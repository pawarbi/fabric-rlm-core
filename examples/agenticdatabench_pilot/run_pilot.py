"""Run fabric-rlm against AgenticDataBench public tasks.

Usage:
    python run_pilot.py --testbed <path/to/AgenticDataBench/testbed> \
        --tasks strategy_2,strategy_3 --outdir <results dir> \
        --lm openrouter/minimax/minimax-m3

The runner maps one benchmark task to one RLM.task call: every file in the
task's data_sources becomes a File input, and the model is instructed to
write the required output file(s) into the per-task output directory that
AgenticDataBench's evaluate.py expects (outdir/<task_id>/<output_file>).

LM spec:
  - any dspy model string (e.g. "openrouter/minimax/minimax-m3"), or
  - "claude-cli[:model]" to route through a logged-in Claude Code CLI, or
  - "stub:<path.py>" for a scripted-turns plumbing test (see stub_lm.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import re
import time
import traceback
from pathlib import Path

from fabric_rlm import RLM, File, SkillLoader


def build_lm(spec: str, temperature: float | None = None,
              reasoning_effort: str | None = None):
    if spec.startswith("stub:"):
        from stub_lm import StubLM

        return StubLM(spec[len("stub:"):])
    if spec.startswith("claude-cli"):
        from claude_cli_lm import ClaudeCLILM

        _, _, model = spec.partition(":")
        return ClaudeCLILM(model or "sonnet")
    # drop_params lets litellm silently discard params a given route rejects.
    # fabric-rlm sets reasoning_effort for anything matching the gpt-5 family
    # regex, and OpenRouter refuses it for some of those models, which errors
    # every turn rather than degrading.
    cfg: dict = {"model": spec, "drop_params": True}
    if reasoning_effort:
        # litellm's model map does not know this route accepts
        # reasoning_effort and blocks it, so the param must be allow-listed
        # explicitly. Without this, drop_params would silently discard it and
        # the run would look like it used the requested effort when it did not.
        cfg["reasoning_effort"] = reasoning_effort
        cfg["allowed_openai_params"] = ["reasoning_effort"]
    if temperature is not None:
        # fabric-rlm defaults non-reasoning models to temperature 1.0, which
        # is a large variance source when benchmarking: two identical runs
        # here disagreed on 4 of 10 tasks. Pin it for measurement.
        cfg["temperature"] = temperature
    return cfg


def resolve_source(testbed: Path, domain: str, src: str) -> Path:
    """Locate a data_sources entry on disk.

    Entries are spelled three ways across the task set: relative to the domain
    directory, with a redundant "datasets/<domain>/" prefix, or as a bare
    filename for a file that actually sits in a subdirectory (e.g.
    "pesticides.csv" lives in "Crop Yield Prediction Dataset/"). Try the
    literal paths first, then fall back to a basename search so those tasks
    are not dropped as missing.
    """
    root = testbed / "datasets" / domain
    rel = src
    for prefix in (f"datasets/{domain}/", f"{domain}/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
    direct = root / rel
    if direct.exists():
        return direct
    if root.exists():
        base = Path(rel).name
        for candidate in root.rglob(base):
            if candidate.is_file():
                return candidate
    return direct          # non-existent; caller flags it as MISSING


def make_output_validator(task_dir: Path, names: list[str]):
    """Reject a SUBMIT that did not actually write the required files.

    Trace analysis: on the 246-task runs, 13 M3 tasks and 10 Kimi tasks called
    SUBMIT while producing no output file at all, scoring zero for a reason the
    model could have fixed itself. The runtime turns an AssertionError here
    into a repair turn with the message attached, and the check costs no prompt
    tokens unless it fires. Never reads the gold answer.
    """
    def validate(payload):
        missing = [n for n in names if not (task_dir / n).exists()]
        assert not missing, (
            f"required output file(s) not written: {missing}. Write them to "
            f"{task_dir} with exactly those names, then SUBMIT again.")
        for n in names:
            f = task_dir / n
            assert f.stat().st_size > 0, f"{n} was written but is empty."
            if n.lower().endswith(".csv"):
                import pandas as pd
                try:
                    df = pd.read_csv(f)
                except Exception as exc:
                    raise AssertionError(f"{n} is not readable as CSV: {exc}")
                assert len(df) > 0, f"{n} has a header but no data rows."
    return validate


def outputs_agree(dir_a: Path, dir_b: Path, names: list[str],
                  rtol: float = 1e-3) -> bool:
    """Do two blind attempts agree on the files they produced?

    Erring toward disagreement, as the library's ``answers_agree`` does: a
    false disagreement costs one reconciliation run, a false agreement costs
    correctness. Compares shape, column names and cell values (numerics within
    a relative tolerance). Never reads the gold answer.
    """
    import pandas as pd

    for name in names:
        pa, pb = dir_a / name, dir_b / name
        if not pa.exists() or not pb.exists():
            return False
        if pa.suffix.lower() not in (".csv", ".json", ".txt", ".tsv"):
            # Images and binaries: agree only if byte-identical, which they
            # rarely are; treat as disagreement rather than guessing.
            if pa.read_bytes() != pb.read_bytes():
                return False
            continue
        if pa.suffix.lower() != ".csv":
            if pa.read_text(encoding="utf-8", errors="ignore").strip() != \
               pb.read_text(encoding="utf-8", errors="ignore").strip():
                return False
            continue
        try:
            da, db = pd.read_csv(pa), pd.read_csv(pb)
        except Exception:
            return False
        if da.shape != db.shape or set(da.columns) != set(db.columns):
            return False
        db = db[list(da.columns)]
        for col in da.columns:
            sa, sb = da[col], db[col]
            if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
                import numpy as np
                if not np.allclose(sa.fillna(0), sb.fillna(0), rtol=rtol,
                                   atol=1e-6, equal_nan=True):
                    return False
            else:
                if list(sa.astype(str).str.strip().str.lower()) != \
                   list(sb.astype(str).str.strip().str.lower()):
                    return False
    return True


def preview_of(path: Path, max_cols: int = 40) -> str:
    """Compact, deterministic schema preview of a tabular input.

    Trace analysis of the 246-task M3 run: 3.7 leading turns per task, about
    30 percent of all turns, do nothing but discover columns and dtypes, and
    input tokens are 96 percent of volume because the whole transcript is
    resent every turn. Computing this in the harness costs no model tokens
    and removes the reason for those turns.
    """
    try:
        import pandas as pd
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv", ".txt"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=200, low_memory=False)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in (".json", ".jsonl"):
            df = pd.read_json(path, lines=(suffix == ".jsonl"), nrows=200)
        else:
            return ""
        cols = list(df.columns)[:max_cols]
        more = "" if len(df.columns) <= max_cols else f" (+{len(df.columns)-max_cols} more)"
        dtypes = ", ".join(f"{c}:{df[c].dtype}" for c in cols)
        head = df[cols].head(3).to_string(max_colwidth=24)
        return (f"    columns ({len(df.columns)}){more}: {dtypes}\n"
                f"    first rows:\n{head}")
    except Exception as exc:                      # unreadable is not fatal
        return f"    (preview unavailable: {type(exc).__name__})"


def input_key(index: int, source: str) -> str:
    stem = Path(source).stem
    key = re.sub(r"\W+", "_", stem).strip("_").lower() or f"file_{index}"
    if key[0].isdigit():
        key = f"f_{key}"
    return key


def run_task(task: dict, testbed: Path, outdir: Path, lm, max_turns: int,
             skills: list[str] | None = None,
             skill_dir: str | None = None,
             timeout: float = 900.0,
             previews: bool = False,
             verify: bool = False,
             auto_skills: bool = False,
             validate_outputs: bool = False) -> dict:
    task_dir = outdir / task["id"]
    task_dir.mkdir(parents=True, exist_ok=True)

    inputs = {}
    source_lines = []
    # 62 of the 246 public tasks carry no data_sources field and name their
    # files inline instead ("Based on the pub_fund.sqlite database... refer to
    # dataset_pub_fund.md"). Bind the domain directory so those are reachable.
    if not task.get("data_sources"):
        domain_dir = testbed / "datasets" / task["domain"]
        inputs["dataset_dir"] = File(domain_dir)
        if domain_dir.exists():
            names = sorted(p.relative_to(domain_dir).as_posix()
                           for p in domain_dir.rglob("*") if p.is_file())
            source_lines.append("- `dataset_dir`: directory holding this "
                                "task's files: " + ", ".join(names[:40]))
        else:
            source_lines.append(f"- `dataset_dir`: {domain_dir}  (MISSING)")

    for i, src in enumerate(task.get("data_sources", [])):
        path = resolve_source(testbed, task["domain"], src)
        key = input_key(i, src)
        while key in inputs:
            key = f"{key}_{i}"
        inputs[key] = File(path)
        line = f"- `{key}`: {src}" + ("" if path.exists() else "  (MISSING)")
        if previews and path.exists():
            pv = preview_of(path)
            if pv:
                line += "\n" + pv
        source_lines.append(line)

    out_files = ", ".join(task["output_file_name"])
    prompt = (
        f"{task['question']}\n\n"
        f"Input files provided as File handles:\n" + "\n".join(source_lines) + "\n\n"
        f"Write the required output file(s) ({out_files}) with exactly those "
        f"file names into this directory: {task_dir.resolve()}\n"
        f"After writing, reload each output file and verify it has the "
        f"required columns before submitting."
    )

    # compare_image does not compare pixels: it reads .json and .npy sidecars
    # holding the extracted plot data, which the benchmark's own DA-Agent
    # environment generates by instrumenting matplotlib. Nothing generates
    # them here, so an image task would score 0 for harness reasons rather
    # than model reasons. Give the model the same extractor the official
    # harness applies automatically.
    if any(str(o).lower().endswith(".png") for o in task["output_file_name"]):
        prompt += (
            "\n\nFor every .png you save, the grader reads plot metadata from "
            "sidecar files, not the pixels. Immediately after saving each "
            "figure, run the benchmark's own extractor on the SAME figure "
            "object:\n"
            "    import sys\n"
            f"    sys.path.insert(0, r\"{testbed.resolve()}\")\n"
            "    from da_agent.configs.scripts.image import Plotprocess\n"
            "    fig.savefig(png_path)\n"
            "    Plotprocess.plot_process(fig, png_path)\n"
            "This writes the .json and .npy files next to the image. Without "
            "them the figure cannot be graded at all, so treat it as part of "
            "producing the output, and confirm both sidecars exist before "
            "submitting."
        )

    started = time.monotonic()
    record: dict = {"id": task["id"], "domain": task["domain"],
                    "skills": skills or []}
    extra: dict = {}
    if auto_skills:
        extra["enable_skill_autoloading"] = True
        extra["enable_router"] = True
    if validate_outputs:
        extra["output_validator"] = make_output_validator(
            task_dir, list(task["output_file_name"]))
        # 14 of the 27 no-output tasks never called SUBMIT at all, several
        # after exhausting turns. Reserving turns forces a finalize attempt
        # instead of the run ending with nothing on disk.
        extra["reserve_finalize_turns"] = 2
    if skills:
        extra["skills"] = skills
        if skill_dir:
            extra["skill_loader"] = SkillLoader(skill_dir=skill_dir)
    def solve_into(dest: Path, extra_text: str = ""):
        dest.mkdir(parents=True, exist_ok=True)
        text = prompt.replace(str(task_dir.resolve()), str(dest.resolve()))
        rlm = RLM.task(task=text, inputs=inputs, outputs=["summary"], lm=lm,
                       max_turns=max_turns, timeout=timeout, **extra)
        return rlm.run()

    try:
        if verify:
            # fabric_rlm.verified_task solves the same task text twice, so both
            # attempts write to one path and the winner it returns can differ
            # from the file left on disk. Grading here reads files, so each
            # attempt gets its own directory and agreement is decided on the
            # produced files. The gold answer is never consulted -- this is
            # self-consistency between two blind solves, nothing more.
            a_dir, b_dir = task_dir / "_a", task_dir / "_b"
            res_a = solve_into(a_dir)
            res_b = solve_into(b_dir)
            attempts = [("a", a_dir, res_a), ("b", b_dir, res_b)]
            agree = outputs_agree(a_dir, b_dir, task["output_file_name"])
            record["verify_agreed"] = agree
            if agree:
                winner = a_dir
                record["verdict"] = "agree"
            else:
                c_dir = task_dir / "_c"
                res_c = solve_into(c_dir, "")
                attempts.append(("c", c_dir, res_c))
                # Prefer the reconciler, but only if it actually produced the
                # required files; otherwise fall back to whichever attempt did.
                order = [c_dir, a_dir, b_dir]
                winner = next(
                    (d for d in order
                     if all((d / o).exists() for o in task["output_file_name"])),
                    c_dir)
                record["verdict"] = "reconciled"
            for name in os.listdir(winner):
                src_p, dst_p = winner / name, task_dir / name
                if src_p.is_file():
                    shutil.copy2(src_p, dst_p)
            result = next(r for tag, d, r in attempts if d == winner)
            record["attempts"] = len(attempts)
        else:
            result = solve_into(task_dir)
        record["summary"] = str(getattr(result, "summary", ""))[:2000]
        record["status"] = "completed"
        trajectory = getattr(result, "trajectory", None)
        if trajectory is not None and hasattr(trajectory, "to_jsonl"):
            (task_dir / "trajectory.jsonl").write_text(
                trajectory.to_jsonl(), encoding="utf-8"
            )
    except Exception as exc:  # a task must not kill the batch
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        (task_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    record["seconds"] = round(time.monotonic() - started, 1)
    record["outputs_written"] = [
        name for name in task["output_file_name"] if (task_dir / name).exists()
    ]

    # AgenticDataBench's evaluator skips any task without this marker file
    # and requires the keys read in _get_trajectory_info_from_json:
    # trajectory, finished, steps, result, result_files.{added,changed}_files.
    # Harness-specific trajectory parsing keys on the output path containing
    # 'smolagents'/'da-agent'/'claude-code'; ours matches none, so the
    # trajectory list is only carried as metadata.
    dabench_dir = task_dir / "dabench"
    dabench_dir.mkdir(exist_ok=True)
    (dabench_dir / "result.json").write_text(
        json.dumps({
            "trajectory": [],
            "finished": record["status"] == "completed",
            "steps": 0,
            "result": record.get("summary", record.get("error", "")),
            "result_files": {
                "added_files": record["outputs_written"],
                "changed_files": [],
            },
            "harness": "fabric-rlm",
            "seconds": record["seconds"],
        }),
        encoding="utf-8",
    )
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task ids")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lm", default="claude-cli")
    ap.add_argument("--max-turns", type=int, default=25,
                    help="turn ceiling; at 12 it bound on 4 of 10 pilot "
                         "tasks, truncating one mid-write")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-turn subprocess execution timeout in seconds "
                         "(fabric-rlm default is 300)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="reasoning effort for models that accept it "
                         "(e.g. max, high); allow-listed through litellm")
    ap.add_argument("--validate-outputs", action="store_true",
                    help="reject SUBMIT when the required output files are "
                         "missing, empty or unreadable, driving a repair turn")
    ap.add_argument("--auto-skills", action="store_true",
                    help="enable the library's skill autoloading and keyword "
                         "router (both default OFF in RLM.__init__)")
    ap.add_argument("--previews", action="store_true",
                    help="inject deterministic schema previews of tabular "
                         "inputs; removes the ~3.7 discovery turns per task")
    ap.add_argument("--verify", action="store_true",
                    help="blind double-solve with file-level agreement and "
                         "a reconciliation run on disagreement")
    ap.add_argument("--temperature", type=float, default=None,
                    help="pin sampling temperature; recommended for "
                         "measurement runs, see PREREGISTRATION.md")
    ap.add_argument("--skills", default="",
                    help="comma-separated skill names, e.g. output_contract")
    ap.add_argument("--skill-dir", default="",
                    help="directory of contrib skills to layer over the "
                         "packaged ones")
    args = ap.parse_args()
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]

    testbed = Path(args.testbed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(testbed / "tasks" / "dev.jsonl", encoding="utf-8") as fh:
        all_tasks = {json.loads(line)["id"]: json.loads(line) for line in fh}

    lm = build_lm(args.lm, args.temperature, args.reasoning_effort)
    records = []
    for tid in args.tasks.split(","):
        tid = tid.strip()
        if tid not in all_tasks:
            print(f"[skip] unknown task id: {tid}")
            continue
        print(f"[run ] {tid}")
        record = run_task(all_tasks[tid], testbed, outdir, lm, args.max_turns,
                          skills=skills, skill_dir=args.skill_dir or None,
                          timeout=args.timeout, previews=args.previews,
                          verify=args.verify, auto_skills=args.auto_skills,
                          validate_outputs=args.validate_outputs)
        print(f"[done] {tid}: {record['status']}, "
              f"outputs={record['outputs_written']}, {record['seconds']}s")
        records.append(record)

    (outdir / "pilot_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {outdir / 'pilot_records.json'}")
    print("Grade with AgenticDataBench's own evaluator, e.g.:")
    print(f"  python evaluate.py --output_dir {outdir} --gold_dir gold "
          f"--eval_json <pilot tasks jsonl> --result_dir results")


if __name__ == "__main__":
    main()
