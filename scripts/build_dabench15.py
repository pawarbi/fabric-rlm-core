"""Build a 15-question data-analysis bench from InfiAgent-DABench.

DABench (https://huggingface.co/datasets/infiagent/DABench) provides 257
data-analysis questions, each referencing a real CSV table. Answers use
the strict structured format ``@key[value]`` which is straightforward to
grade with regex (numeric tolerance for floats, string-equal otherwise).

This builder:
  * loads the dev questions/labels JSONLs from the local session cache
    (or downloads them from HuggingFace if missing),
  * stratified-samples 5 easy + 5 medium + 5 hard questions, biased
    toward the most common CSV tables to minimise upload size,
  * downloads each referenced CSV from HF to ``bench/adaptive/dabench_tables/``,
  * writes ``bench/adaptive/dabench_15.jsonl`` matching the schema used
    by the comparison harness (question_id/template/prompt/answer/metadata).

The prompt tells the model the lakehouse path of the CSV so the
fabric_rlm interpreter can load + analyze it with pandas (or duckdb).

Usage:
    python scripts/build_dabench15.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
OUT_JSONL = ROOT / "bench" / "adaptive" / "dabench_15.jsonl"
TABLES_DIR = ROOT / "bench" / "adaptive" / "dabench_tables"

LAKE_TABLES_DIR = "/lakehouse/default/Files/fabric_rlm_dabench_tables"

HF_BASE = "https://huggingface.co/datasets/infiagent/DABench/resolve/main"
QUESTIONS_URL = f"{HF_BASE}/da-dev-questions.jsonl"
LABELS_URL = f"{HF_BASE}/da-dev-labels.jsonl"
TABLE_URL_TMPL = f"{HF_BASE}/da-dev-tables/{{name}}"

CACHE_DIR = Path.home() / ".copilot" / "session-state" / "83674b05-6393-422f-bd1f-ee20b1f0502a" / "files" / "dabench"
QUESTIONS_CACHE = CACHE_DIR / "questions.jsonl"
LABELS_CACHE = CACHE_DIR / "labels.jsonl"

PER_LEVEL = 5
SEED = 7

INSTRUCTIONS_TMPL = (
    "You are a data-analysis assistant. You have access to a Python interpreter "
    "with pandas/numpy/scipy installed. A CSV file is available on disk at:\n"
    "    {csv_path}\n"
    "Load the file with pandas (e.g. `pd.read_csv('{csv_path}')`) and answer the "
    "question below. Follow ALL constraints exactly.\n\n"
    "On the final line of your response, output your answer in EXACTLY this format:\n"
    "    {fmt}\n"
    "If the format lists multiple keys, output all of them on a single final line, "
    "comma-separated (e.g. `@key_a[v1], @key_b[v2]`). Use only ASCII brackets "
    "`[` and `]`. Do NOT add any text after the answer line.\n\n"
    "Question:\n{question}\n\n"
    "Constraints:\n{constraints}\n"
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ensure_question_files() -> tuple[list[dict], dict[int, list[list[str]]]]:
    if not QUESTIONS_CACHE.exists():
        QUESTIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {QUESTIONS_URL}")
        urlretrieve(QUESTIONS_URL, QUESTIONS_CACHE)
    if not LABELS_CACHE.exists():
        print(f"downloading {LABELS_URL}")
        urlretrieve(LABELS_URL, LABELS_CACHE)
    qs = _load_jsonl(QUESTIONS_CACHE)
    labels = {row["id"]: row["common_answers"] for row in _load_jsonl(LABELS_CACHE)}
    return qs, labels


def _sample_stratified(qs: list[dict], labels: dict, rng: random.Random) -> list[dict]:
    file_freq = Counter(q["file_name"] for q in qs)
    chosen: list[dict] = []
    chosen_files: set[str] = set()

    for level in ("easy", "medium", "hard"):
        pool = [q for q in qs if q["level"] == level and q["id"] in labels]
        # Bias toward common files first (so we download fewer CSVs);
        # within each file sort by id for determinism, then shuffle the
        # ordered file groups deterministically.
        pool.sort(key=lambda q: (-file_freq[q["file_name"]], q["file_name"], q["id"]))
        # Walk the sorted pool and pick PER_LEVEL questions, preferring
        # files we've already chosen, but always allowing new files when
        # needed to reach the quota.
        picked: list[dict] = []
        seen_ids: set[int] = set()

        for q in pool:
            if len(picked) >= PER_LEVEL:
                break
            if q["id"] in seen_ids:
                continue
            picked.append(q)
            seen_ids.add(q["id"])
            chosen_files.add(q["file_name"])
        chosen.extend(picked)

    # Shuffle deterministically so smoke tests don't always hit the same q
    rng.shuffle(chosen)
    return chosen


def _ensure_table(name: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return out
    url = TABLE_URL_TMPL.format(name=name)
    print(f"  downloading {name} ({url})")
    urlretrieve(url, out)
    return out


def main() -> int:
    qs, labels = _ensure_question_files()
    rng = random.Random(SEED)
    chosen = _sample_stratified(qs, labels, rng)
    print(f"sampled {len(chosen)} questions across",
          dict(Counter(q["level"] for q in chosen)))

    needed_files = sorted({q["file_name"] for q in chosen})
    print(f"need {len(needed_files)} unique tables: {needed_files}")
    for name in needed_files:
        local = _ensure_table(name)
        print(f"    {name}: {local.stat().st_size} bytes")

    rows = []
    for q in chosen:
        gold = labels[q["id"]]  # list of [key, value]
        csv_path = f"{LAKE_TABLES_DIR}/{q['file_name']}"
        prompt = INSTRUCTIONS_TMPL.format(
            csv_path=csv_path,
            fmt=q["format"].strip(),
            question=q["question"].strip(),
            constraints=q.get("constraints", "").strip(),
        )
        rows.append({
            "question_id": f"DABENCH_{q['id']:04d}",
            "domain": "data_analysis",
            "difficulty": q["level"],
            "template": "DABENCH",
            "prompt": prompt,
            # Gold answer = JSON-encoded list of [key, value] pairs.
            # The grader regex-matches @key[value] tokens in the response.
            "answer": json.dumps(gold),
            "metadata": {
                "dabench_id": q["id"],
                "file_name": q["file_name"],
                "concepts": q.get("concepts") or [],
                "format_spec": q["format"],
                "level": q["level"],
            },
        })

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT_JSONL.relative_to(ROOT)}")
    print(f"tables in {TABLES_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
