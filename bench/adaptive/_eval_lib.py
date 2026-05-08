"""Reusable bench harness for SRLM evaluation (Phase 1).

This module locks the schema for all subsequent SRLM features (A/B/C/D).
Subsequent phases will populate ``RolloutObservability`` fields by writing
into ``record.metadata['srlm']`` on each rollout's :class:`RLMResult`.

Stability guarantees (do NOT change post-Phase-1):

* Field names on :class:`RolloutObservability` (Phase 1 schema lock).
* JSON layout written by :func:`run_bench` (one file per question×config×seed).
* The 7 config flags (``default``, ``adaptive_current``, ``adaptive_a/b/c/d``,
  ``adaptive_all``).

Adaptive feature flags ``adaptive_a / b / c / d / all`` are accepted as valid
configs today **but route to the same engine path as ``adaptive_current``**
because the underlying features (Trace-len tiebreaker / TraceLengthSignal /
Self-consistency / VC×Len) have not landed yet. This stub is intentional --
it lets us calibrate the harness now and swap in real implementations as
each feature ships, without changing the bench surface or re-running
calibrated baselines.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------

# All seven configs the SRLM bench evaluates. Subsequent feature implementations
# (Phase 2-5) will alter how the adaptive_a/b/c/d/all configs route, but their
# *names* must remain stable so historical result JSON keeps comparing apples
# to apples.
CONFIG_NAMES: tuple[str, ...] = (
    "default",
    "adaptive_current",
    "adaptive_a",  # Feature A: trace-length tiebreaker (stub)
    "adaptive_b",  # Feature B: TraceLengthSignal (stub)
    "adaptive_c",  # Feature C: self-consistency hard pre-filter (stub)
    "adaptive_d",  # Feature D: VC × Len SRLM-style selection (stub)
    "adaptive_all",  # All four features composed (stub)
)


@dataclass(frozen=True)
class EvalConfig:
    """One bench configuration.

    ``engine`` is the literal forwarded to :class:`fabric_rlm.RLM`. ``adaptive``
    is the kwargs dict forwarded under ``adaptive=`` (only meaningful when
    engine=='adaptive').
    """

    name: str
    engine: str  # "default" | "adaptive"
    adaptive: dict[str, Any] = field(default_factory=dict)
    inner_engine: str | None = None
    notes: str = ""


def _stub_features_note(feature_id: str) -> str:
    return (
        f"STUB: Feature {feature_id} not yet implemented; this config currently "
        "routes through the standard adaptive engine and behaves identically to "
        "adaptive_current. Will be wired up in the corresponding feature phase."
    )


def get_config(name: str) -> EvalConfig:
    """Return the EvalConfig for one of the seven canonical config names.

    Adaptive feature flags currently route to ``adaptive_current`` until their
    features land (see module docstring).
    """
    if name not in CONFIG_NAMES:
        raise ValueError(
            f"Unknown config {name!r}; valid: {', '.join(CONFIG_NAMES)}"
        )
    if name == "default":
        return EvalConfig(
            name="default",
            engine="default",
            notes="Plain RLM(engine='default'); no adaptive wrapper.",
        )
    base_adaptive: dict[str, Any] = {"parallel_rollouts": 3}
    if name == "adaptive_current":
        return EvalConfig(
            name="adaptive_current",
            engine="adaptive",
            adaptive=dict(base_adaptive),
            notes="Current adaptive engine (rung-3 best-of-N at K=3).",
        )
    if name == "adaptive_a":
        return EvalConfig(
            name="adaptive_a",
            engine="adaptive",
            adaptive=dict(base_adaptive),
            notes=_stub_features_note("A"),
        )
    if name == "adaptive_b":
        return EvalConfig(
            name="adaptive_b",
            engine="adaptive",
            adaptive=dict(base_adaptive),
            notes=_stub_features_note("B"),
        )
    if name == "adaptive_c":
        return EvalConfig(
            name="adaptive_c",
            engine="adaptive",
            adaptive=dict(base_adaptive),
            notes=_stub_features_note("C"),
        )
    if name == "adaptive_d":
        return EvalConfig(
            name="adaptive_d",
            engine="adaptive",
            adaptive=dict(base_adaptive),
            notes=_stub_features_note("D"),
        )
    # adaptive_all
    return EvalConfig(
        name="adaptive_all",
        engine="adaptive",
        adaptive=dict(base_adaptive),
        notes=_stub_features_note("A+B+C+D"),
    )


# ---------------------------------------------------------------------------
# Question records + loaders
# ---------------------------------------------------------------------------


@dataclass
class QuestionRecord:
    id: str
    source_file: str
    source_idx: int
    prompt: str
    expected: str | None
    task_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        """Domain bucket inferred from ``source_file``.

        Buckets: easy_calibration, math, dabench, ssb, longcot_holdout.
        """
        sf = self.source_file.lower()
        if "easy_cases" in sf:
            return "easy_calibration"
        if "aqua_rat" in sf:
            return "math"
        if "dabench" in sf:
            return "dabench"
        if "ssb" in sf or "spreadsheetbench" in sf:
            return "ssb"
        if "longcot" in sf:
            return "longcot_holdout"
        return "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def load_easy_cases(path: Path, n: int = 5) -> list[QuestionRecord]:
    rows = _read_jsonl(path)[:n]
    out: list[QuestionRecord] = []
    for i, row in enumerate(rows):
        qid = str(_pick(row, "id", "question_id", "qid") or f"easy-{i}")
        prompt = str(_pick(row, "question", "prompt", "q") or "")
        expected = _pick(row, "answer", "expected", "gold")
        out.append(
            QuestionRecord(
                id=qid,
                source_file=str(path.name),
                source_idx=i,
                prompt=prompt,
                expected=str(expected) if expected is not None else None,
                task_meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"id", "question", "answer"}
                },
            )
        )
    return out


def load_aqua(path: Path, n: int = 5) -> list[QuestionRecord]:
    rows = _read_jsonl(path)[:n]
    out: list[QuestionRecord] = []
    for i, row in enumerate(rows):
        qid = str(_pick(row, "question_id", "id", "qid") or f"aqua-{i}")
        prompt = str(_pick(row, "prompt", "question", "q") or "")
        expected = _pick(row, "answer", "expected", "gold")
        out.append(
            QuestionRecord(
                id=qid,
                source_file=str(path.name),
                source_idx=i,
                prompt=prompt,
                expected=str(expected) if expected is not None else None,
                task_meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"question_id", "prompt", "answer"}
                },
            )
        )
    return out


def load_dabench(path: Path, n: int = 12) -> list[QuestionRecord]:
    rows = _read_jsonl(path)[:n]
    out: list[QuestionRecord] = []
    for i, row in enumerate(rows):
        qid = str(_pick(row, "question_id", "id", "qid") or f"dabench-{i}")
        prompt = str(_pick(row, "prompt", "question", "q") or "")
        expected = _pick(row, "answer", "expected", "gold")
        out.append(
            QuestionRecord(
                id=qid,
                source_file=str(path.name),
                source_idx=i,
                prompt=prompt,
                expected=str(expected) if expected is not None else None,
                task_meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"question_id", "prompt", "answer"}
                },
            )
        )
    return out


def load_ssb(path: Path, n: int = 5) -> list[QuestionRecord]:
    rows = _read_jsonl(path)[:n]
    out: list[QuestionRecord] = []
    for i, row in enumerate(rows):
        qid = str(_pick(row, "question_id", "id", "qid") or f"ssb-{i}")
        # SSB uses 'instruction' / 'prompt_text' (typically identical) for
        # the natural-language task description.
        prompt = str(_pick(row, "prompt_text", "instruction", "prompt", "question") or "")
        # SSB does not carry a single 'answer' string — golden_file points
        # to the spreadsheet ground-truth. We capture answer_position as a
        # best-effort textual expectation marker.
        expected = _pick(row, "answer", "expected", "answer_position", "gold")
        out.append(
            QuestionRecord(
                id=qid,
                source_file=str(path.name),
                source_idx=i,
                prompt=prompt,
                expected=str(expected) if expected is not None else None,
                task_meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"question_id", "prompt_text", "instruction", "answer"}
                },
            )
        )
    return out


def load_longcot(path: Path, n: int = 5) -> list[QuestionRecord]:
    rows = _read_jsonl(path)[:n]
    out: list[QuestionRecord] = []
    for i, row in enumerate(rows):
        qid = str(_pick(row, "question_id", "id", "qid") or f"longcot-{i}")
        prompt = str(_pick(row, "prompt", "question", "q") or "")
        expected = _pick(row, "answer", "expected", "gold")
        out.append(
            QuestionRecord(
                id=qid,
                source_file=str(path.name),
                source_idx=i,
                prompt=prompt,
                expected=str(expected) if expected is not None else None,
                task_meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"question_id", "prompt", "answer"}
                },
            )
        )
    return out


# Canonical 32-question mix locked for Phase 1.
def default_question_set(repo_root: Path) -> list[QuestionRecord]:
    bench = repo_root / "bench"
    return [
        *load_easy_cases(bench / "adaptive" / "easy_cases.jsonl", n=5),
        *load_aqua(bench / "adaptive" / "aqua_rat_15.jsonl", n=5),
        *load_dabench(bench / "adaptive" / "dabench_15.jsonl", n=12),
        *load_ssb(bench / "spreadsheetbench" / "ssb_subset_50.jsonl", n=5),
        *load_longcot(bench / "adaptive" / "longcot_cs_hard_holdout25.jsonl", n=5),
    ]


# ---------------------------------------------------------------------------
# Per-rollout observability — Phase 1 schema lock
# ---------------------------------------------------------------------------


@dataclass
class RolloutObservability:
    """One row per rollout. The schema is the contract for Features A/B/C/D.

    Subsequent feature implementations populate these fields by writing into
    ``rlm_result.trajectory.metadata['srlm']`` (a sibling of the existing
    ``metadata['adaptive']`` block). The harness reads them back here.

    Phase 1 leaves every field except ``trace_length_completion`` and
    ``trace_length_turns`` defaulted to None — those two are derived from
    existing token/turn counts and are populated today.
    """

    selector_key: tuple[Any, ...] | list[Any] | None = None
    trace_length_completion: int | None = None
    trace_length_turns: int | None = None
    vc_raw_text: str | None = None
    vc_parsed: list[float] | None = None
    vc_aggregate: float | None = None
    consensus_cluster_id: str | None = None
    consensus_cluster_size: int | None = None
    srlm_score: float | None = None
    discard_reason: str | None = None


@dataclass
class ResultRow:
    question_id: str
    config_name: str
    seed: int
    passed: bool
    answer: str | None
    n_turns: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    retry_tokens: int
    total_cost_tokens: int
    winner_rung: int | None
    elapsed_s: float
    observability: list[RolloutObservability] = field(default_factory=list)
    error: str | None = None
    domain: str = "unknown"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

_AQUA_ANS_RE = re.compile(r"answer\s*[:\-]\s*([A-E])\b", re.IGNORECASE)


def _normalize(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def validate_question_passed(answer: str | None, question: QuestionRecord) -> bool:
    """Generic, domain-aware grader.

    The default RLM engine doesn't run a per-question validator, so the harness
    needs its own grader. We keep this deliberately simple: substring-match for
    canary domains (easy / longcot / dabench), letter-extraction for AQUA, and
    answer-position match for SSB. For SSB the textual ground truth is only an
    address (e.g. "A1"); reliable grading needs the workbook diff and is out
    of scope for the harness — we return False but record the answer.
    """
    expected = question.expected
    if expected is None or answer is None:
        return False
    domain = question.domain
    a = _normalize(answer)
    e = _normalize(expected)
    if not e:
        return False
    if domain == "math":
        m = _AQUA_ANS_RE.search(str(answer))
        if m is not None:
            return m.group(1).strip().upper() == str(expected).strip().upper()
        # Fall back to substring: "Answer: B" -> normalized contains "answer: b"
        return e in a
    if domain == "ssb":
        # SSB needs file diff to grade; substring match on the cell address is
        # not reliable, but is a useful signal-of-life. Treat as "unknown".
        return e in a
    # easy_calibration / dabench / longcot_holdout / unknown: substring match.
    return e in a or a == e


# ---------------------------------------------------------------------------
# LM construction (mirrors tests/behavior/runner.py:make_lm)
# ---------------------------------------------------------------------------

_REASONING_PREFIXES = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


def make_lm(model: str) -> Any:
    """Construct a dspy.LM via OpenRouter, cache disabled.

    Mirrors :func:`tests.behavior.runner.make_lm`; duplicated here so the
    bench harness doesn't import from the test tree.
    """
    import os

    import dspy

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    kwargs: dict[str, Any] = {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": api_key,
        "max_tokens": 16_000 if _is_reasoning(model) else 4_096,
        "cache": False,
    }
    if not _is_reasoning(model):
        kwargs["temperature"] = 1.0
    return dspy.LM(model=f"openrouter/{model}", **kwargs)


# ---------------------------------------------------------------------------
# Per-rollout extraction
# ---------------------------------------------------------------------------


def _extract_answer(result: Any) -> str | None:
    if result is None:
        return None
    payload = getattr(result, "payload", None) or {}
    for key in ("answer", "output", "result", "report"):
        if isinstance(payload, dict) and payload.get(key) is not None:
            return str(payload[key])
    if isinstance(payload, dict) and payload:
        # Best-effort: stringify whatever single field is present.
        for v in payload.values():
            if v is not None:
                return str(v)
    return None


def _trace_completion_tokens(result: Any) -> int:
    val = getattr(result, "total_completion_tokens", None)
    if val is not None:
        return int(val)
    traj = getattr(result, "trajectory", None)
    turns = getattr(traj, "turns", None) if traj else None
    if turns:
        return sum(int(getattr(t, "completion_tokens", None) or 0) for t in turns)
    return 0


def _trace_turn_count(result: Any) -> int:
    traj = getattr(result, "trajectory", None)
    turns = getattr(traj, "turns", None) if traj else None
    return len(turns) if turns else 0


def _build_observability_for_default(result: Any) -> list[RolloutObservability]:
    """Default engine: one rollout, observability built from result directly."""
    srlm_meta: dict[str, Any] = {}
    traj = getattr(result, "trajectory", None)
    if traj is not None:
        srlm_meta = (getattr(traj, "metadata", None) or {}).get("srlm", {}) or {}
    return [
        RolloutObservability(
            selector_key=srlm_meta.get("selector_key"),
            trace_length_completion=srlm_meta.get(
                "trace_length_completion", _trace_completion_tokens(result)
            ),
            trace_length_turns=srlm_meta.get(
                "trace_length_turns", _trace_turn_count(result)
            ),
            vc_raw_text=srlm_meta.get("vc_raw_text"),
            vc_parsed=srlm_meta.get("vc_parsed"),
            vc_aggregate=srlm_meta.get("vc_aggregate"),
            consensus_cluster_id=srlm_meta.get("consensus_cluster_id"),
            consensus_cluster_size=srlm_meta.get("consensus_cluster_size"),
            srlm_score=srlm_meta.get("srlm_score"),
            discard_reason=srlm_meta.get("discard_reason"),
        )
    ]


def _build_observability_for_adaptive(result: Any) -> list[RolloutObservability]:
    """Adaptive engine: one rollout per attempt logged in metadata['adaptive'].

    Future features will additionally write per-attempt entries into
    metadata['srlm']['attempts'] keyed by (rung, rollout_index); we honor
    that shape but fall back to the existing 'adaptive' attempt summaries.
    """
    traj = getattr(result, "trajectory", None)
    meta = getattr(traj, "metadata", None) or {}
    adaptive_meta = meta.get("adaptive") or {}
    srlm_meta = meta.get("srlm") or {}
    srlm_attempts = srlm_meta.get("attempts") or {}

    out: list[RolloutObservability] = []
    for att in adaptive_meta.get("attempts") or []:
        key = (att.get("rung"), att.get("rollout_index"))
        sa = srlm_attempts.get(f"{key[0]}.{key[1]}") or {}
        out.append(
            RolloutObservability(
                selector_key=sa.get("selector_key"),
                trace_length_completion=sa.get(
                    "trace_length_completion", att.get("completion_tokens") or 0
                ),
                trace_length_turns=sa.get(
                    "trace_length_turns", att.get("turns_used") or 0
                ),
                vc_raw_text=sa.get("vc_raw_text"),
                vc_parsed=sa.get("vc_parsed"),
                vc_aggregate=sa.get("vc_aggregate"),
                consensus_cluster_id=sa.get("consensus_cluster_id"),
                consensus_cluster_size=sa.get("consensus_cluster_size"),
                srlm_score=sa.get("srlm_score"),
                discard_reason=sa.get("discard_reason"),
            )
        )
    if not out:
        # No attempt log (e.g., adaptive rejected at construction): synthesize
        # one rollout from the result itself so the schema invariant holds.
        out = _build_observability_for_default(result)
    return out


# ---------------------------------------------------------------------------
# Single (question, config, seed) execution
# ---------------------------------------------------------------------------


def _build_rlm(question: QuestionRecord, config: EvalConfig, lm: Any, max_seconds: int) -> Any:
    """Construct an :class:`RLM` for a single (question, config) pair.

    Uses ``RLM.from_task`` so the inline task/inputs/outputs propagate through
    both the default and adaptive engines (see test_adaptive_runtime.py).
    """
    from fabric_rlm import RLM

    kwargs: dict[str, Any] = dict(
        task=question.prompt,
        inputs=None,
        outputs=["answer"],
        lm=lm,
        max_turns=10,
        timeout=float(max_seconds),
        engine=config.engine,
    )
    if config.inner_engine is not None:
        kwargs["inner_engine"] = config.inner_engine
    if config.engine == "adaptive":
        # Inject a permissive validator so the adaptive runner doesn't insist
        # on an external grader; the harness re-grades after the fact via
        # validate_question_passed(). The runner still uses verdict.passed
        # to decide whether to escalate, so we surface our grader here.
        adaptive_kwargs = dict(config.adaptive)
        if "validator" not in adaptive_kwargs:
            def _validator(result: Any, _q: QuestionRecord = question) -> bool:
                ans = _extract_answer(result)
                return validate_question_passed(ans, _q)

            adaptive_kwargs["validator"] = _validator
        kwargs["adaptive"] = adaptive_kwargs
    return RLM.from_task(**kwargs)


def run_one(
    question: QuestionRecord,
    config: EvalConfig,
    seed: int,
    model: str,
    max_seconds: int = 300,
    *,
    lm_factory: Callable[[str], Any] | None = None,
) -> ResultRow:
    """Run one (question × config × seed). Errors are captured, never raised."""
    lm_factory = lm_factory or make_lm
    random.seed(seed)
    t0 = time.time()
    err: str | None = None
    result: Any = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm = lm_factory(model)
            rlm = _build_rlm(question, config, lm, max_seconds)
            result = rlm.run()
    except Exception as exc:  # noqa: BLE001 — bench must not crash on one row
        err = f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - t0
    answer = _extract_answer(result)
    passed = validate_question_passed(answer, question) if err is None else False

    p = int(getattr(result, "total_prompt_tokens", None) or 0)
    c = int(getattr(result, "total_completion_tokens", None) or 0)
    r_tokens = getattr(result, "total_reasoning_tokens", None)
    r_int = int(r_tokens) if r_tokens is not None else None
    retry_tokens = 0  # populated by features that retry; placeholder today

    if config.engine == "adaptive":
        observability = _build_observability_for_adaptive(result)
        traj = getattr(result, "trajectory", None)
        meta = getattr(traj, "metadata", None) or {}
        winner_rung = (meta.get("adaptive") or {}).get("winner_rung")
    else:
        observability = _build_observability_for_default(result)
        winner_rung = None

    return ResultRow(
        question_id=question.id,
        config_name=config.name,
        seed=seed,
        passed=bool(passed),
        answer=answer,
        n_turns=_trace_turn_count(result),
        prompt_tokens=p,
        completion_tokens=c,
        reasoning_tokens=r_int,
        retry_tokens=retry_tokens,
        total_cost_tokens=p + c + (r_int or 0) + retry_tokens,
        winner_rung=int(winner_rung) if winner_rung is not None else None,
        elapsed_s=elapsed,
        observability=observability,
        error=err,
        domain=question.domain,
    )


# ---------------------------------------------------------------------------
# Bench loop (idempotent + resumable)
# ---------------------------------------------------------------------------


def _result_path(results_dir: Path, question_id: str, config_name: str, seed: int) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.\-]+", "_", question_id)
    return results_dir / f"{safe_qid}__{config_name}__seed{seed}.json"


def _serialize_result(row: ResultRow) -> dict[str, Any]:
    d = asdict(row)
    # asdict already converts nested dataclasses to dicts.
    return d


def run_bench(
    questions: Sequence[QuestionRecord],
    configs: Sequence[EvalConfig],
    seeds: Sequence[int],
    model: str,
    results_dir: Path,
    *,
    max_seconds: int = 300,
    lm_factory: Callable[[str], Any] | None = None,
    on_skip: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Run the cartesian (question × config × seed). Idempotent + resumable.

    For every (qid, config, seed) the row is written as a single JSON file.
    A second invocation with the same args performs zero LM calls because
    each existing file is detected and skipped before construction.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ResultRow] = []
    skipped = 0
    ran = 0

    for q in questions:
        for cfg in configs:
            for seed in seeds:
                path = _result_path(results_dir, q.id, cfg.name, seed)
                if path.exists():
                    skipped += 1
                    if on_skip is not None:
                        on_skip(path)
                    continue
                row = run_one(
                    q, cfg, seed, model,
                    max_seconds=max_seconds, lm_factory=lm_factory,
                )
                path.write_text(
                    json.dumps(_serialize_result(row), indent=2, default=str),
                    encoding="utf-8",
                )
                rows.append(row)
                ran += 1

    return {
        "n_questions": len(questions),
        "n_configs": len(configs),
        "n_seeds": len(seeds),
        "n_ran": ran,
        "n_skipped": skipped,
        "results_dir": str(results_dir),
    }


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# Aggregation: bootstrap CI clustered by question
# ---------------------------------------------------------------------------


def bootstrap_ci(
    passed_list: Sequence[bool] | Sequence[int],
    n_resamples: int = 2000,
    seed: int = 0,
    *,
    cluster_ids: Sequence[Any] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI on the mean of a binary list.

    If ``cluster_ids`` is provided, resampling is done at the cluster level
    (questions are the unit of variance, not individual rollouts).
    Returns ``(point_estimate, lo, hi)``.
    """
    if not passed_list:
        return (0.0, 0.0, 0.0)
    arr = [1 if p else 0 for p in passed_list]
    point = sum(arr) / len(arr)

    rng = random.Random(seed)
    samples: list[float] = []

    if cluster_ids is None:
        n = len(arr)
        for _ in range(n_resamples):
            picks = [arr[rng.randrange(n)] for _ in range(n)]
            samples.append(sum(picks) / n)
    else:
        # Cluster bootstrap: resample clusters with replacement, then average
        # within each picked cluster.
        clusters: dict[Any, list[int]] = {}
        for c, v in zip(cluster_ids, arr):
            clusters.setdefault(c, []).append(v)
        keys = list(clusters.keys())
        if not keys:
            return (point, point, point)
        for _ in range(n_resamples):
            picked = [clusters[keys[rng.randrange(len(keys))]] for _ in range(len(keys))]
            flat = [x for grp in picked for x in grp]
            samples.append(sum(flat) / len(flat))

    samples.sort()
    lo_idx = max(0, int(math.floor((alpha / 2) * len(samples))))
    hi_idx = min(len(samples) - 1, int(math.ceil((1 - alpha / 2) * len(samples))) - 1)
    return (point, samples[lo_idx], samples[hi_idx])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate result-row dicts (loaded from JSON) into a summary report."""
    rows = list(rows)
    by_config: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_config.setdefault(r["config_name"], []).append(r)

    summary: dict[str, Any] = {"per_config": {}, "per_domain": {}}
    for cfg_name, rs in by_config.items():
        passes = [bool(r["passed"]) for r in rs]
        qids = [r["question_id"] for r in rs]
        point, lo, hi = bootstrap_ci(passes, cluster_ids=qids)
        tokens = [int(r["total_cost_tokens"]) for r in rs]
        elapsed = [float(r["elapsed_s"]) for r in rs]
        summary["per_config"][cfg_name] = {
            "n_total": len(rs),
            "n_passed": sum(1 for p in passes if p),
            "accuracy": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "mean_total_tokens": (sum(tokens) / len(tokens)) if tokens else 0.0,
            "mean_elapsed_s": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
        }
        # per-domain breakdown
        per_dom: dict[str, list[bool]] = {}
        for r, p in zip(rs, passes):
            per_dom.setdefault(r.get("domain", "unknown"), []).append(p)
        summary["per_domain"][cfg_name] = {
            dom: {
                "n": len(ps),
                "n_passed": sum(1 for x in ps if x),
                "accuracy": (sum(1 for x in ps if x) / len(ps)) if ps else 0.0,
            }
            for dom, ps in per_dom.items()
        }
    return summary


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# SRLM bench summary", ""]
    lines.append("## Per-config aggregate")
    lines.append("")
    lines.append(
        "| config | n | passed | accuracy | 95% CI | mean total tokens | mean elapsed s |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for name, s in sorted(summary.get("per_config", {}).items()):
        lines.append(
            f"| {name} | {s['n_total']} | {s['n_passed']} | "
            f"{s['accuracy']:.3f} | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | "
            f"{s['mean_total_tokens']:.0f} | {s['mean_elapsed_s']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-domain accuracy")
    lines.append("")
    domains = sorted({d for cfg in summary.get("per_domain", {}).values() for d in cfg.keys()})
    header = "| config | " + " | ".join(domains) + " |"
    sep = "|---|" + "|".join(["---:"] * len(domains)) + "|"
    lines.append(header)
    lines.append(sep)
    for name in sorted(summary.get("per_domain", {})):
        per_dom = summary["per_domain"][name]
        cells = []
        for d in domains:
            entry = per_dom.get(d)
            if entry is None:
                cells.append("—")
            else:
                cells.append(f"{entry['n_passed']}/{entry['n']}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = [
    "CONFIG_NAMES",
    "EvalConfig",
    "QuestionRecord",
    "ResultRow",
    "RolloutObservability",
    "bootstrap_ci",
    "default_question_set",
    "get_config",
    "load_aqua",
    "load_dabench",
    "load_easy_cases",
    "load_longcot",
    "load_ssb",
    "load_all_results",
    "make_lm",
    "render_markdown_report",
    "run_bench",
    "run_one",
    "summarize",
    "validate_question_passed",
]
