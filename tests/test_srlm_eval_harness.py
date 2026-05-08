"""Unit tests for the SRLM bench harness (Phase 1, no LM calls)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

# Bench harness lives outside the package import path; load via sys.path.
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BENCH_ADAPTIVE = REPO_ROOT / "bench" / "adaptive"
if str(BENCH_ADAPTIVE) not in sys.path:
    sys.path.insert(0, str(BENCH_ADAPTIVE))

from _eval_lib import (  # type: ignore  # noqa: E402
    CONFIG_NAMES,
    EvalConfig,
    QuestionRecord,
    ResultRow,
    RolloutObservability,
    bootstrap_ci,
    default_question_set,
    get_config,
    load_aqua,
    load_dabench,
    load_easy_cases,
    load_longcot,
    load_ssb,
    run_bench,
    run_one,
    summarize,
    validate_question_passed,
)


# ---------------------------------------------------------------------------
# Loader tests (per source file)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_loader_easy_cases(tmp_path: Path) -> None:
    p = tmp_path / "easy_cases.jsonl"
    _write_jsonl(p, [{"id": "e1", "bucket": "math", "template": "t",
                      "question": "What is 1+1?", "answer": "2",
                      "validator": "exact_match"}])
    [q] = load_easy_cases(p)
    assert q.id == "e1"
    assert q.prompt == "What is 1+1?"
    assert q.expected == "2"
    assert q.domain == "easy_calibration"


def test_loader_aqua(tmp_path: Path) -> None:
    p = tmp_path / "aqua_rat_15.jsonl"
    _write_jsonl(p, [{"question_id": "a1", "domain": "quant",
                      "prompt": "Pick A/B/C/D/E", "answer": "B"}])
    [q] = load_aqua(p)
    assert q.id == "a1"
    assert q.expected == "B"
    assert q.domain == "math"


def test_loader_dabench(tmp_path: Path) -> None:
    p = tmp_path / "dabench_15.jsonl"
    _write_jsonl(p, [{"question_id": "d1", "prompt": "Q", "answer": "42"}])
    [q] = load_dabench(p)
    assert q.id == "d1"
    assert q.expected == "42"
    assert q.domain == "dabench"


def test_loader_ssb(tmp_path: Path) -> None:
    p = tmp_path / "ssb_subset_50.jsonl"
    _write_jsonl(p, [{"question_id": "s1", "instruction": "Edit cell A1",
                      "prompt_text": "Edit cell A1", "answer_position": "A1"}])
    [q] = load_ssb(p)
    assert q.id == "s1"
    assert q.prompt == "Edit cell A1"
    assert q.expected == "A1"
    assert q.domain == "ssb"


def test_loader_longcot(tmp_path: Path) -> None:
    p = tmp_path / "longcot_cs_hard_holdout25.jsonl"
    _write_jsonl(p, [{"question_id": "l1", "prompt": "Hard puzzle", "answer": "yes"}])
    [q] = load_longcot(p)
    assert q.id == "l1"
    assert q.expected == "yes"
    assert q.domain == "longcot_holdout"


def test_loader_tolerates_alternate_field_names(tmp_path: Path) -> None:
    p = tmp_path / "easy_cases.jsonl"
    # Some files use 'q' / 'gold' / 'expected' instead of canonical names.
    _write_jsonl(p, [{"id": "x", "q": "alt prompt", "gold": "ans"}])
    [q] = load_easy_cases(p)
    assert q.prompt == "alt prompt"
    assert q.expected == "ans"


def test_default_question_set_has_32_questions() -> None:
    qs = default_question_set(REPO_ROOT)
    assert len(qs) == 32
    counts: dict[str, int] = {}
    for q in qs:
        counts[q.domain] = counts.get(q.domain, 0) + 1
    assert counts == {
        "easy_calibration": 5,
        "math": 5,
        "dabench": 12,
        "ssb": 5,
        "longcot_holdout": 5,
    }


# ---------------------------------------------------------------------------
# RolloutObservability + ResultRow are JSON-serializable
# ---------------------------------------------------------------------------


def test_rollout_observability_json_serializable() -> None:
    obs = RolloutObservability(
        selector_key=("a", 1),
        trace_length_completion=42,
        trace_length_turns=3,
        vc_raw_text="0.9",
        vc_parsed=[0.9],
        vc_aggregate=0.9,
        consensus_cluster_id="abc",
        consensus_cluster_size=2,
        srlm_score=1.5,
        discard_reason=None,
    )
    s = json.dumps(asdict(obs), default=str)
    back = json.loads(s)
    assert back["trace_length_completion"] == 42
    assert back["consensus_cluster_size"] == 2


def test_rollout_observability_defaults_are_none() -> None:
    obs = RolloutObservability()
    d = asdict(obs)
    expected_none = {
        "selector_key", "trace_length_completion", "trace_length_turns",
        "vc_raw_text", "vc_parsed", "vc_aggregate",
        "consensus_cluster_id", "consensus_cluster_size",
        "srlm_score", "discard_reason",
    }
    assert set(d.keys()) == expected_none
    for k in expected_none:
        assert d[k] is None


def test_result_row_json_serializable() -> None:
    row = ResultRow(
        question_id="q1", config_name="default", seed=0, passed=True,
        answer="hi", n_turns=1, prompt_tokens=10, completion_tokens=5,
        reasoning_tokens=None, retry_tokens=0, total_cost_tokens=15,
        winner_rung=None, elapsed_s=0.5,
        observability=[RolloutObservability(trace_length_turns=1)],
    )
    s = json.dumps(asdict(row), default=str)
    back = json.loads(s)
    assert back["question_id"] == "q1"
    assert back["observability"][0]["trace_length_turns"] == 1


# ---------------------------------------------------------------------------
# Stub routing: a/b/c/d/all are valid configs and route to adaptive engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_get_config_returns_eval_config_for_each_name(name: str) -> None:
    cfg = get_config(name)
    assert isinstance(cfg, EvalConfig)
    assert cfg.name == name
    if name == "default":
        assert cfg.engine == "default"
    else:
        assert cfg.engine == "adaptive"


def test_stub_features_route_through_adaptive_current_today() -> None:
    base = get_config("adaptive_current").adaptive
    # Phase 2: Feature A (adaptive_a) is now wired to prefer_shorter_traces=True
    # and is no longer a stub. The remaining b/c/d/all stay stubbed.
    for name in ("adaptive_b", "adaptive_c", "adaptive_d", "adaptive_all"):
        cfg = get_config(name)
        # Same engine + same adaptive kwargs as adaptive_current today.
        assert cfg.engine == "adaptive"
        assert cfg.adaptive == base
        # Stub note documents the deferred behavior.
        assert "STUB" in cfg.notes


def test_adaptive_a_wires_prefer_shorter_traces() -> None:
    """Phase 2: adaptive_a is no longer a stub — it sets prefer_shorter_traces=True."""
    cfg = get_config("adaptive_a")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("prefer_shorter_traces") is True
    # Still inherits the K=3 baseline.
    assert cfg.adaptive.get("parallel_rollouts") == 3
    assert "STUB" not in cfg.notes
    assert "Feature A" in cfg.notes


def test_get_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_config("nope")


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------


def test_bootstrap_ci_all_pass_returns_one() -> None:
    point, lo, hi = bootstrap_ci([True] * 10, n_resamples=200, seed=0)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_all_fail_returns_zero() -> None:
    point, lo, hi = bootstrap_ci([False] * 10, n_resamples=200, seed=0)
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_handles_clustered_input() -> None:
    # 3 questions × 2 seeds; flip on q3
    passes = [True, True, True, True, False, False]
    qids = ["q1", "q1", "q2", "q2", "q3", "q3"]
    point, lo, hi = bootstrap_ci(passes, n_resamples=500, seed=0, cluster_ids=qids)
    assert 0 <= lo <= point <= hi <= 1
    assert abs(point - (4 / 6)) < 1e-9


def test_bootstrap_ci_empty() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Domain-bucket tagging via summarize()
# ---------------------------------------------------------------------------


def test_summarize_groups_by_domain_with_expected_counts() -> None:
    qs = default_question_set(REPO_ROOT)
    rows = [
        {
            "question_id": q.id,
            "config_name": "default",
            "seed": 0,
            "passed": True,
            "domain": q.domain,
            "total_cost_tokens": 100,
            "elapsed_s": 1.0,
        }
        for q in qs
    ]
    s = summarize(rows)
    per_dom = s["per_domain"]["default"]
    assert set(per_dom.keys()) == {
        "easy_calibration", "math", "dabench", "ssb", "longcot_holdout",
    }
    assert per_dom["easy_calibration"]["n"] == 5
    assert per_dom["math"]["n"] == 5
    assert per_dom["dabench"]["n"] == 12
    assert per_dom["ssb"]["n"] == 5
    assert per_dom["longcot_holdout"]["n"] == 5


# ---------------------------------------------------------------------------
# validate_question_passed
# ---------------------------------------------------------------------------


def test_validate_question_passed_substring_for_easy_domain() -> None:
    q = QuestionRecord(
        id="q", source_file="easy_cases.jsonl", source_idx=0,
        prompt="2+2?", expected="4",
    )
    assert validate_question_passed("the answer is 4", q) is True
    assert validate_question_passed("the answer is 5", q) is False


def test_validate_question_passed_aqua_letter_extraction() -> None:
    q = QuestionRecord(
        id="q", source_file="aqua_rat_15.jsonl", source_idx=0,
        prompt="...", expected="C",
    )
    assert validate_question_passed("Reasoning ...\nAnswer: C", q) is True
    assert validate_question_passed("Reasoning ...\nAnswer: D", q) is False


def test_validate_question_passed_handles_none() -> None:
    q = QuestionRecord(id="q", source_file="x", source_idx=0, prompt="", expected="x")
    assert validate_question_passed(None, q) is False


# ---------------------------------------------------------------------------
# Idempotency of run_bench (no LM calls on second invocation)
# ---------------------------------------------------------------------------


class _CountingLMFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: str):
        self.calls += 1
        # Return an inert object — run_bench should never reach LM execution
        # the second time around because the result file already exists.
        return object()


def test_run_bench_is_idempotent_no_lm_calls_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub out _build_rlm so the FIRST run also doesn't touch a real LM.
    import _eval_lib as _lib

    class _StubResult:
        payload = {"answer": "stub-answer"}
        total_prompt_tokens = 1
        total_completion_tokens = 2
        total_reasoning_tokens = None
        total_cached_tokens = None

        class trajectory:
            metadata: dict = {}
            turns: list = []

    class _StubRLM:
        def __init__(self) -> None:
            pass

        def run(self):
            return _StubResult()

    def _fake_build_rlm(question, config, lm, max_seconds):
        return _StubRLM()

    monkeypatch.setattr(_lib, "_build_rlm", _fake_build_rlm)

    factory = _CountingLMFactory()
    q = QuestionRecord(
        id="qx", source_file="easy_cases.jsonl", source_idx=0,
        prompt="2+2?", expected="stub-answer",
    )
    cfg = get_config("default")
    results_dir = tmp_path / "results"

    agg1 = run_bench([q], [cfg], [0], "fake-model", results_dir, lm_factory=factory)
    assert agg1["n_ran"] == 1
    assert agg1["n_skipped"] == 0
    assert factory.calls == 1
    files = list(results_dir.glob("*.json"))
    assert len(files) == 1

    # Second run: zero LM factory calls.
    skipped: list[Path] = []
    agg2 = run_bench(
        [q], [cfg], [0], "fake-model", results_dir,
        lm_factory=factory, on_skip=skipped.append,
    )
    assert agg2["n_ran"] == 0
    assert agg2["n_skipped"] == 1
    assert len(skipped) == 1
    # Critical invariant: factory was NOT called again.
    assert factory.calls == 1


def test_run_one_records_error_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _eval_lib as _lib

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(_lib, "_build_rlm", _boom)

    q = QuestionRecord(
        id="qe", source_file="easy_cases.jsonl", source_idx=0,
        prompt="2+2?", expected="4",
    )
    row = run_one(q, get_config("default"), 0, "fake-model", lm_factory=lambda m: object())
    assert row.error is not None
    assert "simulated failure" in row.error
    assert row.passed is False
    assert row.answer is None
