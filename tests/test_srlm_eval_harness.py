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
    _validate_dabench,
    _validate_longcot,
    _validate_ssb,
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


def test_default_question_set_excludes_ssb() -> None:
    # SSB dropped in Phase 3: ``_validate_ssb`` is a substring stub that
    # returns ~0% across configs, contributing pure noise to aggregate scores.
    # See ``default_question_set`` docstring for the Phase 4 follow-up.
    qs = default_question_set(REPO_ROOT)
    assert len(qs) == 27
    counts: dict[str, int] = {}
    for q in qs:
        counts[q.domain] = counts.get(q.domain, 0) + 1
    assert counts == {
        "easy_calibration": 5,
        "math": 5,
        "dabench": 12,
        "longcot_holdout": 5,
    }
    assert "ssb" not in counts


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
        "candidate_answer_preview",
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
    # Phase 2: Feature A (adaptive_a) wired (prefer_shorter_traces=True).
    # Phase 3: Feature C (adaptive_c) wired (prefer_consensus=True);
    # adaptive_all now composes A+C. Remaining b/d still stubbed.
    for name in ("adaptive_b", "adaptive_d"):
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


def test_adaptive_c_wires_prefer_consensus() -> None:
    """Phase 3: adaptive_c sets prefer_consensus=True; not a stub anymore."""
    cfg = get_config("adaptive_c")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("prefer_consensus") is True
    # adaptive_c does NOT enable Feature A — keep flags orthogonal so
    # bench A/B can decompose contributions.
    assert cfg.adaptive.get("prefer_shorter_traces") is not True
    assert cfg.adaptive.get("parallel_rollouts") == 3
    assert "STUB" not in cfg.notes
    assert "Feature C" in cfg.notes


def test_adaptive_c_minrung3_forces_rung3_with_consensus() -> None:
    cfg = get_config("adaptive_c_minrung3")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("prefer_consensus") is True
    assert cfg.adaptive.get("start_rung") == 3
    assert cfg.force_min_rung == 3


def test_adaptive_all_composes_a_and_c() -> None:
    """adaptive_all is no longer a pure stub: A and C are wired, B and D still stub."""
    cfg = get_config("adaptive_all")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("prefer_shorter_traces") is True
    assert cfg.adaptive.get("prefer_consensus") is True
    assert "STUB" not in cfg.notes


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
        "easy_calibration", "math", "dabench", "longcot_holdout",
    }
    assert per_dom["easy_calibration"]["n"] == 5
    assert per_dom["math"]["n"] == 5
    assert per_dom["dabench"]["n"] == 12
    assert per_dom["longcot_holdout"]["n"] == 5
    assert "ssb" not in per_dom


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


# ---------------------------------------------------------------------------
# Phase 2 — B1 (cost aggregation) + B2 (selector_key propagation)
# ---------------------------------------------------------------------------


def _make_adaptive_stub_result(attempts: list[dict], winner_idx: int = -1):
    """Build a minimal stand-in for an RLMResult coming out of AdaptiveRunner.

    ``attempts`` is a list of per-attempt summary dicts mimicking what
    ``AttemptRecord.to_summary()`` produces. The ``winner_idx`` selects which
    attempt's tokens become ``total_*_tokens`` on the wrapping RLMResult
    (mimicking how the runner exposes the winner's tokens at the top level).
    """
    win = attempts[winner_idx]

    class _Traj:
        metadata = {
            "adaptive": {
                "stop_reason": "passed",
                "elapsed_seconds": 1.0,
                "winner_rung": win.get("rung"),
                "winner_rollout_index": win.get("rollout_index"),
                "attempts": attempts,
            }
        }
        turns: list = []

    class _R:
        payload = {"answer": "x"}
        submitted = True
        total_prompt_tokens = win.get("prompt_tokens") or 0
        total_completion_tokens = win.get("completion_tokens") or 0
        total_reasoning_tokens = win.get("reasoning_tokens")
        total_cached_tokens = None
        trajectory = _Traj()

    return _R()


def test_run_one_adaptive_total_cost_sums_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1: total_cost_tokens must equal sum across ALL attempts, not winner-only."""
    import _eval_lib as _lib

    attempts = [
        {"rung": 3, "rollout_index": 0, "prompt_tokens": 100, "completion_tokens": 318, "reasoning_tokens": 0, "turns_used": 1, "srlm": {}},
        {"rung": 3, "rollout_index": 1, "prompt_tokens": 100, "completion_tokens": 567, "reasoning_tokens": 0, "turns_used": 1, "srlm": {}},
        {"rung": 3, "rollout_index": 2, "prompt_tokens": 100, "completion_tokens": 173, "reasoning_tokens": 0, "turns_used": 1, "srlm": {"selector_key": [1, 0, 0, 0, 0, -173, 0], "selector_won": True}},
    ]
    stub = _make_adaptive_stub_result(attempts, winner_idx=2)

    class _StubRLM:
        def run(self):
            return stub

    monkeypatch.setattr(_lib, "_build_rlm", lambda *_a, **_kw: _StubRLM())

    q = QuestionRecord(
        id="qcost", source_file="dabench_15.jsonl", source_idx=0,
        prompt="x?", expected="x",
    )
    row = run_one(q, get_config("adaptive_a"), 0, "fake-model", lm_factory=lambda m: object())
    expected_p = 300  # 3 * 100
    expected_c = 318 + 567 + 173  # 1058
    assert row.prompt_tokens == expected_p, f"winner-only cost leaked: {row.prompt_tokens}"
    assert row.completion_tokens == expected_c, f"winner-only cost leaked: {row.completion_tokens}"
    assert row.total_cost_tokens == expected_p + expected_c


def test_run_one_adaptive_propagates_selector_key_into_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2: every per-rollout observability record must carry the srlm dict
    written by select_best_of_n (selector_key + trace_length_completion).
    """
    import _eval_lib as _lib

    attempts = [
        {
            "rung": 3, "rollout_index": 0,
            "prompt_tokens": 50, "completion_tokens": 200, "turns_used": 1,
            "srlm": {"selector_key": [1, 0, 0, 0, 0, -200, 0], "trace_length_completion": 200},
        },
        {
            "rung": 3, "rollout_index": 1,
            "prompt_tokens": 50, "completion_tokens": 100, "turns_used": 1,
            "srlm": {
                "selector_key": [1, 0, 0, 0, 0, -100, -1],
                "trace_length_completion": 100,
                "selector_won": True,
            },
        },
    ]
    stub = _make_adaptive_stub_result(attempts, winner_idx=1)

    class _StubRLM:
        def run(self):
            return stub

    monkeypatch.setattr(_lib, "_build_rlm", lambda *_a, **_kw: _StubRLM())

    q = QuestionRecord(
        id="qsk", source_file="dabench_15.jsonl", source_idx=0,
        prompt="x?", expected="x",
    )
    row = run_one(q, get_config("adaptive_a"), 0, "fake-model", lm_factory=lambda m: object())
    assert len(row.observability) == 2
    # Critical invariant: NO observability record may have selector_key=None
    # for a Feature A run that produced multiple attempts. The duck flagged
    # this as B2 — without per-rollout selector_key we can't prove the
    # tiebreaker actually fired.
    sks = [obs.selector_key for obs in row.observability]
    assert all(sk is not None for sk in sks), f"B2 regression: selector_keys={sks}"
    # And both trace_length_completion values should make it through
    tlcs = [obs.trace_length_completion for obs in row.observability]
    assert tlcs == [200, 100], f"trace lengths not propagated: {tlcs}"

# ---------------------------------------------------------------------------
# Phase 2.5 — Domain-specific validators (dabench / ssb / longcot)
# ---------------------------------------------------------------------------


# DABench: list-of-pairs JSON answers with numeric tolerance.
class TestValidateDabench:
    def test_exact_json_match(self) -> None:
        assert _validate_dabench(
            '[["correlation_pclass_fare", "-0.55"]]',
            '[["correlation_pclass_fare", "-0.55"]]',
        ) is True

    def test_numeric_tolerance(self) -> None:
        # rounding noise within 1% should still pass
        assert _validate_dabench(
            '[["correlation_pclass_fare", "-0.5499"]]',
            '[["correlation_pclass_fare", "-0.55"]]',
        ) is True

    def test_obvious_fail(self) -> None:
        assert _validate_dabench(
            '[["correlation_pclass_fare", "0.99"]]',
            '[["correlation_pclass_fare", "-0.55"]]',
        ) is False

    def test_prose_with_name_and_number(self) -> None:
        # Model didn't emit JSON; prose mentions the name and the value.
        prose = "The correlation_pclass_fare is approximately -0.55."
        assert _validate_dabench(
            prose, '[["correlation_pclass_fare", "-0.55"]]'
        ) is True

    def test_prose_missing_name_fails(self) -> None:
        prose = "The value is -0.55."
        assert _validate_dabench(
            prose, '[["correlation_pclass_fare", "-0.55"]]'
        ) is False

    def test_multi_pair_match(self) -> None:
        assert _validate_dabench(
            '[["average_fare_Mr", "24.44"], ["average_fare_Mrs", "45.14"]]',
            '[["average_fare_Mrs", "45.14"], ["average_fare_Mr", "24.44"]]',
        ) is True

    def test_yes_no_string_value(self) -> None:
        assert _validate_dabench(
            '[["significance", "Yes"]]',
            '[["significance", "Yes"]]',
        ) is True
        assert _validate_dabench(
            '[["significance", "No"]]',
            '[["significance", "Yes"]]',
        ) is False

    def test_returns_false_on_garbage(self) -> None:
        assert _validate_dabench("not json", '[["x", "1"]]') is False
        assert _validate_dabench("anything", "{not json") is False

    def test_none_input_returns_false(self) -> None:
        assert _validate_dabench(None, '[["x", "1"]]') is False
        assert _validate_dabench("anything", None) is False


# SSB: signal-of-life range token match.
class TestValidateSsb:
    def test_range_token_in_answer(self) -> None:
        assert _validate_ssb(
            "Filled formula in H3:H6 with the recoupment formula.", "H3:H6"
        ) is True

    def test_case_insensitive(self) -> None:
        assert _validate_ssb("filled cells h3:h6 done", "H3:H6") is True

    def test_no_mention_returns_false(self) -> None:
        assert _validate_ssb("Did some work in column G", "H3:H6") is False

    def test_none_input_returns_false(self) -> None:
        assert _validate_ssb(None, "H3:H6") is False
        assert _validate_ssb("text", None) is False

    def test_empty_expected_returns_false(self) -> None:
        assert _validate_ssb("text", "") is False


# LongCoT: dict-with-list-value JSON answers.
class TestValidateLongcot:
    EXPECTED = '{"final_capacities": [5, 14, 7, 9]}'

    def test_exact_dict_match(self) -> None:
        assert _validate_longcot(self.EXPECTED, self.EXPECTED) is True

    def test_obvious_fail(self) -> None:
        assert _validate_longcot(
            '{"final_capacities": [5, 14, 7, 8]}', self.EXPECTED
        ) is False

    def test_bare_list_answer_matches_dict_expected(self) -> None:
        assert _validate_longcot("[5, 14, 7, 9]", self.EXPECTED) is True

    def test_inline_list_in_prose(self) -> None:
        prose = "After computing, the final capacities are [5, 14, 7, 9]. Done."
        assert _validate_longcot(prose, self.EXPECTED) is True

    def test_inline_list_wrong_values(self) -> None:
        prose = "Got: [5, 14, 7, 8]"
        assert _validate_longcot(prose, self.EXPECTED) is False

    def test_returns_false_on_garbage(self) -> None:
        assert _validate_longcot("not json", self.EXPECTED) is False

    def test_none_input_returns_false(self) -> None:
        assert _validate_longcot(None, self.EXPECTED) is False
        assert _validate_longcot("x", None) is False


# Dispatch through validate_question_passed.
class TestValidateDispatch:
    def test_dabench_dispatch(self) -> None:
        q = QuestionRecord(
            id="d", source_file="dabench_15.jsonl", source_idx=0,
            prompt="x", expected='[["x", "1.5"]]',
        )
        assert validate_question_passed('[["x", "1.5"]]', q) is True
        assert validate_question_passed('[["x", "9.9"]]', q) is False

    def test_ssb_dispatch(self) -> None:
        q = QuestionRecord(
            id="s", source_file="ssb_subset_50.jsonl", source_idx=0,
            prompt="x", expected="A1:B2",
        )
        assert validate_question_passed("touched A1:B2 cells", q) is True
        assert validate_question_passed("did nothing", q) is False

    def test_longcot_dispatch(self) -> None:
        q = QuestionRecord(
            id="l", source_file="longcot_cs_hard_holdout25.jsonl", source_idx=0,
            prompt="x", expected='{"final_capacities": [1, 2, 3]}',
        )
        assert validate_question_passed("answer is [1, 2, 3]", q) is True
        assert validate_question_passed("answer is [9, 9, 9]", q) is False


# ---------------------------------------------------------------------------
# Phase 2.5 — Bench configs: minrung3 wiring + force_min_rung
# ---------------------------------------------------------------------------


def test_adaptive_current_minrung3_config() -> None:
    cfg = get_config("adaptive_current_minrung3")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("start_rung") == 3
    assert cfg.adaptive.get("parallel_rollouts") == 3
    assert cfg.force_min_rung == 3


def test_adaptive_a_minrung3_config() -> None:
    cfg = get_config("adaptive_a_minrung3")
    assert cfg.engine == "adaptive"
    assert cfg.adaptive.get("start_rung") == 3
    assert cfg.adaptive.get("prefer_shorter_traces") is True
    assert cfg.force_min_rung == 3


def test_ladder_policy_start_rung_short_circuits_baseline() -> None:
    """Offline test: LadderPolicy(start_rung=3).baseline_config() returns a
    rung-3 config (parallel_rollouts > 1), bypassing rungs 0/1/2."""
    from fabric_rlm.experimental.adaptive_policy import LadderPolicy

    pol = LadderPolicy(start_rung=3, parallel_rollouts=3)
    cfg = pol.baseline_config()
    assert cfg.rung == 3
    assert cfg.parallel_rollouts == 3


def test_ladder_policy_start_rung_zero_is_unchanged() -> None:
    """Default behavior is byte-identical to before: start_rung=0 => rung 0."""
    from fabric_rlm.experimental.adaptive_policy import LadderPolicy

    pol = LadderPolicy()  # start_rung defaults to 0
    cfg = pol.baseline_config()
    assert cfg.rung == 0
    assert cfg.parallel_rollouts == 1


def test_ladder_policy_start_rung_clamped_to_max_rung() -> None:
    """If start_rung exceeds max_rung (no strong_lm), clamp to max_rung."""
    from fabric_rlm.experimental.adaptive_policy import LadderPolicy

    pol = LadderPolicy(start_rung=99)
    cfg = pol.baseline_config()
    assert cfg.rung == pol.max_rung == 3


def test_ladder_policy_start_rung_first_decision_jumps_to_target() -> None:
    """Regression: ``next_decision([])`` (the path the AdaptiveRunner actually
    takes for the first attempt) must respect ``start_rung``. The default
    ``ValidatorOnly`` signal returns ``escalate(0)`` for empty attempts; the
    policy must still bump that to the configured ``start_rung`` so the runner
    starts at rung 3 / K=3 instead of rung 0 / K=1.
    """
    from fabric_rlm.experimental.adaptive_policy import LadderPolicy

    pol = LadderPolicy(start_rung=3, parallel_rollouts=3)
    verdict, cfg = pol.next_decision([])
    assert verdict.action == "escalate"
    assert cfg is not None
    assert cfg.rung == 3
    assert cfg.parallel_rollouts == 3

    # And the no-knob case stays at rung 0.
    pol0 = LadderPolicy()
    _, cfg0 = pol0.next_decision([])
    assert cfg0 is not None
    assert cfg0.rung == 0
    assert cfg0.parallel_rollouts == 1


def test_force_min_rung_propagates_through_rlm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end offline: passing adaptive={'start_rung': 3} via RLM(engine='adaptive')
    results in a LadderPolicy whose baseline_config is rung 3.

    Mirrors the monkeypatched-AdaptiveRunner pattern in test_adaptive_runtime.py.
    """
    import warnings

    from fabric_rlm import RLM
    from fabric_rlm.experimental import adaptive_runner as ar_mod

    captured: dict = {}

    class _CapturingRunner:
        def __init__(self, *, rlm_factory, policy, **_kw) -> None:
            captured["policy"] = policy
            captured["factory"] = rlm_factory

        def run(self, inputs, **_kw):
            class _R:
                class trajectory:
                    metadata: dict = {}
                payload = {"answer": "stub"}
                submitted = True
                failure_reason = None

            class _AR:
                result = _R()
                passed = True
                attempts = []
                winner = None
                stop_reason = "ok"
                elapsed_seconds = 0.0
            return _AR()

    monkeypatch.setattr(ar_mod, "AdaptiveRunner", _CapturingRunner)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rlm = RLM(
            signature="q -> a",
            lm="gpt-4.1-mini",
            engine="adaptive",
            adaptive={"validator": lambda r: True, "start_rung": 3},
        )
        rlm.run({"q": "hi"})

    pol = captured["policy"]
    assert pol.start_rung == 3
    cfg = pol.baseline_config()
    assert cfg.rung == 3
    assert cfg.parallel_rollouts >= 2
