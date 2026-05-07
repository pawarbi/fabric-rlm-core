"""Behavior gating test: runs the pinned suite against the configured model and
fails if PR results regress vs the committed baseline.

Two pytest markers select the model:

* ``@pytest.mark.primary``         — blocking; uses ``openai/gpt-4.1-mini``
* ``@pytest.mark.secondary_free``  — non-blocking (continue-on-error in CI);
  uses a free-tier OpenRouter model

Both tests skip locally without ``OPENROUTER_API_KEY``.  Inside GitHub Actions
the missing key is treated as a hard error so a misconfigured workflow can't
silently pass.

Behavior result JSON is written to ``--behavior-results=<path>`` if that arg is
passed on the pytest CLI; the workflow uploads it as an artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from . import baseline_loader as bl
from . import runner as runner_mod
from .questions import QUESTIONS, get_question, questions_sha256


_BASELINE_PATH = Path(__file__).parent / "baselines.json"
_PRIMARY_MODEL = os.environ.get("BEHAVIOR_PRIMARY_MODEL", "openai/gpt-4.1-mini")
_SECONDARY_FREE_MODEL = os.environ.get(
    "BEHAVIOR_SECONDARY_FREE_MODEL", "deepseek/deepseek-chat-v3.1:free"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_api_key() -> None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail(
            "OPENROUTER_API_KEY is required in GitHub Actions; behavior CI cannot run "
            "without it.  Configure the secret on the repository."
        )
    pytest.skip("OPENROUTER_API_KEY not set; skipping behavior gate locally.")


def _load_baseline_or_skip() -> bl.Baseline:
    if not _BASELINE_PATH.exists():
        pytest.skip(
            f"No baseline at {_BASELINE_PATH}; run "
            "`python -m tests.behavior.runner --calibrate` to bootstrap."
        )
    baseline = bl.load_baseline(_BASELINE_PATH)
    if baseline.questions_sha256 != questions_sha256():
        pytest.fail(
            "questions.py has changed since the baseline was calibrated "
            f"(baseline sha={baseline.questions_sha256[:12]}..., "
            f"current sha={questions_sha256()[:12]}...).  Recalibrate with "
            "`python -m tests.behavior.runner --calibrate`."
        )
    return baseline


def _write_results(request: pytest.FixtureRequest, payload: dict[str, Any]) -> None:
    try:
        out = request.config.getoption("--behavior-results", default=None)
    except (ValueError, KeyError):
        out = None
    if not out:
        return
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _run_gate(model: str, request: pytest.FixtureRequest) -> None:
    _require_api_key()
    baseline = _load_baseline_or_skip()
    if model not in baseline.models:
        pytest.skip(f"No baseline calibrated for model {model!r}; skipping.")

    mb = baseline.models[model]
    pr_results: dict[str, bool] = {}
    detail: list[dict[str, Any]] = []
    for q in QUESTIONS:
        res = runner_mod.run_question(
            q, model, max_turns=baseline.max_turns, timeout_s=float(baseline.timeout_s)
        )
        pr_results[q.qid] = res.passed
        detail.append({
            "qid": res.qid,
            "passed": res.passed,
            "reason": res.reason,
            "error_class": res.error_class,
            "n_turns": res.n_turns,
            "elapsed_s": res.elapsed_s,
            "attempts": res.attempts,
        })

    outcome = bl.evaluate_gates(mb, pr_results)
    payload = {
        "model": model,
        "baseline_passes": mb.baseline_passes,
        "min_passes": mb.min_passes,
        "stable_qids": list(mb.stable_qids()),
        "pr_results": pr_results,
        "passed": outcome.passed,
        "reasons": outcome.reasons,
        "detail": detail,
    }
    _write_results(request, payload)

    if not outcome.passed:
        msg = ["Behavior CI regression detected:"]
        msg.extend(f"  - {r}" for r in outcome.reasons)
        msg.append("Per-question detail:")
        for d in detail:
            tag = "PASS" if d["passed"] else "FAIL"
            msg.append(f"  [{tag}] {d['qid']}  reason={d['reason']!r}  err={d['error_class']!r}")
        pytest.fail("\n".join(msg), pytrace=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.primary
def test_behavior_baseline_primary(request: pytest.FixtureRequest) -> None:
    """Blocking gate against the primary calibrated model."""
    _run_gate(_PRIMARY_MODEL, request)


@pytest.mark.secondary_free
def test_behavior_baseline_secondary_free(request: pytest.FixtureRequest) -> None:
    """Informational gate against a free-tier model.

    The CI workflow runs this with ``continue-on-error: true`` so it never
    blocks merge — but failures still surface in the PR check log.
    """
    _run_gate(_SECONDARY_FREE_MODEL, request)
