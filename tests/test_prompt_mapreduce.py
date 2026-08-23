"""Parallel map-reduce idiom: capability + PDF-skill guidance.

The worker supports ``await predict(...)`` inside a running event loop
(nest_asyncio), so ``asyncio.gather`` over chunked sub-LM calls works. These
tests pin:

1. The idiom actually executes: ``asyncio.gather`` over multiple ``predict()``
   calls works inside the worker's running-loop context, isolates failures
   (``return_exceptions=True``), and runs concurrently.
2. The PDF skill (which already taught page-level gather) teaches it in a
   *bounded, fault-tolerant* form.

Note: an always-on base-prompt paragraph teaching this idiom was prototyped and
A/B-tested (branch ``feat/mapreduce-prompt``). Neither gpt-3.5-turbo nor
gpt-4.1-mini adopted predict()-based map-reduce even when single-pass
demonstrably failed (the orchestrator guessed items it could not see rather than
delegating). The always-on block was therefore dropped as dead weight; the
capability and the routed PDF-skill guidance remain.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import dspy

import fabric_rlm
from fabric_rlm import _worker


# ---------------------------------------------------------------------------
# 1. PDF-skill content: bounded, fault-tolerant gather.
# ---------------------------------------------------------------------------

def _pdf_skill_text() -> str:
    skill = Path(fabric_rlm.__file__).parent / "skills" / "pdf_document_analysis.md"
    return skill.read_text(encoding="utf-8")


def test_pdf_skill_teaches_bounded_fault_tolerant_gather():
    text = _pdf_skill_text()
    assert "asyncio.gather" in text
    # bounded fan-out (not naive gather-over-everything)
    assert "batch" in text.lower() or "at a time" in text.lower()
    # one bad page must not lose the batch
    assert "return_exceptions=True" in text


def test_pdf_skill_checks_form_widgets_before_falling_back_to_rendering():
    text = _pdf_skill_text()
    widget_pos = text.index("page.widgets()")
    render_pos = text.index("Render candidate or all pages")

    assert widget_pos < render_pos
    assert "field_name" in text
    assert "field_value" in text


# ---------------------------------------------------------------------------
# 2. Capability: the idiom actually runs.
# ---------------------------------------------------------------------------

def test_gather_over_predict_executes_and_preserves_results(monkeypatch):
    """`await asyncio.gather(predict(...), ...)` works inside a running loop."""

    class FakePredict:
        def __init__(self, sig):
            pass

        async def acall(self, **kwargs):
            await asyncio.sleep(0.01)
            return dspy.Prediction(label=f"L:{kwargs.get('text')}")

    monkeypatch.setattr(dspy, "Predict", FakePredict)
    monkeypatch.setattr(_worker, "_get_lm", lambda: object())

    chunks = ["a", "b", "c", "d", "e"]

    async def driver():
        # This mirrors exactly what the prompt tells the model to write.
        calls = [_worker.predict("text -> label", text=c) for c in chunks]
        return await asyncio.gather(*calls, return_exceptions=True)

    results = asyncio.run(driver())
    assert [r.label for r in results] == [f"L:{c}" for c in chunks]


def test_gather_return_exceptions_isolates_one_bad_chunk(monkeypatch):
    """One failing chunk must not lose the whole batch (return_exceptions=True)."""

    class FakePredict:
        def __init__(self, sig):
            pass

        async def acall(self, **kwargs):
            t = kwargs.get("text")
            if t == "boom":
                raise ValueError("bad chunk")
            return dspy.Prediction(label=f"L:{t}")

    monkeypatch.setattr(dspy, "Predict", FakePredict)
    monkeypatch.setattr(_worker, "_get_lm", lambda: object())

    chunks = ["a", "boom", "c"]

    async def driver():
        calls = [_worker.predict("text -> label", text=c) for c in chunks]
        return await asyncio.gather(*calls, return_exceptions=True)

    results = asyncio.run(driver())
    assert results[0].label == "L:a"
    assert isinstance(results[1], Exception)
    assert results[2].label == "L:c"


def test_gather_runs_concurrently(monkeypatch):
    """Bounded parallel fan-out is actually concurrent (latency win)."""
    import time

    class FakePredict:
        def __init__(self, sig):
            pass

        async def acall(self, **kwargs):
            await asyncio.sleep(0.05)
            return dspy.Prediction(ok=True)

    monkeypatch.setattr(dspy, "Predict", FakePredict)
    monkeypatch.setattr(_worker, "_get_lm", lambda: object())

    async def driver():
        calls = [_worker.predict("t -> ok", t=str(i)) for i in range(8)]
        return await asyncio.gather(*calls)

    t0 = time.time()
    results = asyncio.run(driver())
    elapsed = time.time() - t0
    assert len(results) == 8
    # 8 x 50ms sequential would be ~400ms; concurrent should be well under.
    assert elapsed < 0.25
