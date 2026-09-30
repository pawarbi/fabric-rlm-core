"""run_each (per-item child runs), the override sentence, and the written-table completeness check."""

from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import RLM, AskEach, File
from fabric_rlm.analytical_integrity import check_written_tables_complete
from fabric_rlm.prompts import ask_each_section


class RoleLM:
    """Parent and child runs share one LM; the system prompt tells them apart."""

    def __init__(self, parent_code: str, child_code: str):
        self.parent_code, self.child_code = parent_code, child_code
        self.systems: list[str] = []

    def __call__(self, *, messages):
        system = messages[0]["content"]
        self.systems.append(system)
        code = self.child_code if "  item: " in system else self.parent_code
        return f"```python\n{code}\n```"


def _docs(tmp_path: Path, n: int = 3) -> dict[str, File]:
    out = {}
    for i in range(n):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"Section 1. The fee is {100 * (i + 1)} dollars.\n", encoding="utf-8")
        out[f"d{i}"] = File(p)
    return out


PARENT = ("rows = run_each([docs[k] for k in sorted(docs)], 'Read the document and report its fee.', {'fee': int})\n"
          "SUBMIT(fees=[r['fee'] if r else None for r in rows], ok=rows.stats['ok'])")
CHILD = "import re\nt = item.read_text()\nSUBMIT(fee=int(re.search(r'fee is (\\d+)', t).group(1)))"


def test_run_each_runs_one_child_per_item_and_returns_aligned_payloads(tmp_path: Path):
    lm = RoleLM(PARENT, CHILD)
    result = RLM.task("Report each fee.", inputs={"docs": _docs(tmp_path)}, outputs={"fees": list, "ok": int}, lm=lm,
                      ask_each=AskEach(lm=lambda **k: {}, sub_runs=True, sub_run_turns=3), max_turns=3, timeout=120).run()
    assert result.payload == {"fees": [100, 200, 300], "ok": 3}
    parent_system = lm.systems[0]
    assert "run_each(items, task, outputs" in parent_system and "keep it in this run" in parent_system
    child_systems = [s for s in lm.systems if "  item: " in s]
    assert len(child_systems) == 3 and all("run_each(items" not in s for s in child_systems)  # no recursion
    records = [c for t in result.turns for c in (t.source_calls or []) if c.get("query_type") == "run_each"]
    assert records and records[0]["ok"] == 3
    assert result.trajectory.metadata["run_each"]["runs"] == 3 and result.trajectory.metadata["run_each"]["turns"] >= 3


def test_run_each_is_absent_unless_sub_runs_is_on(tmp_path: Path):
    code = "try:\n    run_each\n    present = 1\nexcept NameError:\n    present = 0\nSUBMIT(present=present)"
    lm = RoleLM(code, code)
    result = RLM.task("Is it there?", inputs={"docs": _docs(tmp_path)}, outputs={"present": int}, lm=lm,
                      ask_each=AskEach(lm=lambda **k: {}), max_turns=2, timeout=60).run()
    assert result.payload == {"present": 0} and "run_each" not in lm.systems[0]


def test_a_child_that_fails_is_none_with_an_error(tmp_path: Path):
    child = "t = item.read_text()\nif '200' in t:\n    raise ValueError('cannot read')\nSUBMIT(fee=1)"
    parent = ("rows = run_each([docs[k] for k in sorted(docs)], 'fee', {'fee': int})\n"
              "SUBMIT(got=[r is not None for r in rows], errors=len(rows.errors))")
    result = RLM.task("x", inputs={"docs": _docs(tmp_path)}, outputs={"got": list, "errors": int}, lm=RoleLM(parent, child),
                      ask_each=AskEach(lm=lambda **k: {}, sub_runs=True, sub_run_turns=2), max_turns=3, timeout=120).run()
    assert result.payload == {"got": [True, False, True], "errors": 1}


def test_sub_run_settings_are_checked():
    with pytest.raises(ValueError):
        AskEach(sub_run_turns=1)
    with pytest.raises(ValueError):
        AskEach(sub_run_concurrency=0)


def test_run_each_guidance_only_for_several_documents_with_sub_runs():
    assert "run_each(" not in ask_each_section(documents=True)
    assert "run_each(" not in ask_each_section(documents=True, several_documents=False)
    assert "run_each(" in ask_each_section(documents=True, several_documents=True)


def test_document_guidance_says_to_look_for_overriding_text():
    text = ask_each_section(documents=True)
    assert "notwithstanding" in text and "an exception can itself have an exception" in text


# ----------------------------------------------------------------- written-table completeness
def test_table_with_empty_cells_is_flagged_with_the_columns(tmp_path: Path):
    path = tmp_path / "out.csv"
    path.write_text("id,a,b\n1,x,\n2,,\n3,y,z\n", encoding="utf-8")
    problems = check_written_tables_complete({}, {"out_path": str(path)})
    assert len(problems) == 1
    assert "3 empty cells out of 9" in problems[0] and "b (2)" in problems[0] and "`not found: <why>`" in problems[0]


def test_table_with_every_cell_filled_or_marked_passes(tmp_path: Path):
    path = tmp_path / "out.csv"
    path.write_text("id,a\n1,n/a\n2,not found: no clause\n", encoding="utf-8")
    assert check_written_tables_complete({}, {"out_path": str(path)}) == []


def test_only_tables_this_run_wrote_are_judged(tmp_path: Path):
    import os, time

    path = tmp_path / "old.csv"
    path.write_text("id,a\n1,\n", encoding="utf-8")
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    assert check_written_tables_complete({}, {"out_path": str(path)}, started_at=time.time()) == []
    assert check_written_tables_complete({}, {"source": File(path)}) == []  # an input to read is never judged


def test_run_is_asked_once_to_fill_gaps_then_submits(tmp_path: Path):
    out = tmp_path / "t.csv"
    code = ("import os\n"
            "first = not os.path.exists(out_path)\n"
            "open(out_path, 'w').write('id,v\\n1,\\n' if first else 'id,v\\n1,n/a\\n')\n"
            "SUBMIT(path=out_path)")

    class LM:
        def __init__(self):
            self.feedback: list[str] = []

        def __call__(self, *, messages):
            self.feedback.append(messages[-1]["content"])
            return f"```python\n{code}\n```"

    lm = LM()
    result = RLM.task("Write the table.", inputs={"out_path": str(out)}, outputs={"path": str}, lm=lm, max_turns=4, timeout=60).run()
    assert result.submitted and out.read_text().strip().endswith("n/a")
    assert any("empty cells" in f for f in lm.feedback)


# ----------------------------------------------------------------- partial answers, coercion, inspect, tools engine
def test_a_child_out_of_turns_returns_its_last_answer_marked_partial(tmp_path: Path):
    # The child's SUBMIT is rejected (missing field) and it has no turn left: the parent still gets
    # the fields it did submit, flagged as partial and unchecked, instead of nothing.
    child = "SUBMIT(fee=7)"
    parent = ("rows = run_each([docs[k] for k in sorted(docs)], 'fee', {'fee': int, 'source': str})\n"
              "SUBMIT(fees=[r['fee'] if r else None for r in rows], partial=rows.stats['partial'], "
              "flags=[bool(e.get('partial')) for e in rows.errors])")
    result = RLM.task("x", inputs={"docs": _docs(tmp_path, 2)}, outputs={"fees": list, "partial": int, "flags": list},
                      lm=RoleLM(parent, child), ask_each=AskEach(lm=lambda **k: {}, sub_runs=True, sub_run_turns=2),
                      max_turns=3, timeout=120).run()
    assert result.payload == {"fees": [7, 7], "partial": 2, "flags": [True, True]}


def test_child_values_are_converted_to_the_requested_types_when_lossless():
    from fabric_rlm.interpreter import _coerce_child_payload

    out = _coerce_child_payload({"fee": "$48,701,040", "days": "4", "ok": "yes", "note": 3, "bad": "four"},
                                {"fee": float, "days": int, "ok": bool, "note": str, "bad": int})
    assert out == {"fee": 48701040.0, "days": 4, "ok": True, "note": "3", "bad": "four"}


def test_inspect_shows_child_runs_and_sub_run_totals(tmp_path: Path):
    lm = RoleLM(PARENT, CHILD)
    result = RLM.task("Report each fee.", inputs={"docs": _docs(tmp_path)}, outputs={"fees": list, "ok": int}, lm=lm,
                      ask_each=AskEach(lm=lambda **k: {}, sub_runs=True, sub_run_turns=3), max_turns=3, timeout=120).run()
    assert len(result.child_runs) == 3 and all(child.submitted for _, child in result.child_runs)
    html = result.inspect().to_html()
    assert "Child runs (3)" in html and "doc0.txt" in html and html.count("RLM run inspector") == 4
    assert html.count("<style") == 1  # nested views reuse the page styles


def test_inspect_of_a_plain_run_is_unchanged():
    code = "SUBMIT(answer=1)"
    result = RLM.task("x", outputs={"answer": int}, lm=RoleLM(code, code), max_turns=2, timeout=60).run()
    html = result.inspect().to_html()
    assert "Child runs" not in html and "ask_each items" not in html and "Unresolved checks" not in html


def test_run_each_works_in_the_tools_engine(tmp_path: Path):
    dspy = pytest.importorskip("dspy")

    class TwoEngineLM(dspy.LM):
        """Parent runs the tool-call engine (DSPy field format); children run the default engine (fenced code)."""

        def __init__(self):
            super().__init__(model="openai/scripted-offline", cache=False)

        def __call__(self, prompt=None, messages=None, **kwargs):
            text = "\n".join(str(m.get("content", "")) for m in (messages or []))
            if "  item: " in text:
                return [f"```python\n{CHILD}\n```"]
            return ["[[ ## reasoning ## ]]\nOne child run per document.\n\n"
                    f"[[ ## code ## ]]\n```python\n{PARENT}\n```\n\n[[ ## completed ## ]]\n"]

        def copy(self, **kwargs):
            return self

    def noop() -> int:
        """A registered tool, so the run uses the tool-call engine."""
        return 1

    result = RLM.task("Report each fee.", inputs={"docs": _docs(tmp_path)}, outputs={"fees": list, "ok": int}, lm=TwoEngineLM(),
                      ask_each=AskEach(lm=lambda **k: {}, sub_runs=True, sub_run_turns=3), tools=[noop],
                      max_turns=3, timeout=120).run()
    assert result.payload is not None and result.payload.get("fees") == [100, 200, 300]