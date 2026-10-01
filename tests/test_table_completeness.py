"""Written-table completeness check, and the override sentence in the document guidance."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fabric_rlm import RLM, File
from fabric_rlm.analytical_integrity import check_written_tables_complete
from fabric_rlm.prompts import ask_each_section


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


def test_xlsx_and_tsv_are_checked(tmp_path: Path):
    import pandas as pd

    x = tmp_path / "out.xlsx"
    pd.DataFrame({"id": [1, 2], "v": ["a", None]}).to_excel(x, index=False)
    t = tmp_path / "out.tsv"
    t.write_text("id\tv\n1\t\n", encoding="utf-8")
    assert len(check_written_tables_complete({"a": str(x), "b": str(t)}, {})) == 2


def test_only_tables_this_run_wrote_are_judged(tmp_path: Path):
    path = tmp_path / "old.csv"
    path.write_text("id,a\n1,\n", encoding="utf-8")
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    assert check_written_tables_complete({}, {"out_path": str(path)}, started_at=time.time()) == []
    assert check_written_tables_complete({}, {"source": File(path)}) == []  # an input to read is never judged


def test_run_is_asked_to_fill_gaps_then_submits(tmp_path: Path):
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


def test_document_guidance_says_to_look_for_overriding_text():
    text = ask_each_section(documents=True)
    assert "notwithstanding" in text and "an exception can itself have an exception" in text


def test_on_the_last_turn_the_answer_is_kept_and_the_gap_is_reported(tmp_path: Path):
    # Rejecting on the final turn would leave no answer at all; a table with a few
    # blank cells is worth more than nothing, and the gap is still on the result.
    out = tmp_path / "t.csv"
    code = "open(out_path, 'w').write('id,v\\n1,\\n')\nSUBMIT(path=out_path)"

    class LM:
        def __call__(self, *, messages):
            return f"```python\n{code}\n```"

    result = RLM.task("Write the table.", inputs={"out_path": str(out)}, outputs={"path": str}, lm=LM(), max_turns=1, timeout=60).run()
    assert result.submitted
    assert any("empty cells" in p for p in result.integrity_problems)


def test_strict_mode_still_rejects_on_the_last_turn(tmp_path: Path):
    out = tmp_path / "t.csv"
    code = "open(out_path, 'w').write('id,v\\n1,\\n')\nSUBMIT(path=out_path)"

    class LM:
        def __call__(self, *, messages):
            return f"```python\n{code}\n```"

    result = RLM.task("Write the table.", inputs={"out_path": str(out)}, outputs={"path": str}, lm=LM(), max_turns=1,
                      timeout=60, analytical_integrity="strict").run()
    assert not result.submitted
