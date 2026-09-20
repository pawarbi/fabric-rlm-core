"""A run may not hand back the path of a file it never wrote.

Found in Fabric: a 16-turn insurance review failed before ``wb.save`` in every
build turn, then submitted ``report_path`` with its KPIs. The payload had the
right type, the screen passed it, and the notebook crashed opening a workbook
that was not there. The check is narrow on purpose: a local path the task
supplied as an input, returned as an output, in a folder that exists.
"""

from __future__ import annotations

from pathlib import Path

from fabric_rlm import RLM
from fabric_rlm.analytical_integrity import check_submitted_paths_exist


def test_a_returned_input_path_with_no_file_is_a_problem(tmp_path: Path) -> None:
    target = str(tmp_path / "review.xlsx")
    problems = check_submitted_paths_exist({"report_path": target, "total": 3.0}, {"report_path": target})
    assert len(problems) == 1
    assert "report_path" in problems[0] and target in problems[0] and "never saved" in problems[0]


def test_nothing_is_flagged_when_the_file_is_there_or_the_case_is_doubtful(tmp_path: Path) -> None:
    written = tmp_path / "review.xlsx"
    written.write_bytes(b"x")
    missing = str(tmp_path / "other.xlsx")
    cases = [
        ({"report_path": str(written)}, {"report_path": str(written)}),          # saved
        ({"out_dir": str(tmp_path)}, {"out_dir": str(tmp_path)}),                # a folder that exists
        ({"report_path": missing}, {"template": str(written)}),                  # a path the run chose itself
        ({"month": "2025-06"}, {"month": "2025-06"}),                            # not a path
        ({"note": "see v1.2"}, {"note": "see v1.2"}),
        ({"p": "abfss://w@onelake.dfs.fabric.microsoft.com/l/Files/r.xlsx"},
         {"p": "abfss://w@onelake.dfs.fabric.microsoft.com/l/Files/r.xlsx"}),     # not checkable from here
        ({"p": "/no/such/folder/anywhere/r.xlsx"}, {"p": "/no/such/folder/anywhere/r.xlsx"}),   # another machine's path
        ({"report_path": missing}, {}),
        (None, {"report_path": missing}),
    ]
    for payload, inputs in cases:
        assert check_submitted_paths_exist(payload, inputs) == [], (payload, inputs)


def _scripted(*blocks: str):
    queue = ["```python\n" + block + "\n```" for block in blocks]
    return lambda messages=None, prompt=None, **kw: [queue.pop(0) if queue else "```python\nprint('idle')\n```"]


def test_a_run_is_sent_back_until_the_file_exists(tmp_path: Path) -> None:
    target = str(tmp_path / "brief.md")
    result = RLM.task(
        "Write the brief to brief_path and return it.",
        inputs={"brief_path": target},
        outputs={"brief_path": str},
        lm=_scripted(
            "SUBMIT(brief_path=brief_path)",
            "open(brief_path, 'w').write('# Brief')\nSUBMIT(brief_path=brief_path)",
        ),
        max_turns=4,
    ).run()

    assert result.submitted and Path(target).read_text() == "# Brief"
    assert [turn.turn_type for turn in result.turns] == ["normal", "verifier_repair"]
    assert result.integrity_ok
    assert "analytical-integrity screen rejected a submission" in result.report()


def test_the_screen_can_be_switched_off(tmp_path: Path) -> None:
    target = str(tmp_path / "brief.md")
    result = RLM.task(
        "Return brief_path.",
        inputs={"brief_path": target},
        outputs={"brief_path": str},
        lm=_scripted("SUBMIT(brief_path=brief_path)"),
        analytical_integrity=False,
        max_turns=2,
    ).run()
    assert result.submitted and len(result.turns) == 1


def test_a_missing_input_file_is_announced_before_the_run(tmp_path: Path) -> None:
    import warnings

    import pytest

    from fabric_rlm import File

    present = tmp_path / "orders.csv"
    present.write_text("a\n1\n")
    missing = tmp_path / "ordres.csv"
    lm = _scripted("SUBMIT(n=1)")

    with pytest.warns(UserWarning, match=r"input 'data' is File\(.*ordres\.csv.*no file exists there"):
        RLM.task("Count.", inputs={"data": File(str(missing))}, outputs={"n": int},
                 lm=lm, analytical_integrity=False, max_turns=1).run()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RLM.task("Count.", inputs={"data": [File(str(present))], "month": "2025-06"}, outputs={"n": int},
                 lm=_scripted("SUBMIT(n=1)"), analytical_integrity=False, max_turns=1).run()
