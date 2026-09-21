"""A run may not finish on a workbook it wrote that no longer opens.

Found in Fabric: a run that could not read its conditional formatting back
wrote a bad key into openpyxl's rule store, ``wb.save`` raised partway and left
2,298 bytes of zip at ``report_path``. The last turn submitted the figures, the
screen passed them, ``integrity_ok`` was true, and the notebook crashed opening
the report. openpyxl, python-docx and pyarrow write straight to the
destination, so any save that raises leaves a file like this behind.

The path was an input only, never part of the payload, so the check reads both.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import pytest

from fabric_rlm import RLM, File
from fabric_rlm.analytical_integrity import check_written_files_open

openpyxl = pytest.importorskip("openpyxl")


def _workbook(path: Path) -> Path:
    wb = openpyxl.Workbook()
    wb.active["A1"] = 1
    wb.save(path)
    return path


def _what_the_failed_save_left(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("docProps/app.xml", "<Properties/>")
        archive.writestr("xl/theme/theme1.xml", "<theme/>")
    return path


def test_a_workbook_cut_short_by_a_failed_save_is_a_problem(tmp_path: Path) -> None:
    target = str(_what_the_failed_save_left(tmp_path / "streaks.xlsx"))
    # The shape that was missed: the path is an input, the payload holds figures only.
    problems = check_written_files_open({"n_analyzed": 166}, {"report_path": target})
    assert len(problems) == 1
    assert "report_path" in problems[0] and target in problems[0] and "does not open" in problems[0]
    assert "[Content_Types].xml" in problems[0]


@pytest.mark.parametrize("suffix", [".xlsx", ".xlsm", ".docx", ".pptx"])
@pytest.mark.parametrize("damage", ["empty", "truncated", "text"])
def test_every_office_container_is_judged_the_same_way(tmp_path: Path, suffix: str, damage: str) -> None:
    good = _workbook(tmp_path / "good.xlsx").read_bytes()
    target = tmp_path / f"report{suffix}"
    target.write_bytes({"empty": b"", "truncated": good[: len(good) // 2], "text": b"Traceback (most recent call last)"}[damage])
    problems = check_written_files_open({"out": str(target)}, {})
    assert len(problems) == 1 and "out is" in problems[0]


def test_a_parquet_file_without_its_footer_is_a_problem(tmp_path: Path) -> None:
    whole = tmp_path / "whole.parquet"
    whole.write_bytes(b"PAR1" + b"\x00" * 64 + b"PAR1")
    cut = tmp_path / "cut.parquet"
    cut.write_bytes(b"PAR1" + b"\x00" * 64)
    assert check_written_files_open({"a": str(whole)}, {}) == []
    assert len(check_written_files_open({"a": str(cut)}, {})) == 1


def test_nothing_is_flagged_when_the_file_is_fine_or_the_case_is_doubtful(tmp_path: Path) -> None:
    good = str(_workbook(tmp_path / "good.xlsx"))
    broken = str(_what_the_failed_save_left(tmp_path / "broken.xlsx"))
    note = tmp_path / "notes.csv"
    note.write_text("not,a,container\n", encoding="utf-8")
    folder = tmp_path / "table.parquet"
    folder.mkdir()
    cases = [
        ({"report_path": good}, {"report_path": good}),                      # opens
        ({}, {"report_path": str(tmp_path / "not_written.xlsx")}),           # absent: the file-exists check owns that
        ({"p": str(note)}, {}),                                              # a format with no exact signature
        ({"p": str(folder)}, {}),                                            # a Spark-style folder, not a file
        ({"p": "abfss://w@onelake.dfs.fabric.microsoft.com/l/Files/r.xlsx"}, {}),   # not checkable from here
        ({"p": "broken.xlsx"}, {}),                                          # not a path
        ({"rows": [broken]}, {}),                                            # only top-level strings are read
        ({}, {"data": File(broken)}),                                        # a File input is read, not written
        (None, None),
    ]
    for payload, inputs in cases:
        assert check_written_files_open(payload, inputs) == [], (payload, inputs)


def test_a_broken_file_that_was_there_before_the_run_is_not_this_runs_problem(tmp_path: Path) -> None:
    old = _what_the_failed_save_left(tmp_path / "old.xlsx")
    hour_ago = time.time() - 3600
    os.utime(old, (hour_ago, hour_ago))
    assert check_written_files_open({}, {"template": str(old)}, started_at=time.time()) == []
    assert len(check_written_files_open({}, {"template": str(old)}, started_at=None)) == 1
    # A mounted store stamps files with its own clock: a small skew must not hide a file this run wrote.
    skewed = time.time() - 30
    os.utime(old, (skewed, skewed))
    assert len(check_written_files_open({}, {"template": str(old)}, started_at=time.time())) == 1


def _scripted(*blocks: str):
    queue = ["```python\n" + block + "\n```" for block in blocks]
    return lambda messages=None, prompt=None, **kw: [queue.pop(0) if queue else "```python\nprint('idle')\n```"]


BREAK_THE_SAVE = (
    "import openpyxl\n"
    "wb = openpyxl.Workbook()\n"
    "wb.active['A1'] = 1\n"
    "wb.active.conditional_formatting._cf_rules['E3:E17'] = []\n"   # what the run did: a str key
    "try:\n"
    "    wb.save(report_path)\n"
    "except Exception as exc:\n"
    "    print('save failed:', type(exc).__name__)\n"
)
GOOD_SAVE = "import openpyxl\nwb = openpyxl.Workbook()\nwb.active['A1'] = 1\nwb.save(report_path)\nprint('saved')"


def test_a_run_is_sent_back_until_the_workbook_opens(tmp_path: Path) -> None:
    target = tmp_path / "report.xlsx"
    result = RLM.task(
        "Build the workbook at report_path and return the row count.",
        inputs={"report_path": str(target)},
        outputs={"rows": int},
        lm=_scripted(BREAK_THE_SAVE, "SUBMIT(rows=1)", GOOD_SAVE, "SUBMIT(rows=1)"),
        max_turns=6,
    ).run()

    assert result.submitted and result.integrity_ok
    assert [turn.turn_type for turn in result.turns] == ["normal", "normal", "verifier_repair", "normal"]
    assert "save failed" in result.turns[0].stdout, "openpyxl no longer raises on this; break the save another way"
    assert openpyxl.load_workbook(target).active["A1"].value == 1
    assert "analytical-integrity screen rejected a submission" in result.report()


def test_a_workbook_that_opens_costs_nothing(tmp_path: Path) -> None:
    target = tmp_path / "report.xlsx"
    result = RLM.task(
        "Build the workbook at report_path and return the row count.",
        inputs={"report_path": str(target)},
        outputs={"rows": int},
        lm=_scripted(GOOD_SAVE, "SUBMIT(rows=1)"),
        max_turns=3,
    ).run()
    assert result.submitted and result.integrity_ok and len(result.turns) == 2


def test_the_screen_can_be_switched_off(tmp_path: Path) -> None:
    target = tmp_path / "report.xlsx"
    result = RLM.task(
        "Build the workbook at report_path and return the row count.",
        inputs={"report_path": str(target)},
        outputs={"rows": int},
        lm=_scripted(BREAK_THE_SAVE, "SUBMIT(rows=1)"),
        analytical_integrity=False,
        max_turns=3,
    ).run()
    assert result.submitted and len(result.turns) == 2
