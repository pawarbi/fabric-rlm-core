"""Checking what a workbook's formulas evaluate to, not just that they exist.

openpyxl never evaluates a formula and a file saved from Python has no cached
values, so "reopen and verify" cannot tell a right formula from a wrong one.
In Fabric two models shipped what-if models and dashboards that looked
finished and recalculated to wrong numbers. With this check wired in as an
``output_validator`` one run went from 20 wrong cells to none on the next
submission, and the other was correctly refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("formulas")
openpyxl = pytest.importorskip("openpyxl")

from fabric_rlm import (  # noqa: E402
    RLM,
    formula_errors,
    recalculate_workbook,
    workbook_formula_validator,
)


def _model(path: Path, *, wrong: bool = False, broken: bool = False) -> Path:
    wb = openpyxl.Workbook()
    a = wb.active
    a.title = "Assumptions"
    a["A2"], a["B2"] = "Migration share", 0.25
    a["A3"], a["B3"] = "Cost per policy", 15
    b = wb.create_sheet("Baseline")
    b.append(["Method", "Policies", "Premium", "Lapse"])
    b.append(["Autopay", 100, 1000.0, 0.02])
    b.append(["Manual", 40, 400.0, 0.15])
    s = wb.create_sheet("My Scenarios")
    lapse_row = 2 if not wrong else 3           # the wrong model lapses migrated premium at the manual rate
    s["B2"] = (
        "=Baseline!C3*(1-Assumptions!B2)*(1-Baseline!D3)"
        f"+Baseline!C3*Assumptions!B2*(1-Baseline!D{lapse_row})+Baseline!C2*(1-Baseline!D2)"
    )
    # Excel accepts the sheet name on both ends of a range; the engine does not.
    s["C2"] = "=Assumptions!B2*Baseline!B3*Assumptions!B3+IF(Assumptions!B2>0,SUM(Baseline!$B$2:Baseline!$B$3),0)"
    if broken:
        s["D2"] = '="text"*2'
    wb.save(path)
    return path


def _premium(share: float) -> float:
    return 400.0 * (1 - share) * 0.85 + 400.0 * share * 0.98 + 1000.0 * 0.98


def test_recalculates_formulas_that_openpyxl_only_stores(tmp_path: Path) -> None:
    path = _model(tmp_path / "model.xlsx")
    assert openpyxl.load_workbook(path, data_only=True)["My Scenarios"]["B2"].value is None

    values = recalculate_workbook(path)

    assert values["My Scenarios!B2"] == pytest.approx(_premium(0.25))
    assert values["'my scenarios'!$b$2"] == pytest.approx(_premium(0.25))   # case, quotes, anchors
    assert values["My Scenarios!C2"] == pytest.approx(0.25 * 40 * 15 + 140)
    assert "Assumptions!B2" in values and "Nowhere!A1" not in values


def test_changing_an_input_recalculates_and_leaves_the_file_alone(tmp_path: Path) -> None:
    path = _model(tmp_path / "model.xlsx")
    before = path.read_bytes()

    values = recalculate_workbook(path, {"Assumptions!B2": 0.6, "assumptions!b3": 40})

    assert values["My Scenarios!B2"] == pytest.approx(_premium(0.6))
    assert values["My Scenarios!C2"] == pytest.approx(0.6 * 40 * 40 + 140)
    assert path.read_bytes() == before


def test_an_unknown_sheet_in_inputs_is_a_clear_error(tmp_path: Path) -> None:
    path = _model(tmp_path / "model.xlsx")
    with pytest.raises(KeyError, match="no sheet 'Inputs'"):
        recalculate_workbook(path, {"Inputs!A1": 1})
    with pytest.raises(ValueError, match="must look like"):
        recalculate_workbook(path, {"A1": 1})


def test_formula_errors_lists_cells_excel_would_show_as_errors(tmp_path: Path) -> None:
    assert formula_errors(_model(tmp_path / "ok.xlsx")) == {}
    errors = formula_errors(_model(tmp_path / "broken.xlsx", broken=True))
    assert list(errors) == ["MY SCENARIOS!D2"] and errors["MY SCENARIOS!D2"].startswith("#")


SCENARIOS = [
    {"inputs": {}, "expected": {"My Scenarios!B2": _premium(0.25)}},
    {"inputs": {"Assumptions!B2": 0.8}, "expected": {"My Scenarios!B2": _premium(0.8)}},
]


def test_validator_accepts_a_correct_model(tmp_path: Path) -> None:
    path = _model(tmp_path / "model.xlsx")
    workbook_formula_validator(SCENARIOS)({"report_path": str(path)})
    workbook_formula_validator({"My Scenarios!B2": _premium(0.25)}, path_field="p")({"p": str(path)})


def test_validator_names_the_wrong_cell_and_both_values(tmp_path: Path) -> None:
    path = _model(tmp_path / "wrong.xlsx", wrong=True)
    with pytest.raises(AssertionError) as raised:
        workbook_formula_validator(SCENARIOS)({"report_path": str(path)})
    message = str(raised.value)
    assert "My Scenarios!B2 recalculates to" in message and "expected" in message
    assert "scenario 2 with {'Assumptions!B2': 0.8}" in message
    assert "do not hard-code" in message


def test_hard_coded_numbers_cannot_satisfy_a_second_scenario(tmp_path: Path) -> None:
    path = _model(tmp_path / "model.xlsx")
    wb = openpyxl.load_workbook(path)
    wb["My Scenarios"]["B2"] = _premium(0.25)          # a constant that matches scenario 1 only
    wb.save(path)
    with pytest.raises(AssertionError, match="scenario 2"):
        workbook_formula_validator(SCENARIOS)({"report_path": str(path)})


def test_validator_rejects_a_missing_file_and_empty_expectations(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="does not name a saved workbook"):
        workbook_formula_validator(SCENARIOS)({"report_path": str(tmp_path / "none.xlsx")})
    with pytest.raises(ValueError, match="at least one expected cell"):
        workbook_formula_validator([{"inputs": {}, "expected": {}}])


def test_a_run_is_sent_back_until_the_workbook_recalculates_correctly(tmp_path: Path) -> None:
    path = tmp_path / "model.xlsx"
    helper = Path(__file__).read_text(encoding="utf-8")
    build = helper[helper.index("def _model("):helper.index("def _premium(")]
    prelude = "import openpyxl\nfrom pathlib import Path\n" + build

    def turn(wrong: bool) -> str:
        return (
            f"```python\n{prelude}\n_model(Path(r'{path}'), wrong={wrong})\n"
            f"SUBMIT(report_path=r'{path}')\n```"
        )

    replies = iter([turn(True), turn(False)])
    result = RLM.task(
        task="build the model",
        outputs={"report_path": str},
        lm=lambda messages=None, prompt=None, **kw: [next(replies)],
        output_validator=workbook_formula_validator(SCENARIOS),
        analytical_integrity=False,
        max_turns=4,
        timeout=120,
    ).run()

    assert result.submitted, result.report()
    assert [t.turn_type for t in result.turns] == ["normal", "verifier_repair"]
    assert recalculate_workbook(path)["My Scenarios!B2"] == pytest.approx(_premium(0.25))
