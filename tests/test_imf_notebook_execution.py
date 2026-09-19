"""Offline execution of the shipped notebook's cells, not copied implementations."""

import ast
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("duckdb")
requests = pytest.importorskip("requests")
openpyxl = pytest.importorskip("openpyxl")
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font, PatternFill


NOTEBOOK = Path(__file__).parents[1] / "examples/notebooks/rlm_vs_plain_llm_imf_cpi.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
# Whitespace-normalized snapshot fetched in memory from WEO_URL; SHA-256 matches
# cb0c2459c63b5d7d3d310fb6f0d6c32ee6624b79050dcf644e55a109a07bf360.
CORE_SENTENCE = (
    "Core inflation is expected to return to target only gradually in several major economies: "
    "by mid-2027 in the United Kingdom, by the end of 2027 in Japan and the United States, "
    "and only in 2028 in the euro area."
)


def source(marker):
    matches = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "code"
               and marker in "".join(c["source"])]
    assert len(matches) == 1, marker
    return matches[0]


def execute(marker, namespace, definitions=False):
    tree = ast.parse(source(marker))
    if definitions:
        tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(tree, f"{NOTEBOOK.name}:{marker}", "exec"), namespace)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network access is forbidden in notebook execution tests")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)


@pytest.fixture
def namespace():
    ns = {}
    execute("def download_cpi", ns, definitions=True)
    execute("def grade_report", ns, definitions=True)
    execute("def streak_reference", ns, definitions=True)
    return ns


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "country's CPI.csv"
    header = ["COUNTRY", "INDEX_TYPE", "COICOP_1999", "TYPE_OF_TRANSFORMATION",
              "FREQUENCY", "TIME_PERIOD", "OBS_VALUE"]
    months = [f"{year}-M{month:02}" for year in range(2021, 2026) for month in range(1, 13)]
    records = []
    for country in range(18):
        for i, month in enumerate(months):
            records.append([f"C{country:02}", "CPI", "_T", "YOY_PCH_PA_PT", "M", month,
                            country * 10 + i // 12 + (i % 12) / 10])
    base = records[:60]
    # Missing, duplicated, nonnumeric, infinite and malformed months must not qualify.
    for name, observations in [("MISS", base[:-1]), ("DUP", base[:-1] + base[:1]),
                               ("EXTRA", base + base[:1])]:
        records.extend([[name, *row[1:]] for row in observations])
    for name, column, value in [("BAD", 6, "missing"), ("INF", 6, "inf"),
                                ("MONTH", 5, "2025-M13")]:
        observations = [[name, *row[1:]] for row in base]
        observations[-1][column] = value
        records.extend(observations)
    for column, value in [(1, "OTHER"), (2, "FOOD"), (3, "INDEX"), (4, "A")]:
        observations = [[f"FILTER{column}", *row[1:]] for row in base]
        for row in observations:
            row[column] = value
        records.extend(observations)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(reversed(records))
    # Closed-form expectations, independent of the notebook's SQL and streak algorithm.
    rows = [[f"C{c:02}", *[c * 10 + y + .55 for y in range(5)], c * 10 + 2.55]
            for c in reversed(range(18))]
    truth = {"n_countries": 18, "top10": rows[:10], "median_avg": 87.55,
             "all_countries": [[r[0], r[6]] for r in rows]}
    streaks = [[f"C{c:02}", 60, "2021-M01", "2025-M12", c * 10 + 5.1, "2025-M12"]
               for c in range(1, 18)]
    return path, truth, rows, streaks


def report_workbook(path, truth, gap=None):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["Average year-over-year CPI inflation (%), 2021-2025"])
    ws.merge_cells(f"A1:{'H' if gap is not None else 'G'}1")
    ws.append(["Country", 2021, 2022, 2023, 2024, 2025, "Avg 2021-2025"]
              + (["Gap vs world 2025 (pp)"] if gap is not None else []))
    for cell in ws[2]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9D9D9")
    ws.column_dimensions["A"].width = 32
    for i, row in enumerate(truth["top10"]):
        ws.append(row + ([gap[i]] if gap is not None else []))
    ws["A13"] = "Median (all qualifying countries)"
    ws["G13"] = truth["median_avg"]
    allc = wb.create_sheet("All countries")
    allc.append(["Country", "Avg 2021-2025"])
    for cell in allc[1]:
        cell.font = Font(bold=True)
    for row in truth["all_countries"]:
        allc.append(row)
    for sheet in (ws, allc):
        for row in sheet.iter_rows(min_row=3 if sheet is ws else 2, min_col=2):
            for cell in row:
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.00"
    if gap is not None:
        wb.create_sheet("IMF Outlook")
    wb.save(path)
    return wb


def test_cells_compile_and_outputs_are_empty():
    for cell in CELLS:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            code = "".join(cell["source"])
            if not code.startswith("%pip"):
                compile(code, cell["id"], "exec")


def test_ground_truth_and_grading_cells(namespace, reference, tmp_path):
    path, truth, rows, streaks = reference
    namespace["DATA_PATH"] = str(path)
    execute("con = duckdb.connect()", namespace)
    assert namespace["truth"] == truth
    assert namespace["streak_reference"](path) == (18, streaks)
    report = tmp_path / "report.xlsx"
    report_workbook(report, truth)
    namespace.update(REPORT_PATH=report, result=SimpleNamespace(submitted=True, payload={"n_countries": 18, "median_avg": 87.55}))
    execute("def grade_report", namespace)
    assert all(namespace["checks"].values())
    namespace["result"].submitted = False
    with pytest.raises(AssertionError):
        execute("def grade_report", namespace)
    namespace["result"].submitted = True
    namespace["result"].payload["n_countries"] = 17
    with pytest.raises(AssertionError):
        execute("def grade_report", namespace)
    namespace["con"].close()


@pytest.mark.parametrize("gap", [None, [1.25] * 10])
@pytest.mark.parametrize("mutation", [
    "sheet", "title", "merge", "header", "bold", "fill", "fill_pattern", "width",
    "format", "value", "country", "median_label", "median", "median_format",
    "all_header", "all_bold", "all_country", "all_value", "all_format", "all_order", "all_extra",
])
def test_report_rejects_wrong_workbooks(namespace, reference, tmp_path, gap, mutation):
    truth = reference[1]
    path = tmp_path / "report.xlsx"
    wb = report_workbook(path, truth, gap)
    assert all(namespace["grade_report"](path, truth, gap).values())
    ws, allc = wb["Report"], wb["All countries"]
    if mutation == "sheet":
        wb.create_sheet("Unexpected")
    elif mutation == "merge":
        ws.unmerge_cells(f"A1:{'H' if gap else 'G'}1")
    elif mutation in {"bold", "fill", "fill_pattern"}:
        if mutation == "bold":
            ws["G2"].font = Font(bold=False)
        else:
            ws["G2"].fill = PatternFill("solid" if mutation == "fill" else None, fgColor="FFFFFF" if mutation == "fill" else "D9D9D9")
    elif mutation == "width":
        ws.column_dimensions["A"].width = 31
    elif mutation in {"format", "median_format", "all_format"}:
        {"format": ws["G12"], "median_format": ws["G13"], "all_format": allc["B18"]}[mutation].number_format = "General"
    elif mutation == "all_bold":
        allc["B1"].font = Font(bold=False)
    elif mutation == "all_order":
        a, b = [c.value for c in allc[17]], [c.value for c in allc[18]]
        for col in (1, 2):
            allc.cell(17, col, b[col - 1])
            allc.cell(18, col, a[col - 1])
    elif mutation == "all_extra":
        allc.append(["XXX", 0])
    else:
        cell, value = {
            "title": (ws["A1"], "wrong"), "header": (ws["F2"], 2026),
            "value": (ws["F12"], ws["F12"].value + .01), "country": (ws["A12"], "XXX"),
            "median_label": (ws["A13"], "Median"), "median": (ws["G13"], 0),
            "all_header": (allc["B1"], "Average"), "all_country": (allc["A18"], "XXX"),
            "all_value": (allc["B18"], 1000),
        }[mutation]
        cell.value = value
    wb.save(path)
    assert not all(namespace["grade_report"](path, truth, gap).values()), mutation


@pytest.mark.parametrize("value", [None, "1.23", True, float("nan"), float("inf"), 1.24])
def test_numeric_checker_rejects_invalid_values(namespace, value):
    assert not namespace["ok"](value, 1.23)
    assert namespace["ok"](1.23000000001, 1.23)


@pytest.mark.parametrize("failure", ["stream", "http", "empty", "html", "header_only", "wrong_schema"])
@pytest.mark.parametrize("existing", [False, True])
def test_atomic_download_failure_and_retry(namespace, monkeypatch, tmp_path, failure, existing):
    path = tmp_path / "cpi.csv"
    old = b"previous file, possibly partial"
    if existing:
        path.write_bytes(old)
    header = b"COUNTRY,INDEX_TYPE,COICOP_1999,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE\n"
    valid = header + b"USA,CPI,_T,YOY_PCH_PA_PT,M,2021-M01,1.2\n"
    response = MagicMock()
    response.__enter__.return_value = response
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value = response
    monkeypatch.setattr(requests, "Session", lambda: session)

    def chunks():
        yield header
        assert path.read_bytes() == old if existing else not path.exists()
        raise requests.exceptions.ChunkedEncodingError("interrupted")

    response.iter_content.return_value = {
        "stream": chunks(), "http": [], "empty": [], "html": [b"<html>error</html>"],
        "header_only": [header], "wrong_schema": [b"a,b\n1,2\n"],
    }[failure]
    if failure == "http":
        response.raise_for_status.side_effect = requests.HTTPError("503")
    with pytest.raises((ValueError, requests.RequestException)):
        namespace["download_cpi"]("https://offline.invalid/cpi", path)
    assert path.read_bytes() == old if existing else not path.exists()
    response.__exit__.assert_called()
    response.raise_for_status.side_effect = None
    response.iter_content.return_value = [valid[:30], valid[30:]]
    namespace["download_cpi"]("https://offline.invalid/cpi", path)
    assert path.read_bytes() == valid
    assert not Path(str(path) + ".part").exists()
    # A second successful run must download again, not trust the existing header.
    response.iter_content.return_value = [valid.replace(b"1.2", b"2.3")]
    namespace["download_cpi"]("https://offline.invalid/cpi", path)
    assert path.read_bytes().endswith(b"2.3\n")
    assert session.get.call_count == 3
    assert session.get.call_args.kwargs["stream"] is True
    assert session.mount.call_args.args[1].max_retries.total == 3


def streak_workbook(path, analyzed, expected):
    wb = openpyxl.Workbook()
    st = wb.active
    st.title = "Streaks"
    st.append(["Longest high-inflation streaks (YoY >= 10%), 2021-2025"])
    st.merge_cells("A1:F1")
    st.append(["Country", "Months", "Start", "End", "Peak YoY", "Peak month"])
    for cell in st[2]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9D9D9")
    for col in ("A", "F"):
        st.column_dimensions[col].width = 14
    for row in expected[:15]:
        st.append(row)
    for row in range(3, 18):
        st.cell(row, 5).number_format = "0.00"
    st.conditional_formatting.add("E3:E17", ColorScaleRule(start_type="min", start_color="FFFFFF", end_type="max", end_color="FF0000"))
    chart = BarChart()
    chart.title = "Peak YoY, top 10"
    chart.add_data(Reference(st, min_col=5, min_row=3, max_row=12))
    chart.set_categories(Reference(st, min_col=1, min_row=3, max_row=12))
    st.add_chart(chart, "H2")
    summary = wb.create_sheet("Summary")
    for row in [["Countries analyzed", analyzed], ["Countries with a streak", len(expected)],
                ["Longest streak country", expected[0][0]], ["Longest streak months", expected[0][1]]]:
        summary.append(row)
    wb.save(path)
    return wb


@pytest.mark.parametrize("mutation", [None, "late_row", "summary", "format", "width", "fill", "chart", "anchor", "series", "categories", "chart_title", "scale"])
def test_streak_grader(namespace, reference, tmp_path, mutation):
    path = tmp_path / "streaks.xlsx"
    expected = reference[3]
    wb = streak_workbook(path, 18, expected)
    st = wb["Streaks"]
    if mutation == "late_row":
        st["F17"] = "2021-M01"
    elif mutation == "summary":
        wb["Summary"]["B2"] = 1
    elif mutation == "format":
        st["E17"].number_format = "General"
    elif mutation == "width":
        st.column_dimensions["F"].width = 15
    elif mutation == "fill":
        st["F2"].fill = PatternFill("solid", fgColor="FFFFFF")
    elif mutation == "chart":
        st._charts.clear()
    elif mutation == "anchor":
        st._charts[0].anchor = "H3"
    elif mutation == "series":
        st._charts[0].series[0].val.numRef.f = "'Streaks'!$E$3:$E$11"
    elif mutation == "categories":
        st._charts[0].series[0].cat.numRef.f = "'Streaks'!$B$3:$B$12"
    elif mutation == "chart_title":
        st._charts[0].title = "Wrong"
    elif mutation == "scale":
        st.conditional_formatting["E3:E17"][0].colorScale.color[1].rgb = "0000FF"
    wb.save(path)
    result = SimpleNamespace(submitted=True, payload={"n_analyzed": 18, "n_with_streak": 17,
                                                     "longest_country": "C01", "longest_len": 60})
    checks = namespace["grade_streaks"](path, 18, expected, result)
    assert all(checks.values()) == (mutation is None), checks
    namespace.update(DATA_PATH=reference[0], REPORT2_PATH=path, result2=result)
    if mutation is None:
        execute("def streak_reference", namespace)
    else:
        with pytest.raises(AssertionError):
            execute("def streak_reference", namespace)


@pytest.mark.parametrize("skills", [True, False])
@pytest.mark.parametrize("field", [None, "submitted", "n_analyzed", "n_with_streak", "longest_country", "longest_len"])
@pytest.mark.parametrize("missing", [False, True])
def test_streak_submission(namespace, reference, tmp_path, skills, field, missing):
    result = SimpleNamespace(submitted=True, payload={"n_analyzed": 18, "n_with_streak": 17,
                                                     "longest_country": "C01", "longest_len": 60},
                             n_turns=1, total_prompt_tokens=0, total_completion_tokens=0)
    if field == "submitted":
        result.submitted = False
    elif field:
        if missing:
            del result.payload[field]
        else:
            result.payload[field] = "WRONG" if field == "longest_country" else -1
    path = tmp_path / "streaks.xlsx"
    streak_workbook(path, 18, reference[3])
    namespace.update(DATA_PATH=reference[0], REPORT2_PATH=path, result2=result,
                     analyzed=18, expected_streaks=reference[3], DATA_DIR=str(tmp_path), STAMP="test",
                     Path=Path, os=os, time=time, STREAKS_TASK="offline", File=lambda p: p, lm_mini=None)

    def task(**kwargs):
        def run():
            streak_workbook(kwargs["inputs"]["report_path"], 18, reference[3])
            return result
        return SimpleNamespace(run=run)

    namespace["RLM"] = SimpleNamespace(task=task)
    marker = "def streak_reference" if skills else "result3 = rlm3.run()"
    if field is None:
        execute(marker, namespace)
    else:
        with pytest.raises(AssertionError):
            execute(marker, namespace)


def test_report_paths_use_fresh_shared_stamp(namespace, tmp_path):
    # Execute actual setup assignments without downloads, Fabric clients or model calls.
    setup = ast.parse(source("def download_cpi"))
    setup.body = [n for n in setup.body if isinstance(n, (ast.Import, ast.ImportFrom))
                  or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STAMP" for t in n.targets)]
    namespace["DATA_DIR"] = str(tmp_path)
    paths = ["REPORT_PATH", "REPORT2_PATH", "NOSKILLS_PATH", "REPORT3_PATH"]
    previous = set()
    for _ in range(2):
        exec(compile(setup, "notebook-setup", "exec"), namespace)
        for name in paths:
            tree = ast.parse(source(f"{name} ="))
            tree.body = [n for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
            exec(compile(tree, "notebook-path", "exec"), namespace)
        current = {namespace[name] for name in paths}
        assert len(current) == 4
        assert not current & previous
        assert all(namespace["STAMP"] in Path(p).name for p in current)
        previous = current


@pytest.mark.parametrize("result_name,path_name", [
    ("result", "REPORT_PATH"), ("result2", "REPORT2_PATH"),
    ("result3", "NOSKILLS_PATH"), ("result4", "REPORT3_PATH"),
])
def test_failed_rerun_removes_stale_report(namespace, reference, tmp_path, monkeypatch, result_name, path_name):
    path = tmp_path / ("streaks_noskills_test.xlsx" if path_name == "NOSKILLS_PATH" else "report.xlsx")
    namespace.update(DATA_DIR=str(tmp_path), STAMP="test", DATA_PATH=reference[0], WEO_PATH="offline.pdf",
                     Path=Path, os=os, time=time, TASK="offline", STREAKS_TASK="offline",
                     TWO_SOURCE_TASK="offline", lm_mini=None, FabricLM=lambda *a, **kw: None)
    namespace[path_name] = str(path)

    def task(**kwargs):
        assert kwargs["inputs"]["report_path"] == str(path)
        assert not path.exists(), "Previous report survived into a new model call"
        return SimpleNamespace(run=lambda: SimpleNamespace(submitted=False, payload={}))

    fake = SimpleNamespace(task=task)
    namespace.update(RLM=fake, File=lambda p: p)
    monkeypatch.setitem(sys.modules, "fabric_rlm", SimpleNamespace(RLM=fake, File=lambda p: p))
    tree = ast.parse(source(f"\n{result_name} = "))
    # Run through the submission assertion, excluding subsequent display/grading.
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assert)
               and isinstance(n.test, ast.Attribute) and n.test.attr == "submitted")
    tree.body = tree.body[:end + 1]
    for _ in range(2):
        streak_workbook(path, 18, reference[3])
        with pytest.raises(AssertionError):
            exec(compile(tree, "notebook-failed-rerun", "exec"), namespace)
        assert not path.exists()


def test_streak_reference_edges_and_ties(namespace, tmp_path):
    path = tmp_path / "edges.csv"
    months = [f"{y}-M{m:02}" for y in range(2021, 2026) for m in range(1, 13)]
    patterns = {"EARLY": [10, 12, 12] + [0] * 54 + [11, 12, 13],
                "END": [0] * 56 + [10, 11, 12, 12], "NONE": [9.99] * 60,
                "SAME": [10, 12, 12] + [0] * 57}
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["COUNTRY", "INDEX_TYPE", "COICOP_1999", "TYPE_OF_TRANSFORMATION", "FREQUENCY", "TIME_PERIOD", "OBS_VALUE"])
        for country, values in patterns.items():
            writer.writerows([country, "CPI", "_T", "YOY_PCH_PA_PT", "M", month, value]
                             for month, value in zip(months, values))
    assert namespace["streak_reference"](path) == (4, [
        ["END", 4, "2025-M09", "2025-M12", 12, "2025-M11"],
        ["EARLY", 3, "2021-M01", "2021-M03", 12, "2021-M02"],
        ["SAME", 3, "2021-M01", "2021-M03", 12, "2021-M02"],
    ])


@pytest.mark.parametrize("mutation", [None, "late_country", "gap", "gap_format", "label", "header", "bold", "format", "rate", "economy", "timing", "quote", "driver", "payload", "failed", "swap_uk_japan", "swap_uk_us", "year_only", "swap_quote"])
def test_two_source_cell(namespace, reference, tmp_path, mutation):
    truth, rows = reference[1:3]
    path = tmp_path / "combined.xlsx"
    gap = [round(r[5] - 4.1, 2) for r in rows[:10]]
    wb = report_workbook(path, truth, gap)
    outlook = wb["IMF Outlook"]
    sentence = CORE_SENTENCE
    driver = "higher energy and food prices"
    for address, value in {
        "A1": "IMF World Economic Outlook Update, July 2026", "A2": "Measure", "B2": 2025, "C2": 2026, "D2": 2027,
        "A3": "World headline inflation (%)", "B3": 4.1, "C3": 4.7, "D3": 3.9,
        "A5": "Core inflation returns to target", "A6": "Economy", "B6": "Timing",
        "A7": "United Kingdom", "B7": "by mid-2027", "A8": "Japan", "B8": "by the end of 2027",
        "A9": "United States", "B9": "by the end of 2027", "A10": "euro area", "B10": "only in 2028",
        "A12": "Source sentence", "A13": sentence, "A15": "Stated driver of the 2026 increase", "A16": driver,
    }.items():
        outlook[address] = value
    for address in ["A2", "B2", "C2", "D2", "A5", "A6", "B6", "A12", "A15"]:
        outlook[address].font = Font(bold=True)
    for address in ["B3", "C3", "D3"]:
        outlook[address].number_format = "0.00"
    payload = {"n_countries": 18, "median_avg": 87.55, "world_2025": 4.1,
               "world_2026": 4.7, "revision_2026_pp": .3, "revision_2027_pp": .2, "core_target_economies": 4}
    if mutation == "late_country":
        wb["All countries"]["B18"] = 999
    elif mutation == "gap":
        wb["Report"]["H12"] = 999
    elif mutation == "gap_format":
        wb["Report"]["H12"].number_format = "General"
    elif mutation == "bold":
        outlook["A15"].font = Font(bold=False)
    elif mutation == "format":
        outlook["D3"].number_format = "General"
    elif mutation == "payload":
        payload["revision_2027_pp"] = 0
    elif mutation in {"swap_uk_japan", "swap_uk_us"}:
        other = "B8" if mutation == "swap_uk_japan" else "B9"
        outlook["B7"], outlook[other] = outlook[other].value, outlook["B7"].value
    elif mutation == "year_only":
        for address in ["B7", "B8", "B9"]:
            outlook[address] = "2027"
    elif mutation == "swap_quote":
        outlook["A13"] = sentence.replace("United Kingdom", "Japan").replace("Japan and", "United Kingdom and")
    elif mutation and mutation != "failed":
        address, value = {"label": ("A12", "Wrong"), "header": ("D2", 2028),
                          "rate": ("D3", 9), "economy": ("A10", "Mars"),
                          "timing": ("B10", "2029"), "quote": ("A13", "return to target only gradually"),
                          "driver": ("A16", "energy and food fabrication")}[mutation]
        outlook[address] = value
    wb.save(path)
    # Keep the actual pinned sentence at the PDF boundary, including when the workbook lies.
    document = MagicMock()
    document.__enter__.return_value = [SimpleNamespace(get_text=lambda: sentence + " The increase reflects " + driver + ".")]
    namespace.update(REPORT3_PATH=path, truth=truth, rows=rows, WEO_PATH="offline.pdf",
                     fitz=SimpleNamespace(open=lambda path: document),
                     result4=SimpleNamespace(submitted=mutation != "failed", payload=payload))
    if mutation is None:
        execute("WORLD_2025, WORLD_2026", namespace)
        assert all(namespace["two_source_checks"].values())
    else:
        with pytest.raises(AssertionError):
            execute("WORLD_2025, WORLD_2026", namespace)
        if mutation in {"swap_uk_japan", "swap_uk_us", "year_only"}:
            assert namespace["two_source_checks"]["source_sentence_quoted"]
            assert not namespace["two_source_checks"]["timings_correct"]
