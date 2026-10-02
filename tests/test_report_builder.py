"""RLM.report: plan checks, the compute engine, narrative linkage and rendering.

The model runs are stubbed; what is tested is everything the host does around them:
a plan that names something the model lacks or a partial period goes back, every figure
comes from the engine's own queries, and a number the narrative cannot trace goes back.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from fabric_rlm import report_builder as rb
from fabric_rlm.report_builder import (
    DaxBackend,
    ModelNames,
    ReportPlan,
    check_narrative,
    check_plan,
    compute_findings,
    untraceable_numbers,
)

GUID = "e8715c25-0515-4a9a-b74c-7f69737d9441"


class FakeModel:
    def __init__(self, rows=None, total=(100.0, 150.0), dataset="Sales Model"):
        self.dataset = dataset
        self.workspace = None
        self.rows = rows if rows is not None else [("north", 60.0, 100.0), ("south", 40.0, 30.0), (None, 0.0, 20.0)]
        self.total = total
        self.queries = []

    def measures(self):
        return pd.DataFrame([
            {"Measure Name": "Total Sales", "Measure Expression": "SUM(Sales[Amount])"},
            {"Measure Name": "On Time Rate", "Measure Expression": "DIVIDE([On Time], [Orders])"},
            {"Measure Name": "This Year Sales", "Measure Expression": "[Total Sales]"},
            {"Measure Name": "Last Year Sales", "Measure Expression": "CALCULATE([Total Sales], SAMEPERIODLASTYEAR('Date'[Date]))"},
        ])

    def columns(self):
        return pd.DataFrame([
            {"Table Name": "Store", "Column Name": "Region", "Data Type": "String"},
            {"Table Name": "Date", "Column Name": "Date", "Data Type": "DateTime"},
        ])

    def dax(self, query):
        self.queries.append(query)
        if "SUMMARIZECOLUMNS" in query:
            return pd.DataFrame(self.rows, columns=["Store[Region]", "[b]", "[a]"])
        if query.startswith('EVALUATE ROW("b"'):
            return pd.DataFrame([self.total], columns=["[b]", "[a]"])
        if query.startswith('EVALUATE ROW("p0"'):
            n = query.count('"p')
            return pd.DataFrame([[10.0 + i for i in range(n)]], columns=[f"[p{i}]" for i in range(n)])
        raise AssertionError(query)


class Coverage:
    def __init__(self, partial=()):
        self.partial = set(partial)
        self.latest_complete = "2026-Q2"
        self.date_column = "'Date'[Date]"

    def status(self, key):
        return "partial" if key in self.partial else "complete"


def coverage_for(partial=()):
    cov = Coverage(partial)
    return lambda measure, grain: cov


NAMES = ModelNames.read(FakeModel())
CHANGE = {"kind": "change", "measures": ["[Total Sales]"], "group_by": ["Store[Region]"],
          "periods": ["2025-Q2", "2026-Q2"], "compare_measures": [], "filters": {},
          "reading": "Which regions moved most between the second quarters of the two years."}


def test_plan_names_resolve_to_the_models_spelling():
    plan = check_plan({**CHANGE, "measures": ["total sales"], "group_by": ["'store'[region]"]}, NAMES, coverage_for())
    assert plan.measures == ["Total Sales"] and plan.group_by == [("Store", "Region")] and plan.grain == "quarter"


def test_plan_with_an_unknown_measure_goes_back_with_candidates():
    with pytest.raises(AssertionError, match="not a measure.*Total Sales"):
        check_plan({**CHANGE, "measures": ["Sales Total Amount"]}, NAMES, coverage_for())


def test_plan_with_a_partial_period_goes_back_with_the_latest_complete_one():
    with pytest.raises(AssertionError, match="2026-Q3 is partial.*2026-Q2"):
        check_plan({**CHANGE, "periods": ["2025-Q3", "2026-Q3"]}, NAMES, coverage_for({"2026-Q3"}))


def test_plan_with_mixed_period_kinds_goes_back():
    with pytest.raises(AssertionError, match="same kind"):
        check_plan({**CHANGE, "periods": ["2025-Q2", "2026-06"]}, NAMES, coverage_for())


def test_plan_comparison_held_in_two_measures_needs_no_periods():
    plan = check_plan({**CHANGE, "measures": [], "periods": [], "compare_measures": ["Last Year Sales", "This Year Sales"]},
                      NAMES, coverage_for())
    assert plan.compare_measures == ["Last Year Sales", "This Year Sales"] and plan.measures == ["This Year Sales"]


def test_trend_plan_must_list_periods_in_order():
    trend = {**CHANGE, "kind": "trend", "group_by": [], "periods": ["2026-03", "2026-01", "2026-02"]}
    with pytest.raises(AssertionError, match="in order"):
        check_plan(trend, NAMES, coverage_for())


def plan(**over):
    base = dict(kind="change", measures=["Total Sales"], group_by=[("Store", "Region")], periods=["2025-Q2", "2026-Q2"],
                compare_measures=[], filters=[], reading="r", grain="quarter")
    base.update(over)
    return ReportPlan(**base)


def test_dax_backend_writes_period_filters_and_findings_add_up():
    model = FakeModel()
    backend = DaxBackend(model, "'Date'[Date]")
    f = compute_findings(plan(), backend)
    assert "DATE(2025,4,1)" in model.queries[0] and "DATE(2026,7,1)" in model.queries[1]
    g = f["groupings"]["'Store'[Region]"]
    assert g["additive"] and g["grew"] == 2 and g["fell"] == 1
    assert g["groups"][0]["group"] == "north" and g["groups"][0]["change"] == 40.0
    assert g["groups"][1]["group"] == "(blank)"
    south = next(x for x in g["groups"] if x["group"] == "south")
    assert south["share_before"] == pytest.approx(40.0) and south["share_after"] == pytest.approx(20.0)
    assert f["total"]["change_pct"] == pytest.approx(50.0)


def test_a_rate_gets_no_shares():
    model = FakeModel(rows=[("a", 0.26, 0.02), ("b", 0.9, 0.95)], total=(0.7, 0.6))
    f = compute_findings(plan(measures=["On Time Rate"]), DaxBackend(model, "'Date'[Date]"))
    g = f["groupings"]["'Store'[Region]"]
    assert not g["additive"] and g["ratio"] and "share_before" not in g["groups"][0]


def test_measure_pairs_need_no_date_filter():
    model = FakeModel()
    compute_findings(plan(periods=[], compare_measures=["Last Year Sales", "This Year Sales"], measures=["This Year Sales"]),
                     DaxBackend(model, ""))
    assert "[Last Year Sales]" in model.queries[0] and "DATE(" not in model.queries[0]


def test_trend_series_comes_from_one_query():
    model = FakeModel()
    f = compute_findings(plan(kind="trend", group_by=[], periods=["2026-01", "2026-02", "2026-03"], grain="month"),
                         DaxBackend(model, "'Date'[Date]"))
    s = f["series"]["Total Sales"]
    assert [p["value"] for p in s["points"]] == [10.0, 11.0, 12.0] and s["max_period"] == "2026-03"
    assert len(model.queries) == 1


@pytest.mark.parametrize("text,ok", [
    ("Sales rose 40 to 100.", True),
    ("Sales reached 1.5M in 2026-Q2.", True),
    ("Growth of 50% and 50.0 percent.", True),
    ("A change of 1,830,581 against 1,830,580.74 computed.", True),
    ("Top 5 groups over 12 months in 2026.", True),
    ("An invented 377 appears.", False),
    ("Shares moved 26% to 2% for the vendor.", True),
])
def test_numbers_in_text_trace_to_figures(text, ok):
    figs = rb.figures({"a": 40.0, "b": 100.0, "c": 1_500_000.0, "d": 50.0, "e": 1_830_580.74, "rate": [0.26, 0.02]})
    assert (untraceable_numbers(text, figs) == []) is ok


def findings():
    return compute_findings(plan(), DaxBackend(FakeModel(), "'Date'[Date]"))


def layout(**over):
    base = {
        "title": "Sales by region, 2025-Q2 to 2026-Q2",
        "summary": "Sales rose from 100 to 150, up 50%. North added 40 and south lost 10. "
                   "Two regions grew and one fell, so the gain was concentrated.",
        "sections": [
            {"title": "Headline", "block": "kpis", "grouping": "", "text": "Total sales reached 150 against 100 a year earlier, a rise of 50 or 50%."},
            {"title": "Gains and losses", "block": "gains_losses", "grouping": "'Store'[Region]",
             "text": "North added 40 and the unassigned group added 20, while south fell by 10 over the same quarter."},
            {"title": "Mix", "block": "mix", "grouping": "'Store'[Region]",
             "text": "South fell from 40% of sales to 20% as north grew faster than every other region did."},
        ],
        "insights": ["North added 40.", "South fell 10."],
    }
    base.update(over)
    return base


def test_a_good_narrative_passes():
    check_narrative(layout(), findings(), None)


def test_an_untraceable_number_goes_back_listed():
    bad = layout(summary=layout()["summary"] + " Margins improved by 377 basis points this quarter.")
    with pytest.raises(AssertionError, match="377"):
        check_narrative(bad, findings(), None)


def test_requested_sections_must_be_present():
    with pytest.raises(AssertionError, match="Returns by category"):
        check_narrative(layout(), findings(), ["Gains and losses", "Returns by category"])
    check_narrative(layout(), findings(), ["Gains and losses by region", "Headline"])


def test_mix_on_a_rate_goes_back():
    model = FakeModel(rows=[("a", 0.26, 0.02), ("b", 0.9, 0.95)], total=(0.7, 0.6))
    f = compute_findings(plan(measures=["On Time Rate"]), DaxBackend(model, "'Date'[Date]"))
    lay = layout(summary="The rate fell from 70% to 60%. Vendor a fell from 26% to 2%. Vendor b rose from 90% to 95% over the quarter.",
                 sections=[layout()["sections"][2] | {"text": "Vendor a fell from 26% to 2% while vendor b rose from 90% to 95% in the quarter."}] * 3,
                 insights=["a fell to 2%."])
    with pytest.raises(AssertionError, match="mix needs an additive"):
        check_narrative(lay, f, None)


def built(**over):
    rep = rb.BuiltReport("q", "Sales Model", plan(), findings(), layout(), NAMES, "'Date'[Date]", ["q1", "q2"])
    for k, v in over.items():
        setattr(rep, k, v)
    return rep


def test_html_has_a_chart_per_visual_section_and_the_method():
    html = built().to_html()
    assert html.count("<svg") == 2 and "How this was checked" in html and "No value" in html
    assert "<h2>Headline</h2>" in html and "SUM(Sales[Amount])" in html
    assert GUID not in html


def test_bar_labels_leave_room_for_the_values():
    svg = rb.svg_bars([("a very long category name here", 1_000_000.0), ("short", -400_000.0)], "t")
    xs = [float(x) for x in re.findall(r'<rect x="([\d.]+)"', svg)]
    label_end = float(re.search(r'<text x="([\d.]+)" y="\d+" text-anchor="end">A very', svg).group(1))
    assert min(xs) > label_end, "bars start after the label column"
    value_x = [float(x) for x in re.findall(r'<text x="([\d.]+)" y="\d+">1.000M', svg)]
    assert value_x and value_x[0] + rb._text_width("1.000M") <= 760


def test_markdown_has_the_same_sections_and_linked_charts(tmp_path):
    path = built().save(tmp_path / "r.md")
    text = path.read_text(encoding="utf-8")
    for title in ("## Headline", "## Gains and losses", "## Mix", "## How this was checked"):
        assert title in text
    assert "![Gains and losses](r_charts/chart_3.svg)" in text
    svg = (tmp_path / "r_charts" / "chart_3.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg xmlns=") and "var(--" not in svg
    assert "| North |" in text


def test_display_name_never_shows_a_guid(monkeypatch):
    assert rb.display_name(SimpleNamespace(dataset="Retail Analysis", workspace=None)) == "Retail Analysis"
    assert rb.display_name(SimpleNamespace(dataset=GUID, workspace=None)) != GUID


def test_build_report_plans_computes_and_narrates(monkeypatch):
    calls = []

    class StubRun:
        def __init__(self, task, inputs, outputs, output_validator, **kw):
            self.task, self.inputs, self.validator = task, inputs, output_validator
            calls.append(self)

        def run(self):
            if "findings" in self.inputs:
                payloads = [layout(summary="Sales grew by 999 percent overall this quarter across every one of the regions."), layout()]
            else:
                payloads = [{**CHANGE, "measures": ["Revenue"]}, CHANGE]
            self.rejections = []
            for p in payloads:
                try:
                    self.validator(p)
                except AssertionError as exc:
                    self.rejections.append(str(exc))
                    continue
                return SimpleNamespace(submitted=True, payload=p, failure_reason=None)
            return SimpleNamespace(submitted=False, payload=None, failure_reason="rejected")

    from fabric_rlm import RLM

    monkeypatch.setattr(RLM, "task", classmethod(lambda cls, task, inputs=None, outputs=None, **kw: StubRun(task, inputs, outputs, **kw)))
    monkeypatch.setattr("fabric_rlm.semantic_checks.period_coverage", lambda model, measure, grain, find_as_of: Coverage())
    model = FakeModel(dataset=GUID)
    rep = RLM.report("Which regions moved most?", inputs={"sm": model}, lm=object(), name="Sales Model",
                     sections=["Gains and losses"])
    assert rep.verified, rep.failure
    assert "not a measure" in calls[0].rejections[0] and "999" in calls[1].rejections[0]
    assert "Gains and losses" in calls[1].task and rep.queries and rep.findings["total"]["after"] == 150.0
    assert GUID not in rep.to_html()


def test_named_summary_insights_and_method_sections_are_not_repeated():
    secs = layout()["sections"] + [
        {"title": "Insights", "block": "text", "grouping": "", "text": "These findings rest only on the figures computed for both quarters above."},
        {"title": "How this was checked", "block": "text", "grouping": "", "text": "Each figure came from the model with the queries listed below this line."},
    ]
    secs[0] = secs[0] | {"title": "Summary"}
    html = built(layout=layout(sections=secs)).to_html()
    for title in ("Summary", "Insights", "How this was checked"):
        assert html.count(f"<h2>{title}</h2>") == 1
    assert html.count(layout()["summary"]) == 1 and "North added 40.</li>" in html


@pytest.mark.parametrize("change,why", [
    ({"summary": layout()["summary"] + " Revenue was $150 in total."}, "currency"),
    ({"summary": layout()["summary"] + " The <NA> group added 20."}, "no value"),
    ({"summary": layout()["summary"] + " Sales rose 50.000001 percent."}, "Round"),
    ({"sections": layout()["sections"] + [layout()["sections"][1]]}, "repeat"),
    ({"sections": [layout()["sections"][0], layout()["sections"][1],
                   {"title": "Words", "block": "text", "grouping": "", "text": "North added 40 and south lost 10 across the two quarters in question, which is the whole story here."}]}, "two sections with a chart"),
])
def test_narrative_style_rules_go_back(change, why):
    with pytest.raises(AssertionError, match=why):
        check_narrative(layout(**change), findings(), None)


def test_narrative_view_rounds_and_uses_display_labels():
    view = rb.narrative_view(findings())
    g = view["groupings"]["'Store'[Region]"]
    assert {x["group"] for x in g["groups"]} == {"North", "South", "No value"}
    assert view["total"]["change_pct"] == 50.0 and "groups_shown" in g


def test_period_changes_chart_draws_each_step():
    f = compute_findings(plan(kind="trend", group_by=[], periods=["2026-01", "2026-02", "2026-03"], grain="month"),
                         DaxBackend(FakeModel(), "'Date'[Date]"))
    rep = rb.BuiltReport("q", "Sales Model", plan(kind="trend"), f, {}, NAMES)
    svg = rep._block_html({"block": "period_changes"})
    assert svg.count("<rect") == 2 and "2026-02" in svg
