"""The ledger: what a run established, written as it goes.

The failure this exists for: a model computes at turn six and writes its answer
at turn twenty from a transcript, and figures drift in between. Observed drifts
included a shift comparison that was never queried, a defect rate that was off
by a tenth of a point, and a sigma level of 5.57 for a DPMO that is 3.49.

It was first prototyped by describing `record()` in the prompt and asking the
model to define it. Over 26 turns it defined nothing, reassigned the bound path,
and hand-typed a figure into the report. Hence a bound object, and hence
`test_record_ignores_a_claimed_value`, which is the property that makes typing a
figure pointless.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from fabric_rlm import Ledger, SemanticModel, bare_numbers, cited_labels, format_value
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.ledger import iter_unverified


@pytest.fixture
def ledger(tmp_path):
    return Ledger(str(tmp_path / "findings.jsonl"), reset=True)


class FakeFrame:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    @property
    def iloc(self):
        outer = self

        class _I:
            def __getitem__(self, key):
                r, c = key
                return list(outer._rows[r].values())[c]

        return _I()


@pytest.fixture
def fake_sempy(monkeypatch):
    calls = []
    fabric = types.ModuleType("sempy.fabric")

    def evaluate_dax(dataset, query, **kw):
        calls.append(query)
        # one row unless the query asks for a grouping
        if "SUMMARIZECOLUMNS" in query:
            return FakeFrame([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])
        return FakeFrame([{"v": 1234.5}])

    fabric.evaluate_dax = evaluate_dax
    fabric.list_tables = lambda *a, **k: FakeFrame([{"Name": "T"}])
    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    return calls


# -- the property that matters --------------------------------------------


def test_record_ignores_a_claimed_value(ledger, fake_sempy):
    """The recorded figure is the query result, full stop.

    There is no argument for the model to pass a value, so there is nothing to
    disagree with the query. This is the difference between catching drift and
    not having any."""
    model = SemanticModel("D", validate=False, ledger=ledger)
    returned = model.record("total", 'EVALUATE ROW("v", [X])', format="currency")
    assert returned == 1234.5
    assert ledger.facts()["total"]["value"] == 1234.5
    assert ledger.facts()["total"]["source"] == 'EVALUATE ROW("v", [X])'


def test_a_worker_write_is_visible_to_the_parent(tmp_path):
    """The ledger crosses the wire as a path, so both processes append to one
    file. Without this the parent's `entries()` would be empty after a run."""
    path = str(tmp_path / "f.jsonl")
    parent = Ledger(path, reset=True)
    worker = decode_from_worker_wire(encode_for_worker({"lg": parent}))["lg"]
    worker.assert_value("from_worker", 7, "src")
    assert [e["label"] for e in parent.entries()] == ["from_worker"]


def test_model_carries_its_ledger_across_the_wire(tmp_path):
    path = str(tmp_path / "f.jsonl")
    m = SemanticModel("D", validate=False, ledger=Ledger(path, reset=True))
    back = decode_from_worker_wire(encode_for_worker({"m": m}))["m"]
    assert back.ledger is not None and back.ledger.path == path
    json.dumps(encode_for_worker({"m": m}))          # must stay JSON-safe


def test_a_model_without_a_ledger_says_so(fake_sempy):
    with pytest.raises(RuntimeError, match="no ledger"):
        SemanticModel("D", validate=False).record("x", "EVALUATE 1")


def test_record_refuses_a_grouped_query(ledger, fake_sempy):
    """Reading row zero off a grouped result is how three different quarters
    once came back with the same number."""
    model = SemanticModel("D", validate=False, ledger=ledger)
    with pytest.raises(ValueError, match="single-row"):
        model.record("bad", "EVALUATE SUMMARIZECOLUMNS(T[a], \"v\", [X])")
    assert ledger.facts() == {}, "a rejected record must not be written"


# -- reading it back -------------------------------------------------------


def test_recall_reads_back_figures_and_dead_ends(ledger):
    ledger.assert_value("arr", 18_118_056_834.46, "dax", format="currency", note="book")
    ledger.observe("New/Churn measures are blank outside a report")
    text = ledger.recall()
    assert "arr = 18118056834.46" in text
    assert "blank outside a report" in text


def test_observations_cannot_be_cited_as_numbers(ledger):
    ledger.observe("checked shift totals, nothing there")
    assert ledger.facts() == {}
    assert ledger.missing_labels("{{anything}}") == ["anything"]


def test_last_write_wins(ledger):
    ledger.assert_value("x", 1, "a")
    ledger.assert_value("x", 2, "b")
    assert ledger.facts()["x"]["value"] == 2


def test_unreadable_lines_do_not_break_reading(ledger):
    ledger.assert_value("good", 1, "a")
    with open(ledger.path, "a", encoding="utf-8") as fh:
        fh.write("not json\n")
    assert [e["label"] for e in ledger.entries()] == ["good"]


# -- putting it in a report ------------------------------------------------


def test_render_substitutes_and_formats(ledger):
    ledger.assert_value("arr", 18_118_056_834.46, "d", format="currency")
    ledger.assert_value("share", 0.1631, "d", format="percent")
    out = ledger.render("Book is {{arr}}, top ten hold {{share}}.")
    assert out == "Book is $18.1B, top ten hold 16.3%."


def test_unknown_label_is_left_visible(ledger):
    """Silently dropping it would read as prose and hide the error."""
    assert ledger.render("{{nope}}") == "{{nope}}"
    assert ledger.missing_labels("{{nope}}") == ["nope"]


@pytest.mark.parametrize("value,kind,expected", [
    (18_118_056_834.46, "currency", "$18.1B"),
    (453_910_015.22, "currency", "$453.9M"),
    (742.0, "currency", "$742"),
    (0.1631, "percent", "16.3%"),
    (23.5, "percent", "23.5%"),
    (2.5, "ratio", "2.50x"),
    (6309, "count", "6,309"),
    (0.0234, "count", "0.023"),
])
def test_formats_read_the_way_a_report_should(value, kind, expected):
    assert format_value(value, kind) == expected


def test_cited_labels_are_ordered_and_deduped():
    assert cited_labels("{{b}} {{a}} {{b}}") == ["b", "a"]


@pytest.mark.parametrize("text,flagged", [
    ("we saw 1,234,567 accounts", ["1,234,567"]),
    ("a rate of 3.14 percent", ["3.14"]),
    ("in 2026 across 8 lines", []),          # a year and a small count
    ("{{arr}} and {{share}}", []),           # cited, not typed
])
def test_bare_numbers_flags_only_what_was_typed(text, flagged):
    assert bare_numbers(text) == flagged


# -- integrity -------------------------------------------------------------


def test_iter_unverified_reports_drift_and_dead_sources(ledger):
    ledger.assert_value("ok", 100.0, "q1")
    ledger.assert_value("drifted", 100.0, "q2")
    ledger.assert_value("dead", 1.0, "q3")

    def check(entry):
        if entry["source"] == "q3":
            raise RuntimeError("query gone")
        return 100.0 if entry["source"] == "q1" else 250.0

    problems = dict(iter_unverified(ledger, check))
    assert "ok" not in problems
    assert "250" in problems["drifted"]
    assert "did not run" in problems["dead"]


def test_describe_points_at_record_when_a_ledger_is_present(tmp_path):
    lg = Ledger(str(tmp_path / "f.jsonl"), reset=True)
    with_ledger = SemanticModel("D", validate=False, ledger=lg).__rlm_describe__()
    without = SemanticModel("D", validate=False).__rlm_describe__()
    assert ".record(" in with_ledger
    assert ".record(" not in without


def test_ledger_creates_its_directory(tmp_path):
    lg = Ledger(str(tmp_path / "deep" / "nested" / "f.jsonl"))
    lg.assert_value("x", 1, "s")
    assert lg.facts()["x"]["value"] == 1


def test_reset_clears_but_default_appends(tmp_path):
    path = str(tmp_path / "f.jsonl")
    Ledger(path, reset=True).assert_value("first", 1, "s")
    assert len(Ledger(path).entries()) == 1            # default appends
    assert len(Ledger(path, reset=True).entries()) == 0


def test_bad_format_is_rejected_at_write_time(ledger):
    with pytest.raises(ValueError, match="format must be one of"):
        ledger.assert_value("x", 1, "s", format="dollars")


# -- asserted values are not citable ---------------------------------------
#
# Observed: given a ledger and 26 turns, the model ignored the query-executing
# recorder for 23 turns, then at turn 24 wrote "Record all key figures - we
# have 3 turns left" and called the ledger's own record() with numbers recalled
# from memory and sources like "calc". Binding the object was not enough,
# because the task could be completed without ever touching it. So an entry now
# carries whether its value came from running its source, and only those count.


def test_an_asserted_value_is_marked_unverified(ledger):
    ledger.assert_value("claimed", 138138217, "calc")
    assert ledger.facts()["claimed"]["verified"] is False
    assert ledger.unverified() == ["claimed"]


def test_a_query_result_is_marked_verified(ledger, fake_sempy):
    SemanticModel("D", validate=False, ledger=ledger).record("real", "EVALUATE 1")
    assert ledger.facts()["real"]["verified"] is True
    assert ledger.unverified() == []


def test_only_verified_entries_can_be_cited(ledger, fake_sempy):
    ledger.assert_value("claimed", 1, "calc")
    SemanticModel("D", validate=False, ledger=ledger).record("real", "EVALUATE 1")
    text = "{{claimed}} and {{real}}"
    assert ledger.missing_labels(text) == ["claimed"], \
        "a number the caller asserted must not satisfy a citation"


def test_every_query_is_logged_even_without_record(ledger, fake_sempy):
    """Recording has to be a side effect of querying, not an alternative to it:
    the model used dax() on eight turns and record() on none."""
    model = SemanticModel("D", validate=False, ledger=ledger)
    model.dax('EVALUATE ROW("v", [X])')
    sources = [e.get("source") for e in ledger.entries()]
    assert 'EVALUATE ROW("v", [X])' in sources
    assert ledger.facts() == {}, "an unlabelled query log is not a citable fact"


def test_query_logging_never_breaks_the_query(ledger, fake_sempy, monkeypatch):
    """A ledger that cannot be written must not take the run down with it."""
    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(Ledger, "observe", boom)
    model = SemanticModel("D", validate=False, ledger=ledger)
    assert len(model.dax('EVALUATE ROW("v", [X])')) == 1


def test_writing_a_figure_straight_into_the_ledger_is_refused(ledger):
    """Offered as `record(label, value, source)` it was used twice to write
    down numbers recalled from memory, with sources like "calc". Failing on the
    spot costs one turn; letting it through costs the run, because the value
    cannot be cited and that is only discovered at final validation."""
    with pytest.raises(AttributeError, match="from the source that produces it"):
        ledger.record("arr_2020Q4", 138138217, "calc")
    assert ledger.entries() == []


def test_the_refusal_names_the_call_to_use_instead(ledger):
    with pytest.raises(AttributeError) as err:
        ledger.record("x", 1, "calc")
    assert "model.record(" in str(err.value)
    assert "notes.observe(" in str(err.value)


def test_describe_no_longer_advertises_recording_a_figure(ledger):
    described = ledger.__rlm_describe__()
    assert "observe(" in described and "recall()" in described
    assert "record(label, value" not in described


# -- a write-up phase that never sees a number -----------------------------


def test_brief_withholds_values(ledger, fake_sempy):
    """A writer that cannot see a figure cannot type one, so citing is the only
    way to get a number onto the page."""
    m = SemanticModel("D", validate=False, ledger=ledger)
    m.record("total", 'EVALUATE ROW("v", [X])', format="currency",
             note="whole book, all years")
    brief = ledger.brief()
    assert "{{total}}" in brief
    assert "currency" in brief
    assert "whole book, all years" in brief
    assert "1234" not in brief, "the value must not reach the writer"


def test_brief_carries_observations(ledger):
    ledger.observe("New/Churn measures blank outside a report")
    assert "observed: New/Churn measures blank" in ledger.brief()


def test_brief_omits_unverified_entries(ledger):
    ledger.assert_value("claimed", 99, "calc")
    assert "{{claimed}}" not in ledger.brief()


def test_brief_is_honest_when_empty(ledger):
    assert ledger.brief() == "(nothing recorded)"


# -- lineage beyond semantic models ----------------------------------------


def test_a_file_logs_its_own_reads(tmp_path, ledger):
    """The pattern generalises: a bound source records its own access, so
    lineage does not depend on the model cooperating."""
    from fabric_rlm import File

    src = tmp_path / "memo.txt"
    src.write_text("escalation threshold 0.20", encoding="utf-8")
    File(str(src), ledger=ledger).read_text()
    logged = [e for e in ledger.entries() if not e.get("label")]
    assert logged and logged[0]["source"] == str(src)
    assert "memo.txt" in logged[0]["note"]


def test_a_file_carries_its_ledger_across_the_wire(tmp_path, ledger):
    from fabric_rlm import File

    src = tmp_path / "memo.txt"
    src.write_text("x", encoding="utf-8")
    back = decode_from_worker_wire(
        encode_for_worker({"f": File(str(src), ledger=ledger)}))["f"]
    back.read_text()
    assert any(e.get("source") == str(src) for e in ledger.entries())


def test_a_figure_read_from_a_file_is_unverified(tmp_path, ledger):
    """A number in a document cannot be re-executed. A human checks it by
    opening the file; the machine cannot, so it is not citable."""
    from fabric_rlm import File

    src = tmp_path / "memo.txt"
    src.write_text("threshold 0.20", encoding="utf-8")
    File(str(src), ledger=ledger).record("threshold", 0.20, format="percent")
    assert ledger.facts()["threshold"]["verified"] is False
    assert ledger.missing_labels("{{threshold}}") == ["threshold"]


def test_a_file_without_a_ledger_still_reads(tmp_path):
    from fabric_rlm import File

    src = tmp_path / "memo.txt"
    src.write_text("hello", encoding="utf-8")
    assert File(str(src)).read_text() == "hello"
