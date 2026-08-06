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
    worker.record("from_worker", 7, "src")
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
    ledger.record("arr", 18_118_056_834.46, "dax", format="currency", note="book")
    ledger.observe("New/Churn measures are blank outside a report")
    text = ledger.recall()
    assert "arr = 18118056834.46" in text
    assert "blank outside a report" in text


def test_observations_cannot_be_cited_as_numbers(ledger):
    ledger.observe("checked shift totals, nothing there")
    assert ledger.facts() == {}
    assert ledger.missing_labels("{{anything}}") == ["anything"]


def test_last_write_wins(ledger):
    ledger.record("x", 1, "a")
    ledger.record("x", 2, "b")
    assert ledger.facts()["x"]["value"] == 2


def test_unreadable_lines_do_not_break_reading(ledger):
    ledger.record("good", 1, "a")
    with open(ledger.path, "a", encoding="utf-8") as fh:
        fh.write("not json\n")
    assert [e["label"] for e in ledger.entries()] == ["good"]


# -- putting it in a report ------------------------------------------------


def test_render_substitutes_and_formats(ledger):
    ledger.record("arr", 18_118_056_834.46, "d", format="currency")
    ledger.record("share", 0.1631, "d", format="percent")
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
    ledger.record("ok", 100.0, "q1")
    ledger.record("drifted", 100.0, "q2")
    ledger.record("dead", 1.0, "q3")

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
    lg.record("x", 1, "s")
    assert lg.facts()["x"]["value"] == 1


def test_reset_clears_but_default_appends(tmp_path):
    path = str(tmp_path / "f.jsonl")
    Ledger(path, reset=True).record("first", 1, "s")
    assert len(Ledger(path).entries()) == 1            # default appends
    assert len(Ledger(path, reset=True).entries()) == 0


def test_bad_format_is_rejected_at_write_time(ledger):
    with pytest.raises(ValueError, match="format must be one of"):
        ledger.record("x", 1, "s", format="dollars")
