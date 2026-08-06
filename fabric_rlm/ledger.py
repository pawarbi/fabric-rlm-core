"""An append-only record of what a run established, written as it goes.

    from fabric_rlm import RLM, SemanticModel, Ledger

    ledger = Ledger("/lakehouse/default/Files/run/findings.jsonl")
    RLM.task(
        task=BRIEF,
        inputs={"model": SemanticModel("Sales", ledger=ledger), "notes": ledger},
        outputs=["report"],
    ).run()
    ledger.entries()          # everything the run established, with its provenance

Inside the run, `model.record("total_arr", 'EVALUATE ROW("v", [Total Sales])')`
runs the query, appends the result, and returns it. The recorded value is the
query result by construction, so the model never types a figure and there is
nothing to drift between computing at turn six and writing at turn twenty.

Why this is a bound object rather than a helper described in the prompt: it was
first prototyped by describing `record()` in the task text and asking the model
to define it. Given a 26-turn budget it defined nothing, reassigned the bound
path to a relative one, and hand-typed a figure into the report. A ledger the
model has to opt into is a ledger it will skip.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")

_FORMATS = ("currency", "percent", "count", "ratio", "raw")


@dataclass(frozen=True)
class Ledger:
    """A JSONL file of established facts, appended to during a run.

    One entry per figure: its label, its value, and the source that produced it.
    `source` is whatever makes the entry checkable later - a DAX query, a SQL
    string, a file path and line, a verbatim quote. The ledger does not care
    which; it cares that there is one.
    """

    path: str
    reset: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("Ledger needs a file path.")
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self.reset or not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8"):
                pass

    # -- writing ----------------------------------------------------------

    def _append(
        self,
        label: str,
        value: Any,
        source: str = "",
        *,
        format: str = "count",
        note: str = "",
        verified: bool = False,
    ) -> Any:
        """Write an entry. Called by a source that has just run something.

        Deliberately not part of the surface an agent sees. Offered as
        `record(label, value, source)` it was used twice to write down numbers
        recalled from memory, with sources like "calc" - turns spent down a
        path that cannot satisfy a citation, discovered only at final
        validation. A figure enters the ledger by being produced, or not at all.
        """
        label = str(label).strip()
        if not label:
            raise ValueError("a ledger entry needs a label")
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")
        entry = {
            "label": label,
            "value": value,
            "source": source,
            "format": format,
            "note": note,
            # False unless the value came from executing `source`. A caller
            # supplying both a value and a source string is asserting, not
            # demonstrating, and an agent asked to record at the end of a run
            # will happily assert numbers it half-remembers - observed doing
            # exactly that, with sources like "calc".
            "verified": bool(verified),
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return value

    def record(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse, and say what to call instead.

        An agent reaching for `notes.record(label, value, ...)` is about to
        write down a number it is holding rather than one it just produced.
        Failing here costs it one turn; letting it through costs the run,
        because an asserted value cannot be cited and it finds out at the end.
        """
        raise AttributeError(
            "A figure cannot be written straight into the ledger. Record it "
            "from the source that produces it - for a semantic model, "
            'model.record("label", "EVALUATE ...") runs the query and records '
            "what it returned. Use notes.observe(note) for a caveat or a dead "
            "end that is not a figure."
        )

    def assert_value(
        self,
        label: str,
        value: Any,
        source: str,
        *,
        format: str = "count",
        note: str = "",
    ) -> Any:
        """Record a figure this process did not produce, marked unverified.

        For the caller that legitimately has a value from outside any source
        object. Never citable: `missing_labels` ignores unverified entries.
        """
        return self._append(label, value, source, format=format, note=note,
                            verified=False)

    def observe(self, note: str, source: str = "") -> None:
        """Record something that is not a figure: a dead end, a caveat, a
        decision about what to look at next. These are readable by `recall()`
        but cannot be cited as a number."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"label": None, "note": note,
                                 "source": source}, default=str) + "\n")

    # -- reading ----------------------------------------------------------

    def entries(self) -> list[dict[str, Any]]:
        """Every line, in order, skipping any that are unreadable."""
        if not os.path.exists(self.path):
            return []
        out: list[dict[str, Any]] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def facts(self, *, verified_only: bool = False) -> dict[str, dict[str, Any]]:
        """Labelled figures, last write wins.

        `verified_only` keeps just the entries whose value came from executing
        their source. Anything a caller merely asserted is dropped, which is
        what a citation check should use.
        """
        out = {e["label"]: e for e in self.entries() if e.get("label")}
        if verified_only:
            out = {k: v for k, v in out.items() if v.get("verified")}
        return out

    def unverified(self) -> list[str]:
        """Labels whose value was asserted rather than produced by a source."""
        return sorted(lb for lb, e in self.facts().items()
                      if not e.get("verified"))

    def recall(self) -> str:
        """The record as text, for the model to read back before writing.

        This is the point of the whole thing: at write time it reads what it
        established rather than recalling it from a transcript full of schema
        dumps and tracebacks.
        """
        lines: list[str] = []
        for e in self.entries():
            if e.get("label"):
                lines.append(f"{e['label']} = {e['value']}"
                             + (f"   # {e['note']}" if e.get("note") else ""))
            elif e.get("note"):
                lines.append(f"- {e['note']}")
        return "\n".join(lines) or "(nothing recorded yet)"

    def brief(self) -> str:
        """What was found, without the numbers.

        For a write-up phase that should cite the record rather than retype it.
        Values are withheld deliberately: a writer that never sees a figure
        cannot type one, so citing becomes the only way to put a number on the
        page. Notes come through, because they carry why the figure mattered -
        without them the report is true and lifeless.
        """
        lines: list[str] = []
        for entry in self.entries():
            label = entry.get("label")
            if label:
                if not entry.get("verified"):
                    continue
                note = entry.get("note") or ""
                lines.append(f"{{{{{label}}}}}  ({entry.get('format', 'count')})"
                             + (f" - {note}" if note else ""))
            elif entry.get("note"):
                lines.append(f"observed: {entry['note']}")
        return "\n".join(lines) or "(nothing recorded)"

    # -- using ------------------------------------------------------------

    def render(self, text: str) -> str:
        """Replace every {{label}} with its recorded value, formatted.

        An unknown label is left as written rather than silently dropped, so a
        caller can tell the difference between "no such fact" and "zero"."""
        facts = self.facts()

        def sub(m: "re.Match[str]") -> str:
            entry = facts.get(m.group(1).strip())
            if entry is None:
                return m.group(0)
            return format_value(entry.get("value"), entry.get("format", "count"))

        return _PLACEHOLDER.sub(sub, text)

    def missing_labels(self, text: str) -> list[str]:
        """Labels the text cites that were never recorded."""
        facts = self.facts(verified_only=True)
        return sorted({lb.strip() for lb in _PLACEHOLDER.findall(text)
                       if lb.strip() not in facts})

    def __rlm_describe__(self) -> str:
        n = len(self.facts())
        return (f"Ledger at {self.path} ({n} fact(s) so far) - "
                "observe(note) records a dead end or a caveat; recall() reads "
                "back everything established so far. Figures are recorded from "
                "the source that produces them, not written here directly.")

    def __repr__(self) -> str:
        return f"Ledger({self.path!r})"


def format_value(value: Any, kind: str = "count") -> str:
    """How a recorded figure reads in a report.

    Taking number-writing away from the model also takes formatting away, so
    the ledger has to give it back: an executive reads $18.1B, not
    18,118,056,834.46.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "percent":
        return f"{v * 100:.1f}%" if abs(v) <= 1.5 else f"{v:.1f}%"
    if kind == "currency":
        a = abs(v)
        for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= cut:
                return f"${v / cut:,.1f}{suffix}"
        return f"${v:,.0f}"
    if kind == "ratio":
        return f"{v:,.2f}x"
    if kind == "raw":
        return str(value)
    return f"{v:,.0f}" if abs(v) >= 1 else f"{v:,.3f}"


def cited_labels(text: str) -> list[str]:
    """Every {{label}} a piece of text cites, in first-seen order."""
    seen: list[str] = []
    for label in _PLACEHOLDER.findall(text):
        label = label.strip()
        if label and label not in seen:
            seen.append(label)
    return seen


def bare_numbers(text: str, *, floor: float = 999, decimals: int = 2) -> list[str]:
    """Figures typed directly into text instead of cited from the ledger.

    Placeholders are stripped first, so only what the model wrote itself is
    flagged. Years pass, and so do small integers, which are thresholds and
    list counts rather than findings.
    """
    stripped = _PLACEHOLDER.sub(" ", text)
    out: list[str] = []
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", stripped):
        raw = m.group(0).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if 1900 <= v <= 2100 and dec == 0:
            continue
        if abs(v) > floor or dec >= decimals:
            out.append(m.group(0))
    return out


def iter_unverified(ledger: Ledger, check: Any) -> Iterable[tuple[str, str]]:
    """Yield (label, problem) for entries whose source no longer reproduces.

    `check` takes an entry and returns the value its source produces now, or
    raises. Kept generic so the same integrity pass works for DAX, SQL, or a
    substring check against a document.
    """
    for label, entry in ledger.facts().items():
        try:
            got = check(entry)
        except Exception as exc:  # noqa: BLE001 - reported, not handled
            yield label, f"its source did not run: {type(exc).__name__}: {exc}"
            continue
        try:
            claimed, actual = float(entry["value"]), float(got)
        except (TypeError, ValueError):
            if str(entry["value"]) != str(got):
                yield label, f"recorded {entry['value']!r}, source gives {got!r}"
            continue
        if abs(claimed - actual) > max(abs(actual) * 0.005, 1e-9):
            yield label, f"recorded {claimed:,.4f}, source gives {actual:,.4f}"
