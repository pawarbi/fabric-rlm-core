"""Experimental transfer of a verified analytical procedure across sources.

The procedure is portable; source names and fields are not. Binding is
deliberately conservative: when the target question or schema is ambiguous,
the caller receives no plan and can continue with the ordinary RLM path.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from fabric_rlm.knowledge import LearnedLesson, SourceProfile


_WORD = re.compile(r"[A-Za-z0-9]+")
_RANKING = re.compile(r"\b(most|highest|largest|top|greatest)\b", re.I)
_WITHIN = re.compile(r"\b(within|inside)\b", re.I)


def _words(text: str) -> set[str]:
    parts = _WORD.findall(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text))
    return {part.lower() for part in parts if len(part) > 1}


def _matches(column: str, clause: str) -> bool:
    words = _words(column)
    task_words = _words(clause)
    return bool(words & task_words)


@dataclass(frozen=True)
class TransferableProcedure:
    origin_source: str
    origin_lesson: str
    kind: str
    evidence_count: int


@dataclass(frozen=True)
class BoundProcedure:
    procedure: TransferableProcedure
    target_source: str
    coarse: str
    detail: str
    measure: str

    def guidance(self) -> str:
        return (
            f"Verified transferable procedure: sum {self.measure} by {self.coarse}, "
            f"select the leading {self.coarse}, then filter to that exact value "
            f"before summing {self.measure} by {self.detail}. Recompute both "
            "reported totals from the bound source."
        )


def extract_procedure(lesson: LearnedLesson) -> TransferableProcedure | None:
    """Retain an abstract operation only after independently verified runs."""
    rule = lesson.structured_rule
    coarse = rule.get("coarse_grain") or ()
    fine = rule.get("drilldown_grain") or ()
    if not (
        lesson.kind == "preferred_strategy"
        and lesson.status == "active"
        and rule.get("strategy") == "coarse_to_candidate_drilldown"
        and set(coarse) < set(fine)
        and int(rule.get("verified_runs", 0)) >= 2
        and len(lesson.evidence_ids) >= 2
        and {"verified_runs", "candidate_restriction_observed"} <= set(lesson.basis)
        and len(lesson.source_dependencies) == 1
    ):
        return None
    return TransferableProcedure(
        origin_source=lesson.source_dependencies[0],
        origin_lesson=lesson.lesson_id,
        kind="coarse_to_candidate_drilldown",
        evidence_count=len(lesson.evidence_ids),
    )


def bind_csv(
    procedure: TransferableProcedure,
    profile: SourceProfile,
    question: str,
) -> BoundProcedure | None:
    """Bind by target schema and question; refuse missing or ambiguous roles."""
    if (
        profile.family != "csv"
        or profile.source_id == procedure.origin_source
        or profile.diagnostics.get("snapshot_exact") is not True
        or not _RANKING.search(question)
    ):
        return None
    match = _WITHIN.search(question)
    if match is None:
        return None
    before, after = question[:match.start()], question[match.end():]
    schema = profile.schema
    categories = [name for name, meta in schema.items() if meta.get("type") == "string"]
    numbers = [name for name, meta in schema.items() if meta.get("type") in {"integer", "number", "float"}]
    coarse = [name for name in categories if _matches(name, before) and _matches(name, after)]
    detail = [name for name in categories if _matches(name, after) and name not in coarse]
    question_words = _words(question)
    scored = [(len(_words(name) & question_words), name) for name in numbers]
    best_score = max((score for score, _ in scored), default=0)
    measure = [name for score, name in scored if score == best_score and score > 0]
    if len(numbers) == 1 and not measure:
        measure = numbers
    if len(coarse) != 1 or len(detail) != 1 or len(measure) != 1:
        return None
    return BoundProcedure(procedure, profile.source_id, coarse[0], detail[0], measure[0])
