"""Task-scoped retrieval of learned lessons and their prompt rendering.

A package may hold hundreds of lessons; a task gets the few that bear on
it. Retrieval ranks active lessons by how their subject, rule vocabulary
and kind relate to the task text, then renders the structured rules into
short natural-language guidance. The rendering states what was observed
and how confident the package is; it never carries data values, and it
closes with the reminder that the source remains the authority when
evidence conflicts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import re

from fabric_rlm.knowledge import KnowledgePackage, LearnedLesson


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "which", "into",
        "over", "each", "per", "are", "was", "were", "has", "have", "how",
        "what", "when", "where", "who", "why", "all", "any", "its", "their",
        "then", "than", "also", "only", "not", "use", "used", "using", "they",
        "them", "our", "you", "your", "about", "between", "among", "across",
        "report", "show", "give", "list", "find", "identify", "say", "whether",
    }
)
_KIND_TRIGGERS: dict[str, frozenset[str]] = {
    "time_semantics": frozenset(
        {"current", "currently", "latest", "now", "today", "recent", "this",
         "quarter", "month", "year", "period", "week", "date", "asof", "fiscal"}
    ),
    "context_requirement": frozenset(
        {"growth", "previous", "prior", "change", "changed", "trend", "trends",
         "retention", "nrr", "grr", "churn", "compare", "compared", "versus",
         "deteriorated", "deterioration", "declined", "decline", "improved",
         "improvement", "yoy", "qoq", "delta", "variance", "movement", "increase",
         "decrease", "last", "over"}
    ),
    "expensive_grain": frozenset(
        {"by", "breakdown", "segment", "segments", "group", "groups", "each",
         "per", "across", "every", "detail", "detailed", "customer", "customers"}
    ),
    "valid_grain": frozenset(
        {"by", "breakdown", "segment", "segments", "group", "groups", "each",
         "per", "across", "every", "region", "product"}
    ),
    "preferred_strategy": frozenset(
        {"segment", "segments", "drill", "drilldown", "candidate", "candidates",
         "deteriorated", "deterioration", "rank", "ranked", "top", "worst",
         "largest", "biggest", "impact", "narrow"}
    ),
    "invalid_path": frozenset(),
    "metric_equivalence": frozenset({"same", "equivalent", "reconcile", "match"}),
    "metric_non_equivalence": frozenset({"same", "equivalent", "reconcile", "match"}),
    "semantic_fact": frozenset(),
    "query_behavior": frozenset(),
    "relationship_path": frozenset({"join", "relationship", "related", "link"}),
    "cross_source_mapping": frozenset({"join", "combine", "both", "sources", "compare"}),
}
_KIND_ORDER = (
    "time_semantics",
    "context_requirement",
    "preferred_strategy",
    "valid_grain",
    "expensive_grain",
    "invalid_path",
    "metric_equivalence",
    "metric_non_equivalence",
    "relationship_path",
    "cross_source_mapping",
    "semantic_fact",
    "query_behavior",
)
_CONFIDENCE_WEIGHT = {"high": 0.3, "medium": 0.2, "low": 0.1}
_SECTION_TITLES = {
    "time_semantics": "Time semantics",
    "context_requirement": "Measure behavior",
    "preferred_strategy": "Validated strategies",
    "valid_grain": "Validated strategies",
    "expensive_grain": "Observed query risk",
    "invalid_path": "Known invalid references",
    "metric_equivalence": "Metric relationships",
    "metric_non_equivalence": "Metric relationships",
    "relationship_path": "Relationships",
    "cross_source_mapping": "Cross-source mappings",
    "semantic_fact": "Source facts",
    "query_behavior": "Query behavior",
}
_BASIS_TEXT = {
    "source_declared": "source declared",
    "name_pattern": "name pattern only",
    "preflight_estimate": "preflight estimate",
    "repeated_timeout": "repeated timeouts",
    "single_timeout": "one timeout",
    "verified_success": "verified runs",
    "single_success": "one run",
    "contradicted_by_failure": "contradicted by a failure",
    "degenerate_unfiltered": "identity observed under an unfiltered context",
    "distinct_when_filtered": "distinct once filtered",
    "repeated_runs": "repeated across runs",
    "verified_runs": "verified runs",
    "catalog_validation": "catalog validation",
    "reproducible_identity": "reproducible identity",
}


def _tokens(text: object) -> set[str]:
    words = _WORD.findall(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or "")))
    return {word.lower() for word in words if len(word) > 1 and word.lower() not in _STOPWORDS}


def _rule_tokens(value: object) -> set[str]:
    if isinstance(value, str):
        return _tokens(value)
    if isinstance(value, Mapping):
        return set().union(*(_rule_tokens(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_rule_tokens(item) for item in value)) if value else set()
    return set()


def lesson_score(lesson: LearnedLesson, task_tokens: set[str]) -> float:
    """How much a lesson bears on a task, from vocabulary overlap.

    Subject and rule names count fully, the kind's trigger words count
    half (a question about "current quarter" wants the time-semantics
    rule even if it never names the construct), and confidence breaks
    ties so a verified lesson outranks a nominated one.
    """
    subject_tokens = _tokens(lesson.subject) | _rule_tokens(lesson.structured_rule)
    triggers = _KIND_TRIGGERS.get(lesson.kind, frozenset())
    score = float(len(subject_tokens & task_tokens))
    score += 0.5 * len(triggers & task_tokens)
    if score <= 0:
        return 0.0
    return score + _CONFIDENCE_WEIGHT.get(lesson.confidence, 0.0)


def retrieve_lessons(
    package: KnowledgePackage,
    task_text: str | None,
    *,
    limit: int = 8,
    statuses: Iterable[str] = ("active",),
    source_ids: Iterable[str] | None = None,
) -> tuple[LearnedLesson, ...]:
    """The lessons worth putting in front of a task, best first.

    Only lessons in ``statuses`` (active by default: candidates are never
    shown to an agent) whose vocabulary meets the task's are returned, at
    most ``limit`` of them, ordered by score, then by kind so time
    semantics come before query-cost notes at equal relevance.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    allowed = set(statuses)
    wanted_sources = set(source_ids) if source_ids is not None else None
    task_tokens = _tokens(task_text)
    scored: list[tuple[float, int, str, LearnedLesson]] = []
    for lesson in package.lessons:
        if lesson.status not in allowed:
            continue
        if wanted_sources is not None and not set(lesson.source_dependencies) & wanted_sources:
            continue
        score = lesson_score(lesson, task_tokens)
        if score <= 0:
            continue
        kind_rank = _KIND_ORDER.index(lesson.kind) if lesson.kind in _KIND_ORDER else len(_KIND_ORDER)
        scored.append((-score, kind_rank, lesson.lesson_id, lesson))
    scored.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in scored[:limit])


def _grain_text(grain: object) -> str:
    if not isinstance(grain, (list, tuple)) or not grain:
        return "the total"
    return " x ".join(str(item) for item in grain)


def _render_rule(lesson: LearnedLesson) -> str:
    rule = lesson.structured_rule
    kind = lesson.kind
    if kind == "time_semantics":
        constructs = rule.get("current_period_constructs") or ()
        listed = ", ".join(str(c) for c in list(constructs)[:4])
        return (
            f"\"Current\" is defined by an explicit model construct ({listed}). "
            "Do not infer the current period from MAX(Date)."
        )
    if kind == "context_requirement":
        measure = rule.get("measure", lesson.subject)
        base = rule.get("base_measure")
        observed = rule.get("observed")
        if observed == "identity_under_unfiltered_context" and base:
            detail = f"unfiltered it returned exactly {base}, a degenerate comparison"
        elif observed == "constant_under_unfiltered_context":
            detail = f"unfiltered it returned a constant {rule.get('constant', 'value')}"
        else:
            detail = "its name marks it as time-relative"
        return (
            f"{measure} requires an explicit period context; {detail}. "
            "Establish the period (or use the base measure with explicit periods) before trusting it."
        )
    if kind == "expensive_grain":
        parts = [f"{_grain_text(rule.get('grain'))} exceeded the query budget"]
        estimate, limit = rule.get("estimated_groups"), rule.get("max_groups")
        if estimate is not None and limit is not None:
            parts.append(f"(estimated {int(estimate):,} groups, limit {int(limit):,})")
        elif rule.get("outcome") == "preflight_timeout":
            parts.append("(the cardinality estimate did not finish in time)")
        return " ".join(parts) + ". Narrow the grain or filter before querying it."
    if kind == "valid_grain":
        measures = ", ".join(str(m) for m in list(rule.get("measures") or [])[:3]) or "measures"
        detail = []
        if rule.get("max_rows_observed") is not None:
            detail.append(f"{int(rule['max_rows_observed']):,} rows")
        if rule.get("max_seconds_observed") is not None:
            detail.append(f"{float(rule['max_seconds_observed']):.1f} s")
        suffix = f" ({', '.join(detail)})" if detail else ""
        return f"{measures} by {_grain_text(rule.get('grain'))} is a reliable analysis grain{suffix}."
    if kind == "preferred_strategy":
        coarse = _grain_text(rule.get("coarse_grain"))
        fine = _grain_text(rule.get("drilldown_grain"))
        return (
            f"Analyze at {coarse} first, then restrict to the candidate tuples before "
            f"drilling into {fine}; keep the candidates as tuples, not independent lists."
        )
    if kind == "invalid_path":
        return (
            f"No {rule.get('reference_kind', 'reference')} named {rule.get('reference', lesson.subject)} "
            "exists in this model; check the catalog before using a similar name."
        )
    if kind in {"metric_equivalence", "metric_non_equivalence"}:
        measures = " and ".join(str(m) for m in (rule.get("measures") or [lesson.subject]))
        relation = "were identical" if kind == "metric_equivalence" else "differed"
        return (
            f"{measures} {relation} across every filtered context observed "
            f"({int(rule.get('contexts', 0))} contexts); this is a reproduced identity, not a definition."
        )
    pairs = ", ".join(f"{key} {value}" for key, value in rule.items() if isinstance(value, (str, int, float)))
    return f"{lesson.subject}: {pairs}."


def _confidence_note(lesson: LearnedLesson) -> str:
    basis = ", ".join(_BASIS_TEXT.get(item, item.replace("_", " ")) for item in lesson.basis)
    note = f"{lesson.subject}: {lesson.confidence}"
    return f"{note} ({basis})" if basis else note


def render_learned_guidance(lessons: Sequence[LearnedLesson]) -> str:
    """The prompt section for a set of retrieved lessons; empty for none."""
    if not lessons:
        return ""
    sections: dict[str, list[str]] = {}
    for lesson in lessons:
        title = _SECTION_TITLES.get(lesson.kind, "Source facts")
        sections.setdefault(title, []).append(f"- {_render_rule(lesson)}")
    lines = [
        "## Learned source guidance",
        "",
        "Facts learned from earlier verified runs on these sources. They narrow the",
        "search; they never replace checking the source when evidence conflicts.",
    ]
    for title in dict.fromkeys(_SECTION_TITLES[kind] for kind in _KIND_ORDER):
        if title in sections:
            lines.append("")
            lines.append(title)
            lines.extend(sections[title])
    lines.append("")
    lines.append("Confidence: " + "; ".join(_confidence_note(lesson) for lesson in lessons) + ".")
    return "\n".join(lines)


__all__ = ["lesson_score", "render_learned_guidance", "retrieve_lessons"]
