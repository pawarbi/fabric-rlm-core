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
    # A time-semantics lesson matters when the task asks about the current
    # or latest period; a task that names its window ("2025/Q4 through
    # 2026/Q2") does not need it, so period nouns alone do not trigger it.
    "time_semantics": frozenset(
        {"current", "currently", "latest", "now", "today", "recent", "this",
         "asof", "fiscal", "ytd", "qtd", "mtd", "todate"}
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
    "query_behavior": frozenset({"same", "equivalent", "reconcile", "match", "identical"}),
    "relationship_path": frozenset({"join", "relationship", "related", "link"}),
    "cross_source_mapping": frozenset({"join", "combine", "both", "sources", "compare"}),
}
# Kinds that need one of their trigger words in the task before vocabulary
# overlap counts at all.
_TRIGGER_REQUIRED = frozenset({"time_semantics"})
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
    "declared": "declared by the source owner",
    "source_declared": "source declared",
    "schema_name_pattern": "inferred from schema names",
    "boolean_period_flag": "boolean flag in a period table",
    "name_pattern": "name pattern only",
    "preflight_estimate": "preflight estimate",
    "repeated_timeout": "repeated timeouts",
    "single_timeout": "one timeout",
    "repeated_execution": "executed in several runs",
    "single_execution": "executed once",
    "verified_success": "verified runs",
    "single_success": "one run",
    "contradicted_by_failure": "contradicted by a failure",
    "degenerate_unfiltered": "identity observed under an unfiltered context",
    "distinct_when_filtered": "distinct once filtered",
    "repeated_runs": "repeated across runs",
    "verified_runs": "verified runs",
    "candidate_restriction_observed": "candidate restriction seen in the trajectory",
    "catalog_validation": "catalog validation",
    "observed_identity": "observed identity of values",
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
    if lesson.kind in {"valid_grain", "expensive_grain"}:
        # A grain lesson is about its dimensions; the measures it happened
        # to carry would match every task that names the same measure.
        subject_tokens = _tokens(lesson.subject) | _rule_tokens(lesson.structured_rule.get("grain"))
    else:
        subject_tokens = _tokens(lesson.subject) | _rule_tokens(lesson.structured_rule)
    triggers = _KIND_TRIGGERS.get(lesson.kind, frozenset())
    if lesson.kind in _TRIGGER_REQUIRED and not (triggers & task_tokens):
        # A lesson about "current" has nothing to say to a task that names
        # its own window, however many model names the two share.
        return 0.0
    score = float(len(subject_tokens & task_tokens))
    score += 0.5 * len(triggers & task_tokens)
    if lesson.kind == "semantic_fact" and "declared" in lesson.basis:
        # A declaration by the source owner is not a hypothesis to be
        # matched against the question: it applies to every task on the
        # source, below whatever the question names explicitly.
        return 0.5 + score + _CONFIDENCE_WEIGHT.get(lesson.confidence, 0.0)
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
    ``source_ids`` restricts the lessons to those depending on the given
    sources, which is how the registered-operation planner sees only what
    applies to the sources its operations read.
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
            f"The schema names a current-period construct ({listed}); \"current\" is most "
            "likely defined there. Use it, and do not infer the current period from MAX(Date)."
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
        runs = rule.get("runs")
        if runs is not None:
            detail.append(f"{int(runs)} run{'s' if int(runs) != 1 else ''}")
        if rule.get("max_rows_observed") is not None:
            detail.append(f"up to {int(rule['max_rows_observed']):,} rows")
        if rule.get("max_seconds_observed") is not None:
            detail.append(f"{float(rule['max_seconds_observed']):.1f} s")
        suffix = f" ({', '.join(detail)})" if detail else ""
        return (
            f"{measures} by {_grain_text(rule.get('grain'))} executed successfully in prior "
            f"runs{suffix}: an observed feasible query grain, not a validated analysis."
        )
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
    if kind == "query_behavior" and rule.get("observation") == "identical_in_observed_contexts":
        measures = " and ".join(str(m) for m in (rule.get("measures") or [lesson.subject]))
        return (
            f"{measures} returned identical values in every filtered context observed "
            f"({int(rule.get('contexts', 0))} contexts). That is an observed coincidence of values, "
            "not a verified equivalence of definitions; do not substitute one for the other."
        )
    if kind == "semantic_fact" and rule.get("fact"):
        fact = rule.get("fact")
        if fact == "grain":
            grain = " x ".join(str(c) for c in (rule.get("grain") or []))
            return f"Declared grain: one row per {grain}. Aggregate at this grain, or say why the answer's grain differs."
        if fact == "period_column":
            return (
                f"Declared period column: {rule.get('column')}. Define \"current\", \"latest\" and any "
                "period comparison by this column, not by the maximum of another field."
            )
        if fact == "units":
            return f"{rule.get('column')} is measured in {rule.get('unit')}; report it with that unit."
        if fact == "definition":
            return f"{rule.get('name')}: {rule.get('definition')}"
        if fact == "note":
            return str(rule.get("text") or lesson.subject)
    if kind in {"metric_equivalence", "metric_non_equivalence"}:
        measures = " and ".join(str(m) for m in (rule.get("measures") or [lesson.subject]))
        relation = "are equivalent by definition" if kind == "metric_equivalence" else "are not equivalent"
        return f"{measures} {relation}."
    pairs = ", ".join(f"{key} {value}" for key, value in rule.items() if isinstance(value, (str, int, float)))
    return f"{lesson.subject}: {pairs}."


def _source_tag(lesson: LearnedLesson) -> str:
    return ", ".join(lesson.source_dependencies)


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
        # Every line names the source it was learned on, so a rule about one
        # bound input is never read as a rule about another.
        sections.setdefault(title, []).append(f"- [{_source_tag(lesson)}] {_render_rule(lesson)}")
    lines = [
        "## Learned source guidance",
        "",
        "Facts learned from earlier runs on these sources, each tagged with the",
        "input it applies to. They narrow the search; they never replace checking",
        "the source when evidence conflicts.",
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
