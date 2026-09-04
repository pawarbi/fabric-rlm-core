"""Source-agnostic analytical integrity checks.

Source adapters decide how evidence is retrieved: ``File`` reads a path,
``LakehouseSource`` runs a bounded query, ``SemanticModel`` evaluates a
measure. Nothing in this module knows which of them produced a number. It
answers a different question: given the evidence, is the reasoning faithful
to what was asked?

Three failures from one live trajectory motivate the helpers here. The run
compared floats with ``<`` and reported a segment as deteriorating over a
difference of 1e-7. It was asked to rank by business impact and sorted by
current ARR. It selected Product x Region x Customer Group candidates, then
filtered a later query by three independent ``.unique()`` lists, admitting
combinations it never chose. None of those depended on the source; the same
code over a CSV would have failed the same way.

The helpers are deliberately thin and free of business thresholds. The
analysis supplies the materiality rule; this module only applies it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_NOISE_RELATIVE_TOLERANCE",
    "AnalyticalIntegrityError",
    "DirectionalClaim",
    "IntegrityReport",
    "RankingRequest",
    "change_direction",
    "check_answer_hygiene",
    "check_directional_claims",
    "check_ranking_disclosure",
    "check_zero_change_items",
    "task_asks_about_change",
    "classify_claim_level",
    "concept_head",
    "declared_ranking_phrases",
    "infer_requested_ranking",
    "is_material_change",
    "parse_directional_claims",
    "restrict_to_candidate_tuples",
    "validate_analysis_integrity",
    "validate_evidence_lineage",
    "validate_grain",
    "validate_ranking",
]


class AnalyticalIntegrityError(AssertionError):
    """An analytical claim is not faithful to the evidence or the request.

    Subclasses ``AssertionError`` so the runtime's verifier loop treats it as
    a recoverable rejection and feeds the message back to the model.
    """


# Accumulated float64 rounding error, not a business threshold. Summing a few
# thousand doubles in a different order moves the result by parts in 1e-13 to
# 1e-12 (roughly 4,500 ULP at this setting), which is what turns 926400.0 into
# 926400.0000001. At one trillion this is one dollar; anything larger is left
# to the caller's absolute and relative tolerances, which are the only
# materiality rules. Callers may pass ``noise_relative_tolerance=0`` to
# disable even this.
DEFAULT_NOISE_RELATIVE_TOLERANCE = 1e-12

_DIRECTIONS = (None, "increase", "decrease")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"^(?P<sign>[-+]?)\s*(?P<currency>[$€£¥])?\s*(?P<sign2>[-+]?)"
    r"(?P<body>\d[\d,]*(?:\.\d+)?|\.\d+)\s*"
    r"(?P<suffix>%|percent|pct|bps|k|m|mm|b|bn|t|thousand|million|billion|trillion)?$",
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIER = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}


def _as_float(value: Any) -> float | None:
    """Coerce a scalar or a formatted string to a finite float, else None.

    Accepts ``926,400.00``, ``$4.2M``, ``-18%``, ``12 bps`` and numpy scalars.
    Percent and basis points are returned on their printed scale (``18``,
    not ``0.18``): a claim compares two printed values, so the scale only
    has to be consistent between them.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return _as_float(value.item())
        except Exception:
            return None
    if not isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    text = value.strip().replace("−", "-")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    body = match.group("body").replace(",", "")
    try:
        number = float(body)
    except ValueError:
        return None
    suffix = (match.group("suffix") or "").lower()
    number *= _SUFFIX_MULTIPLIER.get(suffix, 1.0)
    if "-" in (match.group("sign") + match.group("sign2")):
        number = -number
    return number if math.isfinite(number) else None


def is_material_change(
    current: Any,
    baseline: Any,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    direction: str | None = None,
    noise_relative_tolerance: float = DEFAULT_NOISE_RELATIVE_TOLERANCE,
) -> bool:
    """True when ``current`` differs from ``baseline`` by more than noise
    and more than the caller's materiality rule.

    ``absolute_tolerance`` is in the values' own unit; ``relative_tolerance``
    is a fraction of ``abs(baseline)`` and is skipped when the baseline is
    zero, because no relative rule is meaningful there. Both default to
    zero. Separately, ``noise_relative_tolerance`` (default one part in
    1e12, about 4,500 ULP) absorbs float64 rounding error such as summation
    order; it is numerical, not analytical, and can be set to zero.
    ``direction`` narrows the answer to ``"increase"`` or ``"decrease"``.
    Missing, NaN and unparsable values are never a material change.

    No currency, percentage or business threshold is assumed. The analysis
    decides what matters and passes it in.
    """
    if direction not in _DIRECTIONS:
        raise ValueError("direction must be None, 'increase' or 'decrease'")
    if absolute_tolerance < 0 or relative_tolerance < 0 or noise_relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    current_value = _as_float(current)
    baseline_value = _as_float(baseline)
    if current_value is None or baseline_value is None:
        return False
    difference = current_value - baseline_value
    if difference == 0 or (
        noise_relative_tolerance > 0
        and math.isclose(
            current_value,
            baseline_value,
            rel_tol=noise_relative_tolerance,
            abs_tol=0.0,
        )
    ):
        return False
    if abs(difference) <= absolute_tolerance:
        return False
    if (
        relative_tolerance > 0
        and baseline_value != 0
        and abs(difference) <= relative_tolerance * abs(baseline_value)
    ):
        return False
    if direction == "increase":
        return difference > 0
    if direction == "decrease":
        return difference < 0
    return True


def change_direction(
    current: Any,
    baseline: Any,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
    noise_relative_tolerance: float = DEFAULT_NOISE_RELATIVE_TOLERANCE,
) -> str:
    """``"increase"``, ``"decrease"``, ``"flat"`` or ``"unknown"``.

    ``"flat"`` covers both exact equality and differences inside the
    materiality rule, so prose built on this never calls noise a trend.
    """
    if _as_float(current) is None or _as_float(baseline) is None:
        return "unknown"
    for candidate in ("increase", "decrease"):
        if is_material_change(
            current,
            baseline,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            direction=candidate,
            noise_relative_tolerance=noise_relative_tolerance,
        ):
            return candidate
    return "flat"


# ---------------------------------------------------------------------------
# Candidate tuples
# ---------------------------------------------------------------------------


def _is_dataframe(value: Any) -> bool:
    return hasattr(value, "merge") and hasattr(value, "columns")


def _candidate_tuples(candidates: Any, keys: Sequence[str]) -> set[tuple[Any, ...]]:
    if _is_dataframe(candidates):
        missing = [key for key in keys if key not in candidates.columns]
        if missing:
            raise AnalyticalIntegrityError(
                f"candidates lack the key columns {missing}; keys={list(keys)}"
            )
        return {tuple(row) for row in candidates[list(keys)].itertuples(index=False, name=None)}
    tuples: set[tuple[Any, ...]] = set()
    for item in candidates:
        if isinstance(item, Mapping):
            missing = [key for key in keys if key not in item]
            if missing:
                raise AnalyticalIntegrityError(
                    f"candidate {dict(item)!r} lacks the keys {missing}"
                )
            tuples.add(tuple(item[key] for key in keys))
        else:
            values = tuple(item) if isinstance(item, (list, tuple)) else (item,)
            if len(values) != len(keys):
                raise AnalyticalIntegrityError(
                    f"candidate {values!r} has {len(values)} values for {len(keys)} keys"
                )
            tuples.add(values)
    return tuples


def restrict_to_candidate_tuples(
    frame: Any,
    candidates: Any,
    *,
    keys: Sequence[str],
) -> Any:
    """Keep only rows of ``frame`` whose ``keys`` form a selected candidate.

    ``candidates`` may be a DataFrame, a list of dicts, or a list of tuples
    in ``keys`` order. The restriction is on the compound identity, never on
    each dimension separately: candidates ``(Cloud, US, Enterprise)`` and
    ``(ADC, EMEA, Telco)`` do not admit ``(Cloud, EMEA, Telco)``, which is
    exactly what three independent ``.isin(...)`` filters would have done.

    Works on a pandas DataFrame (returns a DataFrame) or on a list of dict
    rows (returns a list), so the same call serves a CSV, a Lakehouse query
    result, or a semantic-model aggregate.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("keys must name at least one column")
    if _is_dataframe(frame):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise AnalyticalIntegrityError(
                f"frame lacks the key columns {missing}; keys={keys}"
            )
        if _is_dataframe(candidates):
            selected = candidates[keys].drop_duplicates()
        else:
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover - pandas ships with sempy
                raise RuntimeError(
                    "restrict_to_candidate_tuples on a DataFrame requires pandas"
                ) from exc
            selected = pd.DataFrame(
                sorted(_candidate_tuples(candidates, keys), key=repr), columns=keys
            )
        return frame.merge(selected, on=keys, how="inner")
    wanted = _candidate_tuples(candidates, keys)
    kept = []
    for row in frame:
        if not isinstance(row, Mapping):
            raise AnalyticalIntegrityError(
                "frame must be a DataFrame or a sequence of mapping rows"
            )
        if tuple(row.get(key) for key in keys) in wanted:
            kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "of", "by", "and", "or", "in", "for", "on", "to", "their",
    "its", "its", "business", "overall", "total", "each", "per", "with",
    "descending", "ascending", "order", "desc", "asc", "then", "which", "that",
}
_SUFFIXES = ("ations", "ation", "ition", "tions", "tion", "ings", "ing", "ies", "ers", "er", "ed", "es", "s")


def _stem(word: str) -> str:
    word = word.lower()
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: Any) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text)
    return [_stem(w) for w in words if w.lower() not in _STOPWORDS and len(w) > 1]


def _stems_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    left_set = set(left)
    right_set = set(right)
    if left_set & right_set:
        return True
    for a in left_set:
        for b in right_set:
            if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
                return True
    return False


def concept_head(concept: str) -> str:
    """The token a ranking concept is really about.

    "business impact of deterioration" is about impact, not deterioration;
    "churn risk" is about risk. The head is the last non-stopword before an
    "of", else the last non-stopword.
    """
    text = str(concept or "")
    before_of = re.split(r"\bof\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    tokens = _tokens(before_of) or _tokens(text)
    return tokens[-1] if tokens else ""


def validate_ranking(
    *,
    requested_concept: str,
    operational_definition: str | None,
    ranking_metric: str,
    ranking_values: Sequence[Any] | None = None,
    descending: bool = True,
) -> dict[str, Any]:
    """Check that a ranking answers the concept that was requested.

    A request to rank by "business impact" is not answered by sorting on
    current size, and "business impact of deterioration" is not answered by
    a deterioration rate: severity is not economic magnitude. Lexical overlap
    between the metric name and the concept certifies nothing on its own.
    The operational definition must name the metric and relate it to the
    head of the concept (impact, risk, ...); when the metric does not itself
    name the concept it is recorded as a declared proxy. When
    ``ranking_values`` are given they must already be in the requested
    order. Raises ``AnalyticalIntegrityError`` otherwise; returns a small
    summary for the answer to display.
    """
    concept_tokens = _tokens(requested_concept)
    metric_tokens = _tokens(ranking_metric)
    definition = (operational_definition or "").strip()
    if not concept_tokens:
        raise ValueError("requested_concept must name the ranking concept")
    if not metric_tokens:
        raise AnalyticalIntegrityError(
            f"Ranking by {requested_concept!r} needs a named metric; none was given."
        )
    if len(definition.split()) < 3:
        raise AnalyticalIntegrityError(
            f"Ranking by {requested_concept!r} needs an operational definition "
            f"of the metric {ranking_metric!r} (what it measures and how it is "
            "computed). State it before sorting."
        )
    head = concept_head(requested_concept)
    definition_tokens = _tokens(definition)
    names_metric = _stems_overlap(definition_tokens, metric_tokens)
    relates_to_concept = _stems_overlap(definition_tokens, [head]) if head else False
    if not names_metric or not relates_to_concept:
        raise AnalyticalIntegrityError(
            f"The task asked to rank by {requested_concept!r} but the ranking "
            f"metric is {ranking_metric!r} and the operational definition "
            f"({definition[:80]!r}) does not establish how {ranking_metric!r} "
            f"measures {head or requested_concept!r}. Define the metric in terms "
            f"of {head or requested_concept!r}, or derive one that represents it "
            "and sort by that."
        )
    proxy_justified = not _stems_overlap(metric_tokens, [head]) if head else False
    count = 0
    if ranking_values is not None:
        numbers = [_as_float(v) for v in ranking_values]
        if any(n is None for n in numbers):
            raise AnalyticalIntegrityError(
                f"ranking_values for {ranking_metric!r} must all be numeric."
            )
        ordered = sorted(numbers, reverse=descending)
        if numbers != ordered:
            raise AnalyticalIntegrityError(
                f"The result is not sorted by {ranking_metric!r} "
                f"({'descending' if descending else 'ascending'}); the values "
                f"run {numbers[:6]}. Sort by the ranking metric before submitting."
            )
        count = len(numbers)
    return {
        "concept": requested_concept,
        "metric": ranking_metric,
        "definition": definition,
        "proxy": proxy_justified,
        "count": count,
    }


@dataclass(frozen=True)
class RankingRequest:
    """A ranking concept found in the task text."""

    concept: str
    tokens: tuple[str, ...]
    phrase: str


_RANKING_REQUEST_RES = (
    re.compile(
        r"\b(?:rank(?:ed|ing)?|prioriti[sz]e[ds]?|sort(?:ed)?|order(?:ed)?|top\s+\d+|bottom\s+\d+)"
        r"\b[^.?!;\n]{0,80}?\bby\s+(?:the\s+|their\s+|its\s+)?(?P<concept>[^.,;:?!\n]+)",
        re.IGNORECASE,
    ),
)
_CONCEPT_CUTS = re.compile(
    r"\s+(?:and|with|using|for|over|across|within|in|from|so|then|per|among)\s+|\(",
    re.IGNORECASE,
)


def infer_requested_ranking(task_text: str | None) -> RankingRequest | None:
    """Find "rank ... by <concept>" in a task; None when no ranking is asked.

    Only explicit "by <concept>" forms count. Superlatives such as "the
    largest customers" are left alone: they name a size, not a concept
    that needs an operational definition.
    """
    if not task_text:
        return None
    for pattern in _RANKING_REQUEST_RES:
        match = pattern.search(task_text)
        if not match:
            continue
        raw = match.group("concept")
        concept = _CONCEPT_CUTS.split(raw, maxsplit=1)[0].strip(" \t\"'")
        words = concept.split()
        if len(words) > 6:
            concept = " ".join(words[:6])
        tokens = tuple(_tokens(concept))
        if not tokens:
            continue
        return RankingRequest(concept=concept, tokens=tokens, phrase=match.group(0).strip())
    return None


# "ranked by X", "rank them by X", "sorted (descending) by X", "ordered by X"
_RANKED_BY_RE = re.compile(
    r"\b(?:rank(?:ed|ing)?|sort(?:ed)?|order(?:ed)?)\b(?:\s+\w+){0,3}?\s+by\s+(?P<metric>[^.,;:\n]{1,80})",
    re.IGNORECASE,
)
# "Ranking metric: X", "Ranking metric (label): X", "ranking metric = X",
# "Ranking metric (X)". The parenthetical is a label for the concept, the
# metric is what follows the colon; both are kept as declared vocabulary.
_RANKING_METRIC_RE = re.compile(
    r"\brank(?:ed|ing)?\s+(?:metric|measure|field|criterion|key|basis)\b"
    r"\s*(?:\((?P<label>[^)\n]{0,80})\))?\s*(?:[:=]\s*(?P<metric>[^\n]{1,100}))?",
    re.IGNORECASE,
)


def declared_ranking_phrases(text: str | None) -> list[str]:
    """The metric phrases an answer says it ranked by.

    Reads "ranked by X", "rank them by X", "sorted by X", "ordered by X",
    "Ranking metric: X", "Ranking metric (label): X" and "Ranking metric
    (X)". Used so a ranking the answer discloses is never mistaken for
    drift because of how the code named its column, and so a "ranked by"
    phrase is read together with the metric declaration it refers to.
    """
    body = str(text or "")
    phrases = [m.group("metric").strip() for m in _RANKED_BY_RE.finditer(body)]
    phrases.extend(_declared_metric_phrases(body))
    return [p for p in phrases if p]


def _first_clause(text: str) -> str:
    """Up to the first sentence or clause end: ". ", ";" or a line break."""
    return re.split(r"(?<=\S)\.\s|;|\n", text, maxsplit=1)[0].strip()


def _declared_metric_phrases(text: str) -> list[str]:
    """Only the explicit "ranking metric: ..." declarations."""
    phrases: list[str] = []
    for m in _RANKING_METRIC_RE.finditer(text):
        for part in (m.group("label"), m.group("metric")):
            if part and part.strip():
                phrases.append(_first_clause(part))
    return [p for p in phrases if p]


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence or clause that contains ``text[start:end]``."""
    boundaries = re.compile(r"(?<=\S)\.\s|;|\n")
    left = 0
    for m in boundaries.finditer(text, 0, start):
        left = m.end()
    right = len(text)
    m = boundaries.search(text, end)
    if m:
        right = m.start()
    return text[left:right]


def check_ranking_disclosure(text: str | None, request: RankingRequest) -> list[str]:
    """Problems with how an answer discloses the ranking it was asked for.

    The answer must mention the requested concept, so the reader can see
    the ranking metric, and if it says "ranked by X" then X must be the
    concept or an explicitly labelled proxy.
    """
    problems: list[str] = []
    body = str(text or "")
    answer_tokens = _tokens(body)
    head_tokens = [concept_head(request.concept)] if concept_head(request.concept) else list(request.tokens)
    if not _stems_overlap(head_tokens, answer_tokens):
        problems.append(
            f"The task asked to rank by {request.concept!r}, but the answer never "
            f"mentions {request.concept!r}. Show the ranking metric (a column such "
            f"as 'Estimated {request.concept}' or a line 'ranked by ...') so the "
            "ranking is auditable."
        )
    declared = [tok for phrase in _declared_metric_phrases(body) for tok in _tokens(phrase)]
    head = concept_head(request.concept)
    for match in _RANKED_BY_RE.finditer(body):
        metric_tokens = _tokens(match.group("metric"))
        if not metric_tokens:
            continue
        if _stems_overlap(request.tokens, metric_tokens):
            continue
        if "proxy" in body.lower():
            continue
        # "ranked by USD loss" is consistent with a declared "Ranking metric:
        # absolute ARR loss in USD"; the declaration is where the reader
        # audits the choice.
        if declared and _stems_overlap(declared, metric_tokens):
            continue
        # or the concept is tied to it in the same sentence
        sentence = _sentence_around(body, match.start(), match.end())
        if head and _stems_overlap([head], _tokens(sentence)):
            continue
        problems.append(
            f"The answer says {match.group(0).strip()!r} but the task asked to rank "
            f"by {request.concept!r}. Rank by a metric for {request.concept!r}, or "
            "state that the field used is a proxy and why."
        )
    return problems


# "Change: $-0 (-0.0%)", "Change in ARR: $0", "abs drop = $0", "impact metric (ARR drop): $0"
_ZERO_CHANGE_RE = re.compile(
    r"\b(?P<label>(?:net\s+|abs(?:olute)?\s+|total\s+)?(?:change|delta|difference|decline|decrease|drop|loss|"
    r"increase|growth|gain|movement|impact(?:\s+metric)?)(?:\s+in\s+\w+)?(?:\s*\([^)\n]{0,40}\))?)"
    r"\s*[:=]\s*(?:[-+]?\s?[$€£¥]?\s?[-+]?0(?:\.0+)?(?!\d|\.\d))(?:\s?(?:%|USD|EUR|GBP|units?|users?))?(?!\d|\.\d)",
    re.IGNORECASE,
)
_CHANGE_TASK_RE = re.compile(
    r"\b(?:deteriorat|declin|decreas|drop|fall|fell|grow|grew|increas|chang|trend|improv|worsen|movement)",
    re.IGNORECASE,
)


def task_asks_about_change(task_text: str | None) -> bool:
    """True when the task is about movement, so a zero-change item is a finding."""
    return bool(_CHANGE_TASK_RE.search(str(task_text or "")))


def check_zero_change_items(text: str | None, *, absolute_tolerance: float = 0.0) -> list[str]:
    """Items reported with a change of zero inside an answer about change.

    Only meaningful when the task asked for items that moved; the caller
    decides that with :func:`task_asks_about_change`. A listed segment whose
    own line says "Change: $0" or "impact metric: $0" is float noise or a
    flat item that should have been excluded or called flat.
    """
    problems: list[str] = []
    body = str(text or "")
    for match in _ZERO_CHANGE_RE.finditer(body):
        line_start = body.rfind("\n", 0, match.start()) + 1
        line_end = body.find("\n", match.end())
        line = body[line_start: line_end if line_end >= 0 else len(body)].strip()
        problems.append(
            f"{line[:120]!r} reports a {match.group('label').strip().lower()} of zero for an item the "
            "answer presents as having moved. A zero change is float noise or a flat item: exclude "
            "it under the stated materiality rule, or call it flat."
        )
        if len(problems) >= 3:
            break
    return problems


_REPR_LEAK_RE = re.compile(
    r"<bound method |<function [\w.<>]+ at 0x|<class '|<[\w.]+ object at 0x|"
    r"<generator object|<map object|<filter object|\bNaN\b(?=[^\n]*\bdtype\b)|"
    r"dtype: (?:object|float64|int64)\s*$",
    re.MULTILINE,
)


def check_answer_hygiene(text: str | None) -> list[str]:
    """Problems that mean the answer text was never a finished answer.

    Catches Python object representations leaking into prose, such as
    ``<bound method Series.prod of ...>``, which a live repair turn produced
    when a fix introduced a bug and the model submitted anyway. Formatting
    only; no source query is needed.
    """
    body = str(text or "")
    match = _REPR_LEAK_RE.search(body)
    if not match:
        return []
    snippet = body[max(0, match.start() - 40): match.end() + 60].replace("\n", " ")
    return [
        f"The answer contains a raw Python object representation ({snippet.strip()!r}). "
        "Format the values (names, numbers) rather than printing objects, then submit again."
    ]


# ---------------------------------------------------------------------------
# Grain
# ---------------------------------------------------------------------------


def _norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _dimension_parts(label: Any) -> tuple[str | None, str]:
    """(table, leaf) for ``Table[Column]``; (None, leaf) for a bare name."""
    text = str(label)
    match = re.match(r"^\s*'?([^'\[\]]+?)'?\s*\[([^\[\]]+)\]\s*$", text)
    if match:
        return _norm_text(match.group(1)), _norm_text(match.group(2))
    return None, _norm_text(text)


def _norm_dimension(label: Any) -> str:
    return _dimension_parts(label)[1]


def _match_dimensions(
    requested: Sequence[Any], actual: Sequence[Any]
) -> tuple[list[str], list[str]]:
    """Missing and extra labels, with table qualification when leaves collide.

    ``Products[Line Of Business]`` matches ``line_of_business`` because that
    leaf is unique. When a leaf appears more than once on either side
    (``Products[Region]`` and ``Sold To[Region]``) a match needs the table
    too, so those two never collapse into one and a bare ``region`` cannot
    stand in for either.
    """
    req = [(*_dimension_parts(d), str(d)) for d in requested]
    act = [(*_dimension_parts(d), str(d)) for d in actual]
    req_leaf_counts: dict[str, int] = {}
    act_leaf_counts: dict[str, int] = {}
    for _t, leaf, _o in req:
        req_leaf_counts[leaf] = req_leaf_counts.get(leaf, 0) + 1
    for _t, leaf, _o in act:
        act_leaf_counts[leaf] = act_leaf_counts.get(leaf, 0) + 1
    used: set[int] = set()
    missing: list[str] = []
    for r_table, r_leaf, r_orig in req:
        candidates = [
            (i, a_table) for i, (a_table, a_leaf, _o) in enumerate(act)
            if a_leaf == r_leaf and i not in used
        ]
        if not candidates:
            missing.append(r_orig)
            continue
        ambiguous = req_leaf_counts[r_leaf] > 1 or act_leaf_counts[r_leaf] > 1
        if ambiguous:
            exact = [i for i, a_table in candidates if r_table is not None and a_table == r_table]
            if not exact:
                missing.append(r_orig)
                continue
            used.add(exact[0])
        else:
            i, a_table = candidates[0]
            if r_table is not None and a_table is not None and a_table != r_table:
                missing.append(r_orig)
                continue
            used.add(i)
    extra = [orig for i, (_t, _l, orig) in enumerate(act) if i not in used]
    return missing, extra


def validate_grain(
    *,
    requested: Sequence[Any],
    actual: Sequence[Any],
    explanation: str | None = None,
) -> dict[str, Any]:
    """Detect a silent change of analytical grain.

    ``requested`` and ``actual`` are dimension labels; ``Products[Line Of
    Business]`` and ``line_of_business`` compare equal when that leaf name
    is unique, while ``Products[Region]`` and ``Sold To[Region]`` stay
    distinct. A dropped or added dimension is allowed only with an
    ``explanation`` the answer will show.
    """
    missing, extra = _match_dimensions(requested, actual)
    explained = bool((explanation or "").strip())
    if (missing or extra) and not explained:
        raise AnalyticalIntegrityError(
            f"The analytical grain changed silently: requested {list(requested)} "
            f"but the result is at {list(actual)}"
            + (f" (missing {missing})" if missing else "")
            + (f" (added {extra})" if extra else "")
            + ". Restore the requested grain or state why it changed."
        )
    return {
        "requested": [str(d) for d in requested],
        "actual": [str(d) for d in actual],
        "missing": missing,
        "extra": extra,
        "explained": explained,
    }


# ---------------------------------------------------------------------------
# Directional claims in prose
# ---------------------------------------------------------------------------

_DECREASE_STEMS = (
    "declin", "decreas", "fell", "fall", "drop", "deteriorat", "worsen", "shrank",
    "shrink", "slid", "slip", "contract", "eroded", "erod", "lower", "weaken",
)
_INCREASE_STEMS = (
    "increas", "grew", "grow", "rose", "rise", "risen", "improv", "climb", "expand",
    "gain", "strengthen", "higher", "jump", "surg",
)
# The suffix must end at a word boundary: under IGNORECASE a bare ``[KMBT]``
# would otherwise read "100 this year" as one hundred trillion.
_NUMBER_TOKEN = (
    r"[-+]?[$€£¥]?\s?\(?[-+]?\d[\d,]*(?:\.\d+)?\)?"
    r"(?:\s?(?:%|(?:percent|pct|bps|[KMBT](?:illion)?|thousand|million|billion|trillion|mm|bn)(?![A-Za-z])))?"
)
# A unit word after the number ("926,400.00 USD", "18 users") and an
# optional "in <period>" before the "to".
_UNIT_WORD = r"(?:\s?(?:USD|EUR|GBP|dollars?|euros?|units?|users?|customers?|seats?|accounts?|rows?))?"
_PERIOD_TAG = r"(?:\s+(?:in|for|at|during)\s+[\w/\-]+)?"
_FROM_TO_RE = re.compile(
    r"(?P<lead>\b\w+\b(?:\s+\w+){0,4}?)\s+from\s+(?P<a>" + _NUMBER_TOKEN + r")" + _UNIT_WORD + _PERIOD_TAG
    + r"\s+to\s+(?P<b>" + _NUMBER_TOKEN + r")" + _UNIT_WORD + _PERIOD_TAG
    + r"(?P<tail>[^.;\n]{0,80})",
    re.IGNORECASE,
)
# "a decrease of 0.00 USD", "an increase of 0%": a movement word with a zero amount
_CHANGE_OF_RE = re.compile(
    r"\b(?P<word>decrease|decline|drop|reduction|loss|fall|increase|growth|gain|rise|improvement|deterioration)"
    r"\s+of\s+(?P<amount>" + _NUMBER_TOKEN + r")" + _UNIT_WORD,
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DirectionalClaim:
    """A "from A to B" statement with the direction the prose asserts."""

    text: str
    baseline: float
    current: float
    claimed: str
    actual: str


def _claimed_direction(*fragments: str) -> str | None:
    words = re.findall(r"[a-z]+", " ".join(fragments).lower())
    for word in words:
        if any(word.startswith(stem) for stem in _DECREASE_STEMS):
            return "decrease"
        if any(word.startswith(stem) for stem in _INCREASE_STEMS):
            return "increase"
    return None


def parse_directional_claims(
    text: str | None,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> list[DirectionalClaim]:
    """Extract "<moved> from A to B" claims and compare prose to numbers."""
    claims: list[DirectionalClaim] = []
    for match in _FROM_TO_RE.finditer(str(text or "")):
        baseline = _as_float(match.group("a").replace(" ", ""))
        current = _as_float(match.group("b").replace(" ", ""))
        if baseline is None or current is None:
            continue
        claimed = _claimed_direction(match.group("lead"), match.group("tail"))
        if claimed is None:
            continue
        actual = change_direction(
            current,
            baseline,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        claims.append(
            DirectionalClaim(
                text=match.group(0).strip(),
                baseline=baseline,
                current=current,
                claimed=claimed,
                actual=actual,
            )
        )
    return claims


def check_directional_claims(
    text: str | None,
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> list[str]:
    """Problems where prose direction disagrees with the printed numbers.

    Flags "declined from 926,400.00 to 926,400.00" (values effectively
    equal) and "fell from 3.9M to 4.2M" (opposite direction). Pure
    formatting checks: no source query is needed.
    """
    problems: list[str] = []
    for match in _CHANGE_OF_RE.finditer(str(text or "")):
        amount = _as_float(match.group("amount").replace(" ", ""))
        if amount is None:
            continue
        if amount == 0 or abs(amount) <= absolute_tolerance:
            problems.append(
                f"{match.group(0).strip()!r} names a {match.group('word').lower()} whose amount is "
                f"{amount:g}, which is no movement at all under the materiality rule. Call it "
                "flat, or state the unrounded movement and the threshold that makes it material."
            )
    for claim in parse_directional_claims(
        text,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ):
        if claim.actual == "flat":
            problems.append(
                f"{claim.text!r} describes a {claim.claimed} but the values "
                f"({claim.baseline:g} to {claim.current:g}) are effectively equal "
                "under the materiality rule. Call it flat, or state the unrounded "
                "movement and the threshold that makes it material."
            )
        elif claim.actual != claim.claimed:
            problems.append(
                f"{claim.text!r} describes a {claim.claimed} but the values "
                f"({claim.baseline:g} to {claim.current:g}) show a {claim.actual}. "
                "Make the prose agree with the numbers."
            )
    return problems


# ---------------------------------------------------------------------------
# Claim levels and evidence lineage
# ---------------------------------------------------------------------------

_CAUSAL_RE = re.compile(
    r"\b(?:caus(?:e|es|ed|ing)|because of|due to|driven by|drives?|drove|"
    r"led to|leads? to|result(?:s|ed)? in|root cause|attributable to|"
    r"explains? the|responsible for)\b",
    re.IGNORECASE,
)
_INTERPRETATION_RE = re.compile(
    r"\b(?:indicat(?:es|ing)|suggest(?:s|ing)?|implies|signals?|points? to|"
    r"weakening|strengthening|healthy|at risk|concerning|encouraging)\b",
    re.IGNORECASE,
)
_DERIVED_RE = re.compile(
    r"(?:\d(?:\.\d+)?\s?%|\bpercent\b|\brate\b|\bshare\b|\bratio\b|\bper\b|"
    r"\baverage\b|\bmean\b|\bdelta\b|\bchange of\b|\bnet change\b|\bgrowth\b)",
    re.IGNORECASE,
)
_CLAIM_LEVELS = ("observed", "derived", "interpretation", "causal")


def classify_claim_level(text: str | None) -> str:
    """Classify prose as observed, derived, interpretation or causal.

    Causal wording wins, then interpretive wording, then derived-metric
    wording; anything else is an observation. A heuristic screen for the
    verifier, not a semantic judgement.
    """
    body = str(text or "")
    if _CAUSAL_RE.search(body):
        return "causal"
    if _INTERPRETATION_RE.search(body):
        return "interpretation"
    if _DERIVED_RE.search(body):
        return "derived"
    return "observed"


@dataclass
class IntegrityReport:
    """Findings from :func:`validate_analysis_integrity`."""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_for_problems(self) -> None:
        if self.problems:
            raise AnalyticalIntegrityError("\n".join(f"- {p}" for p in self.problems))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": list(self.checks),
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def _claim_sources(claim: Mapping[str, Any]) -> list[str]:
    source = claim.get("source")
    if source is None:
        return []
    if isinstance(source, str):
        return [source]
    return [str(s) for s in source]


def _claim_label(claim: Mapping[str, Any], index: int) -> str:
    return str(claim.get("claim") or claim.get("metric") or f"claim #{index + 1}")


def validate_evidence_lineage(
    claims: Sequence[Mapping[str, Any]],
    *,
    sources: Iterable[str] | None = None,
    joins: Sequence[Mapping[str, Any]] | None = None,
    disclosures: Mapping[str, Any] | None = None,
) -> IntegrityReport:
    """Check provenance and cross-source compatibility of a set of claims.

    Each claim is a mapping. Recognised keys, all optional except where a
    check needs them: ``claim`` (text), ``source`` (alias or list of
    aliases), ``metric``, ``value``, ``unit``, ``time_basis``,
    ``definition``, ``entity``, ``level`` (observed, derived,
    interpretation, causal), ``direction`` (increase or decrease, for
    signals about the same entity), ``causal_evidence``.

    Single-source claims are checked for provenance and for causal
    language without causal evidence. When more than one source
    contributes, the cross-source checks activate: differing
    ``time_basis`` values need ``disclosures["period_alignment"]``; the
    same ``metric`` with different ``unit`` or ``definition`` across
    sources needs ``disclosures["unit_conversion"]`` or
    ``disclosures["metric_definitions_reconciled"]``; every join in
    ``joins`` needs an explicit ``key`` and an inferred ``match`` is only
    a note when it carries a ``confidence``; conflicting ``direction``
    signals about one entity need ``disclosures["contradiction_surfaced"]``.
    Nothing here is specific to any source type.
    """
    report = IntegrityReport()
    known = {str(s) for s in sources} if sources is not None else None
    disclosed = dict(disclosures or {})
    report.checks.append("provenance")
    all_sources: set[str] = set()
    for index, claim in enumerate(claims):
        label = _claim_label(claim, index)
        claim_sources = _claim_sources(claim)
        all_sources.update(claim_sources)
        material = claim.get("value") is not None or claim.get("level") in ("observed", "derived")
        if material and not claim_sources:
            report.problems.append(
                f"{label!r} has no source. Attribute every material claim to the "
                "input that supports it."
            )
        if known is not None:
            unknown = [s for s in claim_sources if s not in known]
            if unknown:
                report.problems.append(
                    f"{label!r} cites unknown source(s) {unknown}; known inputs are "
                    f"{sorted(known)}."
                )
        level = claim.get("level")
        if level is not None and level not in _CLAIM_LEVELS:
            report.problems.append(
                f"{label!r} has level {level!r}; use one of {list(_CLAIM_LEVELS)}."
            )
        text_level = classify_claim_level(claim.get("claim"))
        if (level == "causal" or text_level == "causal") and not claim.get("causal_evidence"):
            report.problems.append(
                f"{label!r} uses causal language without causal evidence. Describe "
                "the association, or name the evidence that establishes cause."
            )
    report.checks.append("claim_levels")

    if len(all_sources) < 2:
        return report
    report.checks.append("cross_source")

    time_bases = {
        str(claim["time_basis"]): _claim_sources(claim)
        for claim in claims
        if claim.get("time_basis") is not None
    }
    if len(time_bases) > 1 and not disclosed.get("period_alignment"):
        report.problems.append(
            "Sources cover different periods "
            + "; ".join(f"{basis} ({', '.join(src)})" for basis, src in time_bases.items())
            + ". Align them to a common valid period, or state the mismatch "
            "(disclosures['period_alignment'])."
        )

    by_metric: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        metric = claim.get("metric")
        if metric is not None:
            by_metric.setdefault(str(metric).strip().lower(), []).append(claim)
    for metric, group in by_metric.items():
        group_sources = {s for claim in group for s in _claim_sources(claim)}
        if len(group_sources) < 2:
            continue
        units = {str(claim["unit"]) for claim in group if claim.get("unit") is not None}
        if len(units) > 1 and not disclosed.get("unit_conversion"):
            report.problems.append(
                f"Metric {metric!r} is reported in different units across sources "
                f"({sorted(units)}). Convert explicitly (disclosures['unit_conversion']) "
                "before comparing."
            )
        definitions = {
            str(claim["definition"]).strip() for claim in group if claim.get("definition")
        }
        if len(definitions) > 1 and not disclosed.get("metric_definitions_reconciled"):
            report.problems.append(
                f"Metric {metric!r} has different definitions across sources; a "
                "similar name does not make them equivalent. Reconcile the "
                "definitions (disclosures['metric_definitions_reconciled']) or keep "
                "them separate."
            )

    for join in joins or ():
        join_sources = [str(s) for s in join.get("sources", ())]
        key = join.get("key")
        match = str(join.get("match") or ("explicit" if key else "unknown")).lower()
        if len(join_sources) < 2:
            continue
        if not key:
            report.problems.append(
                f"Joining {join_sources} has no shared key. Prefer an explicit "
                "identifier such as customer_id or subscription_id; name-based "
                "matching must be declared as inferred with a confidence."
            )
            continue
        if match == "inferred":
            if join.get("confidence") is None:
                report.problems.append(
                    f"Joining {join_sources} on {key!r} is inferred but carries no "
                    "confidence. State how ambiguous the match is."
                )
            else:
                report.notes.append(
                    f"Join of {join_sources} on {key!r} is inferred "
                    f"(confidence {join['confidence']}); treat dependent figures as inferred."
                )

    by_entity: dict[str, dict[str, set[str]]] = {}
    for claim in claims:
        direction = claim.get("direction")
        entity = claim.get("entity")
        if direction is None or entity is None:
            continue
        for source in _claim_sources(claim):
            by_entity.setdefault(str(entity), {}).setdefault(str(direction).lower(), set()).add(source)
    for entity, directions in by_entity.items():
        if len(directions) > 1 and not disclosed.get("contradiction_surfaced"):
            summary = "; ".join(
                f"{direction} per {', '.join(sorted(src))}" for direction, src in directions.items()
            )
            report.problems.append(
                f"Sources disagree about {entity!r}: {summary}. Surface the conflict "
                "instead of resolving it (disclosures['contradiction_surfaced'])."
            )
    return report


# ---------------------------------------------------------------------------
# One entry point
# ---------------------------------------------------------------------------


def validate_analysis_integrity(
    *,
    requested_grain: Sequence[Any] | None = None,
    actual_grain: Sequence[Any] | None = None,
    grain_explanation: str | None = None,
    ranking: Mapping[str, Any] | None = None,
    directional_claims: Sequence[Mapping[str, Any]] | None = None,
    materiality: Mapping[str, float] | None = None,
    candidate_keys: Sequence[str] | None = None,
    candidates: Any = None,
    selected: Any = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
    sources: Iterable[str] | None = None,
    joins: Sequence[Mapping[str, Any]] | None = None,
    disclosures: Mapping[str, Any] | None = None,
    requested_ranking: str | None = None,
    answer_text: str | None = None,
    strict: bool = False,
) -> IntegrityReport:
    """Run every integrity check whose inputs were supplied.

    Nothing is required; a check activates only when its inputs are
    present, so a PDF-only task runs provenance checks and skips ranking,
    and a single-frame CSV task skips the cross-source checks.

    - ``requested_grain`` and ``actual_grain``: silent grain substitution.
    - ``ranking``: ``{"concept", "definition", "metric", "values",
      "descending"}`` passed to :func:`validate_ranking`.
    - ``directional_claims``: ``[{"label", "current", "baseline",
      "claimed"}]`` each checked with ``materiality`` (``absolute_tolerance``
      and ``relative_tolerance``).
    - ``candidate_keys`` with ``candidates`` and ``selected``: every
      selected row must be one of the candidate tuples.
    - ``claims``, ``sources``, ``joins``, ``disclosures``: provenance and
      cross-source reconciliation via :func:`validate_evidence_lineage`.
    - ``answer_text``: prose direction versus printed numbers, and ranking
      disclosure when ``requested_ranking`` names the concept.

    Returns an :class:`IntegrityReport`; with ``strict=True`` it raises
    ``AnalyticalIntegrityError`` when any problem was found.
    """
    report = IntegrityReport()
    tolerances = {
        "absolute_tolerance": float((materiality or {}).get("absolute_tolerance", 0.0)),
        "relative_tolerance": float((materiality or {}).get("relative_tolerance", 0.0)),
    }

    if requested_grain is not None and actual_grain is not None:
        report.checks.append("grain")
        try:
            validate_grain(
                requested=requested_grain,
                actual=actual_grain,
                explanation=grain_explanation,
            )
        except AnalyticalIntegrityError as exc:
            report.problems.append(str(exc))

    if ranking is not None:
        report.checks.append("ranking")
        try:
            validate_ranking(
                requested_concept=str(ranking.get("concept") or requested_ranking or ""),
                operational_definition=ranking.get("definition"),
                ranking_metric=str(ranking.get("metric") or ""),
                ranking_values=ranking.get("values"),
                descending=bool(ranking.get("descending", True)),
            )
        except AnalyticalIntegrityError as exc:
            report.problems.append(str(exc))

    if directional_claims:
        report.checks.append("materiality")
        for index, claim in enumerate(directional_claims):
            label = str(claim.get("label") or f"claim #{index + 1}")
            claimed = str(claim.get("claimed") or "").lower()
            if claimed in ("flat", "unchanged", "stable", "no change"):
                claimed = "flat"
            actual = change_direction(claim.get("current"), claim.get("baseline"), **tolerances)
            if actual == "unknown":
                report.problems.append(f"{label!r} has non-numeric current or baseline values.")
            elif claimed and claimed != actual:
                report.problems.append(
                    f"{label!r} is described as {claimed!r} but "
                    f"{claim.get('baseline')} to {claim.get('current')} is {actual!r} "
                    "under the declared materiality rule."
                )

    if candidate_keys and candidates is not None and selected is not None:
        report.checks.append("candidate_identity")
        try:
            allowed = _candidate_tuples(candidates, list(candidate_keys))
            chosen = _candidate_tuples(selected, list(candidate_keys))
        except AnalyticalIntegrityError as exc:
            report.problems.append(str(exc))
        else:
            strays = sorted(chosen - allowed, key=repr)
            if strays:
                report.problems.append(
                    f"{len(strays)} selected combination(s) were never candidates, e.g. "
                    f"{strays[:3]}. Restrict later steps to the candidate tuples "
                    "(restrict_to_candidate_tuples) instead of independent "
                    "per-dimension lists."
                )

    if claims is not None:
        lineage = validate_evidence_lineage(
            claims, sources=sources, joins=joins, disclosures=disclosures
        )
        report.checks.extend(lineage.checks)
        report.problems.extend(lineage.problems)
        report.notes.extend(lineage.notes)

    if answer_text:
        report.checks.append("answer_consistency")
        report.problems.extend(check_directional_claims(answer_text, **tolerances))
        if requested_ranking:
            request = infer_requested_ranking(f"rank by {requested_ranking}")
            if request is not None:
                report.problems.extend(check_ranking_disclosure(answer_text, request))

    if strict:
        report.raise_for_problems()
    return report
