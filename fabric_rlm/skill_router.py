"""Structured skill router (zero-LM-cost).

The router scores each candidate skill against a question's text and any
declared output-field tokens, honors `excludes` and `depends_on`, always
activates `specificity == "core"` skills, and caps the active set so the
turn-1 prompt stays small. Non-active high-priority skills surface as
*cards* (one-line summaries) so the LM can ask to activate them via the
`activate_skill` tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .skill_loader import Skill, SkillLoader


_KEYWORD_WEIGHT = 2
_OUTPUT_FIELD_WEIGHT = 1


@dataclass(frozen=True)
class RouteDecision:
    """Result of routing a single question."""

    active: tuple[str, ...]
    cards: tuple[str, ...]
    scores: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


class SkillRouter:
    """Pure-Python router over a fixed set of `Skill` objects."""

    def __init__(
        self,
        skills: Iterable[Skill],
        *,
        max_active_skills: int = 2,
        baseline_skill_names: Iterable[str] | None = None,
        candidate_specificities: Iterable[str] | None = None,
        include_dependencies: bool = True,
    ) -> None:
        self._skills: tuple[Skill, ...] = tuple(skills)
        self._by_name: dict[str, Skill] = {s.name: s for s in self._skills}
        self.max_active_skills = max(0, int(max_active_skills))
        # When None, fall back to "specificity == 'core'" as always-on
        # (legacy behavior). When provided (even empty), this list is the
        # exact set of always-on skill names.
        self.baseline_skill_names: tuple[str, ...] | None = (
            tuple(baseline_skill_names) if baseline_skill_names is not None else None
        )
        # When None, no specificity filter is applied to ranked candidates
        # (legacy behavior). When provided, only skills whose specificity is
        # in this set may be auto-elected by score.
        self.candidate_specificities: tuple[str, ...] | None = (
            tuple(candidate_specificities) if candidate_specificities is not None else None
        )
        self.include_dependencies = bool(include_dependencies)

    @classmethod
    def from_loader(
        cls,
        loader: SkillLoader,
        *,
        max_active_skills: int = 2,
        baseline_skill_names: Iterable[str] | None = None,
        candidate_specificities: Iterable[str] | None = None,
        include_dependencies: bool = True,
    ) -> "SkillRouter":
        skills: list[Skill] = []
        for name in loader.list_skills():
            try:
                skills.append(loader.load(name))
            except Exception:
                continue
        return cls(
            skills,
            max_active_skills=max_active_skills,
            baseline_skill_names=baseline_skill_names,
            candidate_specificities=candidate_specificities,
            include_dependencies=include_dependencies,
        )

    def route(
        self,
        question_text: str,
        *,
        explicit_skills: Optional[Iterable[str]] = None,
    ) -> RouteDecision:
        text = question_text or ""
        text_lower = text.lower()
        declared_fields = _extract_declared_fields(text)

        scores: dict[str, int] = {}
        reasons: dict[str, str] = {}
        for skill in self._skills:
            score = 0
            why: list[str] = []
            for kw in skill.applies_when_keywords:
                if not kw:
                    continue
                if kw.lower() in text_lower:
                    score += _KEYWORD_WEIGHT
                    why.append(f"kw:{kw}")
            for fld in skill.applies_when_output_fields:
                if not fld:
                    continue
                if fld in declared_fields:
                    score += _OUTPUT_FIELD_WEIGHT
                    why.append(f"out:{fld}")
            if score > 0:
                scores[skill.name] = score
                reasons[skill.name] = ",".join(why)

        # Always-on baseline. When `baseline_skill_names` is set, use it as
        # the exact override (even if empty). Otherwise fall back to the
        # legacy "specificity == 'core'" rule.
        if self.baseline_skill_names is not None:
            core_names = [n for n in self.baseline_skill_names if n in self._by_name]
        else:
            core_names = [s.name for s in self._skills if s.specificity == "core"]
        active: list[str] = []
        seen: set[str] = set()
        for name in core_names:
            if name not in seen:
                active.append(name)
                seen.add(name)
                reasons.setdefault(name, "always-on:core")

        # Explicit override (caller pinned skills).
        explicit = list(explicit_skills or [])
        for name in explicit:
            if name in self._by_name and name not in seen:
                active.append(name)
                seen.add(name)
                reasons.setdefault(name, "explicit")

        # Score-ranked candidates (excluding cores already added). Stable by
        # (-score, original index) so ties prefer files declared earlier.
        # Optionally filter by specificity for lean-router modes.
        order = {s.name: i for i, s in enumerate(self._skills)}
        allowed_specs: set[str] | None = (
            set(self.candidate_specificities)
            if self.candidate_specificities is not None
            else None
        )
        ranked = sorted(
            (
                n
                for n in scores
                if n not in seen
                and (
                    allowed_specs is None
                    or (n in self._by_name and self._by_name[n].specificity in allowed_specs)
                )
            ),
            key=lambda n: (-scores[n], order.get(n, 1_000_000)),
        )

        # Honor excludes from already-active skills.
        excluded: set[str] = set()
        for name in list(active):
            sk = self._by_name.get(name)
            if sk:
                excluded.update(sk.excludes)

        # Promote ranked candidates up to the cap.
        for name in ranked:
            if len(active) >= self.max_active_skills + len(core_names):
                break
            if name in excluded:
                continue
            sk = self._by_name.get(name)
            if sk is None:
                continue
            active.append(name)
            seen.add(name)
            excluded.update(sk.excludes)

        # Pull in dependencies of every active skill (also add to active so
        # their bodies are loaded; they don't count against the cap because
        # they're declared prerequisites). Skipped entirely when
        # `include_dependencies=False` (lean-router mode).
        if self.include_dependencies:
            for name in list(active):
                for dep in self._dep_closure(name):
                    if dep not in seen and dep in self._by_name:
                        active.append(dep)
                        seen.add(dep)
                        reasons.setdefault(dep, f"dep-of:{name}")

        # Cards = ranked-but-not-active high-priority skills. Prefer
        # utility-with-verifier first, then everything else still in `ranked`.
        cards: list[str] = []
        for name in ranked:
            if name in seen:
                continue
            sk = self._by_name.get(name)
            if sk is None:
                continue
            cards.append(name)
        # Stable order: utilities-with-verifier first.
        cards.sort(
            key=lambda n: (
                0 if (self._by_name[n].specificity == "utility" and self._by_name[n].verifier_present) else 1,
                -scores.get(n, 0),
                order.get(n, 1_000_000),
            )
        )

        return RouteDecision(
            active=tuple(active),
            cards=tuple(cards),
            scores=dict(scores),
            reasons=dict(reasons),
        )

    def _dep_closure(self, name: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        def walk(n: str) -> None:
            sk = self._by_name.get(n)
            if sk is None:
                return
            for d in sk.dependencies:
                if d in seen:
                    continue
                seen.add(d)
                walk(d)
                out.append(d)

        walk(name)
        return out

    def card_text(self, name: str) -> str:
        sk = self._by_name.get(name)
        if sk is None:
            return f"- {name}: (unknown skill)"
        summary = sk.summary or sk.title or sk.name
        verifier_tag = " [verifier]" if sk.verifier_present else ""
        return f"- {sk.name}{verifier_tag}: {summary}"


_DECLARED_FIELDS_RE = re.compile(r"\b([A-Z][A-Za-z]?\d+)\b")


def _extract_declared_fields(text: str) -> set[str]:
    """Heuristic: pull tokens like `Q1`, `Q23`, `A1` from the question text."""

    if not text:
        return set()
    return set(_DECLARED_FIELDS_RE.findall(text))
