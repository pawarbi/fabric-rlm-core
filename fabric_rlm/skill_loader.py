"""Load bundled SKILL playbooks for RLM prompts and worker code."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # PyYAML is part of the project's transitive deps via dspy.
    import yaml as _yaml
except Exception:  # pragma: no cover - graceful degrade if pyyaml is missing
    _yaml = None  # type: ignore[assignment]


_logger = logging.getLogger(__name__)

_SAFE_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_VERIFIER_BLOCK = re.compile(
    r"^## Required verifier\s*\n(?P<body>.*?)^```python\n(?P<code>.*?)\n```",
    re.DOTALL | re.MULTILINE,
)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
_VALID_SPECIFICITIES = ("core", "domain", "utility")


@dataclass(frozen=True)
class Skill:
    name: str
    title: str
    summary: str
    content: str
    dependencies: tuple[str, ...] = ()
    verifier_source: Optional[str] = None
    # Track 5″ frontmatter fields. All optional / back-compat with skills
    # that have no frontmatter block.
    applies_when_keywords: tuple[str, ...] = ()
    applies_when_output_fields: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    specificity: str = "domain"
    verifier_present: bool = False


class SkillLoader:
    """Discover and load Markdown SKILL playbooks by safe name."""

    def __init__(self, skill_dir: str | Path | None = None, *, package: str = "fabric_rlm.skills"):
        self.skill_dir = Path(skill_dir) if skill_dir is not None else None
        self.package = package

    def list_skills(self) -> list[str]:
        if self.skill_dir is not None:
            if not self.skill_dir.exists():
                return []
            names = [path.stem for path in self.skill_dir.glob("*.md") if path.is_file()]
        else:
            names = [
                entry.name.removesuffix(".md")
                for entry in resources.files(self.package).iterdir()
                if entry.is_file() and entry.name.endswith(".md")
            ]
        return sorted(name for name in names if _is_safe_skill_name(name))

    def load(self, name: str) -> Skill:
        safe_name = normalize_skill_name(name)
        content = self._read_skill_text(safe_name)
        if safe_name == "core":
            mode = _resolve_pvr_mode()
            if mode == "off":
                content = _strip_pvr_clauses(content)
            elif mode == "reflect_only":
                content = _strip_plan_verify_clauses(content)
        frontmatter, body_after_fm = _split_frontmatter(content)
        title, summary, legacy_deps = _parse_metadata(safe_name, body_after_fm)
        verifier_source = extract_verifier_source(content)
        fm_deps = _normalize_str_tuple(frontmatter.get("depends_on"))
        dependencies = fm_deps if fm_deps else legacy_deps
        applies_when = frontmatter.get("applies_when") or {}
        if not isinstance(applies_when, dict):
            applies_when = {}
        keywords = _normalize_str_tuple(applies_when.get("keywords"))
        output_fields = _normalize_str_tuple(applies_when.get("output_fields"))
        excludes = _normalize_str_tuple(frontmatter.get("excludes"))
        specificity = frontmatter.get("specificity") or "domain"
        if specificity not in _VALID_SPECIFICITIES:
            _logger.warning(
                "Skill %r has unknown specificity %r; defaulting to 'domain'.",
                safe_name,
                specificity,
            )
            specificity = "domain"
        return Skill(
            name=safe_name,
            title=title,
            summary=summary,
            content=content,
            dependencies=dependencies,
            verifier_source=verifier_source,
            applies_when_keywords=keywords,
            applies_when_output_fields=output_fields,
            excludes=excludes,
            specificity=specificity,
            verifier_present=verifier_source is not None,
        )

    def load_text(self, name: str, *, include_dependencies: bool = False) -> str:
        if not include_dependencies:
            return self.load(name).content
        return compose_skills([name], loader=self, include_dependencies=True)

    def format_index(self) -> str:
        lines: list[str] = []
        for name in self.list_skills():
            skill = self.load(name)
            suffix = f" — {skill.summary}" if skill.summary else ""
            deps = f" (depends on: {', '.join(skill.dependencies)})" if skill.dependencies else ""
            lines.append(f"- {skill.name}: {skill.title}{suffix}{deps}")
        return "\n".join(lines)

    def _read_skill_text(self, name: str) -> str:
        if self.skill_dir is not None:
            path = self.skill_dir / f"{name}.md"
            if not path.is_file():
                raise FileNotFoundError(f"Unknown SKILL: {name}")
            return path.read_text(encoding="utf-8")

        target = resources.files(self.package).joinpath(f"{name}.md")
        if not target.is_file():
            raise FileNotFoundError(f"Unknown SKILL: {name}")
        return target.read_text(encoding="utf-8")


def list_skills() -> list[str]:
    """Return bundled SKILL names available to RLM worker code."""

    return SkillLoader().list_skills()


def load_skill(name: str, *, include_dependencies: bool = False) -> str:
    """Return a bundled SKILL Markdown playbook by safe name."""

    return SkillLoader().load_text(name, include_dependencies=include_dependencies)


def compose_skills(
    names: Iterable[str],
    *,
    loader: SkillLoader | None = None,
    include_dependencies: bool = True,
) -> str:
    """Compose selected SKILL playbooks into a prompt-ready Markdown block."""

    skill_loader = loader or SkillLoader()
    ordered: list[Skill] = []
    seen: set[str] = set()
    loading: set[str] = set()

    def add_skill(name: str) -> None:
        skill = skill_loader.load(name)
        if skill.name in seen:
            return
        if skill.name in loading:
            raise ValueError(f"Cyclic SKILL dependency involving {skill.name!r}")
        loading.add(skill.name)
        if include_dependencies:
            for dependency in skill.dependencies:
                add_skill(dependency)
        loading.remove(skill.name)
        seen.add(skill.name)
        ordered.append(skill)

    for name in names:
        add_skill(name)

    sections: list[str] = []
    for skill in ordered:
        sections.extend([f"## Skill: {skill.name}", "", skill.content.strip(), ""])
    return "\n".join(sections).strip()


def normalize_skill_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("SKILL name must be a string")
    normalized = name.removesuffix(".md")
    if not _is_safe_skill_name(normalized):
        raise ValueError(f"Unsafe SKILL name: {name!r}")
    return normalized


def _is_safe_skill_name(name: str) -> bool:
    return bool(_SAFE_SKILL_NAME.fullmatch(name)) and ".." not in name


_PVR_CLAUSE_RE = re.compile(
    r"^\d+\.\s+\*\*(?:PLAN before action|VERIFY before SUBMIT|"
    r"Honor PRIOR_ATTEMPT_FEEDBACK[^*]*?)\.?\*\*.*?(?=^\d+\.\s+\*\*|^## |\Z)",
    re.DOTALL | re.MULTILINE,
)


_PLAN_VERIFY_CLAUSE_RE = re.compile(
    r"^\d+\.\s+\*\*(?:PLAN before action|VERIFY before SUBMIT)\.?\*\*"
    r".*?(?=^\d+\.\s+\*\*|^## |\Z)",
    re.DOTALL | re.MULTILINE,
)


_VALID_PVR_MODES = ("full", "off", "reflect_only")


def _resolve_pvr_mode() -> str:
    """Resolve effective PVR mode from env vars.

    Reads ``FABRIC_RLM_PVR_MODE`` first (full|off|reflect_only). Falls back
    to legacy ``FABRIC_RLM_PVR`` (0=off, 1=full) for backwards compatibility.
    Defaults to ``full``.
    """
    mode = os.environ.get("FABRIC_RLM_PVR_MODE", "").strip().lower()
    if mode in _VALID_PVR_MODES:
        return mode
    legacy = os.environ.get("FABRIC_RLM_PVR", "1").strip()
    return "off" if legacy == "0" else "full"


def _strip_pvr_clauses(content: str) -> str:
    """Remove the PLAN/VERIFY/REFLECT clauses from core.md when PVR is disabled.

    Used by the FABRIC_RLM_PVR=0 ablation switch — the rest of the skill
    body (single-SUBMIT, no-echo-prompt, advice-not-output) stays in place.
    """
    return _PVR_CLAUSE_RE.sub("", content)


def _strip_plan_verify_clauses(content: str) -> str:
    """Remove only the PLAN/VERIFY clauses, keeping ``Honor PRIOR_ATTEMPT_FEEDBACK``.

    Used by the ``reflect_only`` ablation mode to test whether REFLECT
    alone delivers the PVR wins without the PLAN/VERIFY scaffolding the
    model ignores at minimal reasoning effort.
    """
    return _PLAN_VERIFY_CLAUSE_RE.sub("", content)


def extract_verifier_source(content: str) -> Optional[str]:
    """Return the body of the first ``## Required verifier`` python block, or None.

    The fenced ``\u0060\u0060\u0060python`` block is taken to be the *first* one
    that follows the ``## Required verifier`` heading, but only if no other
    top-level (``## ``) heading appears between them — otherwise the verifier
    section had no code block and we return None.
    """

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    match = _VERIFIER_BLOCK.search(normalized)
    if not match:
        return None
    body = match.group("body") or ""
    if re.search(r"^## ", body, re.MULTILINE):
        return None
    source = match.group("code")
    return source if source.strip() else None


def _split_frontmatter(content: str) -> tuple[dict, str]:
    """Return ``(frontmatter_dict, body_after_frontmatter)``.

    YAML frontmatter is delimited by ``---`` lines at the top of the file.
    Returns ``({}, content)`` when no frontmatter is present, when YAML
    parsing fails, or when PyYAML is unavailable.
    """

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    match = _FRONTMATTER_RE.match(normalized)
    if not match:
        return {}, content
    if _yaml is None:
        return {}, normalized[match.end() :]
    body = match.group("body")
    try:
        data = _yaml.safe_load(body)
    except Exception as exc:  # noqa: BLE001 - tolerate any YAML error gracefully
        _logger.warning("Malformed YAML frontmatter (%s); ignoring.", exc)
        return {}, normalized[match.end() :]
    if not isinstance(data, dict):
        return {}, normalized[match.end() :]
    return data, normalized[match.end() :]


def _normalize_str_tuple(value: Any) -> tuple[str, ...]:  # type: ignore[name-defined]
    if value is None:
        return ()
    if isinstance(value, str):
        # Allow comma-separated single string for ergonomics.
        parts = [p.strip() for p in value.split(",")]
        return tuple(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    return ()


def _parse_metadata(name: str, content: str) -> tuple[str, str, tuple[str, ...]]:
    title = name
    summary = ""
    dependencies: tuple[str, ...] = ()
    for raw_line in content.splitlines()[:20]:
        line = raw_line.strip()
        if line.startswith("# ") and title == name:
            title = line[2:].strip() or name
        elif line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("dependencies:"):
            value = line.split(":", 1)[1].strip()
            dependencies = tuple(
                normalize_skill_name(part.strip())
                for part in value.split(",")
                if part.strip() and part.strip().lower() not in {"none", "n/a"}
            )
    return title, summary, dependencies
