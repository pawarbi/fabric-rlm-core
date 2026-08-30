"""Strict loader for local benchmark source manifests."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType


_IDENTITY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_ENTRY_FIELDS = {
    "file_path",
    "row_count",
    "selection_rationale",
    "sha256",
    "size_bytes",
    "source_db",
    "source_table",
    "table",
}


@dataclass(frozen=True)
class SourceManifest:
    """Verified source inventory derived from a local manifest."""

    path: Path
    sources: Mapping[str, Path]
    selection_rationale: str
    normalization_note: str

    @property
    def table_count(self) -> int:
        return len(self.sources)


def build_source_agnostic_research_prompt(manifest: SourceManifest) -> str:
    """Build a privacy-bounded research brief from verified source metadata."""

    source_lines = "\n".join(
        f"- {identity}: {path}"
        for identity, path in manifest.sources.items()
    )
    return f"""\
Research this unseen local business dataset using a source-agnostic process.

DATASET RATIONALE
{manifest.selection_rationale}

NORMALIZATION NOTE
{manifest.normalization_note}

AUTHORITATIVE SOURCE IDENTITIES
{source_lines}

Return a compact JSON research ledger, not the final deep-insight contract.
Quality and depth win over candidate count.

RESEARCH REQUIREMENTS
- Measure every source schema and grain before selecting analytical paths.
- Build a measured join map with coverage, matched and unmatched counts, and
  explicit fan-out controls. Pre-aggregate one-to-many sources before joins.
- Assess method applicability for decomposition, instrumentation diagnostics,
  change points, cohorts, interactions, drivers, concentration, clustering,
  classification, and regression. State why each method is or is not applicable.
- Develop 6-10 decision-relevant candidate findings, including cross-domain
  candidates only where measured coverage permits.
- Include quantitative rejected candidates, diagnostic alternatives,
  metric-definition sensitivities, and explicit benchmark or target bases.

PRIVACY AND EVIDENCE RULES
- Evidence must be self-contained DuckDB SQL over the source aliases above.
- Include aggregate evidence only: no raw records.
- Never emit personal identifiers, contact details, free-text messages,
  transcript bodies, article bodies, descriptions, subjects, or quotations.
- Do not select or group by free-text or personal-identifier columns. Analyze
  communications only through non-content metadata and aggregate measures.
- Every alias must map to exactly one listed source identity; never depend on
  worker-created tables or views.

The research_json object must contain non-empty analysis_plan, join_map,
method_applicability, and candidates sections.
"""


def _required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _resolve_source_path(root: Path, value: object, path: str) -> Path:
    raw_path = Path(_required_text(value, path))
    resolved = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path} is outside manifest directory {root}") from exc
    if resolved.suffix.casefold() != ".csv":
        raise ValueError(f"{path} must reference a CSV file")
    if not resolved.is_file():
        raise FileNotFoundError(f"{path} does not exist: {resolved}")
    return resolved


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file has no header: {path}") from exc
        if not header or any(not column.strip() for column in header):
            raise ValueError(f"CSV file has an invalid header: {path}")
        return sum(1 for _ in reader)


def load_source_manifest(manifest_path: str | Path) -> SourceManifest:
    """Load and fully verify a manifest before exposing its CSV sources."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"source manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("source manifest must contain a JSON object")

    selection_rationale = _required_text(
        payload.get("selection_rationale"),
        "selection_rationale",
    )
    normalization_note = _required_text(
        payload.get("normalization_note"),
        "normalization_note",
    )
    entries = payload.get("tables")
    if not isinstance(entries, list) or not entries:
        raise ValueError("tables must be a non-empty list")

    root = path.parent.resolve()
    sources: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        entry_path = f"tables[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{entry_path} must be an object")
        missing = sorted(_REQUIRED_ENTRY_FIELDS - entry.keys())
        if missing:
            raise ValueError(
                f"{entry_path} is missing required fields: {', '.join(missing)}"
            )

        identity = _required_text(entry["table"], f"{entry_path}.table")
        if _IDENTITY.fullmatch(identity) is None:
            raise ValueError(
                f"{entry_path}.table must be a simple SQL identifier"
            )
        if identity in sources:
            raise ValueError(f"duplicate table identity: {identity}")

        source_path = _resolve_source_path(
            root,
            entry["file_path"],
            f"{entry_path}.file_path",
        )
        if source_path in seen_paths:
            raise ValueError(f"duplicate file path: {source_path}")

        expected_size = _required_nonnegative_int(
            entry["size_bytes"],
            f"{entry_path}.size_bytes",
        )
        actual_size = source_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {identity}: "
                f"manifest={expected_size}, actual={actual_size}"
            )

        expected_digest = _required_text(
            entry["sha256"],
            f"{entry_path}.sha256",
        ).casefold()
        if _SHA256.fullmatch(expected_digest) is None:
            raise ValueError(f"{entry_path}.sha256 must be a SHA-256 digest")
        actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {identity}")

        expected_rows = _required_nonnegative_int(
            entry["row_count"],
            f"{entry_path}.row_count",
        )
        actual_rows = _csv_row_count(source_path)
        if actual_rows != expected_rows:
            raise ValueError(
                f"row-count mismatch for {identity}: "
                f"manifest={expected_rows}, actual={actual_rows}"
            )

        sources[identity] = source_path
        seen_paths.add(source_path)

    return SourceManifest(
        path=path,
        sources=MappingProxyType(dict(sorted(sources.items()))),
        selection_rationale=selection_rationale,
        normalization_note=normalization_note,
    )
