from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from fabric_rlm._benchmark_manifest import load_source_manifest


def _write_csv(path: Path, rows: list[list[str]]) -> tuple[int, str]:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _write_manifest(
    root: Path,
    entries: list[dict[str, object]],
) -> Path:
    path = root / "export_manifest.json"
    path.write_text(
        json.dumps(
            {
                "exported_at_utc": "2026-04-10T05:37:27Z",
                "selection_rationale": "Transfer benchmark",
                "normalization_note": "Normalized source bundle",
                "tables": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(root: Path, identity: str = "accounts") -> dict[str, object]:
    csv_path = root / f"{identity}.csv"
    size, digest = _write_csv(csv_path, [["id", "name"], ["1", "Example"]])
    return {
        "table": identity,
        "source_db": "source.db",
        "source_table": identity.title(),
        "file_path": str(csv_path),
        "row_count": 1,
        "size_bytes": size,
        "sha256": digest,
        "selection_rationale": "Required source",
    }


def test_load_source_manifest_returns_stable_verified_source_inventory(
    tmp_path: Path,
) -> None:
    first = _entry(tmp_path, "accounts")
    second = _entry(tmp_path, "opportunities")
    manifest_path = _write_manifest(tmp_path, [second, first])

    manifest = load_source_manifest(manifest_path)

    assert tuple(manifest.sources) == ("accounts", "opportunities")
    assert manifest.sources["accounts"] == tmp_path / "accounts.csv"
    assert manifest.table_count == 2
    assert manifest.selection_rationale == "Transfer benchmark"
    assert manifest.normalization_note == "Normalized source bundle"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda entries, root: entries.append(dict(entries[0])), "duplicate table"),
        (
            lambda entries, root: entries[0].update(table="not-valid"),
            "simple SQL identifier",
        ),
        (
            lambda entries, root: entries[0].update(file_path=str(root / "missing.csv")),
            "does not exist",
        ),
        (
            lambda entries, root: entries[0].update(file_path=str(root.parent / "outside.csv")),
            "outside manifest directory",
        ),
        (
            lambda entries, root: entries[0].update(size_bytes=1),
            "size mismatch",
        ),
        (
            lambda entries, root: entries[0].update(sha256="0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda entries, root: entries[0].update(row_count=2),
            "row-count mismatch",
        ),
    ],
)
def test_load_source_manifest_rejects_untrusted_or_inconsistent_entries(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    entries = [_entry(tmp_path)]
    mutation(entries, tmp_path)
    manifest_path = _write_manifest(tmp_path, entries)

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        load_source_manifest(manifest_path)


def test_load_source_manifest_rejects_duplicate_file_paths(tmp_path: Path) -> None:
    first = _entry(tmp_path, "accounts")
    second = dict(first, table="accounts_copy")
    manifest_path = _write_manifest(tmp_path, [first, second])

    with pytest.raises(ValueError, match="duplicate file path"):
        load_source_manifest(manifest_path)


def test_load_source_manifest_handles_quoted_multiline_csv_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "messages.csv"
    size, digest = _write_csv(
        csv_path,
        [["id", "body"], ["1", "first line\nsecond line"]],
    )
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "table": "messages",
                "source_db": "source.db",
                "source_table": "Message",
                "file_path": str(csv_path),
                "row_count": 1,
                "size_bytes": size,
                "sha256": digest,
                "selection_rationale": "Communication metadata",
            }
        ],
    )

    manifest = load_source_manifest(manifest_path)

    assert manifest.table_count == 1
