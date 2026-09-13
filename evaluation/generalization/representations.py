"""Equivalent local source representations; no reference-answer calculations."""

from __future__ import annotations

import json
from pathlib import Path

from .fixtures import VARIANTS


DOMAINS = ("inventory", "manufacturing", "service")
REPRESENTATIONS = ("csv", "parquet", "lakehouse")


def _csv_paths(fixtures: Path, domain: str, variant: str) -> list[Path]:
    if domain not in DOMAINS or variant not in VARIANTS:
        raise ValueError("unknown fixture domain or naming variant")
    paths = [
        path for path in sorted((fixtures / domain / variant).glob("*.csv"))
        if not path.name.endswith("_large.csv")
    ]
    if not paths:
        raise FileNotFoundError(f"no seeded sources for {domain}/{variant}")
    return paths


def prepare_representations(fixtures: Path) -> None:
    import duckdb
    import pyarrow.parquet as parquet
    from deltalake import DeltaTable, write_deltalake

    paths = [
        path for domain in DOMAINS for variant in VARIANTS
        for path in _csv_paths(fixtures, domain, variant)
    ]
    for path in paths:
        for suffix in (".parquet", ".delta", ".source.json"):
            target = path.with_suffix(suffix)
            if target.exists():
                raise FileExistsError(f"refusing to replace {target}")
    with duckdb.connect() as connection:
        for path in paths:
            cursor = connection.execute("SELECT * FROM read_csv_auto(?)", [str(path)])
            columns = [[name, str(dtype)] for name, dtype, *_rest in cursor.description]
            table = cursor.to_arrow_table()
            with path.with_suffix(".parquet").open("xb") as output:
                parquet.write_table(table, output)
            delta_path = path.with_suffix(".delta")
            write_deltalake(str(delta_path), table)
            delta = DeltaTable(str(delta_path), without_files=True)
            catalog = {
                "kind": "delta", "name": path.stem,
                "version": delta.version(), "table_id": delta.metadata().id,
                "columns": columns,
            }
            with path.with_suffix(".source.json").open("x", encoding="utf-8") as output:
                json.dump(catalog, output, indent=2, sort_keys=True)


def domain_sources(
    fixtures: Path, domain: str, variant: str, representation: str = "csv",
) -> dict[str, object]:
    from fabric_rlm import File

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported representation: {representation}")
    sources: dict[str, object] = {}
    for path in _csv_paths(fixtures, domain, variant):
        if representation == "csv":
            sources[path.stem] = File(path)
        elif representation == "parquet":
            parquet_path = path.with_suffix(".parquet")
            if not parquet_path.is_file():
                raise FileNotFoundError(parquet_path)
            sources[path.stem] = File(parquet_path)
        else:
            from fabric_rlm import LakehouseSource

            delta_path = path.with_suffix(".delta")
            catalog = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
            sources[path.stem] = LakehouseSource(
                delta_path.resolve().as_uri(),
                catalog=[{**catalog, "path": str(delta_path)}],
            )
    return sources
