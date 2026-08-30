"""Internal, constrained DuckDB executor for deep-insight audit checks."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from pathlib import Path
import re
from typing import Any

from ._deep_insight_audit import AuditCheck


class DuckDBAuditError(RuntimeError):
    """A DuckDB audit check or source binding is invalid."""


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FORBIDDEN_WORDS = frozenset(
    {
        "ALTER",
        "ATTACH",
        "CALL",
        "COPY",
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "EXPORT",
        "IMPORT",
        "INSERT",
        "INSTALL",
        "LOAD",
        "PRAGMA",
        "SET",
        "UPDATE",
    }
)
_FORBIDDEN_READERS = frozenset(
    {
        "csv_scan",
        "delta_scan",
        "glob",
        "httpfs",
        "iceberg_scan",
        "parquet_scan",
        "read_blob",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_parquet",
        "sqlite_scan",
    }
)


class DuckDBAuditExecutor:
    """Execute untrusted singleton SELECT checks over trusted CSV aliases."""

    def __init__(self, sources: Mapping[str, str | Path]) -> None:
        if not isinstance(sources, Mapping):
            raise DuckDBAuditError("DuckDB audit sources must be a mapping")

        bound_sources: dict[str, Path] = {}
        normalized_aliases: set[str] = set()
        for source_name, raw_path in sources.items():
            if (
                not isinstance(source_name, str)
                or _IDENTIFIER.fullmatch(source_name) is None
            ):
                raise DuckDBAuditError(
                    f"invalid source alias {source_name!r}; "
                    "use a simple SQL identifier"
                )
            normalized_alias = source_name.casefold()
            if normalized_alias in normalized_aliases:
                raise DuckDBAuditError(
                    f"duplicate source alias {source_name!r}; "
                    "aliases are case-insensitive"
                )
            normalized_aliases.add(normalized_alias)
            try:
                path = Path(raw_path)
            except TypeError as exc:
                raise DuckDBAuditError(
                    f"source file for alias {source_name!r} must be a filesystem path"
                ) from exc
            if not path.is_file():
                raise DuckDBAuditError(
                    f"source file for alias {source_name!r} does not exist: {path}"
                )
            bound_sources[source_name] = path

        try:
            duckdb = importlib.import_module("duckdb")
        except ModuleNotFoundError as exc:
            if exc.name != "duckdb":
                raise
            raise DuckDBAuditError(
                "optional dependency 'duckdb' is unavailable; install "
                "'fabric-rlm[analytics]' or install 'duckdb>=1.1'"
            ) from exc

        if not hasattr(duckdb, "extract_statements"):
            raise DuckDBAuditError(
                "installed DuckDB lacks required SQL parser support; "
                "install 'duckdb>=1.1'"
            )
        self._duckdb = duckdb
        self._sources = bound_sources
        self._connection = duckdb.connect(database=":memory:")
        self._source_tables: dict[str, str] = {}
        try:
            for index, (source_name, source_path) in enumerate(
                bound_sources.items()
            ):
                table_name = f"_audit_source_{index}"
                self._connection.execute(
                    f'CREATE TABLE "{table_name}" AS SELECT * FROM read_csv(?)',
                    [str(source_path)],
                )
                self._source_tables[source_name] = table_name
            _disable_external_access(self._connection)
        except Exception:
            self._connection.close()
            raise

    def __call__(self, check: AuditCheck) -> list[list[Any]]:
        verification = check.verification
        path = check.path
        if verification.get("method") != "sql":
            raise DuckDBAuditError(f"{path}: verification method must be 'sql'")

        expression = verification.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise DuckDBAuditError(
                f"{path}: verification expression must be a non-empty string"
            )

        declared = self._validate_source_declaration(
            verification.get("sources"), path
        )
        self._validate_sql(expression, path)
        return self._execute(expression, declared, path)

    def _validate_source_declaration(
        self, raw_sources: object, path: str
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(raw_sources, Mapping):
            raise DuckDBAuditError(f"{path}: verification sources must be a mapping")

        declared: list[tuple[str, str]] = []
        normalized_aliases: set[str] = set()
        for alias, source_name in raw_sources.items():
            if not isinstance(alias, str) or _IDENTIFIER.fullmatch(alias) is None:
                raise DuckDBAuditError(
                    f"{path}: source declaration has invalid query alias {alias!r}"
                )
            normalized_alias = alias.casefold()
            if normalized_alias in normalized_aliases:
                raise DuckDBAuditError(
                    f"{path}: duplicate query source alias {alias!r}"
                )
            normalized_aliases.add(normalized_alias)
            if not isinstance(source_name, str) or source_name not in self._sources:
                raise DuckDBAuditError(f"{path}: unknown source {source_name!r}")
            declared.append((alias, source_name))
        return tuple(declared)

    def _validate_sql(self, expression: str, path: str) -> None:
        masked = _mask_comments_and_quoted_values(expression)
        if ";" in masked:
            raise DuckDBAuditError(f"{path}: only a single SQL statement is allowed")

        tokens = {token.upper() for token in re.findall(r"\b[A-Za-z_]\w*\b", masked)}
        forbidden = tokens & _FORBIDDEN_WORDS
        if forbidden:
            operation = sorted(forbidden)[0]
            raise DuckDBAuditError(
                f"{path}: unsafe or non-read-only SQL operation {operation!r}"
            )

        called_functions = {
            match.group(1).lower()
            for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", masked)
        }
        readers = {
            name
            for name in called_functions
            if name in _FORBIDDEN_READERS
            or name.startswith("read_")
            or name.endswith("_scan")
        }
        if readers:
            reader = sorted(readers)[0]
            raise DuckDBAuditError(
                f"{path}: unsafe external reader {reader!r} is not allowed"
            )
        if any(
            token.lower().startswith(("duckdb_", "_audit_"))
            or token.lower() in {"information_schema", "sqlite_master"}
            for token in tokens
        ):
            raise DuckDBAuditError(
                f"{path}: unsafe access to audit internals is not allowed"
            )

        try:
            statements = self._duckdb.extract_statements(expression)
        except Exception as exc:
            raise DuckDBAuditError(f"{path}: invalid SQL expression: {exc}") from exc
        if len(statements) != 1:
            raise DuckDBAuditError(f"{path}: exactly one SQL statement is required")

        select_type = getattr(self._duckdb.StatementType, "SELECT", None)
        if statements[0].type != select_type:
            raise DuckDBAuditError(
                f"{path}: verification SQL must be one read-only SELECT or WITH query"
            )

    def _execute(
        self,
        expression: str,
        aliases: tuple[tuple[str, str], ...],
        path: str,
    ) -> list[list[Any]]:
        created_aliases: list[str] = []
        try:
            for alias, source_name in aliases:
                quoted_alias = f'"{alias}"'
                source_table = self._source_tables[source_name]
                self._connection.execute(
                    f"CREATE TEMP VIEW {quoted_alias} AS "
                    f'SELECT * FROM "{source_table}"'
                )
                created_aliases.append(alias)

            try:
                self._connection.execute(f"EXPLAIN {expression}")
                cursor = self._connection.execute(expression)
                if cursor.description is None or len(cursor.description) != 1:
                    raise DuckDBAuditError(
                        f"{path}: query must return exactly one row and one column"
                    )
                rows = cursor.fetchmany(2)
            except DuckDBAuditError:
                raise
            except Exception as exc:
                raise DuckDBAuditError(f"{path}: query failed: {exc}") from exc

            if len(rows) != 1 or len(rows[0]) != 1:
                raise DuckDBAuditError(
                    f"{path}: query must return exactly one row and one column"
                )
            return [[rows[0][0]]]
        finally:
            for alias in reversed(created_aliases):
                self._connection.execute(f'DROP VIEW IF EXISTS "{alias}"')

    def close(self) -> None:
        """Release the in-memory source snapshot."""

        self._connection.close()

    def __enter__(self) -> DuckDBAuditExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _disable_external_access(connection: Any) -> None:
    settings = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM duckdb_settings() WHERE name IN "
            "('enable_external_access', 'autoinstall_known_extensions', "
            "'autoload_known_extensions', 'lock_configuration')"
        ).fetchall()
    }
    if "autoinstall_known_extensions" in settings:
        connection.execute("SET autoinstall_known_extensions = false")
    if "autoload_known_extensions" in settings:
        connection.execute("SET autoload_known_extensions = false")
    if "enable_external_access" in settings:
        connection.execute("SET enable_external_access = false")
    if "lock_configuration" in settings:
        connection.execute("SET lock_configuration = true")


def _mask_comments_and_quoted_values(sql: str) -> str:
    output = list(sql)
    index = 0
    block_depth = 0
    quote: str | None = None
    while index < len(sql):
        if block_depth:
            output[index] = " "
            if sql.startswith("/*", index):
                output[index : index + 2] = [" ", " "]
                block_depth += 1
                index += 2
            elif sql.startswith("*/", index):
                output[index : index + 2] = [" ", " "]
                block_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if quote:
            output[index] = " "
            if sql[index] == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    output[index + 1] = " "
                    index += 2
                else:
                    quote = None
                    index += 1
            else:
                index += 1
            continue
        if sql.startswith("--", index):
            end = sql.find("\n", index)
            end = len(sql) if end < 0 else end
            output[index:end] = [" "] * (end - index)
            index = end
        elif sql.startswith("/*", index):
            output[index : index + 2] = [" ", " "]
            block_depth = 1
            index += 2
        elif sql[index] in {"'", '"'}:
            quote = sql[index]
            output[index] = " "
            index += 1
        else:
            index += 1
    return "".join(output)
