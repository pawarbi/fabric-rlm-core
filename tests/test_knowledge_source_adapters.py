from __future__ import annotations

import builtins
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fabric_rlm.artifacts import File
from fabric_rlm.knowledge import KnowledgePackage, canonical_json
from fabric_rlm.knowledge_store import save_knowledge_package


def _module():
    import fabric_rlm.knowledge_sources as knowledge_sources

    return knowledge_sources


def _profile(path: object, **kwargs: object):
    return _module().profile_sources({"sales": path}, **kwargs)[0]


@pytest.mark.parametrize("field", [
    "max_input_bytes",
    "max_records",
    "max_fields",
    "max_nesting_depth",
    "max_diagnostic_bytes",
    "read_chunk_bytes",
])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_profile_limits_require_positive_integers(field: str, value: object) -> None:
    ProfileLimits = _module().ProfileLimits
    values = {
        "max_input_bytes": 1,
        "max_records": 1,
        "max_fields": 1,
        "max_nesting_depth": 1,
        "max_diagnostic_bytes": 1,
        "read_chunk_bytes": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ProfileLimits(**values)


def test_profile_limits_are_immutable_and_have_conservative_defaults() -> None:
    limits = _module().ProfileLimits()

    assert limits.max_input_bytes == 1024 * 1024
    assert limits.max_records == 1000
    assert limits.max_fields == 256
    assert limits.max_nesting_depth == 8
    assert limits.max_diagnostic_bytes == 64 * 1024
    assert limits.read_chunk_bytes == 64 * 1024
    with pytest.raises(Exception):
        limits.max_records = 2


def test_str_path_and_file_inputs_produce_equivalent_profiles(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id,total\n1,2.5\n", encoding="utf-8")

    profiles = [
        _profile(str(path)),
        _profile(path),
        _profile(File(path)),
    ]

    assert profiles[0] == profiles[1] == profiles[2]
    assert profiles[0].locator == "local/sales"


def test_suffix_routing_is_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "ORDERS.CsV"
    path.write_text("id\n1\n", encoding="utf-8")

    assert _profile(path).family == "csv"


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("bad.csv", 'id,name\n1,"unterminated\n'),
        ("bad.json", '{"id":'),
        ("bad.jsonl", '{"id": 1}\nnot-json\n'),
    ],
)
def test_recognized_parser_errors_never_downgrade_to_opaque(
    tmp_path: Path, name: str, contents: str
) -> None:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=name.rsplit(".", 1)[1].upper()):
        _profile(path)


def test_json_rejects_nonstandard_numeric_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"amount": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        _profile(path)


def test_missing_path_and_directory_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sales"):
        _profile(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="regular file"):
        _profile(tmp_path)


def test_non_regular_file_fails_clearly(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform")
    path = tmp_path / "pipe"
    os.mkfifo(path)

    with pytest.raises(ValueError, match="regular file"):
        _profile(path)


def test_csv_profile_is_deterministic_structural_and_value_free(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text(
        "id,total,active,note\n1,2.5,true,alpha\n2,,false,beta\n",
        encoding="utf-8",
    )

    first = _profile(path)
    second = _profile(path)
    encoded = canonical_json(first.to_dict())

    assert first == second
    assert first.schema == {
        "active": {"nullable": False, "type": "boolean"},
        "id": {"nullable": False, "type": "integer"},
        "note": {"nullable": False, "type": "string"},
        "total": {"nullable": True, "type": "number"},
    }
    for value in ("alpha", "beta", "2.5"):
        assert value not in encoded


@pytest.mark.parametrize(
    "header",
    [
        "id,id\n1,2\n",
        "safe,unsafe header\n1,2\n",
        ",name\n1,a\n",
    ],
)
def test_csv_rejects_duplicate_or_unsafe_headers(
    tmp_path: Path, header: str
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(header, encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        _profile(path)


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "authorization",
        "access_token_status",
        "proxy_authorization",
        "token_count",
    ],
)
def test_csv_profiles_and_classifies_sensitive_column_names(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / "sensitive.csv"
    path.write_text(f"id,{name}\n1,private-sentinel\n", encoding="utf-8")

    profile = _profile(path)

    assert name in profile.schema
    assert name in profile.sensitive_columns
    assert "private-sentinel" not in canonical_json(profile.to_dict())


def test_csv_honors_record_field_and_byte_bounds(tmp_path: Path) -> None:
    path = tmp_path / "bounded.csv"
    path.write_text("id,name\n1,a\n2,b\n3,c\n", encoding="utf-8")
    limits = _module().ProfileLimits(
        max_input_bytes=18,
        max_records=1,
        max_fields=2,
        max_nesting_depth=2,
        max_diagnostic_bytes=4096,
        read_chunk_bytes=3,
    )

    profile = _profile(path, limits=limits)

    assert profile.diagnostics["records_inspected"] == 1
    assert profile.diagnostics["records_truncated"] is True
    assert profile.diagnostics["input_truncated"] is True
    assert profile.diagnostics["snapshot_exact"] is False


def test_small_file_value_and_schema_changes_affect_the_right_fingerprints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    original = _profile(path)
    path.write_text("id\n2\n", encoding="utf-8")
    value_change = _profile(path)
    path.write_text("order_id\n2\n", encoding="utf-8")
    schema_change = _profile(path)

    assert original.snapshot_fingerprint != value_change.snapshot_fingerprint
    assert original.schema_fingerprint == value_change.schema_fingerprint
    assert value_change.schema_fingerprint != schema_change.schema_fingerprint


def test_mtime_changes_do_not_affect_profile_identity(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    original = _profile(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert _profile(path) == original


def test_profiling_never_uses_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError("whole-file read_bytes must not be used")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    assert _profile(path).family == "csv"


def test_json_supports_object_and_array_without_retaining_values(
    tmp_path: Path,
) -> None:
    object_path = tmp_path / "one.json"
    object_path.write_text('{"id": 1, "name": "private-value"}', encoding="utf-8")
    array_path = tmp_path / "many.json"
    array_path.write_text(
        '[{"id": 1}, {"id": 2, "active": true}]', encoding="utf-8"
    )

    one = _profile(object_path)
    many = _profile(array_path)

    assert one.family == "json"
    assert one.diagnostics["records_inspected"] == 1
    assert many.schema["active"]["nullable"] is True
    assert "private-value" not in canonical_json(one.to_dict())


def test_json_rejects_non_object_records_and_excessive_nesting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"id": 1}, 2]', encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        _profile(path)

    path.write_text('{"a": {"b": {"c": 1}}}', encoding="utf-8")
    limits = _module().ProfileLimits(max_nesting_depth=2)
    with pytest.raises(ValueError, match="nesting"):
        _profile(path, limits=limits)


@pytest.mark.parametrize(
    "contents",
    [
        '{"id": 1, "id": 2}',
        '{"outer": {"password": "first", "password": "second"}}',
        '[{"nested": {"authorization": "first", "authorization": "second"}}]',
    ],
)
def test_json_rejects_duplicate_keys_at_every_object_depth(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        _profile(path)


def test_json_rejects_oversized_input_before_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized.json"
    path.write_text('{"id": 1}', encoding="utf-8")
    module = _module()

    def forbidden_decoder(text: str) -> object:
        raise AssertionError("oversized JSON must not reach the decoder")

    monkeypatch.setattr(module, "_load_json", forbidden_decoder)

    with pytest.raises(ValueError, match="max_input_bytes"):
        _profile(path, limits=module.ProfileLimits(max_input_bytes=4))


def test_json_profiles_sensitive_names_without_retaining_nested_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensitive.json"
    path.write_text(
        '{"password": "top-secret-sentinel", '
        '"access_token_status": {"nested": "nested-private-sentinel"}, '
        '"token_count": 3}',
        encoding="utf-8",
    )

    profile = _profile(path)
    encoded = canonical_json(profile.to_dict())

    assert profile.sensitive_columns == (
        "access_token_status",
        "password",
        "token_count",
    )
    assert "top-secret-sentinel" not in encoded
    assert "nested-private-sentinel" not in encoded


def test_jsonl_uses_bounded_deterministic_field_union_and_no_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"id": 1, "name": "first-private"}\n'
        '{"id": 2, "active": true, "name": "second-private"}\n',
        encoding="utf-8",
    )
    limits = _module().ProfileLimits(max_records=1)

    profile = _profile(path, limits=limits)
    encoded = canonical_json(profile.to_dict())

    assert set(profile.schema) == {"id", "name"}
    assert profile.diagnostics["records_truncated"] is True
    assert "private" not in encoded


def test_truncated_jsonl_without_a_complete_nonblank_record_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"password": "private-sentinel"}\n', encoding="utf-8")
    limits = _module().ProfileLimits(max_input_bytes=10, read_chunk_bytes=3)

    with pytest.raises(ValueError, match="complete nonblank"):
        _profile(path, limits=limits)


def test_truncated_jsonl_keeps_only_complete_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"id": 1}\n{"password": "private-sentinel"}\n',
        encoding="utf-8",
    )
    limits = _module().ProfileLimits(max_input_bytes=15, read_chunk_bytes=4)

    profile = _profile(path, limits=limits)

    assert set(profile.schema) == {"id"}
    assert profile.diagnostics["input_truncated"] is True
    assert "private-sentinel" not in canonical_json(profile.to_dict())


def test_jsonl_rejects_nested_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        '{"outer": {"password": "first", "password": "second"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="JSONL"):
        _profile(path)


def test_opaque_profile_does_not_decode_content(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"\xff\xfeprivate\x00bytes")

    profile = _profile(path)
    encoded = canonical_json(profile.to_dict())

    assert profile.family == "opaque"
    assert profile.role == "context_only"
    assert profile.schema == {}
    assert "private" not in encoded


@pytest.mark.parametrize("role", ["numeric_evidence", "lookup"])
def test_opaque_rejects_tabular_roles(tmp_path: Path, role: str) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="role"):
        _profile(path, roles={"sales": role})


def test_incompatible_opaque_role_is_rejected_before_missing_file_io(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.bin"

    with pytest.raises(ValueError, match="opaque.*role|role.*opaque"):
        _profile(missing, roles={"sales": "numeric_evidence"})


def test_unknown_role_alias_is_rejected_before_any_file_io(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(ValueError, match="unknown source alias"):
        _module().profile_sources(
            {"sales": missing}, roles={"unknown": "context_only"}
        )


def test_invalid_role_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="role"):
        _profile(path, roles={"sales": "arbitrary"})


def test_registry_is_ordered_explicit_and_customizable(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")

    class FirstAdapter(module._LocalFileAdapter):
        family = "custom"
        allowed_roles = frozenset({"context_only"})
        default_role = "context_only"

        def _matches_path(self, candidate: Path) -> bool:
            return True

        def _profile_file(self, source_id, candidate, role, limits, snapshot):
            return module.SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"local/{source_id}",
                snapshot_fingerprint=snapshot.fingerprint,
                schema_fingerprint="custom-schema",
                diagnostics={"format_code": "custom"},
                role=role,
            )

    registry = module.SourceAdapterRegistry((FirstAdapter(),))
    profile = _profile(path, registry=registry, roles={"sales": "context_only"})

    assert profile.family == "custom"


def test_custom_object_adapter_routes_and_profiles_deterministically() -> None:
    module = _module()

    @dataclass(frozen=True)
    class FakeHandle:
        item_id: str
        version: int

    class FakeAdapter:
        family = "fake"
        allowed_roles = frozenset({"context_only"})
        default_role = "context_only"

        def matches(self, value: object) -> bool:
            return isinstance(value, FakeHandle)

        def profile(self, source_id, value, role, limits):
            return module.SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"fake/{value.item_id}",
                snapshot_fingerprint=f"fake-snapshot-{value.version}",
                schema_fingerprint="fake-schema",
                diagnostics={"snapshot_exact": True},
                role=role,
            )

    handle = FakeHandle("sales-model", 7)
    registry = module.SourceAdapterRegistry((FakeAdapter(),))

    first = _profile(handle, registry=registry)
    second = _profile(handle, registry=registry)

    assert first == second
    assert first.locator == "fake/sales-model"
    assert first.snapshot_fingerprint == "fake-snapshot-7"


def test_custom_object_role_is_rejected_before_adapter_profile() -> None:
    module = _module()

    class FakeHandle:
        pass

    class FakeAdapter:
        family = "fake"
        allowed_roles = frozenset({"context_only"})
        default_role = "context_only"
        profile_called = False

        def matches(self, value: object) -> bool:
            return isinstance(value, FakeHandle)

        def profile(self, source_id, value, role, limits):
            self.profile_called = True
            raise AssertionError("profile must not be called")

    adapter = FakeAdapter()
    registry = module.SourceAdapterRegistry((adapter,))

    with pytest.raises(ValueError, match="role.*fake|fake.*role"):
        _profile(
            FakeHandle(),
            registry=registry,
            roles={"sales": "numeric_evidence"},
        )

    assert adapter.profile_called is False


def test_directory_values_reach_registry_matching(tmp_path: Path) -> None:
    module = _module()
    delta_directory = tmp_path / "sales.delta"
    delta_directory.mkdir()

    class FutureDeltaAdapter:
        family = "delta"
        allowed_roles = frozenset({"numeric_evidence"})
        default_role = "numeric_evidence"

        def matches(self, value: object) -> bool:
            return isinstance(value, Path) and value.suffix == ".delta"

        def profile(self, source_id, value, role, limits):
            return module.SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"delta/{source_id}",
                snapshot_fingerprint="delta-snapshot",
                schema_fingerprint="delta-schema",
                diagnostics={"snapshot_exact": True},
                role=role,
            )

    profile = _profile(
        delta_directory,
        registry=module.SourceAdapterRegistry((FutureDeltaAdapter(),)),
    )

    assert profile.family == "delta"


def test_source_replacement_during_adapter_profile_fails_closed(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "orders.csv"
    replacement = tmp_path / "replacement.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    replacement.write_text("id\n2\n", encoding="utf-8")

    class ReplacingAdapter(module._LocalFileAdapter):
        family = "custom"
        allowed_roles = frozenset({"numeric_evidence"})
        default_role = "numeric_evidence"

        def _matches_path(self, candidate: Path) -> bool:
            return candidate.suffix.lower() == ".csv"

        def _profile_file(self, source_id, candidate, role, limits, snapshot):
            candidate.unlink()
            replacement.replace(candidate)
            return module.SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"local/{source_id}",
                snapshot_fingerprint=snapshot.fingerprint,
                schema_fingerprint="custom-schema",
                role=role,
            )

    with pytest.raises(ValueError, match="sales.*changed.*profil"):
        _profile(path, registry=module.SourceAdapterRegistry((ReplacingAdapter(),)))


def test_same_size_in_place_mutation_with_restored_mtime_fails_closed(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "orders.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    original_stat = path.stat()

    class MutatingAdapter(module._LocalFileAdapter):
        family = "custom"
        allowed_roles = frozenset({"numeric_evidence"})
        default_role = "numeric_evidence"

        def _matches_path(self, candidate: Path) -> bool:
            return candidate.suffix.lower() == ".csv"

        def _profile_file(self, source_id, candidate, role, limits, snapshot):
            candidate.write_text("id\n2\n", encoding="utf-8")
            os.utime(
                candidate,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return module.SourceProfile(
                source_id=source_id,
                family=self.family,
                locator=f"local/{source_id}",
                snapshot_fingerprint=snapshot.fingerprint,
                schema_fingerprint="custom-schema",
                role=role,
            )

    registry = module.SourceAdapterRegistry((MutatingAdapter(),))

    with pytest.raises(ValueError, match="sales.*changed.*profil"):
        _profile(path, registry=registry)


def test_import_and_registry_construction_do_not_import_optional_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_roots = {
        "pandas",
        "polars",
        "duckdb",
        "deltalake",
        "fitz",
        "openpyxl",
        "sempy",
    }
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in optional_roots:
            raise AssertionError(f"optional import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("fabric_rlm.knowledge_sources", None)
    module = importlib.import_module("fabric_rlm.knowledge_sources")

    module.SourceAdapterRegistry.default()


def test_profiles_are_compatible_with_knowledge_persistence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("id,total\n1,2.5\n", encoding="utf-8")
    profile = _profile(path)

    destination = tmp_path / "knowledge.json"
    save_knowledge_package(
        destination,
        KnowledgePackage(package_id="local.package", sources=(profile,)),
    )

    assert destination.is_file()


def test_emitted_profile_respects_canonical_output_size_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(
        ",".join(f"field_{index:03d}" for index in range(40))
        + "\n"
        + ",".join("1" for _ in range(40))
        + "\n",
        encoding="utf-8",
    )
    limits = _module().ProfileLimits(max_diagnostic_bytes=500)

    with pytest.raises(ValueError, match="max_diagnostic_bytes"):
        _profile(path, limits=limits)


def test_profile_sources_is_not_exported_from_package_root() -> None:
    import fabric_rlm

    assert not hasattr(fabric_rlm, "profile_sources")


def test_parquet_adapter_uses_metadata_without_retaining_values(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "orders.parquet"
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            COPY (
                SELECT 1::INTEGER AS id, 'private-sentinel'::VARCHAR AS note
                UNION ALL
                SELECT 2::INTEGER, 'another-private-value'::VARCHAR
            ) TO ? (FORMAT PARQUET)
            """,
            [str(path)],
        )
    finally:
        connection.close()

    profile = _profile(path)
    encoded = canonical_json(profile.to_dict())

    assert profile.family == "parquet"
    assert profile.schema == {
        "id": {"logical_type": "integer", "nullable": True, "type": "integer"},
        "note": {"logical_type": "varchar", "nullable": True, "type": "string"},
    }
    assert profile.diagnostics["row_count"] == 2
    assert profile.diagnostics["row_group_count"] == 1
    assert "private-sentinel" not in encoded
    assert str(path.resolve()) not in encoded


def test_parquet_profiles_and_classifies_sensitive_names_without_values(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "sensitive.parquet"
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(
            """
            COPY (
                SELECT 'private-sentinel'::VARCHAR AS password,
                       'nested-private-sentinel'::VARCHAR AS authorization,
                       2::INTEGER AS token_count
            ) TO ? (FORMAT PARQUET)
            """,
            [str(path)],
        )
    finally:
        connection.close()

    profile = _profile(path)
    encoded = canonical_json(profile.to_dict())

    assert profile.sensitive_columns == (
        "authorization",
        "password",
        "token_count",
    )
    assert "private-sentinel" not in encoded


def test_parquet_rejects_invalid_magic_and_oversized_footer_before_duckdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = tmp_path / "invalid.parquet"
    invalid.write_bytes(b"not parquet")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "duckdb":
            raise AssertionError("DuckDB must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="magic|footer"):
        _profile(invalid)

    oversized = tmp_path / "oversized.parquet"
    oversized.write_bytes(b"PAR1" + b"x" * 20 + (20).to_bytes(4, "little") + b"PAR1")
    limits = _module().ProfileLimits(max_input_bytes=10)
    with pytest.raises(ValueError, match="footer metadata exceeds"):
        _profile(oversized, limits=limits)


def test_parquet_connection_closes_and_description_fetch_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.parquet"
    path.write_bytes(b"PAR1" + b"x" + (1).to_bytes(4, "little") + b"PAR1")
    module = _module()

    class FakeResult:
        def __init__(self, rows):
            self.rows = rows

        def fetchmany(self, size):
            assert size == 3
            return self.rows[:size]

        def fetchone(self):
            return self.rows[0] if self.rows else None

        def fetchall(self):
            raise AssertionError("fetchall must not be used")

    class FakeConnection:
        def __init__(self, fail: bool = False):
            self.closed = False
            self.fail = fail
            self.queries = []

        def execute(self, query, parameters=None):
            self.queries.append(query.strip())
            if self.fail and "DESCRIBE" in query:
                raise RuntimeError("malformed")
            if "DESCRIBE" in query:
                return FakeResult(
                    [
                        ("id", "INTEGER", "YES"),
                        ("password", "VARCHAR", "YES"),
                        ("excess", "VARCHAR", "YES"),
                    ]
                )
            if "parquet_file_metadata" in query:
                return FakeResult([(1, 1)])
            return FakeResult([])

        def close(self):
            self.closed = True

    successful = FakeConnection()
    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda database: successful),
    )
    limits = module.ProfileLimits(max_fields=2)
    with pytest.raises(ValueError, match="max_fields"):
        _profile(path, limits=limits)
    assert successful.closed is True
    assert successful.queries[:3] == [
        "SET memory_limit = '64MB'",
        "SET threads = 1",
        "DESCRIBE SELECT * FROM read_parquet(?)",
    ]

    failing = FakeConnection(fail=True)
    monkeypatch.setitem(
        sys.modules,
        "duckdb",
        SimpleNamespace(connect=lambda database: failing),
    )
    with pytest.raises(ValueError, match="malformed"):
        _profile(path, limits=limits)
    assert failing.closed is True
    assert failing.queries[:3] == [
        "SET memory_limit = '64MB'",
        "SET threads = 1",
        "DESCRIBE SELECT * FROM read_parquet(?)",
    ]


@pytest.mark.parametrize("family", ["csv", "jsonl", "parquet"])
def test_representative_profiles_have_deterministic_bounded_canonical_bytes(
    tmp_path: Path, family: str
) -> None:
    path = tmp_path / f"representative.{family}"
    if family == "csv":
        path.write_text("id,password\n1,private-sentinel\n", encoding="utf-8")
    elif family == "jsonl":
        path.write_text(
            '{"id": 1, "password": "private-sentinel"}\n',
            encoding="utf-8",
        )
    else:
        duckdb = pytest.importorskip("duckdb")
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                """
                COPY (
                    SELECT 1::INTEGER AS id,
                           'private-sentinel'::VARCHAR AS password
                ) TO ? (FORMAT PARQUET)
                """,
                [str(path)],
            )
        finally:
            connection.close()
    limits = _module().ProfileLimits(max_diagnostic_bytes=4096)

    first = canonical_json(_profile(path, limits=limits).to_dict()).encode("utf-8")
    second = canonical_json(_profile(path, limits=limits).to_dict()).encode("utf-8")

    assert first == second
    assert len(first) <= limits.max_diagnostic_bytes
    assert b"private-sentinel" not in first


def test_parquet_value_and_schema_changes_affect_expected_fingerprints(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    path = tmp_path / "orders.parquet"

    def write(query: str) -> None:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                f"COPY ({query}) TO ? (FORMAT PARQUET)",
                [str(path)],
            )
        finally:
            connection.close()

    write("SELECT 1::INTEGER AS id")
    original = _profile(path)
    write("SELECT 2::INTEGER AS id")
    value_change = _profile(path)
    write("SELECT 2::BIGINT AS order_id")
    schema_change = _profile(path)

    assert original.snapshot_fingerprint != value_change.snapshot_fingerprint
    assert original.schema_fingerprint == value_change.schema_fingerprint
    assert value_change.schema_fingerprint != schema_change.schema_fingerprint


def test_missing_parquet_dependency_is_actionable_and_never_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "orders.parquet"
    path.write_bytes(b"PAR1" + b"x" + (1).to_bytes(4, "little") + b"PAR1")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "duckdb":
            raise ModuleNotFoundError("No module named 'duckdb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match=r"fabric-rlm\[analytics\]"):
        _profile(path)
