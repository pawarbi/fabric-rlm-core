"""Metadata-only knowledge profiling for SemanticModel handles."""

from __future__ import annotations

import json
import sys
import traceback
from types import ModuleType

import pytest

from fabric_rlm.knowledge import canonical_json
from fabric_rlm.knowledge_sources import ProfileLimits, profile_sources
from fabric_rlm.semantic_model import SemanticModel


class FakeFrame:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]

    def to_dict(self, orient="dict"):
        assert orient == "records"
        return [dict(row) for row in self._rows]


class FakeSemanticModel(SemanticModel):
    def __init__(
        self,
        *,
        dataset="Sales Model",
        workspace="Analytics Workspace",
        tables=None,
        columns=None,
        measures=None,
        relationships=None,
        failure=None,
        mutate_after_first_pass=False,
    ):
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "validate", False)
        self._metadata = {
            "tables": tables
            or [
                {
                    "Name": "Sales",
                    "Description": "Sales facts",
                    "Type": "Table",
                },
                {"Name": "Date", "Description": "Calendar", "Type": "Table"},
            ],
            "columns": columns
            or [
                {
                    "Table Name": "Sales",
                    "Column Name": "Amount",
                    "Data Type": "Double",
                    "Description": "Revenue value",
                },
                {
                    "Table Name": "Date",
                    "Column Name": "Date",
                    "Data Type": "DateTime",
                    "Description": "Calendar date",
                },
            ],
            "measures": measures
            or [
                {
                    "Table Name": "Sales",
                    "Measure Name": "Total Sales",
                    "Measure Expression": "SUM(Sales[Amount])",
                    "Measure Description": "Revenue across all channels",
                }
            ],
            "relationships": relationships
            or [
                {
                    "From Table": "Sales",
                    "From Column": "DateKey",
                    "To Table": "Date",
                    "To Column": "DateKey",
                    "Multiplicity": "m:1",
                    "Relationship Name": "Sales_Date",
                }
            ],
        }
        self.failure = failure
        self.mutate_after_first_pass = mutate_after_first_pass
        self.calls = []

    def _read(self, name):
        self.calls.append(name)
        if self.failure is not None:
            raise self.failure
        rows = [dict(row) for row in self._metadata[name]]
        if self.mutate_after_first_pass and len(self.calls) > 4 and name == "measures":
            rows[0]["Measure Expression"] = "SUMX(Sales, Sales[Amount])"
        return FakeFrame(rows)

    def tables(self):
        return self._read("tables")

    def columns(self):
        return self._read("columns")

    def measures(self):
        return self._read("measures")

    def relationships(self):
        return self._read("relationships")

    def dax(self, *args, **kwargs):
        raise AssertionError("knowledge profiling must never evaluate DAX")

    def measure(self, *args, **kwargs):
        raise AssertionError("knowledge profiling must never evaluate measures")

    def read_table(self, *args, **kwargs):
        raise AssertionError("knowledge profiling must never read table rows")


def _profile(model, *, role="numeric_evidence", limits=None):
    from fabric_rlm.knowledge_semantic_model import semantic_model_registry

    return profile_sources(
        {"sales": model},
        roles={"sales": role},
        limits=limits,
        registry=semantic_model_registry(),
    )[0]


def test_profiles_only_public_metadata_methods_and_rechecks_them():
    model = FakeSemanticModel()

    profile = _profile(model)

    assert profile.family == "semantic_model"
    assert model.calls == [
        "tables",
        "columns",
        "measures",
        "relationships",
        "tables",
        "columns",
        "measures",
        "relationships",
    ]


def test_real_semantic_model_wrapper_calls_only_stubbed_sempy_metadata_apis(
    monkeypatch,
):
    calls = []
    frames = {
        "list_tables": FakeFrame(
            [{"Name": "Sales", "Description": "", "Type": "Table"}]
        ),
        "list_columns": FakeFrame(
            [
                {
                    "Table Name": "Sales",
                    "Column Name": "Amount",
                    "Data Type": "Double",
                }
            ]
        ),
        "list_measures": FakeFrame(
            [
                {
                    "Table Name": "Sales",
                    "Measure Name": "Revenue",
                    "Measure Expression": "SUM(Sales[Amount])",
                }
            ]
        ),
        "list_relationships": FakeFrame([]),
    }
    fabric = ModuleType("sempy.fabric")

    for name, frame in frames.items():
        def metadata(dataset, *, workspace=None, _name=name, _frame=frame):
            calls.append((_name, dataset, workspace))
            return _frame

        setattr(fabric, name, metadata)

    for forbidden in ("evaluate_dax", "read_table", "execute_tmsl"):
        setattr(
            fabric,
            forbidden,
            lambda *args, _name=forbidden, **kwargs: pytest.fail(
                f"{_name} must not be called"
            ),
        )

    sempy = ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    model = SemanticModel(
        "Sales Model",
        workspace="Analytics Workspace",
        validate=False,
    )

    profile = _profile(model)

    assert profile.family == "semantic_model"
    assert calls == [
        (name, "Sales Model", "Analytics Workspace")
        for _ in range(2)
        for name in (
            "list_tables",
            "list_columns",
            "list_measures",
            "list_relationships",
        )
    ]


def test_metadata_row_order_does_not_affect_profile():
    first_model = FakeSemanticModel()
    second_model = FakeSemanticModel(
        tables=list(reversed(first_model._metadata["tables"])),
        columns=list(reversed(first_model._metadata["columns"])),
        measures=list(reversed(first_model._metadata["measures"])),
        relationships=list(
            reversed(
                [
                    *first_model._metadata["relationships"],
                    {
                        "From Table": "Sales",
                        "From Column": "ProductKey",
                        "To Table": "Product",
                        "To Column": "ProductKey",
                        "Multiplicity": "m:1",
                    },
                ]
            )
        ),
    )
    first_model._metadata["relationships"].append(
        {
            "From Table": "Sales",
            "From Column": "ProductKey",
            "To Table": "Product",
            "To Column": "ProductKey",
            "Multiplicity": "m:1",
        }
    )

    first = _profile(first_model)
    second = _profile(second_model)

    assert first.to_dict() == second.to_dict()


def test_locator_is_an_opaque_fingerprint_of_logical_model_coordinates():
    model = FakeSemanticModel(
        dataset="11111111-2222-3333-4444-555555555555",
        workspace="Finance Display Name",
    )

    profile = _profile(model)
    encoded = json.dumps(profile.to_dict(), sort_keys=True)

    assert profile.locator.startswith("semantic-model/v1/")
    assert model.dataset not in encoded
    assert model.workspace not in encoded
    assert "Finance Display Name" not in encoded


def test_model_identity_changes_locator_even_when_metadata_is_identical():
    first = _profile(FakeSemanticModel(dataset="Model A", workspace="Workspace"))
    second = _profile(FakeSemanticModel(dataset="Model B", workspace="Workspace"))

    assert first.schema_fingerprint == second.schema_fingerprint
    assert first.snapshot_fingerprint == second.snapshot_fingerprint
    assert first.locator != second.locator


def test_values_credentials_and_expression_bodies_are_not_persisted():
    secret = "Bearer top-secret-token"
    model = FakeSemanticModel(
        measures=[
            {
                "Table Name": "Sales",
                "Measure Name": "Private Metric",
                "Measure Expression": 'CALCULATE([Sales], Customer[Email] = "a@b.com")',
                "Measure Description": "Confidential business definition",
                "Authorization": secret,
            }
        ],
        columns=[
            {
                "Table Name": "Sales",
                "Column Name": "Amount",
                "Data Type": "Double",
                "Credential": "password=private-sentinel",
            }
        ],
    )

    encoded = json.dumps(_profile(model).to_dict(), sort_keys=True)

    assert secret not in encoded
    assert "private-sentinel" not in encoded
    assert "a@b.com" not in encoded
    assert "CALCULATE" not in encoded
    assert "Confidential business definition" not in encoded
    assert "expression_fingerprint" in encoded


def test_profile_metadata_is_compatible_with_knowledge_persistence_validation():
    from fabric_rlm.knowledge_store import (
        _validate_diagnostics,
        _validate_schema_descriptor,
    )

    profile = _profile(FakeSemanticModel())

    _validate_schema_descriptor(profile.schema, "profile.schema")
    _validate_diagnostics(profile.diagnostics, "profile.diagnostics")


def test_expression_change_is_snapshot_drift_not_structural_drift():
    original = _profile(FakeSemanticModel())
    changed = _profile(
        FakeSemanticModel(
            measures=[
                {
                    "Table Name": "Sales",
                    "Measure Name": "Total Sales",
                    "Measure Expression": "SUMX(Sales, Sales[Amount])",
                    "Measure Description": "Revenue across all channels",
                }
            ]
        )
    )

    assert original.schema_fingerprint == changed.schema_fingerprint
    assert original.snapshot_fingerprint != changed.snapshot_fingerprint


def test_structural_change_updates_schema_and_snapshot_fingerprints():
    original = _profile(FakeSemanticModel())
    changed = _profile(
        FakeSemanticModel(
            columns=[
                {
                    "Table Name": "Sales",
                    "Column Name": "Amount",
                    "Data Type": "Decimal",
                }
            ]
        )
    )

    assert original.schema_fingerprint != changed.schema_fingerprint
    assert original.snapshot_fingerprint != changed.snapshot_fingerprint


def test_retained_metadata_is_deterministically_bounded_and_disclosed():
    model = FakeSemanticModel(
        tables=[
            {
                "Name": f"Table {index:02d} " + ("N" * 200),
                "Description": "x" * 200,
                "Type": "Table",
            }
            for index in range(10)
        ],
        columns=[
            {
                "Table Name": f"Table {index:02d}",
                "Column Name": "Value",
                "Data Type": "String",
                "Description": "y" * 200,
            }
            for index in range(10)
        ],
        measures=[
            {
                "Table Name": f"Table {index:02d}",
                "Measure Name": "Metric",
                "Measure Expression": f"SUM('Table {index:02d}'[Value])",
                "Measure Description": "z" * 200,
            }
            for index in range(10)
        ],
        relationships=[],
    )
    limits = ProfileLimits(
        max_records=2,
        max_fields=10,
        max_input_bytes=256,
        max_diagnostic_bytes=4096,
    )

    first = _profile(model, limits=limits)
    second = _profile(model, limits=limits)
    encoded = json.dumps(first.to_dict(), sort_keys=True).encode()

    assert first.to_dict() == second.to_dict()
    assert first.diagnostics["records_truncated"] is True
    assert first.diagnostics["text_truncated"] is True
    assert first.diagnostics["fields_truncated"] is False
    assert first.diagnostics["provider_materialization_bounded"] is False
    assert len(encoded) <= limits.max_diagnostic_bytes


def test_field_and_nesting_limits_bound_retained_schema():
    field_limited = _profile(
        FakeSemanticModel(),
        limits=ProfileLimits(max_fields=1),
    )
    for records in field_limited.schema.values():
        assert all(len(descriptor) <= 1 for descriptor in records.values())
    assert field_limited.diagnostics["fields_truncated"] is True

    nesting_limited = _profile(
        FakeSemanticModel(),
        limits=ProfileLimits(max_nesting_depth=1),
    )
    assert dict(nesting_limited.schema) == {}
    assert nesting_limited.diagnostics["nesting_truncated"] is True
    assert nesting_limited.diagnostics["text_truncated"] is False


@pytest.mark.parametrize(
    "role",
    ["numeric_evidence", "lookup", "context_only", "excluded"],
)
def test_supported_roles_profile(role):
    assert _profile(FakeSemanticModel(), role=role).role == role


def test_template_role_is_rejected_before_metadata_io():
    model = FakeSemanticModel()

    with pytest.raises(ValueError, match="role.*semantic_model|semantic_model.*role"):
        _profile(model, role="template")

    assert model.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("sempy missing; token=private-sentinel"),
        PermissionError("Authorization: Bearer private-sentinel"),
    ],
)
def test_metadata_failures_are_sanitized(failure):
    model = FakeSemanticModel(failure=failure)

    with pytest.raises(ValueError) as error:
        _profile(model)

    message = str(error.value)
    assert "private-sentinel" not in message
    assert model.dataset not in message
    assert model.workspace not in message
    assert error.value.__cause__ is None
    formatted = "".join(
        traceback.format_exception(
            type(error.value),
            error.value,
            error.value.__traceback__,
        )
    )
    assert "private-sentinel" not in formatted


def test_provider_frame_conversion_failures_are_sanitized():
    class BadFrame:
        def to_dict(self, orient="dict"):
            raise RuntimeError("token=private-sentinel")

    model = FakeSemanticModel()
    model.tables = lambda: BadFrame()

    with pytest.raises(ValueError) as error:
        _profile(model)

    formatted = "".join(
        traceback.format_exception(
            type(error.value),
            error.value,
            error.value.__traceback__,
        )
    )
    assert error.value.__cause__ is None
    assert "private-sentinel" not in formatted


def test_change_during_profiling_fails_closed():
    model = FakeSemanticModel(mutate_after_first_pass=True)

    with pytest.raises(ValueError, match="changed during profiling"):
        _profile(model)


def test_factory_is_explicit_and_does_not_change_the_default_registry():
    from fabric_rlm.knowledge_semantic_model import (
        SemanticModelKnowledgeAdapter,
        semantic_model_adapter,
        semantic_model_registry,
    )

    adapter = semantic_model_adapter()
    registry = semantic_model_registry()

    assert isinstance(adapter, SemanticModelKnowledgeAdapter)
    assert registry.adapters == (adapter,)
    assert adapter.matches(FakeSemanticModel())


def test_fabric_registry_composes_semantic_model_and_local_adapters():
    from fabric_rlm.knowledge_lakehouse_sources import fabric_source_registry

    registry = fabric_source_registry()

    assert registry.resolve(FakeSemanticModel()).family == "semantic_model"
    assert registry.resolve("orders.csv").family == "csv"


def _wide_model(measure_count=4800, column_count=2500):
    return FakeSemanticModel(
        tables=[{"Name": f"T{i:03d}", "Description": "", "Type": "Table"} for i in range(120)],
        columns=[
            {"Table Name": f"T{i % 120:03d}", "Column Name": f"Column {i:04d}",
             "Data Type": "String", "Description": ""}
            for i in range(column_count)
        ],
        measures=[
            {"Table Name": "T000", "Measure Name": f"Measure {i:04d} POS $ Sales IYA",
             "Measure Expression": f"SUM(T000[C{i}])", "Measure Description": ""}
            for i in range(measure_count)
        ],
        relationships=[],
    )


def test_a_model_with_thousands_of_measures_profiles_at_default_limits():
    # Found in Fabric: a 122-table, 2,533-column, 4,800-measure model failed
    # RLM.learn with "canonical SourceProfile exceeds max_diagnostic_bytes".
    limits = ProfileLimits()
    profile = _profile(_wide_model(), limits=limits)
    # The same encoding profile_sources() measures against the limit.
    encoded = canonical_json(profile.to_dict()).encode("utf-8")

    assert len(encoded) <= limits.max_diagnostic_bytes
    assert profile.diagnostics["records_truncated"] is True
    assert 0 < profile.diagnostics["retained_record_cap"] < limits.max_records
    assert profile.diagnostics["records_inspected"] >= 120 + 2500 + 4800
    assert profile.schema["measures"]                     # something useful is kept
    assert _profile(_wide_model(), limits=limits).to_dict() == profile.to_dict()   # deterministic


def test_shrinking_what_is_retained_does_not_weaken_drift_detection():
    limits = ProfileLimits()
    before = _profile(_wide_model(), limits=limits)
    retained = set(before.schema["measures"])
    assert len(retained) < 4800
    # Which records are retained follows a hash order, so edit every measure in
    # turn until one is found that neither profile retains.
    for index in range(4800):
        changed = _wide_model()
        record = changed._metadata["measures"][index]
        record["Measure Expression"] = "SUM(T000[other])"
        key = f"T000[{record['Measure Name']}]"
        after = _profile(changed, limits=limits)
        if key not in retained and key not in after.schema["measures"]:
            break
    else:
        pytest.fail("every measure was retained")

    assert after.snapshot_fingerprint != before.snapshot_fingerprint


def test_a_model_that_fits_is_profiled_exactly_as_before():
    profile = _profile(FakeSemanticModel())
    assert "retained_record_cap" not in profile.diagnostics


def test_the_generic_profile_size_error_says_which_source_and_how_to_raise_it(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(",".join(f"column_{i:03d}_{'x' * 40}" for i in range(200)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        profile_sources({"wide": path}, roles={"wide": "numeric_evidence"},
                        limits=ProfileLimits(max_diagnostic_bytes=512))
    message = str(raised.value)
    assert "source alias wide" in message and "512" in message
    assert "ProfileLimits(max_diagnostic_bytes=" in message
