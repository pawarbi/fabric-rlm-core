"""Metadata-only knowledge profiling for SemanticModel handles."""

from __future__ import annotations

import json
import traceback

import pytest

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
