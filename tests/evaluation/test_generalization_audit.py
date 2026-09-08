from __future__ import annotations

from pathlib import Path

from evaluation.generalization.audit import run_offline_audit
from evaluation.generalization.fixtures import generate_fixtures
from evaluation.generalization.freeze import create_freeze_manifest, verify_freeze


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SHA = "b5226712a9aa41c3173d5f427e81244c333c0179"


def test_freeze_manifest_covers_core_and_bundled_skills() -> None:
    manifest = create_freeze_manifest(REPO_ROOT, baseline_sha=BASELINE_SHA)

    assert manifest["baseline_sha"] == BASELINE_SHA
    assert "fabric_rlm/runtime.py" in manifest["files"]
    assert "fabric_rlm/skills/core.md" in manifest["files"]
    assert verify_freeze(REPO_ROOT, manifest) == []


def test_audit_distinguishes_arr_examples_from_runtime_assumptions(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    generate_fixtures(fixture_root, large_rows=20_000)

    audit = run_offline_audit(REPO_ROOT, fixture_root)

    assert audit["arr_mentions"]["executable"] == []
    assert audit["arr_mentions"]["documentation_or_comments"]
    assert audit["naming_effects"]["descriptive_time_construct_lessons"] == 1
    assert audit["naming_effects"]["abbreviated_time_construct_lessons"] == 0
    assert audit["naming_effects"]["english_customer_trigger_score"] > 0
    assert audit["naming_effects"]["unfamiliar_account_trigger_score"] == 0


def test_audit_records_source_and_large_file_limits(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    generate_fixtures(fixture_root, large_rows=20_000)

    audit = run_offline_audit(REPO_ROOT, fixture_root)

    assert audit["sources"]["real_file_profiles"] == ["csv"]
    assert audit["sources"]["mocked_adapter_tests"] == ["lakehouse", "semantic_model"]
    assert audit["sources"]["unsupported_or_unavailable"] == [
        "generic_sql",
        "real_lakehouse_no_credentials",
        "real_semantic_model_no_credentials",
    ]
    assert audit["large_file"]["snapshot_exact"] is False
    assert audit["large_file"]["registered_operations"] == 0
    assert audit["learn_only"]["file_lessons"] == 0
    assert audit["learn_only"]["small_csv_operations"] == 1
