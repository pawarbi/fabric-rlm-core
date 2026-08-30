"""Tests for the checkpointed portable deep-insight critic example."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "olist_deep_insight_critic.py"
TAXONOMY = (
    "obviousness",
    "cross_domain_depth",
    "contradiction",
    "denominator_integrity",
    "metric_definition",
    "alternative_explanation",
    "target_basis",
    "benchmark_basis",
    "causal_overclaim",
    "grain_or_join",
    "headline_consistency",
    "actionability",
)


def load_module():
    spec = importlib.util.spec_from_file_location("olist_deep_insight_critic", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discovery(title: str = "Measured retention signal") -> dict:
    return {
        "contract_version": 2,
        "insights": [
            {
                "title": title,
                "rank": 1,
                "action": {"kind": "diagnostic"},
                "diagnostic_assessment": {
                    "decision_readiness": "investigate_first"
                },
                "statement": "A measured source-derived signal needs review.",
            }
        ],
    }


def audit(actual: float = 12.0) -> dict:
    return {
        "status": "passed",
        "total_checks": 1,
        "checks": [
            {
                "path": "insights[0].metric_spec.value",
                "expected": 12.0,
                "actual": actual,
            }
        ],
    }


def rejected_partial(title: str = "Measured retention signal") -> dict:
    checks = [
        {
            "type": category,
            "status": "tested",
            "rationale": f"Tested {category} against authoritative source evidence.",
            "evidence_refs": ["discovery.insights[0].statement"],
        }
        for category in TAXONOMY
    ]
    return {
        "reviewed_insights": [
            {
                "title": title,
                "rank": 1,
                "verdict": "reject",
                "decision_effect": (
                    "Exclude this finding from downstream decision synthesis."
                ),
                "challenges": [
                    {
                        "id": "insight-1-obviousness",
                        "type": "obviousness",
                        "assessment": (
                            "The finding does not materially change the decision."
                        ),
                        "severity": "material",
                        "evidence_refs": ["discovery.insights[0].statement"],
                    }
                ],
                "required_changes": [
                    {
                        "change": (
                            "Supply decision-changing evidence before reconsideration."
                        ),
                        "gate": "investigate_first",
                    }
                ],
                "synthesis_eligible": False,
                "resolutions": [],
            }
        ],
        "portfolio_challenges": [],
        "checks_performed": checks,
        "synthesis_manifest": {
            "approved": [],
            "revised": [],
            "rejected": [title],
            "program_action_titles": [],
            "diagnostic_only_titles": [],
        },
        "quality_summary": {
            "process_rigor": 8,
            "analytical_depth": 8,
            "decision_quality": 8,
            "overall_assessment": (
                "The only source finding is not decision-changing enough."
            ),
            "blocking_issues": [],
        },
    }


def write_inputs(tmp_path: Path, payload: dict | str, audit_value: dict | str):
    payload_path = tmp_path / "payload.json"
    audit_path = tmp_path / "audit.json"
    payload_path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload),
        encoding="utf-8",
    )
    audit_path.write_text(
        audit_value if isinstance(audit_value, str) else json.dumps(audit_value),
        encoding="utf-8",
    )
    return payload_path, audit_path


class FakeRLM:
    calls: list[dict] = []
    partial: dict = {}

    @classmethod
    def from_task(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls()

    def run(self):
        return SimpleNamespace(payload=self.partial, trajectory=["turn"])


class FakeDSPy:
    lm_calls: list[dict] = []
    configured = []

    @classmethod
    def LM(cls, model, **kwargs):
        lm = SimpleNamespace(model=model, kwargs=kwargs)
        cls.lm_calls.append({"model": model, **kwargs})
        return lm

    @classmethod
    def configure(cls, **kwargs):
        cls.configured.append(kwargs)


def install_fake_runtime(monkeypatch, critic, partial: dict):
    FakeRLM.calls = []
    FakeRLM.partial = partial
    FakeDSPy.lm_calls = []
    FakeDSPy.configured = []
    monkeypatch.setattr(
        critic,
        "_load_runtime_dependencies",
        lambda: (
            FakeDSPy,
            FakeRLM,
            lambda trajectory: {
                "turns": len(trajectory),
                "submitted": True,
                "error_turns": 0,
                "validation_failed_turns": 0,
                "wall_time_s": 99.9,
                "state_keys": ["secret"],
            },
        ),
    )


def test_cli_requires_payload_and_audit() -> None:
    critic = load_module()
    with pytest.raises(SystemExit):
        critic.parse_args([])


def test_load_json_rejects_malformed_and_nonstandard_numbers(tmp_path: Path) -> None:
    critic = load_module()
    malformed = tmp_path / "bad.json"
    malformed.write_text('{"value":', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        critic.load_json(malformed, "discovery payload")

    malformed.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        critic.load_json(malformed, "discovery payload")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(contract_version=1), "contract_version 2 or 3"),
        (lambda value: value.pop("insights"), "insights"),
        (lambda value: value.update(insights="bad"), "insights"),
        (lambda value: value["insights"][0].pop("rank"), "rank"),
        (lambda value: value["insights"][0].update(rank=0), "positive"),
        (
            lambda value: value["insights"].append(
                {"title": "Measured retention signal", "rank": 2}
            ),
            "duplicate.*title",
        ),
        (
            lambda value: value["insights"].append(
                {"title": "Another measured signal", "rank": 1}
            ),
            "duplicate.*rank",
        ),
    ],
)
def test_discovery_validation_rejects_malformed_inventory(mutate, message) -> None:
    critic = load_module()
    value = discovery()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        critic.validate_discovery(value)


def test_discovery_validation_accepts_evidence_closure_contract() -> None:
    critic = load_module()
    value = discovery()
    value["contract_version"] = 3

    assert critic.validate_discovery(value)[0]["title"] == value["insights"][0]["title"]


def test_inventory_is_rank_ordered_and_contains_only_host_invariants() -> None:
    critic = load_module()
    value = discovery()
    value["insights"] = [
        {"title": "Second", "rank": 2, "ignored": "value"},
        {
            "title": "First",
            "rank": 1,
            "action": {"kind": "program"},
            "diagnostic_assessment": {"decision_readiness": "act_ready"},
        },
    ]
    assert critic.validate_discovery(value) == [
        {
            "title": "First",
            "rank": 1,
            "action_kind": "program",
            "decision_readiness": "act_ready",
        },
        {"title": "Second", "rank": 2},
    ]


def test_inventory_reads_contract_v2_rank_from_priority() -> None:
    critic = load_module()
    value = discovery()
    for insight in value["insights"]:
        rank = insight.pop("rank")
        insight["priority"] = {
            "rank": rank,
            "impact": "high",
            "urgency": "medium",
        }

    inventory = critic.validate_discovery(value)

    assert [item["rank"] for item in inventory] == [1]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"status": "passed"}, "checks"),
        ({"status": "passed", "checks": []}, "non-empty"),
        (
            {
                "status": "passed",
                "checks": [{"path": "x", "expected": 1.0}],
            },
            "actual",
        ),
        (
            {
                "status": "passed",
                "checks": [{"path": "x", "expected": math.inf, "actual": math.inf}],
            },
            "finite",
        ),
        (
            {
                "status": "passed",
                "checks": [{"path": "x", "expected": 1.0, "actual": 2.0}],
            },
            "not successful",
        ),
    ],
)
def test_audit_validation_rejects_untrusted_attestations(value, message) -> None:
    critic = load_module()
    with pytest.raises(ValueError, match=message):
        critic.validate_audit(value)


def test_audit_validation_accepts_successful_declared_precision_pair() -> None:
    critic = load_module()
    value = {
        "checks": [
            {
                "path": "insights[0].metric_spec.components[0]",
                "expected": -4.0606,
                "actual": -4.06058,
            },
            {
                "path": "insights[1].supporting_claims[0]",
                "expected": 6597.88,
                "actual": 6597.875,
            },
        ]
    }

    assert critic.validate_audit(value) == value


def test_audit_validation_rejects_value_beyond_declared_precision() -> None:
    critic = load_module()
    value = {
        "checks": [
            {
                "path": "insights[0].metric_spec.components[0]",
                "expected": -4.0606,
                "actual": -4.06054,
            }
        ]
    }

    with pytest.raises(ValueError, match="not successful"):
        critic.validate_audit(value)


def test_fingerprint_is_deterministic_and_binds_both_inputs() -> None:
    critic = load_module()
    payload = discovery()
    audit_value = audit()
    first = critic.source_fingerprint(payload, audit_value)
    reordered = {"insights": payload["insights"], "contract_version": 2}
    assert first == critic.source_fingerprint(reordered, audit_value)
    assert first.startswith("sha256:") and len(first) == 71
    assert first != critic.source_fingerprint(discovery("Changed title"), audit_value)
    assert first != critic.source_fingerprint(payload, audit(12.01))


def test_prompt_is_source_agnostic_exhaustive_and_contains_safe_attestation() -> None:
    critic = load_module()
    prompt = critic.build_critic_prompt(
        discovery(), critic.validate_audit(audit()), critic.validate_discovery(discovery())
    )
    lowered = prompt.lower()
    assert "every source insight" in lowered
    assert "every taxonomy" in lowered
    assert "no approval quota" in lowered
    assert "style" in lowered and "rewrite" in lowered
    assert "exact discovery/audit paths" in lowered
    assert "rejecting every" in lowered
    assert '"path":"insights[0].metric_spec.value"' in prompt
    assert "olist" not in lowered
    assert "api_key" not in lowered


def test_model_run_assembles_verifies_and_writes_safe_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    critic = load_module()
    partial = rejected_partial()
    install_fake_runtime(monkeypatch, critic, partial)
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-key")
    payload_path, audit_path = write_inputs(tmp_path, discovery(), audit())
    output = tmp_path / "out"

    record = critic.run_critic(
        payload_path,
        audit_path,
        output_dir=output,
        model="test-model",
        api_base="https://invalid.test",
        max_turns=7,
        timeout=42,
    )

    assert record["critic"]["critic_version"] == 1
    assert record["critic"]["source_contract_version"] == 2
    assert record["critic"]["source_inventory"] == [
        {
            "title": "Measured retention signal",
            "rank": 1,
            "action_kind": "diagnostic",
            "decision_readiness": "investigate_first",
        }
    ]
    assert record["critic"]["synthesis_manifest"]["rejected"] == [
        "Measured retention signal"
    ]
    assert set(partial).isdisjoint(
        {"critic_version", "source_contract_version", "source_fingerprint", "source_inventory"}
    )
    assert len(FakeRLM.calls) == 1
    rlm_call = FakeRLM.calls[0]
    assert rlm_call["outputs"] == critic.CRITIC_OUTPUTS
    assert rlm_call["skills"] == ["deep_insight_critic"]
    assert rlm_call["enable_verifier"] is False
    assert rlm_call["block_network"] is True
    assert rlm_call["max_turns"] == 7
    assert rlm_call["reserve_finalize_turns"] == 4
    assert rlm_call["timeout"] == 42
    assert rlm_call["verbose"] is False
    assert FakeDSPy.lm_calls[0]["cache"] is False

    checkpoint = json.loads((output / "critic.checkpoint.json").read_text())
    assert set(checkpoint) == {"input_fingerprint", "partial"}
    assert checkpoint["partial"] == partial
    run = json.loads((output / "critic-run.json").read_text())
    assert set(run) == {
        "cached",
        "counts",
        "fingerprint",
        "model",
        "normalizations",
        "status",
        "turns",
    }
    assert run["counts"] == {
        "approved": 0,
        "blocking_issues": 0,
        "diagnostic": 0,
        "program": 0,
        "rejected": 1,
        "reviewed": 1,
        "revised": 0,
        "source": 1,
    }
    assert set(run["turns"]) == {
        "error_turns",
        "submitted",
        "turns",
        "validation_failed_turns",
    }
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output.glob("*.json")
    )
    assert "super-secret-key" not in artifact_text
    assert "https://invalid.test" not in artifact_text
    assert "AUTHORITATIVE DISCOVERY" not in artifact_text


def test_matching_valid_checkpoint_resumes_with_zero_model_calls(
    tmp_path: Path, monkeypatch
) -> None:
    critic = load_module()
    install_fake_runtime(monkeypatch, critic, rejected_partial())
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    payload_path, audit_path = write_inputs(tmp_path, discovery(), audit())
    output = tmp_path / "out"
    critic.run_critic(payload_path, audit_path, output_dir=output)
    assert len(FakeRLM.calls) == 1

    monkeypatch.delenv("OPENROUTER_API_KEY")
    record = critic.run_critic(payload_path, audit_path, output_dir=output)
    assert len(FakeRLM.calls) == 1
    assert record["run"]["cached"] is True
    assert record["run"]["turns"]["turns"] == 0


def test_stale_checkpoint_reruns_model(tmp_path: Path, monkeypatch) -> None:
    critic = load_module()
    install_fake_runtime(monkeypatch, critic, rejected_partial())
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    payload_path, audit_path = write_inputs(tmp_path, discovery(), audit())
    output = tmp_path / "out"
    output.mkdir()
    (output / "critic.checkpoint.json").write_text(
        json.dumps({"input_fingerprint": "sha256:stale", "partial": "malformed"}),
        encoding="utf-8",
    )
    critic.run_critic(payload_path, audit_path, output_dir=output)
    assert len(FakeRLM.calls) == 1


def test_matching_malformed_checkpoint_fails_without_model_call(
    tmp_path: Path, monkeypatch
) -> None:
    critic = load_module()
    install_fake_runtime(monkeypatch, critic, rejected_partial())
    payload_value, audit_value = discovery(), audit()
    payload_path, audit_path = write_inputs(tmp_path, payload_value, audit_value)
    output = tmp_path / "out"
    output.mkdir()
    fingerprint = critic.source_fingerprint(payload_value, audit_value)
    (output / "critic.checkpoint.json").write_text(
        json.dumps({"input_fingerprint": fingerprint, "partial": {"bad": []}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matching critic checkpoint is malformed"):
        critic.run_critic(payload_path, audit_path, output_dir=output)
    assert FakeRLM.calls == []


def test_verifier_rejection_is_wrapped_and_never_checkpointed(
    tmp_path: Path, monkeypatch
) -> None:
    critic = load_module()
    partial = rejected_partial()
    partial["synthesis_manifest"]["rejected"] = []
    install_fake_runtime(monkeypatch, critic, partial)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    payload_path, audit_path = write_inputs(tmp_path, discovery(), audit())
    output = tmp_path / "out"
    with pytest.raises(
        ValueError, match="portable deep-insight critic verification failed:"
    ):
        critic.run_critic(payload_path, audit_path, output_dir=output)
    assert not (output / "critic.checkpoint.json").exists()
    assert not (output / "critic.json").exists()


def test_model_infrastructure_errors_propagate(tmp_path: Path, monkeypatch) -> None:
    critic = load_module()

    class BrokenRLM(FakeRLM):
        def run(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        critic,
        "_load_runtime_dependencies",
        lambda: (FakeDSPy, BrokenRLM, lambda trajectory: {}),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    payload_path, audit_path = write_inputs(tmp_path, discovery(), audit())
    with pytest.raises(RuntimeError, match="provider unavailable"):
        critic.run_critic(payload_path, audit_path, output_dir=tmp_path / "out")


def test_native_partial_must_have_exact_typed_shape() -> None:
    critic = load_module()
    with pytest.raises(ValueError, match="exactly"):
        critic.extract_partial(
            SimpleNamespace(payload={**rejected_partial(), "critic_version": 1})
        )
    with pytest.raises(ValueError, match="reviewed_insights must be list"):
        critic.extract_partial(
            SimpleNamespace(
                payload={**rejected_partial(), "reviewed_insights": {}}
            )
        )
    with pytest.raises(ValueError, match="native mapping"):
        critic.extract_partial(SimpleNamespace(payload=json.dumps(rejected_partial())))


def test_critic_normalization_upgrades_high_risk_minor_without_mutation() -> None:
    critic = load_module()
    partial = rejected_partial()
    partial["reviewed_insights"][0]["challenges"][0].update(
        {"type": "causal_overclaim", "severity": "minor"}
    )
    original = copy.deepcopy(partial)

    normalized, changes = critic.normalize_critic_partial(partial)

    assert partial == original
    assert normalized["reviewed_insights"][0]["challenges"][0]["severity"] == (
        "material"
    )
    assert changes == (
        "$.reviewed_insights[0].challenges[0].severity",
    )


def test_critic_normalization_preserves_minor_low_risk_challenge() -> None:
    critic = load_module()
    partial = rejected_partial()
    partial["reviewed_insights"][0]["challenges"][0].update(
        {"type": "obviousness", "severity": "minor"}
    )

    normalized, changes = critic.normalize_critic_partial(partial)

    assert normalized == partial
    assert changes == ()
