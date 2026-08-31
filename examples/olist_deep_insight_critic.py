"""Checkpointed, source-agnostic adversarial review of discovery artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "openrouter/z-ai/glm-5.3-flash"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
CRITIC_OUTPUTS = {
    "reviewed_insights": list,
    "portfolio_challenges": list,
    "checks_performed": list,
    "synthesis_manifest": dict,
    "quality_summary": dict,
}
HIGH_RISK_CHALLENGES = {
    "contradiction",
    "denominator_integrity",
    "grain_or_join",
    "headline_consistency",
    "causal_overclaim",
}


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"JSON contains non-finite number {value}")


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    """Load one strict JSON object without accepting NaN or infinities."""

    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read {label} at {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not finite canonical JSON: {exc}") from exc


def source_fingerprint(
    discovery_payload: Mapping[str, Any],
    audit_artifact: Mapping[str, Any],
) -> str:
    """Bind both complete input artifacts with a deterministic digest."""

    canonical_inputs = _canonical_json(
        {
            "audit_artifact": audit_artifact,
            "discovery_payload": discovery_payload,
        }
    )
    digest = hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def validate_discovery(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate contract-v2 inputs and derive the exact ranked inventory."""

    if not isinstance(payload, Mapping):
        raise ValueError("discovery payload must be an object")
    if payload.get("contract_version") not in {2, 3}:
        raise ValueError("discovery payload must have contract_version 2 or 3")
    insights = payload.get("insights")
    if not isinstance(insights, list) or not insights:
        raise ValueError("discovery payload insights must be a non-empty list")

    inventory = []
    titles: set[str] = set()
    ranks: set[int] = set()
    for index, insight in enumerate(insights):
        label = f"discovery insights[{index}]"
        if not isinstance(insight, dict):
            raise ValueError(f"{label} must be an object")
        title = insight.get("title")
        rank = insight.get("rank")
        priority = insight.get("priority")
        if rank is None and isinstance(priority, Mapping):
            rank = priority.get("rank")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{label} title must be a non-empty string")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError(f"{label} rank must be a positive integer")
        if title in titles:
            raise ValueError(f"duplicate discovery insight title: {title}")
        if rank in ranks:
            raise ValueError(f"duplicate discovery insight rank: {rank}")
        titles.add(title)
        ranks.add(rank)

        item: dict[str, Any] = {"title": title, "rank": rank}
        action = insight.get("action")
        if action is not None:
            if not isinstance(action, dict):
                raise ValueError(f"{label} action must be an object")
            action_kind = action.get("kind")
            if action_kind is not None:
                if action_kind not in {"program", "diagnostic"}:
                    raise ValueError(f"{label} action.kind is unsupported")
                item["action_kind"] = action_kind
        assessment = insight.get("diagnostic_assessment")
        if assessment is not None:
            if not isinstance(assessment, dict):
                raise ValueError(
                    f"{label} diagnostic_assessment must be an object"
                )
            readiness = assessment.get("decision_readiness")
            if readiness is not None:
                if readiness not in {"act_ready", "investigate_first"}:
                    raise ValueError(
                        f"{label} diagnostic_assessment.decision_readiness "
                        "is unsupported"
                    )
                item["decision_readiness"] = readiness
        inventory.append(item)
    return sorted(inventory, key=lambda item: item["rank"])


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _audit_values_match(expected: Any, actual: float) -> bool:
    expected_number = float(expected)
    if math.isclose(actual, expected_number, rel_tol=1e-9, abs_tol=1e-9):
        return True
    if type(expected) is int:
        return False
    expected_decimal = Decimal(str(expected))
    exponent = expected_decimal.as_tuple().exponent
    if exponent >= 0:
        return False
    tolerance = Decimal("0.5") * (Decimal(10) ** exponent)
    difference = abs(Decimal(str(actual)) - expected_decimal)
    return difference <= tolerance


def validate_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a previously successful audit and return its safe attestation."""

    if not isinstance(audit, Mapping):
        raise ValueError("audit artifact must be an object")
    if "status" in audit and audit["status"] != "passed":
        raise ValueError("audit artifact status must be passed")
    checks = audit.get("checks")
    if not isinstance(checks, list):
        raise ValueError("audit artifact checks must be a list")
    if not checks:
        raise ValueError("audit artifact checks must be non-empty")
    if "total_checks" in audit and audit["total_checks"] != len(checks):
        raise ValueError("audit artifact total_checks does not match checks")

    safe_checks = []
    for index, check in enumerate(checks):
        label = f"audit checks[{index}]"
        if not isinstance(check, dict):
            raise ValueError(f"{label} must be an object")
        path = check.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{label} path must be a non-empty string")
        if "expected" not in check:
            raise ValueError(f"{label} expected is required")
        if "actual" not in check:
            raise ValueError(f"{label} actual is required")
        expected = _finite_number(check["expected"], f"{label} expected")
        actual = _finite_number(check["actual"], f"{label} actual")
        if not _audit_values_match(check["expected"], actual):
            raise ValueError(f"{label} expected/actual pair is not successful")
        safe_checks.append(
            {"path": path, "expected": check["expected"], "actual": check["actual"]}
        )
    return {"checks": safe_checks}


def build_critic_prompt(
    discovery_payload: Mapping[str, Any],
    audit_attestation: Mapping[str, Any],
    source_inventory: list[dict[str, Any]],
) -> str:
    """Build the portable adversarial brief from canonical source artifacts."""

    return f"""\
Perform a source-agnostic adversarial analytics review under the
deep_insight_critic skill. Return exactly the five typed partial fields
requested by the host; the host supplies all version, fingerprint, and source
inventory invariants.

REVIEW RULES
- Review every source insight exactly once, in the supplied inventory order.
- Perform every taxonomy check exactly once and in taxonomy order.
- Challenge decision fitness; do not perform a style rewrite.
- There is no approval quota. Rejecting every source insight is valid.
- Evidence refs must use exact discovery/audit paths present below. Do not
  invent SQL, files, notebook cells, recomputations, or unsupported evidence.
- Treat the audit as a prior successful numeric attestation only. Do not rerun
  SQL or claim that the worker independently accessed source data.
- Reject averages-only findings unless sample size, denominator, median or
  distribution tail, and sensitivity to skew are addressed or explicitly
  gated as investigate-first.
- Test censoring, selection effects, obvious confounders, and population
  comparability before treating subgroup differences as decision-ready.
- Require exposure-normalized comparisons for lifecycle-stage activity,
  engagement, workload, and other measures that accumulate with time.
- A model-proposed threshold or external benchmark is not governed evidence.
  Require an approved source, or remove the threshold-based risk/action claim.
- An insight title must never claim more than its evidence tier and
  decision_readiness permit. For investigate-first findings, challenge titles
  or interpretations using affects, drives, causes, root cause, primary lever,
  intervention target, will improve, or failure language.
- Heavy-tail claims require P95 or P99 evidence, not only a mean, median, late
  rate, or P90. Otherwise require neutral tail-risk investigation wording.
- Step-change or level-shift language requires a formal change-point diagnostic
  with explicit pre/post windows and partial-period policy; otherwise require
  possible or candidate wording.
- Concentration claims must state the eligible population and concentration
  curve context. A top-N count without the total eligible population is
  incomplete.
- Challenge qualitative severity such as extreme, severe, high concentration,
  rare, heavy, unusually high, or nearly absent unless an approved benchmark or
  governed threshold supports it.
- Reject opaque statements such as "the primary measured difference is X";
  require population, comparison, period, unit, denominator, and effect.
- Concentration-based program actions require a governed threshold, not a
  model-proposed cutoff.
- Check bound direction. If under-merging can hide repeated entities and make
  the true rate higher, the observed rate is a lower bound, not an upper bound.
- Exhaustively satisfy the skill contract, then call SUBMIT with native values.

HOST SOURCE INVENTORY
{_canonical_json(source_inventory)}

AUTHORITATIVE DISCOVERY PAYLOAD
{_canonical_json(discovery_payload)}

SAFE AUDIT ATTESTATION
{_canonical_json(audit_attestation)}
"""


def extract_partial(result: Any) -> dict[str, Any]:
    """Extract an exact native typed partial from an RLM result."""

    if not hasattr(result, "payload"):
        raise ValueError("critic RLM result has no payload")
    payload = result.payload
    if not isinstance(payload, Mapping):
        raise ValueError("critic RLM payload must be a native mapping")
    if set(payload) != set(CRITIC_OUTPUTS):
        raise ValueError(
            "critic RLM payload must contain exactly "
            + ", ".join(CRITIC_OUTPUTS)
        )
    partial = dict(payload)
    for field, expected_type in CRITIC_OUTPUTS.items():
        if type(partial[field]) is not expected_type:
            raise ValueError(
                f"critic RLM payload field {field} must be "
                f"{expected_type.__name__}"
            )
    return partial


def normalize_critic_partial(
    partial: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Repair critic state combinations that are forbidden by its contract."""

    normalized = deepcopy(dict(partial))
    changes: list[str] = []
    reviewed = normalized.get("reviewed_insights")
    if isinstance(reviewed, list):
        for insight_index, insight in enumerate(reviewed):
            challenges = (
                insight.get("challenges") if isinstance(insight, dict) else None
            )
            if not isinstance(challenges, list):
                continue
            for challenge_index, challenge in enumerate(challenges):
                if (
                    isinstance(challenge, dict)
                    and challenge.get("type") in HIGH_RISK_CHALLENGES
                    and challenge.get("severity") == "minor"
                ):
                    challenge["severity"] = "material"
                    changes.append(
                        "$.reviewed_insights"
                        f"[{insight_index}].challenges[{challenge_index}].severity"
                    )
            if (
                isinstance(insight, dict)
                and insight.get("verdict") == "revise"
                and insight.get("synthesis_eligible") is True
            ):
                resolutions = insight.get("resolutions")
                resolution_by_index = {
                    resolution.get("challenge_index"): resolution.get("status")
                    for resolution in (
                        resolutions if isinstance(resolutions, list) else ()
                    )
                    if isinstance(resolution, dict)
                }
                required_changes = insight.get("required_changes")
                has_investigation_gate = any(
                    isinstance(change, dict)
                    and change.get("gate") == "investigate_first"
                    for change in (
                        required_changes
                        if isinstance(required_changes, list)
                        else ()
                    )
                )
                unresolved_material = any(
                    isinstance(challenge, dict)
                    and challenge.get("severity") in {"material", "blocking"}
                    and resolution_by_index.get(challenge_index)
                    not in {"resolved", "gated"}
                    for challenge_index, challenge in enumerate(challenges)
                )
                if unresolved_material and not has_investigation_gate:
                    insight["synthesis_eligible"] = False
                    changes.append(
                        "$.reviewed_insights"
                        f"[{insight_index}].synthesis_eligible"
                    )
    portfolio = normalized.get("portfolio_challenges")
    if isinstance(portfolio, list):
        for challenge_index, challenge in enumerate(portfolio):
            if (
                isinstance(challenge, dict)
                and challenge.get("type") in HIGH_RISK_CHALLENGES
                and challenge.get("severity") == "minor"
            ):
                challenge["severity"] = "material"
                changes.append(
                    f"$.portfolio_challenges[{challenge_index}].severity"
                )
    return normalized, tuple(changes)


def assemble_critic(
    partial: Mapping[str, Any],
    fingerprint: str,
    inventory: list[dict[str, Any]],
    source_contract_version: int = 2,
) -> dict[str, Any]:
    return {
        "critic_version": 1,
        "source_contract_version": source_contract_version,
        "source_fingerprint": fingerprint,
        "source_inventory": inventory,
        **partial,
    }


def verify_critic(payload: Mapping[str, Any]) -> None:
    """Run the packaged critic verifier, preserving its assertion as context."""

    from fabric_rlm.skill_loader import SkillLoader

    source = SkillLoader().load("deep_insight_critic").verifier_source
    if source is None:
        raise RuntimeError("packaged deep_insight_critic verifier is unavailable")
    namespace: dict[str, Any] = {}
    exec(compile(source, "<deep_insight_critic verifier>", "exec"), namespace)
    try:
        namespace["verify"](payload)
    except AssertionError as exc:
        raise ValueError(
            f"portable deep-insight critic verification failed: {exc}"
        ) from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_runtime_dependencies():
    import dspy
    from fabric_rlm import RLM
    from fabric_rlm.metrics import summarize_trajectory

    return dspy, RLM, summarize_trajectory


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    checkpoint = load_json(path, "critic checkpoint")
    if checkpoint.get("input_fingerprint") != fingerprint:
        return None
    try:
        if set(checkpoint) != {"input_fingerprint", "partial"}:
            raise ValueError("checkpoint must contain exactly input_fingerprint and partial")
        partial = extract_partial(SimpleResult(checkpoint["partial"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"matching critic checkpoint is malformed: {exc}") from exc
    return partial


class SimpleResult:
    """Minimal adapter that reuses native partial validation for checkpoints."""

    def __init__(self, payload: Any):
        self.payload = payload


def _safe_turn_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "turns": summary.get("turns", 0),
        "submitted": bool(summary.get("submitted", False)),
        "error_turns": summary.get("error_turns", 0),
        "validation_failed_turns": summary.get("validation_failed_turns", 0),
    }


def _run_counts(critic: Mapping[str, Any]) -> dict[str, int]:
    manifest = critic["synthesis_manifest"]
    return {
        "source": len(critic["source_inventory"]),
        "reviewed": len(critic["reviewed_insights"]),
        "approved": len(manifest["approved"]),
        "revised": len(manifest["revised"]),
        "rejected": len(manifest["rejected"]),
        "program": len(manifest["program_action_titles"]),
        "diagnostic": len(manifest["diagnostic_only_titles"]),
        "blocking_issues": len(critic["quality_summary"]["blocking_issues"]),
    }


def run_critic(
    payload_path: str | Path,
    audit_path: str | Path,
    *,
    output_dir: str | Path = Path("_local") / "olist_deep_insight_critic",
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    max_turns: int = 12,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Review one audited discovery payload, resuming a valid checkpoint."""

    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
        raise ValueError("turns must be a positive integer")
    if not isinstance(timeout, (int, float)) or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be finite and positive")

    discovery_payload = load_json(payload_path, "discovery payload")
    audit_artifact = load_json(audit_path, "audit artifact")
    inventory = validate_discovery(discovery_payload)
    audit_attestation = validate_audit(audit_artifact)
    fingerprint = source_fingerprint(discovery_payload, audit_artifact)
    output = Path(output_dir)
    checkpoint_path = output / "critic.checkpoint.json"
    partial = _load_checkpoint(checkpoint_path, fingerprint)
    cached = partial is not None

    if partial is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to run the critic")
        dspy, RLM, summarize_trajectory = _load_runtime_dependencies()
        lm = dspy.LM(
            model,
            api_key=api_key,
            api_base=api_base,
            max_tokens=20000,
            temperature=0,
            cache=False,
            reasoning={"max_tokens": 4096, "exclude": True},
        )
        dspy.configure(lm=lm)
        rlm = RLM.from_task(
            task=build_critic_prompt(
                discovery_payload, audit_attestation, inventory
            ),
            outputs=CRITIC_OUTPUTS,
            lm=lm,
            skills=["deep_insight_critic"],
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=max_turns,
            reserve_finalize_turns=min(4, max_turns),
            timeout=timeout,
            verbose=False,
        )
        result = rlm.run()
        partial = extract_partial(result)
        turn_summary = _safe_turn_summary(summarize_trajectory(result.trajectory))
    else:
        turn_summary = {
            "turns": 0,
            "submitted": True,
            "error_turns": 0,
            "validation_failed_turns": 0,
        }

    partial, normalizations = normalize_critic_partial(partial)
    critic = assemble_critic(
        partial,
        fingerprint,
        inventory,
        discovery_payload["contract_version"],
    )
    try:
        verify_critic(critic)
    except ValueError as exc:
        if cached:
            raise ValueError(
                f"matching critic checkpoint is malformed: {exc}"
            ) from exc
        raise

    if not cached or normalizations:
        _atomic_json(
            checkpoint_path,
            {"input_fingerprint": fingerprint, "partial": partial},
        )
    run = {
        "status": "success",
        "fingerprint": fingerprint,
        "model": model,
        "cached": cached,
        "normalizations": {
            "count": len(normalizations),
            "paths": list(normalizations),
        },
        "turns": turn_summary,
        "counts": _run_counts(critic),
    }
    _atomic_json(output / "critic.json", critic)
    _atomic_json(output / "critic-run.json", run)
    return {"critic": critic, "run": run}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a checkpointed adversarial critic over audited insights."
    )
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("_local") / "olist_deep_insight_critic",
        type=Path,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--turns", default=12, type=int)
    parser.add_argument("--timeout", default=1800.0, type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_critic(
        args.payload,
        args.audit,
        output_dir=args.output_dir,
        model=args.model,
        api_base=args.api_base,
        max_turns=args.turns,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "status": record["run"]["status"],
                "fingerprint": record["run"]["fingerprint"],
                "cached": record["run"]["cached"],
                "artifacts": {
                    "critic": str(args.output_dir / "critic.json"),
                    "run": str(args.output_dir / "critic-run.json"),
                    "checkpoint": str(
                        args.output_dir / "critic.checkpoint.json"
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
