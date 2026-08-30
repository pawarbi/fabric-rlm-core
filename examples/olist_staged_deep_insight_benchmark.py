"""Three-stage local Olist deep-insight benchmark.

Stage 1 builds a measured research ledger. Stages 2 and 3 separately build the
contract scaffold and insights before host verification and numeric audit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from typing import Any, NamedTuple


def _load_base_example():
    try:
        import olist_deep_insight_benchmark as base

        return base
    except ModuleNotFoundError as exc:
        if exc.name != "olist_deep_insight_benchmark":
            raise
        path = Path(__file__).with_name("olist_deep_insight_benchmark.py")
        spec = importlib.util.spec_from_file_location(
            "_olist_deep_insight_benchmark_base", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load benchmark helpers from {path}")
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        return base


_BASE = _load_base_example()
CANONICAL_FILES = _BASE.CANONICAL_FILES
discover_sources = _BASE.discover_sources
normalize_payload = _BASE.normalize_payload
audit_to_dict = _BASE.audit_to_dict
_atomic_json = _BASE._atomic_json

DEFAULT_MODEL = "openrouter/z-ai/glm-5.3-flash"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
RESEARCH_SKILLS = ("data_exploration",)
SYNTHESIS_SKILLS = ("deep_insight_discovery", "data_exploration")
RESEARCH_SECTIONS = (
    "analysis_plan",
    "join_map",
    "method_applicability",
    "candidates",
)
SCAFFOLD_OUTPUTS = {
    "analysis_plan": dict,
    "candidates": list,
}
INSIGHT_OUTPUTS = {
    "insights": list,
}
EVIDENCE_CLOSURE_OUTPUTS = {
    "closure_plans": list,
}
CRITIC_CLOSURE_OUTPUTS = {
    "critic_closure_plans": list,
}
ACTION_SYNTHESIS_OUTPUTS = {
    "action_updates": list,
}
TARGETED_INSIGHT_OUTPUTS = {
    "insight": dict,
}
TARGETED_STATEMENT_OUTPUTS = {
    "statement": str,
}
TARGETED_INTERPRETATION_OUTPUTS = {
    "interpretation": str,
}
AUDIT_SUPPORTING_CLAIM_OUTPUTS = {
    "supporting_claim": dict,
}
AUDIT_METRIC_SPEC_OUTPUTS = {
    "metric_spec": dict,
}
AUDIT_METRIC_AND_STATEMENT_OUTPUTS = {
    "metric_spec": dict,
    "statement": str,
}
AUDIT_LEAF_OUTPUTS = {
    "audit_leaf": dict,
}
AUDIT_REJECTION_COMPONENT_OUTPUTS = {
    "rejection_component": dict,
}


class HostAuditMismatch(NamedTuple):
    path: str
    expected: float
    actual: float


def _source_lines(sources: Mapping[str, Path]) -> str:
    return "\n".join(f"- {identity}: {path}" for identity, path in sources.items())


def build_research_prompt(sources: Mapping[str, Path]) -> str:
    """Build the Stage 1 evidence-research brief."""

    return f"""\
Research the canonical public Olist sources listed below. Treat each identity
and path as authoritative:
{_source_lines(sources)}

Return a compact JSON research ledger, not the final deep-insight contract.
Quality and depth win over candidate count.

RESEARCH REQUIREMENTS
- Measure each source schema and grain.
- Build a measured join map with coverage, matched counts, unmatched counts,
  and explicit fan-out controls. Pre-aggregate one-to-many sources before joins.
- Assess method applicability for decomposition, instrumentation diagnostics,
  change points, cohorts, interactions, drivers, concentration, clustering,
  classification, and regression. State why methods are or are not applicable.
- Develop 6-10 candidate findings, including cross-domain candidates where
  measured coverage permits.
- Include quantitative rejected candidates, diagnostic alternatives,
  metric-definition sensitivities, and the benchmark/target basis.
- Favor decision-relevant findings over descriptive counts.

EVIDENCE RULES
- Evidence must be self-contained DuckDB SQL over canonical source aliases.
- Every alias must map to one source identity listed above; never depend on
  worker-created tables or views.
- Include aggregate evidence only: no raw records and no review text in output.

The research_json object must contain non-empty analysis_plan, join_map,
method_applicability, and candidates sections.
"""


def parse_research_json(value: str) -> dict[str, Any]:
    """Strictly parse and validate a Stage 1 research ledger."""

    if not isinstance(value, str):
        raise ValueError("research_json must be a string")
    if not value.strip():
        raise ValueError("research_json is empty")
    try:
        research = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"research_json is not valid JSON: {exc}") from exc
    if not isinstance(research, dict):
        raise ValueError("research_json must contain a JSON object")
    missing = [section for section in RESEARCH_SECTIONS if section not in research]
    if missing:
        raise ValueError(
            "research_json is missing required sections: " + ", ".join(missing)
        )
    empty = [section for section in RESEARCH_SECTIONS if not research[section]]
    if empty:
        raise ValueError(
            "research_json sections must be non-empty: " + ", ".join(empty)
        )
    return research


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _input_fingerprint(*inputs: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for value in inputs:
        digest.update(_compact_json(value).encode("utf-8"))
    return digest.hexdigest()


def _load_synthesis_checkpoint(
    path: Path,
    input_fingerprint: str,
    expected: Mapping[str, type],
    label: str,
) -> dict[str, Any] | None:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} checkpoint is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "input_fingerprint",
        "partial",
    }:
        raise ValueError(
            f"{label} checkpoint must contain exactly input_fingerprint and partial"
        )
    if type(envelope["input_fingerprint"]) is not str:
        raise ValueError(f"{label} checkpoint input_fingerprint must be str")
    if type(envelope["partial"]) is not dict:
        raise ValueError(f"{label} checkpoint partial must be dict")
    if envelope["input_fingerprint"] != input_fingerprint:
        return None
    _validate_partial(envelope["partial"], expected, f"{label} checkpoint partial")
    return envelope["partial"]


def _write_synthesis_checkpoint(
    path: Path, input_fingerprint: str, partial: Mapping[str, Any]
) -> None:
    _atomic_json(
        path,
        {
            "input_fingerprint": input_fingerprint,
            "partial": dict(partial),
        },
    )


def build_contract_scaffold_prompt(
    sources: Mapping[str, Path], research: Mapping[str, Any]
) -> str:
    """Build the Stage 2 contract-scaffold brief."""

    compact_research = _compact_json(research)
    return f"""\
Transform the measured research ledger into the exact
deep_insight_discovery contract scaffold. This is intentionally a partial
payload: SUBMIT exactly the native fields analysis_plan and candidates.

AUTHORITATIVE SOURCE IDENTITIES
{_source_lines(sources)}

STAGE 1 RESEARCH LEDGER
{compact_research}

SCAFFOLD CONTRACT
- Return exactly typed analysis_plan: dict and candidates: list.
- Preserve quantitative rejection evidence and its verification components.
- In analysis_plan.search_space, explicitly cover dimensions_available and
  dimensions_deferred, along with applicable populations and time grains.
- Promote 3-5 high-quality findings and establish their exact promoted titles
  and dimensions_tested. Quality wins over count.
- Use only source-derived research candidates from the embedded ledger.
- Keep diagnostic alternatives and metric-definition sensitivities explicit.
- Do not invent insights in this stage.
- Construct the native partial payload immediately and call SUBMIT by the
  first finalization turn.
"""


def build_insights_prompt(
    sources: Mapping[str, Path],
    research: Mapping[str, Any],
    scaffold: Mapping[str, Any],
) -> str:
    """Build the Stage 3 insight brief."""

    return f"""\
Create only the insights portion of deep_insight_discovery contract v2 from
the authoritative research and exact scaffold below. SUBMIT exactly the native
field insights: list. No broad exploration is permitted.

AUTHORITATIVE SOURCE IDENTITIES
{_source_lines(sources)}

STAGE 1 RESEARCH LEDGER
{_compact_json(research)}

EXACT CONTRACT SCAFFOLD
{_compact_json(scaffold)}

INSIGHT CONTRACT
- Return exactly typed insights: list.
- Create one insight for every promoted candidate and no other insight.
- Insight titles and discovery.dimensions_tested must exactly match the
  promoted titles and dimensions in the scaffold.
- Follow contract v2 diagnostics and metric specs, including competing
  explanations, robustness checks, limitations, and metric components.
- Every verification must use self-contained SQL and an exact alias->source
  identity mapping containing only authoritative identities listed above.
- Do not reference worker-created tables or views.
- Preserve observational language and reconcile metric definitions.
- Construct the native partial payload immediately and call SUBMIT by the
  first finalization turn.
"""


def _pending_evidence_closure_targets(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    targets = []
    insights = payload.get("insights")
    if not isinstance(insights, list):
        return targets
    for insight_index, insight in enumerate(insights):
        if not isinstance(insight, Mapping):
            continue
        assessment = insight.get("diagnostic_assessment")
        if not isinstance(assessment, Mapping):
            continue
        explanations = assessment.get("explanations")
        if not isinstance(explanations, list):
            continue
        for explanation_index, explanation in enumerate(explanations):
            if (
                not isinstance(explanation, Mapping)
                or explanation.get("measurable") is not True
                or explanation.get("disposition") != "unresolved"
            ):
                continue
            explanation_id = explanation.get("explanation_id")
            if not isinstance(explanation_id, str) or not explanation_id:
                explanation_id = (
                    f"insight-{insight_index + 1}-"
                    f"explanation-{explanation_index + 1}"
                )
            targets.append(
                {
                    "explanation": explanation.get("explanation"),
                    "explanation_id": explanation_id,
                    "explanation_index": explanation_index,
                    "insight_index": insight_index,
                    "insight_title": insight.get("title"),
                }
            )
    return targets


def build_evidence_closure_prompt(
    sources: Mapping[str, Path],
    payload: Mapping[str, Any],
) -> str:
    """Build one bounded evidence-closure task for pending explanations."""

    targets = _pending_evidence_closure_targets(payload)
    return f"""\
Close only the exact measurable unresolved diagnostic explanations below.
Return exactly closure_plans: list with exactly one plan per target.

AUTHORITATIVE SOURCE IDENTITIES
{_source_lines(sources)}

EXACT PENDING TARGETS
{_compact_json({"targets": targets})}

CLOSURE PLAN CONTRACT
- Preserve each explanation_id exactly.
- Supply a substantive required_check, final disposition of ruled_out,
  weakened, or supported, finite numeric expected_value, and verification.
- Verification must use one self-contained aggregate DuckDB SQL query and an
  exact alias-to-source identity mapping containing only identities above.
- Test only the declared explanation. Do not add candidates, explanations,
  findings, actions, or source identities.
- Include aggregate evidence only: no raw records.
- Never select, group by, quote, or emit personal identifiers, contact details,
  free-text messages, transcript bodies, article bodies, subjects, descriptions,
  or other source text.
- Do not reference worker-created tables or views.
- Call SUBMIT immediately after constructing the bounded native partial.
"""


def validate_evidence_closure_plan(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate exact target coverage and source authority for a closure plan."""

    _validate_partial(plan, EVIDENCE_CLOSURE_OUTPUTS, "evidence closure plan")
    targets = _pending_evidence_closure_targets(payload)
    expected_ids = [target["explanation_id"] for target in targets]
    plans = plan["closure_plans"]
    if len(plans) != len(expected_ids):
        raise ValueError(
            "evidence closure plan must contain exactly one plan per pending target"
        )
    authorized = frozenset(sources)
    seen: set[str] = set()
    for index, item in enumerate(plans):
        label = f"closure_plans[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        if set(item) != {
            "disposition",
            "expected_value",
            "explanation_id",
            "required_check",
            "verification",
        }:
            raise ValueError(f"{label} has unsupported fields")
        explanation_id = item["explanation_id"]
        if explanation_id not in expected_ids:
            raise ValueError(f"{label} must reference an exact pending target")
        if explanation_id in seen:
            raise ValueError(
                "evidence closure plan must contain exactly one plan per pending target"
            )
        seen.add(explanation_id)
        if item["disposition"] not in {"ruled_out", "weakened", "supported"}:
            raise ValueError(f"{label} disposition is invalid")
        required_check = item["required_check"]
        if not isinstance(required_check, str) or not required_check.strip():
            raise ValueError(f"{label} required_check must be non-empty")
        expected_value = item["expected_value"]
        if (
            type(expected_value) not in {int, float}
            or not math.isfinite(expected_value)
        ):
            raise ValueError(f"{label} expected_value must be finite numeric")
        verification = item["verification"]
        if not isinstance(verification, dict):
            raise ValueError(f"{label} verification must be an object")
        if verification.get("method") != "sql":
            raise ValueError(f"{label} verification method must be sql")
        expression = verification.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"{label} verification expression must be non-empty")
        declared_sources = verification.get("sources")
        if not isinstance(declared_sources, dict) or not declared_sources:
            raise ValueError(f"{label} verification sources are required")
        for alias, identity in declared_sources.items():
            if (
                not isinstance(alias, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias) is None
            ):
                raise ValueError(f"{label} source alias is invalid")
            if identity not in authorized:
                raise ValueError(
                    f"{label} must use an authoritative source identity"
                )
    if seen != set(expected_ids):
        raise ValueError(
            "evidence closure plan must contain exactly one plan per pending target"
        )
    return deepcopy(dict(plan))


def merge_evidence_closure_plan(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    """Upgrade to contract v3 and merge only exact diagnostic evidence leaves."""

    validated = validate_evidence_closure_plan(payload, plan, sources)
    planned = {
        item["explanation_id"]: item
        for item in validated["closure_plans"]
    }
    merged = deepcopy(dict(payload))
    merged["contract_version"] = 3
    for insight_index, insight in enumerate(merged.get("insights", ())):
        assessment = insight.get("diagnostic_assessment")
        if not isinstance(assessment, dict):
            continue
        explanations = assessment.get("explanations")
        if not isinstance(explanations, list):
            continue
        for explanation_index, explanation in enumerate(explanations):
            if not isinstance(explanation, dict):
                continue
            explanation_id = explanation.get("explanation_id")
            if not isinstance(explanation_id, str) or not explanation_id:
                explanation_id = (
                    f"insight-{insight_index + 1}-"
                    f"explanation-{explanation_index + 1}"
                )
                explanation["explanation_id"] = explanation_id
            if explanation.get("measurable") is not True:
                explanation["closure_status"] = "unresolvable"
                continue
            closure = planned.get(explanation_id)
            if closure is not None:
                disposition = closure["disposition"]
                explanation.update(
                    {
                        "closure_status": disposition,
                        "disposition": disposition,
                        "expected_value": closure["expected_value"],
                        "required_check": closure["required_check"],
                        "verification": deepcopy(closure["verification"]),
                    }
                )
                continue
            disposition = explanation.get("disposition")
            if disposition not in {"ruled_out", "weakened", "supported"}:
                raise ValueError(
                    f"measurable explanation {explanation_id} has no closure plan"
                )
            explanation["closure_status"] = disposition
            explanation.setdefault(
                "required_check",
                "Re-execute the existing diagnostic verification.",
            )
    return merged


def _critic_closure_targets(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
) -> list[dict[str, Any]]:
    insight_by_title = {
        insight.get("title"): (index, insight)
        for index, insight in enumerate(payload.get("insights", ()))
        if isinstance(insight, Mapping)
    }
    targets = []
    for review in critic.get("reviewed_insights", ()):
        if not isinstance(review, Mapping):
            continue
        bound = insight_by_title.get(review.get("title"))
        if bound is None:
            continue
        insight_index, insight = bound
        challenges = review.get("challenges")
        resolutions = review.get("resolutions")
        if not isinstance(challenges, list) or not isinstance(resolutions, list):
            continue
        resolution_by_index = {
            resolution.get("challenge_index"): resolution
            for resolution in resolutions
            if isinstance(resolution, Mapping)
        }
        assessment = insight.get("diagnostic_assessment")
        explanations = (
            assessment.get("explanations")
            if isinstance(assessment, Mapping)
            else None
        )
        if not isinstance(explanations, list):
            continue
        explanation_inventory = []
        for explanation_index, explanation in enumerate(explanations):
            if not isinstance(explanation, Mapping):
                continue
            explanation_inventory.append(
                {
                    "explanation": explanation.get("explanation"),
                    "explanation_id": explanation.get("explanation_id")
                    or (
                        f"insight-{insight_index + 1}-"
                        f"explanation-{explanation_index + 1}"
                    ),
                    "measurable": explanation.get("measurable"),
                    "disposition": explanation.get("disposition"),
                }
            )
        for challenge_index, challenge in enumerate(challenges):
            if not isinstance(challenge, Mapping):
                continue
            resolution = resolution_by_index.get(challenge_index)
            if (
                not isinstance(resolution, Mapping)
                or resolution.get("status") != "gated"
                or challenge.get("severity") not in {"material", "blocking"}
            ):
                continue
            targets.append(
                {
                    "assessment": challenge.get("assessment"),
                    "challenge_id": challenge.get("id"),
                    "challenge_type": challenge.get("type"),
                    "explanations": explanation_inventory,
                    "insight_title": review.get("title"),
                    "required_changes": review.get("required_changes"),
                }
            )
    return targets


def build_critic_closure_prompt(
    sources: Mapping[str, Path],
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
) -> str:
    """Build a bounded source-aware task for gated critic challenges."""

    targets = _critic_closure_targets(payload, critic)
    return f"""\
Test only the exact gated material or blocking critic challenges below.
Return exactly critic_closure_plans: list with exactly one plan per challenge.

AUTHORITATIVE SOURCE IDENTITIES
{_source_lines(sources)}

EXACT CRITIC CLOSURE TARGETS
{_compact_json({"targets": targets})}

AUTHORITATIVE DISCOVERY PAYLOAD
{_compact_json(payload)}

CRITIC CLOSURE CONTRACT
- Preserve challenge_id exactly and select exactly one existing explanation_id
  from the same insight. Do not add or rewrite explanations.
- Supply a substantive required_check, final disposition of ruled_out,
  weakened, or supported, finite numeric expected_value, and verification.
- Verification must use one self-contained aggregate DuckDB SQL query and an
  exact alias-to-source identity mapping containing only identities above.
- Do not weaken, remove, rename, or resolve a critic challenge by assertion.
  The later critic rerun decides whether executed evidence resolves it.
- Include aggregate evidence only: no raw records, personal identifiers,
  contact details, free-text messages, transcripts, subjects, or descriptions.
- Call SUBMIT immediately after constructing the bounded native partial.
"""


def merge_critic_closure_plan(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
    plan: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    """Reopen and close only explanations selected for eligible challenges."""

    _validate_partial(plan, CRITIC_CLOSURE_OUTPUTS, "critic closure plan")
    targets = _critic_closure_targets(payload, critic)
    target_by_id = {
        target["challenge_id"]: target
        for target in targets
        if isinstance(target.get("challenge_id"), str)
    }
    plans = plan["critic_closure_plans"]
    if len(plans) != len(target_by_id):
        raise ValueError(
            "critic closure plan must contain exactly one plan per eligible gated challenge"
        )
    authorized = frozenset(sources)
    planned_by_challenge: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(plans):
        label = f"critic_closure_plans[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        challenge_id = item.get("challenge_id")
        target = target_by_id.get(challenge_id)
        if target is None or challenge_id in planned_by_challenge:
            raise ValueError(f"{label} must reference an eligible gated challenge")
        explanation_ids = {
            explanation["explanation_id"]
            for explanation in target["explanations"]
        }
        if item.get("explanation_id") not in explanation_ids:
            raise ValueError(f"{label} must reference an existing explanation")
        if item.get("disposition") not in {
            "ruled_out",
            "weakened",
            "supported",
        }:
            raise ValueError(f"{label} disposition is invalid")
        if (
            not isinstance(item.get("required_check"), str)
            or not item["required_check"].strip()
        ):
            raise ValueError(f"{label} required_check must be non-empty")
        expected_value = item.get("expected_value")
        if (
            type(expected_value) not in {int, float}
            or not math.isfinite(expected_value)
        ):
            raise ValueError(f"{label} expected_value must be finite numeric")
        verification = item.get("verification")
        if (
            not isinstance(verification, dict)
            or verification.get("method") != "sql"
            or not isinstance(verification.get("expression"), str)
            or not verification["expression"].strip()
        ):
            raise ValueError(f"{label} verification must be executable SQL")
        declared_sources = verification.get("sources")
        if not isinstance(declared_sources, dict) or not declared_sources:
            raise ValueError(f"{label} verification sources are required")
        if any(identity not in authorized for identity in declared_sources.values()):
            raise ValueError(f"{label} must use an authoritative source identity")
        planned_by_challenge[challenge_id] = item

    merged = deepcopy(dict(payload))
    merged["contract_version"] = 3
    title_to_insight = {
        insight.get("title"): (index, insight)
        for index, insight in enumerate(merged.get("insights", ()))
        if isinstance(insight, dict)
    }
    for challenge_id, item in planned_by_challenge.items():
        target = target_by_id[challenge_id]
        insight_index, insight = title_to_insight[target["insight_title"]]
        explanations = insight["diagnostic_assessment"]["explanations"]
        target_id = item["explanation_id"]
        selected = None
        for explanation_index, explanation in enumerate(explanations):
            explanation_id = explanation.get("explanation_id") or (
                f"insight-{insight_index + 1}-"
                f"explanation-{explanation_index + 1}"
            )
            explanation["explanation_id"] = explanation_id
            if explanation_id == target_id:
                selected = explanation
        if selected is None:
            raise ValueError("critic closure target explanation disappeared")
        selected.pop("limitation", None)
        selected.update(
            {
                "closure_status": item["disposition"],
                "critic_challenge_id": challenge_id,
                "disposition": item["disposition"],
                "expected_value": item["expected_value"],
                "measurable": True,
                "required_check": item["required_check"],
                "verification": deepcopy(item["verification"]),
            }
        )

    for insight_index, insight in enumerate(merged.get("insights", ())):
        assessment = insight.get("diagnostic_assessment")
        explanations = (
            assessment.get("explanations")
            if isinstance(assessment, dict)
            else ()
        )
        for explanation_index, explanation in enumerate(explanations):
            if not isinstance(explanation, dict):
                continue
            explanation.setdefault(
                "explanation_id",
                f"insight-{insight_index + 1}-explanation-{explanation_index + 1}",
            )
            if explanation.get("measurable") is False:
                explanation["closure_status"] = "unresolvable"
            elif explanation.get("disposition") in {
                "ruled_out",
                "weakened",
                "supported",
            }:
                explanation["closure_status"] = explanation["disposition"]
                explanation.setdefault(
                    "required_check",
                    "Re-execute the existing diagnostic verification.",
                )
            else:
                explanation["closure_status"] = "pending"
                explanation.setdefault(
                    "required_check",
                    "Execute the declared diagnostic alternative check.",
                )
    return merged


def _action_synthesis_targets(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload_titles = {
        insight.get("title")
        for insight in payload.get("insights", ())
        if isinstance(insight, Mapping)
        and isinstance(insight.get("title"), str)
    }
    portfolio_blocked_titles = {
        title
        for challenge in critic.get("portfolio_challenges", ())
        if isinstance(challenge, Mapping)
        and challenge.get("severity") == "blocking"
        for title in challenge.get("affected_insight_titles", ())
        if isinstance(title, str)
    }
    deferred_action_bars = {
        title
        for check in critic.get("checks_performed", ())
        if isinstance(check, Mapping)
        and check.get("status") == "deferred"
        and check.get("severity") in {"material", "blocking"}
        for title in check.get("affected_insight_titles", payload_titles)
        if isinstance(title, str)
    }
    review_by_title = {
        review.get("title"): review
        for review in critic.get("reviewed_insights", ())
        if isinstance(review, Mapping)
    }
    targets = []
    for insight in payload.get("insights", ()):
        if not isinstance(insight, Mapping):
            continue
        title = insight.get("title")
        review = review_by_title.get(title)
        assessment = insight.get("diagnostic_assessment")
        readiness = (
            assessment.get("decision_readiness")
            if isinstance(assessment, Mapping)
            else None
        )
        action = insight.get("action")
        if (
            title in portfolio_blocked_titles
            or title in deferred_action_bars
            or not isinstance(review, Mapping)
            or review.get("verdict") != "approve"
            or review.get("synthesis_eligible") is not True
            or review.get("required_changes")
            or readiness != "act_ready"
            or not isinstance(action, Mapping)
            or action.get("kind") != "diagnostic"
        ):
            continue
        resolutions = review.get("resolutions")
        challenges = review.get("challenges")
        if not isinstance(resolutions, list) or not isinstance(challenges, list):
            continue
        resolved_by_index = {
            resolution.get("challenge_index"): resolution.get("status")
            for resolution in resolutions
            if isinstance(resolution, Mapping)
        }
        if "gated" in resolved_by_index.values():
            continue
        if any(
            challenge.get("severity") in {"material", "blocking"}
            and resolved_by_index.get(index) not in {"resolved", "downgraded"}
            for index, challenge in enumerate(challenges)
            if isinstance(challenge, Mapping)
        ):
            continue
        targets.append(
            {
                "current_action": action,
                "decision_effect": review.get("decision_effect"),
                "title": title,
            }
        )
    return targets


def build_action_synthesis_prompt(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
) -> str:
    """Build a bounded action rewrite for critic-approved ready findings."""

    targets = _action_synthesis_targets(payload, critic)
    return f"""\
Create exactly one action update for each eligible evidence-closed finding
below. Return exactly action_updates: list.

EXACT ELIGIBLE TARGETS
{_compact_json({"targets": targets})}

ACTION UPDATE CONTRACT
- Preserve each title exactly and update no other finding field.
- action must contain exactly owner, segment, decision, target, time_horizon,
  and kind. Every text field must be substantive and kind must be program.
- The action must be bounded by the measured population and critic-approved
  decision effect. Do not claim causality or invent a numeric target.
- Do not include raw records, personal identifiers, contact details,
  communications, transcripts, or source text.
- Call SUBMIT immediately with the native partial.
"""


def merge_action_synthesis(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only actions for exact critic-approved act-ready findings."""

    _validate_partial(updates, ACTION_SYNTHESIS_OUTPUTS, "action synthesis")
    eligible = {
        target["title"]
        for target in _action_synthesis_targets(payload, critic)
    }
    action_updates = updates["action_updates"]
    if not eligible or len(action_updates) != len(eligible):
        raise ValueError(
            "action synthesis has no eligible findings or incomplete coverage"
        )
    by_title: dict[str, dict[str, Any]] = {}
    required_action_fields = {
        "decision",
        "kind",
        "owner",
        "segment",
        "target",
        "time_horizon",
    }
    for index, item in enumerate(action_updates):
        label = f"action_updates[{index}]"
        if not isinstance(item, dict) or set(item) != {"title", "action"}:
            raise ValueError(f"{label} must contain exactly title and action")
        title = item["title"]
        if title not in eligible or title in by_title:
            raise ValueError(f"{label} must reference an exact eligible title")
        action = item["action"]
        if not isinstance(action, dict) or set(action) != required_action_fields:
            raise ValueError(f"{label} action fields are invalid")
        if action.get("kind") != "program":
            raise ValueError(f"{label} action kind must be program")
        if any(
            not isinstance(action[field], str) or not action[field].strip()
            for field in required_action_fields - {"kind"}
        ):
            raise ValueError(f"{label} action text must be substantive")
        by_title[title] = deepcopy(action)
    if set(by_title) != eligible:
        raise ValueError("action synthesis must cover every exact eligible title")

    merged = deepcopy(dict(payload))
    for insight in merged.get("insights", ()):
        if isinstance(insight, dict) and insight.get("title") in by_title:
            insight["action"] = by_title[insight["title"]]
    return merged


def build_insight_repair_prompt(
    sources: Mapping[str, Path],
    research: Mapping[str, Any],
    scaffold: Mapping[str, Any],
    current_insights: Mapping[str, Any],
    verifier_error: str,
) -> str:
    """Build a bounded repair brief from the latest verifier failure."""

    compact_sources = _compact_json(
        {identity: str(path) for identity, path in sources.items()}
    )
    return f"""\
Repair only the insights portion of deep_insight_discovery contract v2.
Return exactly insights: list and call SUBMIT immediately.

AUTHORITATIVE SOURCES JSON
{compact_sources}

STAGE 1 RESEARCH JSON
{_compact_json(research)}

EXACT CONTRACT SCAFFOLD JSON
{_compact_json(scaffold)}

CURRENT INSIGHTS JSON
{_compact_json(current_insights)}

EXACT PORTABLE VERIFIER ERROR
{verifier_error}

REPAIR RULES
- Make the minimum contract correction required by the exact verifier error.
- Preserve verified facts, titles, dimensions, SQL aliases, and evidence unless
  implicated by that error.
- Do not broaden exploration. Do not invent evidence.
- Return no fields other than the exactly typed insights: list.
- Call SUBMIT immediately after constructing the corrected native partial.
"""


def extract_insight_index(error: str, insight_count: int) -> int | None:
    """Extract a valid zero-based insight index from a verifier error."""

    match = re.search(r"\binsight\s+(\d+)\b", str(error), flags=re.IGNORECASE)
    if match is None:
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < insight_count else None


_AUDIT_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_AUDIT_PATH = (
    r"[A-Za-z_]\w*(?:\[\d+\])?"
    r"(?:\.[A-Za-z_]\w*(?:\[\d+\])?)*"
)
_AUDIT_MISMATCH_RE = re.compile(
    rf"(?P<path>{_AUDIT_PATH}): expected "
    rf"(?P<expected>{_AUDIT_NUMBER}), actual (?P<actual>{_AUDIT_NUMBER})"
)
_PATH_TOKEN_RE = re.compile(r"([A-Za-z_]\w*)|\[(\d+)\]")


def parse_host_audit_mismatch(error: str) -> HostAuditMismatch | None:
    """Parse only the host audit's exact, safe numeric-mismatch format."""

    if not isinstance(error, str):
        return None
    match = _AUDIT_MISMATCH_RE.fullmatch(error)
    if match is None:
        return None
    expected = float(match.group("expected"))
    actual = float(match.group("actual"))
    if not math.isfinite(expected) or not math.isfinite(actual):
        return None
    return HostAuditMismatch(match.group("path"), expected, actual)


def _path_tokens(path: str) -> tuple[str | int, ...]:
    tokens: list[str | int] = []
    position = 0
    for match in _PATH_TOKEN_RE.finditer(path):
        if match.start() != position:
            raise ValueError(f"unsupported host audit target path: {path}")
        tokens.append(
            match.group(1) if match.group(1) is not None else int(match.group(2))
        )
        position = match.end()
        if position < len(path) and path[position] == ".":
            position += 1
    if position != len(path):
        raise ValueError(f"unsupported host audit target path: {path}")
    return tuple(tokens)


def _resolve_path(value: Any, tokens: tuple[str | int, ...]) -> Any:
    current = value
    for token in tokens:
        if isinstance(token, str):
            if not isinstance(current, Mapping) or token not in current:
                raise ValueError(f"host audit target does not exist at {token}")
            current = current[token]
        else:
            if not isinstance(current, list) or not 0 <= token < len(current):
                raise ValueError(f"host audit target index is out of range: {token}")
            current = current[token]
    return current


def _without_verification(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_verification(child)
            for key, child in value.items()
            if key != "verification"
        }
    if isinstance(value, list):
        return [_without_verification(child) for child in value]
    return deepcopy(value)


_AUDIT_IMMUTABLE_KEYS = frozenset(
    {"verification", "sql", "expression", "sources", "method"}
)


def _without_audit_immutables(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_audit_immutables(child)
            for key, child in value.items()
            if key not in _AUDIT_IMMUTABLE_KEYS
        }
    if isinstance(value, list):
        return [_without_audit_immutables(child) for child in value]
    return deepcopy(value)


def _numeric_text_variants(number: float) -> tuple[str, ...]:
    number = float(number)
    variants = {format(number, "g"), format(number, ",g")}
    if number.is_integer():
        variants.add(f"{int(number):,}")
    percentage = number * 100
    variants.add(f"{percentage:g}%")
    variants.add(f"{percentage:,g}%")
    return tuple(sorted(variants, key=len, reverse=True))


def _contains_numeric_value(text: str, number: float) -> bool:
    return any(variant in text for variant in _numeric_text_variants(number))


def _audit_target(
    insights: Mapping[str, Any], mismatch: HostAuditMismatch
) -> tuple[int, tuple[str | int, ...], str, dict[str, type]]:
    tokens = _path_tokens(mismatch.path)
    if (
        len(tokens) < 2
        or tokens[0] != "insights"
        or not isinstance(tokens[1], int)
    ):
        raise ValueError(
            f"host audit mismatch is not under an insight: {mismatch.path}"
        )
    insight_index = tokens[1]
    insight_list = insights.get("insights")
    if not isinstance(insight_list, list) or not 0 <= insight_index < len(insight_list):
        raise ValueError(
            f"host audit insight index is out of range: {mismatch.path}"
        )
    relative = tokens[2:]
    for index in range(len(relative) - 1):
        if (
            relative[index] == "supporting_claims"
            and isinstance(relative[index + 1], int)
        ):
            target = relative[: index + 2]
            return (
                insight_index,
                target,
                "supporting_claim",
                AUDIT_SUPPORTING_CLAIM_OUTPUTS,
            )
    if relative and relative[0] == "metric_spec":
        statement = insight_list[insight_index].get("statement")
        if (
            isinstance(statement, str)
            and _contains_numeric_value(statement, mismatch.expected)
        ):
            return (
                insight_index,
                ("metric_spec",),
                "metric_spec",
                AUDIT_METRIC_AND_STATEMENT_OUTPUTS,
            )
        return (
            insight_index,
            ("metric_spec",),
            "metric_spec",
            AUDIT_METRIC_SPEC_OUTPUTS,
        )
    leaf = relative
    while leaf:
        try:
            if isinstance(_resolve_path(insight_list[insight_index], leaf), Mapping):
                return insight_index, leaf, "audit_leaf", AUDIT_LEAF_OUTPUTS
        except ValueError:
            pass
        leaf = leaf[:-1]
    raise ValueError(f"host audit target is not a repairable object: {mismatch.path}")


def _candidate_audit_target(
    scaffold: Mapping[str, Any], mismatch: HostAuditMismatch
) -> tuple[int, int, dict[str, Any]]:
    tokens = _path_tokens(mismatch.path)
    if (
        len(tokens) != 6
        or tokens[0] != "candidates"
        or not isinstance(tokens[1], int)
        or tokens[2:] != (
            "rejection_evidence",
            "verification",
            "components",
            tokens[5],
        )
        or not isinstance(tokens[5], int)
    ):
        raise ValueError(
            "host audit mismatch is not an exact rejected-candidate component: "
            f"{mismatch.path}"
        )
    candidate_index = tokens[1]
    component_index = tokens[5]
    candidates = scaffold.get("candidates")
    if (
        not isinstance(candidates, list)
        or not 0 <= candidate_index < len(candidates)
    ):
        raise ValueError(
            f"host audit candidate index is out of range: {mismatch.path}"
        )
    candidate = candidates[candidate_index]
    if not isinstance(candidate, Mapping):
        raise ValueError(f"host audit candidate must be an object: {mismatch.path}")
    if candidate.get("disposition") != "rejected":
        raise ValueError(
            f"host audit mismatch targets a promoted candidate: {mismatch.path}"
        )
    if candidate.get("rejection_type") != "quantitative":
        raise ValueError(
            f"host audit candidate rejection is not quantitative: {mismatch.path}"
        )
    components = _resolve_path(
        candidate,
        ("rejection_evidence", "verification", "components"),
    )
    if (
        not isinstance(components, list)
        or not 0 <= component_index < len(components)
    ):
        raise ValueError(
            f"host audit candidate component index is out of range: {mismatch.path}"
        )
    component = components[component_index]
    if not isinstance(component, Mapping):
        raise ValueError(
            f"host audit candidate component must be an object: {mismatch.path}"
        )
    expected_values = _expected_values(component)
    if not expected_values:
        raise ValueError(
            f"host audit candidate component is not quantitative: {mismatch.path}"
        )
    return candidate_index, component_index, dict(component)


def build_host_candidate_audit_repair_prompt(
    scaffold: Mapping[str, Any], mismatch: HostAuditMismatch
) -> str:
    """Build a minimal repair brief for one rejected-candidate component."""

    _, _, component = _candidate_audit_target(scaffold, mismatch)
    return f"""\
Repair one quantitative rejected-candidate evidence component. Return exactly
rejection_component: dict and SUBMIT immediately.

EXACT CONTRACT PATH
{mismatch.path}

PRIOR EXPECTED
{mismatch.expected!r}

AUTHORITATIVE EXECUTED ACTUAL
{mismatch.actual!r}

IMPLICATED COMPONENT (verification/SQL/source fields are immutable and omitted)
{_compact_json(_without_audit_immutables(component))}

RULES
- Update expected_value to authoritative actual {mismatch.actual!r}.
- Update numeric explanatory prose in this component when it contains the
  prior expected value.
- Do not add, remove, or modify SQL, expression, sources, method, or nested
  verification.
- Return exactly rejection_component: dict; no wrapper, explanation, or
  other fields.
"""


def build_host_audit_repair_prompt(
    current_insight: Mapping[str, Any],
    mismatch: HostAuditMismatch,
    target_tokens: tuple[str | int, ...],
    output_name: str,
    include_statement: bool = False,
) -> str:
    """Build a minimal numeric repair brief for one host-audited leaf."""

    leaf = _resolve_path(current_insight, target_tokens)
    context = {
        key: current_insight[key]
        for key in ("title", "statement", "interpretation")
        if key in current_insight
    }
    output_contract = (
        "metric_spec: dict and statement: str"
        if include_statement
        else f"{output_name}: dict"
    )
    return f"""\
Repair one host-audited numeric leaf. Return exactly {output_contract} and
SUBMIT immediately.

EXACT CONTRACT PATH
{mismatch.path}

PRIOR EXPECTED
{mismatch.expected!r}

AUTHORITATIVE EXECUTED ACTUAL
{mismatch.actual!r}

IMPLICATED LEAF (verification is immutable and omitted)
{_compact_json(_without_verification(leaf))}

MINIMAL PARENT PROSE
{_compact_json(context)}

RULES
- Reconcile every outer and nested expected_value in the returned leaf with
  authoritative actual {mismatch.actual!r}.
- Update quantitative prose in the leaf. Update statement only when it
  contains the old primary value and statement is part of the output contract.
- Do not add, remove, or modify SQL, verification, or source mappings.
- Do not change the title or unrelated prose, claims, or metrics.
- Return exactly {output_contract}; no wrapper, explanation, or other fields.
"""


def _expected_values(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "expected_value":
                if (
                    isinstance(child, bool)
                    or not isinstance(child, (int, float))
                    or not math.isfinite(float(child))
                ):
                    raise ValueError("audit repair expected_value must be finite numeric")
                values.append(float(child))
            else:
                values.extend(_expected_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_expected_values(child))
    return values


def _verification_values(value: Any, path: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "verification":
                values[child_path] = child
            else:
                values.update(_verification_values(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            values.update(_verification_values(child, f"{path}[{index}]"))
    return values


def _overlay_preserving_audit_immutables(
    old: Mapping[str, Any], proposed: Mapping[str, Any], path: str = ""
) -> dict[str, Any]:
    merged = deepcopy(dict(old))
    for key, value in proposed.items():
        key_path = f"{path}.{key}" if path else key
        if key in _AUDIT_IMMUTABLE_KEYS:
            if key not in old or value != old[key]:
                raise ValueError(
                    "host audit repair attempted to alter immutable "
                    f"SQL/source/expression field at {key_path}"
                )
            continue
        old_value = old.get(key)
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[key] = _overlay_preserving_audit_immutables(
                old_value, value, key_path
            )
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_updated_numeric_prose(
    old: Any,
    repaired: Any,
    mismatch: HostAuditMismatch,
    path: str = "",
) -> None:
    if isinstance(old, Mapping) and isinstance(repaired, Mapping):
        for key, old_value in old.items():
            if key in _AUDIT_IMMUTABLE_KEYS or key not in repaired:
                continue
            child_path = f"{path}.{key}" if path else key
            _validate_updated_numeric_prose(
                old_value, repaired[key], mismatch, child_path
            )
    elif isinstance(old, list) and isinstance(repaired, list):
        for index, (old_value, repaired_value) in enumerate(zip(old, repaired)):
            _validate_updated_numeric_prose(
                old_value,
                repaired_value,
                mismatch,
                f"{path}[{index}]",
            )
    elif (
        isinstance(old, str)
        and _contains_numeric_value(old, mismatch.expected)
        and (
            not isinstance(repaired, str)
            or _contains_numeric_value(repaired, mismatch.expected)
            or not _contains_numeric_value(repaired, mismatch.actual)
        )
    ):
        raise ValueError(
            "host audit repair did not update numeric explanatory prose at "
            f"{path}"
        )


def merge_host_candidate_audit_repair(
    scaffold: Mapping[str, Any],
    mismatch: HostAuditMismatch,
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge exactly one validated rejected-candidate evidence component."""

    candidate_index, component_index, old_component = _candidate_audit_target(
        scaffold, mismatch
    )
    if set(repair) != set(AUDIT_REJECTION_COMPONENT_OUTPUTS):
        raise ValueError(
            "host candidate audit repair returned fields other than "
            "rejection_component"
        )
    proposed = repair["rejection_component"]
    if not isinstance(proposed, Mapping):
        raise ValueError("host audit repair rejection_component must be dict")
    old_expected = _expected_values(old_component)
    if not any(
        math.isclose(value, mismatch.expected, rel_tol=1e-12, abs_tol=1e-12)
        for value in old_expected
    ):
        raise ValueError(
            f"host audit expected value is absent from target: {mismatch.path}"
        )
    component = _overlay_preserving_audit_immutables(old_component, proposed)
    repaired_expected = _expected_values(component)
    if not repaired_expected or any(
        not math.isclose(value, mismatch.actual, rel_tol=1e-12, abs_tol=1e-12)
        for value in repaired_expected
    ):
        raise ValueError(
            "host audit repair did not reconcile every expected_value with "
            f"authoritative actual {mismatch.actual:g}"
        )
    _validate_updated_numeric_prose(old_component, component, mismatch)
    merged = deepcopy(dict(scaffold))
    rejection_evidence = merged["candidates"][candidate_index][
        "rejection_evidence"
    ]
    rejection_evidence["verification"]["components"][component_index] = component
    component_name = old_component.get("name")
    mirrored_value = rejection_evidence.get(component_name)
    if (
        isinstance(component_name, str)
        and type(mirrored_value) in {int, float}
        and math.isclose(
            float(mirrored_value),
            mismatch.expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        rejection_evidence[component_name] = mismatch.actual
    return merged


def merge_host_audit_repair(
    current_insight: Mapping[str, Any],
    mismatch: HostAuditMismatch,
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and exactly merge one model-updated host-audited object."""

    _, target_tokens, output_name, outputs = _audit_target(
        {"insights": [current_insight]},
        HostAuditMismatch(
            re.sub(r"^insights\[\d+\]", "insights[0]", mismatch.path),
            mismatch.expected,
            mismatch.actual,
        ),
    )
    if set(repair) != set(outputs):
        raise ValueError(
            "host audit repair returned fields other than "
            + ", ".join(outputs)
        )
    old_leaf = _resolve_path(current_insight, target_tokens)
    candidate = repair[output_name]
    if not isinstance(candidate, Mapping):
        raise ValueError(f"host audit repair {output_name} must be dict")
    old_expected = _expected_values(old_leaf)
    if not any(
        math.isclose(value, mismatch.expected, rel_tol=1e-12, abs_tol=1e-12)
        for value in old_expected
    ):
        raise ValueError(
            f"host audit expected value is absent from target: {mismatch.path}"
        )
    candidate_expected = _expected_values(candidate)
    if not candidate_expected or any(
        not math.isclose(
            value, mismatch.actual, rel_tol=1e-12, abs_tol=1e-12
        )
        for value in candidate_expected
    ):
        raise ValueError(
            "host audit repair did not reconcile every expected_value with "
            f"authoritative actual {mismatch.actual:g}"
        )
    old_verifications = _verification_values(old_leaf)
    candidate_verifications = _verification_values(candidate)
    if candidate_verifications and candidate_verifications != old_verifications:
        raise ValueError("host audit repair attempted to modify verification or SQL")
    candidate = deepcopy(candidate)
    if (
        len(target_tokens) >= 3
        and target_tokens[0] == "diagnostic_assessment"
        and target_tokens[1] == "explanations"
        and isinstance(target_tokens[2], int)
        and isinstance(old_leaf, Mapping)
    ):
        for key in ("explanation", "measurable", "disposition", "limitation"):
            if key in old_leaf:
                candidate[key] = deepcopy(old_leaf[key])
            else:
                candidate.pop(key, None)
    if not candidate_verifications:
        def restore_verification(old: Any, new: Any) -> None:
            if not isinstance(old, Mapping) or not isinstance(new, dict):
                return
            for key, child in old.items():
                if key == "verification":
                    new[key] = deepcopy(child)
                elif key in new:
                    if isinstance(child, list) and isinstance(new[key], list):
                        for old_item, new_item in zip(child, new[key]):
                            restore_verification(old_item, new_item)
                    else:
                        restore_verification(child, new[key])

        restore_verification(old_leaf, candidate)
        if _verification_values(candidate) != old_verifications:
            raise ValueError(
                "host audit repair omitted structure containing verification"
            )
    merged = deepcopy(current_insight)
    parent = merged
    for token in target_tokens[:-1]:
        parent = parent[token]
    parent[target_tokens[-1]] = candidate
    if "statement" in outputs:
        statement = repair["statement"]
        if not isinstance(statement, str):
            raise ValueError("host audit repair statement must be str")
        old_statement = current_insight.get("statement")
        if (
            not isinstance(old_statement, str)
            or not _contains_numeric_value(old_statement, mismatch.expected)
            or _contains_numeric_value(statement, mismatch.expected)
            or not _contains_numeric_value(statement, mismatch.actual)
        ):
            raise ValueError(
                "host audit repair statement did not replace the old primary "
                "value with the authoritative actual"
            )
        merged["statement"] = statement
    return merged


def merge_targeted_insight_repair(
    current_insight: Mapping[str, Any],
    repaired_insight: Mapping[str, Any],
    verifier_error: str,
) -> dict[str, Any]:
    """Merge only the verifier-addressed leaf from a repaired insight."""

    merged = deepcopy(current_insight)
    supporting_match = re.search(
        r"\bsupporting claim\s+(\d+)\b",
        str(verifier_error),
        flags=re.IGNORECASE,
    )
    if supporting_match is not None:
        index = int(supporting_match.group(1)) - 1
        current_claims = merged.get("supporting_claims")
        repaired_claims = repaired_insight.get("supporting_claims")
        if (
            not isinstance(current_claims, list)
            or not isinstance(repaired_claims, list)
            or not 0 <= index < len(current_claims)
            or not 0 <= index < len(repaired_claims)
        ):
            raise ValueError(
                "targeted insight repair did not return the addressed "
                "supporting claim"
            )
        current_claims[index] = deepcopy(repaired_claims[index])
        return merged

    explanation_match = re.search(
        r"\bdiagnostic explanation\s+(\d+)\b",
        str(verifier_error),
        flags=re.IGNORECASE,
    )
    if explanation_match is not None:
        index = int(explanation_match.group(1)) - 1
        current_assessment = merged.get("diagnostic_assessment")
        repaired_assessment = repaired_insight.get("diagnostic_assessment")
        current_explanations = (
            current_assessment.get("explanations")
            if isinstance(current_assessment, dict)
            else None
        )
        repaired_explanations = (
            repaired_assessment.get("explanations")
            if isinstance(repaired_assessment, Mapping)
            else None
        )
        if (
            not isinstance(current_explanations, list)
            or not isinstance(repaired_explanations, list)
            or not 0 <= index < len(current_explanations)
            or not 0 <= index < len(repaired_explanations)
        ):
            raise ValueError(
                "targeted insight repair did not return the addressed "
                "diagnostic explanation"
            )
        current_explanations[index] = deepcopy(repaired_explanations[index])
        return merged

    if re.search(r"\bmetric_spec\b", str(verifier_error), flags=re.IGNORECASE):
        metric_spec = repaired_insight.get("metric_spec")
        if not isinstance(metric_spec, Mapping):
            raise ValueError(
                "targeted insight repair did not return the addressed metric_spec"
            )
        merged["metric_spec"] = deepcopy(metric_spec)
        return merged

    return deepcopy(repaired_insight)


def build_targeted_insight_repair_prompt(
    sources: Mapping[str, Path],
    scaffold: Mapping[str, Any],
    current_insight: Mapping[str, Any],
    verifier_error: str,
) -> str:
    """Build a compact repair brief for one verifier-identified insight."""

    title = current_insight.get("title")
    matching_candidate = next(
        (
            candidate
            for candidate in scaffold.get("candidates", [])
            if isinstance(candidate, Mapping)
            and candidate.get("disposition") == "promoted"
            and candidate.get("promoted_as") == title
        ),
        None,
    )
    candidate_section = (
        _compact_json(matching_candidate)
        if matching_candidate is not None
        else "not found"
    )
    source_identities = json.dumps(
        list(sources), separators=(",", ":"), ensure_ascii=False
    )
    return f"""\
Repair one deep_insight_discovery contract v2 insight. Return exactly one
native insight: dict and SUBMIT immediately.

AUTHORITATIVE SOURCE IDENTITIES
{source_identities}

MATCHING PROMOTED CANDIDATE
{candidate_section}

CURRENT INSIGHT
{_compact_json(current_insight)}

EXACT PORTABLE VERIFIER ERROR
{verifier_error}

RULES
- Make only the minimum correction required by the exact error.
- Keep the exact title and discovery dimensions_tested preserved.
- Preserve valid evidence; no invented evidence or broad exploration.
- For statement errors, express only metric_spec.expected_value as the primary
  fact. Move threshold, numerator, and denominator details to
  supporting_claims only when they are already evidenced.
- Return no wrapper or other fields: exactly insight: dict.
- SUBMIT immediately.
"""


def build_targeted_statement_repair_prompt(
    current_insight: Mapping[str, Any], verifier_error: str
) -> str:
    """Build a field-only repair brief for a measured-claim statement error."""

    context = {
        "title": current_insight.get("title"),
        "primary_metric": {
            "type": (
                current_insight.get("metric_spec", {}).get("type")
                if isinstance(current_insight.get("metric_spec"), Mapping)
                else None
            ),
            "expected_value": (
                current_insight.get("metric_spec", {}).get("expected_value")
                if isinstance(current_insight.get("metric_spec"), Mapping)
                else None
            ),
        },
    }
    return f"""\
Repair only the statement field of one deep_insight_discovery contract v2
insight. Return exactly one native statement: str and SUBMIT immediately.

RELEVANT INSIGHT FIELDS
{_compact_json(context)}

EXACT PORTABLE VERIFIER ERROR
{verifier_error}

RULES
- State exactly one measured fact: the primary metric_spec.expected_value.
- Use exactly one numeric literal: the supplied primary expected_value, rounded
  only as needed for readable presentation.
- Do not include metric component values or any supporting-claim values.
- Do not repeat component, threshold, numerator, denominator, sample-size, or
  other quantitative facts in the statement.
- Preserve the meaning, population, and direction of the primary metric.
- Do not invent evidence or alter any other insight field.
- Return no wrapper or other fields: exactly statement: str.
- SUBMIT immediately.
"""


def build_targeted_interpretation_repair_prompt(
    current_insight: Mapping[str, Any], verifier_error: str
) -> str:
    """Build a field-only repair brief for quantitative interpretation prose."""

    context = {
        "title": current_insight.get("title"),
        "statement": current_insight.get("statement"),
        "interpretation": current_insight.get("interpretation"),
        "supporting_claims": current_insight.get("supporting_claims"),
    }
    return f"""\
Repair only the interpretation field of one deep_insight_discovery contract v2
insight. Return exactly one native interpretation: str and SUBMIT immediately.

RELEVANT INSIGHT FIELDS
{_compact_json(context)}

EXACT PORTABLE VERIFIER ERROR
{verifier_error}

RULES
- Preserve the analytical meaning, direction, mechanism, and decision relevance.
- Include no quantitative facts, counts, percentages, currency values, numeric
  thresholds, or word-number ratios; those remain in supporting_claims.
- Use no digits and no number words in the returned interpretation, including
  calendar years. Refer generically to pre-break and post-break periods instead.
- Do not alter or repeat the measured statement.
- Do not invent evidence or alter any other insight field.
- Return no wrapper or other fields: exactly interpretation: str.
- SUBMIT immediately.
"""


def is_targeted_statement_error(error: str) -> bool:
    """Return whether a verifier error can be repaired as one statement field."""

    return re.search(
        r"\binsight\s+\d+\s+"
        r"statement\s+must contain one primary measured claim\b",
        str(error),
        flags=re.IGNORECASE,
    ) is not None


def is_targeted_interpretation_error(error: str) -> bool:
    """Return whether a verifier error is isolated to interpretation prose."""

    return re.search(
        r"\binsight\s+\d+\s+"
        r"quantitative facts belong in verified supporting_claims\b",
        str(error),
        flags=re.IGNORECASE,
    ) is not None


def classify_repair_target(error: str) -> str:
    """Route a verifier error to the narrowest safely repairable contract part."""

    lowered = str(error).casefold()
    scaffold_markers = (
        "candidate",
        "candidates ledger",
        "analysis_plan",
        "search_space",
        "kpi_map",
        "dimensions_available",
        "dimensions_deferred",
    )
    if any(marker in lowered for marker in scaffold_markers):
        return "scaffold"
    return "insights"


def build_scaffold_repair_prompt(
    sources: Mapping[str, Path],
    research: Mapping[str, Any],
    current_scaffold: Mapping[str, Any],
    current_insights: Mapping[str, Any],
    verifier_error: str,
) -> str:
    """Build a bounded scaffold repair brief from a verifier failure."""

    compact_sources = _compact_json(
        {identity: str(path) for identity, path in sources.items()}
    )
    return f"""\
Repair only the contract scaffold for deep_insight_discovery contract v2.
Return exactly analysis_plan: dict and candidates: list. Call SUBMIT immediately.

AUTHORITATIVE SOURCES JSON
{compact_sources}

STAGE 1 RESEARCH JSON
{_compact_json(research)}

CURRENT CONTRACT SCAFFOLD JSON
{_compact_json(current_scaffold)}

CURRENT DEPENDENT INSIGHTS JSON
{_compact_json(current_insights)}

EXACT PORTABLE VERIFIER ERROR
{verifier_error}

REPAIR RULES
- Make the minimum contract correction required by the exact verifier error.
- Preserve valid promotions, titles, dimensions, and source-derived facts unless
  implicated by that error.
- For quantitative rejection verification, derived effects must be decomposed
  into source-derived checks or the candidate must be honestly reclassified
  under an allowed rejection type. Never substitute an unrelated metric.
- Do not invent evidence or return insights.
- Return no fields other than the exactly typed analysis_plan and candidates.
- Call SUBMIT immediately after constructing the corrected native partial.
"""


def _validate_partial(
    value: Mapping[str, Any],
    expected: Mapping[str, type],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != set(expected):
        raise ValueError(
            f"{label} must contain exactly " + ", ".join(expected)
        )
    for name, expected_type in expected.items():
        if type(value[name]) is not expected_type:
            raise ValueError(
                f"{label} field {name} must be {expected_type.__name__}"
            )


_SOURCE_IDENTITY = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)
_SQL_RELATION = re.compile(
    r"\b(?:from|join)\s+(?!\()"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*)",
    flags=re.IGNORECASE,
)
_SQL_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[(),]")


def _without_sql_literals_and_comments(expression: str) -> str:
    sanitized = list(expression)
    index = 0
    length = len(expression)
    while index < length:
        if expression.startswith("--", index):
            end = expression.find("\n", index + 2)
            end = length if end < 0 else end
            sanitized[index:end] = " " * (end - index)
            index = end
            continue
        if expression.startswith("/*", index):
            end = expression.find("*/", index + 2)
            end = length if end < 0 else end + 2
            sanitized[index:end] = " " * (end - index)
            index = end
            continue
        delimiter = expression[index]
        if delimiter in {"'", '"', "`", "["}:
            closing = "]" if delimiter == "[" else delimiter
            end = index + 1
            while end < length:
                if expression[end] == closing:
                    if (
                        closing != "]"
                        and end + 1 < length
                        and expression[end + 1] == closing
                    ):
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
            sanitized[index:end] = " " * (end - index)
            index = end
            continue
        if delimiter == "$":
            dollar = re.match(r"\$[A-Za-z_0-9]*\$", expression[index:])
            if dollar:
                marker = dollar.group(0)
                end = expression.find(marker, index + len(marker))
                end = length if end < 0 else end + len(marker)
                sanitized[index:end] = " " * (end - index)
                index = end
                continue
        index += 1
    return "".join(sanitized)


def _cte_names(expression: str) -> set[str]:
    tokens = [match.group(0) for match in _SQL_TOKEN.finditer(expression)]
    lowered = [token.casefold() for token in tokens]
    names: set[str] = set()

    def matching_paren(open_index: int) -> int | None:
        depth = 0
        for token_index in range(open_index, len(tokens)):
            if tokens[token_index] == "(":
                depth += 1
            elif tokens[token_index] == ")":
                depth -= 1
                if depth == 0:
                    return token_index
        return None

    for with_index, token in enumerate(lowered):
        if token != "with":
            continue
        cursor = with_index + 1
        if cursor < len(tokens) and lowered[cursor] == "recursive":
            cursor += 1
        while cursor < len(tokens):
            name = tokens[cursor]
            if not _SOURCE_IDENTITY.fullmatch(name):
                break
            after_name = cursor + 1
            if after_name < len(tokens) and tokens[after_name] == "(":
                after_name = matching_paren(after_name)
                if after_name is None:
                    break
                after_name += 1
            if (
                after_name + 1 >= len(tokens)
                or lowered[after_name] != "as"
                or tokens[after_name + 1] != "("
            ):
                break
            names.add(name.casefold())
            close = matching_paren(after_name + 1)
            if close is None or close + 1 >= len(tokens) or tokens[close + 1] != ",":
                break
            cursor = close + 2
    return names


def _safe_path(path: str, key: Any) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[*]"


def normalize_mechanical_contract(
    payload: Mapping[str, Any], source_names: Any
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Repair deterministic contract consistency without semantic inference."""

    if isinstance(source_names, (str, bytes)):
        raise ValueError("source_names must be a finite collection of identifiers")
    source_paths: dict[str, str | None] = {}
    if isinstance(source_names, Mapping):
        for identity, raw_path in source_names.items():
            if isinstance(raw_path, (str, Path)):
                path_text = str(raw_path)
                source_paths[path_text] = (
                    identity if path_text not in source_paths else None
                )
    try:
        supplied_names = tuple(source_names)
    except TypeError as exc:
        raise ValueError(
            "source_names must be a finite collection of identifiers"
        ) from exc
    if any(
        not isinstance(name, str) or _SOURCE_IDENTITY.fullmatch(name) is None
        for name in supplied_names
    ):
        raise ValueError("source_names must contain only simple SQL identifiers")
    authorized = frozenset(supplied_names)
    normalized = deepcopy(payload)
    changes: list[str] = []

    def record_set(mapping: dict[str, Any], key: str, value: Any, path: str) -> None:
        if mapping.get(key) != value:
            mapping[key] = value
            changes.append(_safe_path(path, key))

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            method = value.get("method")
            expression = value.get("expression")
            declared_sources = value.get("sources")
            if (
                isinstance(method, str)
                and method.casefold() == "sql"
                and isinstance(expression, str)
                and isinstance(declared_sources, dict)
            ):
                for alias, declared_source in tuple(declared_sources.items()):
                    if not isinstance(declared_source, str):
                        continue
                    canonical_identity = source_paths.get(declared_source)
                    if (
                        canonical_identity is not None
                        and declared_source != canonical_identity
                    ):
                        declared_sources[alias] = canonical_identity
                        changes.append(
                            f"{_safe_path(path, 'sources')}.{alias}"
                        )
                sanitized = _without_sql_literals_and_comments(expression)
                ctes = _cte_names(sanitized)
                relations: list[str] = []
                for match in _SQL_RELATION.finditer(sanitized):
                    relation = re.sub(r"\s+", "", match.group(1))
                    if relation.casefold() not in ctes and relation not in relations:
                        relations.append(relation)
                for relation in relations:
                    if (
                        relation in authorized
                        and relation not in declared_sources
                    ):
                        declared_sources[relation] = relation
                        changes.append(
                            f"{_safe_path(path, 'sources')}.{relation}"
                        )

            for key, child in tuple(value.items()):
                visit(child, _safe_path(path, key))

            interpretation = value.get("interpretation")
            if isinstance(interpretation, str):
                prepared_interpretation = re.sub(
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                    r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?"
                    r"[-\s]+(?:19|20)\d{2}\b",
                    "the observed break",
                    interpretation,
                    flags=re.IGNORECASE,
                )
                prepared_interpretation = re.sub(
                    r"\b(?:19|20)\d{2}\b\s*",
                    "",
                    prepared_interpretation,
                )
                sentences = re.split(
                    r"(?<=[.!?])\s+", prepared_interpretation.strip()
                )
                quantitative_words = re.compile(
                    r"\b(?:"
                    r"half\s+(?:of\s+)?(?:the\s+)?|"
                    r"a\s+(?:third|quarter|fifth)\s+of\s+|"
                    r"one\s+in\s+(?:two|three|four|five|six|seven|eight|nine|ten)\b|"
                    r"(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+dozen\b"
                    r")",
                    flags=re.IGNORECASE,
                )
                qualitative_sentences = [
                    sentence
                    for sentence in sentences
                    if sentence
                    and not re.search(
                        r"\d+(?:[,.]\d+)*\s*%?|\$\s*\d", sentence
                    )
                    and quantitative_words.search(sentence) is None
                ]
                qualitative_interpretation = " ".join(qualitative_sentences)
                if (
                    qualitative_interpretation
                    and qualitative_interpretation != interpretation
                ):
                    record_set(
                        value,
                        "interpretation",
                        qualitative_interpretation,
                        path,
                    )

            metric_type = value.get("type")
            components = value.get("components")
            if (
                metric_type in {"delta", "rate", "share", "rate_of_change"}
                and isinstance(components, list)
            ):
                role_values: dict[str, float] = {}
                valid_components = True
                for component in components:
                    if not isinstance(component, dict):
                        valid_components = False
                        break
                    role = component.get("role")
                    expected_value = component.get("expected_value")
                    if (
                        not isinstance(role, str)
                        or type(expected_value) not in {int, float}
                        or not math.isfinite(expected_value)
                        or role in role_values
                    ):
                        valid_components = False
                        break
                    role_values[role] = expected_value
                derived_value: float | None = None
                if valid_components and metric_type == "delta":
                    if set(role_values) == {"current", "comparison"}:
                        derived_value = (
                            role_values["current"] - role_values["comparison"]
                        )
                elif valid_components and metric_type in {"rate", "share"}:
                    if (
                        set(role_values) == {"numerator", "denominator"}
                        and role_values["denominator"] != 0
                    ):
                        derived_value = (
                            role_values["numerator"] / role_values["denominator"]
                        )
                elif valid_components and metric_type == "rate_of_change":
                    if (
                        set(role_values) == {"current", "comparison"}
                        and role_values["comparison"] != 0
                    ):
                        derived_value = (
                            role_values["current"] - role_values["comparison"]
                        ) / abs(role_values["comparison"])
                if derived_value is not None and math.isfinite(derived_value):
                    record_set(
                        value, "expected_value", derived_value, path
                    )

            assessment = value.get("diagnostic_assessment")
            if not isinstance(assessment, dict):
                return
            explanations = assessment.get("explanations")
            competing = value.get("competing_explanations")
            if (
                isinstance(competing, list)
                and isinstance(explanations, list)
                and len(competing) == len(explanations)
                and all(isinstance(item, str) for item in competing)
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("explanation"), str)
                    for item in explanations
                )
            ):
                assessed = [item["explanation"] for item in explanations]
                unmatched_assessed = [
                    index
                    for index, explanation in enumerate(assessed)
                    if explanation not in competing
                ]
                unmatched_competing = [
                    explanation
                    for explanation in competing
                    if explanation not in assessed
                ]
                if len(unmatched_assessed) == len(unmatched_competing) == 1:
                    explanation_index = unmatched_assessed[0]
                    explanations[explanation_index]["explanation"] = (
                        unmatched_competing[0]
                    )
                    changes.append(
                        f"{_safe_path(path, 'diagnostic_assessment')}."
                        f"explanations[{explanation_index}].explanation"
                    )
            if (
                not isinstance(explanations, list)
                or not explanations
                or not all(
                    isinstance(item, dict)
                    and type(item.get("measurable")) is bool
                    for item in explanations
                )
            ):
                return
            measured = [item["measurable"] for item in explanations]
            declaration = (
                "measurable"
                if all(measured)
                else "not_measurable"
                if not any(measured)
                else "mixed"
            )
            record_set(value, "diagnostic_measurability", declaration, path)
            gated = any(
                item["measurable"]
                and item.get("disposition") in {"unresolved", "supported"}
                for item in explanations
            )
            assessment_path = _safe_path(path, "diagnostic_assessment")
            if (
                not gated
                and normalized.get("contract_version") == 3
                and all(
                    (
                        item["measurable"]
                        and item.get("disposition") in {"ruled_out", "weakened"}
                    )
                    or (
                        not item["measurable"]
                        and item.get("disposition") == "not_measurable"
                    )
                    for item in explanations
                )
            ):
                record_set(
                    assessment,
                    "decision_readiness",
                    "act_ready",
                    assessment_path,
                )
            if not gated:
                return
            record_set(
                assessment,
                "decision_readiness",
                "investigate_first",
                assessment_path,
            )
            action = value.get("action")
            if action is None:
                action = {}
                value["action"] = action
            if isinstance(action, dict):
                record_set(action, "kind", "diagnostic", _safe_path(path, "action"))
            confidence = value.get("confidence")
            if (
                isinstance(confidence, dict)
                and isinstance(confidence.get("level"), str)
                and confidence["level"].casefold() == "high"
            ):
                record_set(
                    confidence, "level", "medium", _safe_path(path, "confidence")
                )
            priority = value.get("priority")
            if (
                isinstance(priority, dict)
                and isinstance(priority.get("urgency"), str)
                and priority["urgency"].casefold() == "critical"
            ):
                record_set(
                    priority, "urgency", "high", _safe_path(path, "priority")
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(normalized, "$")
    return normalized, tuple(changes)


def assemble_contract(
    scaffold: Mapping[str, Any], insights: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble detached, strictly shaped partial outputs into contract v2."""

    _validate_partial(scaffold, SCAFFOLD_OUTPUTS, "contract scaffold")
    _validate_partial(insights, INSIGHT_OUTPUTS, "insights partial")
    return {
        "contract_version": 2,
        "analysis_plan": deepcopy(scaffold["analysis_plan"]),
        "candidates": deepcopy(scaffold["candidates"]),
        "insights": deepcopy(insights["insights"]),
    }


def _load_skill_loader():
    from fabric_rlm import SkillLoader

    return SkillLoader


def verify_portable_contract(payload: Mapping[str, Any]) -> None:
    """Run the packaged portable deep-insight verifier on the host."""

    loader_type = _load_skill_loader()
    source = loader_type().load("deep_insight_discovery").verifier_source
    if not source:
        raise RuntimeError("deep_insight_discovery has no packaged verifier")
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(
        compile(source, "<deep_insight_discovery verifier>", "exec"),
        namespace,
    )
    verify = namespace.get("verify")
    if not callable(verify):
        raise RuntimeError("deep_insight_discovery verifier has no verify function")
    try:
        verify(payload)
    except AssertionError as exc:
        raise AssertionError(
            f"portable deep-insight verification failed: {exc}"
        ) from exc


def _load_runtime_dependencies():
    import dspy
    from fabric_rlm import RLM
    from fabric_rlm._deep_insight_audit import (
        DeepInsightAuditError,
        audit_deep_insight,
    )
    from fabric_rlm._duckdb_audit import DuckDBAuditExecutor
    from fabric_rlm.metrics import summarize_trajectory

    return (
        dspy,
        RLM,
        DuckDBAuditExecutor,
        audit_deep_insight,
        summarize_trajectory,
        DeepInsightAuditError,
    )


def _extract_research(result: Any) -> dict[str, Any]:
    if not hasattr(result, "payload"):
        raise ValueError("research RLM result has no payload")
    payload = normalize_payload(result.payload)
    if set(payload) != {"research_json"}:
        raise ValueError("research RLM payload must contain exactly research_json")
    return parse_research_json(payload["research_json"])


def _extract_partial(
    result: Any, expected: Mapping[str, type], label: str
) -> dict[str, Any]:
    if not hasattr(result, "payload"):
        raise ValueError(f"{label} RLM result has no payload")
    payload = normalize_payload(result.payload)
    _validate_partial(payload, expected, f"{label} payload")
    return payload


def run_evidence_closure(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    *,
    lm: Any,
    rlm_type: Any,
    executor_type: Any,
    audit_function: Any,
    summarize_trajectory: Any,
    verify_function: Any,
    max_turns: int,
    timeout: float,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plan, verify, execute, and persist one bounded evidence-closure pass."""

    if not _pending_evidence_closure_targets(payload):
        return {
            "payload": deepcopy(dict(payload)),
            "audit": None,
            "summary": {
                "cached": True,
                "submitted": True,
                "turns": 0,
                "skipped": True,
            },
        }
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("evidence closure max_turns must be a positive integer")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("evidence closure timeout must be finite and positive")

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = _input_fingerprint(
        payload,
        {"sources": {name: str(path) for name, path in sources.items()}},
    )
    plan = (
        _load_synthesis_checkpoint(
            checkpoint,
            fingerprint,
            EVIDENCE_CLOSURE_OUTPUTS,
            "evidence closure",
        )
        if checkpoint is not None and checkpoint.is_file()
        else None
    )
    cached = plan is not None
    if plan is None:
        rlm = rlm_type.from_task(
            task=build_evidence_closure_prompt(sources, payload),
            outputs=EVIDENCE_CLOSURE_OUTPUTS,
            lm=lm,
            skills=list(SYNTHESIS_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=max_turns,
            reserve_finalize_turns=min(3, max_turns),
            timeout=timeout,
            verbose=False,
        )
        result = rlm.run()
        plan = _extract_partial(
            result,
            EVIDENCE_CLOSURE_OUTPUTS,
            "evidence closure",
        )
        summary = dict(summarize_trajectory(result.trajectory))
        summary.setdefault("submitted", True)
        summary["cached"] = False
    else:
        summary = {"cached": True, "submitted": True, "turns": 0}

    validated = validate_evidence_closure_plan(payload, plan, sources)
    closed_payload = merge_evidence_closure_plan(payload, validated, sources)
    closed_payload, _ = normalize_mechanical_contract(closed_payload, sources)
    verify_function(closed_payload)
    with executor_type(sources) as executor:
        audit = audit_function(closed_payload, executor)
    if checkpoint is not None and not cached:
        _write_synthesis_checkpoint(checkpoint, fingerprint, validated)
    return {
        "payload": closed_payload,
        "audit": audit,
        "summary": summary,
    }


def run_critic_evidence_closure(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
    sources: Mapping[str, Path],
    *,
    lm: Any,
    rlm_type: Any,
    executor_type: Any,
    audit_function: Any,
    summarize_trajectory: Any,
    verify_function: Any,
    max_turns: int,
    timeout: float,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one bounded source-aware closure pass from critic challenges."""

    if not _critic_closure_targets(payload, critic):
        return {
            "payload": deepcopy(dict(payload)),
            "audit": None,
            "summary": {
                "cached": True,
                "submitted": True,
                "turns": 0,
                "skipped": True,
            },
        }
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError(
            "critic evidence closure max_turns must be a positive integer"
        )
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "critic evidence closure timeout must be finite and positive"
        )

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = _input_fingerprint(
        payload,
        critic,
        {"sources": {name: str(path) for name, path in sources.items()}},
    )
    plan = (
        _load_synthesis_checkpoint(
            checkpoint,
            fingerprint,
            CRITIC_CLOSURE_OUTPUTS,
            "critic evidence closure",
        )
        if checkpoint is not None and checkpoint.is_file()
        else None
    )
    cached = plan is not None
    if plan is None:
        rlm = rlm_type.from_task(
            task=build_critic_closure_prompt(sources, payload, critic),
            outputs=CRITIC_CLOSURE_OUTPUTS,
            lm=lm,
            skills=list(SYNTHESIS_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=max_turns,
            reserve_finalize_turns=min(3, max_turns),
            timeout=timeout,
            verbose=False,
        )
        result = rlm.run()
        plan = _extract_partial(
            result,
            CRITIC_CLOSURE_OUTPUTS,
            "critic evidence closure",
        )
        summary = dict(summarize_trajectory(result.trajectory))
        summary.setdefault("submitted", True)
        summary["cached"] = False
    else:
        summary = {"cached": True, "submitted": True, "turns": 0}

    closed_payload = merge_critic_closure_plan(
        payload,
        critic,
        plan,
        sources,
    )
    closed_payload, _ = normalize_mechanical_contract(closed_payload, sources)
    verify_function(closed_payload)
    with executor_type(sources) as executor:
        audit = audit_function(closed_payload, executor)
    if checkpoint is not None and not cached:
        _write_synthesis_checkpoint(checkpoint, fingerprint, plan)
    return {
        "payload": closed_payload,
        "audit": audit,
        "summary": summary,
    }


def run_action_synthesis(
    payload: Mapping[str, Any],
    critic: Mapping[str, Any],
    *,
    lm: Any,
    rlm_type: Any,
    summarize_trajectory: Any,
    verify_function: Any,
    max_turns: int,
    timeout: float,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create and verify bounded program actions for exact eligible findings."""

    if not _action_synthesis_targets(payload, critic):
        return {
            "payload": deepcopy(dict(payload)),
            "summary": {
                "cached": True,
                "submitted": True,
                "turns": 0,
                "skipped": True,
            },
        }
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("action synthesis max_turns must be a positive integer")
    if (
        not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("action synthesis timeout must be finite and positive")

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = _input_fingerprint(payload, critic)
    updates = (
        _load_synthesis_checkpoint(
            checkpoint,
            fingerprint,
            ACTION_SYNTHESIS_OUTPUTS,
            "action synthesis",
        )
        if checkpoint is not None and checkpoint.is_file()
        else None
    )
    cached = updates is not None
    if updates is None:
        rlm = rlm_type.from_task(
            task=build_action_synthesis_prompt(payload, critic),
            outputs=ACTION_SYNTHESIS_OUTPUTS,
            lm=lm,
            skills=list(SYNTHESIS_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=max_turns,
            reserve_finalize_turns=min(3, max_turns),
            timeout=timeout,
            verbose=False,
        )
        result = rlm.run()
        updates = _extract_partial(
            result,
            ACTION_SYNTHESIS_OUTPUTS,
            "action synthesis",
        )
        summary = dict(summarize_trajectory(result.trajectory))
        summary.setdefault("submitted", True)
        summary["cached"] = False
    else:
        summary = {"cached": True, "submitted": True, "turns": 0}

    updated_payload = merge_action_synthesis(payload, critic, updates)
    verify_function(updated_payload)
    if checkpoint is not None and not cached:
        _write_synthesis_checkpoint(checkpoint, fingerprint, updates)
    return {"payload": updated_payload, "summary": summary}


def run_staged_benchmark(
    data_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    enable_evidence_closure: bool = False,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    research_turns: int = 18,
    scaffold_turns: int = 10,
    insight_turns: int = 14,
    repair_turns: int = 6,
    max_insight_repairs: int = 4,
    max_scaffold_repairs: int = 3,
    max_audit_repairs: int = 6,
    closure_turns: int = 10,
    timeout: float = 3600,
    research_cache_path: str | Path | None = None,
    scaffold_cache_path: str | Path | None = None,
    insights_cache_path: str | Path | None = None,
    closure_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run research, build and verify the contract, then audit its evidence."""

    if manifest_path is None:
        sources = discover_sources(data_dir)
        research_task = build_research_prompt(sources)
    else:
        from fabric_rlm._benchmark_manifest import (
            build_source_agnostic_research_prompt,
            load_source_manifest,
        )

        manifest = load_source_manifest(manifest_path)
        if Path(data_dir).resolve() != manifest.path.parent:
            raise ValueError(
                "data_dir must match the source manifest directory"
            )
        sources = dict(manifest.sources)
        research_task = build_source_agnostic_research_prompt(manifest)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required to run the staged Olist benchmark"
        )

    runtime_dependencies = _load_runtime_dependencies()
    (
        dspy,
        RLM,
        DuckDBAuditExecutor,
        audit_deep_insight,
        summarize_trajectory,
    ) = runtime_dependencies[:5]
    if len(runtime_dependencies) >= 6:
        DeepInsightAuditError = runtime_dependencies[5]
    else:
        from fabric_rlm._deep_insight_audit import DeepInsightAuditError
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

    cache_path = Path(research_cache_path) if research_cache_path else None
    if cache_path is not None and cache_path.is_file():
        research = parse_research_json(cache_path.read_text(encoding="utf-8"))
        research_summary = {"cached": True, "submitted": True, "turns": 0}
    else:
        research_rlm = RLM.from_task(
            task=research_task,
            outputs={"research_json": str},
            lm=lm,
            skills=list(RESEARCH_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=research_turns,
            reserve_finalize_turns=4,
            timeout=timeout,
            verbose=False,
        )
        research_result = research_rlm.run()
        research = _extract_research(research_result)
        research_summary = summarize_trajectory(research_result.trajectory)
        if cache_path is not None:
            _atomic_json(cache_path, research)

    scaffold_path = Path(scaffold_cache_path) if scaffold_cache_path else None
    scaffold_fingerprint = _input_fingerprint(research)
    scaffold = (
        _load_synthesis_checkpoint(
            scaffold_path,
            scaffold_fingerprint,
            SCAFFOLD_OUTPUTS,
            "contract scaffold",
        )
        if scaffold_path is not None and scaffold_path.is_file()
        else None
    )
    scaffold_was_cached = scaffold is not None
    if scaffold is not None:
        scaffold_summary = {"cached": True, "submitted": True, "turns": 0}
    else:
        scaffold_rlm = RLM.from_task(
            task=build_contract_scaffold_prompt(sources, research),
            outputs=SCAFFOLD_OUTPUTS,
            lm=lm,
            skills=list(SYNTHESIS_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=scaffold_turns,
            reserve_finalize_turns=4,
            timeout=timeout,
            verbose=False,
        )
        scaffold_result = scaffold_rlm.run()
        scaffold = _extract_partial(
            scaffold_result, SCAFFOLD_OUTPUTS, "contract scaffold"
        )
        scaffold_summary = summarize_trajectory(scaffold_result.trajectory)

    mechanical_changes: list[str] = []
    unnormalized_scaffold = deepcopy(scaffold)
    scaffold, scaffold_changes = normalize_mechanical_contract(scaffold, sources)
    mechanical_changes.extend(scaffold_changes)
    if not scaffold_was_cached or scaffold_changes:
        if scaffold_path is not None:
            _write_synthesis_checkpoint(
                scaffold_path, scaffold_fingerprint, scaffold
            )

    insights_path = Path(insights_cache_path) if insights_cache_path else None
    insights_fingerprint = _input_fingerprint(research, scaffold)
    prior_insights_fingerprint = _input_fingerprint(
        research, unnormalized_scaffold
    )
    insights = (
        _load_synthesis_checkpoint(
            insights_path,
            insights_fingerprint,
            INSIGHT_OUTPUTS,
            "insights",
        )
        if insights_path is not None and insights_path.is_file()
        else None
    )
    migrate_insights_checkpoint = False
    if (
        insights is None
        and insights_path is not None
        and insights_path.is_file()
        and prior_insights_fingerprint != insights_fingerprint
    ):
        insights = _load_synthesis_checkpoint(
            insights_path,
            prior_insights_fingerprint,
            INSIGHT_OUTPUTS,
            "insights",
        )
        migrate_insights_checkpoint = insights is not None
    insights_was_cached = insights is not None
    if insights is not None:
        insights_summary = {"cached": True, "submitted": True, "turns": 0}
    else:
        insight_rlm = RLM.from_task(
            task=build_insights_prompt(sources, research, scaffold),
            outputs=INSIGHT_OUTPUTS,
            lm=lm,
            skills=list(SYNTHESIS_SKILLS),
            enable_verifier=False,
            block_network=True,
            engine="default",
            max_turns=insight_turns,
            reserve_finalize_turns=6,
            timeout=timeout,
            verbose=False,
        )
        insight_result = insight_rlm.run()
        insights = _extract_partial(insight_result, INSIGHT_OUTPUTS, "insights")
        insights_summary = summarize_trajectory(insight_result.trajectory)
    payload = assemble_contract(scaffold, insights)
    payload, initial_changes = normalize_mechanical_contract(payload, sources)
    mechanical_changes.extend(initial_changes)
    normalized_insights = {"insights": deepcopy(payload["insights"])}
    insights_changed = normalized_insights != insights
    insights = normalized_insights
    if (
        insights_path is not None
        and (
            not insights_was_cached
            or migrate_insights_checkpoint
            or insights_changed
        )
    ):
        _write_synthesis_checkpoint(
            insights_path, insights_fingerprint, insights
        )
    repairs: list[dict[str, Any]] = []
    repair_attempts = {"scaffold": 0, "insights": 0}
    repair_limits = {
        "scaffold": max_scaffold_repairs,
        "insights": max_insight_repairs,
    }
    while True:
        try:
            verify_portable_contract(payload)
            break
        except AssertionError as exc:
            target = classify_repair_target(str(exc))
            if repair_attempts[target] >= repair_limits[target]:
                raise AssertionError(
                    "portable deep-insight verification failed after "
                    f"{target} target: {repair_attempts[target]} "
                    f"repair attempts: {exc}"
                ) from exc
            repair_attempts[target] += 1
            attempt = repair_attempts[target]

            if target == "insights":
                insight_index = extract_insight_index(
                    str(exc), len(insights["insights"])
                )
                if insight_index is not None:
                    statement_only = is_targeted_statement_error(str(exc))
                    interpretation_only = is_targeted_interpretation_error(
                        str(exc)
                    )
                    repair_rlm = RLM.from_task(
                        task=(
                            build_targeted_statement_repair_prompt(
                                insights["insights"][insight_index],
                                str(exc),
                            )
                            if statement_only
                            else build_targeted_interpretation_repair_prompt(
                                insights["insights"][insight_index],
                                str(exc),
                            )
                            if interpretation_only
                            else build_targeted_insight_repair_prompt(
                                sources,
                                scaffold,
                                insights["insights"][insight_index],
                                str(exc),
                            )
                        ),
                        outputs=(
                            TARGETED_STATEMENT_OUTPUTS
                            if statement_only
                            else TARGETED_INTERPRETATION_OUTPUTS
                            if interpretation_only
                            else TARGETED_INSIGHT_OUTPUTS
                        ),
                        lm=lm,
                        skills=list(SYNTHESIS_SKILLS),
                        enable_verifier=False,
                        block_network=True,
                        engine="default",
                        max_turns=repair_turns,
                        reserve_finalize_turns=3,
                        timeout=timeout,
                        verbose=False,
                    )
                    repair_result = repair_rlm.run()
                    updated_insights = deepcopy(insights["insights"])
                    if statement_only:
                        targeted = _extract_partial(
                            repair_result,
                            TARGETED_STATEMENT_OUTPUTS,
                            "targeted statement repair",
                        )
                        updated_insights[insight_index]["statement"] = targeted[
                            "statement"
                        ]
                    elif interpretation_only:
                        targeted = _extract_partial(
                            repair_result,
                            TARGETED_INTERPRETATION_OUTPUTS,
                            "targeted interpretation repair",
                        )
                        updated_insights[insight_index]["interpretation"] = targeted[
                            "interpretation"
                        ]
                    else:
                        targeted = _extract_partial(
                            repair_result,
                            TARGETED_INSIGHT_OUTPUTS,
                            "targeted insight repair",
                        )
                        updated_insights[insight_index] = (
                            merge_targeted_insight_repair(
                                updated_insights[insight_index],
                                targeted["insight"],
                                str(exc),
                            )
                        )
                    insights = {"insights": updated_insights}
                    payload = assemble_contract(scaffold, insights)
                    payload, changes = normalize_mechanical_contract(
                        payload, sources
                    )
                    mechanical_changes.extend(changes)
                    insights = {"insights": deepcopy(payload["insights"])}
                    if insights_path is not None:
                        _write_synthesis_checkpoint(
                            insights_path, insights_fingerprint, insights
                        )
                    repairs.append(
                        {
                            "target": target,
                            "attempt": attempt,
                            "mode": (
                                "targeted-statement"
                                if statement_only
                                else "targeted-interpretation"
                                if interpretation_only
                                else "targeted"
                            ),
                            "insight_index": insight_index + 1,
                            "insights": summarize_trajectory(
                                repair_result.trajectory
                            ),
                        }
                    )
                    continue

                repair_rlm = RLM.from_task(
                    task=build_insight_repair_prompt(
                        sources,
                        research,
                        scaffold,
                        insights,
                        str(exc),
                    ),
                    outputs=INSIGHT_OUTPUTS,
                    lm=lm,
                    skills=list(SYNTHESIS_SKILLS),
                    enable_verifier=False,
                    block_network=True,
                    engine="default",
                    max_turns=repair_turns,
                    reserve_finalize_turns=3,
                    timeout=timeout,
                    verbose=False,
                )
                repair_result = repair_rlm.run()
                insights = _extract_partial(
                    repair_result, INSIGHT_OUTPUTS, "insight repair"
                )
                payload = assemble_contract(scaffold, insights)
                payload, changes = normalize_mechanical_contract(payload, sources)
                mechanical_changes.extend(changes)
                insights = {"insights": deepcopy(payload["insights"])}
                if insights_path is not None:
                    _write_synthesis_checkpoint(
                        insights_path, insights_fingerprint, insights
                    )
                repairs.append(
                    {
                        "target": target,
                        "attempt": attempt,
                        "mode": "full",
                        "insights": summarize_trajectory(
                            repair_result.trajectory
                        ),
                    }
                )
                continue

            scaffold_repair_rlm = RLM.from_task(
                task=build_scaffold_repair_prompt(
                    sources,
                    research,
                    scaffold,
                    insights,
                    str(exc),
                ),
                outputs=SCAFFOLD_OUTPUTS,
                lm=lm,
                skills=list(SYNTHESIS_SKILLS),
                enable_verifier=False,
                block_network=True,
                engine="default",
                max_turns=repair_turns,
                reserve_finalize_turns=3,
                timeout=timeout,
                verbose=False,
            )
            scaffold_repair_result = scaffold_repair_rlm.run()
            scaffold = _extract_partial(
                scaffold_repair_result,
                SCAFFOLD_OUTPUTS,
                "contract scaffold repair",
            )
            scaffold, changes = normalize_mechanical_contract(scaffold, sources)
            mechanical_changes.extend(changes)
            if scaffold_path is not None:
                _write_synthesis_checkpoint(
                    scaffold_path, scaffold_fingerprint, scaffold
                )
            insights_fingerprint = _input_fingerprint(research, scaffold)

            regeneration_rlm = RLM.from_task(
                task=build_insights_prompt(sources, research, scaffold),
                outputs=INSIGHT_OUTPUTS,
                lm=lm,
                skills=list(SYNTHESIS_SKILLS),
                enable_verifier=False,
                block_network=True,
                engine="default",
                max_turns=repair_turns,
                reserve_finalize_turns=3,
                timeout=timeout,
                verbose=False,
            )
            regeneration_result = regeneration_rlm.run()
            insights = _extract_partial(
                regeneration_result,
                INSIGHT_OUTPUTS,
                "dependent insight regeneration",
            )
            payload = assemble_contract(scaffold, insights)
            payload, changes = normalize_mechanical_contract(payload, sources)
            mechanical_changes.extend(changes)
            insights = {"insights": deepcopy(payload["insights"])}
            if insights_path is not None:
                _write_synthesis_checkpoint(
                    insights_path, insights_fingerprint, insights
                )
            repairs.append(
                {
                    "target": target,
                    "attempt": attempt,
                    "mode": "full",
                    "scaffold": summarize_trajectory(
                        scaffold_repair_result.trajectory
                    ),
                    "insights": summarize_trajectory(
                        regeneration_result.trajectory
                    ),
                }
            )

    closure_summary = {
        "cached": True,
        "submitted": True,
        "turns": 0,
        "skipped": True,
    }
    audit = None
    if enable_evidence_closure:
        closure_record = run_evidence_closure(
            payload,
            sources,
            lm=lm,
            rlm_type=RLM,
            executor_type=DuckDBAuditExecutor,
            audit_function=audit_deep_insight,
            summarize_trajectory=summarize_trajectory,
            verify_function=verify_portable_contract,
            max_turns=closure_turns,
            timeout=timeout,
            checkpoint_path=closure_cache_path,
        )
        payload = closure_record["payload"]
        insights = {"insights": deepcopy(payload["insights"])}
        closure_summary = closure_record["summary"]
        audit = closure_record["audit"]

    audit_repairs: list[dict[str, Any]] = []
    audit_attempt = 0
    while audit is None:
        try:
            with DuckDBAuditExecutor(sources) as executor:
                audit = audit_deep_insight(payload, executor)
            break
        except DeepInsightAuditError as exc:
            mismatch = parse_host_audit_mismatch(str(exc))
            if mismatch is None:
                raise
            if audit_attempt >= max_audit_repairs:
                raise DeepInsightAuditError(
                    "host numeric audit failed after "
                    f"{audit_attempt} audit repair attempts at "
                    f"{mismatch.path}: {exc}"
                ) from exc
            is_candidate_repair = mismatch.path.startswith("candidates")
            if is_candidate_repair:
                _candidate_audit_target(scaffold, mismatch)
                audit_outputs = AUDIT_REJECTION_COMPONENT_OUTPUTS
                repair_task = build_host_candidate_audit_repair_prompt(
                    scaffold, mismatch
                )
            else:
                (
                    insight_index,
                    target_tokens,
                    output_name,
                    audit_outputs,
                ) = _audit_target(insights, mismatch)
                repair_task = build_host_audit_repair_prompt(
                    insights["insights"][insight_index],
                    mismatch,
                    target_tokens,
                    output_name,
                    include_statement="statement" in audit_outputs,
                )
            audit_attempt += 1
            repair_rlm = RLM.from_task(
                task=repair_task,
                outputs=audit_outputs,
                lm=lm,
                skills=list(SYNTHESIS_SKILLS),
                enable_verifier=False,
                block_network=True,
                engine="default",
                max_turns=repair_turns,
                reserve_finalize_turns=3,
                timeout=timeout,
                verbose=False,
            )
            repair_result = repair_rlm.run()
            repair_partial = _extract_partial(
                repair_result,
                audit_outputs,
                "host audit repair",
            )
            if is_candidate_repair:
                candidate_scaffold = merge_host_candidate_audit_repair(
                    scaffold, mismatch, repair_partial
                )
                candidate_scaffold, scaffold_changes = (
                    normalize_mechanical_contract(candidate_scaffold, sources)
                )
                mechanical_changes.extend(scaffold_changes)
                candidate_insights = deepcopy(insights)
            else:
                candidate_scaffold = deepcopy(scaffold)
                updated_insights = deepcopy(insights["insights"])
                updated_insights[insight_index] = merge_host_audit_repair(
                    updated_insights[insight_index],
                    mismatch,
                    repair_partial,
                )
                candidate_insights = {"insights": updated_insights}
            candidate_payload = assemble_contract(
                candidate_scaffold, candidate_insights
            )
            candidate_payload, changes = normalize_mechanical_contract(
                candidate_payload, sources
            )
            mechanical_changes.extend(changes)
            candidate_scaffold = {
                "analysis_plan": deepcopy(candidate_payload["analysis_plan"]),
                "candidates": deepcopy(candidate_payload["candidates"]),
            }
            candidate_insights = {
                "insights": deepcopy(candidate_payload["insights"])
            }
            verify_portable_contract(candidate_payload)
            if is_candidate_repair:
                if scaffold_path is not None:
                    _write_synthesis_checkpoint(
                        scaffold_path,
                        scaffold_fingerprint,
                        candidate_scaffold,
                    )
                insights_fingerprint = _input_fingerprint(
                    research, candidate_scaffold
                )
                if insights_path is not None:
                    _write_synthesis_checkpoint(
                        insights_path,
                        insights_fingerprint,
                        candidate_insights,
                    )
                scaffold = candidate_scaffold
            elif insights_path is not None:
                _write_synthesis_checkpoint(
                    insights_path,
                    insights_fingerprint,
                    candidate_insights,
                )
            insights = candidate_insights
            payload = candidate_payload
            audit_repairs.append(
                {
                    "target_path": mismatch.path,
                    "attempt": audit_attempt,
                    "expected": mismatch.expected,
                    "actual": mismatch.actual,
                    "trajectory": summarize_trajectory(
                        repair_result.trajectory
                    ),
                }
            )

    trajectories = {
        "research": research_summary,
        "contract_scaffold": scaffold_summary,
        "insights": insights_summary,
    }
    stage_skills = {
        "research": RESEARCH_SKILLS,
        "contract_scaffold": SYNTHESIS_SKILLS,
        "insights": SYNTHESIS_SKILLS,
    }
    if enable_evidence_closure:
        trajectories["evidence_closure"] = closure_summary
        stage_skills["evidence_closure"] = SYNTHESIS_SKILLS

    return {
        "research": research,
        "payload": payload,
        "audit": audit,
        "repairs": repairs,
        "audit_repairs": audit_repairs,
        "mechanical_repairs": {
            "count": len(mechanical_changes),
            "paths": list(mechanical_changes),
        },
        "trajectories": trajectories,
        "model": model,
        "stage_skills": stage_skills,
    }


def write_staged_artifacts(
    output_dir: str | Path, record: Mapping[str, Any]
) -> dict[str, Path]:
    """Persist deterministic aggregate staged artifacts without trajectories."""

    output = Path(output_dir)
    research = dict(record["research"])
    payload = normalize_payload(record["payload"])
    audit = audit_to_dict(record["audit"])
    trajectories = {
        stage: dict(summary)
        for stage, summary in record["trajectories"].items()
    }
    repairs = [dict(summary) for summary in record["repairs"]]
    audit_repairs = [
        dict(summary) for summary in record.get("audit_repairs", [])
    ]
    mechanical_repairs = {
        "count": int(record["mechanical_repairs"]["count"]),
        "paths": list(record["mechanical_repairs"]["paths"]),
    }
    stage_skills = {
        stage: list(skills) for stage, skills in record["stage_skills"].items()
    }
    run = {
        "status": "success",
        "model": record["model"],
        "stage_skills": stage_skills,
        "turn_summaries": trajectories,
        "repair_summaries": repairs,
        "audit_repair_summaries": audit_repairs,
        "mechanical_repairs": mechanical_repairs,
        "counts": {
            "research_candidates": len(research["candidates"]),
            "contract_candidates": len(payload["candidates"]),
            "insights": len(payload["insights"]),
            "audit_checks": audit["total_checks"],
            "repairs": len(repairs),
            "insight_repairs": sum(
                repair["target"] == "insights" for repair in repairs
            ),
            "scaffold_repairs": sum(
                repair["target"] == "scaffold" for repair in repairs
            ),
            "audit_repairs": len(audit_repairs),
            "mechanical_repairs": mechanical_repairs["count"],
        },
    }
    paths = {
        "research": output / "research.json",
        "payload": output / "payload.json",
        "audit": output / "audit.json",
        "run": output / "run.json",
    }
    _atomic_json(paths["research"], research)
    _atomic_json(paths["payload"], payload)
    _atomic_json(paths["audit"], audit)
    _atomic_json(paths["run"], run)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the staged local Olist deep-insight benchmark."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--research-turns", default=18, type=int)
    parser.add_argument("--scaffold-turns", default=10, type=int)
    parser.add_argument("--insight-turns", default=14, type=int)
    parser.add_argument("--repair-turns", default=6, type=int)
    parser.add_argument("--max-insight-repairs", default=4, type=int)
    parser.add_argument("--max-scaffold-repairs", default=3, type=int)
    parser.add_argument("--max-audit-repairs", default=6, type=int)
    parser.add_argument("--timeout", default=3600, type=float)
    parser.add_argument(
        "--output-dir",
        default=Path("_local") / "olist_staged_deep_insight_benchmark",
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = run_staged_benchmark(
        args.data_dir,
        model=args.model,
        api_base=args.api_base,
        research_turns=args.research_turns,
        scaffold_turns=args.scaffold_turns,
        insight_turns=args.insight_turns,
        repair_turns=args.repair_turns,
        max_insight_repairs=args.max_insight_repairs,
        max_scaffold_repairs=args.max_scaffold_repairs,
        max_audit_repairs=args.max_audit_repairs,
        timeout=args.timeout,
        research_cache_path=args.output_dir / "research.json",
        scaffold_cache_path=(
            args.output_dir / "contract_scaffold.checkpoint.json"
        ),
        insights_cache_path=args.output_dir / "insights.checkpoint.json",
    )
    paths = write_staged_artifacts(args.output_dir, record)
    print(
        json.dumps(
            {
                "status": "success",
                "audit_checks": record["audit"].total_checks,
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
