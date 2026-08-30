"""Experimental, not on the supported path.

Public symbols re-exported here are the surface promised by ``QUICKSTART.md``
§4b for power users of the adaptive engine. Everything in this submodule may
change in 0.2.x — pin the version if you depend on it directly.
"""

from fabric_rlm.experimental.adaptive_policy import (
    AttemptConfig,
    AttemptRecord,
    Budget,
    DifficultyVerdict,
    LadderPolicy,
    ValidationVerdict,
)
from fabric_rlm.experimental.adaptive_runner import AdaptiveResult, AdaptiveRunner
from fabric_rlm.experimental.analysis_contracts import (
    AnalysisBrief,
    AnalysisDAG,
    EvidenceEntry,
    OperatorNode,
    OperatorResult,
    RunBudget,
)
from fabric_rlm.experimental.analysis_dag import (
    OperatorSpec,
    ValidatedDAG,
    validate_analysis_dag,
)
from fabric_rlm.experimental.analysis_reproducibility import (
    canonical_json,
    derive_seed,
    fingerprint,
)
from fabric_rlm.experimental.bandit_policy import BanditPolicy, BanditState
from fabric_rlm.experimental.effort_ladder_policy import (
    EFFORT_RUNG_COST,
    EffortBanditPolicy,
    EffortLadderPolicy,
)

__all__ = [
    "AdaptiveResult",
    "AdaptiveRunner",
    "AnalysisBrief",
    "AnalysisDAG",
    "AttemptConfig",
    "AttemptRecord",
    "BanditPolicy",
    "BanditState",
    "Budget",
    "canonical_json",
    "derive_seed",
    "DifficultyVerdict",
    "EFFORT_RUNG_COST",
    "EffortBanditPolicy",
    "EffortLadderPolicy",
    "EvidenceEntry",
    "fingerprint",
    "LadderPolicy",
    "OperatorNode",
    "OperatorResult",
    "OperatorSpec",
    "RunBudget",
    "ValidatedDAG",
    "ValidationVerdict",
    "validate_analysis_dag",
]
