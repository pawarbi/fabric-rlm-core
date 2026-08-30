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
from fabric_rlm.experimental.analysis_evidence import EvidenceRegistry
from fabric_rlm.experimental.analysis_operators import (
    additive_decomposition,
    rate_decomposition,
    volume_rate_mix_decomposition,
)
from fabric_rlm.experimental.analysis_benchmarks import (
    SyntheticBenchmark,
    load_synthetic_benchmark,
    write_clustered_benchmark,
    write_correlated_benchmark,
    write_decomposition_benchmark,
    write_panel_benchmark,
    write_shift_benchmark,
    write_time_series_benchmark,
)
from fabric_rlm.experimental.analysis_reproducibility import (
    canonical_json,
    derive_seed,
    fingerprint,
)
from fabric_rlm.experimental.analysis_scoring import (
    BenchmarkCaseScore,
    BenchmarkReport,
    score_binary_classification_case,
    score_decomposition_case,
)
from fabric_rlm.experimental.analysis_validation import (
    SplitFold,
    SplitPlan,
    ValidatedSplitPlan,
    build_grouped_split_plan,
    build_nested_grouped_split_plan,
    build_random_split_plan,
    build_stratified_split_plan,
    build_temporal_split_plan,
    validate_split_plan,
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
    "additive_decomposition",
    "AnalysisBrief",
    "AnalysisDAG",
    "AttemptConfig",
    "AttemptRecord",
    "BanditPolicy",
    "BanditState",
    "BenchmarkCaseScore",
    "BenchmarkReport",
    "build_grouped_split_plan",
    "build_nested_grouped_split_plan",
    "build_random_split_plan",
    "build_stratified_split_plan",
    "build_temporal_split_plan",
    "Budget",
    "canonical_json",
    "derive_seed",
    "DifficultyVerdict",
    "EFFORT_RUNG_COST",
    "EffortBanditPolicy",
    "EffortLadderPolicy",
    "EvidenceEntry",
    "EvidenceRegistry",
    "fingerprint",
    "LadderPolicy",
    "load_synthetic_benchmark",
    "OperatorNode",
    "OperatorResult",
    "OperatorSpec",
    "rate_decomposition",
    "RunBudget",
    "score_binary_classification_case",
    "score_decomposition_case",
    "SplitFold",
    "SplitPlan",
    "SyntheticBenchmark",
    "ValidatedSplitPlan",
    "ValidatedDAG",
    "ValidationVerdict",
    "validate_analysis_dag",
    "validate_split_plan",
    "volume_rate_mix_decomposition",
    "write_clustered_benchmark",
    "write_correlated_benchmark",
    "write_decomposition_benchmark",
    "write_panel_benchmark",
    "write_shift_benchmark",
    "write_time_series_benchmark",
]
