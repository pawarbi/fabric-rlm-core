"""Public API for fabric-rlm."""

from .artifacts import File, LocalArtifactStore
from .ledger import Ledger, bare_numbers, cited_labels, format_value
from .semantic_model import SemanticModel, sempy_available
from .excel_artifacts import (
    ExcelCellValue,
    ExcelTargetRange,
    add_excel_workbook_context,
    iter_target_cells,
    parse_target_ranges,
    summarize_workbook_structure_context,
    summarize_workbook_context,
    validate_target_range_sanity,
)
from .interpreter import (
    ExecResult,
    Interpreter,
    SubprocessPythonInterpreter,
    WorkerProtocolError,
    WorkerTimeout,
)
from .lm import AnthropicLM, FabricLM, OpenAILM, register_backend, resolve_lm
from .metrics import ValidationCheck, ValidationReport
from .replay_lm import (
    DivergenceError,
    ReplayInterpreter,
    ReplayLM,
    replay_trajectory,
)
from .runtime import RLM, RLMResult
from .skill_loader import Skill, SkillLoader, compose_skills, list_skills, load_skill
from .verify import VerifiedResult, answers_agree, verified_task
from .trajectory import Issue, Trajectory, TurnRecord
from .validators import (
    assert_in_range,
    assert_keys,
    assert_list_len,
    assert_list_of,
    assert_matches_regex,
    assert_predicate,
    chain,
    signature_validator,
)

__all__ = [
    "AnthropicLM",
    "DivergenceError",
    "ExecResult",
    "ExcelCellValue",
    "ExcelTargetRange",
    "FabricLM",
    "File",
    "Interpreter",
    "LocalArtifactStore",
    "OpenAILM",
    "Issue",
    "Ledger",
    "ReplayInterpreter",
    "ReplayLM",
    "RLM",
    "RLMResult",
    "SemanticModel",
    "replay_trajectory",
    "Skill",
    "SkillLoader",
    "SubprocessPythonInterpreter",
    "bare_numbers",
    "cited_labels",
    "format_value",
    "sempy_available",
    "Trajectory",
    "TurnRecord",
    "ValidationCheck",
    "ValidationReport",
    "WorkerProtocolError",
    "WorkerTimeout",
    "assert_in_range",
    "assert_keys",
    "assert_list_len",
    "assert_list_of",
    "assert_matches_regex",
    "assert_predicate",
    "add_excel_workbook_context",
    "chain",
    "compose_skills",
    "iter_target_cells",
    "list_skills",
    "load_skill",
    "parse_target_ranges",
    "register_backend",
    "resolve_lm",
    "signature_validator",
    "summarize_workbook_structure_context",
    "summarize_workbook_context",
    "validate_target_range_sanity",
]

# Single source of truth for the package version. pyproject.toml reads this
# statically via [tool.setuptools.dynamic] — bump it here and nowhere else.
__version__ = "0.3.5"
