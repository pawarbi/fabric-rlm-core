"""Public API for fabric-rlm."""

from .artifacts import File, LocalArtifactStore
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
    "FabricLM",
    "File",
    "Interpreter",
    "LocalArtifactStore",
    "OpenAILM",
    "Issue",
    "ReplayInterpreter",
    "ReplayLM",
    "RLM",
    "RLMResult",
    "replay_trajectory",
    "Skill",
    "SkillLoader",
    "SubprocessPythonInterpreter",
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
    "chain",
    "compose_skills",
    "list_skills",
    "load_skill",
    "register_backend",
    "resolve_lm",
    "signature_validator",
]

# Single source of truth for the package version. pyproject.toml reads this
# statically via [tool.setuptools.dynamic] — bump it here and nowhere else.
__version__ = "0.2.4"
