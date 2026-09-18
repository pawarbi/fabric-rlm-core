"""Keep the public API reference aligned with the checked-in signatures."""

from collections import Counter
import inspect
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _section(heading: str) -> str:
    text = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing API reference section: {heading}"
    return match.group(1)


@pytest.mark.parametrize(
    ("heading", "expected_count"),
    [("## RLM.task and RLM", 37), ("## verified_task", 8)],
)
def test_api_parameter_tables_match_signatures_and_defaults(heading, expected_count):
    # Other tests reload runtime; resolve the current class at test execution.
    from fabric_rlm.runtime import RLM
    import fabric_rlm.verify as verify

    callables = (
        (RLM.task, RLM.__init__)
        if heading == "## RLM.task and RLM"
        else (verify.verified_task,)
    )
    parameters = {}
    for function in callables:
        for name, parameter in inspect.signature(function).parameters.items():
            if name in {"self", "cls"} or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if name in parameters:
                assert repr(parameters[name].default) == repr(parameter.default), name
            parameters[name] = parameter

    rows = re.findall(
        r"^\|\s*`([^`]+)`\s*\|\s*([^|]+)\|",
        _section(heading),
        re.MULTILINE,
    )
    assert len(parameters) == expected_count
    assert Counter(name for name, _ in rows) == Counter(
        {name: 1 for name in parameters}
    ), "Each named parameter must appear exactly once, with no extra parameters"

    # The docs use double-quoted strings and a symbolic multiline prompt default.
    display_defaults = {
        "engine": '"auto"',
        "inner_engine": '"v6-custom"',
        "reconcile_guidance": "_RECONCILE_GUIDANCE",
    }
    for name, cell in rows:
        default = parameters[name].default
        if name == "reconcile_guidance":
            assert default is verify._RECONCILE_GUIDANCE
        elif name in display_defaults:
            assert default == {"engine": "auto", "inner_engine": "v6-custom"}[name]
        expected = display_defaults.get(name, repr(default))
        if default is inspect.Parameter.empty:
            expected = "required"
        assert cell.strip().strip("`") == expected, f"{heading}: {name} default"

    outputs_default = parameters["outputs"].default
    if heading == "## RLM.task and RLM":
        assert outputs_default is None
    else:
        assert outputs_default is inspect.Parameter.empty


def test_api_reference_links_to_sources_and_gives_starting_guidance():
    text = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    intro = text.split("\n## ", 1)[0]
    assert "(../fabric_rlm/runtime.py)" in intro
    assert "(../fabric_rlm/verify.py)" in intro
    guidance = _section("## Start here")
    assert "RLM.task(...).run()" in guidance
    assert "max_turns=20" in guidance
    assert "output_validator" in guidance


@pytest.mark.parametrize("filename", ["README.md", "QUICKSTART.md"])
def test_public_guides_link_to_api_reference(filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    assert "(docs/api-reference.md)" in text


def test_task_construction_keeps_documented_ordinary_defaults(monkeypatch):
    from fabric_rlm.runtime import RLM

    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)

    def unused_lm(*args, **kwargs):
        pytest.fail("RLM.task must return an unexecuted RLM")

    rlm = RLM.task("Count rows.", lm=unused_lm)
    defaults = inspect.signature(RLM.__init__).parameters
    assert isinstance(rlm, RLM)
    for name in ("max_turns", "timeout", "enable_verifier", "max_submit_bytes"):
        assert getattr(rlm, name) == defaults[name].default, name
    assert rlm.max_turns == 20
    assert rlm.analytical_integrity == "repair"
    assert rlm.engine == "v6-custom"
    assert rlm._inline_inputs == {}
    assert rlm._inline_outputs == []
