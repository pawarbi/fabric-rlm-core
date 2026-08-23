"""Coverage tests for the ``RLM.from_task`` factory route.

Background
----------
The package supports two ways to construct an :class:`RLM`:

1. The classic dspy-style ``RLM(signature="inputs -> outputs")`` -- the task
   is implied by the signature name and any docstring on a ``dspy.Signature``
   subclass.
2. ``RLM.from_task(task=<prose>, outputs=[...])`` -- the task is an inline
   string that flows directly into the system-prompt's ``## Task`` section
   (see :func:`fabric_rlm.prompts.build_system_prompt`).

Route #2 was the unlock that let gpt-5 actually see the SpreadsheetBench
task text in its system prompt; the previously-used pattern
``RLM("question -> answer").run({"question": prompt_text})`` instead surfaced
only the abstract signature and stuffed the prose into the user message --
which materially hurt large reasoning models. The fix shipped to ``main``
(commit ``ba2fb20``) but until now there was no dedicated test proving the
contract that the ``task`` argument lands in the system prompt.

These tests pin the contract so future refactors can't silently regress it.
The assertions are deliberately section-scoped (extracting ``## Task``,
``## Inputs available in namespace`` and ``## Required output fields ...``)
so a substring landing in the wrong place won't false-pass.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.prompts import build_system_prompt


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class ScriptedLM:
    """Minimal mock LM that records the messages it receives."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


def _wrap(body: str) -> str:
    return "```python\n" + body + "\n```"


def _system_msg(lm: ScriptedLM) -> str:
    return lm.messages[0][0]["content"]


# Section anchors used by :func:`build_system_prompt`. Keep this list in
# sync with ``SYSTEM_PROMPT_TEMPLATE`` in ``fabric_rlm/prompts.py``.
_SECTION_ORDER = (
    "## Task",
    "## Inputs available in namespace",
    "## Required output fields for SUBMIT()",
    "## Answering rules",
)


def _section(prompt: str, heading: str) -> str:
    """Return the body of ``heading`` up to the next known section heading.

    Raises ``AssertionError`` if the heading is missing -- regressions that
    rename or drop a section should fail loudly here, not via a quiet
    substring miss elsewhere.
    """
    if heading not in _SECTION_ORDER:
        raise AssertionError(f"Unknown section heading: {heading!r}")
    start = prompt.find(heading)
    assert start != -1, f"section {heading!r} missing from system prompt"
    idx = _SECTION_ORDER.index(heading)
    end = len(prompt)
    for next_heading in _SECTION_ORDER[idx + 1:]:
        nxt = prompt.find(next_heading, start + len(heading))
        if nxt != -1:
            end = nxt
            break
    return prompt[start:end]


# ---------------------------------------------------------------------------
# Construction contract (one sanity test on internal storage; the rest of
# the suite drives the contract through observable behavior).
# ---------------------------------------------------------------------------


class TestFromTaskConstruction:
    def test_factory_wires_task_outputs_inputs_and_kwargs(self):
        """One direct check that ``from_task`` actually sets the inline
        attributes the runtime reads. Other tests prove the contract
        through observable system-prompt content; this is a focused
        sanity test on the wiring itself."""
        rlm = RLM.from_task(
            "Compute the unique answer.",
            inputs={"foo": 7},
            outputs=["answer", "evidence"],
            lm=ScriptedLM([]),
            max_turns=7,
            timeout=11,
        )
        assert rlm._inline_task == "Compute the unique answer."
        assert rlm._inline_outputs == ["answer", "evidence"]
        assert rlm._inline_inputs == {"foo": 7}
        assert rlm.signature is None
        # kwargs flow through to the constructor.
        assert rlm.max_turns == 7
        assert rlm.timeout == 11

    def test_factory_accepts_typed_output_mapping(self):
        rlm = RLM.from_task(
            "Return a structured result.",
            outputs={"result": dict},
            lm=ScriptedLM([]),
            max_turns=1,
            timeout=5,
        )

        assert rlm._inline_outputs == ["result"]
        assert rlm._inline_output_types == {"result": dict}

    def test_typed_output_mapping_is_defensively_copied(self):
        external = {"result": dict}
        rlm = RLM.from_task(
            "Return a structured result.",
            outputs=external,
            lm=ScriptedLM([]),
            max_turns=1,
            timeout=5,
        )
        external["result"] = str
        external["extra"] = list

        assert rlm._inline_outputs == ["result"]
        assert rlm._inline_output_types == {"result": dict}

    def test_typed_output_mapping_rejects_non_type_values(self):
        with pytest.raises(TypeError, match="must be a Python type"):
            RLM.from_task(
                "Return a structured result.",
                outputs={"result": "dict"},
                lm=ScriptedLM([]),
            )

    def test_typed_output_mapping_rejects_invalid_field_names(self):
        with pytest.raises(ValueError, match="must not be empty"):
            RLM.from_task(
                "Return a structured result.",
                outputs={"": dict},
                lm=ScriptedLM([]),
            )

    def test_typed_output_mapping_rejects_conflicting_explicit_signature(self):
        with pytest.raises(ValueError, match="do not match explicit signature outputs"):
            RLM.from_task(
                "Return a structured result.",
                inputs={"q": "x"},
                outputs={"result": dict},
                signature="q -> answer",
                lm=ScriptedLM([]),
                engine="dspy",
            )

    def test_typed_output_mapping_accepts_reordered_explicit_signature_outputs(self):
        rlm = RLM.from_task(
            "Return both outputs.",
            inputs={"q": "x"},
            outputs={"result": dict, "answer": str},
            signature="q -> answer, result",
            lm=ScriptedLM([]),
            engine="dspy",
        )

        assert rlm._inline_outputs == ["result", "answer"]
        assert rlm._inline_output_types == {"result": dict, "answer": str}

    def test_omitted_inputs_and_outputs_default_to_empty(self):
        rlm = RLM.from_task("X", lm=ScriptedLM([]), max_turns=1, timeout=5)
        assert rlm._inline_outputs == []
        assert rlm._inline_inputs == {}


class TestTaskAliasConstruction:
    def test_task_alias_wires_task_outputs_inputs_and_kwargs(self):
        rlm = RLM.task(
            "Compute the unique answer.",
            inputs={"foo": 7},
            outputs=["answer", "evidence"],
            lm=ScriptedLM([]),
            max_turns=7,
            timeout=11,
        )
        assert rlm._inline_task == "Compute the unique answer."
        assert rlm._inline_outputs == ["answer", "evidence"]
        assert rlm._inline_inputs == {"foo": 7}
        assert rlm.signature is None
        assert rlm.max_turns == 7
        assert rlm.timeout == 11

    def test_task_alias_preserves_subclass_dispatch(self):
        class CustomRLM(RLM):
            pass

        rlm = CustomRLM.task("X", lm=ScriptedLM([]), max_turns=1, timeout=5)

        assert isinstance(rlm, CustomRLM)

    def test_task_alias_runs_through_inline_task_runtime(self):
        lm = ScriptedLM([_wrap("SUBMIT(answer=foo * 2)")])
        rlm = RLM.task(
            "Double foo.",
            inputs={"foo": 21},
            outputs=["answer"],
            lm=lm,
            max_turns=1,
            timeout=5,
        )

        result = rlm.run()

        assert result.submitted
        assert result.payload == {"answer": 42}

    def test_typed_output_mapping_rejects_scalar_then_accepts_object(self):
        lm = ScriptedLM(
            [
                _wrap("SUBMIT(result='South')"),
                _wrap("SUBMIT(result={'top_region': 'South'})"),
            ]
        )
        rlm = RLM.task(
            "Return the result.",
            outputs={"result": dict},
            lm=lm,
            max_turns=2,
            timeout=5,
        )

        result = rlm.run()

        assert result.submitted
        assert result.payload == {"result": {"top_region": "South"}}
        assert result.trajectory.turns[0].validation_errors == [
            "Required output field 'result' must be dict, got str."
        ]
        assert "must be dict, got str" in lm.messages[1][-1]["content"]

    def test_legacy_output_list_keeps_accepting_scalar(self):
        lm = ScriptedLM([_wrap("SUBMIT(result='South')")])
        result = RLM.task(
            "Return the result.",
            outputs=["result"],
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

        assert result.submitted
        assert result.payload == {"result": "South"}

    def test_task_alias_omitted_inputs_and_outputs_default_to_empty(self):
        rlm = RLM.task("X", lm=ScriptedLM([]), max_turns=1, timeout=5)
        assert rlm._inline_outputs == []
        assert rlm._inline_inputs == {}


# ---------------------------------------------------------------------------
# Caller-supplied collections must be copied, not aliased
# ---------------------------------------------------------------------------


class TestCallerCollectionsAreIsolated:
    """Behavioral proof (via system prompt) that the factory copies the
    caller's ``inputs`` dict and ``outputs`` list. Mutating the original
    after construction must not change what the LM sees."""

    def test_outputs_list_mutation_after_construction_is_ignored(self):
        external = ["a"]
        lm = ScriptedLM([_wrap("SUBMIT(a=1)")])
        rlm = RLM.from_task("X", outputs=external,
                            lm=lm, max_turns=1, timeout=5)
        external.append("b")  # must NOT be picked up

        rlm.run()
        outputs_section = _section(_system_msg(lm),
                                   "## Required output fields for SUBMIT()")
        assert "- a" in outputs_section
        assert "- b" not in outputs_section

    def test_inputs_dict_mutation_after_construction_is_ignored(self):
        external = {"foo": 1}
        lm = ScriptedLM([_wrap("SUBMIT(answer=foo)")])
        rlm = RLM.from_task("X", inputs=external, outputs=["answer"],
                            lm=lm, max_turns=1, timeout=5)
        external["foo"] = 999
        external["bar"] = "added later"

        result = rlm.run()
        assert result.submitted
        assert result.payload == {"answer": 1}
        inputs_section = _section(_system_msg(lm),
                                  "## Inputs available in namespace")
        assert "foo:" in inputs_section
        assert "bar:" not in inputs_section


# ---------------------------------------------------------------------------
# System-prompt task-text surface (the SSB unlock)
# ---------------------------------------------------------------------------


class TestSystemPromptTaskSurface:
    def test_inline_task_appears_in_task_section(self):
        """The whole point of from_task: the ``task`` string must land in
        the ``## Task`` section, where reasoning models actually see it."""
        unique = "Double the integer named foo and submit it as 'answer'."
        lm = ScriptedLM([_wrap("SUBMIT(answer=14)")])
        rlm = RLM.from_task(unique, inputs={"foo": 7}, outputs=["answer"],
                            lm=lm, max_turns=1, timeout=5)
        rlm.run()
        task_section = _section(_system_msg(lm), "## Task")
        assert unique in task_section, (
            "from_task(...) MUST surface the task text into the system "
            f"prompt's ## Task section. Got:\n{task_section}"
        )

    def test_inline_inputs_listed_in_inputs_section(self):
        unique_input_name = "spreadsheet_bytes_b64"
        lm = ScriptedLM([_wrap(f"SUBMIT(answer={unique_input_name}[:0])")])
        rlm = RLM.from_task(
            "Use the supplied spreadsheet to answer.",
            inputs={unique_input_name: ""},
            outputs=["answer"],
            lm=lm, max_turns=1, timeout=5,
        )
        rlm.run()
        inputs_section = _section(_system_msg(lm),
                                  "## Inputs available in namespace")
        assert f"{unique_input_name}:" in inputs_section

    def test_inline_outputs_listed_in_outputs_section(self):
        lm = ScriptedLM([_wrap("SUBMIT(Q1=1, Q2=2, Q3=3)")])
        rlm = RLM.from_task("Solve Q1..Q3", outputs=["Q1", "Q2", "Q3"],
                            lm=lm, max_turns=1, timeout=5)
        rlm.run()
        outputs_section = _section(_system_msg(lm),
                                   "## Required output fields for SUBMIT()")
        for field in ("Q1", "Q2", "Q3"):
            assert f"- {field}" in outputs_section


# ---------------------------------------------------------------------------
# Output-validation flow
# ---------------------------------------------------------------------------


class TestSubmitValidationViaFromTask:
    def test_missing_required_output_field_fails_validation(self):
        # Two outputs declared; LM only provides one. With max_turns=1 we
        # exhaust the budget and end with output_validation_failed.
        lm = ScriptedLM([_wrap("SUBMIT(answer=1)")])
        rlm = RLM.from_task("Stub.", outputs=["answer", "evidence"],
                            lm=lm, max_turns=1, timeout=5)
        result = rlm.run()
        assert not result.submitted
        last = result.trajectory.turns[-1]
        assert last.validation_errors, "expected validation_errors to be populated"

    def test_all_required_outputs_present_succeeds(self):
        lm = ScriptedLM([_wrap("SUBMIT(answer=1, evidence='proof')")])
        rlm = RLM.from_task("Stub.", outputs=["answer", "evidence"],
                            lm=lm, max_turns=1, timeout=5)
        result = rlm.run()
        assert result.submitted
        assert result.payload == {"answer": 1, "evidence": "proof"}

    def test_no_outputs_declared_accepts_empty_submit(self):
        """``from_task("X")`` with no ``outputs`` accepts ``SUBMIT()`` --
        callers who want validation must declare outputs explicitly."""
        lm = ScriptedLM([_wrap("SUBMIT()")])
        rlm = RLM.from_task("X", lm=lm, max_turns=1, timeout=5)
        result = rlm.run()
        assert result.submitted
        assert result.payload == {}


# ---------------------------------------------------------------------------
# Input binding into the worker namespace
# ---------------------------------------------------------------------------


class TestInputsBoundIntoWorker:
    def test_inputs_are_addressable_by_name_in_first_turn(self):
        # 'foo' is bound to 21; LM uses it directly in code that submits 42.
        lm = ScriptedLM([_wrap("SUBMIT(answer=foo*2)")])
        rlm = RLM.from_task("Double foo", inputs={"foo": 21}, outputs=["answer"],
                            lm=lm, max_turns=1, timeout=5)
        result = rlm.run()
        assert result.submitted
        assert result.payload == {"answer": 42}

    def test_multiple_inputs_all_bound(self):
        lm = ScriptedLM([_wrap("SUBMIT(answer=a + b + c)")])
        rlm = RLM.from_task("Sum a+b+c",
                            inputs={"a": 1, "b": 2, "c": 3},
                            outputs=["answer"],
                            lm=lm, max_turns=1, timeout=5)
        result = rlm.run()
        assert result.submitted
        assert result.payload == {"answer": 6}

    def test_run_kwargs_override_and_extend_factory_inputs(self):
        """``rlm.run({...})`` merges/overrides the ``inputs`` dict that was
        passed to ``from_task``. This lets a single ``RLM`` instance act as
        a reusable task template invoked with per-call inputs."""
        lm = ScriptedLM([_wrap("SUBMIT(answer=foo + bar)")])
        rlm = RLM.from_task(
            "Add foo and bar",
            inputs={"foo": 1},   # default; will be overridden below
            outputs=["answer"],
            lm=lm, max_turns=1, timeout=5,
        )

        result = rlm.run({"foo": 10, "bar": 5})

        assert result.submitted
        assert result.payload == {"answer": 15}


# ---------------------------------------------------------------------------
# Compatibility with the explicit ``signature=`` kwarg
# ---------------------------------------------------------------------------


class TestSignatureKwargCompatibility:
    """``from_task`` is documented (via inline comment in the production
    code) to accept ``signature=...`` via kwargs. When that happens the
    string-signature path takes precedence for output-field extraction
    and the inline task text is appended to the signature in the
    ``## Task`` section. Pin this so a refactor doesn't drop the
    interaction."""

    def test_signature_kwarg_drives_outputs_and_inline_task_is_appended(self):
        lm = ScriptedLM([_wrap("SUBMIT(answer=1)")])
        rlm = RLM.from_task(
            "Inline task details.",
            signature="q -> answer",
            inputs={"q": "question text"},
            outputs=["ignored_by_signature_path"],
            lm=lm, max_turns=1, timeout=5,
        )

        result = rlm.run()
        assert result.submitted

        sys_msg = _system_msg(lm)
        task_section = _section(sys_msg, "## Task")
        outputs_section = _section(sys_msg,
                                   "## Required output fields for SUBMIT()")

        # Both the signature string and the inline task text appear in ## Task.
        assert "q -> answer" in task_section
        assert "Inline task details." in task_section
        # The signature's parsed output ("answer") wins; the inline output
        # name does not appear in the outputs section.
        assert "- answer" in outputs_section
        assert "ignored_by_signature_path" not in outputs_section


# ---------------------------------------------------------------------------
# Direct check of build_system_prompt() in inline-task mode (unit-level)
# ---------------------------------------------------------------------------


class TestBuildSystemPromptInlineTask:
    """Lower-level pin: the prompt builder used by from_task surfaces the
    task into the rendered template even without going through the full
    runtime. Catches regressions in the prompt-rendering layer
    independently from the runtime wiring."""

    def test_task_description_is_rendered(self):
        prompt = build_system_prompt(
            inline_task="Find the minimum spanning tree.",
            inline_outputs=["edges"],
            inputs={"graph": "adjacency-list"},
        )
        task_section = _section(prompt, "## Task")
        inputs_section = _section(prompt, "## Inputs available in namespace")
        outputs_section = _section(prompt,
                                   "## Required output fields for SUBMIT()")
        assert "Find the minimum spanning tree." in task_section
        assert "graph:" in inputs_section
        assert "- edges" in outputs_section

    def test_no_task_renders_explicit_placeholder_not_python_none(self):
        """When neither a signature nor an inline task is provided the
        ``## Task`` section must render an explicit placeholder rather
        than the literal string ``None``."""
        prompt = build_system_prompt(
            inline_outputs=["answer"],
            inputs={},
        )
        task_section = _section(prompt, "## Task")
        # Explicit placeholder used in ``build_system_prompt``.
        assert "(no task description)" in task_section
        # And no Python ``None`` leaked into the rendered section.
        assert "None" not in task_section
