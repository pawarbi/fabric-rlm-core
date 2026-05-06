from fabric_rlm.prompts import build_initial_user_message, build_system_prompt


def test_prompt_includes_task_inputs_and_outputs() -> None:
    prompt = build_system_prompt(
        inline_task="Add two numbers.",
        inline_outputs=["answer"],
        inputs={"a": 1, "b": 2},
    )

    assert "Add two numbers." in prompt
    assert "a: int" in prompt
    assert "- answer" in prompt
    assert "blank strings" in prompt
    assert "instructions=None" in prompt
    assert "pydantic_schemas" in prompt
    assert "SUBMIT" in prompt
    assert "SKILL playbooks" not in prompt


def test_initial_user_message_names_inputs() -> None:
    message = build_initial_user_message({"question": "x"})

    assert "question" in message
    assert "bound in your namespace" in message


def test_initial_user_message_with_previews_includes_preview_block() -> None:
    """When preview text is provided, render an INPUT PREVIEWS section."""
    message = build_initial_user_message(
        {"work_xlsx": "/tmp/x.xlsx", "question": "compute totals"},
        previews={"work_xlsx": "Sheet1: 5 rows x 3 cols\n  | A | B | C |"},
    )

    assert "bound in your namespace" in message
    assert "INPUT PREVIEWS" in message
    assert "work_xlsx" in message
    assert "5 rows x 3 cols" in message


def test_initial_user_message_without_previews_omits_block() -> None:
    """No previews dict ⇒ no INPUT PREVIEWS heading (back-compat)."""
    message = build_initial_user_message({"work_xlsx": "/tmp/x.xlsx"})

    assert "INPUT PREVIEWS" not in message


def test_initial_user_message_with_empty_previews_omits_block() -> None:
    """Empty / all-None previews ⇒ no preview block emitted."""
    message_empty = build_initial_user_message(
        {"x": 1}, previews={}
    )
    message_none = build_initial_user_message(
        {"x": 1}, previews={"x": None}
    )

    assert "INPUT PREVIEWS" not in message_empty
    assert "INPUT PREVIEWS" not in message_none


def test_initial_user_message_preview_only_for_known_inputs() -> None:
    """Previews keyed on inputs that aren't bound are silently ignored."""
    message = build_initial_user_message(
        {"a": 1},
        previews={"a": "PREV_A", "b": "PREV_B_NOT_BOUND"},
    )

    assert "PREV_A" in message
    assert "PREV_B_NOT_BOUND" not in message


def test_initial_user_message_drops_non_string_preview_values() -> None:
    """Non-string preview values must not be stringified into the prompt."""
    message = build_initial_user_message(
        {"a": 1},
        previews={"a": 12345},  # type: ignore[dict-item]
    )

    assert "INPUT PREVIEWS" not in message
    assert "12345" not in message


def test_initial_user_message_drops_whitespace_only_preview_values() -> None:
    """Whitespace-only preview strings are treated as empty."""
    message = build_initial_user_message({"a": 1}, previews={"a": "   \n\t"})

    assert "INPUT PREVIEWS" not in message

