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
    assert "`predict_sync(signature, instructions=None, pydantic_schemas=None, **kwargs)`" in prompt
    assert '.label' in prompt
    assert "Prediction object" in prompt
    assert "SUBMIT" in prompt
    assert "SKILL playbooks" not in prompt


def test_initial_user_message_names_inputs() -> None:
    message = build_initial_user_message({"question": "x"})

    assert "question" in message
    assert "bound in your namespace" in message


def test_prompt_includes_declared_output_types() -> None:
    prompt = build_system_prompt(
        inline_task="Return structured data.",
        inline_outputs=["result"],
        inline_output_types={"result": dict},
    )

    assert "- result: dict" in prompt


def test_prompt_describes_each_self_describing_input_in_a_list() -> None:
    class Source:
        def __init__(self, label: str):
            self.label = label

        def __rlm_describe__(self) -> str:
            return f"source {self.label}"

    prompt = build_system_prompt(
        inline_task="Compare sources.",
        inline_outputs=["answer"],
        inputs={"sources": [Source("sales"), Source("support")]},
    )

    assert "sources: list[2]" in prompt
    assert "[0] source sales" in prompt
    assert "[1] source support" in prompt
