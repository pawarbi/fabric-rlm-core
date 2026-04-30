from fabric_rlm import RLM


class MockLM:
    def __call__(self, *, messages):
        return "```python\nanswer = a + b\nSUBMIT(answer=answer)\n```"


if __name__ == "__main__":
    rlm = RLM.from_task(
        task="Add the two input numbers and submit answer.",
        inputs={"a": 2, "b": 3},
        outputs=["answer"],
        lm=MockLM(),
        max_turns=1,
    )
    result = rlm.run()
    print(result.payload)

