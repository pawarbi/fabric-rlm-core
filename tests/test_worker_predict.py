import asyncio
import base64

import dspy
from pydantic import BaseModel

from fabric_rlm import _worker
from fabric_rlm.artifacts import File


class StructuredAnswer(BaseModel):
    name: str
    score: int


def test_predict_uses_instructions_pydantic_schemas_and_serializes(monkeypatch) -> None:
    seen = {}

    class FakePredict:
        def __init__(self, sig):
            seen["sig"] = sig

        async def acall(self, **kwargs):
            seen["kwargs"] = kwargs
            return dspy.Prediction(
                answer=StructuredAnswer(name="alpha", score=7),
                nested={"more": [StructuredAnswer(name="beta", score=8)]},
            )

    monkeypatch.setattr(dspy, "Predict", FakePredict)
    monkeypatch.setattr(_worker, "_get_lm", lambda: object())

    result = asyncio.run(
        _worker.predict(
            "text: str -> answer: StructuredAnswer",
            instructions="Extract structured fields.",
            pydantic_schemas={"StructuredAnswer": StructuredAnswer},
            text="alpha",
        )
    )

    assert seen["sig"].instructions == "Extract structured fields."
    assert seen["sig"].output_fields["answer"].annotation is StructuredAnswer
    assert seen["kwargs"] == {"text": "alpha"}
    assert result.answer == {"name": "alpha", "score": 7}
    assert result.nested == {"more": [{"name": "beta", "score": 8}]}


def test_predict_wraps_file_inputs_for_dspy_image_fields(monkeypatch, tmp_path) -> None:
    seen = {}

    class FakePredict:
        def __init__(self, sig):
            seen["sig"] = sig

        def __call__(self, **kwargs):
            seen["kwargs"] = kwargs
            return dspy.Prediction(a="ok")

    monkeypatch.setattr(dspy, "Predict", FakePredict)
    monkeypatch.setattr(_worker, "_get_lm", lambda: object())
    image_file = File(tmp_path / "pixel.png")
    image_file.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )

    result = asyncio.run(
        _worker.predict("image: dspy.Image, q: str -> a: str", image=image_file, q="describe")
    )

    assert seen["sig"].input_fields["image"].annotation is dspy.Image
    assert isinstance(seen["kwargs"]["image"], dspy.Image)
    assert seen["kwargs"]["image"].url.startswith("data:image/")
    assert result.a == "ok"
