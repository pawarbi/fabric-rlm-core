from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from fabric_rlm import File
from fabric_rlm.serializers import (
    SubmitPayloadTooLarge,
    freeze,
    freeze_submit_payload,
    snapshot,
)


@dataclass
class Item:
    name: str
    count: int


def test_freeze_supported_values(tmp_path: Path) -> None:
    file = File(tmp_path / "a.txt")
    value = {
        "item": Item("x", 2),
        "path": tmp_path,
        "file": file,
        "items": [1, 2],
    }

    frozen = freeze(value)

    assert frozen["item"] == {"name": "x", "count": 2}
    assert frozen["path"] == str(tmp_path)
    assert frozen["file"]["path"] == file.path
    assert frozen["items"] == [1, 2]


def test_freeze_opaque_object() -> None:
    frozen = freeze(object())

    assert frozen["__type__"] == "object"
    assert frozen["__serializable__"] is False


def test_snapshot_skips_runtime_and_private_names() -> None:
    ns = {
        "_hidden": 1,
        "File": object(),
        "SUBMIT": object(),
        "predict": object(),
        "answer": 42,
        "helper": lambda: None,
    }

    assert snapshot(ns) == {"answer": 42}


def test_freeze_defaults_remain_bounded() -> None:
    frozen = freeze({"text": "x" * 2_500, "rows": list(range(250))})

    assert frozen["text"].endswith("<truncated, total 2500 chars>")
    assert frozen["rows"][-1] == {"__truncated__": 50}


def test_freeze_can_preserve_unbounded_final_payloads() -> None:
    value = {"text": "x" * 10_000, "rows": list(range(500))}

    frozen = freeze(value, max_string_length=None, max_collection_items=None)

    assert frozen == value


def test_submit_payload_limit_uses_utf8_json_bytes() -> None:
    value = {"text": "é"}
    encoded_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    assert freeze_submit_payload(value, max_bytes=encoded_size) == value
    with pytest.raises(
        SubmitPayloadTooLarge,
        match=rf"max_submit_bytes={encoded_size - 1}.*at least {encoded_size} bytes",
    ):
        freeze_submit_payload(value, max_bytes=encoded_size - 1)


@pytest.mark.parametrize("limit", [0, -1])
def test_submit_payload_limit_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_submit_bytes must be greater than zero"):
        freeze_submit_payload({"answer": 42}, max_bytes=limit)


@pytest.mark.parametrize("limit", [True, 1.5, "100"])
def test_submit_payload_limit_must_be_an_integer(limit: object) -> None:
    with pytest.raises(TypeError, match="max_submit_bytes must be an int"):
        freeze_submit_payload({"answer": 42}, max_bytes=limit)  # type: ignore[arg-type]
