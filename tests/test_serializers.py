from dataclasses import dataclass
from pathlib import Path

from fabric_rlm import File
from fabric_rlm.serializers import freeze, snapshot


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

