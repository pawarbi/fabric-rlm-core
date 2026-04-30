from pathlib import Path

from fabric_rlm import File, LocalArtifactStore
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker


def test_file_read_write_and_data_uri(tmp_path: Path) -> None:
    file = File(tmp_path / "hello.txt")
    file.write_text("hello")

    assert file.exists()
    assert file.read_text() == "hello"
    assert file.toDict()["name"] == "hello.txt"
    assert file.as_data_uri().startswith("text/plain;base64,") or file.as_data_uri().startswith(
        "data:text/plain;base64,"
    )


def test_local_artifact_store_writes_under_root(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "run")
    file = store.write_text("artifacts/a.txt", "ok")

    assert Path(file.path).read_text() == "ok"
    assert Path(file.path).is_relative_to(tmp_path / "run")


def test_worker_wire_file_roundtrip(tmp_path: Path) -> None:
    file = File(tmp_path / "input.txt")
    encoded = encode_for_worker({"file": file})
    decoded = decode_from_worker_wire(encoded)

    assert isinstance(decoded["file"], File)
    assert decoded["file"].path == file.path

