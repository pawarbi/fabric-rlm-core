from pathlib import Path

from fabric_rlm import File, FileDestination, LocalArtifactStore
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


def test_worker_wire_file_destination_roundtrip(tmp_path: Path) -> None:
    destination = FileDestination(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files",
        max_bytes=1024,
    )

    decoded = decode_from_worker_wire(encode_for_worker({"destination": destination}))

    worker_destination = decoded["destination"]
    assert isinstance(worker_destination, FileDestination)
    assert worker_destination.root == destination.root
    assert worker_destination.staging_root == destination.staging_root
    assert worker_destination.max_bytes == 1024
    worker_destination.close()
    assert Path(destination.staging_root).exists()
    destination.close()


def test_file_in_an_fstring_is_its_path(tmp_path: Path) -> None:
    # Generated code writes f"read_csv_auto('{data_file}')"; that has to name
    # the file, not the handle's repr.
    target = tmp_path / "orders.csv"
    target.write_text("a\n1\n", encoding="utf-8")
    handle = File(target)

    assert str(handle) == str(target)
    assert f"{handle}" == str(target)
    assert "{}".format(handle) == str(target)
    assert repr(handle) == f"File({str(target)!r})"
    assert open(f"{handle}", encoding="utf-8").read() == "a\n1\n"
