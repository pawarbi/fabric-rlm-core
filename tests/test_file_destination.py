"""Tests for parent-published files created by an isolated worker."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import pytest

from fabric_rlm import FileDestination
from fabric_rlm import artifacts


ROOT = (
    "abfss://workspace@onelake.dfs.fabric.microsoft.com/"
    "lakehouse/Files/published"
)
fabric_runtime_only = pytest.mark.skipif(
    os.name != "posix",
    reason="secure OneLake publication requires the Linux Fabric runtime",
)


class FakeFabricFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def mkdirs(self, _path: str) -> bool:
        return True

    def cp(self, source: str, target: str, recurse: bool = False) -> bool:
        assert not recurse
        local_path = url2pathname(unquote(urlsplit(source).path))
        if os.name == "nt":
            local_path = local_path.lstrip("\\/")
        self.files[target] = Path(local_path).read_bytes()
        return True

    def mv(
        self,
        source: str,
        target: str,
        create_path: bool,
        overwrite: bool = False,
    ) -> bool:
        assert create_path
        if target in self.files and not overwrite:
            return False
        self.files[target] = self.files.pop(source)
        return True

    def rm(self, path: str, recurse: bool = False) -> bool:
        assert not recurse
        self.files.pop(path, None)
        return True

    def ls(self, path: str):
        if path not in self.files:
            return []
        return [
            SimpleNamespace(
                path=path,
                name=path.rsplit("/", 1)[-1],
                size=len(self.files[path]),
                isFile=True,
                isDir=False,
            )
        ]


def test_sealable_memfd_uses_libc_when_python_does_not_expose_it(
    monkeypatch,
) -> None:
    calls = []

    class FakeMemfdCreate:
        argtypes = None
        restype = None

        def __call__(self, name, flags):
            calls.append((name, flags))
            return 37

    fake_memfd_create = FakeMemfdCreate()
    monkeypatch.delattr(artifacts.os, "memfd_create", raising=False)
    monkeypatch.setattr(
        artifacts.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(
            memfd_create=fake_memfd_create
        ),
    )

    assert artifacts._create_sealable_memfd("fabric-rlm-publish") == 37
    assert calls == [
        (
            b"fabric-rlm-publish",
            artifacts._MFD_CLOEXEC | artifacts._MFD_ALLOW_SEALING,
        )
    ]


def test_sealable_memfd_surfaces_libc_errno(monkeypatch) -> None:
    class FailedMemfdCreate:
        argtypes = None
        restype = None

        def __call__(self, _name, _flags):
            return -1

    monkeypatch.delattr(artifacts.os, "memfd_create", raising=False)
    monkeypatch.setattr(
        artifacts.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(
            memfd_create=FailedMemfdCreate()
        ),
    )
    monkeypatch.setattr(artifacts.ctypes, "get_errno", lambda: 38)

    with pytest.raises(OSError) as exc_info:
        artifacts._create_sealable_memfd("fabric-rlm-publish")

    assert exc_info.value.errno == 38


@fabric_runtime_only
def test_file_destination_stages_and_publishes_to_onelake(
    monkeypatch,
) -> None:
    fs = FakeFabricFs()
    monkeypatch.setattr("fabric_rlm.artifacts._notebook_fs", lambda: fs)
    destination = FileDestination(ROOT)
    staged = destination.stage("reports/revenue.xlsx")
    staged.write_bytes(b"workbook")

    result = destination.publish(staged)

    published = f"{ROOT}/reports/revenue.xlsx"
    assert fs.files[published] == b"workbook"
    assert result == {
        "path": published,
        "name": "revenue.xlsx",
        "size": 8,
    }


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "../private.xlsx",
        "reports/../../private.xlsx",
        ".",
        "./report.xlsx",
        "reports//report.xlsx",
        "reports/%2e%2e/private.xlsx",
        "reports/report.xlsx/",
        r"reports\private.xlsx",
        "abfss://other/Files/private.xlsx",
        "/absolute/private.xlsx",
    ],
)
def test_file_destination_rejects_unsafe_relative_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    destination = FileDestination(ROOT)

    with pytest.raises(ValueError, match="safe relative path"):
        destination.stage(relative_path)


def test_file_destination_rejects_source_outside_staging_root(
    tmp_path: Path,
) -> None:
    destination = FileDestination(ROOT)
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(PermissionError, match="staging directory"):
        destination.publish(outside, relative_path="private.txt")


@fabric_runtime_only
def test_file_destination_rejects_oversized_artifact() -> None:
    destination = FileDestination(ROOT, max_bytes=4)
    staged = destination.stage("large.xlsx")
    staged.write_bytes(b"12345")

    with pytest.raises(ValueError, match="max_bytes=4"):
        destination.publish(staged)


@fabric_runtime_only
def test_file_destination_does_not_overwrite_by_default(monkeypatch) -> None:
    fs = FakeFabricFs()
    monkeypatch.setattr("fabric_rlm.artifacts._notebook_fs", lambda: fs)
    destination = FileDestination(ROOT)
    staged = destination.stage("report.xlsx")
    staged.write_bytes(b"new")
    published = f"{ROOT}/report.xlsx"
    fs.files[published] = b"old"

    with pytest.raises(FileExistsError, match="already exists"):
        destination.publish(staged)

    assert fs.files[published] == b"old"


@fabric_runtime_only
def test_file_destination_overwrites_only_when_explicit(monkeypatch) -> None:
    fs = FakeFabricFs()
    monkeypatch.setattr("fabric_rlm.artifacts._notebook_fs", lambda: fs)
    destination = FileDestination(ROOT)
    staged = destination.stage("report.xlsx")
    staged.write_bytes(b"new")
    published = f"{ROOT}/report.xlsx"
    fs.files[published] = b"old"

    result = destination.publish(staged, overwrite=True)

    assert fs.files[published] == b"new"
    assert result["path"] == published


@pytest.mark.parametrize(
    "root",
    [
        "C:/lakehouse/Files",
        "https://onelake.dfs.fabric.microsoft.com/lakehouse/Files",
        "abfss://workspace@other.example/lakehouse/Files",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Tables",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Tables/Files",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files/../Tables",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files/%2e%2e/Tables",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse//Files",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files?x=1",
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files#fragment",
    ],
)
def test_file_destination_requires_canonical_onelake_files_scope(root: str) -> None:
    with pytest.raises(ValueError, match="Files scope"):
        FileDestination(root)


@fabric_runtime_only
def test_file_destination_publishes_via_temporary_onelake_path(monkeypatch) -> None:
    fs = FakeFabricFs()
    monkeypatch.setattr("fabric_rlm.artifacts._notebook_fs", lambda: fs)
    destination = FileDestination(ROOT)
    staged = destination.stage("revenue.xlsx")
    staged.write_bytes(b"workbook")

    result = destination.publish(staged)

    target = f"{ROOT}/revenue.xlsx"
    assert result == {"path": target, "name": "revenue.xlsx", "size": 8}
    assert fs.files == {target: b"workbook"}


def test_file_destination_rejects_symlinked_staged_source(tmp_path: Path) -> None:
    destination = FileDestination(ROOT)
    outside = tmp_path / "outside.xlsx"
    outside.write_bytes(b"private")
    staged = Path(destination.stage("report.xlsx").path)
    try:
        staged.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    with pytest.raises(PermissionError, match="regular files"):
        destination.publish(staged)


def test_file_destination_close_removes_owned_staging_directory() -> None:
    destination = FileDestination(ROOT)
    staged_root = Path(destination.staging_root)
    destination.stage("report.xlsx").write_bytes(b"workbook")

    destination.close()
    destination.close()

    assert not staged_root.exists()


def test_file_destination_rejects_replaced_staging_directory(tmp_path: Path) -> None:
    destination = FileDestination(ROOT)
    original = Path(destination.staging_root)
    moved = tmp_path / "original-staging"
    original.rename(moved)
    replacement = Path(destination.staging_root)
    replacement.mkdir()
    staged = replacement / "report.xlsx"
    staged.write_bytes(b"private")

    with pytest.raises(PermissionError, match="identity changed"):
        destination.publish(staged, relative_path="report.xlsx")


@fabric_runtime_only
def test_file_destination_rejects_file_changed_during_snapshot(
    monkeypatch,
) -> None:
    fs = FakeFabricFs()
    monkeypatch.setattr("fabric_rlm.artifacts._notebook_fs", lambda: fs)
    destination = FileDestination(ROOT)
    staged = destination.stage("report.xlsx")
    staged.write_bytes(b"workbook")
    real_read = os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            Path(staged.path).write_bytes(b"changed-content")
        return chunk

    monkeypatch.setattr("fabric_rlm.artifacts.os.read", mutate_after_read)

    with pytest.raises(PermissionError, match="changed while"):
        destination.publish(staged)


def test_file_destination_fails_closed_without_secure_linux_runtime(
    monkeypatch,
) -> None:
    destination = FileDestination(ROOT)
    staged = destination.stage("report.xlsx")
    staged.write_bytes(b"workbook")
    monkeypatch.setattr("fabric_rlm.artifacts.os.name", "nt")

    with pytest.raises(RuntimeError, match="Linux Fabric runtime"):
        destination.publish(staged)
