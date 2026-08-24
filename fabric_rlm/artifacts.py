"""Artifact helpers shared by the parent process and worker."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lakehouse import LakehouseSource
from .semantic_model import SemanticModel


@dataclass(frozen=True)
class File:
    """Lightweight file wrapper exposed inside the RLM worker namespace."""

    path: str

    def __init__(self, path: str | Path):
        object.__setattr__(self, "path", str(Path(path).expanduser()))

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix

    def exists(self) -> bool:
        return Path(self.path).exists()

    def read_bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        return Path(self.path).read_text(encoding=encoding)

    def write_bytes(self, data: bytes) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_text(self, text: str, encoding: str = "utf-8") -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)

    def as_data_uri(self, mime: str | None = None) -> str:
        mime = mime or mimetypes.guess_type(self.path)[0] or "application/octet-stream"
        encoded = base64.b64encode(self.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def toDict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path}

    def __frozen__(self) -> dict[str, str]:
        return self.toDict()

    def __repr__(self) -> str:
        return f"File({self.path!r})"

    def __fspath__(self) -> str:
        """Make File usable anywhere os.PathLike is accepted (open, os.path.exists,
        Path(...), shutil, etc.). Without this, models naturally write
        `os.path.exists(file)` and hit ``TypeError: stat: path should be string``.
        """
        return self.path


class LocalArtifactStore:
    """Run-scoped artifact store for local files or mounted Lakehouse paths."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def file(self, *parts: str) -> File:
        return File(self.path(*parts))

    def write_text(self, relative_path: str, text: str, encoding: str = "utf-8") -> File:
        file = self.file(relative_path)
        file.write_text(text, encoding=encoding)
        return file

    def write_bytes(self, relative_path: str, data: bytes) -> File:
        file = self.file(relative_path)
        file.write_bytes(data)
        return file

    def manifest_entry(self, file: File, **metadata: Any) -> dict[str, Any]:
        return {"path": file.path, "name": file.name, **metadata}


def encode_for_worker(value: Any) -> Any:
    """Encode supported Python inputs into JSON-safe values for the worker."""

    if isinstance(value, File):
        return {"__fabric_rlm_file__": value.path}
    if isinstance(value, Path):
        return {"__fabric_rlm_path__": str(value)}
    if isinstance(value, SemanticModel):
        # Only the coordinates cross the wire. The worker rebuilds a live
        # handle, and re-validating there would repeat a network call the
        # parent already made.
        return {"__fabric_rlm_semantic_model__": {
            "dataset": value.dataset, "workspace": value.workspace}}
    if isinstance(value, LakehouseSource):
        return {
            "__fabric_rlm_lakehouse_source__": {
                "root": value.root,
                "tables": list(value.tables),
                "files": list(value.files),
                "catalog": [dict(item) for item in value.catalog or ()],
                "max_sources": value.max_sources,
            }
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [encode_for_worker(v) for v in value]
    if isinstance(value, list):
        return [encode_for_worker(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_for_worker(v) for k, v in value.items()}
    raise TypeError(
        f"Unsupported input type for worker binding: {type(value).__name__}. "
        "Use primitives, dict/list containers, pathlib.Path, fabric_rlm.File, "
        "fabric_rlm.SemanticModel, or fabric_rlm.LakehouseSource."
    )


def decode_from_worker_wire(value: Any) -> Any:
    """Decode JSON-safe input values inside the worker process."""

    if isinstance(value, dict):
        if "__fabric_rlm_file__" in value:
            return File(value["__fabric_rlm_file__"])
        if "__fabric_rlm_path__" in value:
            return Path(value["__fabric_rlm_path__"])
        if "__fabric_rlm_semantic_model__" in value:
            spec = value["__fabric_rlm_semantic_model__"]
            return SemanticModel(
                dataset=spec["dataset"],
                workspace=spec.get("workspace"),
                validate=False,
            )
        if "__fabric_rlm_lakehouse_source__" in value:
            spec = value["__fabric_rlm_lakehouse_source__"]
            return LakehouseSource(
                root=spec["root"],
                tables=spec.get("tables"),
                files=spec.get("files"),
                catalog=spec.get("catalog", []),
                max_sources=spec.get("max_sources", 200),
            )
        return {k: decode_from_worker_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_from_worker_wire(v) for v in value]
    return value
