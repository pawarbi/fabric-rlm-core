"""Artifact helpers shared by the parent process and worker."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .semantic_model import SemanticModel


@dataclass(frozen=True)
class File:
    """Lightweight file wrapper exposed inside the RLM worker namespace.

    Pass a `ledger` and every read is logged, the same way a SemanticModel
    logs every query: a bound source records its own access, so lineage does
    not depend on the model cooperating.

    The caveat that does not apply to a semantic model: `open(file.path)` still
    works and is not logged. sempy is the only way to reach a semantic model,
    so its log is complete; a file has other doors.
    """

    path: str
    ledger: Any = None

    def __init__(self, path: str | Path, ledger: Any = None):
        object.__setattr__(self, "path", str(Path(path).expanduser()))
        object.__setattr__(self, "ledger", ledger)

    def _log(self, what: str) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.observe(f"{what} {self.name}", source=self.path)
        except Exception:
            pass

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix

    def exists(self) -> bool:
        return Path(self.path).exists()

    def read_bytes(self) -> bytes:
        self._log("read bytes from")
        return Path(self.path).read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        self._log("read text from")
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

    def record(self, label: str, value: Any, *, format: str = "count",
               note: str = "") -> Any:
        """Record a figure taken from this file, cited as coming from it.

        Unlike a query, a value read out of a document cannot be re-executed,
        so this is marked unverified: the source is checkable by a human
        opening the file, not by the machine re-running it.
        """
        if self.ledger is None:
            raise RuntimeError("This File has no ledger. Pass one: "
                               "File(path, ledger=Ledger(...)).")
        return self.ledger.assert_value(label, value, source=self.path,
                                        format=format, note=note)

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
        if value.ledger is not None:
            return {"__fabric_rlm_file__": value.path,
                    "__fabric_rlm_file_ledger__": value.ledger.path}
        return {"__fabric_rlm_file__": value.path}
    if isinstance(value, Path):
        return {"__fabric_rlm_path__": str(value)}
    if isinstance(value, SemanticModel):
        # Only the coordinates cross the wire. The worker rebuilds a live
        # handle, and re-validating there would repeat a network call the
        # parent already made. The ledger travels as its path, so parent and
        # worker append to the same file.
        return {"__fabric_rlm_semantic_model__": {
            "dataset": value.dataset, "workspace": value.workspace,
            "ledger": value.ledger.path if value.ledger is not None else None}}
    if isinstance(value, Ledger):
        return {"__fabric_rlm_ledger__": {"path": value.path}}
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
        "fabric_rlm.SemanticModel, or fabric_rlm.Ledger."
    )


def decode_from_worker_wire(value: Any) -> Any:
    """Decode JSON-safe input values inside the worker process."""

    if isinstance(value, dict):
        if "__fabric_rlm_file__" in value:
            lg = value.get("__fabric_rlm_file_ledger__")
            return File(value["__fabric_rlm_file__"],
                        ledger=Ledger(lg) if lg else None)
        if "__fabric_rlm_path__" in value:
            return Path(value["__fabric_rlm_path__"])
        if "__fabric_rlm_semantic_model__" in value:
            spec = value["__fabric_rlm_semantic_model__"]
            ledger_path = spec.get("ledger")
            return SemanticModel(
                dataset=spec["dataset"],
                workspace=spec.get("workspace"),
                validate=False,
                # reset=False: the worker must append to the run's record, not
                # start a new one.
                ledger=Ledger(ledger_path) if ledger_path else None,
            )
        if "__fabric_rlm_ledger__" in value:
            return Ledger(value["__fabric_rlm_ledger__"]["path"])
        return {k: decode_from_worker_wire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_from_worker_wire(v) for v in value]
    return value
