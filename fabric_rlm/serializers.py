"""JSON-safe state serialization for RLM worker namespaces."""

from __future__ import annotations

import dataclasses
import json
import re
import types
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INJECTED_NAMES = {"File", "SUBMIT", "predict", "predict_sync", "load_skill", "activate_skill", "list_skills"}

# CPython's default ``object.__repr__`` produces ``<module.path.Class object at
# 0xHEXADDR>``. The hex address changes every Python process and even between
# turn restarts, so it must be stripped from serialized state — otherwise two
# adjacent turns that share the same opaque object (workbook, dataframe, db
# connection, file handle, …) produce non-identical state snapshots, which
# defeats LM prompt caching and produces noise in trajectory diffs.
#
# We are deliberately conservative: only strip the literal `` at 0xHEX``
# pattern when followed by a closing delimiter (``>``, ``,``, ``)``, ``]``).
# This catches the standard ``<… at 0x…>`` form and nested forms like
# ``functools.partial(<function f at 0x…>, 1)`` and
# ``<bound method X.f of <Y object at 0x…>>``, while avoiding stripping prose
# such as ``"price at 0xff per unit"`` from custom ``__repr__`` outputs.
_DEFAULT_REPR_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+(?=[>,)\]])")


def _stable_repr(value: Any, max_chars: int = 300) -> str:
    return _DEFAULT_REPR_ADDRESS.sub("", repr(value))[:max_chars]


def freeze(
    value: Any,
    *,
    max_string_length: int = 2_000,
    max_collection_items: int = 200,
) -> Any:
    """Recursively convert a value to a JSON-safe representation."""

    if hasattr(value, "__frozen__"):
        return freeze(
            value.__frozen__(),
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if hasattr(value, "toDict"):
        return freeze(
            value.toDict(),
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if hasattr(value, "model_dump"):
        return freeze(
            value.model_dump(),
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return freeze(
            dataclasses.asdict(value),
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return value[:max_string_length] + f"...<truncated, total {len(value)} chars>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in value[:max_collection_items]
        ]
    if isinstance(value, list):
        frozen = [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in value[:max_collection_items]
        ]
        if len(value) > max_collection_items:
            frozen.append({"__truncated__": len(value) - max_collection_items})
        return frozen
    if isinstance(value, set):
        items = list(value)
        return [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in items[:max_collection_items]
        ]
    if isinstance(value, Mapping):
        items = list(value.items())
        frozen = {
            str(k): freeze(
                v,
                max_string_length=max_string_length,
                max_collection_items=max_collection_items,
            )
            for k, v in items[:max_collection_items]
        }
        if len(items) > max_collection_items:
            frozen["__truncated__"] = len(items) - max_collection_items
        return frozen
    return opaque_marker(value)


def opaque_marker(value: Any) -> dict[str, Any]:
    return {
        "__type__": type(value).__name__,
        "__repr__": _stable_repr(value),
        "__serializable__": False,
    }


def snapshot(
    namespace: Mapping[str, Any],
    *,
    injected_names: set[str] | None = None,
    max_string_length: int = 2_000,
    max_collection_items: int = 200,
) -> dict[str, Any]:
    """Build a JSON-safe view of a Python namespace."""

    injected = DEFAULT_INJECTED_NAMES if injected_names is None else injected_names
    out: dict[str, Any] = {}
    for name, value in namespace.items():
        if name.startswith("_") or name in injected:
            continue
        if callable(value) or isinstance(value, (types.ModuleType, type)):
            continue
        try:
            frozen = freeze(
                value,
                max_string_length=max_string_length,
                max_collection_items=max_collection_items,
            )
            json.dumps(frozen)
            out[name] = frozen
        except Exception as exc:
            out[name] = {
                "__error__": str(exc),
                "__type__": type(value).__name__,
                "__serializable__": False,
            }
    return out

