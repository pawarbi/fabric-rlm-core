"""JSON-safe state serialization for RLM worker namespaces."""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INJECTED_NAMES = {"File", "SUBMIT", "predict", "load_skill", "activate_skill", "list_skills"}


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
        "__repr__": repr(value)[:300],
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

