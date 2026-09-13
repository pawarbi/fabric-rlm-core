"""JSON-safe state serialization for RLM worker namespaces."""

from __future__ import annotations

import dataclasses
import datetime as dt
import decimal
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Mapping

DEFAULT_INJECTED_NAMES = {
    "File", "SUBMIT", "ABSTAIN", "predict", "predict_sync", "load_skill", "activate_skill", "list_skills",
    "is_material_change", "restrict_to_candidate_tuples", "validate_analysis_integrity",
}
DEFAULT_MAX_SUBMIT_BYTES = 64 * 1024 * 1024

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


class SubmitPayloadTooLarge(ValueError):
    """Raised when a lossless final payload exceeds its explicit byte limit."""


def validate_max_submit_bytes(value: int) -> int:
    """Validate and return a final-payload byte limit."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"max_submit_bytes must be an int; got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError("max_submit_bytes must be greater than zero")
    return value


_NOT_SCALAR = object()


SECONDS_PER_DAY = 86_400.0


def _is_pandas_missing(value: Any) -> bool:
    """True for ``pd.NaT``/``pd.NA`` without importing pandas.

    Checked by identity against an already-imported pandas: if pandas was never
    imported, the value cannot be one of its singletons. ``NaT`` must be caught
    before the date branch below, because it subclasses ``datetime`` and its
    ``isoformat()`` returns the string ``"NaT"``.
    """

    pandas = sys.modules.get("pandas")
    if pandas is None:
        return False
    return any(
        value is getattr(pandas, name, _NOT_SCALAR) for name in ("NaT", "NA")
    )


def _as_arrow_scalar(value: Any) -> Any:
    """Unwrap a pyarrow-style scalar via ``as_py()``, else ``_NOT_SCALAR``."""

    as_py = getattr(value, "as_py", None)
    if not callable(as_py):
        return _NOT_SCALAR
    try:
        return as_py()
    except Exception:
        return _NOT_SCALAR


def _stable_repr(value: Any, max_chars: int = 300) -> str:
    return _DEFAULT_REPR_ADDRESS.sub("", repr(value))[:max_chars]


def _as_scalar(value: Any) -> Any:
    """Return the Python native behind a 0-d array scalar, else ``_NOT_SCALAR``.

    ``np.float64`` subclasses Python ``float``, but ``np.int64`` and ``np.bool_``
    subclass nothing, so integer and boolean scalars miss the native-type branch
    in ``freeze`` and would be emitted as opaque markers. A correct answer such
    as ``df["qty"].sum()`` then reads back as unusable.

    Detection is duck-typed rather than an ``import numpy``: numpy is an optional
    dependency here, and the same shape covers other array libraries. Requiring
    both ``ndim == 0`` and an empty ``shape`` keeps real containers — arrays,
    Series, DataFrames — out, including single-element 1-D arrays, whose data a
    lone scalar cannot faithfully represent.
    """

    if getattr(value, "ndim", None) != 0:
        return _NOT_SCALAR
    try:
        if tuple(getattr(value, "shape", (0,))) != ():
            return _NOT_SCALAR
    except TypeError:
        return _NOT_SCALAR
    item = getattr(value, "item", None)
    if not callable(item):
        return _NOT_SCALAR
    try:
        unwrapped = item()
    except Exception:
        return _NOT_SCALAR
    # Guard the caller's recursion: only a genuine unwrapping makes progress.
    if type(unwrapped) is type(value):
        return _NOT_SCALAR
    return unwrapped


def freeze(
    value: Any,
    *,
    max_string_length: int | None = 2_000,
    max_collection_items: int | None = 200,
) -> Any:
    """Recursively convert a value to a JSON-safe representation.

    ``None`` disables the corresponding bound. Namespace snapshots use the
    defaults to keep iterative LM feedback compact; final submission payloads
    disable both bounds so answers are never silently truncated.
    """

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
        if max_string_length is None or len(value) <= max_string_length:
            return value
        return value[:max_string_length] + f"...<truncated, total {len(value)} chars>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    # A missing marker is an absent value, not an object worth describing. This
    # must precede the date branch: pd.NaT subclasses datetime.
    if _is_pandas_missing(value):
        return None
    # Scalars that arrive from SQL sources and from date arithmetic. These
    # conversions match LakehouseSource's own row normalization, which now
    # delegates here so the two paths cannot drift apart.
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.time)):  # dt.datetime subclasses dt.date
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds() / SECONDS_PER_DAY
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    scalar = _as_scalar(value)
    if scalar is _NOT_SCALAR:
        scalar = _as_arrow_scalar(value)
    if scalar is not _NOT_SCALAR:
        return freeze(
            scalar,
            max_string_length=max_string_length,
            max_collection_items=max_collection_items,
        )
    if isinstance(value, tuple):
        items = value if max_collection_items is None else value[:max_collection_items]
        return [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in items
        ]
    if isinstance(value, list):
        items = value if max_collection_items is None else value[:max_collection_items]
        frozen = [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in items
        ]
        if max_collection_items is not None and len(value) > max_collection_items:
            frozen.append({"__truncated__": len(value) - max_collection_items})
        return frozen
    if isinstance(value, set):
        items = list(value)
        if max_collection_items is not None:
            items = items[:max_collection_items]
        return [
            freeze(v, max_string_length=max_string_length, max_collection_items=max_collection_items)
            for v in items
        ]
    if isinstance(value, Mapping):
        items = list(value.items())
        selected = items if max_collection_items is None else items[:max_collection_items]
        frozen = {
            str(k): freeze(
                v,
                max_string_length=max_string_length,
                max_collection_items=max_collection_items,
            )
            for k, v in selected
        }
        if max_collection_items is not None and len(items) > max_collection_items:
            frozen["__truncated__"] = len(items) - max_collection_items
        return frozen
    return opaque_marker(value)


def opaque_marker(value: Any) -> dict[str, Any]:
    return {
        "__type__": type(value).__name__,
        "__repr__": _stable_repr(value),
        "__serializable__": False,
    }


def freeze_submit_payload(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MAX_SUBMIT_BYTES,
) -> Any:
    """Convert a final payload losslessly and enforce its UTF-8 JSON byte size."""

    limit = validate_max_submit_bytes(max_bytes)
    frozen = freeze(
        value,
        max_string_length=None,
        max_collection_items=None,
    )
    encoder = json.JSONEncoder(ensure_ascii=False)
    size = 0
    for chunk in encoder.iterencode(frozen):
        size += len(chunk.encode("utf-8"))
        if size > limit:
            raise SubmitPayloadTooLarge(
                f"SUBMIT payload exceeds max_submit_bytes={limit} "
                f"after encoding at least {size} bytes"
            )
    return frozen


def snapshot(
    namespace: Mapping[str, Any],
    *,
    injected_names: set[str] | None = None,
    max_string_length: int | None = 2_000,
    max_collection_items: int | None = 200,
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
