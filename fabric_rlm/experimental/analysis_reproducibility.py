"""Deterministic seeds and fingerprints for experimental analysis runs."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math


_SEED_DOMAIN = "fabric-rlm.analysis.seed.v1"
_FINGERPRINT_DOMAIN = "fabric-rlm.analysis.fingerprint.v1"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _canonical_value(value: object, path: str = "value") -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} must have string object keys")
        return {
            key: _canonical_value(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{path} must be JSON-compatible")


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value with stable ordering and formatting."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint(value: object) -> str:
    """Return a domain-separated SHA-256 fingerprint for a canonical value."""

    payload = f"{_FINGERPRINT_DOMAIN}\0{canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(
    root_seed: int,
    *,
    dataset_id: str,
    operator_id: str,
    repetition: int = 0,
    fold: int = 0,
) -> int:
    """Derive a stable unsigned 32-bit child seed without ambient random state."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    for value, field_name in (
        (repetition, "repetition"),
        (fold, "fold"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    payload = canonical_json(
        {
            "domain": _SEED_DOMAIN,
            "root_seed": root_seed,
            "dataset_id": _required_text(dataset_id, "dataset_id"),
            "operator_id": _required_text(operator_id, "operator_id"),
            "repetition": repetition,
            "fold": fold,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
