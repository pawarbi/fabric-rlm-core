"""Secure local persistence and explicit rebinding for knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from fabric_rlm.knowledge import (
    KnowledgePackage,
    _logical_identifier,
    _logical_locator,
    canonical_json,
)


_ENVELOPE_VERSION = 1
_ENVELOPE_FIELDS = {"format_version", "package", "package_fingerprint"}
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_FORBIDDEN_FIELD_NAMES = {
    "raw_rows",
    "raw_content",
    "rows",
    "content",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "token",
    "tokens",
    "sas",
    "code",
    "stdout",
    "stderr",
    "traceback",
    "prompt",
    "response",
    "reasoning",
    "chain_of_thought",
    "chainofthought",
    "state",
}
_FORBIDDEN_FIELD_TOKENS = {
    "rows",
    "content",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "token",
    "tokens",
    "sas",
    "code",
    "stdout",
    "stderr",
    "traceback",
    "prompt",
    "response",
    "reasoning",
    "state",
}


@dataclass(frozen=True)
class SourceBindingDescriptor:
    """Persisted-identity claims supplied explicitly by the binding owner."""

    source_id: str
    locator: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            _logical_identifier(self.source_id, "source_id"),
        )
        object.__setattr__(self, "locator", _logical_locator(self.locator))


@dataclass(frozen=True)
class SourceBinding:
    """An explicit descriptor paired with an opaque runtime value."""

    descriptor: SourceBindingDescriptor
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, SourceBindingDescriptor):
            raise ValueError("descriptor must be a SourceBindingDescriptor")


@dataclass(frozen=True)
class BoundKnowledgePackage:
    """An immutable package plus separately held opaque runtime bindings."""

    package: KnowledgePackage
    bindings: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.package, KnowledgePackage):
            raise ValueError("package must be a KnowledgePackage")
        if not isinstance(self.bindings, Mapping):
            raise ValueError("bindings must be an object")
        object.__setattr__(
            self,
            "bindings",
            MappingProxyType(dict(self.bindings)),
        )


def _normalized_field_name(name: str) -> str:
    with_word_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _is_forbidden_field(name: str) -> bool:
    normalized = _normalized_field_name(name)
    if normalized in _FORBIDDEN_FIELD_NAMES:
        return True
    return bool(set(normalized.split("_")) & _FORBIDDEN_FIELD_TOKENS)


def _is_unsafe_string(value: str) -> bool:
    if value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE.match(value):
        return True

    lowered = value.lower()
    if lowered.startswith("file:"):
        return True
    if not _URL_SCHEME.match(value):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    return (
        parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    )


def _validate_persisted_payload(value: object, path: str = "package") -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError(f"{path} must have string keys")
            if _is_forbidden_field(name):
                raise ValueError(
                    f"{path}.{name} is a privacy-forbidden persisted field"
                )
            _validate_persisted_payload(item, f"{path}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_persisted_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, str) and _is_unsafe_string(value):
        raise ValueError(f"{path} contains an unsafe string value")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate field: {key}")
        result[key] = value
    return result


def _parse_envelope(data: bytes) -> KnowledgePackage:
    try:
        text = data.decode("utf-8")
        envelope = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("knowledge package file is not valid UTF-8 JSON") from exc
    if not isinstance(envelope, dict):
        raise ValueError("knowledge package envelope must be an object")

    unknown = sorted(set(envelope) - _ENVELOPE_FIELDS)
    if unknown:
        raise ValueError(
            f"knowledge package envelope contains unknown field: {unknown[0]}"
        )
    missing = sorted(_ENVELOPE_FIELDS - set(envelope))
    if missing:
        raise ValueError(
            f"knowledge package envelope is missing field: {missing[0]}"
        )
    if (
        type(envelope["format_version"]) is not int
        or envelope["format_version"] != _ENVELOPE_VERSION
    ):
        raise ValueError(
            f"knowledge package envelope format_version must be {_ENVELOPE_VERSION}"
        )
    fingerprint = envelope["package_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("package_fingerprint must be a non-empty string")

    payload = envelope["package"]
    _validate_persisted_payload(payload)
    try:
        package = KnowledgePackage.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("persisted package contract is malformed") from exc
    if package.fingerprint != fingerprint:
        raise ValueError("package fingerprint mismatch")
    return package


def _envelope_bytes(package: KnowledgePackage) -> bytes:
    if not isinstance(package, KnowledgePackage):
        raise ValueError("package must be a KnowledgePackage")
    payload = package.to_dict()
    _validate_persisted_payload(payload)
    envelope = {
        "format_version": _ENVELOPE_VERSION,
        "package": payload,
        "package_fingerprint": package.fingerprint,
    }
    return (canonical_json(envelope) + "\n").encode("utf-8")


def _write_temporary(destination: Path, data: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore_original(destination: Path, original: bytes) -> None:
    try:
        if destination.exists() and destination.read_bytes() == original:
            return
        with destination.open("wb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        pass


def save_knowledge_package(
    destination: str | os.PathLike[str],
    package: KnowledgePackage,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a canonical local knowledge package envelope."""

    if type(overwrite) is not bool:
        raise ValueError("overwrite must be boolean")
    path = Path(destination)
    data = _envelope_bytes(package)
    temporary = _write_temporary(path, data)
    original: bytes | None = None
    try:
        if overwrite:
            if path.exists():
                original = path.read_bytes()
            try:
                os.replace(temporary, path)
            except OSError:
                if original is not None:
                    _restore_original(path, original)
                raise
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _bind_package(
    package: KnowledgePackage,
    bindings: Mapping[str, SourceBinding],
) -> BoundKnowledgePackage:
    if not isinstance(bindings, Mapping):
        raise ValueError("bindings must be an object")
    if not bindings:
        raise ValueError("bindings must not be empty")
    if any(not isinstance(alias, str) for alias in bindings):
        raise ValueError("binding aliases must be strings")

    expected = {source.source_id for source in package.sources}
    actual = set(bindings)
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing aliases: {', '.join(missing)}")
        if extras:
            details.append(f"extra aliases: {', '.join(extras)}")
        raise ValueError(
            "bindings must use exact source aliases; " + "; ".join(details)
        )

    sources = {source.source_id: source for source in package.sources}
    runtime_values: dict[str, object] = {}
    for alias in sorted(expected):
        binding = bindings[alias]
        if not isinstance(binding, SourceBinding):
            raise ValueError(f"binding {alias} must be a SourceBinding")
        source = sources[alias]
        if (
            binding.descriptor.source_id != source.source_id
            or binding.descriptor.locator != source.locator
        ):
            raise ValueError(
                f"binding {alias} descriptor does not exactly match the package"
            )
        runtime_values[alias] = binding.value
    return BoundKnowledgePackage(package=package, bindings=runtime_values)


def load_knowledge_package(
    source: str | os.PathLike[str],
    *,
    bindings: Mapping[str, SourceBinding],
) -> BoundKnowledgePackage:
    """Validate package integrity, then bind explicitly supplied runtime values."""

    data = Path(source).read_bytes()
    package = _parse_envelope(data)
    return _bind_package(package, bindings)
