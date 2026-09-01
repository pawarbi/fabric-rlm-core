"""Secure local persistence and explicit rebinding for knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

from fabric_rlm.knowledge import (
    KnowledgePackage,
    _logical_identifier,
    _logical_locator,
    canonical_json,
)


_ENVELOPE_VERSION = 1
_ENVELOPE_FIELDS = {"format_version", "package", "package_fingerprint"}
MAX_PACKAGE_BYTES = 4 * 1024 * 1024
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SAFE_METADATA_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CREDENTIAL_FIELD_NAMES = {
    "credential",
    "credentials",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "connection_string",
    "access_token",
    "refresh_token",
    "client_secret",
    "secret",
    "secrets",
    "token",
    "tokens",
    "sas",
    "sas_url",
    "sas_token",
    "shared_access_signature",
}
_CREDENTIAL_FIELD_COMPACT = {
    name.replace("_", "") for name in _CREDENTIAL_FIELD_NAMES
}
_SAFE_STRING_METADATA_FIELD = re.compile(
    r"(?:^|_)(?:code|type|status|kind|category|unit|fingerprint|version)$"
)
_SCHEMA_STRING_FIELDS = {
    "type",
    "delta_type",
    "element_delta_type",
    "key_delta_type",
    "value_delta_type",
    "logical_type",
    "physical_type",
    "format",
    "encoding",
    "unit",
    "timezone",
    "kind",
    "category",
    "labels",
}
_CONNECTION_STRING = re.compile(
    r"(?i)(?:^|;)\s*(?:server|data source|host|database|initial catalog|"
    r"user id|uid|password|pwd|accountkey|sharedaccesssignature)\s*="
)
_AUTHORIZATION_VALUE = re.compile(
    r"(?i)^(?:basic|bearer|digest|negotiate|ntlm|sharedaccesssignature)\s+\S+"
)
_RECOGNIZABLE_SECRET_VALUE = re.compile(
    r"(?i)(?:^AKIA[0-9A-Z]{16}$|^gh[pousr]_[A-Za-z0-9]{20,}$|"
    r"(?:^|[?&;])(?:sig|signature|sharedaccesssignature|accountkey)="
    r"[^&;\s]+)"
)


class KnowledgePersistenceError(Exception):
    """Knowledge package persistence could not complete safely."""


class PersistenceIntegrityError(KnowledgePersistenceError):
    """Publication and restoration both failed, so integrity is uncertain."""


@dataclass(frozen=True)
class SourceBindingDescriptor:
    """Host-attested identity metadata supplied by the binding owner.

    This asserts exact source identity and locator matching only. It does not
    validate source drift because adapter profiling is outside this contract.
    """

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
    if normalized in _CREDENTIAL_FIELD_NAMES:
        return True
    compact = normalized.replace("_", "")
    return compact in _CREDENTIAL_FIELD_COMPACT or bool(
        re.search(
            r"(?:^|_)(?:password|passwd|secret|credential|authorization|cookie|"
            r"private_key|api_key|access_token|refresh_token|id_token|"
            r"auth_token|bearer_token|sas)(?:$|_)",
            normalized,
        )
    )


def _is_unsafe_string(value: str) -> bool:
    decoded = value
    for _ in range(len(value) + 1):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    stripped = decoded.strip()
    if (
        stripped.startswith(("/", "\\"))
        or _WINDOWS_DRIVE_PREFIX.match(stripped)
        or stripped.startswith("-----BEGIN ")
        or _AUTHORIZATION_VALUE.match(stripped)
        or _CONNECTION_STRING.search(stripped)
        or _RECOGNIZABLE_SECRET_VALUE.search(stripped)
    ):
        return True

    lowered = stripped.lower()
    if lowered.startswith(("file:", "data:", "mailto:", "jdbc:")):
        return True
    if not _URL_SCHEME.match(stripped):
        return False
    try:
        parsed = urlsplit(stripped)
    except ValueError:
        return True
    return (
        parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    )


def _validate_string_value(value: str, path: str) -> None:
    if _is_unsafe_string(value):
        raise ValueError(f"{path} contains an unsafe string value")


def _validate_diagnostics(
    value: object,
    path: str,
    field_name: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError(f"{path} must have string keys")
            if _is_forbidden_field(name):
                raise ValueError(
                    f"{path}.{name} is a privacy-forbidden persisted field"
                )
            _validate_diagnostics(item, f"{path}.{name}", name)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_diagnostics(item, f"{path}[{index}]", field_name)
        return
    if isinstance(value, str):
        _validate_string_value(value, path)
        if (
            field_name is None
            or not _SAFE_STRING_METADATA_FIELD.search(
                _normalized_field_name(field_name)
            )
            or not _SAFE_METADATA_CODE.fullmatch(value)
        ):
            raise ValueError(
                f"{path} must use bounded metadata codes, not arbitrary free text"
            )
    elif value is not None and type(value) not in {bool, int, float}:
        raise ValueError(f"{path} must contain bounded metadata values")


def _validate_contract_strings(value: object, path: str = "package") -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            _validate_contract_strings(item, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_contract_strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        _validate_string_value(value, path)


def _validate_schema_descriptor(
    value: object,
    path: str,
    field_name: str | None = None,
) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError(f"{path} must have string keys")
            if _is_forbidden_field(name):
                raise ValueError(
                    f"{path}.{name} is a privacy-forbidden schema descriptor"
                )
            _validate_schema_descriptor(item, f"{path}.{name}", name)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_schema_descriptor(item, f"{path}[{index}]", field_name)
        return
    if isinstance(value, str):
        _validate_string_value(value, path)
        if (
            field_name is not None
            and _normalized_field_name(field_name) not in _SCHEMA_STRING_FIELDS
        ) or not _SAFE_METADATA_CODE.fullmatch(value):
            raise ValueError(f"{path} must be a structural schema descriptor")
    elif value is not None and type(value) not in {bool, int, float}:
        raise ValueError(f"{path} must be a structural schema descriptor")


def _validate_source_schema(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        return
    for column_name, descriptor in value.items():
        if not isinstance(column_name, str):
            raise ValueError(f"{path} must have string keys")
        _validate_schema_descriptor(descriptor, f"{path}.{column_name}")


def _validate_persisted_package(payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("package must be an object")
    _validate_contract_strings(payload)
    sources = payload.get("sources", ())
    if isinstance(sources, (list, tuple)):
        for index, source in enumerate(sources):
            if isinstance(source, Mapping):
                _validate_source_schema(
                    source.get("schema", {}),
                    f"package.sources[{index}].schema",
                )
                _validate_diagnostics(
                    source.get("diagnostics", {}),
                    f"package.sources[{index}].diagnostics",
                )
    operations = payload.get("operations", ())
    if isinstance(operations, (list, tuple)):
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                continue
            for field_name in ("parameter_schema", "parameter_defaults"):
                flexible = operation.get(field_name, {})
                if not isinstance(flexible, Mapping):
                    continue
                for name in flexible:
                    if isinstance(name, str) and _is_forbidden_field(name):
                        raise ValueError(
                            f"package.operations[{index}].{field_name}.{name} "
                            "is a privacy-forbidden persisted field"
                        )


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
    _validate_persisted_package(payload)
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
    _validate_persisted_package(payload)
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
    completed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        completed = True
    finally:
        if not completed:
            temporary.unlink(missing_ok=True)
    return temporary


def _read_bounded(path: Path) -> bytes:
    if path.stat().st_size > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")
    with path.open("rb") as stream:
        data = stream.read(MAX_PACKAGE_BYTES + 1)
    if len(data) > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")
    return data


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    except OSError as exc:
        if exc.errno in {
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.ENOSYS,
        }:
            raise KnowledgePersistenceError(
                "atomic no-clobber publication is unsupported"
            ) from exc
        raise


def save_knowledge_package(
    destination: str | os.PathLike[str],
    package: KnowledgePackage,
    *,
    overwrite: bool = False,
) -> None:
    """Publish with atomic visibility and process-level publication semantics."""

    if type(overwrite) is not bool:
        raise ValueError("overwrite must be boolean")
    path = Path(destination)
    data = _envelope_bytes(package)
    if len(data) > MAX_PACKAGE_BYTES:
        raise KnowledgePersistenceError("knowledge package exceeds maximum byte size")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temporary(path, data)
    backup: Path | None = None
    retain_backup = False
    try:
        if overwrite:
            if path.exists():
                backup = _write_temporary(path, _read_bounded(path))
                backup_with_suffix = backup.with_suffix(".bak")
                os.rename(backup, backup_with_suffix)
                backup = backup_with_suffix
            try:
                os.replace(temporary, path)
            except OSError as publication_error:
                destination_is_original = False
                if backup is not None and path.exists():
                    try:
                        destination_is_original = (
                            _read_bounded(path) == _read_bounded(backup)
                        )
                    except (OSError, KnowledgePersistenceError):
                        destination_is_original = False
                if backup is not None and not destination_is_original:
                    try:
                        os.replace(backup, path)
                        backup = None
                    except OSError as restore_error:
                        retain_backup = True
                        raise PersistenceIntegrityError(
                            "publication failed and restore also failed: "
                            f"{publication_error}; {restore_error}; "
                            f"recovery backup retained at {backup}"
                        ) from restore_error
                raise
        else:
            _publish_no_clobber(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        if backup is not None and not retain_backup:
            backup.unlink(missing_ok=True)


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

    data = _read_bounded(Path(source))
    package = _parse_envelope(data)
    return _bind_package(package, bindings)
