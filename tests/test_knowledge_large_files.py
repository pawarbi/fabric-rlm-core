"""A learned file larger than the parse budget is still verified exactly.

Found in Fabric: every file over 1 MiB learned with ``snapshot_exact`` false,
registered no operation, and then every task given that knowledge was refused
with "stale knowledge sources detected", although nothing had changed. The
parse budget (``max_input_bytes``) was doing double duty as the hashing budget.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from fabric_rlm import File, ProfileLimits, RLM, load_knowledge
from fabric_rlm.knowledge_preflight import drift_message, preflight_knowledge
from fabric_rlm.knowledge_sources import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    profile_sources,
    snapshot_budget,
)


def _large_csv(path: Path, rows: int = 200_000) -> Path:
    # About 2.6 MB, more than twice the default parse budget of 1 MiB.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("order_id,region,amount\n")
        for index in range(rows):
            handle.write(f"{index},{'NSEW'[index % 4]},{index % 97}.5\n")
    assert path.stat().st_size > 2 * 1024 * 1024
    return path


def test_a_file_over_the_parse_budget_learns_exactly_and_registers_its_operation(
    tmp_path: Path,
) -> None:
    source = _large_csv(tmp_path / "orders.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no "cannot be verified" warning
        knowledge = RLM.learn(sources={"orders": File(source)})

    profile = knowledge.package.sources[0]
    assert profile.diagnostics["snapshot_exact"] is True
    assert profile.diagnostics["input_truncated"] is True     # parsing stayed bounded
    assert profile.diagnostics["snapshot_observed_bytes"] == source.stat().st_size
    assert [op.operation for op in knowledge.package.operations] == ["tabular.aggregate"]

    result = preflight_knowledge(
        knowledge.package, knowledge.bindings, limits=knowledge._limits,
        registry=knowledge._registry,
    )
    assert result.drift == {}


def test_a_change_in_the_middle_of_a_large_file_is_now_detected(tmp_path: Path) -> None:
    source = _large_csv(tmp_path / "orders.csv")
    knowledge = RLM.learn(sources={"orders": File(source)})

    data = bytearray(source.read_bytes())
    middle = len(data) // 2
    data[middle:middle + 1] = b"9" if data[middle:middle + 1] != b"9" else b"8"
    source.write_bytes(bytes(data))                           # same size, head and tail intact

    result = preflight_knowledge(
        knowledge.package, knowledge.bindings, limits=knowledge._limits,
        registry=knowledge._registry,
    )
    assert result.drift == {"orders": "snapshot"}


def test_saved_package_for_a_large_file_reloads(tmp_path: Path) -> None:
    source = _large_csv(tmp_path / "orders.csv")
    store = tmp_path / "knowledge.json"
    RLM.learn(sources={"orders": File(source)}, store=store)

    reloaded = load_knowledge(store, sources={"orders": File(source)})

    assert reloaded.package.sources[0].diagnostics["snapshot_exact"] is True


def test_small_file_fingerprints_are_unchanged_by_the_streaming_hash(tmp_path: Path) -> None:
    # A package saved by an earlier version must still match its source.
    import hashlib

    from fabric_rlm.knowledge import _domain_fingerprint

    source = tmp_path / "small.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    data = source.read_bytes()
    legacy = _domain_fingerprint(
        "local-source-snapshot-v1",
        {
            "size_bytes": len(data),
            "content_digest": hashlib.sha256(data).hexdigest(),
            "snapshot_exact": True,
            "observed_bytes": len(data),
            "observation_code": "full",
        },
    )
    profile = profile_sources({"small": source}, roles={"small": "numeric_evidence"})[0]
    assert profile.snapshot_fingerprint == legacy


def test_beyond_the_snapshot_budget_the_refusal_names_the_real_cause(tmp_path: Path) -> None:
    source = _large_csv(tmp_path / "orders.csv")
    limits = ProfileLimits(max_snapshot_bytes=1024 * 1024)

    with pytest.warns(UserWarning, match="cannot be verified exactly"):
        knowledge = RLM.learn(sources={"orders": File(source)}, limits=limits)

    result = preflight_knowledge(
        knowledge.package, knowledge.bindings, limits=limits, registry=knowledge._registry
    )
    assert result.drift == {"orders": "inexact"}
    message = drift_message(result.drift, limits=limits)
    assert "stale" not in message
    assert "max_snapshot_bytes" in message and "1,048,576" in message and "orders" in message


def test_drift_message_keeps_the_stale_wording_for_real_changes() -> None:
    assert drift_message({"b": "snapshot", "a": "schema"}) == (
        "stale knowledge sources detected: a, b"
    )
    mixed = drift_message({"a": "snapshot", "z": "inexact"}, limits=ProfileLimits())
    assert mixed.startswith("stale knowledge sources detected: a; knowledge sources cannot")


def test_snapshot_budget_defaults_and_overrides() -> None:
    assert snapshot_budget(ProfileLimits()) == DEFAULT_MAX_SNAPSHOT_BYTES
    assert snapshot_budget(ProfileLimits(max_snapshot_bytes=10)) == 1024 * 1024   # never below the parse budget
    assert snapshot_budget(ProfileLimits(max_input_bytes=2**30)) == 2**30
    with pytest.raises(ValueError, match="max_snapshot_bytes"):
        ProfileLimits(max_snapshot_bytes=0)
