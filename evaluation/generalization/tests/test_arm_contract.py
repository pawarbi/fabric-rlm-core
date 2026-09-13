"""Pin why arms B and C pass ``inputs=None``.

The arm design looks wrong at first reading. Arm A passes data sources and no
knowledge package; arms B and C pass a knowledge package and ``inputs=None``,
which reads like the two arms differ in data access as well as in knowledge --
a confound that would make every A-vs-B number meaningless.

It is not a confound. A knowledge package carries its own source bindings, and
``_bind_knowledge_inputs`` merges them into the task inputs, so arms B and C
reach exactly the same data through a different door. Passing the same aliases
again is refused by design, because an alias would otherwise have two possible
meanings in one task.

These tests exist so nobody -- including a later reader of this suite -- has to
rediscover that by spending a live run on it. One already was.
"""

from __future__ import annotations

import pytest

from fabric_rlm.runtime import RLM


class _Package:
    fingerprint = "fp"
    operations = ()
    sources = ()


class _Knowledge:
    def __init__(self, bindings):
        self.bindings = bindings
        self.package = _Package()
        self._limits = None
        self._registry = None


def _rlm_with_knowledge(bindings):
    rlm = RLM.__new__(RLM)
    rlm._knowledge = _Knowledge(bindings)
    rlm.knowledge_execution = "auto"
    return rlm


def test_no_knowledge_leaves_inputs_untouched():
    rlm = RLM.__new__(RLM)
    rlm._knowledge = None
    bound, metadata = rlm._bind_knowledge_inputs({"orders": "orders.csv"})
    assert bound == {"orders": "orders.csv"}
    assert metadata == {}


def test_package_bindings_supply_sources_when_task_passes_none(monkeypatch):
    """Arm B is not data-starved: the package binds the sources itself."""
    import fabric_rlm.knowledge_preflight as preflight_module

    monkeypatch.setattr(
        preflight_module,
        "preflight_knowledge",
        lambda *a, **k: type("P", (), {"drift": ()})(),
    )
    rlm = _rlm_with_knowledge({"orders": "orders.csv", "products": "products.csv"})
    bound, metadata = rlm._bind_knowledge_inputs({})
    assert bound == {"orders": "orders.csv", "products": "products.csv"}
    assert metadata["knowledge_fingerprint"] == "fp"


def test_duplicate_alias_is_refused_rather_than_silently_preferred():
    """An alias must not have two meanings in one task."""
    rlm = _rlm_with_knowledge({"orders": "from_package.csv"})
    with pytest.raises(ValueError) as excinfo:
        rlm._bind_knowledge_inputs({"orders": "from_task.csv"})
    message = str(excinfo.value)
    assert "conflict with knowledge source aliases" in message
    assert "orders" in message


def test_non_conflicting_task_inputs_are_kept_alongside_bindings(monkeypatch):
    import fabric_rlm.knowledge_preflight as preflight_module

    monkeypatch.setattr(
        preflight_module,
        "preflight_knowledge",
        lambda *a, **k: type("P", (), {"drift": ()})(),
    )
    rlm = _rlm_with_knowledge({"orders": "orders.csv"})
    bound, _ = rlm._bind_knowledge_inputs({"calendar": "calendar.csv"})
    assert bound == {"orders": "orders.csv", "calendar": "calendar.csv"}
