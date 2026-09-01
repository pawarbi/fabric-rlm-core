from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from fabric_rlm import Knowledge, RLM, load_knowledge
from fabric_rlm.knowledge import KnowledgePackage, RegisteredOperation
from fabric_rlm.knowledge_sources import ProfileLimits
from fabric_rlm.onelake_knowledge_store import OneLakeObjectStat


class ScriptedLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, *, messages):
        self.calls += 1
        self.messages.append([dict(message) for message in messages])
        return self.response


class MemoryOneLakeTransport:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.versions: dict[str, int] = {}

    def stat(self, path: str) -> OneLakeObjectStat | None:
        if path not in self.files:
            return None
        return OneLakeObjectStat(
            size=len(self.files[path]),
            etag=f'"v{self.versions[path]}"',
        )

    def read(self, path: str, max_bytes: int) -> bytes:
        return self.files[path][:max_bytes]

    def mkdirs(self, path: str) -> None:
        return None

    def upload(self, path: str, data: bytes) -> None:
        self.files[path] = data
        self.versions[path] = self.versions.get(path, 0) + 1

    def rename_no_clobber(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
    ) -> None:
        if destination in self.files:
            raise FileExistsError
        self._rename(source, destination, source_etag)

    def rename_overwrite(
        self,
        source: str,
        destination: str,
        *,
        source_etag: str | None,
        destination_etag: str | None,
    ) -> None:
        if destination_etag is None:
            if destination in self.files:
                raise RuntimeError("destination changed")
        elif self.stat(destination).etag != destination_etag:
            raise RuntimeError("destination changed")
        self._rename(source, destination, source_etag)

    def _rename(
        self,
        source: str,
        destination: str,
        source_etag: str | None,
    ) -> None:
        if source_etag is not None and self.stat(source).etag != source_etag:
            raise RuntimeError("source changed")
        self.files[destination] = self.files.pop(source)
        self.versions[destination] = self.versions.pop(source) + 1

    def delete(self, path: str) -> None:
        self.files.pop(path, None)
        self.versions.pop(path, None)


def _submit(code: str) -> str:
    return f"```python\n{code}\n```"


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    path.write_text(
        "order_id,amount\n1,10.5\n2,20.0\n",
        encoding="utf-8",
    )
    return path


def test_rlm_learn_profiles_binds_and_persists_sources(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    destination = tmp_path / "knowledge.json"

    knowledge = RLM.learn(
        sources={"orders": source},
        store=destination,
    )

    assert isinstance(knowledge, Knowledge)
    assert knowledge.package.sources[0].source_id == "orders"
    assert knowledge.package.sources[0].family == "csv"
    assert knowledge.bindings["orders"] is source
    assert destination.is_file()
    persisted = destination.read_text(encoding="utf-8")
    assert str(source) not in persisted
    assert "10.5" not in persisted


def test_load_knowledge_reprofiles_and_rebinds_fresh_handles(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)
    destination = tmp_path / "knowledge.json"
    learned = RLM.learn(sources={"orders": source}, store=destination)
    fresh_source = Path(str(source))

    loaded = load_knowledge(
        destination,
        sources={"orders": fresh_source},
    )

    assert isinstance(loaded, Knowledge)
    assert loaded.package.fingerprint == learned.package.fingerprint
    assert loaded.bindings == {"orders": fresh_source}


def test_load_knowledge_rejects_same_locator_source_drift(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    destination = tmp_path / "knowledge.json"
    RLM.learn(sources={"orders": source}, store=destination)
    source.write_text(
        "order_id,amount\n1,10.5\n2,20.0\n3,30.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale.*orders"):
        load_knowledge(destination, sources={"orders": source})


def test_rlm_learn_is_deterministic_for_equivalent_source_mappings(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)

    first = RLM.learn(sources={"orders": source})
    second = RLM.learn(sources={"orders": source})

    assert second.package.package_id == first.package.package_id
    assert second.package.fingerprint == first.package.fingerprint


def test_public_api_round_trips_an_abfss_onelake_store(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    transport = MemoryOneLakeTransport()
    store = (
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/"
        "lakehouse/Files/knowledge/orders.json"
    )

    learned = RLM.learn(
        sources={"orders": source},
        store=store,
        transport=transport,
    )
    loaded = load_knowledge(
        store,
        sources={"orders": Path(str(source))},
        transport=transport,
    )

    assert loaded.package.fingerprint == learned.package.fingerprint
    assert loaded.bindings["orders"] == source


def test_rlm_task_knowledge_exposes_current_bound_sources_and_labels_fallback(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})
    lm = ScriptedLM(_submit("SUBMIT(answer=orders.name)"))

    result = RLM.task(
        "Return the approved source file name.",
        outputs=["answer"],
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.submitted
    assert result.payload == {"answer": "orders.csv"}
    assert result.trajectory.metadata["knowledge_fingerprint"] == (
        knowledge.package.fingerprint
    )
    assert (
        result.trajectory.metadata["knowledge_mode"]
        == "fallback_no_registered_operations"
    )
    prompt = "\n".join(
        message["content"]
        for call in lm.messages
        for message in call
    )
    assert knowledge.package.fingerprint not in prompt
    assert knowledge.package.package_id not in prompt
    assert knowledge.package.sources[0].locator not in prompt
    assert "snapshot_fingerprint" not in prompt
    assert "schema_fingerprint" not in prompt
    assert "snapshot_exact" not in prompt
    assert "numeric_evidence" not in prompt


def test_rlm_from_task_accepts_knowledge(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})
    lm = ScriptedLM(_submit("SUBMIT(answer=orders.name)"))

    result = RLM.from_task(
        "Return the approved source file name.",
        outputs=["answer"],
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "orders.csv"}
    assert result.trajectory.metadata["knowledge_fingerprint"] == (
        knowledge.package.fingerprint
    )


def test_rlm_task_without_knowledge_has_no_knowledge_metadata() -> None:
    lm = ScriptedLM(_submit("SUBMIT(answer='plain')"))

    result = RLM.task(
        "Return a plain answer.",
        outputs=["answer"],
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "plain"}
    assert "knowledge_fingerprint" not in result.trajectory.metadata
    assert "knowledge_mode" not in result.trajectory.metadata


def test_rlm_task_does_not_execute_declared_operations(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    learned = RLM.learn(sources={"orders": source})
    package = KnowledgePackage(
        package_id=learned.package.package_id,
        sources=learned.package.sources,
        operations=(
            RegisteredOperation(
                operation_id="orders.aggregate.v1",
                operation="safe_aggregate",
                required_sources=("orders",),
                output_schema={"value": {"type": "number"}},
                grain="all_orders",
                host_implementation_id="host.safe_aggregate",
                operation_version="1",
                status="active",
            ),
        ),
    )
    knowledge = Knowledge(package=package, bindings=learned.bindings)
    lm = ScriptedLM(_submit("SUBMIT(answer=orders.name)"))

    result = RLM.task(
        "Return the approved source file name.",
        outputs=["answer"],
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "orders.csv"}
    assert (
        result.trajectory.metadata["knowledge_mode"]
        == "registered_operations_unavailable"
    )


def test_rlm_task_rejects_source_drift_before_calling_lm(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})
    source.write_text(
        "order_id,amount\n1,10.5\n2,20.0\n3,30.0\n",
        encoding="utf-8",
    )
    lm = ScriptedLM(_submit("SUBMIT(answer='unsafe')"))

    with pytest.raises(ValueError, match="stale.*orders"):
        RLM.task(
            "Use learned knowledge.",
            outputs=["answer"],
            knowledge=knowledge,
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 0


def test_rlm_task_rejects_persisted_stale_source_before_calling_lm(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)
    learned = RLM.learn(sources={"orders": source})
    stale_profile = replace(learned.package.sources[0], status="stale")
    stale = Knowledge(
        package=KnowledgePackage(
            package_id=learned.package.package_id,
            sources=(stale_profile,),
        ),
        bindings=learned.bindings,
    )
    lm = ScriptedLM(_submit("SUBMIT(answer='unsafe')"))

    with pytest.raises(ValueError, match="not reusable.*orders"):
        RLM.task(
            "Use learned knowledge.",
            outputs=["answer"],
            knowledge=stale,
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 0


@pytest.mark.parametrize("status", ["quarantined", "retired"])
def test_rlm_task_rejects_non_reusable_source_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    source = _csv(tmp_path)
    learned = RLM.learn(sources={"orders": source})
    blocked_profile = replace(learned.package.sources[0], status=status)
    blocked = Knowledge(
        package=KnowledgePackage(
            package_id=learned.package.package_id,
            sources=(blocked_profile,),
        ),
        bindings=learned.bindings,
    )
    lm = ScriptedLM(_submit("SUBMIT(answer='unsafe')"))

    with pytest.raises(ValueError, match="not reusable.*orders"):
        RLM.task(
            "Use learned knowledge.",
            outputs=["answer"],
            knowledge=blocked,
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 0


def test_rlm_task_reuses_learning_profile_limits(tmp_path: Path) -> None:
    source = tmp_path / "large.csv"
    source.write_text(
        "order_id,amount\n" + "".join(f"{index},1\n" for index in range(150_000)),
        encoding="utf-8",
    )
    limits = ProfileLimits(max_input_bytes=2 * 1024 * 1024)
    knowledge = RLM.learn(sources={"orders": source}, limits=limits)
    lm = ScriptedLM(_submit("SUBMIT(answer=orders.name)"))

    result = RLM.task(
        "Return the approved source file name.",
        outputs=["answer"],
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=5,
    ).run()

    assert result.payload == {"answer": "large.csv"}


def test_rlm_task_rejects_explicit_inputs_that_shadow_knowledge(
    tmp_path: Path,
) -> None:
    source = _csv(tmp_path)
    knowledge = RLM.learn(sources={"orders": source})
    lm = ScriptedLM(_submit("SUBMIT(answer='unsafe')"))

    with pytest.raises(ValueError, match="conflict.*orders"):
        RLM.task(
            "Use learned knowledge.",
            inputs={"orders": "different source"},
            outputs=["answer"],
            knowledge=knowledge,
            lm=lm,
            max_turns=1,
            timeout=5,
        ).run()

    assert lm.calls == 0


def test_rlm_task_rejects_invalid_knowledge_at_construction() -> None:
    with pytest.raises(TypeError, match="knowledge must be a Knowledge"):
        RLM.task(
            "Use learned knowledge.",
            outputs=["answer"],
            knowledge=object(),
            lm=ScriptedLM(_submit("SUBMIT(answer='unsafe')")),
        )


def test_load_knowledge_requires_exact_source_aliases(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    destination = tmp_path / "knowledge.json"
    RLM.learn(sources={"orders": source}, store=destination)

    with pytest.raises(ValueError, match="exact aliases"):
        load_knowledge(
            destination,
            sources={"other": source},
        )


def test_public_learn_remains_usable_after_source_module_reload(
    tmp_path: Path,
) -> None:
    import fabric_rlm.knowledge_sources as knowledge_sources

    importlib.reload(knowledge_sources)
    source = _csv(tmp_path)

    knowledge = RLM.learn(
        sources={"orders": source},
        registry=knowledge_sources.SourceAdapterRegistry.default(),
    )

    assert knowledge.package.sources[0].family == "csv"
