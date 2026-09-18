"""Offline skill lifecycles through the real subprocess worker and scripted LM."""

from __future__ import annotations

import csv
from copy import deepcopy

import pytest

from fabric_rlm import RLM
from fabric_rlm.skill_loader import SkillLoader


READ_CSV = """import csv
with open(source, newline='', encoding='utf-8') as handle:
    amounts = [int(row['amount']) for row in csv.DictReader(handle)]
total = sum(amounts)
print('CSV_VALUES', amounts, 'CSV_TOTAL', total)
"""
TASK = "Sum every amount in source."


class ScriptedLM:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(deepcopy(messages))
        assert self.blocks, "Unexpected LM call: scripted responses exhausted"
        return "```python\n" + self.blocks.pop(0) + "\n```"


@pytest.fixture
def csv_source(tmp_path_factory, monkeypatch):
    # Restore production screens; conftest disables provenance for fake workers.
    monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    # Routing considers input paths too. Keep test names out of those paths.
    path = tmp_path_factory.mktemp("data") / "amounts.csv"
    path.write_text("amount\n17\n25\n8\n", encoding="utf-8")
    return path


@pytest.fixture
def host_loader(tmp_path):
    directory = tmp_path / "playbooks"
    directory.mkdir()
    (directory / "local_guide.md").write_text(
        "---\nspecificity: domain\ndepends_on: []\n---\n"
        "# Local Guide\nSummary: Local guidance for totals.\n\n"
        "## Procedure\nLOCAL_FULL_BODY: Read every amount before summing.\n",
        encoding="utf-8",
    )
    return SkillLoader(directory, include_packaged=False)


def run_csv(csv_source, blocks=None, *, outputs=None, **options):
    lm = ScriptedLM(blocks if blocks is not None else [READ_CSV + "SUBMIT(total=total)"])
    expected_turns = len(lm.blocks)
    rlm = RLM.from_task(
        TASK,
        outputs=outputs if outputs is not None else ["total"],
        lm=lm,
        max_turns=expected_turns,
        timeout=30,
        block_network=True,
        **options,
    )
    result = rlm.run(inputs={"source": str(csv_source)})
    assert result.submitted, result.failure_reason
    with csv_source.open(newline="", encoding="utf-8") as handle:
        expected = sum(int(row["amount"]) for row in csv.DictReader(handle))
    assert result.payload["total"] == expected == 50
    assert not lm.blocks
    assert result.n_turns == len(lm.messages) == expected_turns
    assert result.ran_any_code and result.integrity_ok
    assert all(turn.error is None for turn in result.turns)
    assert "CSV_VALUES [17, 25, 8] CSV_TOTAL 50" in result.turns[0].stdout
    return result, lm


@pytest.mark.parametrize("router", [False, True], ids=["defaults", "router-core-baseline"])
def test_default_core_lifecycle(csv_source, router):
    # Omit the flag for the default case rather than restating its value.
    result, lm = run_csv(csv_source, **({"enable_router": True} if router else {}))
    metadata = result.trajectory.metadata
    prompt = lm.messages[0][0]["content"]
    assert metadata["skills"] == []
    assert metadata["skill_autoloading"] is False
    assert metadata["router_enabled"] is router
    assert ("## Skill: core\n" in prompt) is router
    assert ("Available SKILLs:" in prompt) is router
    assert ("Preloaded SKILLs:" in prompt) is router
    if router:
        assert "core" in metadata["router_active"]
    else:
        assert "router_active" not in metadata
        assert "router_cards" not in metadata
    execution = metadata["verifier_execution"]
    assert execution["checks"] == []
    assert execution["verified"] is False
    assert execution["degraded"] == []


def test_selected_custom_skill_preloads_full_body_without_verifier(csv_source, host_loader):
    result, lm = run_csv(csv_source, skill_loader=host_loader, skills=["local_guide"])
    assert host_loader.load("local_guide").verifier_source is None
    prompt = lm.messages[0][0]["content"]
    assert "Preloaded SKILLs:" in prompt
    assert "## Skill: local_guide\n" in prompt
    assert host_loader.load_text("local_guide").strip() in prompt
    assert "Skill cards (not active" not in prompt
    assert result.payload == {"total": 50}
    assert result.trajectory.metadata["skills"] == ["local_guide"]
    assert result.trajectory.metadata["verifier_execution"]["checks"] == []
    assert result.trajectory.metadata["verifier_execution"]["verified"] is False


def test_custom_host_discovery_does_not_replace_worker_discovery(csv_source, host_loader):
    result, lm = run_csv(
        csv_source,
        [READ_CSV + "names = list_skills()\nprint('WORKER_SKILLS', names)\nSUBMIT(total=total, names=names)"],
        outputs=["total", "names"],
        skill_loader=host_loader,
        enable_skill_autoloading=True,
    )
    prompt = lm.messages[0][0]["content"]
    assert host_loader.list_skills() == ["local_guide"]
    assert "local_guide: Local Guide" in prompt
    assert "LOCAL_FULL_BODY" not in prompt
    assert "excel_extract:" not in prompt
    assert result.payload["names"] == SkillLoader().list_skills()
    assert "excel_extract" in result.payload["names"]
    assert "local_guide" not in result.payload["names"]
    assert result.trajectory.metadata["skill_autoloading"] is True
    assert result.trajectory.metadata["skills"] == []


@pytest.mark.parametrize("router", [False, True], ids=["router-off", "router-on"])
@pytest.mark.parametrize("helper", ["load_skill", "activate_skill"])
def test_worker_loading_versus_host_activation(csv_source, router, helper):
    activated = router and helper == "activate_skill"
    blocks = [
        READ_CSV
        + f"body = {helper}('excel_extract')\n"
        + "assert isinstance(body, str)\n"
        + "assert '## Required verifier' in body\n"
        + "assert 'def verify(payload):' in body\n"
        + "print(body[body.index('# excel_extract'):][:400])\n",
        # The generic verifier walks dict-valued sections, not top-level row keys.
        "detail = {'row': 0}\nprint('DETAIL', detail)\nSUBMIT(total=total, detail=detail)",
    ]
    validated = []

    def validate(payload):
        validated.append(deepcopy(payload))
        assert payload == {"total": 50, "detail": {"row": 1}}

    options = {}
    if router:
        options.update(enable_router=True, router_baseline_skills=[], max_active_skills=0)
    if activated:
        blocks.append("detail['row'] = 1\nprint('DETAIL', detail)\nSUBMIT(total=total, detail=detail)")
        options["output_validator"] = validate
    result, lm = run_csv(csv_source, blocks, outputs=["total", "detail"], **options)
    metadata = result.trajectory.metadata
    prompt = lm.messages[0][0]["content"]
    assert "## Skill: excel_extract\n" not in prompt
    assert "Preloaded SKILLs:" not in prompt
    assert "# excel_extract" in result.turns[0].stdout
    marker = "[FABRIC_RLM_ACTIVATE]:excel_extract"
    assert (marker in result.turns[0].stdout) is (helper == "activate_skill")
    assert "# excel_extract" in lm.messages[1][-1]["content"]
    assert metadata["skills"] == []
    assert metadata["skill_autoloading"] is False
    assert metadata["router_enabled"] is router
    if router:
        assert metadata["router_active"] == []  # Initial routing, not live activation state.
        assert "excel_extract" not in metadata["router_cards"]
        assert "Available SKILLs:" in prompt
        assert "excel_extract:" in prompt
    execution = metadata["verifier_execution"]
    assert execution["degraded"] == []
    if activated:
        assert result.payload == {"total": 50, "detail": {"row": 1}}
        assert validated == [result.payload]  # Skill rejection short-circuits the host hook.
        assert execution["passed"] == ["skill:excel_extract", "output_validator"]
        assert execution["verified"] is True
        history = metadata["verifier_repair_history"]
        assert len(history) == 1
        assert history[0]["skill"] == "excel_extract"
        assert history[0]["turn"] == 2
        assert history[0]["rejected_payload"] == {"total": 50, "detail": {"row": 0}}
        assert "row must be 1-based int >= 1" in history[0]["assertion"]
        assert "`excel_extract` skill verifier" in lm.messages[2][-1]["content"]
        assert [turn.turn_type for turn in result.turns] == ["normal", "normal", "verifier_repair"]
    else:
        assert result.payload == {"total": 50, "detail": {"row": 0}}
        assert validated == []
        assert execution["checks"] == []
        assert execution["verified"] is False
        assert "verifier_repair_history" not in metadata
        assert all(turn.turn_type == "normal" for turn in result.turns)


def test_packaged_transitive_dependencies_preload_in_order(csv_source):
    result, lm = run_csv(csv_source, skills=["pdf_document_analysis"])
    prompt = lm.messages[0][0]["content"]
    names = ["error_handling", "validation", "pdf_document_analysis"]
    positions = [prompt.index(f"## Skill: {name}\n") for name in names]
    assert positions == sorted(positions)
    loader = SkillLoader()
    for name in names:
        assert prompt.count(f"## Skill: {name}\n") == 1
        assert loader.load_text(name).strip() in prompt
    assert "## Skill: core\n" not in prompt
    assert result.trajectory.metadata["skills"] == ["pdf_document_analysis"]
    assert result.trajectory.metadata["verifier_execution"]["checks"] == []
    assert result.payload == {"total": 50}
