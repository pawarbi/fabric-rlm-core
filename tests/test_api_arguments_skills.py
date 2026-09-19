"""Public skill/budget arguments exercised with a scripted LM and real worker."""

from __future__ import annotations

import csv
from copy import deepcopy

import pytest

from fabric_rlm import RLM
from fabric_rlm.skill_loader import SkillLoader


READ_CSV = """import csv
with open(csv_path, newline='', encoding='utf-8') as source:
    values = [int(row['amount']) for row in csv.DictReader(source)]
total = sum(values)
print('CSV_VALUES', values, 'CSV_TOTAL', total)
"""
SUBMIT = "SUBMIT(total=total)"


class ScriptedLM:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(deepcopy(messages))
        assert self.blocks, "Unexpected LM call: scripted responses exhausted"
        return "```python\n" + self.blocks.pop(0) + "\n```"


@pytest.fixture
def csv_case(tmp_path, monkeypatch):
    # Undo conftest's fake-worker accommodation: use production provenance.
    monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
    path = tmp_path / "amounts.csv"
    path.write_text("amount\n17\n25\n8\n", encoding="utf-8")
    return path


@pytest.fixture
def loader(tmp_path):
    directory = tmp_path / "playbooks"
    directory.mkdir()
    specs = {
        "alternate": ("domain", ["ledgerword"], []),
        "ledger": ("domain", ["ledgerword"], ["helper"]),
        "utility": ("utility", ["ledgerword"], []),
        "helper": ("utility", [], []),
        "baseline": ("core", [], []),
        "checker": ("domain", [], []),
        "fat": ("domain", [], []),
    }
    for name, (specificity, keywords, dependencies) in specs.items():
        text = (
            f"---\nspecificity: {specificity}\n"
            f"applies_when:\n  keywords: {keywords!r}\n"
            f"depends_on: {dependencies!r}\n---\n"
            f"# {name.title()}\nSummary: {name} CSV guidance.\n\n"
            f"BODY_{name.upper()}\n"
        )
        if name == "fat":
            text += "Read all rows before computing the amount total.\n" * 400
        if name == "checker":
            text += """
## Required verifier

```python
def verify(payload):
    import csv
    with open(csv_path, newline='', encoding='utf-8') as source:
        expected = sum(int(row['amount']) for row in csv.DictReader(source))
    assert payload['total'] == expected, 'CSV total mismatch'
```
"""
        (directory / f"{name}.md").write_text(text, encoding="utf-8")
    return SkillLoader(directory, include_packaged=False)


def run_csv(csv_case, loader, *, blocks=None, **arguments):
    validated = []

    def validate(payload):
        with csv_case.open(newline="", encoding="utf-8") as source:
            expected = sum(int(row["amount"]) for row in csv.DictReader(source))
        validated.append(dict(payload))
        assert payload["total"] == expected, "Host CSV total mismatch"

    lm = ScriptedLM(blocks if blocks is not None else [READ_CSV + SUBMIT])
    options = dict(skill_loader=loader, max_turns=3, timeout=30)
    options.update(arguments)
    rlm = RLM.from_task(
        "Sum every amount in csv_path. ledgerword",
        outputs=["total"],
        lm=lm,
        output_validator=validate,
        **options,
    )
    result = rlm.run(inputs={"csv_path": str(csv_case)})
    assert result.submitted, result.failure_reason
    assert result.payload == {"total": 50}
    assert validated and validated[-1] == {"total": 50}
    assert not lm.blocks
    assert len(result.trajectory.turns) == len(lm.messages)
    assert all(turn.error is None for turn in result.trajectory.turns)
    assert "CSV_VALUES [17, 25, 8] CSV_TOTAL 50" in result.trajectory.turns[0].stdout
    execution = result.trajectory.metadata["verifier_execution"]
    assert "output_validator" in execution["passed"]
    assert execution["verified"] and not execution["degraded"]
    return result, lm


def test_skills(csv_case, loader):
    absent, absent_lm = run_csv(csv_case, loader, skills=[])
    present, present_lm = run_csv(csv_case, loader, skills=["checker"])
    assert "BODY_CHECKER" not in absent_lm.messages[0][0]["content"]
    assert "BODY_CHECKER" in present_lm.messages[0][0]["content"]
    assert absent.trajectory.metadata["skills"] == []
    assert present.trajectory.metadata["skills"] == ["checker"]
    assert "skill:checker" not in absent.trajectory.metadata["verifier_execution"]["passed"]
    assert "skill:checker" in present.trajectory.metadata["verifier_execution"]["passed"]


def test_skill_loader(csv_case, loader, tmp_path):
    other_dir = tmp_path / "other_playbooks"
    other_dir.mkdir()
    original = (loader.skill_dir / "checker.md").read_text(encoding="utf-8")
    (other_dir / "checker.md").write_text(
        original.replace("BODY_CHECKER", "OTHER_LOADER_BODY"), encoding="utf-8"
    )
    other = SkillLoader(other_dir, include_packaged=False)
    _, first = run_csv(csv_case, loader, skills=["checker"])
    second_result, second = run_csv(csv_case, other, skills=["checker"])
    assert "BODY_CHECKER" in first.messages[0][0]["content"]
    assert "OTHER_LOADER_BODY" not in first.messages[0][0]["content"]
    assert "OTHER_LOADER_BODY" in second.messages[0][0]["content"]
    assert "BODY_CHECKER" not in second.messages[0][0]["content"]
    assert "skill:checker" in second_result.trajectory.metadata["verifier_execution"]["passed"]


def test_enable_verifier(csv_case, loader):
    results = {}
    for enabled in (False, True):
        result, lm = run_csv(
            csv_case, loader, skills=["checker"], enable_verifier=enabled,
            blocks=[READ_CSV + "SUBMIT(total=sum(values[:-1]))", SUBMIT],
        )
        history = result.trajectory.metadata["verifier_repair_history"]
        assert len(history) == 1
        assert history[0]["rejected_payload"] == {"total": 42}
        assert result.trajectory.turns[1].turn_type == "verifier_repair"
        check = "checker" if enabled else "output_validator"
        assert history[0]["skill"] == check
        feedback = lm.messages[1][-1]["content"]
        assert ("`checker` skill verifier" in feedback) is enabled
        assert ("skill:checker" in result.trajectory.metadata["verifier_execution"]["passed"]) is enabled
        results[enabled] = result.payload
    assert results[False] == results[True] == {"total": 50}


@pytest.mark.parametrize(
    "argument,first,second,first_active,second_active",
    [
        ("enable_router", False, True, set(), {"alternate", "ledger"}),
        ("max_active_skills", 1, 2, {"alternate"}, {"alternate", "ledger"}),
        ("router_baseline_skills", [], ["baseline"],
         {"alternate", "ledger"}, {"baseline", "alternate", "ledger"}),
        ("router_candidate_specificities", ["domain"], ["utility"],
         {"alternate", "ledger"}, {"utility"}),
        ("router_include_dependencies", False, True,
         {"alternate", "ledger"}, {"alternate", "ledger", "helper"}),
    ],
    ids=["enable_router", "max_active_skills", "router_baseline_skills",
         "router_candidate_specificities", "router_include_dependencies"],
)
def test_router_arguments(csv_case, loader, argument, first, second, first_active, second_active):
    for value, expected in ((first, first_active), (second, second_active)):
        options = dict(enable_router=True, max_active_skills=2,
                       router_baseline_skills=[], router_candidate_specificities=["domain"],
                       router_include_dependencies=False)
        options[argument] = value
        result, lm = run_csv(csv_case, loader, **options)
        metadata = result.trajectory.metadata
        assert metadata["router_enabled"] is options["enable_router"]
        assert set(metadata.get("router_active", [])) == expected
        prompt = lm.messages[0][0]["content"]
        for name in loader.list_skills():
            assert (f"BODY_{name.upper()}" in prompt) is (name in expected)
        if argument == "max_active_skills":
            assert metadata["router_cards"] == (["ledger"] if value == 1 else [])
            assert ("Skill cards (bodies not preloaded" in prompt) is (value == 1)


def test_enable_skill_autoloading(csv_case, loader):
    for enabled in (False, True):
        result, lm = run_csv(csv_case, loader, enable_skill_autoloading=enabled)
        prompt = lm.messages[0][0]["content"]
        assert ("Available SKILLs:" in prompt) is enabled
        assert ("checker: Checker" in prompt) is enabled
        assert "BODY_CHECKER" not in prompt
        assert result.trajectory.metadata["skill_autoloading"] is enabled


def test_skills_as_cards(csv_case, loader):
    for cards in (False, True):
        result, lm = run_csv(csv_case, loader, skills=["checker"], skills_as_cards=cards)
        prompt = lm.messages[0][0]["content"]
        # Host-loader cards advertise custom skills; worker discovery is separate.
        assert "checker CSV guidance." in prompt
        assert ("Skill cards (bodies not preloaded" in prompt) is cards
        assert ("BODY_CHECKER" in prompt) is not cards
        assert ("## Required verifier" in prompt) is not cards
        # Card presentation omits the body but explicit verifiers still run.
        assert "skill:checker" in result.trajectory.metadata["verifier_execution"]["passed"]


@pytest.mark.parametrize("argument", ["max_prompt_tokens", "digest_after_turn"])
def test_prompt_budget(csv_case, loader, argument):
    prompts = {}
    for enabled in (False, True):
        if argument == "max_prompt_tokens":
            value = 1 if enabled else 1_000_000
        else:
            value = 2 if enabled else None
        _, lm = run_csv(
            csv_case, loader, skills=["fat"], enable_router=True,
            router_baseline_skills=[], router_candidate_specificities=[],
            blocks=[READ_CSV, "print('RECHECK', sum(values))", SUBMIT],
            **{argument: value},
        )
        prompts[enabled] = [call[0]["content"] for call in lm.messages]
    assert all("BODY_FAT" in prompt for prompt in prompts[False])
    digest_turn = 0 if argument == "max_prompt_tokens" else 2
    for index, prompt in enumerate(prompts[True]):
        assert ("BODY_FAT" in prompt) is (index < digest_turn)
        assert ("## Skill (digest): fat" in prompt) is (index >= digest_turn)
    assert len(prompts[True][-1]) < len(prompts[False][-1])


def test_reserve_finalize_turns(csv_case, loader):
    for reserve in (0, 1):
        _, lm = run_csv(
            csv_case, loader, reserve_finalize_turns=reserve,
            blocks=[READ_CSV, "print('RECHECK', sum(values))", SUBMIT],
        )
        for index, call in enumerate(lm.messages):
            assert ("[BUDGET]" in call[-1]["content"]) is (reserve == 1 and index == 2)
        if reserve:
            assert "Only 1 turn(s) remain. Stop exploring;" in lm.messages[2][-1]["content"]


def test_verbose(csv_case, loader, capsys):
    capsys.readouterr()
    run_csv(csv_case, loader, verbose=False)
    quiet = capsys.readouterr()
    run_csv(csv_case, loader, verbose=True)
    loud = capsys.readouterr()
    assert "=== Turn 1/3 (normal) ===" not in quiet.out
    assert "SUBMIT(total=total)" not in quiet.out
    assert "=== Turn 1/3 (normal) ===" in loud.out
    assert "with open(csv_path" in loud.out
    assert "SUBMIT(total=total)" in loud.out
