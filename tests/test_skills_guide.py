"""Offline checks of the actual skill docs, including real subprocess execution."""

from collections import Counter
from copy import deepcopy
import csv
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest

from fabric_rlm.skill_loader import SkillLoader


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
# Match the entire outer fence so nested triple-backtick examples stay intact.
FENCE = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<language>[^\n]*)\n"
    r"(?P<code>.*?)^ {0,3}(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def section(text, heading, end=r"^## |\Z"):
    # Headings inside a fenced playbook are not headings of its containing guide.
    visible = FENCE.sub(lambda match: re.sub(r"[^\n]", " ", match[0]), text)
    match = re.search(
        rf"^{re.escape(heading)}[ \t]*\n(.*?)(?={end})",
        visible,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing section: {heading}"
    return text[match.start(1):match.end(1)]


def blocks(text, language):
    return [m["code"] for m in FENCE.finditer(text) if m["language"].strip() == language]


def trusted_verifier(skill):
    assert skill.verifier_present
    assert skill.verifier_source
    namespace = {}
    # Only execute checked-in project documentation, never downloaded playbooks.
    exec(compile(skill.verifier_source, f"{skill.name}:verifier", "exec"), namespace)
    assert callable(namespace.get("verify"))
    return namespace["verify"]


class ScriptedLM:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(deepcopy(messages))
        assert self.scripts, "Unexpected LM call: scripted responses exhausted"
        return "```python\n" + self.scripts.pop(0) + "\n```"


@pytest.fixture
def template():
    return SkillLoader(skill_dir=DOCS, include_packaged=False).load("skill-template")


@pytest.fixture
def custom_loader(tmp_path):
    guide = (DOCS / "skills-guide.md").read_text(encoding="utf-8")
    examples = blocks(section(guide, "## Custom Skills"), "markdown")
    assert len(examples) == 1, "Expected one complete embedded custom playbook"
    (tmp_path / "csv_row_count.md").write_text(examples[0], encoding="utf-8")
    return SkillLoader(skill_dir=tmp_path, include_packaged=False)


@pytest.fixture
def real_worker_defaults(monkeypatch):
    # conftest disables this screen for fake workers; these workers are real.
    monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)


def test_bundled_catalog_matches_loader_metadata_exactly_once():
    loader = SkillLoader()
    names = loader.list_skills()
    assert names
    guide = (DOCS / "skills-guide.md").read_text(encoding="utf-8")
    catalog = section(guide, "## Bundled Catalog")
    rows = [line.strip("| ").split("|") for line in catalog.splitlines() if line.startswith("|")]
    rows = [[cell.strip() for cell in row] for row in rows]
    assert rows[0] == ["Name", "Purpose/use", "When avoid/caveat", "Dependencies", "Required verifier"]
    assert len(rows[1]) == 5 and all(re.fullmatch(r":?-+:?", cell) for cell in rows[1])
    entries = rows[2:]
    assert all(len(row) == 5 for row in entries)
    assert Counter(row[0] for row in entries) == Counter(f"`{name}`" for name in names)
    for name, purpose, caveat, dependencies, verifier in entries:
        skill = loader.load(name.strip("`"))
        assert skill.name == name.strip("`")
        assert skill.title and skill.summary and purpose and caveat
        expected_dependencies = ", ".join(f"`{dep}`" for dep in skill.dependencies) or "none"
        assert dependencies == expected_dependencies, name
        assert set(skill.dependencies) <= set(names), name
        assert verifier in {"yes", "no"}, name
        assert (verifier == "yes") is skill.verifier_present, name
    # Do not freeze the catalog size; any numeric prose claim must track discovery.
    for count in re.findall(r"\b(\d+)\s+bundled\s+(?:skills|playbooks)\b", guide):
        assert int(count) == len(names)


def test_readme_links_to_skills_guide_and_states_no_default_skills():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    skills = section(text, "## Skills")
    assert "(docs/skills-guide.md)" in skills
    prose = " ".join(skills.replace("*", "").lower().split())
    assert "no skills are selected by default" in prose
    for count in re.findall(r"\b(\d+)\s+bundled\s+(?:skills|playbooks)\b", prose):
        assert int(count) == len(SkillLoader().list_skills())


def test_template_metadata_is_a_working_unbundled_example(template):
    assert template.name == "skill-template"
    assert template.title == "csv_summary"
    assert template.summary
    assert template.dependencies == ()
    assert template.applies_when_keywords == ("csvsummary", "csv summary")
    assert template.applies_when_output_fields == template.excludes == ()
    assert template.specificity == "domain"
    assert {"skill-template", "csv_summary"}.isdisjoint(SkillLoader().list_skills())
    trusted_verifier(template)


@pytest.mark.parametrize(
    "summary",
    [
        {"row_count": 0, "total": 0},
        {"row_count": 3, "total": 50},
        {"row_count": 2, "total": -1.5},
        {"row_count": 1, "total": 10**400},
    ],
    ids=["empty", "integer-total", "negative-finite-total", "large-integer"],
)
def test_template_verifier_accepts_well_formed_payloads(template, summary):
    assert trusted_verifier(template)({"summary": summary}) is None


@pytest.mark.parametrize(
    "payload",
    [
        None, [], {}, {"summary": None}, {"summary": []}, {"summary": {}},
        {"summary": {"row_count": 1}}, {"summary": {"total": 0}},
        {"summary": {"row_count": 1, "total": 0, "extra": 1}},
        {"summary": {"row_count": 1, "total": 0}, "extra": 1},
        *({"summary": {"row_count": count, "total": 0}}
          for count in [-1, True, False, 1.0, "1", None]),
        *({"summary": {"row_count": 1, "total": total}}
          for total in [True, False, "50", None, [], float("nan"), float("inf"), float("-inf")]),
    ],
)
def test_template_verifier_rejects_malformed_payloads_with_assertions(template, payload):
    with pytest.raises(AssertionError):
        trusted_verifier(template)(payload)


def test_template_verifier_does_not_guarantee_source_totals(template, tmp_path):
    source = tmp_path / "amounts.csv"
    source.write_text("amount\n17\n25\n8\n", encoding="utf-8")
    with source.open(newline="", encoding="utf-8") as handle:
        amounts = [int(row["amount"]) for row in csv.DictReader(handle)]
    assert sum(amounts) == 50
    wrong = {"summary": {"row_count": len(amounts) + 1, "total": sum(amounts) + 1}}
    assert trusted_verifier(template)(wrong) is None
    prose = " ".join(template.content.split())
    assert "cannot prove" in prose and "matches the data" in prose


def test_template_real_worker_rejects_then_repairs(tmp_path, template, real_worker_defaults):
    from fabric_rlm.runtime import RLM

    (tmp_path / "csv_summary.md").write_text(template.content, encoding="utf-8")
    source = tmp_path / "amounts.csv"
    source.write_text("amount\n17\n25\n8\n", encoding="utf-8")
    loader = SkillLoader(skill_dir=tmp_path, include_packaged=False)
    read_csv = (
        "import csv\n"
        "with open(csv_path, newline='', encoding='utf-8') as handle:\n"
        "    amounts = [int(row['amount']) for row in csv.DictReader(handle)]\n"
        "summary = {'row_count': len(amounts), 'total': sum(amounts)}\n"
        "print('COMPUTED', summary)\n"
    )
    lm = ScriptedLM([
        read_csv + "summary['row_count'] = -1\nprint('BAD', summary)\nSUBMIT(summary=summary)",
        read_csv + "SUBMIT(summary=summary)",
    ])
    result = RLM.from_task(
        "Count CSV records in csv_path and sum the amount column using csv_summary.",
        inputs={"csv_path": str(source)}, outputs={"summary": dict}, lm=lm,
        skill_loader=loader, skills=["csv_summary"], skills_as_cards=False,
        enable_router=False, enable_skill_autoloading=False, enable_verifier=True,
        max_turns=2, timeout=30, block_network=True,
    ).run()
    assert result.submitted, result.failure_reason
    assert result.payload == {"summary": {"row_count": 3, "total": 50}}
    assert result.n_turns == len(lm.messages) == 2 and not lm.scripts
    assert result.ran_any_code and result.integrity_ok
    assert all(turn.error is None for turn in result.turns)
    assert template.content.strip() in lm.messages[0][0]["content"]
    assert "'total': 50" in result.turns[0].stdout
    metadata = result.trajectory.metadata
    assert metadata["skills"] == ["csv_summary"]
    history = metadata["verifier_repair_history"]
    assert len(history) == 1
    assert history[0]["skill"] == "csv_summary" and history[0]["turn"] == 1
    assert history[0]["rejected_payload"] == {"summary": {"row_count": -1, "total": 50}}
    assert "row_count" in history[0]["assertion"]
    assert "`csv_summary` skill verifier" in lm.messages[1][-1]["content"]
    assert [turn.turn_type for turn in result.turns] == ["normal", "verifier_repair"]
    execution = metadata["verifier_execution"]
    assert execution["passed"] == ["skill:csv_summary"]
    assert execution["verified"] is True and execution["degraded"] == []


def test_embedded_custom_playbook_metadata_and_verifier(custom_loader):
    assert custom_loader.list_skills() == ["csv_row_count"]
    skill = custom_loader.load("csv_row_count")
    assert skill.title == "csv_row_count" and skill.summary
    assert skill.dependencies == skill.excludes == skill.applies_when_output_fields == ()
    assert skill.applies_when_keywords == ("csv row count",)
    assert skill.specificity == "domain"
    verify = trusted_verifier(skill)
    for count in [0, 2, 100]:
        assert verify({"result": {"row_count": count}}) is None
    invalid = [None, [], {}, {"result": None}, {"result": []}, {"result": {}},
               {"result": {"row_count": 2, "extra": 1}},
               {"result": {"row_count": 2}, "extra": 1}]
    invalid.extend({"result": {"row_count": count}} for count in
                   [-1, True, False, 2.0, "2", None, float("nan"), float("inf")])
    for payload in invalid:
        with pytest.raises(AssertionError):
            verify(payload)


def test_custom_guide_host_snippets_and_multiline_csv_worker(custom_loader, tmp_path, real_worker_defaults):
    source = tmp_path / "orders.csv"
    source.write_text('\nname,note\nalice,"first\nsecond"\n\nbob,plain\n', encoding="utf-8")
    with source.open(newline="", encoding="utf-8") as handle:
        records = (row for row in csv.reader(handle) if row)
        next(records, None)
        expected = sum(1 for _ in records)
    assert expected == 2
    lm = ScriptedLM([
        "import csv\n"
        "with open(csv_path, encoding='utf-8', newline='') as source:\n"
        "    records = (row for row in csv.reader(source) if row)\n"
        "    next(records, None)\n"
        "    count = sum(1 for row in records)\n"
        "payload = {'result': {'row_count': count}}\n"
        "print('CSV_RECORDS', count)\n"
        "SUBMIT(result=payload['result'])"
    ])
    guide = (DOCS / "skills-guide.md").read_text(encoding="utf-8")
    snippets = blocks(section(guide, "## Custom Skills"), "python")
    assert len(snippets) == 2  # The verifier inside the Markdown fence is not host code.
    namespace = {"lm": lm}
    for snippet in snippets:
        # Substitute only deployment paths and the LM, retaining the documented API calls.
        snippet = snippet.replace('"/lakehouse/default/Files/skills"', repr(str(custom_loader.skill_dir)))
        snippet = snippet.replace('"/lakehouse/default/Files/orders.csv"', repr(str(source)))
        snippet = snippet.replace('FabricLM("gpt-5.1")', "lm")
        exec(compile(snippet, "skills-guide.md:custom-example", "exec"), namespace)
    rlm = namespace["rlm"]
    result = rlm.run()
    assert result.submitted, result.failure_reason
    assert result.payload == {"result": {"row_count": expected}}
    assert result.n_turns == 1 and not lm.scripts
    assert result.ran_any_code and result.integrity_ok
    assert result.turns[0].error is None
    assert "CSV_RECORDS 2" in result.turns[0].stdout
    assert custom_loader.load_text("csv_row_count").strip() in lm.messages[0][0]["content"]
    metadata = result.trajectory.metadata
    assert metadata["skills"] == ["csv_row_count"]
    assert metadata["router_enabled"] is False
    assert metadata["skill_autoloading"] is False
    execution = metadata["verifier_execution"]
    assert execution["passed"] == ["skill:csv_row_count"]
    assert execution["verified"] is True and execution["degraded"] == []


def test_documented_self_tests_execute_verbatim(template, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    authoring = (DOCS / "authoring-skills.md").read_text(encoding="utf-8")
    snippets = blocks(section(authoring, "## Run the Example Locally"), "python")
    assert len(snippets) == 1
    exec(compile(snippets[0], "authoring-skills.md:self-test", "exec"), {})
    assert "valid and" in capsys.readouterr().out
    snippets = blocks(section(template.content, "## Example Self-Test"), "python")
    assert len(snippets) == 1
    exec(compile(snippets[0], "skill-template.md:self-test", "exec"), {"verify": trusted_verifier(template)})


@pytest.mark.parametrize("filename", ["authoring-skills.md", "skill-template.md", "skills-guide.md"])
def test_documentation_python_snippets_compile(filename):
    text = (DOCS / filename).read_text(encoding="utf-8")
    snippets = blocks(text, "python")
    assert snippets
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{filename}:python-block-{index}", "exec")


@pytest.mark.parametrize("filename", ["authoring-skills.md", "skill-template.md", "skills-guide.md"])
def test_skill_documentation_relative_links_resolve(filename):
    path = DOCS / filename
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    text = re.sub(r"(`+).*?\1", "", text)
    definitions = dict(re.findall(r"^\[([^\]]+)\]:\s*<?([^\s>]+)>?", text, re.MULTILINE))
    targets = re.findall(r"\[[^\]\n]+\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)", text)
    for label, reference in re.findall(r"\[([^\]\n]+)\]\[([^\]\n]*)\]", text):
        key = reference or label
        assert key in definitions, f"{filename}: dangling reference {key}"
        targets.append(definitions[key])
    assert targets, f"{filename} has no documentation links"
    for target in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        assert (path.parent / unquote(url.path)).exists(), f"{filename}: missing {target}"


@pytest.mark.parametrize("filename", ["README.md", "QUICKSTART.md"])
def test_public_guides_link_to_skills_guide(filename):
    assert "(docs/skills-guide.md)" in (ROOT / filename).read_text(encoding="utf-8")


def test_authoring_has_no_stale_normative_activation_claims():
    text = (DOCS / "authoring-skills.md").read_text(encoding="utf-8")
    prose = " ".join(FENCE.sub("", text).replace("`", "").replace("*", "").lower().split())
    assert "always active" not in prose
    # Preserve the correct negation, but reject the old affirmative card-only rule.
    prose = prose.replace("not inherently card-only", "not inherently restricted to cards")
    assert not re.search(r"card[- ]only", prose)
