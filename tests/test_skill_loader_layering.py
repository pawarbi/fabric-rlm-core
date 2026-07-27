"""Custom skill directories layer over the packaged skills, not replace them.

Before this behaviour existed, ``SkillLoader(skill_dir=...)`` made every bundled
skill unloadable, so following the documented Lakehouse custom-skill workflow
silently cost you ``excel_modify`` and the rest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import SkillLoader

CUSTOM = """\
---
applies_when:
  keywords: ["widget"]
excludes: []
depends_on: []
specificity: domain
---
# {name}
Summary: {summary}

## Body

Do the widget thing.
"""


@pytest.fixture
def custom_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    (d / "widget_playbook.md").write_text(
        CUSTOM.format(name="widget_playbook", summary="Custom widget playbook."),
        encoding="utf-8",
    )
    return d


def test_packaged_skills_remain_loadable_with_a_custom_dir(custom_dir: Path):
    loader = SkillLoader(skill_dir=custom_dir)
    assert loader.load("excel_modify").name == "excel_modify"
    assert loader.load("core").name == "core"
    assert loader.load("widget_playbook").name == "widget_playbook"


def test_list_skills_is_the_union(custom_dir: Path):
    packaged = set(SkillLoader().list_skills())
    listed = set(SkillLoader(skill_dir=custom_dir).list_skills())
    assert packaged <= listed
    assert "widget_playbook" in listed


def test_several_directories_are_all_searched(tmp_path: Path, custom_dir: Path):
    other = tmp_path / "more"
    other.mkdir()
    (other / "gadget_playbook.md").write_text(
        CUSTOM.format(name="gadget_playbook", summary="Custom gadget playbook."),
        encoding="utf-8",
    )
    loader = SkillLoader(skill_dir=[custom_dir, other])
    names = set(loader.list_skills())
    assert {"widget_playbook", "gadget_playbook", "excel_modify"} <= names
    assert loader.load("gadget_playbook").name == "gadget_playbook"


def test_custom_file_overrides_a_bundled_skill_of_the_same_name(tmp_path: Path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "excel_modify.md").write_text(
        CUSTOM.format(name="excel_modify", summary="Overridden on purpose."),
        encoding="utf-8",
    )
    assert SkillLoader(skill_dir=d).load("excel_modify").summary == "Overridden on purpose."


def test_last_directory_wins_on_conflict(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    for p, tag in ((a, "from A"), (b, "from B")):
        p.mkdir()
        (p / "dup.md").write_text(CUSTOM.format(name="dup", summary=tag), encoding="utf-8")
    assert SkillLoader(skill_dir=[a, b]).load("dup").summary == "from B"


def test_include_packaged_false_restores_hermetic_behaviour(custom_dir: Path):
    loader = SkillLoader(skill_dir=custom_dir, include_packaged=False)
    assert loader.list_skills() == ["widget_playbook"]
    with pytest.raises(FileNotFoundError):
        loader.load("excel_modify")


def test_missing_skill_error_names_where_it_looked(custom_dir: Path):
    with pytest.raises(FileNotFoundError) as exc:
        SkillLoader(skill_dir=custom_dir).load("no_such_skill")
    msg = str(exc.value)
    assert "no_such_skill" in msg and str(custom_dir) in msg


def test_nonexistent_custom_dir_still_serves_packaged_skills(tmp_path: Path):
    loader = SkillLoader(skill_dir=tmp_path / "does_not_exist")
    assert "excel_modify" in loader.list_skills()
    assert loader.load("excel_modify").name == "excel_modify"


def test_skill_dir_property_stays_backward_compatible(custom_dir: Path):
    assert SkillLoader(skill_dir=custom_dir).skill_dir == custom_dir
    assert SkillLoader().skill_dir is None


def test_no_arguments_behaves_exactly_as_before():
    loader = SkillLoader()
    assert "excel_modify" in loader.list_skills()
    assert loader.load("core").name == "core"
