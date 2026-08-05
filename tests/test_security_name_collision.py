"""A denied name that collides with something the caller wanted must say so.

`fabric` is on the forbidden-modules list because of the SSH automation package.
In a Fabric notebook - this library's primary environment - a model reaching for
a Power BI semantic model naturally tries `import fabric`, gets told "module
'fabric' is disabled (network egress off)", reads that as "the fabric module is
unavailable", and abandons `sempy.fabric`, which works fine.

Measured on a semantic-model eval before this hint existed: 11 refusals across
three runs, each citing a blockage that did not apply to the path abandoned.
"""

from __future__ import annotations

import pytest

from fabric_rlm.security import SecurityPolicy


@pytest.fixture
def policy():
    return SecurityPolicy.default()


def test_bare_import_fabric_is_still_blocked(policy):
    """The rule itself must not be weakened - only the message improved."""
    assert policy.validate_code("import fabric") is not None


def test_the_block_points_at_sempy(policy):
    v = policy.validate_code("import fabric")
    assert "sempy.fabric" in v
    assert "SSH" in v, "must say which 'fabric' the rule targets"


def test_the_hint_survives_the_call_path_too(policy):
    """Both the import site and the call site produce a violation; a model may
    hit either first, so both must carry the hint."""
    v = policy.validate_code('import fabric\nfabric.run("ls")')
    assert v is not None and "sempy.fabric" in v


@pytest.mark.parametrize("code", [
    "import sempy.fabric as fabric",
    "import sempy.fabric as fabric\nfabric.evaluate_dax('d', 'EVALUATE 1')",
    "from sempy import fabric\nfabric.list_measures('d')",
    "from sempy.fabric import evaluate_dax\nevaluate_dax('d', 'E')",
    "import sempy\nsempy.fabric.evaluate_dax('d', 'E')",
])
def test_sempy_paths_remain_allowed(policy, code):
    """The working path must stay working. This is the regression that matters:
    if any of these ever start failing, semantic-model access breaks."""
    assert policy.validate_code(code) is None, code


@pytest.mark.parametrize("mod", ["requests", "httpx", "paramiko", "smtplib"])
def test_other_denied_modules_get_no_stray_hint(policy, mod):
    """The hint is specific to a real collision, not decoration on every block."""
    v = policy.validate_code(f"import {mod}")
    assert v is not None
    assert "sempy" not in v


def test_hint_table_is_reachable():
    """Guard against the helper being dropped while the table stays."""
    from fabric_rlm.security import _COLLISION_HINTS, _name_collision_hint

    assert "fabric" in _COLLISION_HINTS
    assert _name_collision_hint("fabric")
    assert _name_collision_hint("requests") == ""
