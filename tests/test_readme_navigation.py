"""Offline navigation and quickstart checks for the two public entry points."""

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^ {0,3}(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _navigation_targets(text):
    # Example paths and markup inside code are not rendered navigation.
    text = FENCE.sub("", text)
    text = re.sub(r"(`+).*?\1", "", text, flags=re.DOTALL)
    targets = []
    for match in re.finditer(
        r'(?=(!?)\[(?:[^\[\]\n]|\[[^\]\n]*\])*\]'
        r'\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\s*\))', text,
    ):
        if match.start() and text[match.start() - 1] == "!":
            continue
        targets.append((match[2] or match[3], bool(match[1])))
    definitions = dict(re.findall(r"^\[([^\]]+)\]:\s*<?([^\s>]+)>?", text, re.MULTILINE))
    definitions = {" ".join(key.lower().split()): value for key, value in definitions.items()}
    for image, label, reference in re.findall(r"(!?)\[([^\]\n]+)\]\[([^\]\n]*)\]", text):
        key = " ".join((reference or label).lower().split())
        assert key in definitions, f"Dangling Markdown reference: {key}"
        targets.append((definitions[key], bool(image)))
    targets.extend((target, False) for target in definitions.values())

    class Assets(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for name, value in attrs:
                if name == "src" and value:
                    targets.append((value, True))
                elif name == "srcset" and value:
                    targets.extend((item.strip().split()[0], True)
                                   for item in value.split(",") if item.strip())

    Assets().feed(text)
    return targets


@pytest.mark.parametrize("filename", ["README.md", "docs/usage-guide.md"])
def test_local_navigation_and_relative_assets_exist(filename):
    path = ROOT / filename
    targets = _navigation_targets(path.read_text(encoding="utf-8"))
    assert targets, f"{filename}: no navigation found"
    missing = []
    for target, asset in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        local = unquote(url.path)
        if re.search(r"[<>{}]|\.\.\.", local):
            continue
        if local.startswith(("/", "\\")):
            assert not asset, f"{filename}: asset must be relative: {target}"
            continue
        if not (path.parent / local).exists():
            missing.append(target)
    assert not missing, f"{filename}: missing local targets: {missing}"


def test_navigation_extraction_ignores_code_and_reads_images_and_srcset():
    text = '''
`[not a link](missing-inline.md)`
```python
example = "[not a link](missing-example.md)"
```
[guide](docs/usage-guide.md#installation)
![chart](docs/assets/chart.png)
[reference][guide]
[guide]: docs/usage-guide.md
[![License](https://example.org/badge.svg)](LICENSE)
<picture><source srcset="assets/dark.svg 1x, assets/dark-2x.svg 2x">
<img src="assets/light.svg"></picture>
'''
    assert _navigation_targets(text) == [
        ("docs/usage-guide.md#installation", False),
        ("docs/assets/chart.png", True),
        ("LICENSE", False),
        ("https://example.org/badge.svg", True),
        ("docs/usage-guide.md", False),
        ("docs/usage-guide.md", False),
        ("assets/dark.svg", True),
        ("assets/dark-2x.svg", True),
        ("assets/light.svg", True),
    ]


def test_readme_stays_a_short_front_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 250
    assert not re.search(r"<\s*details\b", text, re.IGNORECASE)
    assert "(docs/usage-guide.md)" in text


@pytest.mark.parametrize("submitted,failure_reason", [
    (True, None), (False, None), (False, "Budget exhausted"), (True, "Validation failed"),
])
def test_quickstart_prints_payload_and_inspects_result(submitted, failure_reason, monkeypatch):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(r"^## Quick start in Fabric\s*\n(.*?)(?=^## |\Z)", text,
                        re.MULTILINE | re.DOTALL)
    assert section is not None
    blocks = re.findall(r"^```python\s*\n(.*?)^```", section[1], re.MULTILINE | re.DOTALL)
    examples = [block for block in blocks if "RLM.task(" in block]
    assert len(examples) == 1
    code = compile(examples[0], "README.md:quickstart", "exec")
    result = SimpleNamespace(submitted=submitted, failure_reason=failure_reason,
                             payload={"answer": 42}, inspect=Mock())
    # Replace the entire import surface so this example cannot call a provider or read a file.
    module = ModuleType("fabric_rlm")
    module.FabricLM = Mock()
    module.File = Mock()
    module.LakehouseSource = Mock()
    module.SemanticModel = Mock()
    module.RLM = Mock()
    module.RLM.task.return_value.run.return_value = result
    monkeypatch.setitem(sys.modules, "fabric_rlm", module)
    printer = Mock()
    namespace = {"print": printer}
    exec(code, namespace)
    printer.assert_called_once_with(result.payload)
    result.inspect.assert_called_once_with()
    module.RLM.task.return_value.run.assert_called_once_with()
    call = module.RLM.task.call_args.kwargs
    assert call["inputs"]["actuals"] is module.SemanticModel.return_value
    assert call["inputs"]["sales_detail"] is module.LakehouseSource.return_value
    assert call["inputs"]["targets"] is module.File.return_value
    assert call["inputs"]["report_month"] == "2026-08"
    module.SemanticModel.assert_called_once_with("<semantic-model-id>", workspace="<workspace-id>")
    module.LakehouseSource.assert_called_once_with(
        "abfss://<workspace-id>@onelake.dfs.fabric.microsoft.com/<lakehouse-id>/Tables/dbo/sales"
    )
