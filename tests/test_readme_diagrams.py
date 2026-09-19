"""Accessible, local-only diagram assets with matching light/dark semantics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://www.w3.org/2000/svg"}


@pytest.mark.parametrize("name", ["multisource", "rlm-loop"])
def test_diagram_themes_share_accessible_content(name):
    texts = []
    for theme in ("light", "dark"):
        root = ET.parse(ROOT / "docs/assets" / f"{name}-{theme}.svg").getroot()
        assert root.attrib["role"] == "img"
        assert root.attrib.get("viewBox")
        identifiers = {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
        assert set(root.attrib["aria-labelledby"].split()) <= identifiers
        assert root.find("s:title", NS).text
        assert root.find("s:desc", NS).text
        assert root.find(".//s:script", NS) is None
        assert root.find(".//s:foreignObject", NS) is None
        texts.append(["".join(node.itertext()) for node in root.findall(".//s:text", NS)])
    assert texts[0] == texts[1]
    text = " ".join(texts[0])
    if name == "multisource":
        for label in ("2 semantic models", "2 Delta tables", "3 CSV files", "1 PDF",
                      "Skills (optional context)", "Microsoft Fabric notebook", "Excel workbook"):
            assert label in text
    else:
        for label in ("Main execution loop", "Optional recursive delegation", "SUBMIT candidate",
                      "Configured checks", "Accepted result"):
            assert label in text
        assert "not an automatic full RLM loop" in text


def test_readme_places_workflow_before_quickstart_and_explains_optional_recursion():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.index("multisource-light.svg") < text.index("## Quick start in Fabric")
    assert text.index("## How it works") < text.index("rlm-loop-light.svg")
    assert "recursive delegation is optional" in text
    assert "checks run afterward" in text
