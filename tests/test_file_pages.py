"""File.pages(): documents as a list of labelled pages or chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from fabric_rlm import File
from fabric_rlm.artifacts import Page


def test_pdf_gives_one_page_per_pdf_page(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for text in ("first page rule", "second page rule", "third page rule"):
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    pages = File(path).pages()
    assert [p.number for p in pages] == [1, 2, 3]
    assert [p.label for p in pages] == ["page 1", "page 2", "page 3"]
    assert "second page rule" in pages[1] and isinstance(pages[1], str)


def test_text_with_page_markers_keeps_the_page_numbers(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_text("cover\n\n<!-- page 7 -->\n\nseven\n\n<!-- Page 8 -->\n\neight rule\n", encoding="utf-8")
    pages = File(path).pages()
    assert [p.label for p in pages] == ["before page 1", "page 7", "page 8"]
    assert pages[2].number == 8 and "eight rule" in pages[2]


def test_form_feeds_split_pages(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("one\fTwo\fthree", encoding="utf-8")
    assert [(p.number, p.strip()) for p in File(path).pages()] == [(1, "one"), (2, "Two"), (3, "three")]


def test_text_without_markers_is_chunked_at_headings_with_labels(tmp_path: Path):
    body = "\n\n".join(f"Clause {i}. " + "word " * 60 for i in range(12))
    path = tmp_path / "doc.md"
    path.write_text(f"# Payment Terms\n\n{body}\n\n# Late Charges\n\nA levy of 1.5% applies per fortnight.\n", encoding="utf-8")
    pages = File(path).pages(max_chars=1000)
    assert len(pages) > 2 and all(p.number is None for p in pages)
    assert pages[0].label.startswith("chunk 1 · Payment Terms")
    last = pages[-1]
    assert "1.5% applies" in last and "Late Charges" in last.label
    assert all(len(p) <= 1400 for p in pages)
    # no text is lost
    joined = " ".join(" ".join(p.split()) for p in pages)
    assert joined.count("Clause") == 12


def test_page_is_json_serializable_as_plain_text():
    import json

    page = Page("rule text", "page 3", 3)
    assert json.dumps([page]) == '["rule text"]'
