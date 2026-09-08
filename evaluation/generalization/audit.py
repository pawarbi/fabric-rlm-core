from __future__ import annotations

import ast
import io
import os
import tokenize
from pathlib import Path
from typing import Iterable

from fabric_rlm.knowledge import KnowledgePackage, LearnedLesson, SourceProfile
from fabric_rlm.knowledge_api import learn
from fabric_rlm.knowledge_lessons import structural_lessons
from fabric_rlm.knowledge_retrieval import _tokens, lesson_score


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            start = body[0].lineno
            end = getattr(body[0], "end_lineno", start)
            lines.update(range(start, end + 1))
    return lines


def _comment_lines(source: str) -> set[int]:
    return {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }


def _arr_mentions(root: Path) -> dict[str, list[dict[str, object]]]:
    executable: list[dict[str, object]] = []
    documentation: list[dict[str, object]] = []
    for path in sorted((root / "fabric_rlm").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "ARR" not in source:
            continue
        doc_lines = _docstring_lines(ast.parse(source))
        comments = _comment_lines(source)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if "ARR" not in line:
                continue
            record = {
                "path": path.relative_to(root).as_posix(),
                "line": line_number,
                "text": line.strip(),
            }
            if line_number in doc_lines or line_number in comments:
                documentation.append(record)
            else:
                executable.append(record)
    return {
        "executable": executable,
        "documentation_or_comments": documentation,
    }


def _semantic_profile(source_id: str, column_name: str) -> SourceProfile:
    return SourceProfile(
        source_id=source_id,
        family="semantic_model",
        locator=f"semantic_model/v1/{source_id}",
        snapshot_fingerprint=f"snapshot-{source_id}",
        schema_fingerprint=f"schema-{source_id}",
        schema={
            "tables": {"Period": {"type": "table"}},
            "columns": {column_name: {"type": "boolean"}},
            "measures": {},
            "relationships": {},
        },
        diagnostics={"snapshot_exact": True},
    )


def _trigger_lesson() -> LearnedLesson:
    return LearnedLesson(
        lesson_id="lesson.expensive_grain.trigger",
        kind="expensive_grain",
        subject="opaque dimension",
        structured_rule={"grain": ["opaque_dimension"]},
        confidence="medium",
        status="active",
        source_dependencies=("source",),
        source_fingerprints={"source": "schema"},
        basis=("preflight_estimate",),
        dependency_scope="snapshot",
    )


def _profile_families(paths: Iterable[Path]) -> list[str]:
    packages = [
        learn(sources={f"source_{index}": str(path)})
        for index, path in enumerate(paths)
    ]
    return sorted({profile.family for package in packages for profile in package.package.sources})


def run_offline_audit(
    repo_root: str | Path,
    fixture_root: str | Path,
) -> dict[str, object]:
    root = Path(repo_root)
    fixtures = Path(fixture_root)
    descriptive = KnowledgePackage(
        package_id="audit.descriptive",
        sources=(_semantic_profile("descriptive", "Period[IsCurrentQuarter]"),),
    )
    abbreviated = KnowledgePackage(
        package_id="audit.abbreviated",
        sources=(_semantic_profile("abbreviated", "prd[icq]"),),
    )
    descriptive_lessons = structural_lessons(descriptive)
    abbreviated_lessons = structural_lessons(abbreviated)
    trigger = _trigger_lesson()

    small_csv = fixtures / "inventory" / "descriptive" / "inventory_snapshots.csv"
    large_csv = (
        fixtures
        / "service"
        / "descriptive"
        / "ticket_event_history_large.csv"
    )
    small_knowledge = learn(sources={"inventory": str(small_csv)})
    large_knowledge = learn(sources={"events": str(large_csv)})
    large_profile = large_knowledge.package.sources[0]

    return {
        "arr_mentions": _arr_mentions(root),
        "naming_effects": {
            "descriptive_time_construct_lessons": len(descriptive_lessons),
            "abbreviated_time_construct_lessons": len(abbreviated_lessons),
            "english_customer_trigger_score": lesson_score(
                trigger, _tokens("customer")
            ),
            "unfamiliar_account_trigger_score": lesson_score(
                trigger, _tokens("acct")
            ),
        },
        "sources": {
            "real_file_profiles": _profile_families([small_csv]),
            "mocked_adapter_tests": ["lakehouse", "semantic_model"],
            "unsupported_or_unavailable": [
                "generic_sql",
                *(
                    []
                    if os.environ.get("FABRIC_WORKSPACE_ID")
                    and os.environ.get("FABRIC_LAKEHOUSE_ID")
                    else ["real_lakehouse_no_credentials"]
                ),
                *(
                    []
                    if os.environ.get("FABRIC_WORKSPACE_ID")
                    and os.environ.get("FABRIC_SEMANTIC_MODEL_ID")
                    else ["real_semantic_model_no_credentials"]
                ),
            ],
        },
        "large_file": {
            "path": large_csv.relative_to(fixtures).as_posix(),
            "size_bytes": large_csv.stat().st_size,
            "snapshot_exact": large_profile.diagnostics.get("snapshot_exact"),
            "registered_operations": len(large_knowledge.package.operations),
        },
        "learn_only": {
            "file_lessons": len(small_knowledge.package.lessons),
            "small_csv_operations": len(small_knowledge.package.operations),
        },
    }


__all__ = ["run_offline_audit"]
