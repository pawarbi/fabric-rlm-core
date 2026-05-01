"""LongCoT benchmark utilities for fair fabric-rlm comparisons.

The helpers in this module are intentionally dependency-light so Fabric
notebooks can run from a pre-materialized JSONL dataset artifact without
importing Hugging Face `datasets` or pyarrow at runtime.  The optional
`load_longcot_dataset` helper still supports local/offline preprocessing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


LONGCOT_DATASET_NAME = "LongHorizonReasoning/LongCoT"
LONGCOT_GUIDANCE_URLS = {
    "dataset": "https://huggingface.co/datasets/LongHorizonReasoning/LongCoT",
    "benchmark_repo": "https://github.com/LongHorizonReasoning/longcot",
    "leaderboard": "https://longcot.ai/",
    "paper": "https://arxiv.org/abs/2604.14140",
}
DOMAINS = ("logic", "cs", "chemistry", "chess", "math")
DIFFICULTIES = ("easy", "medium", "hard")
LONGCOT_FULL_DIFFICULTIES = ("medium", "hard")
LONGCOT_MINI_DIFFICULTIES = ("easy",)

CS_JSON_OBJECT_TEMPLATES = {"HM", "MFMC", "Scheduling", "TM", "MCM", "LLVM"}
CS_INTEGER_TEMPLATES = {"VLIW", "CodeTrace"}
CS_INTEGER_LIST_TEMPLATES = {"Backprop", "DistMem"}
SUPPORTED_CS_TEMPLATES = (
    CS_JSON_OBJECT_TEMPLATES | CS_INTEGER_TEMPLATES | CS_INTEGER_LIST_TEMPLATES
)

INT_PATTERN = re.compile(r"-?\d+")
INT_CSV_PATTERN = re.compile(r"-?\d+(?:\s*,\s*-?\d+)+")


@dataclass
class LongCoTExample:
    id: str
    domain: str
    question: str
    answer: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    difficulty: str | None = None
    template: str | None = None
    canary: str | None = None

    @property
    def question_id(self) -> str:
        return self.id

    @property
    def prompt(self) -> str:
        return self.question


@dataclass
class LongCoTRunRecord:
    id: str
    domain: str
    output: str
    normalized_output: str
    expected: str | None
    correct: bool | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    supported: bool
    correct: bool | None
    wrong_formatting: bool
    verifier: str
    reason: str | None = None
    normalized_output: str | None = None
    expected_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return response_to_text(response[0]) if response else ""
    if isinstance(response, tuple):
        return response_to_text(response[0]) if response else ""
    if isinstance(response, dict):
        for key in ("content", "text", "message"):
            if key in response:
                return response_to_text(response[key])
        if "choices" in response:
            return response_to_text(response["choices"])
    for attr in ("content", "text", "message"):
        if hasattr(response, attr):
            return response_to_text(getattr(response, attr))
    return str(response)


def extract_solution(text: Any) -> str | None:
    raw = response_to_text(text)
    idx = raw.lower().rfind("solution")
    if idx < 0:
        return None
    eq = raw.find("=", idx)
    if eq < 0:
        return None
    value = raw[eq + 1 :].strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.strip("`").strip()
    return value or None


def normalize_solution(text: Any) -> str:
    """Extract and normalize the required `solution = ...` answer."""

    raw = extract_solution(text)
    if raw is None:
        raw = response_to_text(text).strip()
    raw = raw.strip().strip("`").strip()
    if raw.endswith("."):
        raw = raw[:-1].strip()
    return re.sub(r"\s+", " ", raw)


def format_longcot_prompt(example: LongCoTExample) -> str:
    return (
        "Solve this LongCoT problem. Think through the problem, but the final "
        "answer must include a line exactly in the form `solution = ...`.\n\n"
        f"Domain: {example.domain}\n"
        f"Question:\n{example.question}"
    )


def format_question_only_prompt(example: LongCoTExample) -> str:
    """Return exactly the benchmark prompt text to avoid baseline prompt leakage."""

    return example.question


def verify_exact(example: LongCoTExample, output: Any) -> bool | None:
    if example.answer is None:
        return None
    return normalize_solution(output) == normalize_solution(example.answer)


def verify_response(
    example: LongCoTExample,
    response_text: Any,
    *,
    allow_official_verifier: bool = False,
) -> VerificationResult:
    """Verify a response without silently scoring unsupported templates.

    When the official `longcot` package is available and explicitly allowed, it
    is preferred.  Otherwise this module only scores deterministic CS templates
    implemented locally; all other domains/templates are marked unsupported.
    """

    text = response_to_text(response_text)
    wrong_formatting = extract_solution(text) is None
    expected_sha = stable_sha256(example.answer) if example.answer is not None else None

    if allow_official_verifier:
        official = _verify_with_official_longcot(example, text)
        if official is not None:
            return official

    if example.answer is None:
        return VerificationResult(
            supported=False,
            correct=None,
            wrong_formatting=wrong_formatting,
            verifier="none",
            reason="missing_reference_answer",
            normalized_output=normalize_solution(text),
            expected_sha256=expected_sha,
        )

    if example.domain == "cs":
        try:
            correct, wrong = verify_cs_response(example.template, example.answer, text)
        except ValueError as exc:
            return VerificationResult(
                supported=False,
                correct=None,
                wrong_formatting=wrong_formatting,
                verifier="cs_local",
                reason=str(exc),
                normalized_output=normalize_solution(text),
                expected_sha256=expected_sha,
            )
        return VerificationResult(
            supported=True,
            correct=correct,
            wrong_formatting=wrong,
            verifier=f"cs_local:{example.template}",
            normalized_output=normalize_solution(text),
            expected_sha256=expected_sha,
        )

    return VerificationResult(
        supported=False,
        correct=None,
        wrong_formatting=wrong_formatting,
        verifier="unsupported_without_official_longcot",
        reason=f"domain_not_supported_locally:{example.domain}",
        normalized_output=normalize_solution(text),
        expected_sha256=expected_sha,
    )


def verify_cs_response(
    template: str | None,
    answer_text: Any,
    response_text: Any,
) -> tuple[bool, bool]:
    """Deterministic local verifier for LongCoT CS templates."""

    if template is None:
        raise ValueError("Unsupported CS template: None")

    expected = parse_answer(answer_text)
    text = response_to_text(response_text)
    solution = extract_solution(text)
    search_text = solution if solution is not None else text
    wrong_formatting = solution is None

    if template in CS_JSON_OBJECT_TEMPLATES:
        candidate = extract_last_json_object(search_text)
        if candidate is None:
            candidate = extract_balanced_object(search_text, use_last=True)
        if candidate is None:
            return False, wrong_formatting
        parsed_candidate = parse_answer(candidate)
        return parsed_candidate == expected, wrong_formatting

    if template in CS_INTEGER_TEMPLATES:
        match = INT_PATTERN.search(search_text or "") or INT_PATTERN.search(text or "")
        if match is None:
            return False, wrong_formatting
        return int(match.group(0)) == int(str(expected).strip()), wrong_formatting

    if template in CS_INTEGER_LIST_TEMPLATES:
        expected_list = coerce_int_list(expected) or parse_csv_ints(str(expected))
        predicted = coerce_int_list(parse_list_from_text(search_text))
        if predicted is None:
            csv = INT_CSV_PATTERN.search(search_text or "")
            predicted = parse_csv_ints(csv.group(0)) if csv else None
        if predicted is None:
            predicted = coerce_int_list(parse_list_from_text(text, use_last=True))
        return predicted == expected_list, wrong_formatting

    raise ValueError(f"Unsupported CS template: {template}")


def example_from_row(row: dict[str, Any], *, fallback_id: str = "") -> LongCoTExample:
    question = row.get("question") or row.get("prompt") or row.get("problem")
    if not question:
        raise ValueError("LongCoT row does not contain question/problem/prompt")
    excluded = {
        "id",
        "qid",
        "question_id",
        "domain",
        "category",
        "question",
        "prompt",
        "answer",
        "solution",
        "target",
        "gold",
        "split",
        "difficulty",
        "template",
        "canary",
    }
    return LongCoTExample(
        id=str(row.get("question_id") or row.get("id") or row.get("qid") or fallback_id),
        domain=str(row.get("domain") or row.get("category") or "unknown"),
        question=str(question),
        answer=_first_present(row, ["answer", "solution", "target", "gold"]),
        split=str(row.get("split")) if row.get("split") is not None else None,
        difficulty=str(row.get("difficulty")) if row.get("difficulty") is not None else None,
        template=str(row.get("template")) if row.get("template") is not None else None,
        canary=str(row.get("canary")) if row.get("canary") is not None else None,
        metadata={k: v for k, v in row.items() if k not in excluded},
    )


def example_to_row(example: LongCoTExample, *, include_answer: bool = False) -> dict[str, Any]:
    row = {
        "question_id": example.question_id,
        "domain": example.domain,
        "difficulty": example.difficulty,
        "template": example.template,
        "prompt": example.question,
        "split": example.split,
        "canary": example.canary,
        "metadata": example.metadata,
    }
    if include_answer:
        row["answer"] = example.answer
    elif example.answer is not None:
        row["answer_sha256"] = stable_sha256(example.answer)
    return {k: v for k, v in row.items() if v is not None}


def load_jsonl_dataset(path: str | Path) -> list[LongCoTExample]:
    examples: list[LongCoTExample] = []
    for index, row in enumerate(read_jsonl(path)):
        examples.append(example_from_row(row, fallback_id=str(index)))
    return examples


def load_longcot_dataset(
    *,
    split: str = "train",
    limit: int | None = None,
    dataset_name: str = LONGCOT_DATASET_NAME,
    config_name: str = "all",
) -> list[LongCoTExample]:
    """Load LongCoT through Hugging Face for offline preprocessing only."""

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split)
    examples: list[LongCoTExample] = []
    for index, row in enumerate(dataset):
        examples.append(example_from_row(dict(row), fallback_id=str(index)))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def filter_examples(
    examples: Iterable[LongCoTExample],
    *,
    domains: str | Sequence[str] | None = None,
    difficulties: str | Sequence[str] | None = None,
    templates: str | Sequence[str] | None = None,
    splits: str | Sequence[str] | None = None,
    limit: int | None = None,
) -> list[LongCoTExample]:
    domain_set = _normalize_str_set(domains, lower=True)
    difficulty_set = normalize_difficulties(difficulties)
    template_set = _normalize_str_set(templates)
    split_set = _normalize_str_set(splits)

    selected: list[LongCoTExample] = []
    for example in examples:
        if domain_set is not None and example.domain.lower() not in domain_set:
            continue
        if difficulty_set is not None and (example.difficulty or example.split) not in difficulty_set:
            continue
        if template_set is not None and example.template not in template_set:
            continue
        if split_set is not None and example.split not in split_set:
            continue
        selected.append(example)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def normalize_difficulties(values: str | Sequence[str] | None) -> set[str] | None:
    raw = _normalize_str_list(values)
    if not raw:
        return None
    out: list[str] = []
    for value in raw:
        lowered = value.lower()
        if lowered == "longcot":
            out.extend(LONGCOT_FULL_DIFFICULTIES)
        elif lowered == "longcot-mini":
            out.extend(LONGCOT_MINI_DIFFICULTIES)
        else:
            out.append(lowered)
    return set(out)


def completed_question_ids(
    records_path: str | Path,
    *,
    require_strategies: Sequence[str] = ("direct", "rlm"),
) -> set[str]:
    path = Path(records_path)
    if not path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        qid = row.get("question_id") or row.get("id")
        if not qid:
            continue
        if all(
            isinstance(row.get(strategy), dict) and not row[strategy].get("skipped")
            for strategy in require_strategies
        ):
            completed.add(str(qid))
    return completed


def safe_append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {target}:{line_number}") from exc
    return rows


def summarize_strategy(records: Iterable[dict[str, Any]], strategy: str) -> dict[str, Any]:
    entries = [record.get(strategy, {}) for record in records]
    attempted = [entry for entry in entries if not entry.get("skipped")]
    supported = [entry for entry in attempted if entry.get("supported") is True]
    unsupported = [entry for entry in attempted if entry.get("supported") is False]
    scored = [entry for entry in supported if entry.get("correct") is not None]
    correct = [entry for entry in scored if entry.get("correct") is True]
    wrong_formatting = [entry for entry in attempted if entry.get("wrong_formatting")]
    errored = [entry for entry in attempted if entry.get("error")]
    return {
        "total_records": len(entries),
        "attempted": len(attempted),
        "skipped": len(entries) - len(attempted),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "scored": len(scored),
        "correct": len(correct),
        "accuracy": len(correct) / len(scored) if scored else None,
        "wrong_formatting": len(wrong_formatting),
        "errors": len(errored),
    }


def stable_sha256(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_answer(answer_text: Any) -> Any:
    if not isinstance(answer_text, str):
        return answer_text
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(answer_text)
        except Exception:
            pass
    return answer_text


def extract_last_json_object(text: Any) -> str | None:
    raw = response_to_text(text)
    decoder = json.JSONDecoder()
    best: tuple[int, int] | None = None
    for match in re.finditer(r"\{", raw):
        try:
            parsed, end_rel = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            end = match.start() + end_rel
            if best is None or end > best[1]:
                best = (match.start(), end)
    return None if best is None else raw[best[0] : best[1]]


def extract_balanced_object(text: Any, *, use_last: bool = False) -> str | None:
    return _extract_balanced(text, "{", "}", use_last=use_last)


def extract_balanced_list(text: Any, *, use_last: bool = False) -> str | None:
    return _extract_balanced(text, "[", "]", use_last=use_last)


def parse_list_from_text(text: Any, *, use_last: bool = False) -> list[Any] | None:
    bracketed = extract_balanced_list(text, use_last=use_last)
    if not bracketed:
        return None
    parsed = parse_answer(bracketed)
    return parsed if isinstance(parsed, list) else None


def coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            out.append(item)
        elif re.fullmatch(INT_PATTERN, str(item).strip()):
            out.append(int(str(item).strip()))
        else:
            return None
    return out


def parse_csv_ints(text: str) -> list[int] | None:
    parts = [part.strip() for part in (text or "").split(",")]
    if len(parts) < 2 or any(not re.fullmatch(INT_PATTERN, part) for part in parts):
        return None
    return [int(part) for part in parts]


def run_longcot_subset(
    examples: Iterable[LongCoTExample],
    runner: Callable[[LongCoTExample], Any],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    records: list[LongCoTRunRecord] = []
    for index, example in enumerate(examples):
        if limit is not None and index >= limit:
            break
        output = runner(example)
        normalized = normalize_solution(output)
        correct = verify_exact(example, output)
        records.append(
            LongCoTRunRecord(
                id=example.id,
                domain=example.domain,
                output=str(output),
                normalized_output=normalized,
                expected=normalize_solution(example.answer) if example.answer is not None else None,
                correct=correct,
                metadata=example.metadata,
            )
        )

    scored = [record for record in records if record.correct is not None]
    correct_count = sum(1 for record in scored if record.correct)
    return {
        "count": len(records),
        "scored_count": len(scored),
        "correct_count": correct_count,
        "accuracy": correct_count / len(scored) if scored else None,
        "records": [record.__dict__ for record in records],
    }


def _verify_with_official_longcot(
    example: LongCoTExample,
    response_text: str,
) -> VerificationResult | None:
    try:
        import longcot  # type: ignore[import-not-found]

        try:
            from longcot._types import (  # type: ignore[import-not-found]
                ChemistryVerifyOptions,
                MathVerifyOptions,
                Question,
                VerifyOptions,
            )
        except Exception:
            Question = getattr(longcot, "Question", None)
            VerifyOptions = None
            ChemistryVerifyOptions = None
            MathVerifyOptions = None
        if Question is None:
            return None
    except Exception:
        return None

    if not hasattr(longcot, "verify"):
        return None

    try:
        question = Question(
            question_id=example.question_id,
            domain=example.domain,
            difficulty=example.difficulty,
            prompt=example.question,
            problem=example.metadata.get("problem"),
            answer=parse_answer(example.answer),
        )
        options = None
        if VerifyOptions and ChemistryVerifyOptions and MathVerifyOptions:
            options = VerifyOptions(
                math=MathVerifyOptions(enable_fallback=False),
                chemistry=ChemistryVerifyOptions(enable_fallback=False),
            )
        correct = (
            longcot.verify(question, response_text, options=options)
            if options is not None
            else longcot.verify(question, response_text)
        )
    except Exception as exc:
        return VerificationResult(
            supported=False,
            correct=None,
            wrong_formatting=extract_solution(response_text) is None,
            verifier="official_longcot",
            reason=f"official_verifier_error:{exc!r}",
            normalized_output=normalize_solution(response_text),
            expected_sha256=stable_sha256(example.answer) if example.answer is not None else None,
        )

    return VerificationResult(
        supported=True,
        correct=bool(correct),
        wrong_formatting=extract_solution(response_text) is None,
        verifier="official_longcot_no_fallback",
        normalized_output=normalize_solution(response_text),
        expected_sha256=stable_sha256(example.answer) if example.answer is not None else None,
    )


def _extract_balanced(text: Any, open_char: str, close_char: str, *, use_last: bool) -> str | None:
    raw = response_to_text(text)
    found: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(raw):
        if char == open_char:
            if depth == 0:
                start = index
            depth += 1
        elif char == close_char:
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                found.append(raw[start : index + 1])
                start = None
    if not found:
        return None
    return found[-1] if use_last else found[0]


def _normalize_str_list(values: str | Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = re.split(r"[,;]", values)
    else:
        raw_values = list(values)
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _normalize_str_set(
    values: str | Sequence[str] | None,
    *,
    lower: bool = False,
) -> set[str] | None:
    normalized = _normalize_str_list(values)
    if not normalized:
        return None
    if lower:
        return {value.lower() for value in normalized}
    return set(normalized)


def _first_present(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return None
