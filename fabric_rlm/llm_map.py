"""Host-side ``llm_map``: apply one instruction to every item with an LM, in parallel.

The model writes one call instead of a batching loop. Iteration, concurrency,
per-item validation and retries run here on the host, where the LM object
lives, rather than in code the model has to get right every turn. A run that
labelled 5,860 complaints with a model-written loop spent all of its turns
debugging that loop; the same work as a scripted map finished in minutes.

Running on the host has three consequences worth stating:

* the worker's per-exec timeout does not bound the map, because the parent
  re-arms its read deadline after every tool call;
* ``block_network=True`` still works, because no call leaves the worker;
* the map LM can be a live object (``FabricLM(...)``), which ``sub_lm`` cannot.

Items and results cross the worker boundary as JSON, so an item is text, a
number, a bool, or a dict/list of those. Outputs are validated against a small
declarative schema (``str``, ``int``, ``float``, ``bool``, or a list of allowed
strings); a failed item is retried with the validation error in the prompt,
and one that never validates comes back as ``None`` with its error, never as a
guess.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

LLM_MAP_TOOL = "__fabric_rlm_llm_map__"

DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 64
DEFAULT_RETRIES = 2
MAX_RETRIES = 5
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 100
# A batch is also cut when its items add up to this many characters, so a few
# long items never make one oversized prompt.
MAX_BATCH_CHARS = 60_000
MAX_ITEMS = 100_000
MAX_ITEM_CHARS = 50_000
_SCALAR_TYPES = ("str", "int", "float", "bool")
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


class LLMMapError(ValueError):
    """An ``llm_map`` request the host refuses to run, with a usable message."""


# ----------------------------------------------------------------- request validation
def normalize_request(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the JSON request sent by the worker and return a clean copy."""

    items = kwargs.get("items")
    if not isinstance(items, list):
        raise LLMMapError("llm_map items must be a list (or a DataFrame / Series in the worker).")
    if not items:
        raise LLMMapError("llm_map received no items.")
    if len(items) > MAX_ITEMS:
        raise LLMMapError(
            f"llm_map received {len(items):,} items; the limit is {MAX_ITEMS:,} per call. "
            "Filter or split the items first."
        )
    instructions = kwargs.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise LLMMapError("llm_map needs instructions: what to do with each item.")
    output = kwargs.get("output")
    if not isinstance(output, dict) or not output:
        raise LLMMapError(
            "llm_map needs output: a dict of field name -> type (str, int, float, bool) "
            "or a list of allowed strings."
        )
    fields: dict[str, dict[str, Any]] = {}
    for name, spec in output.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise LLMMapError(f"llm_map output field names must be plain identifiers, got {name!r}.")
        if isinstance(spec, str) and spec in _SCALAR_TYPES:
            fields[name] = {"type": spec}
        elif isinstance(spec, dict) and isinstance(spec.get("choices"), list) and spec["choices"]:
            choices = [str(c) for c in spec["choices"]]
            if len(set(c.strip().lower() for c in choices)) != len(choices):
                raise LLMMapError(f"llm_map output {name!r} has duplicate choices.")
            fields[name] = {"type": "choice", "choices": choices}
        else:
            raise LLMMapError(
                f"llm_map output {name!r} must be str, int, float, bool, or a list of allowed strings; got {spec!r}."
            )
    input_name = kwargs.get("input_name") or "item"
    if not isinstance(input_name, str) or not input_name.isidentifier() or input_name in fields:
        raise LLMMapError("llm_map input_name must be an identifier that is not also an output field.")
    concurrency = _bounded_int(kwargs.get("concurrency", DEFAULT_CONCURRENCY), "concurrency", 1, MAX_CONCURRENCY)
    retries = _bounded_int(kwargs.get("retries", DEFAULT_RETRIES), "retries", 0, MAX_RETRIES)
    batch_size = _bounded_int(kwargs.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size", 1, MAX_BATCH_SIZE)
    texts: list[str] = []
    for index, item in enumerate(items):
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
        if len(text) > MAX_ITEM_CHARS:
            raise LLMMapError(
                f"llm_map item {index} is {len(text):,} characters; the limit is {MAX_ITEM_CHARS:,}. "
                "Trim or chunk long items first."
            )
        texts.append(text)
    return {"items": texts, "instructions": instructions.strip(), "fields": fields,
            "input_name": input_name, "concurrency": concurrency, "retries": retries, "batch_size": batch_size}


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LLMMapError(f"llm_map {name} must be an integer from {low} to {high}, got {value!r}.")
    return value


# ----------------------------------------------------------------- output validation
def _coerce(field: str, spec: Mapping[str, Any], value: Any) -> Any:
    """Return the value in its declared type, or raise ValueError saying why not."""

    kind = spec["type"]
    if value is None:
        raise ValueError(f"`{field}` is missing")
    if kind == "str":
        # Empty text is a legitimate answer ("no quote on this page"); only an absent field is an error.
        return str(value).strip()
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"`{field}` is missing")
    if kind == "choice":
        text = str(value).strip().strip("\"'").strip()
        for choice in spec["choices"]:
            if text.lower() == choice.lower():
                return choice
        raise ValueError(f"`{field}` must be exactly one of {spec['choices']}, got {text!r}")
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ValueError(f"`{field}` must be true or false, got {value!r}")
    if kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"`{field}` must be an integer, got {value!r}")
        number = float(str(value).replace(",", "").strip()) if not isinstance(value, (int, float)) else value
        if not float(number).is_integer():
            raise ValueError(f"`{field}` must be an integer, got {value!r}")
        return int(number)
    if kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"`{field}` must be a number, got {value!r}")
        number = float(str(value).replace(",", "").strip()) if not isinstance(value, (int, float)) else float(value)
        if not math.isfinite(number):
            raise ValueError(f"`{field}` must be a finite number, got {value!r}")
        return number
    raise ValueError(f"unknown type {kind!r} for `{field}`")


def validate_output(fields: Mapping[str, Mapping[str, Any]], raw: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    errors: list[str] = []
    for name, spec in fields.items():
        try:
            out[name] = _coerce(name, spec, raw.get(name))
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    return out


# ----------------------------------------------------------------- execution
def _signature(dspy: Any, request: Mapping[str, Any], feedback: str | None = None) -> Any:
    fields: dict[str, Any] = {request["input_name"]: (str, dspy.InputField())}
    # Every output is declared as str so the adapter never rejects a value
    # before our own coercion runs; our coercion produces the retry feedback.
    hints = {"str": "free text", "int": "an integer", "float": "a number", "bool": "true or false"}
    for name, spec in request["fields"].items():
        if spec["type"] == "choice":
            desc = "exactly one of: " + " | ".join(spec["choices"])
        else:
            desc = hints[spec["type"]]
        fields[name] = (str, dspy.OutputField(desc=desc))
    instructions = request["instructions"]
    if feedback:
        instructions += (
            "\n\nYour previous answer for this item was rejected: " + feedback
            + ". Answer again, following the output rules exactly."
        )
    return dspy.make_signature(fields, instructions)


def _field_rules(request: Mapping[str, Any]) -> str:
    hints = {"str": "text", "int": "an integer", "float": "a number", "bool": "true or false"}
    rules = []
    for name, spec in request["fields"].items():
        if spec["type"] == "choice":
            rules.append(f'"{name}": exactly one of {json.dumps(spec["choices"])}')
        else:
            rules.append(f'"{name}": {hints[spec["type"]]}')
    return ", ".join(rules)


def _batch_signature(dspy: Any, request: Mapping[str, Any]) -> Any:
    fields = {
        "items": (str, dspy.InputField(desc="numbered items; each starts with [n] on its own line")),
        "answers": (str, dspy.OutputField(desc=(
            "a JSON array with exactly one object per item, in item order. Each object has "
            '"n": the item number, and ' + _field_rules(request)
        ))),
    }
    instructions = (
        request["instructions"]
        + "\n\nApply this to EACH numbered item independently; one item must not influence another's answer."
    )
    return dspy.make_signature(fields, instructions)


def _batches(items: list[str], size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    chars = 0
    for index, text in enumerate(items):
        if current and (len(current) >= size or chars + len(text) > MAX_BATCH_CHARS):
            batches.append(current)
            current, chars = [], 0
        current.append(index)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


def _parse_batch(text: Any, indexes: list[int], fields: Mapping[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Map a batch answer back to items. Anything missing or invalid is an error for that item only."""

    good: dict[int, dict[str, Any]] = {}
    bad: dict[int, str] = {}
    raw = str(text or "").strip()
    start, end = raw.find("["), raw.rfind("]")
    try:
        rows = json.loads(raw[start:end + 1]) if start != -1 and end > start else None
    except json.JSONDecodeError:
        rows = None
    if not isinstance(rows, list):
        return good, {i: "the batch answer was not a JSON array" for i in indexes}
    by_number: dict[int, Any] = {}
    for row in rows:
        if isinstance(row, dict):
            try:
                n = int(row.get("n"))
            except (TypeError, ValueError):
                continue
            # A number answered twice is ambiguous; neither answer is trusted.
            by_number[n] = None if n in by_number else row
    for position, index in enumerate(indexes):
        row = by_number.get(position)
        if row is None:
            bad[index] = "this item was missing or duplicated in the batch answer"
            continue
        try:
            good[index] = validate_output(fields, row)
        except ValueError as exc:
            bad[index] = str(exc)[:500]
    return good, bad


def run_llm_map(lm: Any, kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a validated map and return ``(result, telemetry_record)``.

    ``result`` is JSON-serializable: ``results`` aligned with the input (a dict
    per item, or None when the item never validated), ``errors`` (index, error)
    for those items, and ``stats``. The telemetry record carries counts and
    field names only, never item text or output values.
    """

    started = time.monotonic()
    request = normalize_request(kwargs)
    if is_decision_model(lm):
        return _run_decision_map(lm, request, started)
    import dspy

    items = request["items"]
    base_signature = _signature(dspy, request)
    lock = threading.Lock()
    counters = {"calls": 0, "retried": 0, "lm_errors": 0, "batches": 0, "batch_fallbacks": 0}

    def one(index: int, feedback: str | None = None, attempts: int | None = None) -> tuple[int, dict[str, Any] | None, str | None]:
        last_error = feedback or "no attempt"
        total = request["retries"] + 1 if attempts is None else attempts
        for attempt in range(total):
            signature = base_signature if feedback is None else _signature(dspy, request, feedback)
            with lock:
                counters["calls"] += 1
                if attempt or attempts is not None:
                    counters["retried"] += 1
            try:
                with dspy.context(lm=lm):
                    prediction = dspy.Predict(signature)(**{request["input_name"]: items[index]})
                raw = {name: getattr(prediction, name, None) for name in request["fields"]}
                return index, validate_output(request["fields"], raw), None
            except ValueError as exc:
                feedback = last_error = str(exc)[:500]
            except Exception as exc:  # noqa: BLE001 - an LM/adapter failure is this item's error, not the map's
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                feedback = "your answer could not be parsed; return every output field"
                with lock:
                    counters["lm_errors"] += 1
        return index, None, last_error

    results: list[dict[str, Any] | None] = [None] * len(items)
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=request["concurrency"]) as pool:
        if request["batch_size"] == 1:
            outcomes = list(pool.map(one, range(len(items))))
        else:
            batch_signature = _batch_signature(dspy, request)

            def batch(indexes: list[int]) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
                with lock:
                    counters["calls"] += 1
                    counters["batches"] += 1
                text = "\n\n".join(f"[{n}]\n{items[i]}" for n, i in enumerate(indexes))
                try:
                    with dspy.context(lm=lm):
                        prediction = dspy.Predict(batch_signature)(items=text)
                except Exception as exc:  # noqa: BLE001 - the batch falls back to per-item calls
                    with lock:
                        counters["lm_errors"] += 1
                    return {}, {i: f"batch call failed: {type(exc).__name__}" for i in indexes}
                return _parse_batch(getattr(prediction, "answers", None), indexes, request["fields"])

            outcomes = []
            fallback: list[tuple[int, str]] = []
            for good, bad in pool.map(batch, _batches(items, request["batch_size"])):
                outcomes.extend((i, v, None) for i, v in good.items())
                fallback.extend(bad.items())
            with lock:
                counters["batch_fallbacks"] = len(fallback)
            if request["retries"]:
                outcomes.extend(pool.map(lambda pair: one(pair[0], pair[1], request["retries"]), fallback))
            else:
                outcomes.extend((i, None, e) for i, e in fallback)
        for index, value, error in outcomes:
            results[index] = value
            if error is not None:
                errors.append({"index": index, "error": error})
    errors.sort(key=lambda e: e["index"])
    seconds = round(time.monotonic() - started, 3)
    stats = {"items": len(items), "ok": len(items) - len(errors), "failed": len(errors),
             "retried": counters["retried"], "calls": counters["calls"], "lm_errors": counters["lm_errors"],
             "batch_size": request["batch_size"], "batches": counters["batches"],
             "batch_fallbacks": counters["batch_fallbacks"], "seconds": seconds, "model": _model_name(lm)}
    record = {"query_type": "llm_map", "executed": True, "output_fields": list(request["fields"]),
              "concurrency": request["concurrency"], "retries_allowed": request["retries"],
              "execution_seconds": seconds, "total_seconds": seconds, "returned_rows": stats["ok"],
              **{k: v for k, v in stats.items() if k not in ("seconds",)}}
    return {"results": results, "errors": errors, "stats": stats}, record


def _model_name(lm: Any) -> str:
    return str(getattr(lm, "model", None) or getattr(lm, "model_name", None) or type(lm).__name__)


# ----------------------------------------------------------------- decision models
class DecisionLM:
    """A decision ("System One") model as the ``llm_map`` backend: Jev, Kev, Solar Decide, ...

    These models do not generate text. They take a state plus typed questions
    and return a choice with per-option probabilities and a confidence, or the
    probability that a statement is true, in a few hundred milliseconds. That
    is exactly an ``llm_map`` item, so a list-of-choices output becomes a
    Choice question and a ``bool`` output becomes a Noul question. They cannot
    answer ``str``, ``int`` or ``float`` fields; use a text LM for those.

    ``DecisionLM("typesafe/jev-1.13")`` calls OpenRouter's decisions endpoint
    with ``OPENROUTER_API_KEY``. For TypeSafe directly, pass
    ``api_base="https://api.typesafe.ai/v1/systemone"``, ``model="jev-latest"``
    and ``api_key=os.environ["TYPESAFE_API_KEY"]``.
    """

    is_decision_model = True
    OPENROUTER_DECISIONS = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model: str, *, api_key: str | None = None, api_base: str | None = None,
                 timeout: float = 60.0, num_retries: int = 3, bool_threshold: float = 0.5) -> None:
        import os

        self.model = model
        self.api_base = api_base or self.OPENROUTER_DECISIONS
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.num_retries = num_retries
        self.bool_threshold = bool_threshold
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"DecisionLM({self.model!r})"

    def decide(self, state: str, questions: Mapping[str, Any]) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.num_retries + 1):
            request = urllib.request.Request(self.api_base, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode())
                usage = data.get("usage") or {}
                with self._lock:
                    self.history.append({"model": data.get("model"), "cost": usage.get("cost"), "usage": usage})
                return data
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
        raise RuntimeError(f"decision model call failed: {last}")


def is_decision_model(lm: Any) -> bool:
    return bool(getattr(lm, "is_decision_model", False)) and callable(getattr(lm, "decide", None))


def _decision_questions(request: Mapping[str, Any]) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    unsupported = [n for n, s in request["fields"].items() if s["type"] not in ("choice", "bool")]
    if unsupported:
        raise LLMMapError(
            f"llm_map is using a decision model, which answers only a list of choices or bool; "
            f"{unsupported} are {[request['fields'][n]['type'] for n in unsupported]}. "
            "Ask those as choices (e.g. bucket a number into ranges) or as bool questions."
        )
    for name, spec in request["fields"].items():
        label = name.replace("_", " ")
        if spec["type"] == "choice":
            questions[name] = {"type": "choice", "instructions": f"{request['instructions']}\n\nDecide: {label}.",
                               "criteria": {c: c for c in spec["choices"]}}
        else:
            questions[name] = {"type": "noul", "instructions": f"{request['instructions']}\n\nTrue or false: {label}."}
    return questions


def _run_decision_map(lm: Any, request: Mapping[str, Any], started: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One decision call per item; the model's own calibration replaces text retries."""

    items = request["items"]
    questions = _decision_questions(request)
    threshold = float(getattr(lm, "bool_threshold", 0.5))
    lock = threading.Lock()
    counters = {"calls": 0, "lm_errors": 0}

    def one(index: int) -> tuple[int, dict[str, Any] | None, str | None]:
        with lock:
            counters["calls"] += 1
        try:
            answers = (lm.decide(items[index], questions) or {}).get("answers") or {}
            row: dict[str, Any] = {}
            for name, spec in request["fields"].items():
                answer = answers.get(name) or {}
                if spec["type"] == "choice":
                    row[name] = _coerce(name, spec, answer.get("choice"))
                    row[f"{name}_confidence"] = float(answer.get("confidence", 0.0))
                else:
                    p = float(answer["noul"])
                    row[name] = p >= threshold
                    row[f"{name}_p"] = p
            return index, row, None
        except LLMMapError:
            raise
        except Exception as exc:  # noqa: BLE001 - one item's failure is that item's error
            with lock:
                counters["lm_errors"] += 1
            return index, None, f"{type(exc).__name__}: {str(exc)[:300]}"

    results: list[dict[str, Any] | None] = [None] * len(items)
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=request["concurrency"]) as pool:
        for index, value, error in pool.map(one, range(len(items))):
            results[index] = value
            if error is not None:
                errors.append({"index": index, "error": error})
    seconds = round(time.monotonic() - started, 3)
    stats = {"items": len(items), "ok": len(items) - len(errors), "failed": len(errors), "retried": 0,
             "calls": counters["calls"], "lm_errors": counters["lm_errors"], "batch_size": 1, "batches": 0,
             "batch_fallbacks": 0, "seconds": seconds, "model": _model_name(lm), "decision_model": True}
    record = {"query_type": "llm_map", "executed": True, "output_fields": list(request["fields"]),
              "concurrency": request["concurrency"], "retries_allowed": 0,
              "execution_seconds": seconds, "total_seconds": seconds, "returned_rows": stats["ok"],
              **{k: v for k, v in stats.items() if k != "seconds"}}
    return {"results": results, "errors": errors, "stats": stats}, record
