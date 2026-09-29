"""Host-side ``ask_each``: ask one question about every item with an LM, in parallel.

The model writes one call instead of a batching loop. Iteration, concurrency,
per-item validation, retries and throttling run here on the host, where the LM
object lives, rather than in code the model has to get right every turn. A run
that labelled 5,860 complaints with a model-written loop spent all of its turns
debugging that loop; the same work as one call finished in minutes.

``ask_each`` is off unless the caller turns it on with ``RLM(ask_each=...)``.
When it is off the worker has no ``ask_each`` and the prompt does not mention
it. Turning it on is also the caller's permission for item text to leave the
host for the chosen model: the worker itself makes no network call, so
``block_network=True`` still holds for the worker, but the items it passes are
sent to that model.

The LM is called through the same adapter as the run's main ``lm``, so anything
that works as ``lm=`` works here: ``FabricLM(...)``, an OpenRouter or Azure AI
Foundry model string, ``OpenAILM(...)``, a ``dspy.LM`` or a plain callable.
A ``DecisionLM`` (a decision model such as Jev) answers choice and ``bool``
fields with probabilities.

Items and results cross the worker boundary as JSON, so an item is text, a
number, a bool, or a dict/list of those. Each answer is validated against a
small declarative schema (``str``, ``int``, ``float``, ``bool``, or a list of
allowed strings); a failed item is retried with the validation error in the
prompt, and one that never validates comes back as ``None`` with its error,
never as a guess.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

ASK_EACH_TOOL = "__fabric_rlm_ask_each__"
# Set inside the worker when the host turns ask_each on, so helpers that run in
# the worker (SemanticModel.find_measures) can use it. None everywhere else.
WORKER_ASK_EACH: Any = None

DEFAULT_CONCURRENCY = 8
DEFAULT_RETRIES = 2
MAX_RETRIES = 5
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 100
# A batch is also cut when its items add up to this many characters, so a few
# long items never make one oversized prompt.
MAX_BATCH_CHARS = 60_000
MAX_ITEM_CHARS = 50_000
_SCALAR_TYPES = ("str", "int", "float", "bool")
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}
# Throttling: a rate-limited call waits and is tried again without using one of
# the item's retries, and the map runs fewer calls at once until calls succeed.
_THROTTLE_MARKERS = ("429", "rate limit", "ratelimit", "too many requests", "capacitylimitexceeded",
                     "throttl", "overloaded", "quota exceeded")
_BACKOFF_FIRST = 1.0
_BACKOFF_MAX = 30.0
_NOT_ANSWERED = "not answered"


@dataclass
class AskEach:
    """Turns on ``ask_each`` in a run and sets its limits.

    ``RLM(ask_each=True)`` uses the run's main LM; ``RLM(ask_each=lm)`` uses
    ``lm``; ``RLM(ask_each=AskEach(lm=..., max_seconds=...))`` sets the limits
    too. ``lm`` is anything that works as ``RLM(lm=...)``, or a ``DecisionLM``.

    ``max_seconds`` caps one ``ask_each`` call: items not answered in time come
    back as ``None`` with a time-limit error, and ``stats["unfinished"]`` counts
    them. ``max_concurrency`` caps the calls in flight (the model may ask for
    fewer). ``max_items`` caps the items in one call.

    ``text_lm`` (optional) is a second model the run can pick per call with
    ``ask_each(..., model="text")``, e.g. a cheap text model to pull quotes from
    pages after a decision model has screened them.
    """

    lm: Any = None
    max_seconds: float = 900.0
    max_concurrency: int = 16
    max_items: int = 100_000
    text_lm: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float)) or self.max_seconds <= 0:
            raise ValueError(f"AskEach max_seconds must be a positive number, got {self.max_seconds!r}.")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int) or not 1 <= self.max_concurrency <= 256:
            raise ValueError(f"AskEach max_concurrency must be an integer from 1 to 256, got {self.max_concurrency!r}.")
        if isinstance(self.max_items, bool) or not isinstance(self.max_items, int) or self.max_items < 1:
            raise ValueError(f"AskEach max_items must be a positive integer, got {self.max_items!r}.")


def normalize_ask_each(value: Any) -> AskEach | None:
    """``None``/``False`` -> off; ``True`` -> main LM; an ``AskEach`` as is; anything else is the LM."""

    if value is None or value is False:
        return None
    if value is True:
        return AskEach()
    if isinstance(value, AskEach):
        return value
    return AskEach(lm=value)


class AskEachError(ValueError):
    """An ``ask_each`` request the host refuses to run, with a usable message."""


# ----------------------------------------------------------------- request validation
def normalize_request(kwargs: Mapping[str, Any], config: AskEach | None = None) -> dict[str, Any]:
    """Validate the JSON request sent by the worker and return a clean copy."""

    config = config or AskEach()
    items = kwargs.get("items")
    if not isinstance(items, list):
        raise AskEachError("ask_each items must be a list (or a DataFrame / Series in the worker).")
    if not items:
        raise AskEachError("ask_each received no items.")
    if len(items) > config.max_items:
        raise AskEachError(
            f"ask_each received {len(items):,} items; the limit is {config.max_items:,} per call. "
            "Filter or split the items first."
        )
    question = kwargs.get("question")
    if not isinstance(question, str) or not question.strip():
        raise AskEachError("ask_each needs a question: what to ask or do for each item.")
    output = kwargs.get("output")
    if not isinstance(output, dict) or not output:
        raise AskEachError(
            "ask_each needs output: a dict of field name -> type (str, int, float, bool) "
            "or a list of allowed strings."
        )
    fields: dict[str, dict[str, Any]] = {}
    for name, spec in output.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise AskEachError(f"ask_each output field names must be plain identifiers, got {name!r}.")
        if isinstance(spec, str) and spec in _SCALAR_TYPES:
            fields[name] = {"type": spec}
        elif isinstance(spec, dict) and isinstance(spec.get("choices"), list) and spec["choices"]:
            choices = [str(c) for c in spec["choices"]]
            if len(set(c.strip().lower() for c in choices)) != len(choices):
                raise AskEachError(f"ask_each output {name!r} has duplicate choices.")
            fields[name] = {"type": "choice", "choices": choices}
        else:
            raise AskEachError(
                f"ask_each output {name!r} must be str, int, float, bool, or a list of allowed strings; got {spec!r}."
            )
    concurrency = _bounded_int(kwargs.get("concurrency", DEFAULT_CONCURRENCY), "concurrency", 1, 256)
    concurrency = min(concurrency, config.max_concurrency)
    retries = _bounded_int(kwargs.get("retries", DEFAULT_RETRIES), "retries", 0, MAX_RETRIES)
    batch_size = _bounded_int(kwargs.get("batch_size", DEFAULT_BATCH_SIZE), "batch_size", 1, MAX_BATCH_SIZE)
    max_seconds = kwargs.get("max_seconds")
    if max_seconds is None:
        max_seconds = config.max_seconds
    elif isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or max_seconds <= 0:
        raise AskEachError(f"ask_each max_seconds must be a positive number, got {max_seconds!r}.")
    max_seconds = min(float(max_seconds), float(config.max_seconds))
    texts: list[str] = []
    for index, item in enumerate(items):
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
        if len(text) > MAX_ITEM_CHARS:
            raise AskEachError(
                f"ask_each item {index} is {len(text):,} characters; the limit is {MAX_ITEM_CHARS:,}. "
                "Trim or chunk long items first."
            )
        texts.append(text)
    return {"items": texts, "question": question.strip(), "fields": fields, "concurrency": concurrency,
            "retries": retries, "batch_size": batch_size, "max_seconds": max_seconds}


def _bounded_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AskEachError(f"ask_each {name} must be an integer from {low} to {high}, got {value!r}.")
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
    if isinstance(value, bool):
        raise ValueError(f"`{field}` must be a number, got {value!r}")
    try:
        number = value if isinstance(value, (int, float)) else float(str(value).replace(",", "").strip())
    except ValueError:
        raise ValueError(f"`{field}` must be a number, got {value!r}") from None
    if not math.isfinite(float(number)):
        raise ValueError(f"`{field}` must be a finite number, got {value!r}")
    if kind == "int":
        if not float(number).is_integer():
            raise ValueError(f"`{field}` must be an integer, got {value!r}")
        return int(number)
    if kind == "float":
        return float(number)
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


# ----------------------------------------------------------------- prompts and parsing
def _field_rules(fields: Mapping[str, Mapping[str, Any]]) -> str:
    hints = {"str": "text", "int": "an integer", "float": "a number", "bool": "true or false"}
    rules = []
    for name, spec in fields.items():
        if spec["type"] == "choice":
            rules.append(f'"{name}": exactly one of {json.dumps(spec["choices"])}')
        else:
            rules.append(f'"{name}": {hints[spec["type"]]}')
    return ", ".join(rules)


def _single_messages(request: Mapping[str, Any], text: str, feedback: str | None = None) -> list[dict[str, str]]:
    system = (
        "You answer one question about one item. Reply with only a JSON object, no other text, with these keys: "
        + _field_rules(request["fields"]) + "."
    )
    user = f"{request['question']}\n\nItem:\n{text}"
    if feedback:
        user += f"\n\nYour previous answer was rejected: {feedback}. Answer again, following the output rules exactly."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _batch_messages(request: Mapping[str, Any], texts: list[str]) -> list[dict[str, str]]:
    system = (
        "You answer one question about each of several numbered items, each item independently: one item must "
        "not influence another's answer. Reply with only a JSON array, no other text, with exactly one object "
        'per item in item order. Each object has "n": the item number, and ' + _field_rules(request["fields"]) + "."
    )
    body = "\n\n".join(f"[{n}]\n{text}" for n, text in enumerate(texts))
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{request['question']}\n\nItems:\n{body}"}]


def _json_part(text: str, opener: str, closer: str) -> Any:
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _parse_single(text: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    raw = _json_part(str(text or ""), "{", "}")
    if not isinstance(raw, dict):
        raise ValueError("the answer was not a JSON object")
    return validate_output(fields, raw)


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
    rows = _json_part(str(text or ""), "[", "]")
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


# ----------------------------------------------------------------- throttling
def is_throttle_error(exc: BaseException) -> bool:
    """A rate limit or capacity refusal: wait and try again, it is not the item's fault."""

    for attr in ("status_code", "status", "code"):
        if getattr(exc, attr, None) == 429:
            return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _THROTTLE_MARKERS)


class _Limiter:
    """Caps calls in flight; halves the cap and pauses everyone on a throttle, recovers one step at a time."""

    def __init__(self, limit: int, deadline: float) -> None:
        self.ceiling = limit
        self.limit = limit
        self.active = 0
        self.deadline = deadline
        self.pause_until = 0.0
        self.throttles = 0
        self.lowest = limit
        self._streak = 0
        self._cond = threading.Condition()

    def acquire(self) -> bool:
        with self._cond:
            while True:
                now = time.monotonic()
                if now >= self.deadline:
                    return False
                if self.active < self.limit and now >= self.pause_until:
                    self.active += 1
                    return True
                wait = max(self.pause_until - now, 0.05) if now < self.pause_until else 0.5
                self._cond.wait(min(wait, self.deadline - now))

    def release(self, *, success: bool) -> None:
        with self._cond:
            self.active -= 1
            if success and self.limit < self.ceiling:
                self._streak += 1
                if self._streak >= self.limit:
                    self.limit += 1
                    self._streak = 0
            self._cond.notify_all()

    def throttled(self, attempt: int) -> None:
        with self._cond:
            self.throttles += 1
            self.limit = max(1, self.limit // 2)
            self.lowest = min(self.lowest, self.limit)
            self._streak = 0
            delay = min(_BACKOFF_MAX, _BACKOFF_FIRST * 2 ** attempt) * (0.75 + random.random() / 2)
            self.pause_until = max(self.pause_until, time.monotonic() + delay)
            self._cond.notify_all()


class _TimeUp(Exception):
    pass


# ----------------------------------------------------------------- usage
def _usage_from(entry: Any) -> tuple[int | None, int | None, float | None]:
    usage = entry.get("usage") if isinstance(entry, dict) else getattr(entry, "usage", None)
    if usage is not None and not isinstance(usage, dict):
        usage = {k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")}
    usage = usage or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    cost = entry.get("cost") if isinstance(entry, dict) else getattr(entry, "cost", None)
    if cost is None:
        cost = usage.get("cost")
    return (int(prompt) if isinstance(prompt, (int, float)) and not isinstance(prompt, bool) else None,
            int(completion) if isinstance(completion, (int, float)) and not isinstance(completion, bool) else None,
            float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None)


def _sum_usage(entries: list[Any]) -> dict[str, Any]:
    prompts, completions, costs = [], [], []
    for entry in entries:
        p, c, cost = _usage_from(entry)
        prompts.append(p)
        completions.append(c)
        costs.append(cost)

    def total(values: list[Any]) -> Any:
        known = [v for v in values if v is not None]
        return sum(known) if known else None

    cost = total(costs)
    return {"prompt_tokens": total(prompts), "completion_tokens": total(completions),
            "cost": round(cost, 6) if cost is not None else None}


def _history_len(lm: Any) -> int | None:
    history = getattr(lm, "history", None)
    return len(history) if isinstance(history, list) else None


def _model_name(lm: Any) -> str:
    return str(getattr(lm, "model", None) or getattr(lm, "model_name", None) or type(lm).__name__)


# ----------------------------------------------------------------- execution
def run_ask_each(lm: Any, kwargs: Mapping[str, Any], config: AskEach | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a validated map and return ``(result, telemetry_record)``.

    ``result`` is JSON-serializable: ``results`` aligned with the input (a dict
    per item, or None when the item never validated), ``errors`` (index, error)
    for those items, and ``stats``. The telemetry record carries counts, field
    names and usage only, never item text or output values.
    """

    config = config or AskEach()
    started = time.monotonic()
    request = normalize_request(kwargs, config)
    deadline = started + request["max_seconds"]
    if is_decision_model(lm):
        return _run_decision_map(lm, request, started, deadline)
    from .runtime import _call_lm, _response_to_text

    items = request["items"]
    limiter = _Limiter(request["concurrency"], deadline)
    lock = threading.Lock()
    counters = {"calls": 0, "retried": 0, "lm_errors": 0, "batches": 0, "batch_fallbacks": 0}
    responses: list[Any] = []
    history_start = _history_len(lm)

    def call(messages: list[dict[str, str]]) -> str:
        """One LM call, waiting out throttling. Raises _TimeUp when the time limit ends the wait."""
        attempt = 0
        while True:
            if not limiter.acquire():
                raise _TimeUp()
            try:
                with lock:
                    counters["calls"] += 1
                response = _call_lm(lm, messages)
            except Exception as exc:  # noqa: BLE001 - classified below
                limiter.release(success=False)
                if is_throttle_error(exc):
                    limiter.throttled(attempt)
                    attempt += 1
                    continue
                raise
            limiter.release(success=True)
            if history_start is None:
                with lock:
                    responses.append(response)
            return _response_to_text(response)

    def one(index: int, feedback: str | None = None, attempts: int | None = None) -> tuple[int, dict[str, Any] | None, str | None]:
        last_error = feedback or "no attempt"
        total = request["retries"] + 1 if attempts is None else attempts
        for attempt in range(total):
            if attempt or attempts is not None:
                with lock:
                    counters["retried"] += 1
            try:
                text = call(_single_messages(request, items[index], feedback))
                return index, _parse_single(text, request["fields"]), None
            except _TimeUp:
                return index, None, _time_up(request)
            except ValueError as exc:
                feedback = last_error = str(exc)[:500]
            except Exception as exc:  # noqa: BLE001 - an LM failure is this item's error, not the map's
                last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                feedback = "your answer could not be read; reply with only the JSON object"
                with lock:
                    counters["lm_errors"] += 1
        return index, None, last_error

    def batch(indexes: list[int]) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
        with lock:
            counters["batches"] += 1
        try:
            text = call(_batch_messages(request, [items[i] for i in indexes]))
        except _TimeUp:
            return {}, {i: _time_up(request) for i in indexes}
        except Exception as exc:  # noqa: BLE001 - the batch falls back to per-item calls
            with lock:
                counters["lm_errors"] += 1
            return {}, {i: f"batch call failed: {type(exc).__name__}" for i in indexes}
        return _parse_batch(text, indexes, request["fields"])

    results: list[dict[str, Any] | None] = [None] * len(items)
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=request["concurrency"]) as pool:
        if request["batch_size"] == 1:
            outcomes = list(pool.map(one, range(len(items))))
        else:
            outcomes = []
            fallback: list[tuple[int, str]] = []
            for good, bad in pool.map(batch, _batches(items, request["batch_size"])):
                outcomes.extend((i, v, None) for i, v in good.items())
                fallback.extend(bad.items())
            retry = [(i, e) for i, e in fallback if not e.startswith(_NOT_ANSWERED)]
            outcomes.extend((i, None, e) for i, e in fallback if e.startswith(_NOT_ANSWERED))
            counters["batch_fallbacks"] = len(retry)
            if request["retries"]:
                outcomes.extend(pool.map(lambda pair: one(pair[0], pair[1], request["retries"]), retry))
            else:
                outcomes.extend((i, None, e) for i, e in retry)
    for index, value, error in outcomes:
        results[index] = value
        if error is not None:
            errors.append({"index": index, "error": error})
    errors.sort(key=lambda e: e["index"])
    if history_start is not None:
        usage = _sum_usage(list(getattr(lm, "history", [])[history_start:]))
    else:
        usage = _sum_usage(responses)
    extra = {"retried": counters["retried"], "lm_errors": counters["lm_errors"], "batches": counters["batches"],
             "batch_fallbacks": counters["batch_fallbacks"], "throttled": limiter.throttles,
             "lowest_concurrency": limiter.lowest}
    return _finish(lm, request, started, results, errors, counters["calls"], usage, extra)


def _time_up(request: Mapping[str, Any]) -> str:
    return (f"{_NOT_ANSWERED}: the ask_each time limit ({request['max_seconds']:g} s) ran out. "
            "Ask about fewer items, raise concurrency, or pass a larger max_seconds.")


def _finish(lm: Any, request: Mapping[str, Any], started: float, results: list[Any], errors: list[dict[str, Any]],
            calls: int, usage: Mapping[str, Any], extra: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seconds = round(time.monotonic() - started, 3)
    unfinished = sum(1 for e in errors if str(e.get("error", "")).startswith(_NOT_ANSWERED))
    stats = {"items": len(results), "ok": len(results) - len(errors), "failed": len(errors) - unfinished,
             "unfinished": unfinished, "calls": calls, "batch_size": request["batch_size"],
             "concurrency": request["concurrency"], "seconds": seconds, "model": _model_name(lm),
             **extra, **usage}
    record = {"query_type": "ask_each", "executed": True, "output_fields": list(request["fields"]),
              "retries_allowed": request["retries"], "max_seconds": request["max_seconds"],
              "execution_seconds": seconds, "total_seconds": seconds, "returned_rows": stats["ok"],
              **{k: v for k, v in stats.items() if k != "seconds"}}
    return {"results": results, "errors": errors, "stats": stats}, record


# ----------------------------------------------------------------- decision models
_OPENROUTER_DECISIONS = "https://openrouter.ai/api/alpha/decisions"
_KEY_FOR_HOST = {"openrouter.ai": "OPENROUTER_API_KEY", "api.typesafe.ai": "TYPESAFE_API_KEY"}


class DecisionLM:
    """A decision ("System One") model as the ``ask_each`` backend: Jev, Kev, Solar Decide, ...

    These models do not generate text. They take a state plus typed questions
    and return a choice with per-option probabilities and a confidence, or the
    probability that a statement is true, in a few hundred milliseconds. That
    is exactly an ``ask_each`` item, so a list-of-choices output becomes a
    Choice question and a ``bool`` output becomes a Noul question. They cannot
    answer ``str``, ``int`` or ``float`` fields; use a text LM for those.

    ``DecisionLM("typesafe/jev-1.13")`` calls OpenRouter's decisions endpoint
    with ``OPENROUTER_API_KEY``. For TypeSafe directly, pass
    ``api_base="https://api.typesafe.ai/v1/systemone"`` and ``model="jev-latest"``
    (it reads ``TYPESAFE_API_KEY``). Any other endpoint needs ``api_key=``. A key
    read from the environment is only ever sent to the host it belongs to.
    Item text is sent to that endpoint.
    """

    is_decision_model = True
    OPENROUTER_DECISIONS = _OPENROUTER_DECISIONS

    def __init__(self, model: str, *, api_key: str | None = None, api_base: str | None = None,
                 timeout: float = 60.0, num_retries: int = 3, bool_threshold: float = 0.5) -> None:
        from urllib.parse import urlparse

        self.model = model
        self.api_base = api_base or _OPENROUTER_DECISIONS
        host = (urlparse(self.api_base).hostname or "").lower()
        env_name = next((name for suffix, name in _KEY_FOR_HOST.items()
                         if host == suffix or host.endswith("." + suffix)), None)
        self.api_key = api_key or (os.environ.get(env_name) if env_name else None)
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
        raise AskEachError(
            f"ask_each is using a decision model, which answers only a list of choices or bool; "
            f"{unsupported} are {[request['fields'][n]['type'] for n in unsupported]}. "
            "Ask those as choices (e.g. bucket a number into ranges) or as bool questions."
        )
    for name, spec in request["fields"].items():
        label = name.replace("_", " ")
        if spec["type"] == "choice":
            questions[name] = {"type": "choice", "instructions": f"{request['question']}\n\nDecide: {label}.",
                               "criteria": {c: c for c in spec["choices"]}}
        else:
            questions[name] = {"type": "noul", "instructions": f"{request['question']}\n\nTrue or false: {label}."}
    return questions


def _run_decision_map(lm: Any, request: Mapping[str, Any], started: float, deadline: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One decision call per item; the model's own probabilities replace text retries."""

    items = request["items"]
    questions = _decision_questions(request)
    threshold = float(getattr(lm, "bool_threshold", 0.5))
    lock = threading.Lock()
    counters = {"calls": 0, "lm_errors": 0}
    history_start = _history_len(lm)

    def one(index: int) -> tuple[int, dict[str, Any] | None, str | None]:
        if time.monotonic() >= deadline:
            return index, None, _time_up(request)
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
    history = getattr(lm, "history", [])
    usage = _sum_usage(list(history[history_start:])) if history_start is not None else _sum_usage([])
    extra = {"retried": 0, "lm_errors": counters["lm_errors"], "batches": 0, "batch_fallbacks": 0,
             "throttled": 0, "lowest_concurrency": request["concurrency"], "decision_model": True}
    return _finish(lm, request, started, results, errors, counters["calls"], usage, extra)


def summarize_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals across a run's ``ask_each`` calls, for ``trajectory.metadata["ask_each"]``."""

    def total(key: str) -> Any:
        known = [r.get(key) for r in records if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        return sum(known) if known else None

    cost = total("cost")
    return {"calls": len(records), "items": total("items") or 0, "ok": total("ok") or 0,
            "failed": total("failed") or 0, "unfinished": total("unfinished") or 0,
            "lm_calls": total("calls") or 0, "throttled": total("throttled") or 0,
            "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
            "cost": round(cost, 6) if cost is not None else None,
            "seconds": round(total("execution_seconds") or 0.0, 3),
            "models": sorted({str(r.get("model")) for r in records if r.get("model")})}
