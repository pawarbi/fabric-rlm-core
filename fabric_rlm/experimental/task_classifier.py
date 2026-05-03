"""One-shot task-type classifier — seeds bandit priors with informative Beta(α, β).

Borrowed from λ-RLM (`nktkt/lambda-rlm`). The bandit's cold-start ``Beta(1, 1)``
wastes ~3 attempts on rung 0 even when the question is obviously hard. A
single low-effort classifier call (~$0.001) lets the bandit start at the
right rung the first time it sees a task family.

The classifier is deliberately **task-family agnostic** — its 9 output
classes describe *reasoning shape*, not domain. They apply equally to a
Spark log ``GLOBAL_REDUCE``, a SQL ``FORMAT`` answer, a code-review
``MULTI_HOP``, and a CS puzzle. No template names appear here.

Failure mode: if the LM call fails, the prompt is malformed, or the model
returns an out-of-vocabulary class, :func:`classify` returns
:attr:`TaskClass.UNKNOWN` and the bandit falls back to its uniform prior
— same behaviour as today's bandit on a brand-new task_key.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class TaskClass(str, Enum):
    """Universal reasoning-shape taxonomy. Domain-free by design.

    Values map to per-class Beta priors in :data:`DEFAULT_CLASS_PRIORS`.
    """

    LOOKUP = "lookup"             # single-fact retrieval — rung 0 strongly
    SEARCH = "search"             # scan + filter — rung 0
    AGGREGATE = "aggregate"       # sum/count/group — rung 1
    PAIRWISE = "pairwise"         # pair-by-pair compare — rung 1
    MULTI_HOP = "multi_hop"       # chained reasoning — rung 2
    CS_PUZZLE = "cs_puzzle"       # algorithmic — rung 3
    CODE_GEN = "code_gen"         # write code — rung 2
    FORMAT = "format"             # transform-shape — rung 0
    REFUSAL = "refusal"           # ambiguous / unanswerable — rung 0, mark
    UNKNOWN = "unknown"           # fallback — no prior override


# Per-class Beta(α, β) priors, keyed by rung. Only the listed rungs get
# overridden; unlisted rungs keep the cold-start Beta(1, 1).
#
# Mass tuning: (α=4, β=1) is a confident "this rung tends to win" nudge that
# Thompson-sampling will overwrite within ~3 contradicting observations.
# (α=2, β=1) is a soft hint. (α=1, β=4) is a confident "skip this rung".
DEFAULT_CLASS_PRIORS: dict[TaskClass, dict[int, tuple[float, float]]] = {
    TaskClass.LOOKUP:    {0: (4.0, 1.0), 4: (1.0, 4.0)},
    TaskClass.SEARCH:    {0: (4.0, 1.0), 4: (1.0, 4.0)},
    TaskClass.FORMAT:    {0: (4.0, 1.0), 4: (1.0, 4.0)},
    TaskClass.AGGREGATE: {1: (3.0, 1.0)},
    TaskClass.PAIRWISE:  {1: (3.0, 1.0)},
    TaskClass.CODE_GEN:  {2: (3.0, 1.0)},
    TaskClass.MULTI_HOP: {2: (4.0, 1.0), 0: (1.0, 3.0)},
    TaskClass.CS_PUZZLE: {3: (4.0, 1.0), 0: (1.0, 3.0), 1: (1.0, 2.0)},
    TaskClass.REFUSAL:   {0: (2.0, 1.0)},
    TaskClass.UNKNOWN:   {},
}


_VALID_VALUES = {c.value for c in TaskClass}


def _normalize(raw: Any) -> TaskClass:
    """Map an LM output (string, possibly noisy) to a :class:`TaskClass`."""

    if raw is None:
        return TaskClass.UNKNOWN
    text = str(raw).strip().lower()
    if not text:
        return TaskClass.UNKNOWN
    text = text.split()[0].strip(".,;:!?\"'")
    if text in _VALID_VALUES:
        return TaskClass(text)
    return TaskClass.UNKNOWN


_PROMPT = (
    "Classify the question below into ONE of these reasoning categories:\n"
    "  lookup      — single fact retrieval\n"
    "  search      — scan a body of text/data for matches\n"
    "  aggregate   — sum, count, group, or roll up\n"
    "  pairwise    — compare items pair by pair\n"
    "  multi_hop   — chain several inferences together\n"
    "  cs_puzzle   — algorithmic / mathematical puzzle\n"
    "  code_gen    — write a code snippet\n"
    "  format      — reshape input into a new format\n"
    "  refusal     — ambiguous, unanswerable, or harmful\n"
    "Answer with the single category word and nothing else.\n\n"
    "Question:\n{question}\n\nCategory:"
)


def classify(question: str, lm: Any | None) -> TaskClass:
    """Return a :class:`TaskClass` for ``question``.

    Total: never raises. ``lm=None``, an empty question, an LM that throws,
    or a model that produces an out-of-vocabulary token all map to
    :attr:`TaskClass.UNKNOWN`. Caller can then either retry with a
    different LM or accept the uniform-prior fallback.

    ``lm`` is duck-typed: anything with a ``__call__`` that takes a
    string-keyword ``messages`` or a positional prompt and returns a string
    (or list-of-strings) works. This matches both ``dspy.LM`` and the
    project's own :class:`fabric_rlm.lm.FabricLM`.
    """

    if lm is None or not question or not str(question).strip():
        return TaskClass.UNKNOWN

    prompt = _PROMPT.format(question=str(question)[:4000])

    try:
        result = _invoke(lm, prompt)
    except Exception:
        return TaskClass.UNKNOWN

    if isinstance(result, list):
        result = result[0] if result else None

    return _normalize(result)


def _invoke(lm: Any, prompt: str) -> Any:
    """Call ``lm`` with several common signatures so we don't bind to one shape."""

    # dspy.LM signature: lm(prompt: str) -> list[str]
    try:
        return lm(prompt)
    except TypeError:
        pass

    # messages-style:  lm(messages=[{"role":"user","content":...}])
    try:
        return lm(messages=[{"role": "user", "content": prompt}])
    except TypeError:
        pass

    # Some inner LMs expose `.complete(prompt)` or `.generate(prompt)`
    for attr in ("complete", "generate", "predict"):
        fn = getattr(lm, attr, None)
        if callable(fn):
            return fn(prompt)

    raise TypeError(f"Unsupported LM type: {type(lm).__name__}")


def seed_priors(
    state: Any,
    task_key: str,
    task_class: TaskClass,
    *,
    prior_table: dict[TaskClass, dict[int, tuple[float, float]]] | None = None,
    overwrite_existing: bool = False,
) -> bool:
    """Seed ``state.priors[task_key]`` from ``task_class``'s per-rung Beta table.

    Returns ``True`` if priors were written, ``False`` otherwise. Default
    behaviour is **idempotent**: if ``state`` already holds any observations
    for ``task_key`` we leave it alone (the bandit is already learning;
    don't perturb live posteriors). Pass ``overwrite_existing=True`` to
    forcibly reset.

    ``state`` is duck-typed — anything with a ``priors: dict`` attribute
    works. We deliberately don't import :class:`BanditState` here to keep
    this module decoupled from the bandit (and importable in lightweight
    environments without the experimental package).
    """

    if task_class is TaskClass.UNKNOWN:
        return False

    table = prior_table if prior_table is not None else DEFAULT_CLASS_PRIORS
    overrides = table.get(task_class)
    if not overrides:
        return False

    priors = getattr(state, "priors", None)
    if priors is None or not isinstance(priors, dict):
        return False

    existing = priors.get(task_key)
    if existing and not overwrite_existing:
        return False

    priors[task_key] = {int(rung): (float(a), float(b)) for rung, (a, b) in overrides.items()}
    return True


__all__ = [
    "TaskClass",
    "DEFAULT_CLASS_PRIORS",
    "classify",
    "seed_priors",
]
