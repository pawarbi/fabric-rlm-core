"""Pure-Python combinator primitives for the fabric_rlm REPL skill.

Borrowed from λ-RLM (`nktkt/lambda-rlm`, arXiv:2603.20105) §3.1. Designed
to be **task-agnostic**: every primitive operates on generic Python values
(sequences, predicates, callables) and imports nothing from any benchmark
adapter. See `bench/adaptive/SPEC-combinators-skill.md` and the playbook
at `fabric_rlm/skills/combinators.md`.

All functions are pure; the only state is a process-local counter that
records `peek` usage for cost-tracking purposes (see `get_peek_counter`).
"""
from __future__ import annotations

from functools import reduce as _functools_reduce
from itertools import product as _itertools_product
from typing import Any, Callable, Iterable, Sequence, TypeVar

A = TypeVar("A")
B = TypeVar("B")

_PEEK_COUNTER: dict[str, int] = {"calls": 0, "chars_read": 0}
# Hard cap on the result size of `cross` to protect the subprocess from
# accidental cartesian-product blowup. 10**6 tuples is still ~tens of MB
# in memory; anything bigger must be expressed differently.
_CROSS_MAX_SIZE = 1_000_000


# ---- split ---------------------------------------------------------------

def split(text: str, k: int) -> list[str]:
    """Split ``text`` into ``k`` chunks at word boundaries.

    Chunk count is exactly ``k`` (padded with empty strings if the input
    has fewer than ``k`` words). Each chunk holds whole words; words are
    not broken across chunks.

    >>> split("a b c d", 2)
    ['a b', 'c d']
    >>> split("hello world", 1)
    ['hello world']
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    words = text.split()
    if not words:
        return [""] * k
    # Contiguous split: split into k consecutive groups of (almost) equal size.
    n = len(words)
    base, extra = divmod(n, k)
    chunks: list[str] = []
    start = 0
    for i in range(k):
        # First `extra` chunks get one extra word so all words are consumed.
        size = base + (1 if i < extra else 0)
        chunks.append(" ".join(words[start:start + size]))
        start += size
    return chunks


# ---- peek ----------------------------------------------------------------

def peek(text: str, offset: int, n: int) -> str:
    """Return ``n`` chars of ``text`` starting at ``offset``, cost-tracked.

    Out-of-bounds offsets and over-long windows are silently clamped (no
    error) — peeking past the end returns the empty string. Negative
    offsets or sizes raise ValueError because they almost always indicate
    a bug in the caller.

    >>> peek("abcdefghij", 2, 4)
    'cdef'
    >>> peek("abc", 10, 5)
    ''
    """
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(f"offset must be non-negative int, got {offset!r}")
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be non-negative int, got {n!r}")
    if offset >= len(text):
        return ""
    end = min(len(text), offset + n)
    window = text[offset:end]
    _PEEK_COUNTER["calls"] += 1
    _PEEK_COUNTER["chars_read"] += len(window)
    return window


def get_peek_counter() -> dict[str, int]:
    """Return a copy of the peek counter ({calls, chars_read})."""
    return dict(_PEEK_COUNTER)


def reset_peek_counter() -> None:
    """Zero the peek counter (test helper / between-question hook)."""
    _PEEK_COUNTER["calls"] = 0
    _PEEK_COUNTER["chars_read"] = 0


# ---- map_ / filter_ / reduce_ -------------------------------------------
# Suffixed to avoid shadowing the builtins.

def map_(seq: Iterable[A], fn: Callable[[A], B]) -> list[B]:
    """Apply ``fn`` to every element of ``seq``, returning a list.

    >>> map_([1, 2, 3], lambda x: x * 2)
    [2, 4, 6]
    """
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    return [fn(x) for x in seq]


def filter_(seq: Iterable[A], pred: Callable[[A], bool]) -> list[A]:
    """Return the elements of ``seq`` for which ``pred(x)`` is truthy.

    >>> filter_([1, 2, 3, 4], lambda x: x % 2 == 0)
    [2, 4]
    """
    if not callable(pred):
        raise TypeError(f"pred must be callable, got {type(pred).__name__}")
    return [x for x in seq if pred(x)]


_SENTINEL = object()


def reduce_(seq: Iterable[A], fn: Callable[[Any, A], Any], initial: Any = _SENTINEL) -> Any:
    """Left-fold ``seq`` with ``fn``, optionally seeded by ``initial``.

    >>> reduce_([1, 2, 3, 4], lambda a, b: a + b, 0)
    10
    """
    if initial is _SENTINEL:
        return _functools_reduce(fn, seq)
    return _functools_reduce(fn, seq, initial)


# ---- concat --------------------------------------------------------------

def concat(parts: Sequence[Any], *, sep: str | None = None) -> Any:
    """Concatenate a sequence of strings or a sequence of lists.

    Strings → joined string (with optional ``sep``). Lists → flattened list
    (``sep`` not allowed). Mixed types raise TypeError.

    >>> concat(["ab", "cd"])
    'abcd'
    >>> concat([[1], [2, 3]])
    [1, 2, 3]
    """
    if not parts:
        return "" if sep is None or all(False for _ in parts) else ""
    first = parts[0]
    if isinstance(first, str):
        if not all(isinstance(p, str) for p in parts):
            raise TypeError("concat: all parts must be strings when first is str")
        return (sep if sep is not None else "").join(parts)
    if isinstance(first, list):
        if sep is not None:
            raise TypeError("concat: sep is only valid for strings, not lists")
        if not all(isinstance(p, list) for p in parts):
            raise TypeError("concat: all parts must be lists when first is list")
        out: list[Any] = []
        for p in parts:
            out.extend(p)
        return out
    raise TypeError(f"concat: unsupported element type {type(first).__name__}")


# ---- cross ---------------------------------------------------------------

def cross(*factors: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Cartesian product of ``len(factors)`` sequences, eager.

    Result size is capped at 10**6 tuples — beyond that, callers should
    re-express with :func:`map_` and a generator-aware approach.

    >>> cross([1, 2], ['a', 'b'])
    [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    """
    if not factors:
        raise ValueError("cross requires at least one factor")
    size = 1
    for f in factors:
        size *= len(f)
        if size > _CROSS_MAX_SIZE:
            raise ValueError(
                f"cross product too large ({size} > {_CROSS_MAX_SIZE}); "
                "re-express as a streaming map_ or filter_"
            )
    return list(_itertools_product(*factors))


__all__ = [
    "split",
    "peek",
    "get_peek_counter",
    "reset_peek_counter",
    "map_",
    "filter_",
    "reduce_",
    "concat",
    "cross",
]
