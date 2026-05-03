---
applies_when:
  keywords: ["chunk", "split", "map", "filter", "reduce", "concat", "cross", "cartesian", "rollup", "aggregate", "fold", "scan large", "process file", "iterate"]
  output_fields: []
excludes: []
depends_on: ["core"]
specificity: utility
---

# combinators

Summary: Seven pure-Python primitives (split, peek, map_, filter_, reduce_, concat, cross) for chunk/map/reduce work in the REPL. Borrowed from λ-RLM. Use these instead of re-deriving boilerplate per task.
Dependencies: core

## Purpose

Give the model a deterministic, pre-tested combinator library so chunk/map/reduce pipelines compose without per-task boilerplate. Especially useful when you must process a long input in pieces, fan a transform over a list, or aggregate into a single value.

## Contract: output fields

This is a utility skill — it defines no output fields itself. Output shape is determined by whichever domain skill (or the question) drives the task. Combinator outputs feed back into your `solution` / `final_answer` payload via plain Python.

The primitives' I/O contracts:

- **name** `split(text: str, k: int) -> list[str]`
  **type** function
  **exact definition** Split a string into exactly `k` chunks at word boundaries. Chunk sizes differ by at most 1 word. Empty input yields `[""] * k`. `k <= 0` raises `ValueError`.
  **canonical formula** `chunks[i] = " ".join(words[start_i : start_i + size_i])` where `size_i = floor(n/k) + (1 if i < n % k else 0)`.

- **name** `peek(text: str, offset: int, n: int) -> str`
  **type** function
  **exact definition** Return up to `n` characters starting at byte/char `offset` of `text`. Out-of-bounds offset returns `""`; over-long window is clamped to end of text. Negative `offset` or `n` raises `ValueError`. Calls increment a process-local counter accessible via `get_peek_counter()` for cost tracking.
  **canonical formula** `text[offset : min(len(text), offset + n)]`.

- **name** `map_(seq, fn) -> list`
  **type** function
  **exact definition** Apply `fn` elementwise. `fn` must be callable; non-callable raises `TypeError`.
  **canonical formula** `[fn(x) for x in seq]`.

- **name** `filter_(seq, pred) -> list`
  **type** function
  **exact definition** Keep elements where `pred(x)` is truthy. Non-callable `pred` raises `TypeError`.
  **canonical formula** `[x for x in seq if pred(x)]`.

- **name** `reduce_(seq, fn, initial=...) -> Any`
  **type** function
  **exact definition** Left-fold; if `initial` omitted, uses `seq[0]` and folds the rest. Empty `seq` with no `initial` raises `TypeError`; with `initial` returns `initial` unchanged.
  **canonical formula** `functools.reduce(fn, seq[, initial])`.

- **name** `concat(parts, *, sep=None) -> str | list`
  **type** function
  **exact definition** All-strings → joined string (using `sep` if given). All-lists → flattened list (`sep` rejected). Mixed types raise `TypeError`.
  **canonical formula** `(sep or "").join(parts)` for strings; `sum(parts, [])` for lists.

- **name** `cross(*factors) -> list[tuple]`
  **type** function
  **exact definition** Eager Cartesian product of the input sequences. Empty factor → empty result. Result size cap of 10⁶ tuples (over-cap raises `ValueError`).
  **canonical formula** `list(itertools.product(*factors))`.

## Required verifier

```python
def verify(payload):
    """Sanity-check that a payload built using combinators is well-formed.

    Combinators are pure functions; the only payload-level invariant we can
    check generically is that the model didn't accidentally produce a
    generator/iterator (which would render in the prompt as a useless repr)
    or a recursive structure.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if hasattr(value, "__next__") and not hasattr(value, "__len__"):
                raise AssertionError(
                    f"payload[{key!r}] is an iterator; call list() on it before SUBMIT"
                )
    elif hasattr(payload, "__next__") and not hasattr(payload, "__len__"):
        raise AssertionError("payload is an iterator; materialize with list() first")
```

## Tripwires

- **Eager `cross` blowup** — `cross(*[range(100)] * 7)` is 10¹⁴ tuples; the cap raises `ValueError`. Re-express with `map_` / generator if you genuinely need that scale.
- **`reduce_` on empty without `initial`** — raises `TypeError`. Always pass `initial=` when the sequence may be empty (e.g. after `filter_`).
- **`concat` mixed types** — `concat(["abc", [1, 2]])` raises. Map values to a common type first (e.g. `concat(map_(parts, str))`).
- **`peek` past EOF returns `""`, not error** — checking `if peek(...)` handles this cleanly; comparing `peek(...) == expected` may silently match an empty expected.
- **Re-importing in subprocess** — every REPL turn is a fresh subprocess; re-import combinators at the top of every code block. The router preloads the skill text but not the runtime imports.

## Invariants

- `len(split(text, k)) == k` always.
- `peek` chars-read ≤ `n` argument (clamped if past EOF).
- `map_(seq, fn)` length equals `len(list(seq))`.
- `len(filter_(seq, pred)) <= len(list(seq))`.
- `concat([]) == ""` and `concat([], sep=",") == ""`.
- `len(cross(a, b, c)) == len(a) * len(b) * len(c)` (when ≤ cap).

## Procedure

1. Import: `from fabric_rlm.skills._combinators import split, peek, map_, filter_, reduce_, concat, cross`.
2. **Pattern A — sequence-reduce.** Aggregating over a list? `reduce_(seq, fn, initial=0)`. Always supply `initial` unless you've proven `seq` is non-empty.
3. **Pattern B — search by partial-read.** Long input? `peek(blob, offset, n)` to read a window without paying for the whole thing. Use `get_peek_counter()` after to inspect cost.
4. **Pattern C — batch-map then reduce.** `reduce_(map_(items, transform), combine, initial)`. Compose left-to-right; do not nest more than 3 deep.
5. **Pattern D — cross-product enumeration.** Small finite domains? `cross(*domains)`. If you hit the cap, the question almost certainly wanted a pruning approach, not enumeration.
6. Call `verify(payload)` before `SUBMIT(payload)` to catch unmaterialized iterators.

## Worked example: sequence-reduce

```python
from fabric_rlm.skills._combinators import map_, reduce_

xs = [3, 1, 4, 1, 5, 9, 2, 6]
total = reduce_(map_(xs, lambda x: x * x), lambda a, b: a + b, initial=0)
# total == 173
```

## Worked example: search by partial-read

```python
from fabric_rlm.skills._combinators import peek, get_peek_counter, reset_peek_counter

reset_peek_counter()
blob = open("/tmp/large_log.txt").read()
needle = "ERROR"
for offset in range(0, len(blob), 4000):
    window = peek(blob, offset, 4096)
    if needle in window:
        print(offset, window.index(needle))
        break
print(get_peek_counter())   # {'calls': N, 'chars_read': M}
```

## Worked example: batch-map

```python
from fabric_rlm.skills._combinators import map_, filter_, concat

words = "the quick brown fox jumps over".split()
short = filter_(words, lambda w: len(w) < 5)
upper = map_(short, str.upper)
joined = concat(upper, sep=" ")
# joined == "THE FOX OVER"
```
