# SPEC — `verify_shape_tolerant` (generalize the harness grader)

Branch: `feat/shape-tolerant-grader`
Phase: 1 (Specify)
Owner: agent-skill workflow

## Problem

The 5-way comparison harness (`scripts/build_comparison_5way_notebook.py`,
lines 168-190) hard-codes a per-template `grade()` function. The audit
of the bandit run (25 questions) showed **8 / 19 failures** classified
`wrong_shape_or_format` — cases where the model produced the right
content in a wrapper the per-template parser didn't recognize.

Examples that the strict matcher rejects but a human grader would accept:

| Gold (expected) | Model produced | Strict result |
|---|---|---|
| `[2, 1, 0]` | `solution = [2, 1, 0]` | hits |
| `[2, 1, 0]` | ` ```json\n[2,1,0]\n``` ` | misses (fenced block) |
| `[2, 1, 0]` | `{"answer": [2, 1, 0]}` | misses (key-wrapped) |
| `[2, 1, 0]` | `{"solution": [2,1,0]}` | misses (key-wrapped) |
| `[2, 1, 0]` | `Q1: 2\nQ2: 1\nQ3: 0` | misses (Q-numbered) |
| `{"a": 1, "b": 2}` | `{"a":1,"b":2,"_reasoning":"..."}` | misses (extra key) — *do NOT accept; this is genuinely different* |
| `5` | `**Answer:** 5` | hits |
| `5` | `\\boxed{5}` | hits |

A **template-agnostic** matcher belongs in
`bench/adaptive/longcot_adapter.py` so any task family inherits it (not
just LongCoT-CS).

## Solution

Add `verify_shape_tolerant(expected, response_text) -> bool` to
`longcot_adapter.py`. It:

1. **Normalizes `expected`**: if `str`, attempt `json.loads`, fall back
   to literal.
2. **Extracts candidate values** from `response_text` via several
   strategies, in order:
   - Strip everything inside `<think>...</think>`.
   - Strip TeX `\boxed{...}` wrapper.
   - Strip Markdown bold/italics (`**...**`, `*...*`).
   - Find fenced JSON: ` ```json ... ``` ` or ` ``` ... ``` `.
   - Find last balanced JSON object `{...}` and last balanced array
     `[...]`.
   - Find `solution = X`, `answer = X`, `result = X`, `Answer: X`.
   - Find Q-numbered lines `Q1: a\nQ2: b\nQ3: c` → `[a, b, c]`.
   - Find CSV of ints / floats.
3. **Unwraps** key-bearing dicts: if expected is non-dict (list / int /
   str) and candidate is a single-key dict whose key is in
   `{"answer", "solution", "result", "output", "value", "final"}`,
   substitute the value.
4. **Compares** with type-normalization:
   - `int` ↔ stringified int (`"5" == 5`).
   - `list` element-wise with type-normalization.
   - `dict` key-by-key requiring **identical key sets** (no extra keys).
     Reject if model added stray keys.
   - `str` after stripping whitespace, quotes, trailing punctuation.

Returns `True` if any extraction strategy yields a candidate equal to
expected after normalization, else `False`.

## Generalization (mandatory section per repo rule)

This function does **not** know LongCoT, MFMC, Backprop, VLIW, or any
specific template. It works on any `expected` value (dict, list, int,
str) and any free-form `response_text`. Task-family contracts:

| Task family | Expected shape | Will work? |
|---|---|---|
| LongCoT-CS-JSON (MFMC/MCM/...) | `dict` | yes |
| LongCoT-CS-INT (VLIW/CodeTrace) | `int` | yes |
| LongCoT-CS-LIST (Backprop/DistMem) | `list[int]` | yes |
| Spark-RCA | `str` (RCA category) | yes |
| Code-gen | `str` (function source) | partial — exact-match only |
| Multi-doc QA | `str` (passage span) | yes (whitespace-tolerant) |
| Math word problems | `int` / `float` | yes (boxed handled) |

No template names, no dataset literals, no magic numbers. Only structural
heuristics that apply to any `(expected, response)` pair.

## Test plan (TDD, in `tests/test_shape_tolerant_grader.py`)

For each gold below, assert that **all** of the listed model outputs
score True, and **all** of the negative cases score False.

```
GOLD = [2, 1, 0]
POSITIVE:
  "[2, 1, 0]"
  "solution = [2, 1, 0]"
  "Answer: [2,1,0]"
  "```json\n[2, 1, 0]\n```"
  '{"answer": [2, 1, 0]}'
  '{"solution":[2,1,0]}'
  "Q1: 2\nQ2: 1\nQ3: 0"
  "**Answer:** [2, 1, 0]"
  "<think>let me work it out...</think>\n[2, 1, 0]"
  "2, 1, 0"
NEGATIVE:
  "[2, 1, 1]"        # wrong content
  "[2, 1]"           # wrong length
  "[0, 1, 2]"        # wrong order
  "the answer is unknown"
```

```
GOLD = 5
POSITIVE:
  "5"
  "Answer: 5"
  "**Answer:** 5"
  "\\boxed{5}"
  '{"answer": 5}'
  "After analysis, the answer is 5."
NEGATIVE:
  "6"
  "answer: about 5 or 6"   # ambiguous → reject
```

```
GOLD = {"a": 1, "b": 2}
POSITIVE:
  '{"a": 1, "b": 2}'
  '{"b": 2, "a": 1}'                     # key-order independent
  '{"a":"1","b":"2"}'                     # int↔str normalization
  '```json\n{"a":1,"b":2}\n```'
NEGATIVE:
  '{"a": 1, "b": 2, "c": 3}'              # extra key
  '{"a": 1}'                              # missing key
  '{"answer":{"a":1,"b":2}}'              # double-wrapped — reject (ambiguous)
```

## Wiring

After the function lands, update
`scripts/build_comparison_5way_notebook.py:168-190` to call
`verify_shape_tolerant(gold_answer, response_text)` as a **fallback**
after the existing template-specific path returns False. This way the
strict per-template matcher remains the primary signal (no risk of
false positives on existing passes), and the new tolerant matcher only
recovers genuine shape mismatches.

## Re-grade existing traces

After wiring, re-grade the existing 25-question bandit run
(`comparison_5way_bandit-full-20260502-161343/`) with the new harness
and report the delta. Target: 0–8 additional passes (we don't expect
all 8 to flip, since some have wrong content alongside wrong shape).

## Out of scope

- LLM-judge fallback (separate todo).
- Per-template special cases (e.g. set-vs-list semantics for some
  templates) — `verify_cs_response` keeps them; this function is the
  generic fallback.
- Changes to `verify_cs_response` itself (no risk to existing tests).

## Open questions

- Should we add `verify_shape_tolerant` as the primary matcher in
  `verify_cs_response` itself, or only in the harness? **Decision:**
  only in the harness for now. If post-merge re-grade shows it's safe
  on all 22+ adapter tests, we promote in a follow-up.
