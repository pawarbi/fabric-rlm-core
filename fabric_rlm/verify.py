"""Verified execution: solve independently twice, reconcile disagreement.

The single technique shared by every top DataAgentBench submission is an
independent second derivation of the answer. This module is that pattern as a
library feature, measured before it was written: on a 54-query benchmark A/B at
three runs per query it moved stratified Pass@1 by +0.076 over single solves,
at about 2.9x tokens. The mechanism targets run-to-run flakiness, which in the
same measurements was the dominant error source (17 of 54 queries flipped
between identical runs).

How it works:

1. The task is solved twice in fresh, blind contexts. Nothing is shared; the
   second solve does not know the first exists. Independence is what makes
   agreement evidence: two derivations that could not copy each other landing
   on the same answer usually means the answer is right (agreement precision
   was 79-85% across the measured runs).
2. The two answers are compared structurally in code, not by a model: numbers
   must match exactly, sign included, semicolon-separated lists must contain
   the same item set (a missing member is the classic enumeration failure and
   must count as disagreement), and prose falls back to whole-word containment
   and token overlap. A blank answer never agrees with anything, another blank
   included. The comparison deliberately errs toward "disagree", which costs
   one reconciliation run, never correctness.
3. On disagreement a reconciler runs in a third fresh context. It receives the
   task, the data, and both candidate answers, and is instructed to find the
   exact point of divergence and re-derive rather than pick or average. On
   decisive pairs (exactly one candidate correct) the reconciler chose the
   right one 68-77% of the time depending on model, and re-derivation rescued
   20-39% of pairs where both candidates were wrong.

Where NOT to use this, stated plainly:

* Side-effect tasks. A verified run executes the task two or three times, so a
  task that writes files (excel_modify-style work) duplicates its side effects.
  Use it only for read-only analytical tasks whose product is the answer.
* Generative output (summaries, reports, prose). Two good drafts never agree
  structurally, so every run reconciles: triple cost, no signal. The
  comparison is designed for determinate answers: numbers, entities, lists.
* Consistent errors. When both blind solves make the same mistake they agree
  on it, and no self-consistency scheme can catch that. That failure class
  needs domain knowledge (see skills / authored data context), not more
  ensemble.
* Questions with several valid answers ("name a product that..."): the solves
  may pick different valid answers and reconcile needlessly. Harmless, but you
  pay triple for nothing.

Usage:

    from fabric_rlm import verified_task

    vr = verified_task(
        task="Which product had the highest net revenue?",
        inputs={"orders": File("orders.csv")},
        outputs=["answer"],
        lm={"model": "...", "api_key": "..."},
    )
    vr.result.payload["answer"]   # the winning answer
    vr.verdict                    # "agree" or "reconciled"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .runtime import RLM, RLMResult

_RECONCILE_GUIDANCE = """

Two analysts answered this question independently from the same data and
disagreed. Re-derive the answer yourself from the data, then decide. Their
answers are evidence about where to look, not authorities to average: check the
exact point where they diverge (a filter, a join, a definition, a period).
You MUST run at least one check against the data and print its result before
submitting; a verdict with no executed check in your output is invalid.
State the final answer first, then one short sentence on why the losing answer
was wrong.
"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# Signed numbers. A hyphen glued to a preceding word character is a range or
# date separator ("2024-2025", "Q1-2024"), not a minus. A currency symbol may
# sit between the minus and the digits ("-$5"). A number wrapped in
# parentheses with nothing else inside is the accounting form of a negative.
_NUMBER = re.compile(
    r"\(\s*[$€£¥]?(?P<paren>\d+(?:\.\d+)?)\s*%?\s*\)"
    r"|(?P<sign>(?<!\w)-)?[$€£¥]?(?P<num>\d+(?:\.\d+)?)"
)


def _numbers(text: str) -> list[str]:
    """Every number in ``text`` with its sign, as canonical strings, sorted."""
    found = []
    for match in _NUMBER.finditer(text):
        digits = match.group("paren") or match.group("num")
        negative = match.group("paren") is not None or match.group("sign") is not None
        found.append(("-" if negative else "") + digits)
    return sorted(found)


def answers_agree(a: str, b: str) -> bool:
    """Structural agreement between two answers.

    Errs toward disagreement: a false "disagree" costs one reconciliation run,
    a false "agree" costs correctness. Checks run in this order:

    1. A blank answer never agrees, not even with another blank: two solves
       that produced nothing are not evidence for anything.
    2. The signed numbers on both sides must be the same multiset. This runs
       before any text normalization because normalization strips the minus
       sign along with the rest of the punctuation, so "10" and "-10" would
       otherwise look identical.
    3. Normalized text equality.
    4. Semicolon lists agree only on the same item set.
    5. Whole-word containment for short-vs-verbose phrasings, then token
       overlap.
    """
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if _numbers(a) != _numbers(b):
        return False
    if na == nb:
        return True
    # Numbers identical from here. Semicolon lists agree only on the same item
    # set: a missing member is THE enumeration failure mode, and a substring or
    # overlap test would wave it through.
    if ";" in a or ";" in b:
        items_a = {_norm(x) for x in a.split(";") if _norm(x)}
        items_b = {_norm(x) for x in b.split(";") if _norm(x)}
        return items_a == items_b
    # Short-vs-verbose phrasings of the same single answer. Whole words only:
    # "Mark" is not "Denmark".
    if f" {na} " in f" {nb} " or f" {nb} " in f" {na} ":
        return True
    toks_a = {w for w in na.split() if len(w) > 3}
    toks_b = {w for w in nb.split() if len(w) > 3}
    return len(toks_a & toks_b) >= 0.6 * max(min(len(toks_a), len(toks_b)), 1)


@dataclass
class VerifiedResult:
    """Outcome of a verified run.

    ``result`` is the winning :class:`RLMResult`; ``attempts`` holds every
    solve that ran (two on agreement, three on reconciliation) so token
    accounting and trajectories stay auditable.
    """

    result: RLMResult
    verdict: str                      # "agree" | "reconciled"
    answer_a: str
    answer_b: str
    attempts: list[RLMResult] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.total_prompt_tokens or 0 for r in self.attempts)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.total_completion_tokens or 0 for r in self.attempts)


def verified_task(
    task: str,
    *,
    outputs: list[str],
    inputs: dict[str, Any] | None = None,
    field_name: str | None = None,
    reconcile_guidance: str = _RECONCILE_GUIDANCE,
    agree: Any = None,
    **rlm_kwargs: Any,
) -> VerifiedResult:
    """Solve ``task`` twice blind; reconcile in a third fresh context on disagreement.

    Accepts the same keyword arguments as :meth:`RLM.from_task` (``lm``,
    ``skills``, ``max_turns``, ``timeout``, ...). ``field_name`` selects which
    output field is compared; it defaults to the first entry of ``outputs``.
    Pass ``agree`` to override the comparison with your own
    ``(a: str, b: str) -> bool``.
    """
    fname = field_name or outputs[0]
    check = agree or answers_agree

    def solve(text: str) -> tuple[RLMResult, str]:
        res = RLM.from_task(text, outputs=outputs, inputs=inputs, **rlm_kwargs).run()
        val = (res.payload or {}).get(fname, "")
        return res, ("" if val is None else str(val))

    res_a, ans_a = solve(task)
    res_b, ans_b = solve(task)
    attempts = [res_a, res_b]

    if check(ans_a, ans_b):
        winner = res_a if ans_a.strip() or not ans_b.strip() else res_b
        return VerifiedResult(result=winner, verdict="agree",
                              answer_a=ans_a, answer_b=ans_b, attempts=attempts)

    reconcile_task = (
        f"{task}{reconcile_guidance}\n"
        f"Analyst 1 answered: {ans_a.strip() or '(no answer)'}\n\n"
        f"Analyst 2 answered: {ans_b.strip() or '(no answer)'}\n"
    )
    res_c, ans_c = solve(reconcile_task)
    attempts.append(res_c)
    if ans_c.strip():
        winner = res_c
    elif ans_a.strip():
        winner = res_a
    else:
        winner = res_b
    return VerifiedResult(result=winner, verdict="reconciled",
                          answer_a=ans_a, answer_b=ans_b, attempts=attempts)
