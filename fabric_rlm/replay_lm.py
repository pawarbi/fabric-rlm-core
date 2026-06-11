"""ReplayLM: replay a recorded trajectory through the *real* RLM loop.

A recorded :class:`~fabric_rlm.trajectory.Trajectory` already stores both sides
of every turn: the raw LM response (:attr:`TurnRecord.response_text`) and the
worker outcome (``stdout``/``stderr``/``error``/``submitted``/``state``/
``submit_payload``). That is everything needed to drive the orchestration loop
again with **zero API calls and zero subprocesses**.

This module provides two cooperating fakes:

* :class:`ReplayLM` -- a callable LM that returns the recorded ``response_text``
  for each turn, in order. Drop it in as ``lm=`` on an :class:`RLM`.
* :class:`ReplayInterpreter` -- a stand-in for the subprocess interpreter that
  returns the recorded :class:`ExecResult` for each turn, in order.

and a one-call helper :func:`replay_trajectory` that wires both into an existing
``RLM`` and runs it.

Why this matters
----------------
The value is hermetic, deterministic regression coverage of the parts of the
loop that change most often -- feedback formatting, validation, repair routing,
stuck-loop / max-turns stop conditions -- without paying for (or waiting on) a
model. Record a handful of representative trajectories once, freeze them as
``.jsonl`` golden files, and every future refactor of truncation / compaction /
repair logic runs against a green wall instead of hope.

Divergence is the signal
-------------------------
If a code change makes the loop ask for *more* LM turns than the recording has,
:class:`ReplayLM` raises :class:`DivergenceError` -- the LM call sits outside the
loop's per-turn ``try/except`` so the error propagates cleanly. If the loop
stops *earlier* than the recording (e.g. it now submits sooner), unused recorded
responses remain and :func:`replay_trajectory` raises :class:`DivergenceError`
after the run. Either way, "the loop behaves differently now" becomes a red test
rather than a silent production surprise.

Scope
-----
Targets the default (``engine="v6-custom"``) loop. Record golden trajectories
with skills/verifiers that do not issue *extra* interpreter executions (e.g.
``enable_verifier=False`` or verifier-free skills) so the LM-call count, the
interpreter-execute count, and the recorded turn count stay in lockstep. Extra
verifier/skill executions are *detectable* under ``strict=True`` (they desync
the recorded queues and raise :class:`DivergenceError`), but ``strict=False``
disables those checks and can mask them.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .interpreter import ExecResult
from .trajectory import Trajectory, TurnRecord

__all__ = [
    "DivergenceError",
    "ReplayLM",
    "ReplayInterpreter",
    "replay_trajectory",
]


class DivergenceError(RuntimeError):
    """Raised when the replayed loop diverges from the recorded trajectory.

    The two divergence modes are over-consumption (the loop requests more turns
    than were recorded) and under-consumption (the loop stops before consuming
    every recorded turn). Both indicate that orchestration behavior changed
    relative to the recording.
    """


# ---------------------------------------------------------------------------
# ReplayLM
# ---------------------------------------------------------------------------


class ReplayLM:
    """A callable LM that returns recorded ``response_text`` values in order.

    Usage::

        lm = ReplayLM.from_trajectory(traj)
        rlm = RLM("question -> answer", lm=lm)
        result = rlm.run(inputs)

    Because :func:`fabric_rlm.resolve_lm` passes plain callables through
    untouched, a ``ReplayLM`` can be used anywhere an ``lm=`` is accepted.
    """

    def __init__(self, responses: Sequence[str], *, strict: bool = True) -> None:
        self._responses: list[str] = list(responses)
        self._index = 0
        self.strict = strict
        # Every ``messages`` list the loop sent us, captured for assertions.
        self.calls: list[list[dict[str, Any]]] = []

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, *, strict: bool = True) -> "ReplayLM":
        return cls([t.response_text for t in trajectory.turns], strict=strict)

    def __call__(self, prompt: Any = None, *, messages: Any = None, **_: Any) -> str:
        # Mirror the two call conventions the loop supports (``messages=`` kwarg
        # for chat LMs; positional prompt for legacy single-string LMs).
        if messages is not None:
            try:
                self.calls.append([dict(m) for m in messages])
            except TypeError:
                self.calls.append(list(messages))
        if self._index >= len(self._responses):
            raise DivergenceError(
                "ReplayLM exhausted: the loop requested LM response "
                f"#{self._index + 1} but the recording has only "
                f"{len(self._responses)} turn(s). The loop is asking for more "
                "turns than were recorded -- behavior diverged from the golden "
                "trajectory."
            )
        response = self._responses[self._index]
        self._index += 1
        return response

    @property
    def consumed(self) -> int:
        return self._index

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._responses)


# ---------------------------------------------------------------------------
# ReplayInterpreter
# ---------------------------------------------------------------------------


def _exec_result_from_turn(turn: TurnRecord) -> ExecResult:
    """Reconstruct the :class:`ExecResult` the worker returned for ``turn``.

    ``ok`` is derived as "no error was recorded" -- the worker reports ``ok`` iff
    ``error is None``, and that is exactly the inverse relationship recorded on
    the turn.
    """
    return ExecResult(
        ok=turn.error is None,
        submitted=bool(turn.submitted),
        stdout=turn.stdout or "",
        stderr=turn.stderr or "",
        state=dict(turn.state or {}),
        error=turn.error,
        submit_payload=turn.submit_payload,
    )


class ReplayInterpreter:
    """A stand-in interpreter that returns recorded :class:`ExecResult`s in order.

    Implements the duck-typed surface the loop relies on: the context-manager
    protocol, ``configure_lm`` / ``set_inputs`` (no-ops), and ``execute``.

    In ``strict`` mode (the default) every executed code block is compared to
    the recorded ``TurnRecord.code`` for that turn. A mismatch means the loop
    extracted *different* code than it did when the trajectory was recorded
    (e.g. a regression in code-block extraction or repair routing) -- the
    divergence is recorded on :attr:`divergence_error` so :func:`replay_trajectory`
    can surface it even though ``RLM.run`` swallows exceptions raised from
    ``execute`` into a ``worker_error`` result.
    """

    def __init__(
        self,
        results: Sequence[ExecResult],
        *,
        expected_codes: Sequence[str] | None = None,
        strict: bool = True,
    ) -> None:
        self._results: list[ExecResult] = list(results)
        self._expected_codes: list[str | None] = (
            list(expected_codes)
            if expected_codes is not None
            else [None] * len(self._results)
        )
        self._index = 0
        self.strict = strict
        # Code strings the loop asked us to execute, captured for assertions.
        self.executed: list[str] = []
        # First detected divergence (code mismatch or over-consumption), if any.
        self.divergence_error: DivergenceError | None = None

    @classmethod
    def from_trajectory(
        cls, trajectory: Trajectory, *, strict: bool = True
    ) -> "ReplayInterpreter":
        return cls(
            [_exec_result_from_turn(t) for t in trajectory.turns],
            expected_codes=[t.code for t in trajectory.turns],
            strict=strict,
        )

    def __enter__(self) -> "ReplayInterpreter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def configure_lm(self, spec: Any) -> dict:
        return {}

    def set_inputs(self, inputs: Any) -> None:
        return None

    def execute(self, code: str) -> ExecResult:
        self.executed.append(code)
        if self._index >= len(self._results):
            err = DivergenceError(
                "ReplayInterpreter exhausted: the loop executed code block "
                f"#{self._index + 1} but the recording has only "
                f"{len(self._results)} turn(s). The loop executed more code than "
                "was recorded (e.g. a verifier/skill ran an extra block, or the "
                "turn count changed) -- behavior diverged from the recording."
            )
            if self.divergence_error is None:
                self.divergence_error = err
            raise err

        expected_code = self._expected_codes[self._index]
        result = self._results[self._index]
        self._index += 1

        if self.strict and expected_code is not None and code != expected_code:
            err = DivergenceError(
                f"Code mismatch on recorded turn {self._index}: the loop executed "
                "different code than it did when the trajectory was recorded. "
                "This usually means code-block extraction, truncation handling, "
                "or repair routing changed.\n"
                f"--- expected ---\n{expected_code}\n"
                f"--- got ---\n{code}"
            )
            if self.divergence_error is None:
                self.divergence_error = err
            # Return the recorded result anyway so the run completes; the
            # recorded divergence is surfaced by replay_trajectory afterward.
        return result

    @property
    def consumed(self) -> int:
        return self._index

    @property
    def remaining(self) -> int:
        return len(self._results) - self._index


# ---------------------------------------------------------------------------
# Loop wiring
# ---------------------------------------------------------------------------


@contextmanager
def _patched_interpreter(factory: Any) -> Iterator[None]:
    """Temporarily replace ``runtime.Interpreter`` with ``factory``.

    The loop constructs its interpreter via ``with Interpreter(...)``; swapping
    the module attribute is the least invasive way to inject a fake without
    changing the public ``RLM`` constructor signature. Always restored.
    """
    import fabric_rlm.runtime as runtime_mod

    saved = runtime_mod.Interpreter
    runtime_mod.Interpreter = factory
    try:
        yield
    finally:
        runtime_mod.Interpreter = saved


def replay_trajectory(
    rlm: Any,
    trajectory: Trajectory,
    *,
    inputs: dict[str, Any] | None = None,
    strict: bool = True,
) -> Any:
    """Re-run ``rlm`` against a recorded ``trajectory`` with no API calls.

    Feeds the recorded LM responses and worker results through the real loop so
    feedback formatting, validation, repair routing, and stop conditions are
    exercised exactly as in production. Returns the :class:`RLMResult`.

    The ``rlm`` instance is used as-is (its signature, validators, skills, and
    settings all apply); only its LM and interpreter are swapped for the
    duration of the call and restored afterward.

    With ``strict=True`` (default) a :class:`DivergenceError` is raised if the
    loop:

    * requests more LM turns than were recorded (over-consumption),
    * executes a code block that differs from the recording (extraction/repair
      regression), or
    * finishes earlier than the recording, leaving recorded turns unused
      (under-consumption).

    ``strict=False`` disables code-matching and the post-run consumption checks;
    over-consumption still raises because there is no recorded response to
    return.

    Only the default ``engine="v6-custom"`` loop is supported: it is the path
    that constructs the swappable ``runtime.Interpreter``. Other engines use a
    different interpreter and would not be intercepted, so they are rejected
    rather than silently hitting the real subprocess.

    Not thread-safe: it mutates the module-global ``runtime.Interpreter`` for the
    duration of the call. Intended for serial test / diagnostic use.
    """
    engine = getattr(rlm, "engine", "v6-custom")
    if engine != "v6-custom":
        raise ValueError(
            "replay_trajectory only supports the default engine='v6-custom' "
            f"loop (got engine={engine!r}). Other engines use a different "
            "interpreter that replay cannot intercept."
        )

    lm = ReplayLM.from_trajectory(trajectory, strict=strict)
    interp = ReplayInterpreter.from_trajectory(trajectory, strict=strict)

    saved_lm = getattr(rlm, "outer_lm", None)
    rlm.outer_lm = lm
    try:
        with _patched_interpreter(lambda *a, **k: interp):
            result = rlm.run(inputs)
    finally:
        rlm.outer_lm = saved_lm

    if not strict:
        return result

    # Surface code-mismatch / interpreter over-consumption first: RLM.run wraps
    # interpreter.execute in a broad ``except Exception`` that would otherwise
    # convert a raised DivergenceError into a benign worker_error result.
    if interp.divergence_error is not None:
        raise interp.divergence_error
    if interp.remaining > 0:
        raise DivergenceError(
            f"Replay consumed {interp.consumed} of "
            f"{interp.consumed + interp.remaining} recorded interpreter turn(s): "
            "the loop executed fewer code blocks than recorded."
        )
    if lm.remaining > 0:
        raise DivergenceError(
            f"Replay stopped after consuming {lm.consumed} of "
            f"{lm.consumed + lm.remaining} recorded turn(s): the loop finished "
            "earlier than the recording (e.g. it submitted or aborted sooner). "
            "Behavior diverged from the golden trajectory."
        )
    return result
