"""Does timeout recovery generalize past the one shape that motivated it?

The motivating case was a slow full-file scan. These cover other ways a worker
stops responding, plus the states the runtime can be in when it happens, so the
feature is not silently specific to "pandas read a big CSV".
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM, SkillLoader


class ScriptedLM:
    def __init__(self, turns):
        self.turns = list(turns)
        self.i = 0
        self.seen: list[str] = []

    def __call__(self, messages=None, prompt=None, **kwargs):
        if messages:
            self.seen.append(messages[-1].get("content", ""))
        if self.i >= len(self.turns):
            return ["SUBMIT(answer='exhausted')"]
        turn = self.turns[self.i]
        self.i += 1
        return [turn]


def code(body: str) -> str:
    return f"```python\n{body}\n```"


DONE = code("SUBMIT(answer='ok')")

# Distinct ways a worker stops answering in time.
#
# Unbounded recursion used to be in here and was removed: it is not reliably a
# hang. `sys.setrecursionlimit(10**7)` plus infinite recursion races the timeout
# against a C-stack overflow, and which one wins depends on the platform, the
# Python version and how fast the machine is. On 3.10 it segfaulted at ~1.2s and
# lost to a 2s timeout; on 3.11 frames moved to a heap-allocated data stack so it
# merely ran slowly and timed out normally; on windows/3.13 it went back to
# failing. A test whose outcome depends on which of three behaviours the platform
# picks is testing the platform, not recovery.
#
# Nothing is lost by dropping it. Worker *death* is covered deterministically by
# DEATHS below, which kills the process outright, and hangs are covered by the
# five shapes here that hang reliably everywhere.
HANGS = {
    "sleep": "import time\ntime.sleep(30)",
    "busy_loop": "x = 0\nwhile True:\n    x += 1",
    # Exercise allocator-heavy code without relying on OS overcommit or OOM
    # behavior, which differs across Python versions and CI platforms.
    "allocation_loop": "while True:\n    bytearray(4096)",
    "runaway_string": "s = 'a'\nfor _ in range(40):\n    s = s + s",
    "blocking_read": "import sys\nsys.stdin.read()",
}


@pytest.mark.parametrize("name,body", sorted(HANGS.items()))
def test_recovers_from_each_hang_shape(name, body):
    lm = ScriptedLM([code(body), DONE])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=2, recover_worker_timeouts=1).run()
    assert result.submitted is True, f"did not recover from {name}"
    assert result.payload["answer"] == "ok"


# A worker does not only hang: it also dies. An OOM kill is the usual way a
# Fabric worker goes, and unbounded recursion on Python 3.10 overflows the C
# stack and segfaults *before* any timeout fires, arriving here rather than as a
# WorkerTimeout. That case is deliberately not in HANGS above, because which
# behaviour it produces varies by platform. These kill the process outright, so
# the death path is covered identically on every Python.
DEATHS = {
    "hard_exit": "import os\nos._exit(1)",
    "sys_exit": "import sys\nsys.exit(3)",
    "segfault": "import ctypes\nctypes.string_at(0)",
    "kill_self": (
        "import os, signal\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
    ),
}


@pytest.mark.parametrize("name,body", sorted(DEATHS.items()))
def test_recovers_when_the_worker_dies(name, body):
    lm = ScriptedLM([code(body), DONE])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=10, recover_worker_timeouts=1).run()
    assert result.submitted is True, f"did not recover from {name}"
    assert result.payload["answer"] == "ok"


def test_worker_death_is_not_reported_as_a_timeout():
    """A crash and a hang need different advice, so they must stay distinct."""
    lm = ScriptedLM([code(DEATHS["hard_exit"])] * 6)
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=3,
                      timeout=10, recover_worker_timeouts=1).run()
    assert result.submitted is False
    assert result.failure_reason == "worker_died"


def test_death_recovery_tells_the_model_about_memory_not_speed():
    lm = ScriptedLM([code(DEATHS["hard_exit"]), DONE])
    RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
             timeout=10, recover_worker_timeouts=1).run()
    nudge = "\n".join(lm.seen)
    assert "killed the Python worker process" in nudge
    assert "still running after" not in nudge, "a crash was described as a timeout"


def test_recovers_after_several_good_turns():
    """State built over earlier turns is lost; the run still finishes."""
    lm = ScriptedLM([
        code("a = 1\nprint(a)"),
        code("b = a + 1\nprint(b)"),
        code(HANGS["sleep"]),
        code("SUBMIT(answer='after')"),
    ])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=8,
                      timeout=2, recover_worker_timeouts=1).run()
    assert result.submitted is True
    assert result.payload["answer"] == "after"


def test_prior_turns_are_kept_in_the_trajectory():
    """The point of recovering is not discarding the work already done."""
    lm = ScriptedLM([
        code("a = 1"),
        code(HANGS["sleep"]),
        code("SUBMIT(answer='kept')"),
    ])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=8,
                      timeout=2, recover_worker_timeouts=1).run()
    turns = [t for t in result.trajectory if getattr(t, "code", None)]
    assert len(turns) >= 3, "earlier turns were dropped"
    assert any("Worker timed out" in (t.error or "") for t in turns)


def test_file_inputs_are_rebound():
    """File handles, not just plain values, must survive the restart."""
    from fabric_rlm import File
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "d.txt"
        p.write_text("hello", encoding="utf-8")
        lm = ScriptedLM([
            code(HANGS["sleep"]),
            code("SUBMIT(answer=doc.read_text())"),
        ])
        result = RLM.task(task="t", inputs={"doc": File(p)}, outputs=["answer"],
                          lm=lm, max_turns=4, timeout=2,
                          recover_worker_timeouts=1).run()
    assert result.submitted is True
    assert result.payload["answer"] == "hello"


def test_multiple_recoveries_within_budget():
    lm = ScriptedLM([
        code(HANGS["sleep"]),
        code(HANGS["busy_loop"]),
        code("SUBMIT(answer='third')"),
    ])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=8,
                      timeout=2, recover_worker_timeouts=2).run()
    assert result.submitted is True
    assert result.payload["answer"] == "third"


def test_recovery_respects_the_turn_budget():
    """Recovery consumes turns; it must not loop past max_turns."""
    lm = ScriptedLM([code(HANGS["sleep"])] * 10)
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=3,
                      timeout=2, recover_worker_timeouts=99).run()
    assert result.submitted is False
    turns = [t for t in result.trajectory if getattr(t, "code", None)]
    assert len(turns) <= 3, "ran past max_turns"


def test_recovery_works_with_skills_loaded():
    lm = ScriptedLM([code(HANGS["sleep"]), DONE])
    result = RLM.task(task="analyse this csv", outputs=["answer"], lm=lm,
                      max_turns=4, timeout=2, recover_worker_timeouts=1,
                      skills=["data_exploration"],
                      skill_loader=SkillLoader()).run()
    assert result.submitted is True


def test_sub_lm_still_configured_after_restart():
    """configure_lm must be re-applied, or nested predict calls break.

    The sub-LM is configured inside the worker, so the spec has to be
    serializable -- a model string, not a Python callable.
    """
    lm = ScriptedLM([
        code(HANGS["sleep"]),
        code("SUBMIT(answer=str(callable(predict_sync)))"),
    ])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=2, recover_worker_timeouts=1,
                      sub_lm="openai/gpt-4o-mini").run()
    assert result.submitted is True
    assert result.payload["answer"] == "True", "sub-LM was not reconfigured"
