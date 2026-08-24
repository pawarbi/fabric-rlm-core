"""Recovery must not leak workers or file handles.

A Fabric notebook session is long-lived: the same kernel runs many RLM calls.
If each recovery left an orphaned subprocess behind, a notebook doing repeated
analyses would accumulate them until the executor ran out of resources. These
also cover the environment differences that matter there -- inputs behind a
mounted path, and repeated construction in one process.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

from fabric_rlm import File, RLM


class ScriptedLM:
    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def __call__(self, messages=None, prompt=None, **kwargs):
        turn = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return [turn]


def code(b):
    return f"```python\n{b}\n```"


HANG = code("import time\ntime.sleep(30)")
DONE = code("SUBMIT(answer='ok')")


def _child_count() -> int:
    """Live python child processes of this process, or -1 if unavailable."""
    try:
        import psutil
    except ImportError:
        return -1
    me = psutil.Process(os.getpid())
    return sum(1 for c in me.children(recursive=True) if c.is_running())


def test_no_worker_leak_across_many_recoveries():
    before = _child_count()
    if before < 0:
        pytest.skip("psutil not installed")
    for _ in range(5):
        RLM.task(task="t", outputs=["answer"], lm=ScriptedLM([HANG, DONE]),
                 max_turns=4, timeout=2, recover_worker_timeouts=1).run()
    after = _child_count()
    assert after <= before + 1, (
        f"workers leaked: {before} children before, {after} after five recoveries")


def test_repeated_runs_in_one_process_stay_correct():
    """A notebook kernel reuses the process; results must not drift."""
    for i in range(4):
        result = RLM.task(
            task="t", inputs={"n": i}, outputs=["answer"],
            lm=ScriptedLM([HANG, code("SUBMIT(answer=str(n * 10))")]),
            max_turns=4, timeout=2, recover_worker_timeouts=1).run()
        assert result.submitted is True
        assert result.payload["answer"] == str(i * 10), "input bled between runs"


def test_recovery_with_a_mounted_style_path():
    """Inputs bound by absolute path (as a Lakehouse mount gives) re-bind."""
    with tempfile.TemporaryDirectory() as tmp:
        nested = pathlib.Path(tmp) / "Files" / "data"
        nested.mkdir(parents=True)
        csv = nested / "t.csv"
        csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        lm = ScriptedLM([
            HANG,
            code("import csv as c\n"
                 "rows = list(c.reader(open(table.path)))\n"
                 "SUBMIT(answer=str(len(rows)))"),
        ])
        result = RLM.task(task="t", inputs={"table": File(csv)},
                          outputs=["answer"], lm=lm, max_turns=4, timeout=2,
                          recover_worker_timeouts=1).run()
    assert result.submitted is True
    assert result.payload["answer"] == "3"


def test_worker_is_shut_down_after_a_recovered_run():
    """The restarted worker must still be cleaned up by the context manager."""
    before = _child_count()
    if before < 0:
        pytest.skip("psutil not installed")
    RLM.task(task="t", outputs=["answer"], lm=ScriptedLM([HANG, DONE]),
             max_turns=4, timeout=2, recover_worker_timeouts=1).run()
    assert _child_count() <= before + 1
