"""Scripted LM for plumbing tests: replays turns from a Python file.

The turns file must define ``TURNS = ["...", ...]``; each call returns the
next entry. This exists to validate the adapter end to end (File inputs,
subprocess execution, output placement, evaluator ingestion) without a live
model. It says nothing about model capability.
"""

from __future__ import annotations

import runpy


class StubLM:
    def __init__(self, turns_path: str):
        ns = runpy.run_path(turns_path)
        self.turns = list(ns["TURNS"])
        self.i = 0

    def __call__(self, messages=None, prompt=None, **kwargs):
        if self.i >= len(self.turns):
            raise RuntimeError("StubLM: out of scripted turns")
        turn = self.turns[self.i]
        self.i += 1
        return [turn]
