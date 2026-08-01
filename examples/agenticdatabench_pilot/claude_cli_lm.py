"""An LM callable backed by a logged-in Claude Code CLI.

fabric-rlm accepts any callable that takes ``messages=[{role, content}, ...]``
and returns text (see ``resolve_lm`` in fabric_rlm/lm.py). This shim flattens
the chat transcript into one prompt and runs ``claude -p``, which uses the
CLI's own login rather than an API key. Requires ``claude`` on PATH and a
completed ``/login`` in a terminal session.
"""

from __future__ import annotations

import shutil
import subprocess


class ClaudeCLILM:
    def __init__(self, model: str = "sonnet", timeout: int = 600):
        self.model = model
        self.timeout = timeout
        self.exe = shutil.which("claude")
        if not self.exe:
            raise RuntimeError("claude CLI not found on PATH")

    def __call__(self, messages=None, prompt=None, **kwargs):
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages]
        flat = (
            "\n\n".join(parts)
            + "\n\nYou are the ASSISTANT in the transcript above. "
            "Reply with the assistant's next message only."
        )
        proc = subprocess.run(
            [self.exe, "-p", "--output-format", "text", "--model", self.model],
            input=flat,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {proc.stderr[:500]}")
        return [proc.stdout]
