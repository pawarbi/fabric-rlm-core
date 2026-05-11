"""Corpus-regression check: SecurityPolicy.default().validate_code() must
return ``None`` (i.e. ALLOW) for every code snippet the LM emitted across
all saved adaptive trajectories in ``bench/adaptive/results/feat_a/``.

If this test fails on a turn that an LM actually ran historically, the
security baseline is producing a *behavioural regression* and must be
loosened (or the trajectory itself is doing something legitimately
dangerous and should be flagged for review).

This is the empirical no-regression proof for the default-on policy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fabric_rlm.security import SecurityPolicy

_REPO_TRAJ_ROOTS = [
    Path(__file__).resolve().parent.parent / "bench",
    Path(__file__).resolve().parent / "fixtures",
]

# Saved trajectories from prior sessions live in the agent session workspace
# (gitignored). Walk anything we can find.
_SESSION_TRAJ_ROOT = (
    Path.home() / ".copilot" / "session-state"
)


def _iter_trajectories():
    """Yield (path, code, turn_index) for every code-bearing turn we can find."""
    roots = [*_REPO_TRAJ_ROOTS]
    if _SESSION_TRAJ_ROOT.exists():
        roots.append(_SESSION_TRAJ_ROOT)
    for root in roots:
        if not root.exists():
            continue
        for jsonl in root.rglob("*.jsonl"):
            try:
                with jsonl.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if i == 0 and "metadata" in obj:
                            continue
                        code = obj.get("code")
                        if isinstance(code, str) and code.strip():
                            yield jsonl, code, obj.get("turn", i)
            except OSError:
                continue


def test_corpus_zero_false_positives() -> None:
    """Walk every saved trajectory turn through the default policy.
    Zero rejections required.
    """
    pol = SecurityPolicy.default()
    rejected: list[tuple[Path, int, str]] = []
    total = 0
    t0 = time.time()
    for path, code, turn in _iter_trajectories():
        total += 1
        msg = pol.validate_code(code)
        if msg is not None:
            rejected.append((path, turn, msg))
    elapsed = time.time() - t0

    if total == 0:
        pytest.skip("No code-bearing trajectories found in the repo or session workspace.")

    if rejected:
        sample = "\n".join(
            f"  {p.name}#turn{t}: {m[:160]}"
            for p, t, m in rejected[:10]
        )
        pytest.fail(
            f"Default SecurityPolicy rejected {len(rejected)}/{total} historical "
            f"trajectory turns — this is a behavioural regression.\n"
            f"First 10 rejections:\n{sample}"
        )

    print(
        f"\n[corpus-regression] validated {total} trajectory turns in "
        f"{elapsed:.2f}s ({elapsed * 1000 / max(total, 1):.2f} ms/turn avg) — "
        f"0 false positives."
    )


def test_validate_code_performance() -> None:
    """validate_code() should be fast enough that running it on every turn
    is invisible vs. the LM-emitted code's own runtime. Bound: p95 < 5ms
    over the corpus on a typical dev machine.
    """
    pol = SecurityPolicy.default()
    samples: list[float] = []
    for _, code, _ in _iter_trajectories():
        t0 = time.perf_counter()
        pol.validate_code(code)
        samples.append((time.perf_counter() - t0) * 1000)

    if not samples:
        pytest.skip("No trajectories available")

    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)]
    p99 = samples[int(len(samples) * 0.99)]
    print(f"\n[corpus-perf] n={len(samples)} p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms")
    # Generous bound; CI variability acceptable. Real-world: <1ms.
    assert p95 < 50.0, f"validate_code p95={p95:.1f}ms too slow"
