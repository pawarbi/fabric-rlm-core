"""Bandit-augmented escalation policy.

Per-trajectory analysis of the 0.1.10 validation run showed a clear pattern:
on hard algorithmic benchmark templates the cheap LM
produces nonsense at rung 0 every time, and budget spent on rungs 0/1/2/3
is pure waste. The only rung that ever produces a win on those problems
is rung 4 (strong LM) — and even then often only because the cheap rungs
had already burned the attempt budget.

:class:`BanditPolicy` learns ``(task_key, rung) -> Beta(α, β)`` posteriors
from past attempts and Thompson-samples a starting rung the next time it
sees the same ``task_key``. Once started, escalation behavior is identical
to :class:`LadderPolicy` — only the entry point is bandit-driven.

Persistence is opt-in: pass ``state=BanditState.from_path(path)`` to load
priors from a JSON file and call ``state.save()`` after each task.

The policy is deliberately conservative:

* Cold-start ``Beta(1, 1)`` (uniform) means the first ``warmup`` attempts on
  any unseen ``task_key`` always go through the standard ladder. The bandit
  only takes over once it has signal.
* The Thompson-sampled rung is then weighted by an inverse-cost prior so
  that, given equal sampled success probabilities, the cheaper rung wins.
  This prevents the bandit from over-eagerly jumping to the strong LM.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adaptive_policy import (
    AttemptConfig,
    AttemptRecord,
    DifficultyVerdict,
    LadderPolicy,
)


# Default rung costs reflect the standard cross-model LadderPolicy:
# rungs 0-3 are cheap-LM, rung 4 is strong-LM (~30× cheap-LM cost).
# For an effort-climbing ladder (single LM, varying reasoning_effort) the
# costs differ — pass ``rung_cost`` to :class:`BanditPolicy` to override.
_RUNG_COST: dict[int, float] = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 6.0,   # parallel rollouts ×N (default N=3)
    4: 30.0,  # strong LM
}


# Approx cost ratios for an effort-only ladder on a reasoning model like
# gpt-5: each effort step roughly 3× the previous. Rung 4 is high+parallel-N
# (default N=3). Use with :class:`EffortLadderPolicy`.
_EFFORT_RUNG_COST: dict[int, float] = {
    0: 1.0,    # minimal
    1: 3.0,    # low (or base + retry)
    2: 8.0,    # medium
    3: 25.0,   # high
    4: 75.0,   # high × N parallel
}


def _rung_cost(rung: int) -> float:
    return _RUNG_COST.get(rung, 1.0 + rung)


@dataclass
class BanditState:
    """Persistent ``(task_key, rung) -> Beta(α, β)`` posteriors.

    This is the data substrate; the policy reads from it and the caller
    writes to it after each completed task.
    """

    priors: dict[str, dict[int, tuple[float, float]]] = field(default_factory=dict)
    path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def from_path(cls, path: str | Path) -> "BanditState":
        """Load state from ``path`` (creates empty state if file missing)."""

        p = Path(path)
        priors: dict[str, dict[int, tuple[float, float]]] = {}
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                for task_key, rung_map in (raw or {}).items():
                    priors[task_key] = {
                        int(rung): (float(ab[0]), float(ab[1]))
                        for rung, ab in (rung_map or {}).items()
                    }
            except (OSError, ValueError, KeyError):
                priors = {}
        return cls(priors=priors, path=p)

    def save(self) -> None:
        """Atomically write ``self.priors`` to ``self.path`` (no-op if path is None)."""

        if self.path is None:
            return
        with self._lock:
            payload = {
                task_key: {str(rung): list(ab) for rung, ab in rung_map.items()}
                for task_key, rung_map in self.priors.items()
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)

    def beta_for(self, task_key: str, rung: int) -> tuple[float, float]:
        """Return Beta(α, β) for ``(task_key, rung)`` — defaults to ``(1, 1)``."""

        return self.priors.get(task_key, {}).get(rung, (1.0, 1.0))

    def total_observations(self, task_key: str) -> int:
        """Count of recorded outcomes (passes + fails) across all rungs for ``task_key``."""

        rung_map = self.priors.get(task_key, {})
        return sum(int(a + b) - 2 for a, b in rung_map.values() if a + b > 2)

    def record(self, task_key: str, rung: int, passed: bool) -> None:
        """Update the posterior for ``(task_key, rung)``."""

        with self._lock:
            rung_map = self.priors.setdefault(task_key, {})
            alpha, beta = rung_map.get(rung, (1.0, 1.0))
            if passed:
                alpha += 1.0
            else:
                beta += 1.0
            rung_map[rung] = (alpha, beta)


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Draw one sample from ``Beta(alpha, beta)`` using the gamma-ratio method."""

    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    s = x + y
    if s == 0.0:
        return 0.5
    return x / s


@dataclass
class BanditPolicy(LadderPolicy):
    """Ladder policy with a Thompson-sampling bandit on the *starting* rung.

    Subclassing rather than wrapping keeps the runner integration trivial —
    only ``next_decision`` differs, and only when ``len(attempts) == 0`` and
    the bandit has accumulated enough signal.

    Parameters
    ----------
    state
        Shared :class:`BanditState`. ``None`` disables the bandit (degrades
        cleanly to plain :class:`LadderPolicy` behavior).
    task_key
        Identifier the bandit groups outcomes by — typically a template
        name (``"MFMC"``, ``"MCM"``) or a coarse domain tag. The empty
        string also disables the bandit for this run.
    warmup
        Minimum number of recorded outcomes for ``task_key`` before the
        bandit takes over the starting-rung decision. Below this it
        defers to the ladder so we don't bias on a single noisy sample.
    rng
        Optional ``random.Random`` for deterministic tests.
    """

    state: BanditState | None = None
    task_key: str = ""
    warmup: int = 2
    rng: random.Random = field(default_factory=random.Random)
    # Override per-rung cost map to match the underlying ladder. ``None``
    # uses the cross-model default in :data:`_RUNG_COST`. For an effort-only
    # ladder pass :data:`_EFFORT_RUNG_COST` (or any custom dict).
    rung_cost: dict[int, float] | None = None

    def _cost_for(self, rung: int) -> float:
        if self.rung_cost is not None:
            return self.rung_cost.get(rung, 1.0 + rung)
        return _rung_cost(rung)

    def _bandit_should_drive(self) -> bool:
        if self.state is None or not self.task_key:
            return False
        return self.state.total_observations(self.task_key) >= self.warmup

    def _bandit_eligible_rungs(self) -> list[int]:
        """Return rungs eligible for Thompson sampling.

        Default: every rung is eligible. Subclasses can override to gate
        expensive new rungs behind evidence on cheaper ones (prevents
        uniform-prior exploration from torching budget on a broken or
        very expensive new rung before any evidence accumulates).
        """
        return list(range(self.max_rung + 1))

    def _bandit_starting_rung(self) -> tuple[int, float]:
        """Thompson-sample a rung, with cost only used as a small tie-breaker.

        Returns ``(chosen_rung, sampled_p)``. We do *not* divide by rung cost
        — that would make any fixed cost ratio (e.g. strong LM 30× cheap LM)
        permanently dominate the posterior signal. The whole point of the
        bandit is that when cheap-LM evidence shows ~0% success rate, the
        true *expected* cost of rung 0 is the cost of failing AND then
        climbing the ladder — much higher than rung 4's nominal cost. So
        Thompson on the raw posterior is the right decision rule.

        Cost only intercedes when sampled probabilities are within a small
        epsilon of each other, which prevents the bandit from gratuitously
        jumping rungs when several rungs look equally good.
        """

        assert self.state is not None
        epsilon = 0.05  # 5% absolute difference is the tie-window
        eligible = self._bandit_eligible_rungs()
        if not eligible:
            eligible = list(range(self.max_rung + 1))
        candidates: list[tuple[int, float]] = []
        for rung in eligible:
            alpha, beta = self.state.beta_for(self.task_key, rung)
            candidates.append((rung, _beta_sample(alpha, beta, self.rng)))

        # First find max sampled p
        best_rung, best_p = max(candidates, key=lambda rp: rp[1])
        # Then among rungs within epsilon of the best, prefer cheapest
        tied = [(r, p) for r, p in candidates if (best_p - p) <= epsilon]
        if tied:
            best_rung, best_p = min(tied, key=lambda rp: self._cost_for(rp[0]))
        return best_rung, best_p

    def next_decision(
        self,
        attempts: list[AttemptRecord],
    ) -> tuple[DifficultyVerdict, AttemptConfig | None]:
        if not attempts and self._bandit_should_drive():
            chosen, p = self._bandit_starting_rung()
            if chosen > 0:
                verdict = DifficultyVerdict.escalate(
                    chosen,
                    reason=(
                        f"bandit: starting at rung {chosen} "
                        f"(p≈{p:.2f}, key={self.task_key!r})"
                    ),
                )
                cfg = self._build_config(chosen)
                return verdict, cfg
        return super().next_decision(attempts)
