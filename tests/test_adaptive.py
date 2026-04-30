from fabric_rlm.experimental.adaptive import AdaptiveOrchestrator, AdaptiveStrategy, first_passing
import pytest

pytestmark = pytest.mark.experimental


def test_first_passing_returns_first_valid_strategy() -> None:
    result = first_passing(
        [
            ("cheap", lambda: 1),
            ("backup", lambda: 2),
        ],
        validator=lambda value: value == 2,
    )

    assert result.name == "backup"
    assert result.value == 2
    assert result.passed


def test_adaptive_orchestrator_avoids_fanout_when_primary_passes() -> None:
    calls = []
    orchestrator = AdaptiveOrchestrator(validator=lambda value: value == 1)

    result = orchestrator.run(
        AdaptiveStrategy("cheap", lambda: calls.append("cheap") or 1),
        [AdaptiveStrategy("backup", lambda: calls.append("backup") or 1)],
    )

    assert result.passed
    assert result.winning_strategy == "cheap"
    assert not result.fanout_used
    assert calls == ["cheap"]


def test_adaptive_orchestrator_fans_out_after_primary_failure() -> None:
    orchestrator = AdaptiveOrchestrator(validator=lambda value: value == 3)

    result = orchestrator.run(
        AdaptiveStrategy("cheap", lambda: 1),
        [
            AdaptiveStrategy("backup-a", lambda: 2),
            AdaptiveStrategy("backup-b", lambda: 3),
        ],
    )

    assert result.passed
    assert result.fanout_used
    assert result.value == 3
    assert {attempt.name for attempt in result.attempts}.issuperset({"cheap", "backup-b"})


