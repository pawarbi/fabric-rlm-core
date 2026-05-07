"""pytest configuration for tests/behavior.

Registers the ``--behavior-results=<path>`` option used by the gating test to
write a structured per-run summary that the CI workflow uploads as an artifact.
"""

from __future__ import annotations


def pytest_addoption(parser):  # noqa: D401 — pytest hook
    parser.addoption(
        "--behavior-results",
        default=None,
        help="path to write behavior CI JSON results (used by tests/behavior/test_behavior_baseline.py)",
    )
