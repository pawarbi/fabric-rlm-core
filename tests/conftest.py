"""Suite-wide fixtures.

The claim-provenance screen rejects a number typed into SUBMIT that no
executed output showed. Most runtime tests drive the loop with fake
interpreters that execute nothing and print nothing, while their scripted
answers type small literals (``SUBMIT(answer=1)``); under the screen every
one of them would spend a repair turn the script did not budget. Those
tests are about other behaviour, so the screen is off for them here and on
everywhere it is the subject: ``tests/test_generalization.py``, the
integrity runtime tests, and the live behavior gate, which must see the
production default.
"""

from __future__ import annotations

import pytest

_SCREEN_ON = (
    "test_generalization.py",
    "test_analytical_integrity_runtime.py",
    "test_analytical_integrity.py",
)


@pytest.fixture(autouse=True)
def _claim_provenance_default(request, monkeypatch):
    path = str(getattr(request.node, "fspath", "") or "").replace("\\", "/")
    if "/tests/behavior/" in path or path.endswith(_SCREEN_ON):
        monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
        return
    monkeypatch.setenv("FABRIC_RLM_CLAIM_PROVENANCE", "0")
