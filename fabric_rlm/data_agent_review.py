"""The old path of the Data Agent review, kept so existing imports keep working.

The review itself lives in :mod:`fabric_rlm.experimental.data_agent_review`;
the source modelling it shares with the sweep, the brief, the reports and the
KPIs lives in :mod:`fabric_rlm.source_model`. Every name that was importable
from here resolves to its new home.
"""

from __future__ import annotations

from . import source_model as _source_model
from .experimental import data_agent_review as _review

__all__ = list(_review.__all__)


def __getattr__(name: str) -> object:
    for module in (_review, _source_model):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(dir(_review)) | set(dir(_source_model)) | set(globals()))
