from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from fabric_rlm import File
from fabric_rlm.serializers import (
    SubmitPayloadTooLarge,
    freeze,
    freeze_submit_payload,
    snapshot,
)


@dataclass
class Item:
    name: str
    count: int


def test_freeze_supported_values(tmp_path: Path) -> None:
    file = File(tmp_path / "a.txt")
    value = {
        "item": Item("x", 2),
        "path": tmp_path,
        "file": file,
        "items": [1, 2],
    }

    frozen = freeze(value)

    assert frozen["item"] == {"name": "x", "count": 2}
    assert frozen["path"] == str(tmp_path)
    assert frozen["file"]["path"] == file.path
    assert frozen["items"] == [1, 2]


def test_freeze_opaque_object() -> None:
    frozen = freeze(object())

    assert frozen["__type__"] == "object"
    assert frozen["__serializable__"] is False


def test_snapshot_skips_runtime_and_private_names() -> None:
    ns = {
        "_hidden": 1,
        "File": object(),
        "SUBMIT": object(),
        "predict": object(),
        "answer": 42,
        "helper": lambda: None,
    }

    assert snapshot(ns) == {"answer": 42}


def test_freeze_defaults_remain_bounded() -> None:
    frozen = freeze({"text": "x" * 2_500, "rows": list(range(250))})

    assert frozen["text"].endswith("<truncated, total 2500 chars>")
    assert frozen["rows"][-1] == {"__truncated__": 50}


def test_freeze_can_preserve_unbounded_final_payloads() -> None:
    value = {"text": "x" * 10_000, "rows": list(range(500))}

    frozen = freeze(value, max_string_length=None, max_collection_items=None)

    assert frozen == value


def test_submit_payload_limit_uses_utf8_json_bytes() -> None:
    value = {"text": "é"}
    encoded_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    assert freeze_submit_payload(value, max_bytes=encoded_size) == value
    with pytest.raises(
        SubmitPayloadTooLarge,
        match=rf"max_submit_bytes={encoded_size - 1}.*at least {encoded_size} bytes",
    ):
        freeze_submit_payload(value, max_bytes=encoded_size - 1)


@pytest.mark.parametrize("limit", [0, -1])
def test_submit_payload_limit_must_be_positive(limit: int) -> None:
    with pytest.raises(ValueError, match="max_submit_bytes must be greater than zero"):
        freeze_submit_payload({"answer": 42}, max_bytes=limit)


@pytest.mark.parametrize("limit", [True, 1.5, "100"])
def test_submit_payload_limit_must_be_an_integer(limit: object) -> None:
    with pytest.raises(TypeError, match="max_submit_bytes must be an int"):
        freeze_submit_payload({"answer": 42}, max_bytes=limit)  # type: ignore[arg-type]


# --- numpy / array-library scalars -------------------------------------
#
# ``np.int64`` is NOT a subclass of Python ``int`` (unlike ``np.float64``, which
# does subclass ``float``). Integer scalars therefore fell through every branch
# of ``freeze`` and were emitted as opaque, non-serializable markers — so a
# numerically correct answer such as ``df["qty"].sum()`` was recorded as
# unusable. Observed in 4/150 trials of the dbo evaluation, all graded wrong.


def test_freeze_numpy_integer_scalar_is_a_plain_int() -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(np.int64(93257855))
    assert frozen == 93257855
    assert isinstance(frozen, int)
    assert json.dumps(frozen) == "93257855"


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64",
                                   "uint8", "uint32", "uint64"])
def test_freeze_numpy_integer_widths(dtype: str) -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(getattr(np, dtype)(34))
    assert frozen == 34
    assert isinstance(frozen, int)


def test_freeze_numpy_bool_scalar_is_a_plain_bool() -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(np.bool_(True))
    assert frozen is True


def test_freeze_numpy_float_scalar_round_trips() -> None:
    np = pytest.importorskip("numpy")
    assert freeze(np.float32(1.5)) == 1.5
    assert freeze(np.float64(8.1319)) == 8.1319


def test_freeze_zero_dimensional_array_is_a_scalar() -> None:
    np = pytest.importorskip("numpy")
    assert freeze(np.array(7)) == 7


def test_freeze_pandas_integer_aggregate_is_serializable() -> None:
    pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    total = pd.DataFrame({"qty": [1, 2, 3]})["qty"].sum()
    frozen = freeze(total)
    assert frozen == 6
    assert isinstance(frozen, int)
    json.dumps(frozen)


def test_freeze_submit_payload_accepts_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    payload = freeze_submit_payload({"value": np.int64(334), "ok": np.bool_(False)})
    assert payload == {"value": 334, "ok": False}
    json.dumps(payload)


def test_snapshot_reports_numpy_scalar_as_a_value() -> None:
    np = pytest.importorskip("numpy")
    out = snapshot({"total": np.int64(42)})
    assert out["total"] == 42


# Guard the conversion against over-reach: real arrays and frames carry data
# that a single scalar cannot represent, and must stay opaque.


def test_freeze_multi_element_array_stays_opaque() -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(np.array([1, 2, 3]))
    assert frozen["__serializable__"] is False


def test_freeze_single_element_1d_array_stays_opaque() -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(np.array([5]))
    assert frozen["__serializable__"] is False


def test_freeze_dataframe_stays_opaque() -> None:
    pd = pytest.importorskip("pandas")
    frozen = freeze(pd.DataFrame({"a": [1, 2]}))
    assert frozen["__serializable__"] is False


# --- scalars from SQL sources and date arithmetic ----------------------
#
# These reach a SUBMIT payload whenever the model computes in the worker rather
# than through LakehouseSource.query, which normalizes them separately. Both
# paths now share one conversion policy.


def test_freeze_decimal_becomes_a_float() -> None:
    from decimal import Decimal
    frozen = freeze(Decimal("350.75"))
    assert frozen == 350.75
    assert isinstance(frozen, float)


def test_freeze_date_becomes_an_iso_string() -> None:
    import datetime as dt
    assert freeze(dt.date(2024, 1, 15)) == "2024-01-15"


def test_freeze_datetime_becomes_an_iso_string() -> None:
    import datetime as dt
    assert freeze(dt.datetime(2024, 1, 15, 12, 30)) == "2024-01-15T12:30:00"


def test_freeze_time_becomes_an_iso_string() -> None:
    import datetime as dt
    assert freeze(dt.time(12, 30)) == "12:30:00"


def test_freeze_timedelta_becomes_fractional_days() -> None:
    import datetime as dt
    assert freeze(dt.timedelta(days=36)) == 36.0
    assert freeze(dt.timedelta(days=1, hours=12)) == 1.5


def test_freeze_bytes_become_hex() -> None:
    assert freeze(b"\x00\xff") == "00ff"


def test_freeze_pandas_timestamp_and_timedelta() -> None:
    pd = pytest.importorskip("pandas")
    assert freeze(pd.Timestamp("2024-01-15")).startswith("2024-01-15")
    assert freeze(pd.Timedelta(days=36)) == 36.0


def test_freeze_pandas_missing_values_become_null() -> None:
    pd = pytest.importorskip("pandas")
    assert freeze(pd.NaT) is None
    assert freeze(pd.NA) is None


def test_freeze_numpy_datetime_scalars() -> None:
    np = pytest.importorskip("numpy")
    assert freeze(np.datetime64("2024-01-15")).startswith("2024-01-15")
    assert freeze(np.timedelta64(36, "D")) == 36.0


def test_freeze_pyarrow_scalar_unwraps() -> None:
    pa = pytest.importorskip("pyarrow")
    assert freeze(pa.scalar(5)) == 5


def test_freeze_submit_payload_with_sql_scalars() -> None:
    import datetime as dt
    from decimal import Decimal
    payload = freeze_submit_payload({
        "total_revenue": Decimal("350.75"),
        "first_date": dt.date(2024, 1, 15),
        "days_between": dt.timedelta(days=36),
    })
    assert payload == {
        "total_revenue": 350.75,
        "first_date": "2024-01-15",
        "days_between": 36.0,
    }
    json.dumps(payload)
