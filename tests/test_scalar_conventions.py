"""One set of scalar conventions on every path a value takes: the serializer, a worker SUBMIT, a typed output, a lakehouse query cell and a registered-operation packet cell."""

from __future__ import annotations

import datetime as dt
import decimal
import json

import pytest

from fabric_rlm.serializers import freeze, freeze_submit_payload


def _is_opaque(value: object) -> bool:
    return isinstance(value, dict) and value.get("__serializable__") is False


def test_nanosecond_datetime64_and_timedelta64_read_as_dates_and_days() -> None:
    np = pytest.importorskip("numpy")
    assert freeze(np.datetime64("2026-07-01T13:45:00", "ns")) == "2026-07-01T13:45:00"
    assert freeze(np.datetime64("2026-07-01T13:45:00.123456789")) == "2026-07-01T13:45:00.123456"
    assert freeze(np.timedelta64(2, "D").astype("timedelta64[ns]")) == 2.0
    assert freeze(np.timedelta64(90, "m").astype("timedelta64[ns]")) == 0.0625
    assert freeze(np.datetime64("NaT", "ns")) is None
    assert freeze(np.array(np.datetime64("2026-07-01", "ns"))) == "2026-07-01T00:00:00"
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"day": pd.to_datetime(["2026-07-01", "2026-07-03"])})
    assert freeze(frame["day"].to_numpy()[0]) == "2026-07-01T00:00:00"  # pandas' default unit is the nanosecond
    assert freeze(frame["day"].values.max()) == "2026-07-03T00:00:00"
    assert freeze((frame["day"] - frame["day"].iloc[0]).to_numpy()[1]) == 2.0


def test_dictionary_keys_follow_the_scalar_conventions() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"id": [1, 1, 2], "qty": [1, 2, 3], "day": pd.to_datetime(["2026-07-01", "2026-07-01", "2026-07-02"])})
    assert freeze(frame.groupby("day")["qty"].sum().to_dict()) == {"2026-07-01T00:00:00": 3, "2026-07-02T00:00:00": 3}
    assert freeze(frame.groupby("id")["qty"].sum().to_dict()) == {"1": 3, "2": 3}
    assert freeze({dt.date(2026, 7, 1): 5, np.int64(7): 1, True: 2, "k": 3}) == {"2026-07-01": 5, "7": 1, "True": 2, "k": 3}
    json.dumps(freeze({pd.Timestamp("2026-07-01"): [np.int64(1)]}))


def test_string_subclasses_become_plain_strings() -> None:
    np = pytest.importorskip("numpy")
    frozen = freeze(np.str_("s"))
    assert frozen == "s" and type(frozen) is str
    assert type(freeze({np.str_("k"): np.str_("v")})["k"]) is str


def test_a_worker_submit_carries_frame_values_as_dates_days_and_numbers() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    from fabric_rlm import Interpreter

    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'qty': [1, 2, 3], 'day': pd.to_datetime(['2026-07-01', '2026-07-02', '2026-07-03'])})\n"
        "SUBMIT(total=df['qty'].sum(), flag=(df['qty'] > 1).any(), first=df['day'].to_numpy()[0],\n"
        "       span=(df['day'].max() - df['day'].min()).to_numpy(), by_day=df.groupby('day')['qty'].sum().to_dict(),\n"
        "       rows=df.head(1).to_dict('records'), one=df['qty'].to_numpy()[:1])\n"
    )
    with Interpreter() as interpreter:
        result = interpreter.execute(code)
    payload = result.submit_payload
    assert result.error is None and payload is not None
    assert payload["total"] == 6 and payload["flag"] is True
    assert payload["first"] == "2026-07-01T00:00:00" and payload["span"] == 2.0
    assert payload["by_day"] == {"2026-07-01T00:00:00": 1, "2026-07-02T00:00:00": 2, "2026-07-03T00:00:00": 3}
    assert payload["rows"] == [{"qty": 1, "day": "2026-07-01T00:00:00"}]
    assert _is_opaque(payload["one"])  # a one-element array is still a container
    json.dumps(payload)


class _OneShot:
    def __init__(self, code: str) -> None:
        self.code = code

    def __call__(self, *, messages: list) -> str:
        return "```python\n" + self.code + "\n```"


@pytest.mark.parametrize(
    ("declared", "code", "expected"),
    [
        ({"total": int}, "import numpy as np\nSUBMIT(total=np.int64(6))", {"total": 6}),
        ({"flag": bool}, "import numpy as np\nSUBMIT(flag=np.bool_(True))", {"flag": True}),
        ({"mean": float}, "import numpy as np\nSUBMIT(mean=np.float64(2.5))", {"mean": 2.5}),
        ({"amount": float}, "import decimal\nSUBMIT(amount=decimal.Decimal('2.50'))", {"amount": 2.5}),
        ({"day": str}, "import datetime as dt\nSUBMIT(day=dt.date(2026, 7, 1))", {"day": "2026-07-01"}),
        ({"day": str}, "import numpy as np\nSUBMIT(day=np.datetime64('2026-07-01', 'ns'))", {"day": "2026-07-01T00:00:00"}),
        ({"day": str}, "import pandas as pd\nSUBMIT(day=pd.Timestamp('2026-07-01'))", {"day": "2026-07-01T00:00:00"}),
        ({"days": float}, "import datetime as dt\nSUBMIT(days=dt.timedelta(days=2))", {"days": 2.0}),
    ],
    ids=["numpy-int", "numpy-bool", "numpy-float", "decimal", "date", "datetime64-ns", "timestamp", "timedelta"],
)
def test_typed_outputs_accept_the_converted_scalars(declared, code, expected) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    from fabric_rlm import RLM

    result = RLM.from_task("Return the value.", outputs=declared, lm=_OneShot(code), max_turns=1, timeout=30).run()
    assert result.submitted, getattr(result, "failure_reason", None)
    assert result.payload == expected


def test_lakehouse_query_cells_share_the_conventions_and_natives_skip_the_serializer() -> None:
    duckdb = pytest.importorskip("duckdb")
    from fabric_rlm.lakehouse import _json_value

    row = duckdb.connect().execute(
        "SELECT 12.50::DECIMAL(18,2) AS dec, DATE '2026-07-01' AS d, INTERVAL 5 DAY AS iv, INTERVAL 90 MINUTE AS iv2, "
        "[1, 2, 3] AS l, {'a': 1} AS s, MAP {'k': 'v'} AS m, uuid() AS u, '\\x00\\xff'::BLOB AS b, 'x' AS t, 7 AS n, TRUE AS bo, NULL AS nul"
    ).fetchone()
    dec, d, iv, iv2, l, s, m, u, b, t, n, bo, nul = row
    assert _json_value(dec) == 12.5 and _json_value(d) == "2026-07-01"
    assert _json_value(iv) == 5.0 and _json_value(iv2) == 0.0625  # a duration is fractional days, as in a payload
    assert _json_value(l) == [1, 2, 3] and _json_value(s) == {"a": 1} and _json_value(m) == {"k": "v"}
    assert _json_value(u) == str(u) and _json_value(b) == "00ff"
    for native in (t, n, bo, nul):
        assert _json_value(native) is native  # the common cells come back untouched
    frozen = freeze_submit_payload({"iv": iv, "l": l})
    assert frozen == {"iv": 5.0, "l": [1, 2, 3]}


def test_packet_cells_share_the_conventions_and_keep_their_finiteness_rule() -> None:
    from fabric_rlm.knowledge_execution import _scalar

    assert _scalar(dt.timedelta(days=1, hours=12), "f") == 1.5
    assert _scalar(dt.date(2026, 7, 1), "f") == "2026-07-01"
    assert _scalar(b"\x00\xff", "f") == "00ff"
    assert _scalar(decimal.Decimal("2.50"), "f") == 2.5
    assert _scalar(float("nan"), "f") is None
    with pytest.raises(ValueError, match="must be finite"):
        _scalar(float("inf"), "f")
    with pytest.raises(ValueError, match="must be finite"):
        _scalar(decimal.Decimal("NaN"), "f")
    with pytest.raises(ValueError, match="must be a scalar value"):
        _scalar([1, 2], "f")
    np = pytest.importorskip("numpy")
    assert _scalar(np.int64(3), "f") == 3 and type(_scalar(np.int64(3), "f")) is int
    assert _scalar(np.datetime64("2026-07-01T00:00:00", "ns"), "f") == "2026-07-01T00:00:00"
    assert _scalar(np.timedelta64(36, "h").astype("timedelta64[ns]"), "f") == 1.5
    assert _scalar(np.float64("nan"), "f") is None
    assert type(_scalar(np.str_("s"), "f")) is str
