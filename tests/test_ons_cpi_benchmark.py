from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "ons_cpi_rlm_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ons_cpi_rlm_benchmark", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_rlm_uses_excel_extract_and_data_exploration_skills():
    bench = load_module()

    assert bench.DEFAULT_SKILLS == ("excel_extract", "data_exploration")
    kwargs = bench.rlm_kwargs("task text", lm=object(), max_turns=7, timeout=30)

    assert kwargs["skills"] == ["excel_extract", "data_exploration"]
    assert kwargs["outputs"] == ["payload_json"]
    assert kwargs["max_turns"] == 7
    assert kwargs["timeout"] == 30


def test_normalize_payload_accepts_json_string_and_dict():
    bench = load_module()

    assert bench.normalize_payload('{"answers": [{"id": "q1"}]}') == {
        "answers": [{"id": "q1"}]
    }
    assert bench.normalize_payload({"results": [{"cdid": "D7BT"}]}) == {
        "results": [{"cdid": "D7BT"}]
    }


def test_extract_payload_preserves_dict_submitted_as_payload_json():
    bench = load_module()

    class Result:
        outputs = {"payload_json": {"results": [{"cdid": "D7BT"}]}}

    assert bench._extract_payload(Result()) == {"results": [{"cdid": "D7BT"}]}


def test_scores_exact_lookup_answers_by_cdid_period_and_value():
    bench = load_module()
    payload = {
        "results": [
            {"cdid": "D7BT", "period": "2024 MAR", "value": 133},
            {"cdid": "D7G7", "period": "2024 MAR", "value": 3.2},
            {"cdid": "CHAW", "period": "2024 MAR", "value": 383},
            {"cdid": "L522", "period": "2024 MAR", "value": 131.6},
            {"cdid": "ZPTX", "period": "2025 JAN", "value": 1954},
        ]
    }

    score = bench.score_exact_lookup_payload(payload)

    assert score["n_value_ok"] == 5
    assert score["n_total"] == 5
    assert all(row["value_ok"] for row in score["details"])


def test_scores_reasoning_answers_by_value_cdid_and_title():
    bench = load_module()
    payload = {
        "answers": [
            {
                "id": "q1_cpi_index_mar24",
                "matched_title": "CPI INDEX 00: ALL ITEMS 2015=100",
                "matched_cdid": "D7BT",
                "period": "2024 MAR",
                "value": 133.0,
            },
            {
                "id": "q2_cpi_annual_rate_mar24",
                "matched_title": "CPI ANNUAL RATE 00: ALL ITEMS 2015=100",
                "matched_cdid": "D7G7",
                "period": "2024 MAR",
                "value": 3.2,
            },
        ]
    }

    score = bench.score_reasoning_payload(payload)

    assert score["n_value_ok"] == 2
    assert score["n_cdid_ok"] == 2
    assert score["n_title_ok"] == 2
    assert score["n_total"] == 5
