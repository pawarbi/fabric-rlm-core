"""Data-backed public facade contracts, using real subprocess workers.

Only model responses are scripted. Existing CSV coverage for knowledge and
capture_evidence lives in test_knowledge_learning_runtime.py (capture parity,
active lesson injection, and enrich round trips). Remote egress rejection is
covered by test_block_network.py::test_remote_connect_refused_in_worker; here
block_network is exercised against a real loopback CSV server, not the internet.
"""

from __future__ import annotations

import csv
import socketserver
import threading

import pytest

from fabric_rlm import RLM
from fabric_rlm.netguard import ENV_FLAG
from fabric_rlm.security import SecurityPolicy
from test_generalization import _code
from test_halve_max_iter_param import _CountingScriptedLM
from test_runtime_mock_lm import ScriptedLM


READ_TOTAL = """import csv
with open(source, newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
included = [row for row in rows if row['included'] == 'yes']
total = sum(int(row['amount']) for row in included)
print('included total', total)
"""


@pytest.fixture
def sales_csv(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_text(
        "region,amount,included\n"
        "east,12,yes\n"
        "east,18,yes\n"
        "west,20,yes\n"
        "west,900,no\n",
        encoding="utf-8",
    )
    # Independent host oracle, including the excluded-row trap.
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(int(row["amount"]) for row in rows) == 950
    assert sum(int(row["amount"]) for row in rows if row["included"] == "yes") == 50
    return source


@pytest.fixture(autouse=True)
def deterministic_guards(monkeypatch, _claim_provenance_default):
    # Run the actual production provenance screen, despite conftest's opt-out.
    monkeypatch.delenv("FABRIC_RLM_CLAIM_PROVENANCE", raising=False)
    monkeypatch.delenv("FABRIC_RLM_ANALYTICAL_INTEGRITY", raising=False)
    monkeypatch.delenv(ENV_FLAG, raising=False)


@pytest.mark.parametrize("signature", [None, "source -> total: int, count: int"])
def test_task_inputs_outputs_signature_and_lm_repair(sales_csv, signature):
    task = "Reconcile included sales only; return their total and row count."
    lm = ScriptedLM([
        _code(READ_TOTAL + "SUBMIT(total=total)"),
        _code("SUBMIT(total=total, count=len(included))"),
    ])
    result = RLM.task(
        task=task,
        inputs={"source": sales_csv},
        outputs={"total": int, "count": int},
        signature=signature,
        lm=lm,
        max_turns=2,
        timeout=30,
    ).run()

    assert result.submitted, result.failure_reason
    assert result.payload == {"total": 50, "count": 3}
    assert len(lm.messages) == len(result.turns) == 2
    assert task in "\n".join(message["content"] for message in lm.messages[0])
    assert result.turns[0].stdout.strip() == "included total 50"
    assert result.turns[0].validation_errors == ["Missing required output field 'count'."]
    assert result.turns[1].turn_type == "validation_repair"
    assert "count" in lm.messages[1][-1]["content"]


def test_worker_csv_error_is_repaired_without_rebinding_inputs(sales_csv):
    lm = ScriptedLM([
        _code(READ_TOTAL + "SUBMIT(total=sum(int(row['missing']) for row in included))"),
        _code("SUBMIT(total=sum(int(row['amount']) for row in included))"),
    ])
    result = RLM.task(
        "Total included sales.", inputs={"source": sales_csv}, outputs=["total"],
        lm=lm, max_turns=2, timeout=30,
    ).run()

    assert result.submitted and result.payload == {"total": 50}
    assert len(result.turns) == 2
    assert "KeyError" in result.turns[0].error
    assert "missing" in lm.messages[1][-1]["content"]
    assert result.turns[1].error is None


@pytest.mark.parametrize("max_turns", [1, 2])
def test_max_turns_bounds_csv_work_not_just_configuration(sales_csv, max_turns):
    lm = ScriptedLM([_code(READ_TOTAL), _code("SUBMIT(total=total)")])
    result = RLM.task(
        "Total included sales.", inputs={"source": sales_csv}, outputs=["total"],
        lm=lm, max_turns=max_turns, timeout=30,
    ).run()

    assert len(lm.messages) == len(result.turns) == max_turns
    assert result.final_state["total"] == 50
    if max_turns == 1:
        assert not result.submitted and result.payload is None
        assert result.failure_reason == "max_turns"
    else:
        assert result.submitted and result.payload == {"total": 50}


@pytest.mark.parametrize("recoveries", [0, 1])
def test_timeout_budget_restarts_and_rebinds_csv_input(sales_csv, recoveries):
    lm = ScriptedLM([
        _code(READ_TOTAL),
        _code("import time\ntime.sleep(30)"),
        _code("assert 'total' not in globals()\n" + READ_TOTAL + "SUBMIT(total=total)"),
    ])
    result = RLM.task(
        "Total included sales after recovering slow work.",
        inputs={"source": sales_csv}, outputs=["total"], lm=lm,
        # Recovery itself consumes one turn in addition to the timed-out code.
        max_turns=4, timeout=5, recover_worker_timeouts=recoveries,
    ).run()

    assert result.turns[0].stdout.strip() == "included total 50"
    assert "timeout" in result.turns[1].error.lower()
    if recoveries:
        assert result.submitted and result.payload == {"total": 50}
        assert len(lm.messages) == len(result.turns) == 3
        assert "restarted" in lm.messages[2][-1]["content"].lower()
        assert result.turns[2].error is None
    else:
        assert not result.submitted and result.payload is None
        assert result.failure_reason == "worker_timeout"
        assert len(lm.messages) == len(result.turns) == 2


@pytest.mark.parametrize("threshold", [2, None])
def test_stuck_loop_threshold_stops_repeated_csv_column_errors(sales_csv, threshold):
    broken = _code(READ_TOTAL + "print(rows[0]['missing'])")
    lm = ScriptedLM([broken, broken, _code("SUBMIT(total=total)")])
    result = RLM.task(
        "Total included sales.", inputs={"source": sales_csv}, outputs=["total"],
        lm=lm, max_turns=3, timeout=30, stuck_loop_threshold=threshold,
    ).run()

    assert all("KeyError" in turn.error for turn in result.turns[:2])
    if threshold is None:
        assert result.submitted and result.payload == {"total": 50}
        assert len(lm.messages) == 3
    else:
        assert not result.submitted and result.payload is None
        assert result.failure_reason == "stuck_loop"
        assert len(lm.messages) == len(result.turns) == 2


@pytest.mark.parametrize("mode", ["no_validator", "returns_none", "asserts", "returns_false"])
def test_output_validator_known_source_contract_and_false_caveat(sales_csv, mode):
    seen = []

    def validate(payload):
        seen.append(dict(payload))
        if mode == "returns_false":
            return payload["total"] == 50
        assert payload["total"] == 50, "Included source total must equal 50"
        return None

    # A computed/printed wrong answer passes provenance: it is not a truth oracle.
    first = "SUBMIT(total=total)" if mode == "returns_none" else (
        "wrong = total + 1\nprint(wrong)\nSUBMIT(total=wrong)"
    )
    lm = ScriptedLM([_code(READ_TOTAL + first), _code("SUBMIT(total=total)")])
    result = RLM.task(
        "Total included sales.", inputs={"source": sales_csv}, outputs=["total"],
        lm=lm, max_turns=2, timeout=30, analytical_integrity="strict",
        output_validator=None if mode == "no_validator" else validate,
    ).run()

    assert result.submitted, result.failure_reason
    expected = 50 if mode in {"returns_none", "asserts"} else 51
    assert result.payload == {"total": expected}
    assert result.integrity_ok
    if mode == "asserts":
        assert seen == [{"total": 51}, {"total": 50}]
        assert len(lm.messages) == 2
        history = result.trajectory.metadata["verifier_repair_history"]
        assert history[0]["skill"] == "output_validator"
        assert "must equal 50" in lm.messages[1][-1]["content"]
    else:
        assert len(lm.messages) == 1
        assert seen == ([] if mode == "no_validator" else [{"total": expected}])
        assert "verifier_repair_history" not in result.trajectory.metadata


def test_context_validator_reopens_saved_xlsx_and_repairs_cells(sales_csv, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    destination = tmp_path / "sales-summary.xlsx"
    checked = []

    def validate(payload, context):
        assert payload == {"total": 50, "artifact": str(destination)}
        assert context["inputs"]["source"] == sales_csv
        assert context["inputs"]["destination"] == destination
        assert context["state"]["total"] == 50
        assert len(context["trajectory"].turns) == context["turn"]
        assert destination.is_file(), "Workbook must actually be saved"
        workbook = openpyxl.load_workbook(destination, read_only=True, data_only=True)
        try:
            cells = list(workbook["Summary"].values)
        finally:
            workbook.close()
        checked.append((context["turn"], cells))
        assert cells == [("included_total", "row_count"), (50, 3)], "Saved cells must match source total 50 and count 3"

    lm = ScriptedLM([
        _code(READ_TOTAL + """from openpyxl import Workbook
workbook = Workbook()
sheet = workbook.active
sheet.title = 'Summary'
sheet.append(['included_total', 'row_count'])
sheet.append([total + 1, len(included)])
workbook.save(destination)
SUBMIT(total=total, artifact=str(destination))"""),
        _code("sheet['A2'] = total\nworkbook.save(destination)\nSUBMIT(total=total, artifact=str(destination))"),
    ])
    result = RLM.task(
        "Save an XLSX summary of included sales.",
        inputs={"source": sales_csv, "destination": destination},
        outputs=["total", "artifact"], lm=lm, max_turns=2, timeout=30,
        output_validator_context=validate,
    ).run()

    assert result.submitted, result.failure_reason
    assert checked == [
        (1, [("included_total", "row_count"), (51, 3)]),
        (2, [("included_total", "row_count"), (50, 3)]),
    ]
    assert result.turns[1].turn_type == "verifier_repair"
    assert "Saved cells must match" in lm.messages[1][-1]["content"]
    assert result.trajectory.metadata["verifier_repair_history"][0]["skill"] == "output_validator_context"


@pytest.mark.parametrize("enabled", [True, False])
def test_security_scrubs_only_fake_secret_probe_at_task_boundary(sales_csv, monkeypatch, enabled):
    key = "FABRIC_RLM_TEST_FAKE_SECRET_KEY"
    monkeypatch.setenv(key, "synthetic-not-a-credential")
    lm = ScriptedLM([_code(
        READ_TOTAL + "import os\n"
        f"secret_present = {key!r} in os.environ\n"
        "SUBMIT(total=total, secret_present=secret_present)"
    )])
    result = RLM.task(
        "Total included sales and report whether the test key exists, never its value.",
        inputs={"source": sales_csv}, outputs=["total", "secret_present"],
        lm=lm, max_turns=1, timeout=30,
        security=SecurityPolicy.default() if enabled else SecurityPolicy.disabled(),
    ).run()

    assert result.submitted, result.failure_reason
    assert result.payload == {"total": 50, "secret_present": not enabled}
    assert "synthetic-not-a-credential" not in str(result.payload)
    assert "synthetic-not-a-credential" not in result.turns[0].stdout


@pytest.mark.parametrize("block_network", [True, False])
def test_block_network_preserves_loopback_csv_access(sales_csv, block_network):
    csv_bytes = sales_csv.read_bytes()

    class CsvHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.sendall(csv_bytes)

    with socketserver.TCPServer(("127.0.0.1", 0), CsvHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            lm = ScriptedLM([_code("""import csv, os, socket
with socket.create_connection(tuple(address), timeout=5) as connection:
    with connection.makefile('r', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
total = sum(int(row['amount']) for row in rows if row['included'] == 'yes')
print('included total', total)
""" + f"guard_enabled = os.environ.get({ENV_FLAG!r}) == '1'\n"
                + "SUBMIT(total=total, guard_enabled=guard_enabled)")])
            result = RLM.task(
                "Total included sales from the local CSV server.",
                inputs={"address": list(server.server_address)},
                outputs=["total", "guard_enabled"], lm=lm, max_turns=1, timeout=30,
                security=SecurityPolicy.disabled(), block_network=block_network,
            ).run()
        finally:
            server.shutdown()
            thread.join(timeout=5)

    assert not thread.is_alive()
    assert result.submitted, result.failure_reason
    assert result.turns[0].error is None
    assert result.payload == {"total": 50, "guard_enabled": block_network}


@pytest.mark.parametrize("engine", ["default", "dspy"])
@pytest.mark.parametrize("limit", [256, 4096])
def test_max_submit_bytes_rejects_oversize_and_allows_csv_repair(sales_csv, engine, limit):
    first = READ_TOTAL + "SUBMIT(total=total, detail=str(rows) * 8)"
    repair = READ_TOTAL + "SUBMIT(total=total, detail='bounded')"
    lm = (_CountingScriptedLM([first, repair]) if engine == "dspy"
          else ScriptedLM([_code(first), _code(repair)]))
    # Omit engine entirely for the default-engine regression contract.
    options = {"engine": "dspy"} if engine == "dspy" else {}
    result = RLM.task(
        "Total included sales with bounded detail.", inputs={"source": str(sales_csv)},
        signature="source -> total: int, detail: str",
        outputs=["total", "detail"], lm=lm, max_turns=2, timeout=30,
        max_submit_bytes=limit, **options,
    ).run()

    assert result.submitted, result.failure_reason
    assert result.payload["total"] == 50
    calls = lm.calls if engine == "dspy" else len(lm.messages)
    if limit == 256:
        assert calls == 2
        assert result.payload["detail"] == "bounded"
        assert "exceeds max_submit_bytes=256" in str(result.turns[0])
    else:
        assert calls == 1
        assert len(result.payload["detail"].encode("utf-8")) > 256


@pytest.mark.parametrize("integrity", ["strict", "off"])
def test_analytical_integrity_checks_actual_emitted_numeric_literal(sales_csv, integrity):
    lm = ScriptedLM([
        _code(READ_TOTAL + "SUBMIT(total=999)"),
        _code("SUBMIT(total=total)"),
    ])
    result = RLM.task(
        "Total included sales.", inputs={"source": sales_csv}, outputs=["total"],
        lm=lm, max_turns=2, timeout=30, analytical_integrity=integrity,
    ).run()

    assert result.submitted, result.failure_reason
    assert result.turns[0].stdout.strip() == "included total 50"
    if integrity == "strict":
        assert result.payload == {"total": 50}
        assert len(lm.messages) == 2
        assert "999 was typed into SUBMIT" in lm.messages[1][-1]["content"]
        assert result.turns[1].turn_type == "verifier_repair"
        assert result.trajectory.metadata["verifier_repair_history"][0]["skill"] == "analytical_integrity"
        assert result.integrity_ok
    else:
        assert result.payload == {"total": 999}
        assert len(lm.messages) == 1
        assert "verifier_repair_history" not in result.trajectory.metadata
