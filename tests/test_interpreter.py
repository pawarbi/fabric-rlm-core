from pathlib import Path
import threading

import pytest

from fabric_rlm import (
    File,
    FileDestination,
    Interpreter,
    LakehouseSource,
    WorkerTimeout,
)


def test_interpreter_persists_state() -> None:
    with Interpreter(timeout=5) as interp:
        first = interp.execute("x = 10")
        second = interp.execute("x += 5\nprint(x)")

    assert first.ok
    assert second.ok
    assert second.stdout.strip() == "15"
    assert second.state["x"] == 15


def test_set_inputs_has_a_control_plane_timeout_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interp = Interpreter(timeout=0.01)
    monkeypatch.setattr(interp, "_send", lambda _message: None)
    response = threading.Timer(
        0.05,
        lambda: interp._stdout_queue.put('{"ok": true}'),
    )
    response.start()
    try:
        assert interp.set_inputs({}) == {"ok": True}
    finally:
        response.join()


def test_interpreter_survives_syntax_and_runtime_errors() -> None:
    with Interpreter(timeout=5) as interp:
        syntax = interp.execute("if True print('bad')")
        runtime = interp.execute("1 / 0")
        recovery = interp.execute("answer = 42")

    assert not syntax.ok
    assert "SyntaxError" in syntax.error
    assert not runtime.ok
    assert "ZeroDivisionError" in runtime.error
    assert recovery.ok
    assert recovery.state["answer"] == 42


def test_interpreter_survives_system_exit() -> None:
    with Interpreter(timeout=5) as interp:
        system_exit = interp.execute("raise SystemExit('stop requested')")
        recovery = interp.execute("answer = 42")

    assert not system_exit.ok
    assert "SystemExit" in system_exit.error
    assert recovery.ok
    assert recovery.state["answer"] == 42


def test_submit_is_first_class_response() -> None:
    with Interpreter(timeout=5) as interp:
        result = interp.execute("SUBMIT(answer=42, status='done')")

    assert result.ok
    assert result.submitted
    assert result.submit_payload == {"answer": 42, "status": "done"}


def test_submit_preserves_large_string_and_collection_payloads() -> None:
    with Interpreter(timeout=5) as interp:
        result = interp.execute(
            "rows = [[i, 'value-' + str(i)] for i in range(500)]\n"
            "csv_text = 'header\\n' + ('x' * 10000)\n"
            "SUBMIT(prediction=csv_text, rows=rows)"
        )

    assert result.submit_payload is not None
    assert len(result.submit_payload["prediction"]) == 10_007
    assert len(result.submit_payload["rows"]) == 500
    assert result.submit_payload["rows"][-1] == [499, "value-499"]


def test_submit_rejects_payload_above_explicit_byte_limit() -> None:
    with Interpreter(timeout=5, max_submit_bytes=100) as interp:
        result = interp.execute("SUBMIT(answer='é' * 100)")

    assert not result.ok
    assert not result.submitted
    assert result.submit_payload is None
    assert "SUBMIT payload exceeds" in result.error
    assert "exceeds max_submit_bytes=100" in result.error


@pytest.mark.parametrize("limit", [0, -1])
def test_interpreter_rejects_nonpositive_submit_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="max_submit_bytes must be greater than zero"):
        Interpreter(max_submit_bytes=limit)


def test_top_level_await() -> None:
    with Interpreter(timeout=5) as interp:
        result = interp.execute("import asyncio\nawait asyncio.sleep(0)\nvalue = 123")

    assert result.ok
    assert result.state["value"] == 123


def test_reset_clears_user_state() -> None:
    with Interpreter(timeout=5) as interp:
        interp.execute("x = 1")
        reset = interp.reset()

    assert reset["ok"]
    assert "x" not in reset["state"]


def test_set_inputs_decodes_file(tmp_path: Path) -> None:
    input_file = File(tmp_path / "input.txt")
    input_file.write_text("fabric")

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"input_file": input_file})
        result = interp.execute("text = input_file.read_text()")

    assert result.ok
    assert result.state["text"] == "fabric"


def test_lakehouse_query_runs_in_parent_without_exposing_credentials(
    monkeypatch,
) -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.companies",
                "path": "abfss://workspace/lakehouse/Tables/dbo/companies",
            }
        ],
    )
    observed = {}

    def fake_query(bound_source, *, sql, sources, max_rows):
        observed.update(
            source=bound_source,
            sql=sql,
            sources=sources,
            max_rows=max_rows,
        )
        return {"columns": ["region"], "rows": [["North America"]], "truncated": False}

    monkeypatch.setattr(
        "fabric_rlm.interpreter.execute_lakehouse_query",
        fake_query,
    )

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"lakehouse": source})
        result = interp.execute(
            "data = lakehouse.query("
            "\"SELECT region FROM companies\", "
            "sources={\"companies\": \"dbo.companies\"})\n"
            "print(data['rows'][0][0])"
        )

    assert result.ok
    assert result.stdout.strip() == "North America"
    assert observed == {
        "source": source,
        "sql": "SELECT region FROM companies",
        "sources": {"companies": "dbo.companies"},
        "max_rows": 1000,
    }


def test_lakehouse_query_rejects_worker_catalog_tampering(monkeypatch) -> None:
    source = LakehouseSource(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "dbo.companies",
                "path": "abfss://workspace/lakehouse/Tables/dbo/companies",
            }
        ],
    )
    monkeypatch.setattr(
        "fabric_rlm.interpreter.execute_lakehouse_query",
        lambda *_args, **_kwargs: pytest.fail("tampered source must not execute"),
    )

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"lakehouse": source})
        result = interp.execute(
            "lakehouse.catalog[0]['path'] = 'abfss://other/private'\n"
            "lakehouse.query("
            "\"SELECT * FROM companies\", "
            "sources={\"companies\": \"dbo.companies\"})"
        )

    assert not result.ok
    assert "not bound to this worker" in (result.error or "")


def test_file_publish_runs_in_parent_for_bound_destination(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = FileDestination(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files"
    )
    observed = {}

    def fake_publish(bound_destination, *, local_path, relative_path, overwrite):
        observed.update(
            destination=bound_destination,
            local_path=local_path,
            relative_path=relative_path,
            overwrite=overwrite,
        )
        return {"path": "abfss://lakehouse/Files/report.xlsx", "name": "report.xlsx", "size": 8}

    monkeypatch.setattr("fabric_rlm.interpreter.publish_file", fake_publish)

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"destination": destination})
        result = interp.execute(
            "staged = destination.stage('report.xlsx')\n"
            "staged.write_bytes(b'workbook')\n"
            "published = destination.publish(staged)\n"
            "print(published['path'])"
        )

    assert result.ok
    assert result.stdout.strip() == "abfss://lakehouse/Files/report.xlsx"
    assert observed == {
        "destination": destination,
        "local_path": observed["local_path"],
        "relative_path": "report.xlsx",
        "overwrite": False,
    }
    assert Path(observed["local_path"]).parent == Path(destination.staging_root)


def test_file_publish_rejects_worker_destination_tampering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = FileDestination(
        "abfss://workspace@onelake.dfs.fabric.microsoft.com/lakehouse/Files"
    )
    monkeypatch.setattr(
        "fabric_rlm.interpreter.publish_file",
        lambda *_args, **_kwargs: pytest.fail("tampered destination must not publish"),
    )

    with Interpreter(timeout=5) as interp:
        interp.set_inputs({"destination": destination})
        result = interp.execute(
            "object.__setattr__(destination, 'root', 'abfss://other/Files')\n"
            "staged = destination.stage('report.xlsx')\n"
            "staged.write_bytes(b'workbook')\n"
            "destination.publish(staged)"
        )

    assert not result.ok
    assert "not bound to this worker" in (result.error or "")


def test_timeout_kills_worker() -> None:
    interp = Interpreter(timeout=0.2).start()
    try:
        with pytest.raises(WorkerTimeout):
            interp.execute("import time\ntime.sleep(5)")
        assert not interp.is_running
    finally:
        interp.shutdown()


def test_shutdown_closes_worker_pipes() -> None:
    interp = Interpreter(timeout=5).start()
    proc = interp.proc
    assert proc is not None

    interp.shutdown()

    assert interp.proc is None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed
