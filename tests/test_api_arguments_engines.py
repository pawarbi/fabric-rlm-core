"""Data-backed argument coverage with real engines, workers, and adaptive policy.

Only model responses are scripted. No runner, interpreter, predictor, backend
resolver, or tool dispatch is replaced. The nested model uses loopback HTTP.

Argument map:
* tools, engine='dspy'/'auto', signature supplied/omitted: host_tool_csv_total.
* sub_lm dict (openai/model, api_base, fake api_key), block_network=False:
  worker_sub_lm_roundtrip (real worker predict and DSPy/OpenAI HTTP client).
* engine='adaptive', inner_engine='default'/'dspy', adaptive policy/validator/
  max_attempts/max_total_turns/max_parallel/max_wall_seconds/on_attempt:
  adaptive_csv_escalation (including the attempt-budget stop).
* halve_max_iter_on_retry: existing test_halve_max_iter_param.py tests
  test_halving_default_behavior, test_no_halving_when_disabled, and
  test_no_halving_recovers_when_default_would_starve execute real DSPy; their
  constructor spy delegates to the original constructor. Not duplicated here.

This does not cover live model quality, remote providers, DSPy's host-side
llm_query sub-LM path, adaptive parallel rollouts, or every budget boundary.
"""

from __future__ import annotations

import csv
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import dspy
import pytest

from fabric_rlm import RLM
from fabric_rlm.experimental.adaptive_policy import LadderPolicy, ValidationVerdict


class _ScriptedLM(dspy.LM):
    def __init__(self, codes):
        super().__init__(model="openai/scripted-offline", cache=False)
        self.codes = list(codes)
        self.calls = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        assert self.codes, f"Unexpected LM call: scripted responses exhausted; messages={messages!r}"
        code = self.codes.pop(0)
        return [
            "[[ ## reasoning ## ]]\nCompute from the supplied CSV.\n\n"
            f"[[ ## code ## ]]\n```python\n{code}\n```\n\n"
            "[[ ## completed ## ]]\n"
        ]

    def copy(self, **kwargs):
        # The real ladder clones LMs when changing reasoning effort. Preserve
        # the response stream and observations across those genuine clones.
        clone = super().copy(**kwargs)
        clone.codes = self.codes
        clone.calls = self.calls
        return clone


@pytest.fixture
def sales_csv(request):
    amount = getattr(request, "param", 19)
    return (
        "id,region,status,amount\n"
        f"n1,north,active,{amount}\n"
        "n2,north,active,23\n"
        "s1,south,active,11\n"
        "void,north,cancelled,900\n"
    )


@pytest.mark.parametrize("sales_csv", [19, 47], indirect=True)
@pytest.mark.parametrize("engine", ["dspy", "auto"])
@pytest.mark.parametrize("explicit_signature", [False, True])
def test_host_tool_csv_total(sales_csv, engine, explicit_signature):
    host_calls = []

    def regional_total(region: str) -> int:
        """Return the active sales total for a region from the host CSV."""
        total = sum(
            int(row["amount"])
            for row in csv.DictReader(io.StringIO(sales_csv))
            if row["region"] == region and row["status"] == "active"
        )
        host_calls.append((region, total, os.getpid()))
        return total

    lm = _ScriptedLM([
        "import csv, io\n"
        "rows = list(csv.DictReader(io.StringIO(sales)))\n"
        "north = int(regional_total(region='north'))\n"
        "south = sum(int(r['amount']) for r in rows "
        "if r['region'] == 'south' and r['status'] == 'active')\n"
        "total = north + south\n"
        "print(total)",
        "SUBMIT(total=total)",
    ])
    kwargs = {"signature": "question: str, sales: str -> total: int"} if explicit_signature else {}
    rlm = RLM.from_task(
        "Sum active sales using the host regional_total tool for north.",
        inputs={"question": "What is the active total?", "sales": sales_csv},
        outputs={"total": int},
        lm=lm,
        tools=[regional_total],
        engine=engine,
        max_turns=3,
        timeout=20,
        **kwargs,
    )
    result = rlm.run()

    rows = list(csv.DictReader(io.StringIO(sales_csv)))
    expected_north = int(rows[0]["amount"]) + int(rows[1]["amount"])
    assert host_calls == [("north", expected_north, os.getpid())]
    assert rlm.engine == "v7-dspy"
    assert (rlm.signature is not None) == explicit_signature
    assert result.submitted, result.failure_reason
    assert result.payload == {"total": expected_north + 11}
    assert result.trajectory.metadata["engine"] == "v7-dspy"
    assert len(lm.calls) == len(result.trajectory.turns) == 2
    assert not lm.codes
    assert all(not turn.error for turn in result.trajectory.turns)
    assert str(expected_north + 11) in result.trajectory.turns[0].stdout


@pytest.fixture
def loopback_sub_lm(monkeypatch):
    # Prevent provider metadata downloads in the child and avoid inherited
    # proxy settings routing even localhost requests outside this machine.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    requests = []
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                assert self.path == "/v1/chat/completions"
                assert body["model"] == "csv-subtotal"
                user_prompt = body["messages"][-1]["content"]
                subset = user_prompt.split("BEGIN_SUBSET\n", 1)[1].split("\nEND_SUBSET", 1)[0]
                subtotal = sum(int(row["amount"]) for row in csv.DictReader(io.StringIO(subset)))
                content = f"[[ ## subtotal ## ]]\n{subtotal}\n\n[[ ## completed ## ]]\n"
                response = {
                    "id": "chatcmpl-loopback",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                }
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as exc:
                errors.append(repr(exc))
                self.send_error(400, "Invalid local test request")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    spec = {
        "model": "openai/csv-subtotal",
        "api_base": f"http://127.0.0.1:{server.server_port}/v1",
        "api_key": "fake-loopback-key",
        "cache": False,
        "num_retries": 0,
        "timeout": 5,
        "max_tokens": 128,
    }
    try:
        yield spec, requests, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("sales_csv", [19, 47], indirect=True)
def test_worker_sub_lm_roundtrip(sales_csv, loopback_sub_lm):
    spec, requests, errors = loopback_sub_lm
    lm = _ScriptedLM([
        "import csv, io\n"
        "rows = list(csv.DictReader(io.StringIO(sales)))\n"
        "selected = [r for r in rows if r['region'] == 'north' and r['status'] == 'active']\n"
        "subset = 'id,amount\\n' + ''.join(r['id'] + ',' + r['amount'] + '\\n' for r in selected)\n"
        "nested = await predict('subset: str -> subtotal: int', "
        "subset='BEGIN_SUBSET\\n' + subset + '\\nEND_SUBSET')\n"
        "south = sum(int(r['amount']) for r in rows if r['region'] == 'south')\n"
        "total = nested.subtotal + south\n"
        "print(total)",
        "SUBMIT(total=total)",
    ])
    rlm = RLM.from_task(
        "Send only active north sales to the sub-LM; add south locally.",
        inputs={"sales": sales_csv},
        outputs={"total": int},
        lm=lm,
        sub_lm=spec,
        engine="default",
        block_network=False,
        max_turns=3,
        # DSPy/LiteLLM imports in a fresh Windows worker can take >45s;
        # the actual HTTP request still has its own five-second timeout.
        timeout=90,
        recover_worker_timeouts=0,
    )
    result = rlm.run()

    assert not errors, errors
    assert result.submitted, (result.failure_reason, [t.error for t in result.trajectory.turns], requests)
    assert len(requests) == 1, requests
    path, authorization, body = requests[0]
    assert path == "/v1/chat/completions"
    assert authorization == "Bearer fake-loopback-key"
    user_prompt = body["messages"][-1]["content"]
    first_amount = int(next(csv.DictReader(io.StringIO(sales_csv)))["amount"])
    assert f"BEGIN_SUBSET\nid,amount\nn1,{first_amount}\nn2,23\n\nEND_SUBSET" in user_prompt
    assert all(excluded not in user_prompt for excluded in ("s1,", "void,", "900"))
    assert result.payload == {"total": first_amount + 23 + 11}
    assert len(lm.calls) == len(result.trajectory.turns) == 2
    assert not lm.codes
    assert all(not turn.error for turn in result.trajectory.turns)
    assert str(first_amount + 23 + 11) in result.trajectory.turns[0].stdout


@pytest.mark.experimental
@pytest.mark.parametrize("inner_engine", ["default", "dspy"])
@pytest.mark.parametrize("max_attempts", [1, 5])
def test_adaptive_csv_escalation(sales_csv, inner_engine, max_attempts):
    read_rows = "import csv, io\nrows = list(csv.DictReader(io.StringIO(sales)))\n"
    weak_code = read_rows + "total = sum(int(r['amount']) for r in rows)\nprint(total)\nSUBMIT(total=total)"
    strong_code = (
        read_rows
        + "total = sum(int(r['amount']) for r in rows if r['status'] == 'active')\n"
        "print(total)\nSUBMIT(total=total)"
    )
    weak = _ScriptedLM([weak_code] * 4)
    strong = _ScriptedLM([strong_code])
    expected = sum(int(r["amount"]) for r in csv.DictReader(io.StringIO(sales_csv)) if r["status"] == "active")
    validated = []
    attempts = []

    def validate(result):
        validated.append(result)
        passed = result.submitted and result.payload == {"total": expected}
        return ValidationVerdict(
            passed=passed,
            feedback=None if passed else "Exclude cancelled CSV rows from the active total.",
        )

    # Use the shipped policy and a resolvable LM object, not a factory masquerading
    # as an LM spec. Rungs 0..3 stay weak; rung 4 selects the strong object.
    policy = LadderPolicy(base_max_turns=2, strong_lm_spec=strong, parallel_rollouts=1)
    with pytest.warns(UserWarning, match="experimental"):
        rlm = RLM.from_task(
            "Compute the active sales total from CSV.",
            inputs={"question": "Sum active sales only.", "sales": sales_csv},
            outputs={"total": int},
            lm=weak,
            engine="adaptive",
            inner_engine=inner_engine,
            max_turns=2,
            timeout=20,
            adaptive={
                "policy": policy,
                "validator": validate,
                "max_attempts": max_attempts,
                "max_total_turns": 10,
                "max_parallel": 1,
                "max_wall_seconds": 120,
                "on_attempt": attempts.append,
            },
        )
    if max_attempts == 1:
        with pytest.warns(UserWarning, match="cannot reach"):
            result = rlm.run()
    else:
        result = rlm.run()

    assert len(attempts) == len(validated) == max_attempts
    assert [a.rung for a in attempts] == list(range(max_attempts))
    assert [a.config.max_turns for a in attempts] == [2] + [4] * (max_attempts - 1)
    assert all(a.result is checked for a, checked in zip(attempts, validated))
    assert all(a.result.submitted and a.result.failure_reason is None for a in attempts)
    assert all(a.turns_used == 1 for a in attempts)
    assert all(not t.error for a in attempts for t in a.result.trajectory.turns)
    assert all(a.result.payload == {"total": expected + 900} for a in attempts[:4])
    assert len(weak.calls) == min(max_attempts, 4)
    assert len(strong.calls) == (1 if max_attempts == 5 else 0)
    if inner_engine == "dspy":
        assert all(a.result.trajectory.metadata["engine"] == "v7-dspy" for a in attempts)
    meta = result.trajectory.metadata["adaptive"]
    assert meta["attempts"] == [a.to_summary() for a in attempts]
    if max_attempts == 5:
        assert attempts[-1].config.lm_spec is strong
        assert [a.verdict.passed for a in attempts] == [False] * 4 + [True]
        assert result is attempts[-1].result
        assert result.payload == {"total": expected}
        assert meta["winner_rung"] == 4
        assert meta["stop_reason"] == "validator passed"
        assert "Exclude cancelled CSV rows" in attempts[-1].config.failure_feedback
        if inner_engine == "dspy":
            assert "Exclude cancelled CSV rows" in json.dumps(strong.calls)
        assert not strong.codes
    else:
        # A submitted but semantically wrong partial result is not a pass.
        assert result is attempts[0].result
        assert not attempts[0].verdict.passed
        assert meta["winner_rung"] == 0
        assert meta["stop_reason"] == "budget: max_attempts reached"
