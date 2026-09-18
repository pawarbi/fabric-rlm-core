"""Release claims backed by shipped docs, real workers, and offline model doubles.

Only nested default-engine calls use HTTP, through the existing loopback fixture.
These tests do not establish live Fabric authentication or provider compatibility.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import re
import socket
from unittest.mock import Mock

import dspy
import pytest

import fabric_rlm
from fabric_rlm import File, RLM, SemanticModel
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.serializers import freeze
from test_api_arguments_engines import _ScriptedLM, loopback_sub_lm  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    connect = socket.socket.connect

    def local_connect(sock, address):
        assert address[0] in {"127.0.0.1", "::1", "localhost"}, address
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)


def _python_blocks(text):
    text = re.sub(r"^> ?", "", text, flags=re.MULTILINE)
    blocks = re.findall(r"^```python\s*\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    assert blocks, "Expected Python examples in the document"
    return blocks


def _without_magics(source):
    # A cell magic owns the entire cell; line magics do not own later Python.
    if source.lstrip().startswith("%%"):
        return ""
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


@pytest.mark.parametrize("filename", [
    "README.md", "docs/usage-guide.md", "QUICKSTART.md", "docs/fabric-runtime-deps.md",
])
def test_release_python_examples_parse(filename):
    for index, block in enumerate(_python_blocks((ROOT / filename).read_text(encoding="utf-8"))):
        ast.parse(_without_magics(block), filename=f"{filename}:example {index}")


def test_fabric_installation_preserves_dependency_resolution_and_restart():
    text = (ROOT / "docs/fabric-runtime-deps.md").read_text(encoding="utf-8")
    blocks = _python_blocks(text)
    installs = [line.strip() for block in blocks for line in block.splitlines()
                if re.match(r"\s*%pip\s+install\b", line)]
    assert installs
    assert any('"fabric-rlm[analytics]"' in line for line in installs)
    assert any(".whl[analytics]" in line for line in installs)
    for line in installs:
        assert "--no-deps" not in line
        assert not re.search(r"\bdspy(?:-ai)?\b", line, re.IGNORECASE), line
    prose = " ".join(text.lower().split())
    assert "pyproject.toml" in prose
    assert re.search(r"restart.{0,80}after installation", prose)
    assert "before running the remaining" in prose
    assert "does not reload" in prose
    assert 'both `engine="default"` and `engine="dspy"` can analyze' in prose
    assert "alone does not require the dspy engine" in prose


def test_quickstart_default_sub_lm_example_is_a_serializable_configuration(monkeypatch):
    text = (ROOT / "QUICKSTART.md").read_text(encoding="utf-8")
    examples = []
    for block in _python_blocks(text):
        tree = ast.parse(_without_magics(block))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                if "sub_lm" in kwargs and isinstance(kwargs.get("engine"), ast.Constant):
                    if kwargs["engine"].value == "default":
                        assert isinstance(kwargs["sub_lm"], (ast.Dict, ast.Constant))
                        examples.append(block)
    assert len(examples) == 1
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-config-only-key")
    outer = _ScriptedLM([])
    namespace = {"RLM": RLM, "lm": outer}
    exec(compile(examples[0], "<QUICKSTART default sub_lm>", "exec"), namespace)
    rlm = namespace["rlm"]
    assert rlm.engine == "v6-custom"
    assert json.loads(json.dumps(rlm.sub_lm_spec)) == {
        "model": "openai/gpt-5-mini", "api_key": "synthetic-config-only-key",
    }
    assert not outer.calls, "Constructing the documented configuration must not call a provider"


def test_readme_warns_about_raw_feedback():
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").lower().split())
    assert "not automatically embedded" in text
    assert "not content-redacted" in text
    assert re.search(r"print raw.{0,100}reach.{0,50}model provider", text)
    assert not re.search(r"never (?:sees?|receives?|sends?).{0,30}raw", text)


def test_usage_guide_explains_parent_only_authentication():
    text = " ".join((ROOT / "docs/usage-guide.md").read_text(encoding="utf-8").lower().split())
    assert "parent process" in text
    assert "neither the token nor the `credential_provider` setting is serialized" in text
    assert "worker-side semantic-model queries still require working sempy automatic authentication" in text


def test_semantic_model_wire_and_worker_snapshot_drop_parent_auth(monkeypatch):
    check = Mock(side_effect=AssertionError("Serialization must not query SemPy"))
    token = Mock(side_effect=AssertionError("Serialization must not request a token"))
    monkeypatch.setattr(SemanticModel, "check", check)
    monkeypatch.setattr("fabric_rlm.semantic_model._NotebookUtilsPbiCredential.get_token", token)
    model = SemanticModel("Release Sales", workspace="Release Workspace",
                          credential_provider="notebookutils", validate=False)
    # Synthetic parent working state must not accidentally enter wire/snapshots.
    secret = "synthetic-parent-token-not-a-real-credential"
    object.__setattr__(model, "_catalog", {"token": secret})
    wire = encode_for_worker({"sales": model})
    assert wire == {"sales": {"__fabric_rlm_semantic_model__": {
        "dataset": "Release Sales", "workspace": "Release Workspace",
    }}}
    encoded = json.dumps(wire)
    assert "credential_provider" not in encoded
    assert secret not in encoded
    back = decode_from_worker_wire(json.loads(encoded))["sales"]
    assert isinstance(back, SemanticModel)
    assert back == model
    assert back.credential_provider is None
    assert back.validate is False
    snapshot = freeze(back)
    assert snapshot["credential_provider"] is None
    assert secret not in json.dumps(snapshot)
    assert secret not in json.dumps(freeze(model))
    check.assert_not_called()
    token.assert_not_called()


@pytest.mark.parametrize("helper", ["predict_sync", "await predict"])
def test_default_serialized_sub_lm_real_worker_roundtrip(helper, loopback_sub_lm):
    spec, requests, errors = loopback_sub_lm
    sales = "id,region,amount\nn1,north,19\nn2,north,23\ns1,south,11\n"
    outer = _ScriptedLM([
        "import csv, io, os\n"
        "rows = list(csv.DictReader(io.StringIO(sales)))\n"
        "subset = 'id,amount\\n' + ''.join(r['id'] + ',' + r['amount'] + '\\n' "
        "for r in rows if r['region'] == 'north')\n"
        f"nested = {helper}('subset: str -> subtotal: int', "
        "subset='BEGIN_SUBSET\\n' + subset + '\\nEND_SUBSET')\n"
        "total = nested.subtotal + sum(int(r['amount']) for r in rows if r['region'] == 'south')\n"
        "print(total)",
        "SUBMIT(total=total, worker_pid=os.getpid())",
    ])
    result = RLM.task(
        "Send north's CSV subset to the nested model and add south locally.",
        inputs={"sales": sales}, outputs={"total": int, "worker_pid": int},
        lm=outer, sub_lm=spec, engine="default", block_network=False,
        max_turns=3, timeout=90, recover_worker_timeouts=0,
    ).run()
    assert not errors, errors
    assert result.submitted, (result.failure_reason, [t.error for t in result.trajectory.turns])
    assert result.payload["total"] == 53
    assert result.payload["worker_pid"] != os.getpid()
    assert len(requests) == 1
    path, authorization, body = requests[0]
    assert path == "/v1/chat/completions"
    assert authorization == "Bearer fake-loopback-key"
    assert body["model"] == "csv-subtotal"
    prompt = body["messages"][-1]["content"]
    assert "BEGIN_SUBSET\nid,amount\nn1,19\nn2,23\n\nEND_SUBSET" in prompt
    assert "s1," not in prompt
    assert len(outer.calls) == len(result.trajectory.turns) == 2
    assert not outer.codes
    assert all(not turn.error for turn in result.trajectory.turns)


def test_dspy_live_sub_lm_object_runs_llm_query_in_parent():
    class SubtotalLM(dspy.LM):
        def __init__(self):
            super().__init__("openai/offline-subtotal", cache=False)
            self.calls = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            self.calls.append((prompt, os.getpid()))
            return [str(sum(int(row["amount"]) for row in csv.DictReader(io.StringIO(prompt))))]

    sub_lm = SubtotalLM()
    outer = _ScriptedLM([
        "import os\n"
        "total = int(llm_query(prompt=sales))\n"
        "print(total)",
        "SUBMIT(total=total, worker_pid=os.getpid())",
    ])
    sales = "id,amount\na,19\nb,23\n"
    result = RLM.task(
        "Use llm_query to total the supplied CSV.", inputs={"sales": sales},
        outputs={"total": int, "worker_pid": int}, engine="dspy",
        lm=outer, sub_lm=sub_lm, max_turns=3, timeout=30,
    ).run()
    assert result.submitted, (result.failure_reason, [t.error for t in result.trajectory.turns])
    assert result.payload["total"] == 42
    assert result.payload["worker_pid"] != os.getpid()
    assert sub_lm.calls == [(sales, os.getpid())]
    assert len(outer.calls) == len(result.trajectory.turns) == 2
    assert not outer.codes
    assert all(not turn.error for turn in result.trajectory.turns)


def test_printed_file_rows_reach_next_outer_prompt(tmp_path):
    class SnapshotLM(_ScriptedLM):
        def __call__(self, prompt=None, messages=None, **kwargs):
            # The default loop appends feedback to the same message list.
            return super().__call__(prompt=prompt, messages=deepcopy(messages), **kwargs)

    sentinel = "SYNTHETIC_PRIVATE_ROW_RELEASE_TEST"
    path = tmp_path / "sales.csv"
    path.write_text(f"id,amount\n{sentinel},19\npublic,23\n", encoding="utf-8")
    outer = SnapshotLM([
        "import csv, io, os\n"
        "raw = sales.read_text()\n"
        "rows = list(csv.DictReader(io.StringIO(raw)))\n"
        "print(raw)",
        "SUBMIT(row_count=len(rows), worker_pid=os.getpid())",
    ])
    result = RLM.task(
        "Read the CSV and count rows.", inputs={"sales": File(path)},
        outputs={"row_count": int, "worker_pid": int}, lm=outer,
        engine="default", block_network=True, max_turns=3, timeout=30,
    ).run()
    assert result.submitted, result.failure_reason
    assert result.payload["row_count"] == 2
    assert result.payload["worker_pid"] != os.getpid()
    assert len(outer.calls) == len(result.trajectory.turns) == 2
    assert sentinel not in json.dumps(outer.calls[0])
    assert f"{sentinel},19" in result.trajectory.turns[0].stdout
    assert f"{sentinel},19" in json.dumps(outer.calls[1])
    assert all(not turn.error for turn in result.trajectory.turns)


def test_verified_public_exports_match_implementation():
    from fabric_rlm import verify

    for name in ("VerifiedResult", "answers_agree", "verified_task"):
        assert fabric_rlm.__all__.count(name) == 1
        assert getattr(fabric_rlm, name) is getattr(verify, name)
    assert len(fabric_rlm.__all__) == len(set(fabric_rlm.__all__))
    assert all(hasattr(fabric_rlm, name) for name in fabric_rlm.__all__)


def test_api_tour_notebook_is_valid_json_with_parseable_python_cells():
    path = ROOT / "examples/notebooks/rlm_api_tour.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["metadata"], dict)
    parsed = 0
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert isinstance(cell["metadata"], dict)
        source = cell["source"]
        assert isinstance(source, (str, list))
        source = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] != "code":
            continue
        assert isinstance(cell["outputs"], list)
        python = _without_magics(source)
        if python.strip():
            ast.parse(python, filename=f"{path.name}:cell {index}")
            parsed += 1
    assert parsed > 0
