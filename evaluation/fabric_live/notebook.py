"""Minimal Fabric notebook REST client: ipynb -> Fabric source, create, execute.

The skill documents tools/notebook.py but only its docs are installed here, so
this implements the same contract directly.

Secrets are passed as job parameters at execution time and never written into
the notebook definition that persists in the workspace.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API = "https://api.fabric.microsoft.com/v1"
SKILL_HEADER = {"x-ms-fabric-skill": "notebook-authoring-cli"}
AZ = "az.cmd" if sys.platform == "win32" else "az"


def token() -> str:
    out = subprocess.run(
        [AZ, "account", "get-access-token", "--resource",
         "https://api.fabric.microsoft.com", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def call(method: str, url: str, body: dict | None = None, tok: str | None = None):
    tok = tok or token()
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    headers.update(SKILL_HEADER)
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=300) as r:
            raw = r.read().decode("utf-8", "replace")
            loc = r.headers.get("Location")
            status = r.status
    except HTTPError as e:
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: "
                           f"{e.read().decode('utf-8','replace')[:500]}") from None
    payload = None
    if raw.strip() not in ("", "null", '"null"'):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
    return status, payload, loc


def ipynb_to_fabric(nb: dict) -> str:
    meta = nb.get("metadata", {})
    kernel = {"name": "jupyter", "jupyter_kernel_name": "python3.12"}
    nb_meta = {"kernel_info": kernel}
    if "dependencies" in meta:
        deps = dict(meta["dependencies"])
        lh = deps.get("lakehouse", {})
        if lh.get("default_lakehouse"):
            lh = dict(lh)
            lh["known_lakehouses"] = [{"id": lh["default_lakehouse"]}]
            deps["lakehouse"] = lh
        nb_meta["dependencies"] = deps

    lines = ["# Fabric notebook source", "", "# METADATA ********************", ""]
    for line in json.dumps(nb_meta, indent=2).split("\n"):
        lines.append(f"# META {line}")
    lines.append("")

    cell_meta = {"language": "python", "language_group": "jupyter_python"}
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            lines += ["# MARKDOWN ********************", ""]
            for line in text.split("\n"):
                lines.append(f"# {line}" if line else "#")
            lines.append("")
            continue
        # A cell tagged "parameters" must use Fabric's dedicated marker, or the
        # job scheduler has nowhere to inject executionData parameters and the
        # notebook runs with the placeholder values still in place.
        tags = (cell.get("metadata") or {}).get("tags") or []
        marker = ("# PARAMETERS CELL ********************"
                  if "parameters" in tags else "# CELL ********************")
        lines += [marker, ""]
        for line in text.split("\n"):
            lines.append(line)
        lines += ["", "# METADATA ********************", ""]
        for line in json.dumps(cell_meta, indent=2).split("\n"):
            lines.append(f"# META {line}")
        lines.append("")
    return "\n".join(lines)


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def platform_file(name: str) -> str:
    return json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/"
                   "gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": "Notebook", "displayName": name, "description": ""},
        "config": {"version": "2.0",
                   "logicalId": "00000000-0000-0000-0000-000000000000"},
    }, indent=2)


def wait_lro(location: str, tok: str, timeout: int = 600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        status, payload, _ = call("GET", location, tok=tok)
        state = (payload or {}).get("status") if isinstance(payload, dict) else None
        if state in ("Succeeded", "Completed"):
            return payload
        if state in ("Failed", "Deduped", "Cancelled"):
            raise RuntimeError(f"LRO {state}: {json.dumps(payload)[:600]}")
    raise TimeoutError("LRO timed out")


def cmd_create(a):
    tok = token()
    nb = json.loads(Path(a.from_file).read_text(encoding="utf-8"))
    src = ipynb_to_fabric(nb)
    Path(a.from_file).with_suffix(".fabric.py").write_text(src, encoding="utf-8")
    body = {
        "displayName": a.name,
        "definition": {
            "format": "FabricGitSource",
            "parts": [
                {"path": "notebook-content.py", "payload": b64(src),
                 "payloadType": "InlineBase64"},
                {"path": ".platform", "payload": b64(platform_file(a.name)),
                 "payloadType": "InlineBase64"},
            ],
        },
    }
    status, payload, loc = call(
        "POST", f"{API}/workspaces/{a.workspace_id}/notebooks", body, tok)
    if status == 202 and loc:
        payload = wait_lro(loc, tok)
        status2, payload, _ = call("GET", f"{loc}/result", tok=tok)
    print(json.dumps({"status": status,
                      "id": (payload or {}).get("id"),
                      "displayName": (payload or {}).get("displayName")}, indent=2))


def cmd_update(a):
    tok = token()
    nb = json.loads(Path(a.from_file).read_text(encoding="utf-8"))
    src = ipynb_to_fabric(nb)
    Path(a.from_file).with_suffix(".fabric.py").write_text(src, encoding="utf-8")
    body = {
        "definition": {
            "format": "FabricGitSource",
            "parts": [
                {"path": "notebook-content.py", "payload": b64(src),
                 "payloadType": "InlineBase64"},
                {"path": ".platform", "payload": b64(platform_file(a.name)),
                 "payloadType": "InlineBase64"},
            ],
        },
    }
    status, payload, loc = call(
        "POST",
        f"{API}/workspaces/{a.workspace_id}/notebooks/{a.notebook_id}/updateDefinition",
        body, tok)
    if status == 202 and loc:
        wait_lro(loc, tok)
    print(f"updated {a.notebook_id}: HTTP {status}")


def cmd_list(a):
    _, payload, _ = call("GET", f"{API}/workspaces/{a.workspace_id}/notebooks")
    for it in (payload or {}).get("value", []):
        print(f'{it["id"]}  {it["displayName"]}')


def cmd_execute(a):
    tok = token()
    params = {}
    for pair in a.parameter or []:
        k, _, v = pair.partition("=")
        # Fabric parameters are typed. Sending everything as "string" means a
        # boolean False arrives in the notebook as the string "False", which is
        # truthy -- silently inverting the flag. Infer the type instead.
        if v in ("True", "False"):
            params[k] = {"value": v == "True", "type": "bool"}
        elif re.fullmatch(r"-?\d+", v):
            params[k] = {"value": int(v), "type": "int"}
        elif re.fullmatch(r"-?\d*\.\d+", v):
            params[k] = {"value": float(v), "type": "float"}
        else:
            params[k] = {"value": v, "type": "string"}
    print("parameters: " + ", ".join(
        f"{k}={'<redacted>' if 'KEY' in k.upper() else p['value']}:{p['type']}"
        for k, p in params.items()))
    body = {"executionData": {"parameters": params}} if params else None
    url = (f"{API}/workspaces/{a.workspace_id}/items/{a.notebook_id}"
           f"/jobs/instances?jobType=RunNotebook")
    status, payload, loc = call("POST", url, body or {}, tok)
    print(f"submitted: HTTP {status}")
    if not a.wait:
        print(loc or "")
        return
    if not loc:
        print("no Location header; cannot poll")
        return
    deadline = time.time() + a.timeout
    last = None
    while time.time() < deadline:
        time.sleep(10)
        _, p, _ = call("GET", loc, tok=tok)
        state = (p or {}).get("status")
        if state != last:
            print(f"  {int(time.time()%100000)}  status={state}")
            last = state
        if state in ("Completed", "Succeeded"):
            print(json.dumps(p, indent=2)[:1200]); return
        if state in ("Failed", "Cancelled", "Deduped"):
            print(json.dumps(p, indent=2)[:3000]); sys.exit(1)
    print("timed out waiting; job may still be running:", loc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("create", "list", "execute", "update"):
        s = sub.add_parser(nm)
        s.add_argument("--workspace-id", required=True)
        if nm == "create":
            s.add_argument("--name", required=True)
            s.add_argument("--from-file", required=True)
        if nm == "update":
            s.add_argument("--notebook-id", required=True)
            s.add_argument("--from-file", required=True)
            s.add_argument("--name", default="notebook")
        if nm == "execute":
            s.add_argument("--notebook-id", required=True)
            s.add_argument("--parameter", action="append")
            s.add_argument("--wait", action="store_true")
            s.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    {"create": cmd_create, "list": cmd_list, "execute": cmd_execute,
     "update": cmd_update}[a.cmd](a)
