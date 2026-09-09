"""Upload the frozen library and the questions-only file to the lakehouse Files area.

Deliberately uploads NO reference answers and NO grader code -- the agent must
not be able to read the answers it is being scored against. Grading happens
afterwards, locally, against ground_truth.json which never leaves this machine.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

WS = "sandeep_ws"
LH = "da_agent_tests.Lakehouse"
DEST = "Files/rlm_excel_eval"
BASE = f"https://onelake.dfs.fabric.microsoft.com/{WS}/{LH}"
API = "2023-08-03"

HERE = Path(__file__).resolve().parent
STAGE = HERE
GT = HERE.parent / "dbo_eval" / "ground_truth.json"


def token() -> str:
    out = subprocess.run(
        ["az.cmd" if sys.platform == "win32" else "az", "account", "get-access-token",
         "--resource", "https://storage.azure.com", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


TOK = token()


def send(url: str, method: str, data: bytes = b"", extra: dict | None = None) -> None:
    headers = {"Authorization": f"Bearer {TOK}", "x-ms-version": API,
               "Content-Length": str(len(data))}
    headers.update(extra or {})
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=300) as r:
            if r.status >= 300:
                raise RuntimeError(f"{method} {url} -> {r.status}")
    except HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {body}") from None


def upload(local: Path, remote_name: str) -> None:
    data = local.read_bytes()
    url = f"{BASE}/{DEST}/{remote_name}"
    send(f"{url}?resource=file", "PUT")
    step = 4 * 1024 * 1024
    for pos in range(0, len(data), step):
        chunk = data[pos:pos + step]
        send(f"{url}?action=append&position={pos}", "PATCH", chunk,
             {"Content-Type": "application/octet-stream"})
    send(f"{url}?action=flush&position={len(data)}", "PATCH")
    print(f"  uploaded {remote_name}  ({len(data)/1024:.1f} KB)")


# make the directory (idempotent)
try:
    send(f"{BASE}/{DEST}?resource=directory", "PUT")
    print(f"created {DEST}")
except RuntimeError as exc:
    print(f"directory: {exc} (continuing -- may already exist)")

# 1. the frozen library
upload(STAGE / "fabric_rlm_pr75.zip", "fabric_rlm_pr75.zip")

# 1b. an installable wheel. The zip carries only the package directory, with no
# pyproject.toml, so side-loading it onto sys.path silently skipped the declared
# dependency pins (notably dspy>=3.2.1,<3.3). Installing the wheel makes pip
# resolve and enforce them, matching how a real user gets the library.
WHEEL = HERE.parent / "fabric-rlm-core-pr75" / "dist_eval" / "fabric_rlm-0.6.0-py3-none-any.whl"
upload(WHEEL, WHEEL.name)

# 2. questions ONLY -- strip reference and hazard
gt = json.loads(GT.read_text(encoding="utf-8"))
questions = [{"id": r["id"], "question": r["question"]} for r in gt if r["id"] != "q23"]
qfile = STAGE / "questions.json"
qfile.write_text(json.dumps(questions, indent=2), encoding="utf-8")
print(f"  prepared {len(questions)} questions (q23 withdrawn; no references included)")
upload(qfile, "questions.json")

assert not any("reference" in q or "hazard" in q for q in questions), \
    "reference answers must never be uploaded"
print("\nverified: no reference answers or grader code uploaded")
print(f"destination: abfss://{WS}@onelake.dfs.fabric.microsoft.com/{LH}/{DEST}")
