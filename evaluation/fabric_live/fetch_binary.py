"""Download a OneLake file as bytes.

read_files.py decodes to text, which silently corrupts binary payloads: an
.xlsx came back 21321 "chars" against 22415 bytes on the lakehouse and openpyxl
rejected it with "Bad offset for central directory". Workbooks are the graded
artifact here, so they need a byte-exact path.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

WS = "sandeep_ws"
LH = "da_agent_tests.Lakehouse"


def token() -> str:
    return subprocess.run(
        ["az.cmd" if sys.platform == "win32" else "az", "account", "get-access-token",
         "--resource", "https://storage.azure.com", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    remote = sys.argv[1]
    out = Path(sys.argv[2])
    url = f"https://onelake.dfs.fabric.microsoft.com/{WS}/{LH}/{remote.lstrip('/')}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req) as response:
        payload = response.read()
    out.write_bytes(payload)
    print(f"{remote} -> {out} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
