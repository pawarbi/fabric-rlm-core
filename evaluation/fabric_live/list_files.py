"""List a directory in the lakehouse Files area."""
import json
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

WS = "sandeep_ws"
LH = "da_agent_tests.Lakehouse"

tok = subprocess.run(
    ["az.cmd" if sys.platform == "win32" else "az", "account", "get-access-token",
     "--resource", "https://storage.azure.com", "--query", "accessToken", "-o", "tsv"],
    capture_output=True, text=True, check=True).stdout.strip()

directory = sys.argv[1] if len(sys.argv) > 1 else "Files/rlm_excel_eval"
url = (f"https://onelake.dfs.fabric.microsoft.com/{WS}?recursive=true"
       f"&resource=filesystem&directory={LH}/{directory}")
try:
    req = Request(url, headers={"Authorization": f"Bearer {tok}",
                                "x-ms-version": "2023-08-03"}, method="GET")
    with urlopen(req, timeout=120) as r:
        paths = json.loads(r.read().decode())["paths"]
    for p in paths:
        kind = "dir " if p.get("isDirectory") == "true" else "file"
        size = p.get("contentLength", "0")
        print(f"{kind}  {int(size):>10}  {p['name']}")
    if not paths:
        print("(empty)")
except HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
