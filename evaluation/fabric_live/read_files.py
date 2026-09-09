"""Read a small text file out of the lakehouse Files area (diagnostics)."""
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

WS = "sandeep_ws"
LH = "da_agent_tests.Lakehouse"
BASE = f"https://onelake.dfs.fabric.microsoft.com/{WS}/{LH}"

tok = subprocess.run(
    ["az.cmd" if sys.platform == "win32" else "az", "account", "get-access-token",
     "--resource", "https://storage.azure.com", "--query", "accessToken", "-o", "tsv"],
    capture_output=True, text=True, check=True).stdout.strip()

args = sys.argv[1:]
save_to = None
if "--out" in args:
    i = args.index("--out")
    save_to = args[i + 1]
    del args[i:i + 2]

for rel in args or ["Files/rlm_excel_eval/progress.log"]:
    url = f"{BASE}/{rel}"
    try:
        req = Request(url, headers={"Authorization": f"Bearer {tok}",
                                    "x-ms-version": "2023-08-03"}, method="GET")
        with urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8", "replace")
        if save_to:
            open(save_to, "w", encoding="utf-8").write(body)
            print(f"===== {rel} ===== -> {save_to} ({len(body)} chars)")
        else:
            print(f"===== {rel} =====")
            print(body)
    except HTTPError as e:
        print(f"===== {rel} ===== HTTP {e.code} ({'not found' if e.code == 404 else e.reason})")
