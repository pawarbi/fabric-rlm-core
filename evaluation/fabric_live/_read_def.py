import base64, json, sys, time
sys.path.insert(0, ".")
from notebook import call, token, API
tok = token()
ws  = "82ad2591-974a-4ad4-ace6-e24879274a4b"
nb  = "0a46cb70-e22a-4867-9365-d0ea89ffa645"
st, p, loc = call("POST", f"{API}/workspaces/{ws}/notebooks/{nb}/getDefinition", {}, tok)
if st == 202 and loc:
    for _ in range(30):
        time.sleep(4)
        s2, p2, _ = call("GET", loc, tok=tok)
        if (p2 or {}).get("status") in ("Succeeded","Completed"): break
    s3, p, _ = call("GET", f"{loc}/result", tok=tok)
for part in (p or {}).get("definition", {}).get("parts", []):
    if part["path"].endswith("notebook-content.py"):
        txt = base64.b64decode(part["payload"]).decode()
        print("\n".join(txt.split("\n")[:24]))
