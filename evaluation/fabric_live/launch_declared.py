"""Launch an arm-D (declared=) run. Keeps the key off the command line I type."""
import json
import os
import subprocess
import sys
from pathlib import Path

WS = "82ad2591-974a-4ad4-ace6-e24879274a4b"
NB_SMOKE = "0a46cb70-e22a-4867-9365-d0ea89ffa645"
NB_FULL = "2935d52f-4574-443d-a8d0-552936a7445a"

smoke = "--smoke" in sys.argv
run_tag = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else (
    "glm-declared-smoke-2" if smoke else "glm-declared-1")

declared = Path(__file__).with_name("declared_dbo.json")
if not declared.exists():
    declared = Path(r"C:\Users\sandeeppawar\.copilot\session-state"
                    r"\4fc1d5d0-25b1-4323-b111-f91ae7bc94cd\files\dbo_eval\declared_dbo.json")
payload = json.dumps(json.loads(declared.read_text(encoding="utf-8")), ensure_ascii=True)

key = os.environ.get("RLM_OR_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
if not key:
    sys.exit("no key in RLM_OR_KEY / OPENROUTER_API_KEY")

# ensure_ascii=True above guarantees no em-dash survives into the job parameter,
# which the console mangles when echoing.
assert payload.isascii(), "declared payload is not ASCII"

args = [
    sys.executable, str(Path(__file__).with_name("notebook.py")), "execute",
    "--workspace-id", WS,
    "--notebook-id", NB_SMOKE if smoke else NB_FULL,
    "--parameter", f"RUN_TAG={run_tag}",
    "--parameter", "MODEL=z-ai/glm-5.3-flash",
    "--parameter", "LEARN=True",
    "--parameter", "MAX_TURNS=22",
    "--parameter", "WORKBOOK_DUTY=True",
    "--parameter", "SKILLS=",
    "--parameter", f"DECLARED_JSON={payload}",
    "--parameter", f"OPENROUTER_API_KEY={key}",
]
print(f"run_tag={run_tag}  declared={len(payload)} chars ascii={payload.isascii()}")
sys.exit(subprocess.run(args).returncode)
