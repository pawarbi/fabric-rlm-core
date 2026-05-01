"""Build the 3 adaptive-engine tier notebooks (easy / medium / hard).

All three share an identical scaffold (wheel install -> import fabric_rlm ->
loop over cases -> write summary.json). Only the case selection and validator
differ. Generating from one template keeps them in lockstep so a fix to one
ladder boilerplate fixes all three.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

WHEEL_PATH = "/lakehouse/default/Files/fabric_rlm_longcot/wheels/fabric_rlm-0.1.10-py3-none-any.whl"
FIXTURE_ROOT = "/lakehouse/default/Files/fabric_rlm_adaptive_validation/fixtures"

LH_ID = "9d10bce5-1edc-4875-83c4-ac0a98a02775"
LH_NAME = "diagnostic"
WS_ID = "82ad2591-974a-4ad4-ace6-e24879274a4b"

CONFIGURE_CELL = (
    "%%configure -f\n"
    '{"vCores": 4, "defaultLakehouse": '
    '{"name": "' + LH_NAME + '", "id": "' + LH_ID + '", "workspaceId": "' + WS_ID + '"}}'
)

SETUP_CELL = '''import sys, json, time, traceback, uuid, platform as _platform
from pathlib import Path

TIER = "{tier}"
RUN_ID = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
FILES_ROOT = Path('/lakehouse/default/Files')
RUN_ROOT = FILES_ROOT / 'fabric_rlm_adaptive_validation' / TIER / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

summary = {{
    'tier': TIER,
    'run_id': RUN_ID,
    'started_at': time.time(),
    'python': _platform.python_version(),
    'stages': [],
    'cases': [],
    'passed': False,
    'error': None,
}}
SUMMARY_PATH = RUN_ROOT / 'summary.json'

def write_summary():
    summary['updated_at'] = time.time()
    summary['elapsed_seconds'] = summary['updated_at'] - summary['started_at']
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')

def stage(name, **fields):
    summary['stages'].append({{'stage': name, 't': time.time(), **fields}})
    print(f'[stage] {{name}}', fields if fields else '')
    write_summary()

stage('setup', run_root=str(RUN_ROOT))
'''

INSTALL_CELL = '''WHEEL_PATH = "''' + WHEEL_PATH + '''"
stage('wheel_check', exists=Path(WHEEL_PATH).exists(),
      size=Path(WHEEL_PATH).stat().st_size if Path(WHEEL_PATH).exists() else 0)
import subprocess
out = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                       '--force-reinstall', '--no-deps', WHEEL_PATH],
                      capture_output=True, text=True)
stage('pip_wheel', rc=out.returncode, stderr_tail=out.stderr[-400:])
if out.returncode != 0:
    summary['error'] = 'wheel install failed'; write_summary()
    raise SystemExit('wheel install failed')

try:
    import dspy
    stage('dspy_present', version=getattr(dspy,'__version__','?'))
except ImportError:
    out2 = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                            'dspy>=3.1.2'], capture_output=True, text=True)
    stage('pip_dspy', rc=out2.returncode, stderr_tail=out2.stderr[-400:])
    if out2.returncode != 0:
        summary['error'] = 'dspy install failed'; write_summary()
        raise SystemExit('dspy install failed')
    import dspy
    stage('dspy_installed', version=getattr(dspy,'__version__','?'))

for mod in [m for m in list(sys.modules) if m == 'fabric_rlm' or m.startswith('fabric_rlm.')]:
    sys.modules.pop(mod, None)
import fabric_rlm
stage('imported', version=getattr(fabric_rlm, '__version__', '?'))
'''

LOAD_CASES_TPL = '''FIXTURE_ROOT = "''' + FIXTURE_ROOT + '''"

# tier-specific case loader
{loader}

stage('cases_loaded', n=len(CASES))
'''

LOADERS = {
    "easy": '''import json as _json
CASES = []
with open(FIXTURE_ROOT + "/easy_cases.jsonl") as fh:
    for line in fh:
        rec = _json.loads(line)
        if rec.get("validator") == "exact_match" and rec.get("template") == "math":
            CASES.append({"id": rec["id"], "question": rec["question"], "answer": str(rec["answer"])})
CASES = CASES[:5]
''',
    "medium": '''import json as _json
CASES = []
with open(FIXTURE_ROOT + "/longcot_cs_hard_pilot20.jsonl") as fh:
    rows = [_json.loads(line) for line in fh]
mcm = [r for r in rows if r.get("template") == "MCM"][:3]
for r in mcm:
    CASES.append({
        "id": "med-mcm-" + str(r["question_id"]),
        "question": r["prompt"],
        "answer": r["answer"],
        "template": "MCM",
    })
''',
    "hard": '''import json as _json
CASES = []
with open(FIXTURE_ROOT + "/longcot_cs_hard_pilot20.jsonl") as fh:
    rows = [_json.loads(line) for line in fh]
picked_templates = ["MFMC", "Backprop", "DistMem", "VLIW"]
for tmpl in picked_templates:
    matches = [r for r in rows if r.get("template") == tmpl]
    if matches:
        r = matches[0]
        CASES.append({
            "id": "hard-" + tmpl.lower() + "-" + str(r["question_id"]),
            "question": r["prompt"],
            "answer": r["answer"],
            "template": tmpl,
        })
''',
}

VALIDATE_CELL = '''import sys
# longcot_adapter.py is uploaded alongside the fixtures.
sys.path.insert(0, "/lakehouse/default/Files/fabric_rlm_adaptive_validation/fixtures")
try:
    from longcot_adapter import verify_cs_response
    HAS_LONGCOT = True
except Exception as _e:
    HAS_LONGCOT = False
    print("longcot_adapter import failed:", _e)

def normalize(s):
    return "".join((s or "").lower().split())

def make_validator(case):
    """Pick the right validator for the case.

    LongCoT-style cases (have a 'template' field for an MFMC/Backprop/etc.
    template) get the deterministic ``verify_cs_response`` from
    ``longcot_adapter`` -- substring-matching a 5-10KB JSON expected answer
    against a free-form model response is too strict to be informative.

    Easy/short-answer cases keep the simple normalized-substring check.
    """
    expected_answer = case.get("answer")
    if expected_answer is None:
        expected_answer = ""
    elif not isinstance(expected_answer, str):
        try:
            import json as _json
            expected_answer = _json.dumps(expected_answer, sort_keys=True)
        except Exception:
            expected_answer = str(expected_answer)
    template = case.get("template")
    if template and HAS_LONGCOT:
        def validator(result):
            if not result.submitted or not result.payload:
                return False
            ans = result.payload.get("answer") or ""
            try:
                correct, _wrong_fmt = verify_cs_response(template, expected_answer, ans)
                return bool(correct)
            except Exception:
                # Fall back to substring match if the verifier raises (e.g.
                # an unsupported template); better some signal than no signal.
                return normalize(expected_answer) in normalize(ans)
        return validator

    norm_expected = normalize(expected_answer)
    def validator(result):
        if not result.submitted or not result.payload:
            return False
        ans = result.payload.get("answer") or ""
        return norm_expected in normalize(ans)
    return validator
'''

RUN_CELL_TPL = '''from fabric_rlm import RLM, FabricLM

try:
    cheap = FabricLM('gpt-4.1-mini', temperature=0.0, cache=False)
    strong = FabricLM('gpt-5', reasoning_effort='medium', cache=False)
    stage('lms_built', cheap='gpt-4.1-mini', strong='gpt-5')
except Exception as exc:
    summary['error'] = 'FabricLM build failed: ' + repr(exc)
    summary['traceback'] = traceback.format_exc()
    write_summary()
    raise

passed_count = 0
for i, case in enumerate(CASES, 1):
    case_record = {{'id': case['id'], 'template': case.get('template')}}
    summary['cases'].append(case_record)
    write_summary()
    try:
        validator = make_validator(case)
        rlm = RLM(
            signature='question -> answer',
            lm=cheap,
            engine='adaptive',
            adaptive=dict(
                strong_lm=strong,
                validator=validator,
                max_attempts={max_attempts},
                parallel_rollouts={parallel_rollouts},
            ),
        )
        t0 = time.perf_counter()
        result = rlm.run({{'question': case['question']}})
        elapsed = time.perf_counter() - t0
        meta = (result.trajectory.metadata or {{}}).get('adaptive', {{}}) if result.trajectory else {{}}
        passed_now = bool(result.submitted and validator(result))
        if passed_now:
            passed_count += 1
        ans_val = (result.payload or {{}}).get('answer') if result.payload else None
        case_record.update({{
            'passed': passed_now,
            'submitted': result.submitted,
            'elapsed_seconds': elapsed,
            'winner_rung': meta.get('winner_rung'),
            'stop_reason': meta.get('stop_reason'),
            'attempts': [{{'rung': a.get('rung'), 'passed': a.get('passed')}}
                          for a in meta.get('attempts', [])],
            'expected_preview': str(case['answer'])[:200],
            'answer_preview': (str(ans_val)[:200] if ans_val is not None else None),
            'failure_reason': result.failure_reason,
            'total_prompt_tokens': result.total_prompt_tokens,
            'total_completion_tokens': result.total_completion_tokens,
        }})
        stage('case_' + str(i) + '_done', id=case['id'], passed=passed_now, rung=meta.get('winner_rung'))
    except Exception as exc:
        case_record.update({{
            'passed': False,
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        }})
        stage('case_' + str(i) + '_error', id=case['id'], error=repr(exc))

summary['passed_count'] = passed_count
summary['total_cases'] = len(CASES)
summary['passed'] = passed_count == len(CASES) and len(CASES) > 0
write_summary()
print('TIER=' + TIER + ' PASSED=' + str(passed_count) + '/' + str(len(CASES)))
'''


def make_cell(source: str, kind: str = "code") -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(tier: str, *, max_attempts: int, parallel_rollouts: int) -> dict:
    cells = [
        make_cell(CONFIGURE_CELL),
        make_cell("# fabric-rlm 0.1.10 adaptive - **" + tier + "** tier validation\n", "markdown"),
        make_cell(SETUP_CELL.format(tier=tier)),
        make_cell(INSTALL_CELL),
        make_cell(LOAD_CASES_TPL.format(loader=LOADERS[tier])),
        make_cell(VALIDATE_CELL),
        make_cell(RUN_CELL_TPL.format(max_attempts=max_attempts, parallel_rollouts=parallel_rollouts)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kernel_info": {"name": "jupyter", "jupyter_kernel_name": "python3.11"},
            "dependencies": {
                "lakehouse": {
                    "default_lakehouse": LH_ID,
                    "default_lakehouse_name": LH_NAME,
                    "default_lakehouse_workspace_id": WS_ID,
                }
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    configs = {
        "easy":   {"max_attempts": 2, "parallel_rollouts": 1},
        "medium": {"max_attempts": 4, "parallel_rollouts": 1},
        "hard":   {"max_attempts": 6, "parallel_rollouts": 2},
    }
    for tier, cfg in configs.items():
        nb = build_notebook(tier, **cfg)
        path = NB_DIR / ("adaptive_test_" + tier + ".ipynb")
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", path, "cells=", len(nb["cells"]))
