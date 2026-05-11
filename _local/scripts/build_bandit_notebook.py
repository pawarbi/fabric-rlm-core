"""Build the bandit-policy validation notebook for fabric-rlm 0.1.11.dev0.

The notebook runs the medium + hard cases multiple times, sharing a single
``BanditState`` across iterations so the bandit's per-template Beta
posteriors accumulate. Each iteration logs which rung the bandit picked,
whether the case passed, and the total attempts used. Comparing iteration
1 (uniform-prior cold start) to iteration 5 should show the bandit
learning to skip rung 0 on hopeless-for-cheap-LM templates.

Reuses the same scaffold pattern as ``build_tier_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

WHEEL_PATH = "/lakehouse/default/Files/fabric_rlm_longcot/wheels/fabric_rlm-0.1.11.dev0-py3-none-any.whl"
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

TIER = "bandit"
RUN_ID = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
FILES_ROOT = Path('/lakehouse/default/Files')
RUN_ROOT = FILES_ROOT / 'fabric_rlm_adaptive_validation' / TIER / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Bandit state lives at a stable path across notebook re-runs so priors
# accumulate over time. Each iteration within a single notebook run also
# shares this state in-memory before saving.
BANDIT_STATE_PATH = FILES_ROOT / 'fabric_rlm_adaptive_validation' / 'bandit' / 'state.json'
BANDIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

summary = {
    'tier': TIER,
    'run_id': RUN_ID,
    'started_at': time.time(),
    'python': _platform.python_version(),
    'stages': [],
    'iterations': [],
    'passed': False,
    'error': None,
}
SUMMARY_PATH = RUN_ROOT / 'summary.json'

def write_summary():
    summary['updated_at'] = time.time()
    summary['elapsed_seconds'] = summary['updated_at'] - summary['started_at']
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')

def stage(name, **fields):
    summary['stages'].append({'stage': name, 't': time.time(), **fields})
    print(f'[stage] {name}', fields if fields else '')
    write_summary()

stage('setup', run_root=str(RUN_ROOT), bandit_state=str(BANDIT_STATE_PATH))
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
    stage('dspy_present', version=getattr(dspy, '__version__', '?'))
except ImportError:
    out2 = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                            'dspy>=3.1.2'], capture_output=True, text=True)
    stage('pip_dspy', rc=out2.returncode, stderr_tail=out2.stderr[-400:])
    if out2.returncode != 0:
        summary['error'] = 'dspy install failed'; write_summary()
        raise SystemExit('dspy install failed')
    import dspy
    stage('dspy_installed', version=getattr(dspy, '__version__', '?'))

for mod in [m for m in list(sys.modules) if m == 'fabric_rlm' or m.startswith('fabric_rlm.')]:
    sys.modules.pop(mod, None)
import fabric_rlm
stage('imported', version=getattr(fabric_rlm, '__version__', '?'))
'''

LOAD_CASES_CELL = '''import json as _json
FIXTURE_ROOT = "''' + FIXTURE_ROOT + '''"

CASES = []
with open(FIXTURE_ROOT + "/longcot_cs_hard_pilot20.jsonl") as fh:
    rows = [_json.loads(line) for line in fh]
# 3 MCM (medium) + 4 hard templates. Same set as the medium+hard tier
# notebooks combined, so the bandit can demonstrate skipping rung 0 on
# the hopeless-for-cheap-LM templates while still trying it on MCM.
mcm = [r for r in rows if r.get("template") == "MCM"][:3]
for r in mcm:
    CASES.append({"id": "mcm-" + str(r["question_id"]), "question": r["prompt"],
                  "answer": r["answer"], "template": "MCM"})
for tmpl in ["MFMC", "Backprop", "DistMem", "VLIW"]:
    matches = [r for r in rows if r.get("template") == tmpl]
    if matches:
        r = matches[0]
        CASES.append({"id": tmpl.lower() + "-" + str(r["question_id"]),
                      "question": r["prompt"], "answer": r["answer"], "template": tmpl})

stage('cases_loaded', n=len(CASES), templates=sorted({c['template'] for c in CASES}))
'''

VALIDATE_CELL = '''import sys
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
                correct, _ = verify_cs_response(template, expected_answer, ans)
                return bool(correct)
            except Exception:
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

RUN_CELL = '''from fabric_rlm import RLM, FabricLM
from fabric_rlm.experimental import BanditState, BanditPolicy
from fabric_rlm.experimental.adaptive_policy import LadderPolicy

# Load (or create) shared bandit state. Re-runs of this notebook will
# accumulate priors over time.
state = BanditState.from_path(BANDIT_STATE_PATH)
def _state_obs_total(s):
    return sum(s.total_observations(k) for k in s.priors.keys())
stage('bandit_state_loaded', total_observations=_state_obs_total(state),
      task_keys=sorted(state.priors.keys()))

cheap = FabricLM('gpt-4.1-mini', temperature=0.0, cache=False)
strong = FabricLM('gpt-5', reasoning_effort='medium', cache=False)
stage('lms_built', cheap='gpt-4.1-mini', strong='gpt-5')

# Run the case set N_ITERATIONS times. Each iteration shares the bandit
# state with the next, so a hopeless-for-cheap-LM template's posterior
# accumulates and the starting rung migrates upward over iterations.
N_ITERATIONS = 3

# Capture the per-iteration ladder used for each template so we can
# visualize the bandit's learning curve.
ladder_history = []  # list of {iter, case_id, template, starting_rung, attempts, passed, elapsed}

for iter_idx in range(1, N_ITERATIONS + 1):
    iter_record = {'iter': iter_idx, 'cases': []}
    summary['iterations'].append(iter_record)
    write_summary()
    stage('iteration_start', iter=iter_idx,
          state_obs=_state_obs_total(state),
          state_keys=sorted(state.priors.keys()))
    for i, case in enumerate(CASES, 1):
        case_record = {'id': case['id'], 'template': case['template']}
        iter_record['cases'].append(case_record)
        write_summary()
        try:
            validator = make_validator(case)
            policy = BanditPolicy(
                state=state,
                task_key=case['template'],
                warmup=2,
                strong_lm_spec=strong,
            )
            rlm = RLM(
                signature='question -> answer',
                lm=cheap,
                engine='adaptive',
                adaptive=dict(
                    policy=policy,
                    validator=validator,
                    max_attempts=6,
                    parallel_rollouts=1,
                ),
            )
            t0 = time.perf_counter()
            result = rlm.run({'question': case['question']})
            elapsed = time.perf_counter() - t0
            meta = (result.trajectory.metadata or {}).get('adaptive', {}) if result.trajectory else {}
            attempts = meta.get('attempts', [])
            passed_now = bool(result.submitted and validator(result))
            starting_rung = attempts[0].get('rung') if attempts else None
            # Feed observation back into the bandit. Each rung in the
            # attempt log gets a record so the posterior reflects
            # which rungs actually worked / failed.
            for a in attempts:
                rung = a.get('rung')
                a_passed = bool(a.get('passed'))
                if rung is not None:
                    state.record(case['template'], rung, a_passed)
            case_record.update({
                'passed': passed_now,
                'submitted': result.submitted,
                'elapsed_seconds': elapsed,
                'starting_rung': starting_rung,
                'winner_rung': meta.get('winner_rung'),
                'stop_reason': meta.get('stop_reason'),
                'attempts': [{'rung': a.get('rung'), 'passed': a.get('passed')} for a in attempts],
                'n_attempts': len(attempts),
            })
            ladder_history.append({
                'iter': iter_idx, 'case_id': case['id'], 'template': case['template'],
                'starting_rung': starting_rung, 'n_attempts': len(attempts),
                'passed': passed_now, 'elapsed': elapsed,
            })
            stage('case_done', iter=iter_idx, id=case['id'], template=case['template'],
                  starting_rung=starting_rung, n_attempts=len(attempts), passed=passed_now)
        except Exception as exc:
            case_record.update({
                'passed': False, 'error': repr(exc),
                'traceback': traceback.format_exc(),
            })
            stage('case_error', iter=iter_idx, id=case['id'], error=repr(exc))
        # Persist after each case so a crash leaves usable state.
        try:
            state.save()
        except Exception as save_exc:
            stage('state_save_error', error=repr(save_exc))
    # Iteration summary
    passed_count = sum(1 for c in iter_record['cases'] if c.get('passed'))
    iter_record['passed_count'] = passed_count
    iter_record['total_cases'] = len(CASES)
    stage('iteration_done', iter=iter_idx, passed=passed_count, total=len(CASES))

# Final summary table
print()
print('=== Bandit ladder history ===')
print(f'{"iter":<5} {"template":<10} {"start_rung":<11} {"n_attempts":<11} {"passed":<7} {"elapsed":<8}')
for h in ladder_history:
    print(f'{h["iter"]:<5} {h["template"]:<10} {str(h["starting_rung"]):<11} {h["n_attempts"]:<11} {str(h["passed"]):<7} {h["elapsed"]:.1f}')

# Aggregate: by-template starting-rung over iterations
print()
print('=== Starting rung by (template, iter) ===')
templates = sorted({h['template'] for h in ladder_history})
print('template     ' + '  '.join(f'iter{i}' for i in range(1, N_ITERATIONS + 1)))
for tmpl in templates:
    cells = []
    for it in range(1, N_ITERATIONS + 1):
        match = [h for h in ladder_history if h['template'] == tmpl and h['iter'] == it]
        cells.append(str(match[0]['starting_rung']) if match else '-')
    print(f'{tmpl:<12} ' + '  '.join(f'{c:<5}' for c in cells))

summary['ladder_history'] = ladder_history
summary['final_state_obs'] = _state_obs_total(state)
summary['final_state_keys'] = sorted(state.priors.keys())
summary['passed'] = True  # smoke success — see ladder_history for accuracy
write_summary()
print()
print('TIER=bandit RUNS=' + str(N_ITERATIONS) + ' STATE_OBS=' + str(_state_obs_total(state)))
'''


def make_cell(source: str, kind: str = "code") -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook() -> dict:
    cells = [
        make_cell(CONFIGURE_CELL),
        make_cell("# fabric-rlm 0.1.11.dev0 — **BanditPolicy** validation\n\n"
                  "Runs the same 7-case set (3 MCM + MFMC/Backprop/DistMem/VLIW) "
                  "across N iterations, sharing a single ``BanditState``. "
                  "Iteration 1 cold-starts uniform; later iterations should "
                  "skip rung 0 on hopeless-for-cheap-LM templates.\n", "markdown"),
        make_cell(SETUP_CELL),
        make_cell(INSTALL_CELL),
        make_cell(LOAD_CASES_CELL),
        make_cell(VALIDATE_CELL),
        make_cell(RUN_CELL),
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
    nb = build_notebook()
    path = NB_DIR / "adaptive_test_bandit.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path, "cells=", len(nb["cells"]))
