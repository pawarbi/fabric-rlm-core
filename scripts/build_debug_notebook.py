"""Build a payload-introspection notebook: 1 easy case, dump everything."""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_tier_notebooks import (
    CONFIGURE_CELL, SETUP_CELL, INSTALL_CELL, LH_ID, LH_NAME, WS_ID, make_cell,
)

DEBUG_RUN_CELL = '''from fabric_rlm import RLM, FabricLM

cheap = FabricLM('gpt-4.1-mini', temperature=0.0, cache=False)
strong = FabricLM('gpt-5', reasoning_effort='medium', cache=False)
stage('lms_built')

def always_false_validator(result):
    return False  # force the engine to climb every rung

rlm = RLM(
    signature='question -> answer',
    lm=cheap,
    engine='adaptive',
    adaptive=dict(strong_lm=strong, validator=always_false_validator,
                  max_attempts=2, parallel_rollouts=1),
)
result = rlm.run({'question': 'What is 17 multiplied by 23? Reply with just the number.'})

# Dump everything
def safe(v, n=400):
    try:
        return str(v)[:n]
    except Exception:
        return repr(v)[:n]

dump = {
    'submitted': result.submitted,
    'failure_reason': result.failure_reason,
    'payload_type': type(result.payload).__name__,
    'payload_repr': safe(result.payload, 800),
    'payload_keys': list(result.payload.keys()) if isinstance(result.payload, dict) else None,
    'payload_answer_value': safe(result.payload.get('answer')) if isinstance(result.payload, dict) else None,
    'payload_answer_type': type(result.payload.get('answer')).__name__ if isinstance(result.payload, dict) else None,
    'trajectory_n_turns': len(result.trajectory.turns) if result.trajectory else 0,
    'trajectory_metadata_keys': list((result.trajectory.metadata or {}).keys()) if result.trajectory else [],
}

if result.trajectory and result.trajectory.turns:
    dump['turns_preview'] = []
    for t in result.trajectory.turns[:8]:
        dump['turns_preview'].append({
            'turn': t.turn,
            'turn_type': getattr(t, 'turn_type', '?'),
            'submitted': t.submitted,
            'code': safe(t.code, 300),
            'stdout': safe(t.stdout, 300),
            'response_text': safe(getattr(t, 'response_text', ''), 300),
            'error': safe(getattr(t, 'error', None), 200),
        })

# Adaptive metadata details
adaptive_meta = (result.trajectory.metadata or {}).get('adaptive', {}) if result.trajectory else {}
dump['adaptive_attempts_full'] = adaptive_meta.get('attempts', [])
dump['adaptive_winner_rung'] = adaptive_meta.get('winner_rung')
dump['adaptive_stop_reason'] = adaptive_meta.get('stop_reason')

summary['debug_dump'] = dump
summary['passed'] = True  # mark "ran successfully" for poller
write_summary()
print('PAYLOAD KEYS:', dump['payload_keys'])
print('ANSWER VALUE:', dump['payload_answer_value'])
'''

cells = [
    make_cell(CONFIGURE_CELL),
    make_cell("# adaptive payload introspection\n", "markdown"),
    make_cell(SETUP_CELL.format(tier="debug")),
    make_cell(INSTALL_CELL),
    make_cell(DEBUG_RUN_CELL),
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3.11", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kernel_info": {"name": "jupyter", "jupyter_kernel_name": "python3.11"},
        "dependencies": {"lakehouse": {"default_lakehouse": LH_ID, "default_lakehouse_name": LH_NAME, "default_lakehouse_workspace_id": WS_ID}},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = Path(__file__).resolve().parents[1] / "notebooks" / "adaptive_debug_payload.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out_path)
