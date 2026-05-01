# Adaptive bench — how to run

This bench requires a real LM. Because the cheap and ceiling configs both
default to `fabric/<model>` (Fabric's built-in OpenAI endpoint), it is meant
to be **run inside a Fabric notebook**, not on a local dev machine.

## In a Fabric notebook

```python
# 1. Install the wheel (replace path with the actual location)
%pip install abfss://sandeep_ws@onelake.dfs.fabric.microsoft.com/diagnostic.Lakehouse/Files/fabric_rlm_longcot/wheels/fabric_rlm-0.1.10-py3-none-any.whl

# 2. Pull the bench fixtures + runner into the notebook environment
import shutil, urllib.request, os, sys
os.makedirs("/lakehouse/default/Files/_adaptive_bench", exist_ok=True)
# Or git-clone the repo, or copy bench/adaptive/ contents from OneLake.
# Option A: clone repo
!git clone --depth 1 https://github.com/<your-fork>/fabric-rlm-core /tmp/frlm
%cd /tmp/frlm

# 3. Smoke-test (1 easy case, 1 mode, no cost concern)
!python -m bench.adaptive.run_bench \
    --output /lakehouse/default/Files/_adaptive_bench/smoke.json \
    --buckets easy --modes baseline --limit 1

# 4. Full run — all 4 modes × 33 cases. This will hit gpt-5 with reasoning_effort=high
#    on the ceiling pass, so confirm your Fabric quota first.
!python -m bench.adaptive.run_bench \
    --output /lakehouse/default/Files/_adaptive_bench/results-0.1.10.json \
    --cheap-lm fabric/gpt-4.1-mini \
    --strong-lm fabric/gpt-5 \
    --modes baseline retry_only adaptive ceiling \
    --buckets easy longcot spark
```

## Outputs

The runner writes one JSON file containing:

- `config` – exact CLI args
- `results` – one row per (case × mode) with `passed`, `score`, `turns_used`,
  `wall_seconds`, `prompt_tokens`, `completion_tokens`, `ncu`, `attempts`
- `aggregates.by_mode_bucket` – per-bucket totals
- `aggregates.by_mode_template` – per-template breakdown (catches per-family
  regressions that overall averages would hide)
- `aggregates.totals_by_mode` – grand totals
- `win_conditions` – the 6 conditions from the plan, each marked passed/failed
  (advisory; the runner does not crash on a fail)

After the run completes, copy the JSON back into the repo at
`bench/adaptive/results-0.1.10.json` and commit alongside the wheel.

## What I am skipping locally

`run_bench.py` imports `fabric_rlm.lm._fabric_factory` lazily, so the file
imports cleanly on a non-Fabric box (the smoke test in this repo's tests
verifies that). The actual `RLM(...).run()` call is what fails locally because
`synapse.ml.fabric.service_discovery` is only importable inside Fabric.

If you want to dry-run on a local machine with `OPENROUTER_API_KEY` set,
override the LM specs:

```bash
python -m bench.adaptive.run_bench \\
    --output /tmp/local-smoke.json \\
    --cheap-lm openrouter/openai/gpt-4o-mini \\
    --strong-lm openrouter/openai/gpt-5 \\
    --buckets easy --limit 2 --modes baseline adaptive
```

(That requires `dspy.LM` to know how to route the spec — which it does for
common providers via litellm — and `OPENAI_API_KEY` / `OPENROUTER_API_KEY` env
vars, depending on the chosen prefix.)
