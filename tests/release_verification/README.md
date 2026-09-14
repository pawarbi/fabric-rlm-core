# Release verification notebooks

A maintainer imports these into a Fabric workspace and runs them against a
release. Each installs the pinned release and fails the run when a check
fails, so a green run is a statement about that version in that runtime.

- `verify_block_network_fabric.ipynb`: the worker stays sealed from the
  network under the default security policy, including in a run whose
  model is reachable.
- `verify_timeout_recovery_fabric.ipynb`: a worker timeout is recovered and
  the run finishes.
- `verify_delta_lakehouse_skill_in_fabric.ipynb`: the Delta lakehouse skill
  against a configured abfss root, with the analytics extra installed.

`tests/test_public_notebooks.py` checks that each notebook installs the
released version and ends with the assertion that fails the run.
