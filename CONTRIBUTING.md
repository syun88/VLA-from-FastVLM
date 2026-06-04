# Contributing

`VLA-from-FastVLM` is an early-stage OSS bridge for researchers who want to test
FastVLM-style visual backbones inside Vision-Language-Action policies and
LeRobot workflows.

## Good first contributions

- Reproduce the LeRobot smoke test in a fresh environment and report the exact
  dependency versions.
- Add compatibility notes for new LeRobot releases.
- Improve dataset adapters, evaluation scripts, or documentation for real robot
  rollout.
- Add small tests that do not require downloading model checkpoints.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you are working inside a LeRobot virtual environment, follow the dependency
pins in `README.md` first and then add this repository to `PYTHONPATH`.

## Local checks

The default CI keeps checks lightweight so contributors do not need to download
FastVLM checkpoints:

```bash
python -m ruff check src scripts tests --select E9,F63,F7,F82
python -m compileall -q src scripts
python -m pytest
python -m build --sdist --wheel --no-isolation
```

Model-loading, dataset, and real robot rollout tests are welcome, but they
should be documented separately because they require large checkpoints,
datasets, or hardware.

## Pull requests

Please include:

- The LeRobot version and Python version used.
- Whether the change was tested with local checkpoints, Hub checkpoints, or no
  model download.
- Any known limitations, especially around hardware, dependency pins, or robot
  safety assumptions.

## Issue triage

Use issues for reproducible bugs, roadmap items, compatibility tracking, and
real robot rollout reports. Include command lines, environment details, and
failure logs when possible.
