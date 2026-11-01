# Contributing

`VLA-from-FastVLM` is an MLX-native research project on Apple Silicon. Contributions should
preserve exact provenance and avoid turning simulation evidence into hardware claims.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,sim]'
```

## Local checks

```bash
python -m ruff check src scripts tests
python -m pytest
python -m compileall -q src scripts
```

Tests that require an Apple GPU, a FastVLM checkpoint, or the Meta-World dataset must be kept
separate from lightweight unit tests and must state their exact local asset requirements.

## Pull requests

Include the hardware/OS, Python and MLX versions, base-model revision, dataset revision, whether
the change was tested with a real checkpoint, and whether any result is pilot or confirmatory.
Do not commit model weights, datasets, credentials, large rollout videos, or optimizer state.

## Scientific reporting

- Use episode-disjoint splits and report all planned seeds and evaluation episodes.
- Report failures and negative outcomes.
- Do not call cached-feature action-head training full-model fine-tuning.
- Do not describe simulation success as physical-robot capability.
- Do not make FastVLM superiority claims from one favorable run or offline loss alone.
- Record Git SHA and W&B run IDs for experiments.
