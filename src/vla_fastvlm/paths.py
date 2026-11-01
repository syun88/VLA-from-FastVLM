from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Local artifact root for checkpoints, caches, and evaluation outputs.

    Keep machine-specific locations out of the public repository. Set
    FASTVLA_WORK_ROOT to the external drive or other artifact location.
    """

    return Path(os.environ.get("FASTVLA_WORK_ROOT", "artifacts")).expanduser()


def dataset_root() -> Path:
    override = os.environ.get("FASTVLA_DATASET_ROOT")
    if override:
        return Path(override).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return (
        hf_home
        / "hub"
        / "datasets--lerobot--metaworld_mt50"
        / "snapshots"
        / "a59f742d218c903328164257ecf180f9b18018a1"
    )


def fastvlm_mlx_path() -> Path:
    override = os.environ.get("FASTVLA_FASTVLM_MLX")
    if override:
        return Path(override).expanduser()
    return workspace_root() / "models" / "apple-fastvlm-0.5b-mlx"


def smolvlm_mlx_path() -> Path:
    override = os.environ.get("FASTVLA_SMOLVLM_MLX")
    if override:
        return Path(override).expanduser()
    return workspace_root() / "models" / "smolvlm2-500m-mlx"


def feature_cache_root() -> Path:
    return workspace_root() / "dataset" / "fastvla" / "features"


def runs_root() -> Path:
    return workspace_root() / "checkpoints" / "fastvla" / "runs"


def evaluations_root() -> Path:
    return workspace_root() / "checkpoints" / "fastvla" / "evals"


def benchmarks_root() -> Path:
    return workspace_root() / "checkpoints" / "fastvla" / "benchmarks"
