#!/usr/bin/env python
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro
from PIL import Image

from vla_fastvlm.fastvla import MLXFastVLAPolicy


@dataclass
class Args:
    checkpoint: str
    task: str = "reach-v3"
    parquet: str = "/Volumes/data/huggingface/lerobot/lerobot/metaworld_mt50/data/chunk-000/file-000.parquet"
    row: int = 100


def main(args: Args) -> None:
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    task_metadata = metadata["dataset"]["tasks"][args.task]
    normalization = task_metadata["normalization"]
    state = np.asarray(normalization["state_mean"], dtype=np.float32)
    policy = MLXFastVLAPolicy.load(checkpoint, load_backbone=True)
    image = None
    if policy.config.uses_vlm:
        table = pq.read_table(args.parquet, columns=["observation.image"])
        value = table["observation.image"][args.row].as_py()
        image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    chunk, timing = policy.predict_chunk(
        image,
        state,
        str(task_metadata["instruction"]),
    )
    report = {
        "checkpoint": str(checkpoint),
        "task": args.task,
        "shape": list(chunk.shape),
        "finite": bool(np.isfinite(chunk).all()),
        "min": float(chunk.min()),
        "max": float(chunk.max()),
        "mean": float(chunk.mean()),
        "latency": timing,
        "git_sha": metadata.get("git_sha"),
        "wandb_run_id": metadata.get("wandb_run_id"),
        "base_model": policy.config.base_model,
        "base_revision": policy.config.base_revision,
    }
    if not report["finite"]:
        raise RuntimeError(f"Checkpoint produced non-finite actions: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
