#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import tyro

from vla_fastvlm.fastvla import MLXFastVLAPolicy
from vla_fastvlm.tracking import ExperimentTracker, git_metadata, runtime_metadata
from vla_fastvlm.training import MultiTaskCache


@dataclass
class Args:
    checkpoint: str
    cache_root: str = "/Volumes/data/dataset/fastvla/features"
    tasks: str = "button-press-topdown-v3,push-v3,reach-v3"
    split: str = "test"
    output_root: str = "/Volumes/data/checkpoints/fastvla/offline_eval"
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"


def main(args: Args) -> None:
    tasks = tuple(name.strip() for name in args.tasks.split(",") if name.strip())
    dataset = MultiTaskCache.from_cache_root(args.cache_root, tasks)
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    policy = MLXFastVLAPolicy.load(checkpoint, load_backbone=True)
    if policy.action_policy is None:
        raise RuntimeError("Checkpoint action head did not load")
    policy.action_policy.eval()
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=f"offline-{Path(args.checkpoint).parent.name}",
        config={**args.__dict__, **git_metadata(Path(__file__).resolve().parents[1]), "runtime": runtime_metadata()},
        tags=("offline-eval", args.split),
        mode=args.wandb_mode,
    )
    results = {}
    try:
        for task in tasks:
            split = json.loads(
                (Path(args.cache_root) / task / "split.json").read_text(encoding="utf-8")
            )
            indices = dataset.indices_for_episodes(task, split[args.split])
            cache = dataset.tasks[task]
            squared: list[np.ndarray] = []
            absolute: list[np.ndarray] = []
            if policy.config.backbone_mode == "lora":
                for index in indices:
                    local = dataset.local_indices[int(index)]
                    input_ids = mx.array(np.asarray(cache.input_ids[local])[None, :].astype(np.int32))
                    visual = mx.array(np.asarray(cache.visual_tokens[local])[None, :, :].astype(np.float32))
                    hidden = policy.backbone.hidden_from_cached_visual(
                        input_ids,
                        visual,
                        image_token_position=int(cache.manifest["image_token_position"]),
                    )
                    batch = dataset.batch([index])
                    prediction = policy.action_policy(
                        visual_tokens=batch["visual_tokens"],
                        global_hidden=hidden[:, -1, :],
                        state=batch["state"],
                    )
                    predicted = np.asarray(prediction[0], dtype=np.float32)
                    target = np.asarray(batch["actions"][0], dtype=np.float32)
                    mask = np.asarray(batch["action_mask"][0], dtype=np.float32)[..., None]
                    squared.append(((predicted - target) ** 2) * mask)
                    absolute.append(np.abs(predicted - target) * mask)
            else:
                for start in range(0, len(indices), 128):
                    batch = dataset.batch(indices[start : start + 128])
                    prediction = policy.action_policy(
                        visual_tokens=batch.get("visual_tokens"),
                        global_hidden=batch.get("global_hidden"),
                        state=batch.get("state"),
                    )
                    predicted = np.asarray(prediction, dtype=np.float32)
                    target = np.asarray(batch["actions"], dtype=np.float32)
                    mask = np.asarray(batch["action_mask"], dtype=np.float32)[..., None]
                    squared.append(((predicted - target) ** 2) * mask)
                    absolute.append(np.abs(predicted - target) * mask)
            squared_array = np.concatenate([x.reshape(-1) for x in squared])
            absolute_array = np.concatenate([x.reshape(-1) for x in absolute])
            # Normalized loss is used for reproducible model comparison; physical RMSE is
            # reported separately after per-dimension hand-action denormalization.
            normalized_rmse = float(np.sqrt(np.mean(squared_array)))
            normalized_mae = float(np.mean(absolute_array))
            results[task] = {
                "samples": int(len(indices)),
                "normalized_rmse": normalized_rmse,
                "normalized_mae": normalized_mae,
            }
            tracker.log(
                {
                    f"offline/{task}/normalized_rmse": normalized_rmse,
                    f"offline/{task}/normalized_mae": normalized_mae,
                    f"offline/{task}/samples": len(indices),
                }
            )
    finally:
        tracker.finish()
    output = Path(args.output_root) / Path(args.checkpoint).parent.name / f"{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "tasks": results,
        "mean_normalized_rmse": float(np.mean([item["normalized_rmse"] for item in results.values()])),
        "checkpoint_metadata": metadata,
        **tracker.provenance(),
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
