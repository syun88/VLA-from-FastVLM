#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from vla_fastvlm.fastvla import FastVLAConfig, TrainConfig
from vla_fastvlm.training import LoRAActionTrainer, MultiTaskCache


@dataclass
class Args:
    cache_root: str = "/Volumes/data/dataset/fastvla/features"
    tasks: str = "button-press-topdown-v3,push-v3,reach-v3"
    run_name: str = "fastvlm-lora-action-head"
    output_root: str = "/Volumes/data/checkpoints/fastvla/runs"
    base_model: str = "apple/FastVLM-0.5B"
    base_model_path: str = "/Volumes/data/checkpoints/fastvla/models/apple-fastvlm-0.5b-mlx"
    base_revision: str = "16375720c2d673fa583e57e9876afde27549c7d0"
    seed: int = 42
    max_steps: int = 2000
    lora_learning_rate: float = 1e-4
    action_head_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    head_hidden_dim: int = 512
    spatial_grid: int = 4
    spatial_token_dim: int = 64
    action_horizon: int = 8
    save_every: int = 500
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"


def _task_names(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main(args: Args) -> None:
    tasks = _task_names(args.tasks)
    dataset = MultiTaskCache.from_cache_root(args.cache_root, tasks)
    norms = [dataset.normalization(task) for task in tasks]
    state_dims = {int(norm["state_dim"]) for norm in norms}
    action_dims = {int(norm["action_dim"]) for norm in norms}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError("All tasks must share a state/action schema")
    train_parts = []
    validation_by_task = {}
    split_by_task = {}
    for task in tasks:
        split = json.loads(
            (Path(args.cache_root) / task / "split.json").read_text(encoding="utf-8")
        )
        split_by_task[task] = split
        train_parts.append(dataset.indices_for_episodes(task, split["train"]))
        validation_by_task[task] = dataset.indices_for_episodes(
            task, split["validation"]
        )
    train_indices = np.concatenate(train_parts)
    policy_config = FastVLAConfig(
        base_model=args.base_model,
        base_model_path=args.base_model_path,
        base_revision=args.base_revision,
        backbone_mode="lora",
        state_dim=state_dims.pop(),
        action_dim=action_dims.pop(),
        action_horizon=args.action_horizon,
        head_hidden_dim=args.head_hidden_dim,
        spatial_grid=args.spatial_grid,
        spatial_token_dim=args.spatial_token_dim,
        use_global_feature=True,
        use_spatial_features=True,
        use_state=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        state_names=("gripper_x", "gripper_y", "gripper_z", "gripper_open"),
        action_names=("dx", "dy", "dz", "gripper"),
        action_space="metaworld_delta_xyz_gripper",
    )
    train_config = TrainConfig(
        run_name=args.run_name,
        output_dir=args.output_root,
        cache_root=args.cache_root,
        seed=args.seed,
        epochs=1,
        max_steps=args.max_steps,
        batch_size=1,
        lora_learning_rate=args.lora_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_every_epochs=max(1, args.save_every // 100),
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,  # type: ignore[arg-type]
        wandb_tags=("lora", f"seed-{args.seed}"),
    )
    output_dir = Path(args.output_root) / f"{args.run_name}-seed{args.seed}"
    trainer = LoRAActionTrainer(
        policy_config,
        train_config,
        dataset,
        output_dir=output_dir,
    )
    _, _ = trainer.fit(
        train_indices=train_indices,
        validation_by_task=validation_by_task,
    )
    final = output_dir / "final"
    metadata_path = final / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["splits"] = split_by_task
    metadata["action_head_learning_rate_requested"] = args.action_head_learning_rate
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved={final}")


if __name__ == "__main__":
    main(tyro.cli(Args))
