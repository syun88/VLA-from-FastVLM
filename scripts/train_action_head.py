#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from vla_fastvlm.fastvla import FastVLAConfig, TrainConfig
from vla_fastvlm.training import CachedActionTrainer, MultiTaskCache


@dataclass
class Args:
    cache_root: str = "/Volumes/data/dataset/fastvla/features"
    tasks: str = "button-press-topdown-v3,push-v3,reach-v3"
    run_name: str = "fastvlm-frozen-action-head"
    output_root: str = "/Volumes/data/checkpoints/fastvla/runs"
    backbone_mode: str = "frozen"
    seed: int = 42
    epochs: int = 40
    max_steps: int | None = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    eval_every_epochs: int = 2
    save_every_epochs: int = 10
    head_hidden_dim: int = 512
    backbone_family: str = "fastvlm"
    base_model: str = "apple/FastVLM-0.5B"
    base_model_path: str = "/Volumes/data/checkpoints/fastvla/models/apple-fastvlm-0.5b-mlx"
    base_revision: str = "16375720c2d673fa583e57e9876afde27549c7d0"
    visual_hidden_size: int = 896
    spatial_grid: int = 4
    spatial_token_dim: int = 64
    action_horizon: int = 8
    use_global_feature: bool = True
    use_spatial_features: bool = True
    use_state: bool = True
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"
    resume_from: str | None = None


def _task_names(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main(args: Args) -> None:
    tasks = _task_names(args.tasks)
    dataset = MultiTaskCache.from_cache_root(args.cache_root, tasks)
    normalizations = [dataset.normalization(task) for task in tasks]
    state_dims = {int(norm["state_dim"]) for norm in normalizations}
    action_dims = {int(norm["action_dim"]) for norm in normalizations}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(f"Incompatible task schemas: states={state_dims}, actions={action_dims}")

    split_by_task: dict[str, dict] = {}
    train_parts: list[np.ndarray] = []
    validation_by_task: dict[str, np.ndarray] = {}
    for task in tasks:
        split_path = Path(args.cache_root) / task / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
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
        backbone_family=args.backbone_family,
        backbone_mode=args.backbone_mode,
        state_dim=state_dims.pop(),
        action_dim=action_dims.pop(),
        action_horizon=args.action_horizon,
        head_hidden_dim=args.head_hidden_dim,
        visual_hidden_size=args.visual_hidden_size,
        spatial_grid=args.spatial_grid,
        spatial_token_dim=args.spatial_token_dim,
        use_global_feature=args.use_global_feature
        and args.backbone_mode not in {"state_only", "vision_only"},
        use_spatial_features=args.use_spatial_features
        and args.backbone_mode != "state_only",
        use_state=args.use_state and args.backbone_mode != "vision_only",
        state_names=("gripper_x", "gripper_y", "gripper_z", "gripper_open"),
        action_names=("dx", "dy", "dz", "gripper"),
        action_space="metaworld_delta_xyz_gripper",
    )
    train_config = TrainConfig(
        run_name=args.run_name,
        output_dir=args.output_root,
        cache_root=args.cache_root,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_every_epochs=args.eval_every_epochs,
        save_every_epochs=args.save_every_epochs,
        wandb_project=args.wandb_project,
        wandb_group=args.run_name,
        wandb_mode=args.wandb_mode,  # type: ignore[arg-type]
        wandb_tags=(args.backbone_mode, f"seed-{args.seed}"),
    )
    output_dir = Path(args.output_root) / f"{args.run_name}-seed{args.seed}"
    trainer = CachedActionTrainer(
        policy_config,
        train_config,
        dataset,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )
    policy, _ = trainer.fit(
        train_indices=train_indices,
        validation_by_task=validation_by_task,
    )
    final = trainer.output_dir / "final"
    metadata_path = final / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["splits"] = split_by_task
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved={final}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
