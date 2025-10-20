#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from vla_fastvlm.data import (
    AlohaDataset,
    AlohaIterableDataset,
    create_aloha_dataloader,
)
from vla_fastvlm.model.policy import FastVLMPolicy, FastVLMPolicyConfig
from vla_fastvlm.model.fastvlm_adapter import FastVLMBackboneConfig
from vla_fastvlm.training import Trainer, TrainingConfig
from vla_fastvlm.utils import configure_logging


@dataclass
class TrainArgs:
    output_dir: str = "outputs/train/aloha_fastvlm"
    dataset_repo_id: str = "lerobot/aloha_sim_insertion_human_image"
    train_split: str = "train"
    eval_split: Optional[str] = "validation"
    streaming: bool = False
    limit_train_samples: Optional[int] = None
    limit_eval_samples: Optional[int] = 1024
    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 4

    model_id: str = "apple/FastVLM-base"
    freeze_backbone: bool = True
    hidden_dim: int = 1024
    fusion_dim: int = 1024
    dropout: float = 0.1

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    mixed_precision: Optional[str] = "bf16"
    seed: int = 42


def main(args: TrainArgs) -> None:
    configure_logging()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    backbone_config = FastVLMBackboneConfig(
        model_id=args.model_id,
        freeze_backbone=args.freeze_backbone,
    )
    policy_config = FastVLMPolicyConfig(
        backbone=backbone_config,
        hidden_dim=args.hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
    )
    policy = FastVLMPolicy(policy_config)

    if args.streaming:
        train_dataset = AlohaIterableDataset(split=args.train_split, repo_id=args.dataset_repo_id)
    else:
        train_dataset = AlohaDataset(
            split=args.train_split,
            repo_id=args.dataset_repo_id,
            limit_samples=args.limit_train_samples,
        )
    train_loader = create_aloha_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        num_workers=args.num_workers,
    )

    if args.eval_split:
        if args.streaming:
            eval_dataset = AlohaIterableDataset(
                split=args.eval_split,
                repo_id=args.dataset_repo_id,
            )
        else:
            eval_dataset = AlohaDataset(
                split=args.eval_split,
                repo_id=args.dataset_repo_id,
                limit_samples=args.limit_eval_samples,
            )
        eval_loader = create_aloha_dataloader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        eval_loader = None

    trainer_config = TrainingConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )

    trainer = Trainer(
        model=policy,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=trainer_config,
    )
    trainer.fit()


if __name__ == "__main__":
    tyro.cli(main)
