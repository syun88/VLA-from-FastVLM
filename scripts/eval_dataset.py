#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import tyro
from tqdm import tqdm

from vla_fastvlm.data import AlohaDataset, AlohaIterableDataset, create_aloha_dataloader
from vla_fastvlm.device import move_batch_to_device
from vla_fastvlm.utils import configure_logging, load_policy_from_checkpoint


@dataclass
class EvalArgs:
    checkpoint_dir: str = "outputs/train/aloha_fastvlm/checkpoints/step-1000"
    dataset_repo_id: str = "lerobot/aloha_sim_insertion_human_image"
    split: str = "validation"
    streaming: bool = False
    batch_size: int = 8
    num_workers: int = 4
    limit_samples: Optional[int] = None


def main(args: EvalArgs) -> None:
    configure_logging()
    policy, device = load_policy_from_checkpoint(args.checkpoint_dir)

    if args.streaming:
        dataset = AlohaIterableDataset(split=args.split, repo_id=args.dataset_repo_id)
    else:
        dataset = AlohaDataset(
            split=args.split,
            repo_id=args.dataset_repo_id,
            limit_samples=args.limit_samples,
        )
    dataloader = create_aloha_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    total_loss = 0.0
    total_samples = 0
    policy.eval()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        tensor_batch = move_batch_to_device(batch, device)
        with torch.inference_mode():
            outputs = policy.compute_loss(tensor_batch)
        total_loss += outputs["mse"].item() * tensor_batch["actions"].shape[0]
        total_samples += tensor_batch["actions"].shape[0]

    mse = total_loss / max(total_samples, 1)
    print(f"MSE on split '{args.split}': {mse:.6f}")


if __name__ == "__main__":
    tyro.cli(main)
