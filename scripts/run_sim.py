#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tyro

from vla_fastvlm.sim import AlohaSimulationConfig, run_aloha_simulation
from vla_fastvlm.utils import configure_logging, load_policy_from_checkpoint


@dataclass
class SimArgs:
    checkpoint_dir: str = "outputs/train/aloha_fastvlm/checkpoints/step-1000"
    num_episodes: int = 5
    max_episode_steps: int = 400
    task_instruction: str = "Insert the peg into the socket."
    device_preference: Optional[str] = None
    record_video: bool = True
    video_dir: str = "outputs/eval/aloha_fastvlm"


def main(args: SimArgs) -> None:
    configure_logging()
    policy, device = load_policy_from_checkpoint(args.checkpoint_dir, device_preference=args.device_preference)

    sim_config = AlohaSimulationConfig(
        max_episode_steps=args.max_episode_steps,
        task_instruction=args.task_instruction,
        num_episodes=args.num_episodes,
        record_video=args.record_video,
        video_dir=args.video_dir,
        device_preference=device.type,
    )
    results = run_aloha_simulation(policy, sim_config)
    for episode in results:
        success = "✅" if episode["success"] else "❌"
        print(
            f"Episode {episode['episode']:03d}: reward={episode['total_reward']:.2f} "
            f"steps={episode['steps']} success={success}",
        )


if __name__ == "__main__":
    tyro.cli(main)
