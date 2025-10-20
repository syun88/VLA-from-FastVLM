#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Optional

import tyro

from vla_fastvlm.real import AlohaHardwareInterface, AlohaRealConfig, AlohaRealRunner
from vla_fastvlm.utils import configure_logging, load_policy_from_checkpoint


def import_string(dotted_path: str):
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@dataclass
class RealArgs:
    checkpoint_dir: str = "outputs/train/aloha_fastvlm/checkpoints/step-1000"
    interface_cls: str = "my_robot.module.CustomAlohaInterface"
    interface_kwargs_json: str = "{}"
    task_instruction: str = "Insert the peg into the socket."
    control_frequency: float = 15.0
    max_episode_steps: int = 400
    device_preference: Optional[str] = None


def main(args: RealArgs) -> None:
    configure_logging()
    interface_kwargs = json.loads(args.interface_kwargs_json)
    interface_cls = import_string(args.interface_cls)
    if not issubclass(interface_cls, AlohaHardwareInterface):
        raise TypeError(f"{args.interface_cls} must subclass AlohaHardwareInterface")
    interface: AlohaHardwareInterface = interface_cls(**interface_kwargs)

    policy, device = load_policy_from_checkpoint(args.checkpoint_dir, device_preference=args.device_preference)
    real_config = AlohaRealConfig(
        control_frequency=args.control_frequency,
        max_episode_steps=args.max_episode_steps,
        task_instruction=args.task_instruction,
        device_preference=device.type,
    )

    with AlohaRealRunner(policy, interface, real_config) as runner:
        stats = runner.run_episode()
        print(f"Ran {stats['steps']} steps in {stats['duration_sec']:.2f}s")


if __name__ == "__main__":
    tyro.cli(main)
