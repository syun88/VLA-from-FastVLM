from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from vla_fastvlm.device import get_best_device


class AlohaHardwareInterface(ABC):
    """Abstract interface that must be implemented for a concrete Aloha robot setup."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connections to motors and sensors."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release hardware resources."""

    @abstractmethod
    def reset(self) -> None:
        """Reset robot to a neutral state before starting control."""

    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Return latest observation with at least:
            - 'image': np.ndarray of shape (H, W, 3) in uint8.
            - 'state': np.ndarray of shape (state_dim,) with joint positions (float32).
        """

    @abstractmethod
    def send_action(self, action: np.ndarray) -> None:
        """Command the robot with the next action expressed as target joint positions."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop any ongoing motion (safety stop)."""


@dataclass
class AlohaRealConfig:
    control_frequency: float = 15.0
    task_instruction: str = "Insert the peg into the socket."
    device_preference: Optional[str] = None
    max_episode_steps: int = 400
    auto_connect: bool = True
    safety_stop_on_exit: bool = True


class AlohaRealRunner:
    """Streaming control loop that queries the hardware interface and feeds actions."""

    def __init__(
        self,
        policy,
        interface: AlohaHardwareInterface,
        config: AlohaRealConfig | None = None,
    ) -> None:
        self.policy = policy
        self.interface = interface
        self.config = config or AlohaRealConfig()
        self.device = get_best_device(self.config.device_preference)

    def __enter__(self):
        if self.config.auto_connect:
            self.interface.connect()
            self.interface.reset()
        self.policy.to(self.device)
        self.policy.eval()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.config.safety_stop_on_exit:
            self.interface.stop()
        self.interface.disconnect()

    def run_episode(self) -> Dict[str, float]:
        dt = 1.0 / self.config.control_frequency
        step = 0
        start_time = time.perf_counter()

        for step in range(self.config.max_episode_steps):
            observation = self.interface.get_observation()
            image = torch.from_numpy(observation["image"]).permute(2, 0, 1).float() / 255.0
            state = torch.from_numpy(observation["state"]).float()

            action = self.policy.select_action(
                image=image.to(self.device),
                state=state.to(self.device),
                task=self.config.task_instruction,
                device=self.device,
            )

            action_np = action.detach().cpu().numpy().astype(np.float32)
            self.interface.send_action(action_np)

            elapsed = time.perf_counter() - start_time
            sleep_time = (step + 1) * dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return {
            "steps": step + 1,
            "duration_sec": time.perf_counter() - start_time,
        }
