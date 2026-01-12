from __future__ import annotations

from typing import List

import torch

from vla_fastvlm.fastvla.configuration_fastvla import FastVLAConfig
from vla_fastvlm.model.fastvlm_adapter import FastVLMBackbone


class FastVLAProcessor:
    """
    Lightweight processor that mirrors the SmolVLA preparation flow:
    - normalize tasks (broadcast + trailing newline)
    - ensure images / states are the latest step when time-major
    - reuse FastVLMBackbone preprocessing for images/tokenizer.
    """

    def __init__(self, config: FastVLAConfig, backbone: FastVLMBackbone) -> None:
        self.config = config
        self.backbone = backbone

    def normalize_tasks(self, tasks: List[str] | str, batch_size: int) -> List[str]:
        if isinstance(tasks, str):
            tasks = [tasks]
        if len(tasks) == 1 and batch_size > 1:
            tasks = [tasks[0] for _ in range(batch_size)]
        if self.config.add_trailing_newline:
            tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        return tasks

    def prepare_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        if images.ndim == 5:  # (B, T, C, H, W)
            images = images[:, -1]
        return self.backbone._prepare_images_tensor(images, device)

    def prepare_states(self, states: torch.Tensor, device: torch.device) -> torch.Tensor:
        if states.ndim == 3:  # (B, T, D)
            states = states[:, -1]
        return states.to(device)

    def prepare_tasks(self, tasks: List[str] | str, batch_size: int) -> List[str]:
        return self.normalize_tasks(tasks, batch_size)
