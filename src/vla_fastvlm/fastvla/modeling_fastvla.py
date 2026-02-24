from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from vla_fastvlm.fastvla.configuration_fastvla import FastVLAConfig
from vla_fastvlm.fastvla.fastvlm_with_expert import FastVLMWithExpert
from vla_fastvlm.fastvla.processor_fastvla import FastVLAProcessor


class FastVLAPolicy(nn.Module):
    """
    FastVLM → VLA policy modeled after the SmolVLA structure
    (config + processor + backbone-with-expert).
    """

    config_class = FastVLAConfig
    name = "fastvla"

    def __init__(self, config: FastVLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or FastVLAConfig()
        self.model = FastVLMWithExpert(self.config)
        self.processor = FastVLAProcessor(self.config, self.model.backbone)

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        tasks: List[str] | str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute actions for a batch of observations.
        """
        if device is None:
            device = images.device
        images = self.processor.prepare_images(images, device)
        states = self.processor.prepare_states(states, device)
        tasks = self.processor.prepare_tasks(tasks, batch_size=images.shape[0])
        return self.model(images, states, tasks, device=device)

    def compute_loss(self, batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor]:
        """
        Regression MSE loss between predicted actions and targets.
        """
        images = batch["images"]
        states = batch["states"]
        actions = batch["actions"]
        tasks = batch["tasks"]

        predictions = self.forward(images, states, tasks)
        mse = F.mse_loss(predictions, actions)
        return {"loss": mse, "mse": mse.detach()}

    @torch.inference_mode()
    def select_action(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Produce a single action for inference scenarios."""
        self.eval()
        image_batch = image.unsqueeze(0).to(device)
        state_batch = state.unsqueeze(0).to(device)
        tasks = self.processor.prepare_tasks(task, batch_size=1)
        action = self.forward(image_batch, state_batch, tasks, device=device)
        return action.squeeze(0)

    def reset(self) -> None:
        """Included for API compatibility."""
        return
