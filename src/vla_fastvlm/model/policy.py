from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from .fastvlm_adapter import FastVLMBackbone, FastVLMBackboneConfig


@dataclass
class FastVLMPolicyConfig:
    backbone: FastVLMBackboneConfig = FastVLMBackboneConfig()
    state_dim: int = 14
    action_dim: int = 14
    hidden_dim: int = 1024
    fusion_dim: int = 1024
    dropout: float = 0.1
    freeze_backbone: bool = True


class FastVLMPolicy(nn.Module):
    """
    Vision-Language-Action policy composed of FastVLM backbone + action head.
    """

    def __init__(self, config: FastVLMPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or FastVLMPolicyConfig()
        self.backbone = FastVLMBackbone(self.config.backbone)

        self.state_projection = nn.Sequential(
            nn.LayerNorm(self.config.state_dim),
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.SiLU(),
        )

        fusion_input = self.backbone.output_dim + self.config.hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.config.fusion_dim),
            nn.LayerNorm(self.config.fusion_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fusion_dim, self.config.fusion_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(self.config.fusion_dim, self.config.action_dim)

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        tasks: List[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the next action for a batch of observations.
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images to be (B,C,H,W) got {images.shape}")
        if device is None:
            device = images.device

        backbone_features = self.backbone(images, tasks, device=device)
        state_features = self.state_projection(states)
        fused = torch.cat([backbone_features, state_features], dim=-1)
        fused = self.fusion(fused)
        return self.action_head(fused)

    def compute_loss(self, batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor]:
        """Compute regression loss for a batch."""
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
        action = self.forward(image_batch, state_batch, [task], device=device)
        return action.squeeze(0)

    def reset(self) -> None:
        """Provided for API compatibility with LeRobot."""
        return
