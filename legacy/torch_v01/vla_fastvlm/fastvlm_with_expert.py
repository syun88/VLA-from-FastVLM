from __future__ import annotations

from typing import List

import torch
from torch import nn

from vla_fastvlm.fastvla.configuration_fastvla import FastVLAConfig
from vla_fastvlm.model.fastvlm_adapter import FastVLMBackbone


class FastVLMWithExpert(nn.Module):
    """
    FastVLM backbone plus a lightweight action expert head, organized similarly
    to the SmolVLA reference.
    """

    def __init__(self, config: FastVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = FastVLMBackbone(self.config.to_backbone_config())

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
        if device is None:
            device = images.device

        backbone_features = self.backbone(images, tasks, device=device)
        state_features = self.state_projection(states)
        fused = torch.cat([backbone_features, state_features], dim=-1)
        fused = self.fusion(fused)
        return self.action_head(fused)
