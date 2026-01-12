from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vla_fastvlm.model.fastvlm_adapter import FastVLMBackboneConfig


@dataclass
class FastVLAConfig:
    """
    Configuration for adapting FastVLM into a VLA policy.
    Mirrors the structure of the SmolVLA config but keeps a FastVLM backbone.
    """

    vlm_model_name: str = "apple/FastVLM-base"
    state_dim: int = 14
    action_dim: int = 14
    hidden_dim: int = 1024
    fusion_dim: int = 1024
    dropout: float = 0.1
    freeze_backbone: bool = True

    # Preprocessing
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_to_max_length: bool = False
    resize_with_padding: bool = True
    image_size: Optional[int] = 512
    pad_value: float = 0.0
    add_trailing_newline: bool = True

    def to_backbone_config(self) -> FastVLMBackboneConfig:
        """Translate to the backbone adapter config."""
        return FastVLMBackboneConfig(
            model_id=self.vlm_model_name,
            freeze_backbone=self.freeze_backbone,
            force_image_size=self.image_size,
            resize_with_padding=self.resize_with_padding,
            pad_value=self.pad_value,
            tokenizer_max_length=self.tokenizer_max_length,
            tokenizer_padding_side=self.tokenizer_padding_side,
            pad_to_max_length=self.pad_to_max_length,
        )
