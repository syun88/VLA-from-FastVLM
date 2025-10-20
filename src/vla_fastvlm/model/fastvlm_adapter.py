from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoConfig, AutoModel, AutoProcessor


@dataclass
class FastVLMBackboneConfig:
    model_id: str = "apple/FastVLM-base"
    revision: str | None = None
    pooling: str = "mean"  # options: mean | cls
    freeze_backbone: bool = True
    use_attention_mask: bool = True


class FastVLMBackbone(nn.Module):
    """
    Wrap Apple's FastVLM model to produce a fixed-size embedding.
    """

    def __init__(self, config: FastVLMBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or FastVLMBackboneConfig()

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=True,
        )
        hf_config = AutoConfig.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=True,
        )
        hidden_size = getattr(hf_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model, "config", None)
            hidden_size = getattr(hidden_size, "hidden_size", 1024)
        self.output_dim = hidden_size

        if self.config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _prepare_inputs(
        self,
        images: torch.Tensor,
        tasks: list[str],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        FastVLM's processor expects PIL images; convert tensors safely.
        """
        pil_images = [_tensor_to_pil(img) for img in images]
        processor_inputs = self.processor(
            images=pil_images,
            text=tasks,
            return_tensors="pt",
            padding=True,
        )
        return {key: value.to(device) for key, value in processor_inputs.items()}

    def forward(self, images: torch.Tensor, tasks: list[str], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = images.device
        inputs = self._prepare_inputs(images, tasks, device=device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)
        if self.config.pooling == "cls":
            pooled = last_hidden_state[:, 0, :]
        else:
            pooled = last_hidden_state.mean(dim=1)
        return pooled


@functools.lru_cache(maxsize=128)
def _tensor_to_pil(tensor: torch.Tensor):
    """Convert torch tensor (C,H,W) in [0,1] or [0,255] to PIL image."""
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return to_pil_image(tensor)
