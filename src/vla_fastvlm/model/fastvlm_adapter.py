from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, CLIPVisionConfig, PretrainedConfig


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

        model_path = Path(self.config.model_id)
        local_files_only = model_path.exists()
        load_kwargs = {
            "revision": self.config.revision,
            "trust_remote_code": True,
        }
        if local_files_only:
            load_kwargs["local_files_only"] = True

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        )
        hf_config = AutoConfig.from_pretrained(
            self.config.model_id,
            **load_kwargs,
        )
        vision_config = getattr(hf_config, "vision_config", None)
        hydrated_vision_config = self._hydrate_vision_config(
            vision_config,
            model_path=model_path,
            local_files_only=local_files_only,
        )
        if hydrated_vision_config is not None:
            hf_config.vision_config = hydrated_vision_config

        if not isinstance(getattr(hf_config, "vision_config", None), PretrainedConfig):
            vision_dict = self._load_local_json(model_path, "vision_config.json") if local_files_only else None
            fallback = self._build_vision_config(vision_dict)
            if fallback is None:
                fallback = self._build_vision_config(self._default_vision_dict())
            if fallback is None:
                fallback = CLIPVisionConfig(**self._default_vision_dict())
            hf_config.vision_config = fallback

        self._ensure_llava_defaults(hf_config)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            config=hf_config,
            **load_kwargs,
        )
        hidden_size = getattr(hf_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model, "config", None)
            hidden_size = getattr(hidden_size, "hidden_size", 1024)
        self.output_dim = hidden_size

        if self.config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _hydrate_vision_config(
        self,
        vision_config: PretrainedConfig | dict | str | None,
        *,
        model_path: Path,
        local_files_only: bool,
    ) -> PretrainedConfig | None:
        if isinstance(vision_config, PretrainedConfig):
            return vision_config

        vision_dict: dict | None = None
        if isinstance(vision_config, str):
            vision_dict = self._load_local_json(model_path, vision_config) if local_files_only else None
            if vision_dict is None:
                vision_dict = self._load_remote_json(
                    self.config.model_id,
                    vision_config,
                    revision=self.config.revision,
                )
        elif isinstance(vision_config, dict):
            vision_dict = vision_config

        return self._build_vision_config(vision_dict)

    def _ensure_llava_defaults(self, config: PretrainedConfig) -> None:
        if not hasattr(config, "vision_feature_layer"):
            layer = getattr(config, "mm_vision_select_layer", -2)
            config.vision_feature_layer = layer

        if not hasattr(config, "vision_feature_select_strategy"):
            strategy = getattr(config, "mm_vision_select_feature", None)
            if strategy in (None, "patch"):
                strategy = "default"
            elif strategy == "token":
                strategy = "full"
            config.vision_feature_select_strategy = strategy

        if not hasattr(config, "image_seq_length"):
            image_size = getattr(getattr(config, "vision_config", None), "image_size", None)
            patch_size = getattr(getattr(config, "vision_config", None), "patch_size", None)
            seq_length = None
            if isinstance(image_size, int) and isinstance(patch_size, int) and patch_size > 0:
                grid = image_size // patch_size
                if grid > 0:
                    seq_length = grid * grid
            if seq_length is None:
                seq_length = getattr(config, "mm_hidden_size", 576)
            config.image_seq_length = seq_length

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

    @staticmethod
    def _build_vision_config(config_dict: dict | None) -> PretrainedConfig | None:
        if config_dict is None:
            return None

        model_type = config_dict.get("model_type") if isinstance(config_dict, dict) else None
        if model_type is not None:
            kwargs = dict(config_dict)
            kwargs.pop("model_type", None)
            try:
                return AutoConfig.for_model(model_type, **kwargs)
            except Exception:
                pass

        try:
            return CLIPVisionConfig(**config_dict)  # type: ignore[arg-type]
        except Exception:
            pass

        try:
            return PretrainedConfig.from_dict(config_dict)  # type: ignore[arg-type]
        except Exception:
            return None

    @staticmethod
    def _default_vision_dict() -> dict[str, int | str]:
        return {
            "model_type": "clip_vision_model",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "image_size": 1024,
            "patch_size": 16,
        }

    @staticmethod
    def _load_local_json(base_path: Path, filename: str) -> dict | None:
        candidate = base_path / filename
        if not candidate.exists():
            return None
        try:
            return json.loads(candidate.read_text())
        except Exception:
            return None

    @staticmethod
    def _load_remote_json(model_id: str, filename: str, revision: str | None = None) -> dict | None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            return None
        try:
            downloaded_path = hf_hub_download(
                model_id,
                filename,
                revision=revision,
                repo_type="model",
            )
            return json.loads(Path(downloaded_path).read_text())
        except Exception:
            return None


@functools.lru_cache(maxsize=128)
def _tensor_to_pil(tensor: torch.Tensor):
    """Convert torch tensor (C,H,W) in [0,1] or [0,255] to PIL image."""
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return to_pil_image(tensor)
