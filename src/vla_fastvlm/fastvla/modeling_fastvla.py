from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .backbone import FastVLMBackbone
from .configuration_fastvla import FastVLAConfig


def _adaptive_grid_pool(tokens: mx.array, grid: int) -> mx.array:
    """Pool a square token grid to grid x grid with average pooling."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected [B, N, D] tokens, got {tokens.shape}")
    batch, token_count, width = tokens.shape
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"Visual token count {token_count} is not a square grid")
    if grid < 1 or side % grid != 0:
        raise ValueError(f"Grid {grid} does not evenly divide token side {side}")
    step = side // grid
    pooled = tokens.reshape(batch, grid, step, grid, step, width).mean(axis=(2, 4))
    return pooled.reshape(batch, grid * grid, width)


class ActionPolicy(nn.Module):
    """Lightweight action-chunk head over FastVLM global and spatial features."""

    def __init__(self, config: FastVLAConfig, visual_hidden_size: int | None = None) -> None:
        super().__init__()
        self.config = config
        if config.action_dim is None:
            raise ValueError("action_dim must be provided")
        if config.use_state and config.state_dim is None:
            raise ValueError("state_dim must be provided when state is enabled")
        if not config.use_global_feature and not config.use_spatial_features and not config.use_state:
            raise ValueError("Policy has no enabled inputs")

        fusion_dim = 0
        if config.use_global_feature:
            if visual_hidden_size is None:
                raise ValueError("visual_hidden_size is required for global features")
            self.global_projection = nn.Sequential(
                nn.Linear(visual_hidden_size, config.head_hidden_dim // 2),
                nn.GELU(),
            )
            fusion_dim += config.head_hidden_dim // 2
        if config.use_spatial_features:
            if visual_hidden_size is None:
                raise ValueError("visual_hidden_size is required for spatial features")
            self.spatial_projection = nn.Linear(
                visual_hidden_size, config.spatial_token_dim
            )
            fusion_dim += config.spatial_grid * config.spatial_grid * config.spatial_token_dim
        if config.use_state:
            self.state_norm = nn.LayerNorm(config.state_dim)
            self.state_projection = nn.Sequential(
                nn.Linear(config.state_dim, config.state_hidden_dim),
                nn.GELU(),
            )
            fusion_dim += config.state_hidden_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, config.head_hidden_dim),
            nn.GELU(),
        )
        self.action_projection = nn.Linear(
            config.head_hidden_dim, config.action_horizon * config.action_dim
        )

    def __call__(
        self,
        *,
        visual_tokens: mx.array | None,
        global_hidden: mx.array | None,
        state: mx.array | None,
    ) -> mx.array:
        pieces: list[mx.array] = []
        if self.config.use_global_feature:
            if global_hidden is None:
                raise ValueError("global_hidden is required by this policy")
            pieces.append(self.global_projection(global_hidden))
        if self.config.use_spatial_features:
            if visual_tokens is None:
                raise ValueError("visual_tokens are required by this policy")
            pooled = _adaptive_grid_pool(visual_tokens, self.config.spatial_grid)
            projected = self.spatial_projection(pooled)
            pieces.append(projected.reshape(projected.shape[0], -1))
        if self.config.use_state:
            if state is None:
                raise ValueError("state is required by this policy")
            pieces.append(self.state_projection(self.state_norm(state)))
        fused = mx.concatenate(pieces, axis=-1)
        output = self.action_projection(self.fusion(fused))
        return output.reshape(
            output.shape[0], self.config.action_horizon, self.config.action_dim
        )

    def loss(
        self,
        predictions: mx.array,
        targets: mx.array,
        mask: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        squared = (predictions - targets) ** 2
        mask_expanded = mx.expand_dims(mask, axis=-1)
        denominator = mx.maximum(mx.sum(mask_expanded) * predictions.shape[-1], 1.0)
        mse = mx.sum(squared * mask_expanded) / denominator
        first_action_mse = mx.mean(squared[:, 0] * mask_expanded[:, 0])
        return mse, {"loss": mse, "first_action_mse": first_action_mse}


class MLXFastVLAPolicy:
    """End-to-end MLX policy: FastVLM features -> action chunk."""

    def __init__(self, config: FastVLAConfig, *, load_backbone: bool = True) -> None:
        self.config = config
        if config.backbone_family == "fastvlm":
            self.backbone = FastVLMBackbone(config, load_model=False)
        elif config.backbone_family == "smolvlm":
            from .alternate_backbone import SmolVLMBackbone

            self.backbone = SmolVLMBackbone(config, load_model=False)
        else:  # pragma: no cover - validated by FastVLAConfig
            raise ValueError(f"Unsupported backbone family: {config.backbone_family}")
        self.action_policy: ActionPolicy | None = None
        if load_backbone and config.uses_vlm:
            self.backbone.load()
            if config.backbone_mode == "lora":
                self.backbone.to_lora()
        if config.uses_vlm:
            hidden_size = config.visual_hidden_size
            if self.backbone.model is not None:
                hidden_size = self.backbone.hidden_size
            self.action_policy = ActionPolicy(config, visual_hidden_size=hidden_size)
        else:
            self.action_policy = ActionPolicy(config)

    def predict_chunk(
        self,
        image: Image.Image | np.ndarray | None,
        state: np.ndarray,
        instruction: str,
        *,
        visual_tokens: np.ndarray | None = None,
        global_hidden: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if self.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        preprocessing_seconds = 0.0
        vision_seconds = 0.0
        language_seconds = 0.0
        if self.config.uses_vlm and (visual_tokens is None or global_hidden is None):
            if image is None:
                raise ValueError("An image is required for a FastVLM policy")
            features = self.backbone.encode(image, instruction)
            visual_tokens = features.visual_tokens
            global_hidden = features.global_hidden
            preprocessing_seconds = features.preprocessing_seconds
            vision_seconds = features.vision_seconds
            language_seconds = features.language_seconds
        head_start = time.perf_counter()
        state_batch = mx.array(state[None, :].astype(np.float32))
        visual_batch = (
            mx.array(visual_tokens[None, ...].astype(np.float32))
            if visual_tokens is not None
            else None
        )
        global_batch = (
            mx.array(global_hidden[None, :].astype(np.float32))
            if global_hidden is not None
            else None
        )
        actions = self.action_policy(
            visual_tokens=visual_batch,
            global_hidden=global_batch,
            state=state_batch,
        )
        mx.eval(actions)
        head_seconds = time.perf_counter() - head_start
        return np.asarray(actions[0], dtype=np.float32), {
            "preprocessing_seconds": preprocessing_seconds,
            "vision_seconds": vision_seconds,
            "language_seconds": language_seconds,
            "action_head_seconds": head_seconds,
            "end_to_end_seconds": preprocessing_seconds
            + vision_seconds
            + language_seconds
            + head_seconds,
        }

    def save(self, output_dir: str | Path, metadata: dict[str, Any]) -> Path:
        if self.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.action_policy.save_weights(str(output / "action_head.safetensors"))
        (output / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if self.config.backbone_mode == "lora":
            from mlx_vlm.trainer.utils import save_adapter

            save_adapter(self.backbone.model, output / "adapters.safetensors")
        return output

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        load_backbone: bool = True,
    ) -> MLXFastVLAPolicy:
        path = Path(checkpoint_dir)
        values = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config = FastVLAConfig.from_dict(values)
        policy = cls(config, load_backbone=load_backbone)
        if config.backbone_mode == "lora" and load_backbone:
            adapter_file = path / "adapters.safetensors"
            if not adapter_file.is_file():
                raise FileNotFoundError(f"Missing LoRA adapter: {adapter_file}")
            # The pinned MLX-VLM adapter API expects adapter_config.json in the directory.
            policy.backbone.model.load_weights(str(adapter_file), strict=False)
        if policy.action_policy is None:
            raise RuntimeError("Checkpoint has no action policy")
        policy.action_policy.load_weights(str(path / "action_head.safetensors"))
        policy.action_policy.eval()
        return policy
