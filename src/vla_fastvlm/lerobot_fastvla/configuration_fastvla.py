from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("fastvla")
@dataclass
class FastVLAConfig(PreTrainedConfig):
    """LeRobot-compatible FastVLA policy config."""

    # Action-chunk interface expected by LeRobot.
    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # FastVLM backbone
    vlm_model_name: str = "apple/FastVLM-0.5B"
    bootstrap_model_name: str = "apple/FastVLM-0.5B"
    freeze_backbone: bool = True

    # MLP head dimensions (state/action dims are inferred from dataset/env features).
    state_dim: int = 14
    action_dim: int = 14
    hidden_dim: int = 1024
    fusion_dim: int = 1024
    dropout: float = 0.1

    # Preprocessing
    tokenizer_max_length: int = 64
    tokenizer_padding_side: str = "right"
    pad_to_max_length: bool = False
    resize_with_padding: bool = True
    image_size: int | None = 512
    pad_value: float = 0.0
    add_trailing_newline: bool = True

    # Optimizer / scheduler presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "n_action_steps must be <= chunk_size. "
                f"Got n_action_steps={self.n_action_steps}, chunk_size={self.chunk_size}."
            )

    def validate_features(self) -> None:
        if not self.input_features:
            return
        has_visual = any(ft.type is FeatureType.VISUAL for ft in self.input_features.values())
        has_state = any(ft.type is FeatureType.STATE for ft in self.input_features.values())
        if not has_visual:
            raise ValueError("FastVLA requires at least one visual observation feature.")
        if not has_state:
            raise ValueError("FastVLA requires at least one state observation feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

