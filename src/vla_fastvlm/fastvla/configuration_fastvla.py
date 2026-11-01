from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BackboneMode = Literal["frozen", "lora", "state_only", "vision_only"]


@dataclass
class FastVLAConfig:
    """Configuration shared by training, evaluation, and exported checkpoints.

    State and action dimensions deliberately have no robotics-specific default. A task
    adapter supplies the schema, so the core model is not tied to Meta-World.
    """

    base_model: str = "apple/FastVLM-0.5B"
    base_model_path: str | None = None
    base_revision: str = "16375720c2d673fa583e57e9876afde27549c7d0"
    backbone_mode: BackboneMode = "frozen"
    action_horizon: int = 8
    state_dim: int | None = None
    action_dim: int | None = None
    head_hidden_dim: int = 512
    visual_hidden_size: int = 896
    state_hidden_dim: int = 64
    spatial_grid: int = 4
    spatial_token_dim: int = 64
    use_global_feature: bool = True
    use_spatial_features: bool = True
    use_state: bool = True
    dropout: float = 0.0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    image_longest_edge: int = 1024
    max_text_tokens: int = 64
    max_sequence_tokens: int = 512
    action_space: str = "joint_or_cartesian"
    action_names: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.state_dim is not None and self.state_dim < 1:
            raise ValueError("state_dim must be positive when provided")
        if self.action_dim is not None and self.action_dim < 1:
            raise ValueError("action_dim must be positive when provided")
        if self.use_state and self.state_dim is None:
            raise ValueError("use_state=True requires state_dim")
        if not self.use_global_feature and not self.use_spatial_features:
            if self.backbone_mode != "state_only":
                raise ValueError("At least one FastVLM feature path must be enabled")
        if self.backbone_mode not in {"frozen", "lora", "state_only", "vision_only"}:
            raise ValueError(f"Unsupported backbone_mode: {self.backbone_mode}")
        if self.backbone_mode == "state_only":
            self.use_global_feature = False
            self.use_spatial_features = False
            self.use_state = True
        if self.backbone_mode == "vision_only":
            self.use_global_feature = False
            self.use_spatial_features = True
            self.use_state = False

    @property
    def uses_vlm(self) -> bool:
        return self.backbone_mode != "state_only" and (
            self.use_global_feature or self.use_spatial_features
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FastVLAConfig:
        if isinstance(values.get("action_names"), list):
            values = {**values, "action_names": tuple(values["action_names"])}
        if isinstance(values.get("state_names"), list):
            values = {**values, "state_names": tuple(values["state_names"])}
        return cls(**values)


@dataclass
class DatasetConfig:
    repo_id: str = "lerobot/metaworld_mt50"
    revision: str = "a59f742d218c903328164257ecf180f9b18018a1"
    local_root: str = "/Volumes/data/huggingface/lerobot/lerobot/metaworld_mt50"
    tasks: tuple[str, ...] = (
        "button-press-topdown-v3",
        "push-v3",
        "reach-v3",
    )
    train_episodes: int = 35
    validation_episodes: int = 7
    test_episodes: int = 8
    split_seed: int = 20261101
    frame_stride: int = 2
    action_horizon: int = 8
    control_frequency_hz: int = 80


@dataclass
class TrainConfig:
    run_name: str = "fastvla-frozen"
    output_dir: str = "/Volumes/data/checkpoints/fastvla/runs"
    cache_root: str = "/Volumes/data/dataset/fastvla/features"
    seed: int = 42
    epochs: int = 40
    max_steps: int | None = None
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    lora_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    log_every: int = 1
    eval_every_epochs: int = 2
    save_every_epochs: int = 10
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_tags: tuple[str, ...] = ()


@dataclass
class SimConfig:
    task: str
    instruction: str
    episodes: int = 50
    max_steps: int = 500
    camera_name: str = "corner2"
    image_width: int = 480
    image_height: int = 480
    replan_steps: int = 8
    seed_start: int = 200000
    counterfactual_task: str | None = None
    save_videos: int = 3


@dataclass
class ExperimentMetadata:
    git_sha: str
    git_branch: str
    wandb_run_id: str | None = None
    wandb_url: str | None = None
    checkpoint_id: str | None = None
    dataset_id: str = "lerobot/metaworld_mt50"
    dataset_revision: str = "a59f742d218c903328164257ecf180f9b18018a1"
    base_model: str = "apple/FastVLM-0.5B"
    base_revision: str = "16375720c2d673fa583e57e9876afde27549c7d0"
    runtime: dict[str, Any] = field(default_factory=dict)
