from __future__ import annotations

from collections import deque
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from lerobot.configs.types import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from vla_fastvlm.fastvla.configuration_fastvla import FastVLAConfig as CoreFastVLAConfig
from vla_fastvlm.fastvla.fastvlm_with_expert import FastVLMWithExpert

from .configuration_fastvla import FastVLAConfig


class FastVLAPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for FastVLMWithExpert."""

    config_class = FastVLAConfig
    name = "fastvla"

    def __init__(self, config: FastVLAConfig, **kwargs: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._state_key, self._image_keys = self._resolve_input_keys()
        self._infer_io_dims_from_features()

        core_cfg = CoreFastVLAConfig(
            vlm_model_name=self.config.vlm_model_name,
            bootstrap_model_name=self.config.bootstrap_model_name,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_dim=self.config.fusion_dim,
            dropout=self.config.dropout,
            freeze_backbone=self.config.freeze_backbone,
            tokenizer_max_length=self.config.tokenizer_max_length,
            tokenizer_padding_side=self.config.tokenizer_padding_side,
            pad_to_max_length=self.config.pad_to_max_length,
            resize_with_padding=self.config.resize_with_padding,
            image_size=self.config.image_size,
            pad_value=self.config.pad_value,
            add_trailing_newline=self.config.add_trailing_newline,
        )
        self.model = FastVLMWithExpert(core_cfg)
        self.reset()

    def _resolve_input_keys(self) -> tuple[str, list[str]]:
        if not self.config.input_features:
            raise ValueError("FastVLA requires input_features to be set.")

        state_keys = [
            key for key, ft in self.config.input_features.items() if ft.type is FeatureType.STATE
        ]
        image_keys = [
            key for key, ft in self.config.input_features.items() if ft.type is FeatureType.VISUAL
        ]
        if not state_keys:
            raise ValueError("No state feature found in input_features.")
        if not image_keys:
            raise ValueError("No visual feature found in input_features.")
        return state_keys[0], image_keys

    def _infer_io_dims_from_features(self) -> None:
        if self.config.input_features and self._state_key in self.config.input_features:
            self.config.state_dim = self.config.input_features[self._state_key].shape[0]
        if self.config.action_feature is not None:
            self.config.action_dim = self.config.action_feature.shape[0]

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _prepare_inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, list[str]]:
        image_key = self._image_keys[0]
        images = batch[image_key]
        if images.ndim == 5:
            images = images[:, -1]

        states = batch[self._state_key]
        if states.ndim == 3:
            states = states[:, -1]

        task = batch.get("task")
        batch_size = images.shape[0]
        if task is None:
            tasks = [""] * batch_size
        elif isinstance(task, str):
            tasks = [task] * batch_size
        elif isinstance(task, (list, tuple)):
            tasks = [str(t) for t in task]
            if len(tasks) == 1 and batch_size > 1:
                tasks = tasks * batch_size
        else:
            tasks = [str(task)] * batch_size

        if self.config.add_trailing_newline:
            tasks = [t if t.endswith("\n") else f"{t}\n" for t in tasks]

        return images, states, tasks

    def _predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        images, states, tasks = self._prepare_inputs(batch)
        return self.model(images, states, tasks, device=images.device)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions = self._predict_actions(batch)
        return actions.unsqueeze(1)  # [B, chunk=1, D]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        pred_actions = self._predict_actions(batch)
        gt_actions = batch[ACTION]
        if gt_actions.ndim == 3:
            gt_actions = gt_actions[:, 0]
        loss = F.mse_loss(pred_actions, gt_actions)
        return loss, {"loss": loss.item(), "mse": loss.item()}

