from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from vla_fastvlm.data import CachedFeatureDataset
from vla_fastvlm.fastvla.configuration_fastvla import FastVLAConfig, TrainConfig
from vla_fastvlm.fastvla.modeling_fastvla import MLXFastVLAPolicy
from vla_fastvlm.tracking import (
    ExperimentTracker,
    git_metadata,
    mlx_memory_stats,
    runtime_metadata,
)


class MultiTaskCache:
    def __init__(self, task_caches: dict[str, CachedFeatureDataset]) -> None:
        self.tasks = task_caches
        self.owners: list[str] = []
        self.local_indices: list[int] = []
        episodes: list[int] = []
        for task_name, cache in task_caches.items():
            self.owners.extend(task_name for _ in range(len(cache)))
            self.local_indices.extend(range(len(cache)))
            episodes.extend(int(episode) for episode in cache.episode_index)
        self._counts = np.asarray(
            [sum(owner == name for owner in self.owners) for name in task_caches],
            dtype=np.int64,
        )
        self._task_names = list(task_caches)
        self._task_offsets = np.concatenate([[0], np.cumsum(self._counts)[:-1]]).astype(np.int64)
        self.episodes = np.asarray(episodes, dtype=np.int64)

    @classmethod
    def from_cache_root(cls, cache_root: str | Path, task_names: Iterable[str]) -> MultiTaskCache:
        root = Path(cache_root)
        if isinstance(task_names, str):
            task_names = (task_names,)
        caches = {}
        for task_name in task_names:
            task_dir = root / task_name
            caches[task_name] = CachedFeatureDataset(task_dir)
        return cls(caches)

    def __len__(self) -> int:
        return len(self.owners)

    def indices_for_episodes(self, task_name: str, episodes: Iterable[int]) -> np.ndarray:
        cache = self.tasks[task_name]
        local = cache.select_episodes(episodes)
        offset = int(self._task_offsets[self._task_names.index(task_name)])
        return local + offset

    def normalization(self, owner: str) -> dict[str, Any]:
        return self.tasks[owner].normalization

    def batch(self, indices: np.ndarray | list[int]) -> dict[str, mx.array]:
        index_array = np.asarray(indices, dtype=np.int64)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        visual: list[np.ndarray] = []
        globals_: list[np.ndarray] = []
        has_visual = False
        for index in index_array.tolist():
            owner = self.owners[index]
            local = self.local_indices[index]
            cache = self.tasks[owner]
            norm = cache.normalization
            state_mean = np.asarray(norm["state_mean"], dtype=np.float32)
            state_std = np.asarray(norm["state_std"], dtype=np.float32)
            action_mean = np.asarray(norm["action_mean"], dtype=np.float32)
            action_std = np.asarray(norm["action_std"], dtype=np.float32)
            states.append((np.asarray(cache.states[local]) - state_mean) / state_std)
            actions.append((np.asarray(cache.actions[local]) - action_mean) / action_std)
            masks.append(np.asarray(cache.action_masks[local], dtype=np.float32))
            if cache.visual_tokens is not None:
                has_visual = True
                visual.append(np.asarray(cache.visual_tokens[local], dtype=np.float32))
                globals_.append(np.asarray(cache.global_hidden[local], dtype=np.float32))
        result = {
            "state": mx.array(np.stack(states).astype(np.float32)),
            "actions": mx.array(np.stack(actions).astype(np.float32)),
            "action_mask": mx.array(np.stack(masks).astype(np.float32)),
        }
        if has_visual:
            if not all(cache.visual_tokens is not None for cache in self.tasks.values()):
                raise ValueError("Cannot mix visual and state-only feature caches")
            result["visual_tokens"] = mx.array(np.stack(visual).astype(np.float32))
            result["global_hidden"] = mx.array(np.stack(globals_).astype(np.float32))
        return result

    def denormalize_actions(self, owner: str, actions: np.ndarray) -> np.ndarray:
        norm = self.normalization(owner)
        mean = np.asarray(norm["action_mean"], dtype=np.float32)
        std = np.asarray(norm["action_std"], dtype=np.float32)
        return actions * std + mean

    def metadata(self) -> dict[str, Any]:
        return {
            "tasks": {
                name: {
                    "samples": len(cache),
                    "instruction": cache.instruction,
                    "manifest": cache.manifest,
                    "normalization": cache.normalization,
                }
                for name, cache in self.tasks.items()
            }
        }


def _learning_rate_at(step: int, total_steps: int, peak: float, warmup_ratio: float) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.5 * peak * (1.0 + math.cos(math.pi * progress))


def _grad_norm(gradients: Any) -> mx.array:
    leaves = [value for _, value in tree_flatten(gradients) if isinstance(value, mx.array)]
    if not leaves:
        return mx.array(0.0)
    return mx.sqrt(sum(mx.sum(value.astype(mx.float32) ** 2) for value in leaves))


class CachedActionTrainer:
    """Train an action head while FastViTHD and FastVLM are frozen."""

    def __init__(
        self,
        policy_config: FastVLAConfig,
        train_config: TrainConfig,
        dataset: MultiTaskCache,
        *,
        output_dir: str | Path,
        resume_from: str | Path | None = None,
    ) -> None:
        self.policy_config = policy_config
        self.train_config = train_config
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = MLXFastVLAPolicy(policy_config, load_backbone=False)
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy was not initialized")
        self.policy.action_policy.train()
        self.optimizer = optim.AdamW(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.start_step = 0
        self.resume_from = Path(resume_from) if resume_from else None
        if self.resume_from:
            self._resume(self.resume_from)

    def _head_loss(
        self, batch: dict[str, mx.array]
    ) -> tuple[mx.array, dict[str, mx.array]]:
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        head = self.policy.action_policy
        predictions = head(
            visual_tokens=batch.get("visual_tokens"),
            global_hidden=batch.get("global_hidden"),
            state=batch.get("state"),
        )
        return head.loss(predictions, batch["actions"], batch["action_mask"])

    def _evaluate_indices(self, indices: np.ndarray, *, batch_size: int = 128) -> dict[str, float]:
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        self.policy.action_policy.eval()
        total_sq = 0.0
        total_elements = 0
        finite_batches = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = self.dataset.batch(selected)
            predictions = self.policy.action_policy(
                visual_tokens=batch.get("visual_tokens"),
                global_hidden=batch.get("global_hidden"),
                state=batch.get("state"),
            )
            squared = (predictions - batch["actions"]) ** 2
            mask = mx.expand_dims(batch["action_mask"], axis=-1)
            total_sq += float(mx.sum(squared * mask).item())
            total_elements += int(mx.sum(mask).item()) * self.policy_config.action_dim
            finite_batches += int(bool(mx.all(mx.isfinite(predictions)).item()))
        self.policy.action_policy.train()
        return {
            "normalized_mse": total_sq / max(1, total_elements),
            "finite_batch_fraction": finite_batches / max(1, math.ceil(len(indices) / batch_size)),
        }

    def fit(
        self,
        *,
        train_indices: np.ndarray,
        validation_by_task: dict[str, np.ndarray],
    ) -> tuple[MLXFastVLAPolicy, ExperimentTracker]:
        cfg = self.train_config
        rng = np.random.default_rng(cfg.seed)
        steps_per_epoch = max(1, math.ceil(len(train_indices) / cfg.batch_size))
        total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
        provenance = git_metadata(Path(__file__).resolve().parents[3])
        tracker = ExperimentTracker(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config={
                "policy": self.policy_config.to_dict(),
                "training": asdict(cfg),
                "dataset": self.dataset.metadata(),
                **provenance,
                "runtime": runtime_metadata(),
                "trainable_parameter_count": self._trainable_parameter_count(),
            },
            tags=("fastvlm", "frozen", "action-head", *cfg.wandb_tags),
            group=cfg.wandb_group or self.policy_config.backbone_mode,
            mode=cfg.wandb_mode,
        )
        loss_and_grad = nn.value_and_grad(
            self.policy.action_policy, self._head_loss
        )
        started = time.perf_counter()
        global_step = self.start_step
        last_metrics: dict[str, float] = {}
        seen_episodes: set[tuple[str, int]] = set()
        best_validation_mse = float("inf")
        best_checkpoint: str | None = None
        try:
            for epoch in range(cfg.epochs):
                if global_step >= total_steps:
                    break
                self.policy.action_policy.train()
                order = rng.permutation(train_indices)
                epoch_losses: list[float] = []
                epoch_grad_norms: list[float] = []
                for batch_start in range(0, len(order), cfg.batch_size):
                    if global_step >= total_steps:
                        break
                    batch_indices = order[batch_start : batch_start + cfg.batch_size]
                    seen_episodes.update(
                        (
                            self.dataset.owners[int(index)],
                            int(self.dataset.episodes[int(index)]),
                        )
                        for index in batch_indices
                    )
                    batch = self.dataset.batch(batch_indices)
                    (loss, metrics), gradients = loss_and_grad(batch)
                    grad_norm = _grad_norm(gradients)
                    self.optimizer.learning_rate = mx.array(
                        _learning_rate_at(
                            global_step,
                            total_steps,
                            cfg.learning_rate,
                            cfg.warmup_ratio,
                        ),
                        dtype=mx.float32,
                    )
                    self.optimizer.update(self.policy.action_policy, gradients)
                    mx.eval(
                        self.policy.action_policy.parameters(),
                        self.optimizer.state,
                        loss,
                        grad_norm,
                    )
                    loss_value = float(loss.item())
                    grad_value = float(grad_norm.item())
                    epoch_losses.append(loss_value)
                    epoch_grad_norms.append(grad_value)
                    last_metrics = {
                        "train/loss": loss_value,
                        "train/first_action_mse": float(metrics["first_action_mse"].item()),
                        "train/gradient_norm": grad_value,
                        "train/learning_rate": float(self.optimizer.learning_rate.item()),
                        "train/epoch": float(epoch),
                        "train/global_step": float(global_step),
                        "train/samples_processed": float(
                            min(global_step + 1, total_steps) * cfg.batch_size
                        ),
                        "train/episodes_processed": float(len(seen_episodes)),
                        "train/wall_clock_seconds": time.perf_counter() - started,
                        **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
                    }
                    elapsed = max(1e-9, last_metrics["train/wall_clock_seconds"])
                    last_metrics["train/samples_per_second"] = (
                        last_metrics["train/samples_processed"] / elapsed
                    )
                    if global_step % cfg.log_every == 0:
                        tracker.log(last_metrics, step=global_step)
                    global_step += 1

                if epoch % cfg.eval_every_epochs == 0 or epoch == cfg.epochs - 1:
                    eval_metrics: dict[str, float] = {}
                    for task_name, indices in validation_by_task.items():
                        task_metrics = self._evaluate_indices(indices)
                        eval_metrics.update(
                            {f"validation/{task_name}/{k}": v for k, v in task_metrics.items()}
                        )
                    if eval_metrics:
                        mean_mse = float(
                            np.mean(
                                [
                                    value
                                    for key, value in eval_metrics.items()
                                    if key.endswith("normalized_mse")
                                ]
                            )
                        )
                        eval_metrics["validation/normalized_mse"] = mean_mse
                        eval_metrics["train/epoch"] = float(epoch)
                        tracker.log(eval_metrics, step=global_step)
                        last_metrics.update(eval_metrics)
                        if np.isfinite(mean_mse) and mean_mse < best_validation_mse:
                            best_validation_mse = mean_mse
                            best_checkpoint = str(
                                self.save_checkpoint(
                                    self.output_dir / "best",
                                    tracker=tracker,
                                    epoch=epoch,
                                    global_step=global_step,
                                )
                            )
                if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs - 1:
                    self.save_checkpoint(
                        self.output_dir / f"step-{global_step}",
                        tracker=tracker,
                        epoch=epoch,
                        global_step=global_step,
                    )
                print(
                    f"epoch={epoch + 1}/{cfg.epochs} step={global_step} "
                    f"loss={np.mean(epoch_losses):.6f} grad={np.mean(epoch_grad_norms):.4f}",
                    flush=True,
                )
        finally:
            if self.policy.action_policy is not None:
                self.policy.action_policy.eval()
            final_metrics = {
                **last_metrics,
                "train/wall_clock_seconds": time.perf_counter() - started,
                "train/optimizer_steps": float(global_step - self.start_step),
                "validation/best_normalized_mse": best_validation_mse
                if np.isfinite(best_validation_mse)
                else None,
                "validation/best_checkpoint": best_checkpoint,
                **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
            }
            tracker.summary(final_metrics)
            tracker.finish()
        self.save_checkpoint(
            self.output_dir / "final",
            tracker=tracker,
            epoch=cfg.epochs - 1,
            global_step=global_step,
        )
        return self.policy, tracker

    def _trainable_parameter_count(self) -> int:
        if self.policy.action_policy is None:
            return 0
        return int(
            sum(value.size for _, value in tree_flatten(self.policy.action_policy.trainable_parameters()))
        )

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        tracker: ExperimentTracker | None,
        epoch: int,
        global_step: int,
    ) -> Path:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        self.policy.action_policy.save_weights(str(path / "action_head.safetensors"))
        mx.save_safetensors(
            str(path / "optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        metadata = {
            **git_metadata(Path(__file__).resolve().parents[3]),
            **runtime_metadata(),
            "policy": self.policy_config.to_dict(),
            "training": asdict(self.train_config),
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_steps": global_step,
            "trainable_parameter_count": self._trainable_parameter_count(),
            "dataset": self.dataset.metadata(),
            **(tracker.provenance() if tracker is not None else {}),
        }
        (path / "config.json").write_text(
            json.dumps(self.policy_config.to_dict(), indent=2), encoding="utf-8"
        )
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return path

    def _resume(self, checkpoint_dir: Path) -> None:
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        head_path = checkpoint_dir / "action_head.safetensors"
        optimizer_path = checkpoint_dir / "optimizer.safetensors"
        if not head_path.is_file() or not optimizer_path.is_file():
            raise FileNotFoundError(f"Checkpoint is incomplete: {checkpoint_dir}")
        self.policy.action_policy.load_weights(str(head_path))
        state = mx.load(str(optimizer_path), return_metadata=False)
        self.optimizer.state = tree_unflatten(list(state.items()))
        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.is_file():
            self.start_step = int(json.loads(metadata_path.read_text())["global_step"])


class LoRAActionTrainer:
    """Real MLX LoRA updates on FastVLM's language layers plus action chunking."""

    def __init__(
        self,
        policy_config: FastVLAConfig,
        train_config: TrainConfig,
        dataset: MultiTaskCache,
        *,
        output_dir: str | Path,
    ) -> None:
        if policy_config.backbone_mode != "lora":
            raise ValueError("LoRAActionTrainer requires backbone_mode='lora'")
        self.policy_config = policy_config
        self.train_config = train_config
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = MLXFastVLAPolicy(policy_config, load_backbone=True)
        if self.policy.action_policy is None or self.policy.backbone.model is None:
            raise RuntimeError("LoRA policy failed to initialize")
        self.policy.action_policy.train()
        self.composite = _CompositePolicy(
            self.policy.backbone.model, self.policy.action_policy
        )
        self.optimizer = optim.AdamW(
            learning_rate=train_config.lora_learning_rate,
            weight_decay=train_config.weight_decay,
        )

    def _loss(self, batch: dict[str, mx.array], input_ids: mx.array, image_position: int, mask: mx.array) -> tuple[mx.array, dict[str, mx.array]]:
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        hidden = self.policy.backbone.hidden_from_cached_visual(
            input_ids,
            batch["visual_tokens"],
            image_token_position=image_position,
        )
        action_token = hidden[:, -1, :]
        predictions = self.policy.action_policy(
            visual_tokens=batch["visual_tokens"],
            global_hidden=action_token,
            state=batch["state"],
        )
        return self.policy.action_policy.loss(predictions, batch["actions"], batch["action_mask"])

    def fit(
        self,
        *,
        train_indices: np.ndarray,
        validation_by_task: dict[str, np.ndarray],
    ) -> tuple[MLXFastVLAPolicy, ExperimentTracker]:
        cfg = self.train_config
        rng = np.random.default_rng(cfg.seed)
        total_steps = cfg.max_steps or cfg.epochs * len(train_indices)
        provenance = git_metadata(Path(__file__).resolve().parents[3])
        tracker = ExperimentTracker(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config={
                "policy": self.policy_config.to_dict(),
                "training": asdict(cfg),
                "dataset": self.dataset.metadata(),
                **provenance,
                "runtime": runtime_metadata(),
                "trainable_parameter_count": int(
                    sum(
                        value.size
                        for _, value in tree_flatten(
                            self.composite.trainable_parameters()
                        )
                    )
                ),
            },
            tags=("fastvlm", "lora", "action-chunking", *cfg.wandb_tags),
            group=cfg.wandb_group or "fastvlm-lora",
            mode=cfg.wandb_mode,
        )
        if self.policy.backbone.model is None:
            raise RuntimeError("FastVLM model is not loaded")
        value_and_grad = nn.value_and_grad(self.composite, self._loss)
        started = time.perf_counter()
        initial_delta = self._lora_delta_norm()
        step = 0
        losses: list[float] = []
        grad_norms: list[float] = []
        seen_episodes: set[tuple[str, int]] = set()
        best_validation_mse = float("inf")
        best_checkpoint: str | None = None
        validation_every = max(100, total_steps // 8)
        try:
            while step < total_steps:
                order = rng.permutation(train_indices)
                for sample_index in order:
                    if step >= total_steps:
                        break
                    owner = self.dataset.owners[int(sample_index)]
                    local = self.dataset.local_indices[int(sample_index)]
                    cache = self.dataset.tasks[owner]
                    if cache.input_ids is None:  # pragma: no cover - defensive
                        raise RuntimeError("LoRA cache is missing token ids")
                    batch = self.dataset.batch(np.asarray([sample_index]))
                    seen_episodes.add(
                        (owner, int(self.dataset.episodes[int(sample_index)]))
                    )
                    input_ids = mx.array(np.asarray(cache.input_ids[local])[None, :].astype(np.int32))
                    mask = mx.array(np.asarray(cache.attention_mask[local])[None, :].astype(np.int32))
                    image_position = int(cache.manifest["image_token_position"])
                    (loss, metrics), gradients = value_and_grad(
                        batch, input_ids, image_position, mask
                    )
                    grad_norm = _grad_norm(gradients)
                    self.optimizer.learning_rate = mx.array(
                        _learning_rate_at(
                            step, total_steps, cfg.lora_learning_rate, cfg.warmup_ratio
                        ),
                        dtype=mx.float32,
                    )
                    self.optimizer.update(self.composite, gradients)
                    mx.eval(
                        self.composite.trainable_parameters(),
                        self.optimizer.state,
                        loss,
                        grad_norm,
                    )
                    loss_value = float(loss.item())
                    grad_value = float(grad_norm.item())
                    losses.append(loss_value)
                    grad_norms.append(grad_value)
                    if step % cfg.log_every == 0:
                        tracker.log(
                            {
                                "train/loss": loss_value,
                                "train/first_action_mse": float(metrics["first_action_mse"].item()),
                                "train/gradient_norm": grad_value,
                                "train/learning_rate": float(self.optimizer.learning_rate.item()),
                                "train/global_step": float(step),
                                "train/epoch": (step + 1) / max(1, total_steps),
                                "train/samples_processed": float(step + 1),
                                "train/episodes_processed": float(len(seen_episodes)),
                                "train/wall_clock_seconds": time.perf_counter() - started,
                                "train/samples_per_second": (step + 1)
                                / max(1e-9, time.perf_counter() - started),
                                "train/lora_delta_norm": float(self._lora_delta_norm().item()),
                                **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
                            },
                            step=step,
                        )
                    step += 1
                    if step % validation_every == 0 and validation_by_task:
                        validation_metrics = {}
                        for task_name, task_indices in validation_by_task.items():
                            validation = self._evaluate_indices(
                                task_indices, task_name=task_name, limit=128
                            )
                            validation_metrics.update(
                                {
                                    f"validation/{task_name}/{key}": value
                                    for key, value in validation.items()
                                }
                            )
                        mean_validation = float(
                            np.mean(
                                [
                                    value
                                    for key, value in validation_metrics.items()
                                    if key.endswith("normalized_mse")
                                ]
                            )
                        )
                        validation_metrics["validation/normalized_mse"] = mean_validation
                        tracker.log(validation_metrics, step=step)
                        if np.isfinite(mean_validation) and mean_validation < best_validation_mse:
                            best_validation_mse = mean_validation
                            best_checkpoint = str(
                                self.save_checkpoint(
                                    self.output_dir / "best", tracker=tracker, step=step
                                )
                            )
                    if step % max(cfg.save_every_epochs * 100, 200) == 0:
                        self.save_checkpoint(
                            self.output_dir / f"step-{step}", tracker=tracker, step=step
                        )
                    if step % 20 == 0:
                        print(
                            f"lora_step={step} loss={np.mean(losses[-20:]):.6f} "
                            f"grad={np.mean(grad_norms[-20:]):.4f}",
                            flush=True,
                        )
        finally:
            tracker.summary(
                {
                    "train/wall_clock_seconds": time.perf_counter() - started,
                    "train/final_loss_20": float(np.mean(losses[-20:])) if losses else None,
                    "train/lora_initial_delta_norm": float(initial_delta.item()),
                    "train/lora_final_delta_norm": float(self._lora_delta_norm().item()),
                    "validation/best_normalized_mse": best_validation_mse
                    if np.isfinite(best_validation_mse)
                    else None,
                    "validation/best_checkpoint": best_checkpoint,
                    **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
                }
            )
            tracker.finish()
        self.save_checkpoint(self.output_dir / "final", tracker=tracker, step=step)
        return self.policy, tracker

    def _evaluate_indices(
        self, indices: np.ndarray, *, task_name: str, limit: int = 128
    ) -> dict[str, float]:
        if self.policy.action_policy is None:
            raise RuntimeError("Action policy is not initialized")
        self.policy.action_policy.eval()
        self.policy.backbone.model.eval()
        total_loss = 0.0
        count = 0
        for index in indices[:limit]:
            local = self.dataset.local_indices[int(index)]
            cache = self.dataset.tasks[task_name]
            batch = self.dataset.batch(np.asarray([index]))
            input_ids = mx.array(
                np.asarray(cache.input_ids[local])[None, :].astype(np.int32)
            )
            mask = mx.array(
                np.asarray(cache.attention_mask[local])[None, :].astype(np.int32)
            )
            loss, _ = self._loss(
                batch,
                input_ids,
                int(cache.manifest["image_token_position"]),
                mask,
            )
            total_loss += float(loss.item())
            count += 1
        self.policy.action_policy.train()
        self.policy.backbone.model.train()
        return {"normalized_mse": total_loss / max(1, count), "samples": float(count)}

    def _lora_delta_norm(self) -> mx.array:
        if self.policy.backbone.model is None:
            return mx.array(0.0)
        leaves = []
        for name, value in tree_flatten(self.policy.backbone.model.trainable_parameters()):
            if "lora" in name.lower() and name.endswith((".A", ".B", "lora_a", "lora_b")):
                leaves.append(value.astype(mx.float32))
        if not leaves:
            # The installed MLX-VLM LoRA layer names vary by revision.
            leaves = [
                value.astype(mx.float32)
                for name, value in tree_flatten(self.policy.backbone.model.trainable_parameters())
                if ".lora" in name.lower() or name.endswith((".A", ".B"))
            ]
        if not leaves:
            return mx.array(0.0)
        total = sum(mx.sum(value * value) for value in leaves)
        return mx.sqrt(total)

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        tracker: ExperimentTracker | None,
        step: int,
    ) -> Path:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.policy.action_policy is None or self.policy.backbone.model is None:
            raise RuntimeError("LoRA policy is not initialized")
        from mlx_vlm.trainer.utils import save_adapter

        self.policy.action_policy.save_weights(str(path / "action_head.safetensors"))
        save_adapter(self.policy.backbone.model, path / "adapters.safetensors")
        mx.save_safetensors(
            str(path / "optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        metadata = {
            **git_metadata(Path(__file__).resolve().parents[3]),
            **runtime_metadata(),
            "policy": self.policy_config.to_dict(),
            "training": asdict(self.train_config),
            "global_step": step,
            "trainable_parameter_count": int(
                sum(
                    value.size
                    for _, value in tree_flatten(self.composite.trainable_parameters())
                )
            ),
            "lora_delta_norm": float(self._lora_delta_norm().item()),
            "dataset": self.dataset.metadata(),
            **(tracker.provenance() if tracker is not None else {}),
        }
        (path / "config.json").write_text(
            json.dumps(self.policy_config.to_dict(), indent=2), encoding="utf-8"
        )
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return path


class _CompositePolicy(nn.Module):
    """Expose FastVLM LoRA and the action head to one MLX gradient transform."""

    def __init__(self, vlm: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.vlm = vlm
        self.head = head
