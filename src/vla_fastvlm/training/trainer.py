from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional
import logging

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.state import AcceleratorState

from vla_fastvlm.device import move_batch_to_device


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/train"
    num_epochs: int = 10
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str | None = "bf16"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    seed: int = 42
    resume_from: str | None = None
    gradient_checkpointing: bool = False
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])


class Trainer:
    """Lightweight trainer tailored for FastVLM policy fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: Iterable[Dict],
        eval_dataloader: Optional[Iterable[Dict]] = None,
        config: TrainingConfig | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        set_seed(self.config.seed)

        self.accelerator = self._create_accelerator()
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        )

        self.num_training_steps = self._compute_total_training_steps()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._build_scheduler_lambda(self.num_training_steps, self.config.warmup_ratio),
        )

        self.global_step = 0
        self.epoch = 0

    def _create_accelerator(self) -> Accelerator:
        desired_precision = self.config.mixed_precision
        attempted: set[str | None] = set()

        while True:
            try:
                accelerator = Accelerator(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    mixed_precision=desired_precision,
                    log_with=self.config.report_to,
                    project_dir=self.config.output_dir,
                )
                if desired_precision != self.config.mixed_precision:
                    self.config.mixed_precision = desired_precision
                return accelerator
            except ValueError as exc:
                message = str(exc).lower()
                has_mixed_kw = ("mixed precision" in message) or ("mixed_precision" in message)
                if (desired_precision is None) or (desired_precision in attempted) or not has_mixed_kw:
                    raise

                attempted.add(desired_precision)
                fallback = self._fallback_precision(str(desired_precision))
                if fallback == desired_precision:
                    raise

                logger.warning(
                    "Mixed precision '%s' not supported on this device (%s); falling back to '%s'.",
                    desired_precision,
                    exc,
                    fallback,
                )
                AcceleratorState._reset_state()
                desired_precision = fallback

    def _serialize_hparam(self, value):
        if isinstance(value, (int, float, bool, str, torch.Tensor)):
            return value
        if value is None:
            return "none"
        return str(value)

    def _build_tracker_config(self) -> Dict[str, int | float | str | bool | torch.Tensor]:
        raw_config = asdict(self.config)
        return {key: self._serialize_hparam(val) for key, val in raw_config.items()}

    def _fallback_precision(self, failed_precision: str) -> str | None:
        precision = failed_precision.lower()
        if precision in {"bf16", "bfloat16"}:
            if torch.cuda.is_available():
                return "fp16"
            return "no"
        if precision in {"fp16", "float16"}:
            return "no"
        return failed_precision

    def fit(self) -> None:
        output_dir = Path(self.config.output_dir)
        if self.accelerator.is_local_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(exist_ok=True)
            (output_dir / "logs").mkdir(exist_ok=True)
            with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
                json.dump(asdict(self.config), f, indent=2)

        tracker_config = self._build_tracker_config()
        self.accelerator.init_trackers("vla_fastvlm", config=tracker_config)

        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self._train_one_epoch()
            if self.global_step >= self.num_training_steps:
                break

        self.accelerator.end_training()

    def _train_one_epoch(self) -> None:
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                tensor_batch = move_batch_to_device(batch, self.accelerator.device)
                outputs = self.model.compute_loss(tensor_batch)
                loss = outputs["loss"]
                self.accelerator.backward(loss)

                if self.config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1

            if self.accelerator.is_main_process and self.global_step % self.config.logging_steps == 0:
                self.accelerator.log(
                    {
                        "train/loss": loss.detach().item(),
                        "train/mse": outputs["mse"].item(),
                        "train/lr": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": self.epoch,
                    },
                    step=self.global_step,
                )

            if self.global_step % self.config.eval_steps == 0 and self.eval_dataloader is not None:
                metrics = self.evaluate()
                if self.accelerator.is_main_process:
                    self.accelerator.log(metrics, step=self.global_step)

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(suffix=f"step-{self.global_step}")

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        for batch in self.eval_dataloader:
            tensor_batch = move_batch_to_device(batch, self.accelerator.device)
            outputs = self.model.compute_loss(tensor_batch)
            total_loss += outputs["mse"].item() * tensor_batch["actions"].shape[0]
            total_count += tensor_batch["actions"].shape[0]
        self.model.train()
        return {"eval/mse": total_loss / max(total_count, 1)}

    def _compute_total_training_steps(self) -> int:
        if self.config.max_steps:
            return self.config.max_steps
        if hasattr(self.train_dataloader, "__len__") and len(self.train_dataloader) > 0:
            batches_per_epoch = len(self.train_dataloader)
            updates_per_epoch = batches_per_epoch // self.config.gradient_accumulation_steps
            updates_per_epoch = max(updates_per_epoch, 1)
            return updates_per_epoch * self.config.num_epochs
        raise ValueError("Unable to infer total training steps from dataloader; please set max_steps.")

    def _build_scheduler_lambda(self, total_steps: int, warmup_ratio: float):
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
            )

        return lr_lambda

    def _save_checkpoint(self, suffix: str) -> None:
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / suffix
        self.accelerator.save_state(checkpoint_dir)
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            config_path = checkpoint_dir / "policy_config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(unwrapped_model.config), f, indent=2)
            model_path = checkpoint_dir / "policy_state_dict.pt"
            torch.save(unwrapped_model.state_dict(), model_path)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        self.accelerator.print(f"Resuming from checkpoint {path}")
        self.accelerator.load_state(checkpoint_path)
