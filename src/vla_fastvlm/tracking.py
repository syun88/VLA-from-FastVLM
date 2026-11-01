from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def git_metadata(repo_root: str | Path) -> dict[str, str]:
    root = str(repo_root)

    def _run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "git_sha": _run("rev-parse", "HEAD"),
        "git_branch": _run("branch", "--show-current"),
        "git_dirty": str(bool(_run("status", "--porcelain"))).lower(),
    }


def mlx_memory_stats() -> dict[str, float]:
    stats: dict[str, float] = {}
    for name, function in (
        ("active_gb", getattr(mx, "get_active_memory", None)),
        ("cache_gb", getattr(mx, "get_cache_memory", None)),
        ("peak_gb", getattr(mx, "get_peak_memory", None)),
    ):
        if function is not None:
            stats[name] = float(function()) / 1e9
    return stats


def runtime_metadata() -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = {}
    for package in ("mlx", "mlx-vlm", "mlx-lm", "numpy", "wandb", "metaworld", "mujoco"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "device": str(mx.default_device()),
    }


class ExperimentTracker:
    """Best-effort W&B boundary that never makes local training depend on connectivity."""

    def __init__(
        self,
        *,
        project: str,
        name: str,
        config: dict[str, Any],
        tags: tuple[str, ...] = (),
        group: str | None = None,
        mode: str = "online",
    ) -> None:
        self.run = None
        self.mode_requested = mode
        self.mode_effective = mode
        self.start_error: str | None = None
        if mode == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=list(tags),
                group=group,
                mode=mode,
            )
            self.mode_effective = "online" if not self.run.settings._offline else "offline"
        except Exception as exc:  # pragma: no cover - network/environment dependent
            self.start_error = f"{type(exc).__name__}: {exc}"
            self.mode_effective = "disabled"

    @property
    def id(self) -> str | None:
        return None if self.run is None else self.run.id

    @property
    def url(self) -> str | None:
        return None if self.run is None else self.run.url

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as exc:  # pragma: no cover - network/environment dependent
            self.start_error = f"log failure: {type(exc).__name__}: {exc}"

    def summary(self, metrics: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for key, value in metrics.items():
                self.run.summary[key] = value
        except Exception as exc:  # pragma: no cover
            self.start_error = f"summary failure: {type(exc).__name__}: {exc}"

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:  # pragma: no cover
            self.start_error = f"finish failure: {type(exc).__name__}: {exc}"

    def provenance(self) -> dict[str, Any]:
        return {
            "wandb_project": self.run.project if self.run is not None else None,
            "wandb_run_id": self.id,
            "wandb_url": self.url,
            "wandb_mode_requested": self.mode_requested,
            "wandb_mode_effective": self.mode_effective,
            "wandb_error": self.start_error,
            "tracked_at_unix": time.time(),
        }


def ensure_private_wandb_defaults() -> None:
    """Keep credential handling out of source while documenting the local default."""

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", str(Path.home() / ".cache" / "wandb"))
