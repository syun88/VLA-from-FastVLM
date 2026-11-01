#!/usr/bin/env python
from __future__ import annotations

import copy
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import tyro
from PIL import Image

from vla_fastvlm.fastvla import MLXFastVLAPolicy
from vla_fastvlm.paths import benchmarks_root, dataset_root
from vla_fastvlm.tracking import (
    ExperimentTracker,
    git_metadata,
    mlx_memory_stats,
    runtime_metadata,
)


@dataclass
class Args:
    checkpoint: str
    task: str = "reach-v3"
    dataset_root: str = field(default_factory=lambda: str(dataset_root()))
    parquet: str = ""
    image_index: int = 100
    warmup: int = 3
    iterations: int = 30
    resolutions: str = "1024"
    output_root: str = field(default_factory=lambda: str(benchmarks_root()))
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"


def _image_from_parquet(path: str, row: int) -> Image.Image:
    table = pq.read_table(path, columns=["observation.image"])
    value = table["observation.image"][row].as_py()
    return Image.open(io.BytesIO(value["bytes"])).convert("RGB")


def main(args: Args) -> None:
    checkpoint = Path(args.checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    task_meta = metadata["dataset"]["tasks"][args.task]
    norm = task_meta["normalization"]
    state = np.zeros(len(norm["state_mean"]), dtype=np.float32)
    state[: min(3, len(state))] = np.asarray(norm["state_mean"], dtype=np.float32)[: min(3, len(state))]
    instruction = task_meta["instruction"]
    parquet = args.parquet or str(
        Path(args.dataset_root) / "data" / "chunk-000" / "file-000.parquet"
    )
    image = _image_from_parquet(parquet, args.image_index)
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=f"benchmark-{checkpoint.parent.name}",
        config={
            **args.__dict__,
            **git_metadata(Path(__file__).resolve().parents[1]),
            "runtime": runtime_metadata(),
        },
        tags=("benchmark", "latency", "apple-silicon"),
        group="performance-benchmark",
        mode=args.wandb_mode,
    )
    results = {}
    try:
        for resolution in [int(value) for value in args.resolutions.split(",")]:
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            cold_start = time.perf_counter()
            policy = MLXFastVLAPolicy.load(checkpoint, load_backbone=True)
            cold_load_seconds = time.perf_counter() - cold_start
            if policy.config.uses_vlm and policy.backbone.processor is not None:
                policy.backbone.processor.image_processor = copy.deepcopy(
                    policy.backbone.processor.image_processor
                )
                policy.backbone.processor.image_processor.size = {"shortest_edge": resolution}
                policy.config.image_longest_edge = resolution
            visual_token_count = None
            if policy.config.uses_vlm:
                feature_probe = policy.backbone.encode(image, instruction)
                visual_token_count = int(feature_probe.visual_tokens.shape[0])
            for _ in range(args.warmup):
                policy.predict_chunk(image, state, instruction)
            timings = []
            chunk_size = None
            for _ in range(args.iterations):
                chunk, timing = policy.predict_chunk(image, state, instruction)
                chunk_size = int(chunk.shape[0])
                timings.append(timing)
            keys = [
                "preprocessing_seconds",
                "vision_seconds",
                "language_seconds",
                "action_head_seconds",
                "end_to_end_seconds",
            ]
            stage_stats = {
                key: {
                    "p50_ms": float(np.percentile([t[key] for t in timings], 50) * 1000),
                    "p95_ms": float(np.percentile([t[key] for t in timings], 95) * 1000),
                    "mean_ms": float(np.mean([t[key] for t in timings]) * 1000),
                }
                for key in keys
            }
            result = {
                "resolution": resolution,
                "cold_load_seconds": cold_load_seconds,
                "action_chunk_size": chunk_size,
                "visual_token_count": visual_token_count,
                "amortized_action_ms": stage_stats["end_to_end_seconds"]["p50_ms"] / max(1, chunk_size or 1),
                "stages": stage_stats,
                **{f"memory/{key}": value for key, value in mlx_memory_stats().items()},
            }
            results[str(resolution)] = result
            tracker.log(
                {
                    f"benchmark/{resolution}/cold_load_seconds": cold_load_seconds,
                    f"benchmark/{resolution}/end_to_end_p50_ms": stage_stats["end_to_end_seconds"]["p50_ms"],
                    f"benchmark/{resolution}/end_to_end_p95_ms": stage_stats["end_to_end_seconds"]["p95_ms"],
                    f"benchmark/{resolution}/amortized_action_ms": result["amortized_action_ms"],
                    f"benchmark/{resolution}/visual_token_count": visual_token_count,
                    **{f"benchmark/{resolution}/memory/{key}": value for key, value in mlx_memory_stats().items()},
                }
            )
            policy.backbone.close()
            del policy
    finally:
        tracker.finish()
    output = Path(args.output_root) / checkpoint.parent.name / "latency.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint),
        "task": args.task,
        "instruction": instruction,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
        **tracker.provenance(),
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
