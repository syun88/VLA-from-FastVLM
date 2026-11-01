#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Args:
    eval_root: str = "/Volumes/data/checkpoints/fastvla/evals"
    offline_root: str = "/Volumes/data/checkpoints/fastvla/offline_eval"
    benchmark_root: str = "/Volumes/data/checkpoints/fastvla/benchmarks"
    output_dir: str = "results/confirmatory"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def condition_name(metadata: dict) -> str:
    policy = metadata.get("policy", {})
    family = policy.get("backbone_family", "fastvlm")
    mode = policy.get("backbone_mode", "frozen")
    if family == "smolvlm":
        return "SmolVLM2-500M-frozen"
    if mode == "state_only":
        return "state-only"
    if mode == "vision_only":
        return "FastViTHD-spatial-only"
    if mode == "lora":
        return "FastVLM-LoRA"
    if not policy.get("use_spatial_features", True):
        return "FastVLM-frozen-global-only"
    return "FastVLM-frozen-global-spatial"


def main(args: Args) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    episode_rows: list[dict] = []
    for summary_path in sorted(Path(args.eval_root).glob("**/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary["checkpoint"])
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        config_path = checkpoint / "config.json"
        checkpoint_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        metadata = {**metadata, "policy": metadata.get("policy", checkpoint_config)}
        seed = metadata.get("training", {}).get("seed")
        condition = condition_name(metadata)
        for task, episodes in summary["task_results"].items():
            successes = sum(int(episode["success"]) for episode in episodes)
            low, high = wilson_interval(successes, len(episodes))
            run_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "task": task,
                    "episodes": len(episodes),
                    "successes": successes,
                    "success_rate": successes / max(1, len(episodes)),
                    "wilson_low": low,
                    "wilson_high": high,
                    "counterfactual": bool(summary.get("counterfactual")),
                    "checkpoint": str(checkpoint),
                    "wandb_run_id": summary.get("wandb_run_id"),
                    "wandb_url": summary.get("wandb_url"),
                    "summary_path": str(summary_path),
                }
            )
            for episode in episodes:
                episode_rows.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        **episode,
                        "summary_path": str(summary_path),
                    }
                )

    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]) if run_rows else ["condition"])
        writer.writeheader()
        writer.writerows(run_rows)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]) if episode_rows else ["condition"])
        writer.writeheader()
        writer.writerows(episode_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in run_rows:
        if not row["counterfactual"]:
            grouped[(row["condition"], row["task"])].append(row)
    aggregate_rows = []
    for (condition, task), rows in sorted(grouped.items()):
        total = sum(int(row["episodes"]) for row in rows)
        successes = sum(int(row["successes"]) for row in rows)
        low, high = wilson_interval(successes, total)
        per_seed = np.asarray([row["success_rate"] for row in rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "condition": condition,
                "task": task,
                "training_seeds": len(rows),
                "episodes": total,
                "successes": successes,
                "pooled_success_rate": successes / max(1, total),
                "pooled_wilson_low": low,
                "pooled_wilson_high": high,
                "seed_mean_success_rate": float(per_seed.mean()),
                "seed_sd_success_rate": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
            }
        )
    with (output / "task_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]) if aggregate_rows else ["condition"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    offline_rows = []
    for path in sorted(Path(args.offline_root).glob("**/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("checkpoint_metadata", {})
        for task, metrics in payload.get("tasks", {}).items():
            offline_rows.append(
                {
                    "condition": condition_name(metadata),
                    "seed": metadata.get("training", {}).get("seed"),
                    "task": task,
                    **metrics,
                    "summary_path": str(path),
                }
            )
    with (output / "offline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(offline_rows[0]) if offline_rows else ["condition"])
        writer.writeheader()
        writer.writerows(offline_rows)

    benchmark_rows = []
    for path in sorted(Path(args.benchmark_root).glob("**/latency.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(payload["checkpoint"])
        metadata_path = checkpoint / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        config_path = checkpoint / "config.json"
        checkpoint_config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        metadata = {**metadata, "policy": metadata.get("policy", checkpoint_config)}
        for resolution, result in payload.get("results", {}).items():
            benchmark_rows.append(
                {
                    "condition": condition_name(metadata),
                    "seed": metadata.get("training", {}).get("seed"),
                    "resolution": int(resolution),
                    "visual_token_count": result.get("visual_token_count"),
                    "cold_load_seconds": result["cold_load_seconds"],
                    "end_to_end_p50_ms": result["stages"]["end_to_end_seconds"]["p50_ms"],
                    "end_to_end_p95_ms": result["stages"]["end_to_end_seconds"]["p95_ms"],
                    "amortized_action_ms": result["amortized_action_ms"],
                    "peak_gb": result.get("memory/peak_gb"),
                    "summary_path": str(path),
                }
            )
    with (output / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmark_rows[0]) if benchmark_rows else ["condition"])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    manifest = {
        "eval_root": args.eval_root,
        "offline_root": args.offline_root,
        "benchmark_root": args.benchmark_root,
        "closed_loop_runs": len(run_rows),
        "episode_rows": len(episode_rows),
        "offline_rows": len(offline_rows),
        "benchmark_rows": len(benchmark_rows),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
