#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro

from vla_fastvlm.paths import runs_root


def condition_name(metadata: dict) -> str:
    policy = metadata.get("policy", {})
    mode = policy.get("backbone_mode", "frozen")
    if policy.get("backbone_family") == "smolvlm":
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


@dataclass
class Args:
    runs_root: str = ""
    output: str = "results/confirmatory/training_runs.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: Args) -> None:
    import wandb

    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or getattr(api.viewer, "entity", None)
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or authenticate a W&B account")
    records = []
    root = Path(args.runs_root) if args.runs_root else runs_root()
    for metadata_path in sorted(root.glob("*/best/metadata.json")):
        checkpoint = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        policy = metadata.get("policy", config)
        run_id = metadata.get("wandb_run_id")
        validation = {}
        if run_id:
            try:
                run = api.run(f"{entity}/fastvlm-vla-research/{run_id}")
                validation = {
                    key: value
                    for key, value in run.summary.items()
                    if key.startswith("validation/") and isinstance(value, (int, float))
                }
            except Exception as exc:
                validation = {"lookup_error": f"{type(exc).__name__}: {exc}"}
        record = {
            "condition": condition_name({**metadata, "policy": policy}),
            "seed": metadata.get("training", {}).get("seed"),
            "checkpoint": str(checkpoint),
            "action_head_sha256": sha256(checkpoint / "action_head.safetensors"),
            "adapter_sha256": sha256(checkpoint / "adapters.safetensors")
            if (checkpoint / "adapters.safetensors").is_file()
            else None,
            "base_model": policy.get("base_model"),
            "base_revision": policy.get("base_revision"),
            "git_sha": metadata.get("git_sha"),
            "git_dirty_at_run": metadata.get("git_dirty"),
            "wandb_run_id": run_id,
            "trainable_parameter_count": metadata.get("trainable_parameter_count"),
            "lora_delta_norm": metadata.get("lora_delta_norm"),
            "validation": validation,
            "dataset": {
                task: {
                    "samples": values.get("samples"),
                    "normalization": values.get("normalization"),
                    "split": values.get("manifest", {}).get("split"),
                    "feature_run_id": values.get("manifest", {}).get("wandb_run_id"),
                }
                for task, values in metadata.get("dataset", {}).get("tasks", {}).items()
            },
        }
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
