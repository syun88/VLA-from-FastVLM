#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro

from vla_fastvlm.fastvla import MLXFastVLAPolicy


@dataclass
class Args:
    checkpoint: str
    output_dir: str = "/Volumes/data/checkpoints/fastvla/release_candidates/fastvla-fastvlm-05b-metaworld"
    evaluation_glob: str = "/Volumes/data/checkpoints/fastvla/evals/**/summary.json"
    repo_id: str | None = None
    private: bool = True
    upload: bool = False


def _metrics_text(summary_paths: list[Path]) -> str:
    if not summary_paths:
        return "No confirmatory closed-loop result file was supplied at release preparation time."
    lines = [
        "| Evaluation | Episodes | Success | Rate | W&B run |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {path.parent.name} | {summary['total_episodes']} | {summary['successes']} | "
            f"{summary['success_rate']:.3f} | {summary.get('wandb_run_id') or 'local'} |"
        )
    return "\n".join(lines)


def main(args: Args) -> None:
    checkpoint = Path(args.checkpoint)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    serialized_checkpoint = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    policy = MLXFastVLAPolicy.load(checkpoint, load_backbone=True)
    if policy.action_policy is None:
        raise RuntimeError("Checkpoint action head did not restore")

    shutil.copy2(checkpoint / "action_head.safetensors", output / "action_head.safetensors")
    shutil.copy2(checkpoint / "config.json", output / "config.json")
    if policy.config.backbone_mode == "lora":
        shutil.copy2(checkpoint / "adapters.safetensors", output / "adapters.safetensors")
        if (checkpoint / "adapter_config.json").is_file():
            shutil.copy2(checkpoint / "adapter_config.json", output / "adapter_config.json")

    task_normalizations = {
        task: values["normalization"]
        for task, values in metadata["dataset"]["tasks"].items()
    }
    schema = {
        "artifact_type": "FastVLA action policy",
        "backbone_family": policy.config.backbone_family,
        "backbone_mode": policy.config.backbone_mode,
        "base_model": policy.config.base_model,
        "base_revision": policy.config.base_revision,
        "camera": "Metaworld corner2, inverted axes, 480x480 source rendered then FastVLM preprocessing",
        "state_names": list(policy.config.state_names),
        "action_names": list(policy.config.action_names),
        "state_dim": policy.config.state_dim,
        "action_dim": policy.config.action_dim,
        "action_horizon": policy.config.action_horizon,
        "control_frequency_hz": 80,
        "normalization_by_task": task_normalizations,
    }
    (output / "normalization.json").write_text(
        json.dumps(task_normalizations, indent=2), encoding="utf-8"
    )
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "third_party" / "apple-fastvlm-model-license.txt",
        output / "LICENSE-APPLE-AMLRA.txt",
    )
    shutil.copy2(Path(__file__).resolve().parents[1] / "LICENSE", output / "LICENSE-MIT.txt")

    summaries = sorted(Path("/").glob(args.evaluation_glob.lstrip("/")))
    metrics = _metrics_text(summaries)
    modifications = (
        "Added an 8-step MLX action-chunk head. For LoRA checkpoints, attached research LoRA "
        "adapters to FastVLM Qwen2 attention/MLP projection layers while freezing FastViTHD "
        "and the multimodal projector during cached-feature training."
    )
    card = f"""---
license: apple-amlr
library_name: mlx
base_model: apple/FastVLM-0.5B
base_model_relation: finetune
pipeline_tag: robotics
tags:
- vision-language-action
- mlx
- apple-silicon
- metaworld
- research-only
---

# FastVLA FastVLM-0.5B research artifact

This is an **unofficial research derivative**, not an Apple model release. Apple has not
reviewed, endorsed, or sponsored it.

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research
Model License Agreement. This artifact is restricted to non-commercial research purposes. See
`LICENSE-APPLE-AMLRA.txt`; that license controls the FastVLM-derived portions.

## Base model

- Model: `{policy.config.base_model}`
- Revision: `{policy.config.base_revision}`
- Base weights are **not** redistributed here. Obtain them from the official Apple repository
  and accept its license separately.

## Modifications

{modifications}

- Project source: https://github.com/syun88/VLA-from-FastVLM
- Source commit: `{metadata.get('git_sha')}`
- Checkpoint mode: `{policy.config.backbone_mode}`
- Action horizon: `{policy.config.action_horizon}`
- State/action dimensions: `{policy.config.state_dim}` / `{policy.config.action_dim}`

## Training data

`lerobot/metaworld_mt50`, Apache-2.0, revision
`a59f742d218c903328164257ecf180f9b18018a1`. The dataset itself is not redistributed. Selected
simulation tasks and episode splits are stored in `training_metadata.json` and the project
repository.

## Evaluation evidence

{metrics}

These are simulations on the Meta-World benchmark. They are not real robot measurements.

## MLX inference

1. Download the official `apple/FastVLM-0.5B` checkpoint at revision
   `{policy.config.base_revision}` and convert or load it through Apple/MLX-VLM's supported
   MLX path.
2. Load `config.json`, `normalization.json`, `action_head.safetensors`, and, when present,
   `adapters.safetensors` with the project's `MLXFastVLAPolicy.load` path.
3. Apply the task-specific state normalization from `normalization.json`, feed one 480 x 480
   Meta-World camera frame plus the task instruction, and de-normalize the returned 8 x 4
   action chunk with the same task metadata.
4. Replan according to the target control contract; do not treat open-loop action MSE as
   evidence of closed-loop success.

## License and attribution

Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research
Model License Agreement. Use, modification, creation of derivatives, and redistribution are
limited to Research Purposes. Redistribution requires this Agreement, the attribution above,
clear derivative identification, modification disclosure, and prohibits use of Apple names or
trademarks to endorse derivatives. No Apple endorsement is claimed.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    report = {
        "checkpoint": args.checkpoint,
        "restored": True,
        "output_dir": str(output),
        "config": serialized_checkpoint,
        "evaluation_summaries": [str(path) for path in summaries],
    }
    (output / "release_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if args.upload:
        if not args.repo_id:
            raise ValueError("--repo-id is required with --upload")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=output)
        print(f"uploaded private={args.private}: {args.repo_id}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
