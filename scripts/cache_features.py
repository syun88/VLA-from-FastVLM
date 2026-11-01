#!/usr/bin/env python
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

from vla_fastvlm.data import (
    build_action_chunks,
    compute_task_normalization,
    iter_frames_with_images,
    load_or_create_split,
    scan_metaworld,
)
from vla_fastvlm.fastvla import FastVLAConfig, FastVLMBackbone
from vla_fastvlm.paths import (
    dataset_root,
    fastvlm_mlx_path,
    feature_cache_root,
)
from vla_fastvlm.tracking import (
    ExperimentTracker,
    git_metadata,
    mlx_memory_stats,
    runtime_metadata,
)


@dataclass
class Args:
    dataset_root: str = field(default_factory=lambda: str(dataset_root()))
    cache_root: str = field(default_factory=lambda: str(feature_cache_root()))
    tasks: str = "button-press-topdown-v3,push-v3,reach-v3"
    base_model_path: str = field(default_factory=lambda: str(fastvlm_mlx_path()))
    base_model: str = "apple/FastVLM-0.5B"
    base_revision: str = "16375720c2d673fa583e57e9876afde27549c7d0"
    backbone_family: str = "fastvlm"
    visual_hidden_size: int = 896
    dataset_revision: str = "a59f742d218c903328164257ecf180f9b18018a1"
    image_longest_edge: int = 1024
    frame_stride: int = 2
    action_horizon: int = 8
    split_seed: int = 20261101
    train_episodes: int = 35
    validation_episodes: int = 7
    test_episodes: int = 8
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"
    checkpoint_every: int = 50
    max_samples: int | None = None
    resume: bool = True


def _open_memmap(path: Path, shape: tuple[int, ...], dtype: str):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _cache_task(args: Args, task_name: str, task_data, split) -> Path:
    output = Path(args.cache_root) / task_name
    output.mkdir(parents=True, exist_ok=True)
    normalization = compute_task_normalization(task_data, split.train, args.action_horizon)
    actions, action_masks, selected_positions = build_action_chunks(
        task_data, args.action_horizon, frame_stride=args.frame_stride
    )
    count = len(selected_positions)
    if args.max_samples is not None:
        count = min(count, args.max_samples)
        selected_positions = selected_positions[:count]
        actions = actions[:count]
        action_masks = action_masks[:count]
    record_lookup = {index: frame for index, frame in enumerate(task_data.frames)}
    selected_by_key = {
        (record_lookup[position].episode_index, record_lookup[position].frame_index): ordinal
        for ordinal, position in enumerate(selected_positions)
    }
    state_dim = normalization["state_dim"]
    action_dim = normalization["action_dim"]

    paths = {
        "episode_index": output / "episode_index.npy",
        "frame_index": output / "frame_index.npy",
        "states": output / "states.npy",
        "actions": output / "actions.npy",
        "action_masks": output / "action_masks.npy",
        "input_ids": output / "input_ids.npy",
        "attention_mask": output / "attention_mask.npy",
        "visual_tokens": output / "visual_tokens.npy",
        "global_hidden": output / "global_hidden.npy",
        "latencies": output / "latencies.npy",
    }
    progress_path = output / "progress.json"
    completed = 0
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("count") != count:
            raise ValueError(f"Cache progress count mismatch in {output}")
        for key in ("episode_index", "frame_index", "states", "actions", "action_masks", "input_ids", "attention_mask", "visual_tokens", "global_hidden", "latencies"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Resume cache is missing {paths[key]}")
        completed = int(progress["completed"])

    provenance = git_metadata(Path(__file__).resolve().parents[1])
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=f"cache-fastvlm-05b-{task_name}",
        config={
            **asdict(args),
            **provenance,
            "runtime": runtime_metadata(),
            "task": task_name,
            "instruction": task_data.instruction,
            "samples": count,
            "split": split.as_dict(),
            "normalization": normalization,
        },
        tags=("fastvlm", "feature-cache", "metaworld"),
        group="feature-cache-v1",
        mode=args.wandb_mode,
    )
    policy_config = FastVLAConfig(
        base_model=args.base_model,
        base_model_path=args.base_model_path,
        base_revision=args.base_revision,
        backbone_family=args.backbone_family,
        image_longest_edge=args.image_longest_edge,
        visual_hidden_size=args.visual_hidden_size,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=args.action_horizon,
    )
    if args.backbone_family == "fastvlm":
        backbone = FastVLMBackbone(policy_config)
    elif args.backbone_family == "smolvlm":
        from vla_fastvlm.fastvla.alternate_backbone import SmolVLMBackbone

        backbone = SmolVLMBackbone(policy_config)
    else:
        raise ValueError(f"Unsupported backbone family: {args.backbone_family}")

    def initialize(first_features) -> dict[str, np.ndarray]:
        arrays = {
            "episode_index": _open_memmap(paths["episode_index"], (count,), "int64"),
            "frame_index": _open_memmap(paths["frame_index"], (count,), "int64"),
            "states": _open_memmap(paths["states"], (count, state_dim), "float32"),
            "actions": _open_memmap(paths["actions"], (count, args.action_horizon, action_dim), "float32"),
            "action_masks": _open_memmap(paths["action_masks"], (count, args.action_horizon), "float32"),
            "input_ids": _open_memmap(paths["input_ids"], (count, first_features.input_ids.shape[1]), "int32"),
            "attention_mask": _open_memmap(paths["attention_mask"], (count, first_features.attention_mask.shape[1]), "int8"),
            "visual_tokens": _open_memmap(paths["visual_tokens"], (count, *first_features.visual_tokens.shape), "float16"),
            "global_hidden": _open_memmap(paths["global_hidden"], (count, *first_features.global_hidden.shape), "float32"),
            "latencies": _open_memmap(paths["latencies"], (count,), "float32"),
        }
        return arrays

    arrays = None
    if completed:
        arrays = {key: np.load(path, mmap_mode="r+") for key, path in paths.items()}

    started = time.perf_counter()
    writable_index = completed
    selected_seen = 0
    try:
        for frame, image_bytes in iter_frames_with_images(task_data):
            key = (frame.episode_index, frame.frame_index)
            if key not in selected_by_key:
                continue
            action_ordinal = selected_by_key[key]
            if action_ordinal >= count:
                continue
            if selected_seen < completed:
                selected_seen += 1
                continue
            selected_seen += 1
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            features = backbone.encode(image, task_data.instruction)
            if arrays is None:
                arrays = initialize(features)
            if features.input_ids.shape[1] != arrays["input_ids"].shape[1]:
                raise ValueError("Prompt length changed within a task")
            arrays["episode_index"][writable_index] = frame.episode_index
            arrays["frame_index"][writable_index] = frame.frame_index
            arrays["states"][writable_index] = frame.state
            arrays["actions"][writable_index] = actions[action_ordinal]
            arrays["action_masks"][writable_index] = action_masks[action_ordinal]
            arrays["input_ids"][writable_index] = features.input_ids[0]
            arrays["attention_mask"][writable_index] = features.attention_mask[0]
            arrays["visual_tokens"][writable_index] = features.visual_tokens
            arrays["global_hidden"][writable_index] = features.global_hidden
            arrays["latencies"][writable_index] = (
                features.preprocessing_seconds
                + features.vision_seconds
                + features.language_seconds
            )
            writable_index += 1
            if writable_index % 10 == 0 or writable_index == count:
                for array in arrays.values():
                    array.flush()
                progress_path.write_text(
                    json.dumps({"completed": writable_index, "count": count}), encoding="utf-8"
                )
                recent = np.asarray(arrays["latencies"][max(0, writable_index - 50) : writable_index])
                tracker.log(
                    {
                        "cache/samples": writable_index,
                        "cache/fraction": writable_index / count,
                        "cache/wall_clock_seconds": time.perf_counter() - started,
                        "latency/p50_seconds": float(np.percentile(recent, 50)),
                        "latency/p95_seconds": float(np.percentile(recent, 95)),
                        **{f"memory/{k}": v for k, v in mlx_memory_stats().items()},
                    },
                    step=writable_index,
                )
                print(
                    f"{task_name}: {writable_index}/{count} "
                    f"p50={np.percentile(recent, 50):.3f}s",
                    flush=True,
                )
    finally:
        tracker.finish()
        backbone.close()

    if arrays is None or writable_index != count:
        raise RuntimeError(f"Incomplete cache {task_name}: {writable_index}/{count}")
    latencies = np.asarray(arrays["latencies"])
    manifest = {
        "format_version": 1,
        "task_name": task_name,
        "instruction": task_data.instruction,
        "samples": count,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_horizon": args.action_horizon,
        "frame_stride": args.frame_stride,
        "image_longest_edge": args.image_longest_edge,
        "image_token_position": int(np.flatnonzero(arrays["input_ids"][0] == -200)[0]),
        "has_visual": True,
        "base_model": args.base_model,
        "base_model_path": args.base_model_path,
        "base_revision": args.base_revision,
        "backbone_family": args.backbone_family,
        "visual_hidden_size": args.visual_hidden_size,
        "dataset_id": "lerobot/metaworld_mt50",
        "dataset_revision": args.dataset_revision,
        "split": split.as_dict(),
        "latency_summary": {
            "p50_seconds": float(np.percentile(latencies, 50)),
            "p95_seconds": float(np.percentile(latencies, 95)),
        },
        **provenance,
        **tracker.provenance(),
    }
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    return output


def main(args: Args) -> None:
    task_names = tuple(name.strip() for name in args.tasks.split(",") if name.strip())
    tasks = scan_metaworld(args.dataset_root, task_names)
    for task_name in task_names:
        split_path = Path(args.cache_root) / task_name / "split.json"
        split = load_or_create_split(
            tasks[task_name],
            split_path,
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
            seed=args.split_seed,
        )
        output = _cache_task(args, task_name, tasks[task_name], split)
        print(f"cached {task_name} -> {output}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
