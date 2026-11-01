#!/usr/bin/env python
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

from vla_fastvlm.evaluation import evaluate_policy
from vla_fastvlm.tracking import ExperimentTracker, git_metadata, runtime_metadata


@dataclass
class Args:
    checkpoint: str
    tasks: str = "button-press-topdown-v3,push-v3,reach-v3"
    episodes: int = 50
    replan_steps: int = 8
    max_steps: int = 500
    seed_start: int = 200000
    output_root: str = "/Volumes/data/checkpoints/fastvla/evals"
    run_name: str = "fastvla-closed-loop"
    group: str = "fastvla-confirmatory"
    counterfactual: bool = False
    save_videos: int = 3
    wandb_project: str = "fastvlm-vla-research"
    wandb_mode: str = "online"


def main(args: Args) -> None:
    tasks = tuple(name.strip() for name in args.tasks.split(",") if name.strip())
    checkpoint_metadata = json.loads(
        (Path(args.checkpoint) / "metadata.json").read_text(encoding="utf-8")
    )
    run_name = args.run_name
    if run_name == "fastvla-closed-loop":
        run_name = Path(args.checkpoint).parent.name
    if args.counterfactual:
        run_name += "-counterfactual"
    output = (
        Path(args.output_root)
        / run_name
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(args.checkpoint).name}"
    )
    tracker = ExperimentTracker(
        project=args.wandb_project,
        name=run_name,
        config={
            **asdict(args),
            "tasks": tasks,
            "checkpoint_metadata": checkpoint_metadata,
            **git_metadata(Path(__file__).resolve().parents[1]),
            "runtime": runtime_metadata(),
        },
        tags=(
            "closed-loop",
            "metaworld",
            "counterfactual-language" if args.counterfactual else "nominal-language",
        ),
        mode=args.wandb_mode,
    )
    try:
        summary = evaluate_policy(
            args.checkpoint,
            tasks=tasks,
            episodes=args.episodes,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            seed_start=args.seed_start,
            output_dir=output,
            tracker=tracker,
            counterfactual=args.counterfactual,
            save_videos=args.save_videos,
        )
        tracker.log(
            {
                "eval/overall/success_rate": summary["success_rate"],
                "eval/overall/successes": summary["successes"],
                "eval/overall/episodes": summary["total_episodes"],
            }
        )
        tracker.summary(summary)
    finally:
        tracker.finish()
    print(json.dumps({k: v for k, v in summary.items() if k != "task_results"}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
