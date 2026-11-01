#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    results_dir: str = "results/confirmatory"
    output_dir: str = "paper/figures"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(args: Args) -> None:
    result_dir = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_rows = _read(result_dir / "task_summary.csv")
    latency_rows = _read(result_dir / "latency.csv")
    tasks = sorted({row["task"] for row in task_rows})
    conditions = sorted({row["condition"] for row in task_rows})
    values = {
        (row["condition"], row["task"]): float(row["pooled_success_rate"])
        for row in task_rows
    }
    lows = {
        (row["condition"], row["task"]): float(row["pooled_success_rate"])
        - float(row["pooled_wilson_low"])
        for row in task_rows
    }
    highs = {
        (row["condition"], row["task"]): float(row["pooled_wilson_high"])
        - float(row["pooled_success_rate"])
        for row in task_rows
    }

    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(conditions))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for index, condition in enumerate(conditions):
        heights = [values[(condition, task)] for task in tasks]
        lower = [lows[(condition, task)] for task in tasks]
        upper = [highs[(condition, task)] for task in tasks]
        ax.bar(x + index * width, heights, width, label=condition, yerr=[lower, upper], capsize=2)
    ax.set_xticks(x + width * (len(conditions) - 1) / 2, tasks, rotation=15, ha="right")
    ax.set_ylabel("Native success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "success_by_task.png", dpi=220)
    fig.savefig(output / "success_by_task.svg")
    plt.close(fig)

    latency_by_condition: dict[str, list[float]] = defaultdict(list)
    peak_by_condition: dict[str, list[float]] = defaultdict(list)
    for row in latency_rows:
        latency_by_condition[row["condition"]].append(float(row["end_to_end_p50_ms"]))
        if row.get("peak_gb"):
            peak_by_condition[row["condition"]].append(float(row["peak_gb"]))
    success_by_condition = {
        condition: float(
            np.mean([values[(condition, task)] for task in tasks if (condition, task) in values])
        )
        for condition in conditions
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in conditions:
        latencies = latency_by_condition.get(condition, [])
        if not latencies:
            continue
        latency = float(np.mean(latencies))
        peak = float(np.mean(peak_by_condition[condition])) if peak_by_condition[condition] else 0.0
        ax.scatter(latency, success_by_condition[condition], s=60 + 25 * peak)
        ax.annotate(condition, (latency, success_by_condition[condition]), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("End-to-end action-chunk latency, p50 (ms)")
    ax.set_ylabel("Mean native success across tasks")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "success_latency.png", dpi=220)
    fig.savefig(output / "success_latency.svg")
    plt.close(fig)

    manifest = {
        "tasks": tasks,
        "conditions": conditions,
        "figures": ["success_by_task.png", "success_by_task.svg", "success_latency.png", "success_latency.svg"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
