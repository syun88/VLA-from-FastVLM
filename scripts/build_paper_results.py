#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


@dataclass
class Args:
    results_dir: str = "results/confirmatory"
    paper_dir: str = "paper"


def main(args: Args) -> None:
    results = Path(args.results_dir)
    output = Path(args.paper_dir) / "generated" / "results.tex"
    task_rows = _read_csv(results / "task_summary.csv")
    latency_rows = _read_csv(results / "latency.csv")
    offline_rows = _read_csv(results / "offline.csv")

    task_table = [
        r"\subsection{Closed-loop Meta-World results}",
        r"The table reports pooled held-out episodes, successes, and Wilson 95\% intervals across confirmatory training seeds.",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Condition & Task & Seeds & Episodes & Success & Wilson 95\% \\",
        r"\midrule",
    ]
    for row in task_rows:
        task_table.append(
            f"{_tex(row['condition'])} & {_tex(row['task'])} & {row['training_seeds']} & "
            f"{row['episodes']} & {row['successes']} ({100*float(row['pooled_success_rate']):.1f}\\%) & "
            f"[{100*float(row['pooled_wilson_low']):.1f}, {100*float(row['pooled_wilson_high']):.1f}] \\\\"
        )
    task_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    efficiency_table = [
        r"\subsection{Efficiency}",
        r"Latency is the synchronized end-to-end observation-to-action-chunk time on the recorded local hardware; amortized action time divides by the action-chunk size.",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Condition & Resolution & Visual tokens & Cold load (s) & Chunk p50 (ms) & Chunk p95 (ms) & Peak MLX (GB) \\",
        r"\midrule",
    ]
    for row in latency_rows:
        efficiency_table.append(
            f"{_tex(row['condition'])} & {row['resolution']} & {row.get('visual_token_count', '--')} & {float(row['cold_load_seconds']):.2f} & "
            f"{float(row['end_to_end_p50_ms']):.1f} & {float(row['end_to_end_p95_ms']):.1f} & "
            f"{float(row['peak_gb']):.2f} \\\\"
        )
    efficiency_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    offline_by_condition: dict[str, list[float]] = defaultdict(list)
    for row in offline_rows:
        offline_by_condition[row["condition"]].append(float(row["normalized_rmse"]))
    offline_lines = [r"\subsection{Held-out action error}"]
    for condition, values in sorted(offline_by_condition.items()):
        mean = sum(values) / len(values)
        offline_lines.append(
            f"{_tex(condition)}: mean normalized action RMSE {mean:.5f} across {len(values)} task/seed entries."
        )

    content = "\n".join(task_table + efficiency_table + offline_lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    manifest = {
        "task_rows": len(task_rows),
        "latency_rows": len(latency_rows),
        "offline_rows": len(offline_rows),
        "output": str(output),
    }
    (Path(args.paper_dir) / "generated" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
