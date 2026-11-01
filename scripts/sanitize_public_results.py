#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from vla_fastvlm.paths import workspace_root

PRIVATE_KEYS = {"wandb_url", "wandb_entity"}


def _public_path(value: str) -> str:
    marker = "/checkpoints/fastvla/"
    if marker in value:
        return "$FASTVLA_WORK_ROOT/checkpoints/fastvla/" + value.split(marker, 1)[1]
    marker = "/dataset/fastvla/"
    if marker in value:
        return "$FASTVLA_WORK_ROOT/dataset/fastvla/" + value.split(marker, 1)[1]
    value = value.replace(str(Path.home()), "$HOME")
    value = value.replace(str(workspace_root()), "$FASTVLA_WORK_ROOT")
    return value


def _scrub(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            if child_key in PRIVATE_KEYS:
                continue
            if child_key == "base_model_path" and "$FASTVLA_WORK_ROOT" in str(child_value):
                child_value = "$FASTVLA_FASTVLM_MLX"
            elif child_key in {"cache_root", "dataset_root", "output_dir", "output_root", "parquet"}:
                child_value = _public_path(str(child_value))
            result[child_key] = _scrub(child_value, child_key)
        return result
    if isinstance(value, list):
        return [_scrub(item, key) for item in value]
    if isinstance(value, str) and key in {"checkpoint", "base_model_path", "cache_root", "dataset_root", "output_dir", "output_root", "parquet"}:
        return _public_path(value)
    return value


def _rewrite_json(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clean = _scrub(payload)
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")


def _rewrite_jsonl(path: Path) -> None:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lines.append(json.dumps(_scrub(json.loads(line)), separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class Args:
    results_dir: str = "results"


def main(args: Args) -> None:
    root = Path(args.results_dir)
    for path in sorted(root.rglob("*")):
        if path.name == ".DS_Store":
            path.unlink()
        elif path.suffix == ".json":
            _rewrite_json(path)
        elif path.suffix == ".jsonl":
            _rewrite_jsonl(path)
    print(f"sanitized public result files under {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
