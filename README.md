# VLA-from-FastVLM

[![CI](https://github.com/syun88/VLA-from-FastVLM/actions/workflows/ci.yml/badge.svg?branch=research/mlx-fastvla)](https://github.com/syun88/VLA-from-FastVLM/actions/workflows/ci.yml)

CI runs on the pinned ARM64 `macos-15` runner because MLX is an Apple-Silicon runtime. Tagged
releases build and validate the Python distributions and attach them to a GitHub Release; model
weights remain governed by the separate Apple license and are not published by that workflow.

MLX-native research on whether Apple's FastVLM is an effective and efficient backbone for
robotic Vision-Language-Action learning. The active system runs genuine FastVLM image and
language computation on Apple Silicon, trains an action-chunk head and optional language LoRA
with MLX, and evaluates policies in local Meta-World/MuJoCo episodes.

Japanese guide: [`README_JP.md`](README_JP.md)

## Why this matters

FastVLM's motivation is efficient high-resolution vision-language inference. Those gains are
usually reported for VLM time-to-first-token, not manipulation. This project measures whether
the speed and visual-token compression survive adaptation to a VLA, while testing the spatial
information and success-rate costs on controlled robot tasks.

The code is constructed around one scientific question with explicit negative outcomes allowed:

> Under controlled data, action horizon, and evaluation budgets, does FastVLM provide a better
> manipulation accuracy-latency-memory trade-off than state-only, FastViTHD-feature, and
> lightweight-VLM alternatives?

## Current implementation

- Native `mlx.core` / `mlx.nn` / `mlx.optimizers` training and inference for FastVLA.
- Official `apple/FastVLM-0.5B` checkpoint, pinned by revision and converted to MLX format.
- Configurable task schemas for cameras, proprioceptive state, action dimensions, normalization,
  control frequency, action horizon, and future multi-arm adapters.
- 8-step Behavior Cloning action chunks with episode-boundary masks.
- Frozen FastVLM baseline with cached projected FastViTHD features.
- Real MLX LoRA updates on FastVLM Qwen2 attention/MLP projections plus the action head.
- State-only, spatial-only, global-only, and frozen/LoRA ablations.
- Local Meta-World closed-loop evaluation with native success, failure categories, stability,
  generalization seeds, nominal/counterfactual instructions, and latency distributions.
- W&B provenance plus resumable local checkpoints under configurable artifact storage.

The original PyTorch implementation is preserved under `legacy/torch_v01/` and is not imported
or supported by the active package.

## Roadmap

- [x] Audit repository, local assets, upstream FastVLM/MLX-VLM, licenses, W&B, and Hugging Face.
- [x] Establish a pinned MLX FastVLM load/forward path and episode-disjoint Meta-World splits.
- [x] Implement MLX action-head/LoRA training, checkpointing, tracking, and simulator adapters.
- [ ] Complete frozen/LoRA training and all controlled baselines with three seeds.
- [ ] Run confirmatory closed-loop evaluations and counterfactual instruction tests.
- [ ] Publish permitted checkpoint/adapter release and reproducible model card.
- [ ] Prepare a paper draft whose tables and figures are generated from measured result files.

This roadmap is not a claim that unchecked items are complete.

## Setup

```bash
cd /path/to/VLA-from-FastVLM
export FASTVLA_WORK_ROOT=/path/to/local/fastvla-storage
export HF_HOME=/path/to/huggingface-cache
export FASTVLA_FASTVLM_MLX="$FASTVLA_WORK_ROOT/models/apple-fastvlm-0.5b-mlx"
export FASTVLA_DATASET_ROOT="$HF_HOME/hub/datasets--lerobot--metaworld_mt50/snapshots/a59f742d218c903328164257ecf180f9b18018a1"
uv pip install --python python3 -e '.[dev,sim]'
wandb login --verify       # credentials remain in the local credential store
hf auth whoami             # do not copy or print the token
```

The experiment defaults target this local M4 / 32 GB machine. They never require CUDA or a
cloud GPU.

## Reproduce from the verified local assets

```bash
# Offline, isolated checkpoint-format conversion from the already cached Apple weights.
python3 -m mlx_vlm convert \
  --hf-path /path/to/apple/FastVLM-0.5B/snapshot \
  --mlx-path "$FASTVLA_FASTVLM_MLX" \
  --dtype bfloat16

# Cache projected FastViTHD tokens and FastVLM hidden states; resumable every 10 samples.
python3 scripts/cache_features.py \
  --tasks button-press-topdown-v3,push-v3,reach-v3

# Frozen FastVLM + action head.
python3 scripts/train_action_head.py \
  --run-name fastvlm-frozen --backbone-mode frozen --seed 42

# Real FastVLM language LoRA + action head.
python3 scripts/train_lora.py \
  --run-name fastvlm-lora --seed 42 --max-steps 2000

# Offline held-out action error.
python3 scripts/evaluate_offline.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best"

# Closed-loop Meta-World evaluation.
python3 scripts/evaluate_sim.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --episodes 50
```

See [`docs/experimental_protocol.md`](docs/experimental_protocol.md) for comparisons and
statistical rules, and [`docs/upstream_audit.md`](docs/upstream_audit.md) for exact revisions,
local measurements, compatibility findings, and the model-license decision.

## Known limitations

- Results are simulation-only. They do not establish real-robot or Dual Scorpion performance.
- The current LoRA path adapts FastVLM's language layers; FastViTHD and the multimodal projector
  are frozen for cached-feature training. This limitation applies equally to the reported
  comparison and must not be described as full-model fine-tuning.
- The pinned Qwen2 path trains language LoRA at batch size one; action-head-only conditions can
  use larger batches.
- Meta-World data and simulator have known action-scale quirks that are reproduced from the
  dataset/environment contract rather than silently clipped or harmonized.
- Apple FastVLM weights use `apple-amlr`, not this repository's MIT source license. Model
  derivatives are research-only and subject to attribution, agreement-inclusion, modification
  disclosure, and no-endorsement requirements.
- W&B connectivity is best-effort; local result and checkpoint files remain authoritative.

## Public repository hygiene

Committed results retain run IDs and cryptographic checkpoint hashes for provenance, but omit
W&B entity names, private run URLs, home-directory paths, and machine-specific mount paths. W&B
access is resolved from the local authenticated session or `WANDB_ENTITY`. Run
`python3 scripts/sanitize_public_results.py` before publishing new result files. The unit suite
fails if private paths or named W&B run URLs reappear in tracked files.

## Repository layout

```text
configs/                         Experiment configuration references
docs/                            Upstream audit and pre-registered protocol
legacy/torch_v01/                Preserved historical PyTorch implementation
scripts/cache_features.py        Resumable MLX feature extraction
scripts/train_action_head.py     Frozen/state-only/spatial/global action-head training
scripts/train_lora.py            Genuine MLX FastVLM LoRA + action-head training
scripts/evaluate_offline.py      Episode-held-out action metrics
scripts/evaluate_sim.py          Closed-loop Meta-World evaluation
src/vla_fastvlm/fastvla/         MLX FastVLM backbone, policy config, and action head
src/vla_fastvlm/data/            LeRobot v3 scanning, splits, chunks, and cache views
src/vla_fastvlm/training/        MLX optimizers, gradients, tracking, and checkpoints
src/vla_fastvlm/evaluation/      Meta-World adapter and rollout evaluation
```

## License

- This repository's source code: MIT (`LICENSE`).
- Apple FastVLM code and weights: separate Apple licenses in `third_party/ml-fastvlm/` and the
  official checkpoint repository. The `apple-amlr` weight license controls derivatives and is
  reproduced, with attribution, in any permitted model release.
- `lerobot/metaworld_mt50`: Apache-2.0 per its dataset card. Dataset bytes are not committed.
