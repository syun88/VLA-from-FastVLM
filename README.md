# VLA-from-FastVLM

日本語版ガイドは [`README_JP.md`](README_JP.md) を参照してください。

End-to-end Vision-Language-Action (VLA) pipeline that fine-tunes Apple’s FastVLM models on the [lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) dataset, evaluates on held-out demonstrations, and rolls policies inside the `gym_aloha` MuJoCo simulator or on real Aloha hardware.

---

## Quick Start

```bash
# 1) Environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[sim,real]"

# 2) (Optional) Cache the dataset locally
python - <<'PY'
from datasets import load_dataset
load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
PY

# 3) Fine-tune FastVLM (10 epochs by default)
# python scripts/train.py \
#   --output-dir=outputs/train/fastvlm_aloha \
#   --model-id=apple/FastVLM-base \
#   --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
#   --batch-size=2 --num-workers=2 --num-epochs=10 --mixed-precision=bf16
# or
python scripts/train.py \
  --output-dir outputs/train/fastvlm_aloha \
  --model-id "$FASTVLM_BACKBONE_PATH/llava-fastvithd_0.5b_stage3" \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --batch-size 2 \
  --num-workers 0 \
  --num-epochs 10 \
  --max-steps 1000

# 4) Evaluate MSE on the dataset (falls back to train split automatically)
python scripts/eval_dataset.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
  --split=validation

# 5) Run MuJoCo simulation rollouts with MP4 logging
python scripts/run_sim.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --num-episodes=10 \
  --video-dir=outputs/eval/pi0_aloha
```

Each CLI has `--help` powered by `tyro`; all flags use kebab-case (`--checkpoint-dir`, `--num-epochs`, …).

---

## Requirements

- Python ≥ 3.10 with PyTorch ≥ 2.1 (CUDA, MPS, or CPU).
- MuJoCo 3.1+ for simulation (`MUJOCO_GL` is auto-selected per OS).
- `gym-aloha` ≥ 0.1.3 (installed through the `[sim]` extra).
- Optional: Apple MPS backend (macOS 13+) or NVIDIA GPU.
- For real hardware: implement a subclass of `AlohaHardwareInterface` and install the `[real]` extra (serial / HID drivers).

---

## Project Layout

```
.
├── configs/                # Example YAML configs
├── scripts/                # CLI entry points: train / eval / sim / real
├── src/vla_fastvlm/        # Python package
│   ├── data/               # Hugging Face dataset wrappers
│   ├── model/              # FastVLM adapter and policy head
│   ├── training/           # Accelerate-based trainer & config dataclasses
│   ├── sim/                # MuJoCo rollout helpers
│   ├── real/               # Real robot loop abstractions
│   └── utils/              # Logging, checkpoint, and device helpers
└── outputs/                # Default artifacts (checkpoints, eval videos, logs)
```

---

## Dataset Notes

- The public Aloha dataset only exposes a `train` split. `eval_dataset.py` defaults to `--allow-missing-split` so `--split=validation` gracefully falls back to `train` and prints a notice. Disable with `--allow-missing-split=false` if you prefer hard failures.
- Use `--streaming` to iterate from the Hub without local downloads. Both `AlohaDataset` and `AlohaIterableDataset` accept `--limit-samples` (offline only) for quicker smoke tests.
- Set `HF_HOME` to relocate the Hugging Face cache. The legacy `TRANSFORMERS_CACHE` warning can be ignored or replaced with `HF_HOME`.

---

## Training (`scripts/train.py`)

- Hyperparameters live in the `TrainArgs` dataclass. Defaults: 10 epochs, batch size 4, streaming disabled.
- `num_epochs` is the stopping criterion when `max_steps` is `null`; adjust either flag to control runtime.
- Outputs under `outputs/train/<run_name>/`:
  - `training_config.json`: frozen `TrainingConfig` including `num_epochs`/`max_steps`.
  - `checkpoints/step-*/`: `policy_state_dict.pt` (PyTorch weights) and `policy_config.json`.
  - TensorBoard events (`--report-to tensorboard`). Launch with `tensorboard --logdir outputs/train/fastvlm_aloha`.
- Resume from checkpoints by passing `--checkpoint-dir <previous_run>/checkpoints/step-XXXX --resume-from step-XXXX`.

---

## Evaluation (`scripts/eval_dataset.py`)

- Computes per-action MSE on a dataset split using the frozen policy.
- Batch-related flags mirror training (`--batch-size`, `--num-workers`, `--streaming`).
- On Apple MPS you may see `pin_memory` warnings; they are harmless because pinned memory is CPU-only today.
- Sample output (from `eval_log.txt`):
  ```
  [eval_dataset] Split 'validation' not found; using 'train' instead.
  MSE on split 'train': 0.004554
  ```
- For faster smoke tests, combine `--limit-samples` and `--batch-size 1`.

---

## Simulation (`scripts/run_sim.py`)

- Wraps `gym_aloha/AlohaInsertion-v0` with auto-rendering, MP4 export, and device selection.
- Rendering backend defaults per OS: `egl` (Linux), `glfw` (macOS), `d3d11` (Windows). Override with `MUJOCO_GL`.
- The FastVLM adapter resizes simulator frames to ≥512 px using either the model’s image processor or a torchvision fallback. Expect a one-time warning if the fallback activates.
- Key flags: `--num-episodes`, `--max-episode-steps`, `--task-instruction`, `--video-dir`, `--record-video=false`.
- Result summary:
  ```
  Episode 000: reward=5.10 steps=287 success=✅
  ```

---

## Real Robot Runner (`scripts/run_real.py`)

- Provide an `AlohaHardwareInterface` subclass (e.g. `my_robot.interface.CustomAlohaInterface`) and optional JSON kwargs via `--interface-kwargs-json`.
- The runner opens the hardware connection, streams observations through the policy, and prints episode statistics.
- The same device selection logic applies (`--device-preference`, `FASTVLM_FORCE_DEVICE`).

---

## CLI Reference

| Script | Purpose | Notable Flags |
| ------ | ------- | ------------- |
| `scripts/train.py` | Fine-tune FastVLM | `--output-dir`, `--model-id`, `--dataset-repo-id`, `--num-epochs`, `--max-steps`, `--mixed-precision` |
| `scripts/eval_dataset.py` | Offline dataset MSE | `--checkpoint-dir`, `--split`, `--allow-missing-split`, `--limit-samples`, `--streaming` |
| `scripts/run_sim.py` | MuJoCo rollouts with optional video | `--checkpoint-dir`, `--num-episodes`, `--max-episode-steps`, `--video-dir`, `--device-preference` |
| `scripts/run_real.py` | Real robot control loop | `--checkpoint-dir`, `--interface-cls`, `--interface-kwargs-json`, `--task-instruction` |

Run any script with `--help` to view the full schema.

---

## Troubleshooting

- **Dataset split missing**: keep the default `--allow-missing-split` or use `--split=train`.
- **Slow image processor warnings**: upgrade to the latest fast processors or set `TRANSFORMERS_USE_FAST=1`. The fallback resizer keeps inference stable.
- **`pin_memory` warning on MPS**: expected; Apple’s backend ignores pinned memory for now.
- **MuJoCo EGL errors on macOS**: the script now auto-selects `glfw`. Override manually if you prefer another backend.
- **`gym_aloha` import errors**: ensure `pip install -e ".[sim]"` succeeded and MuJoCo is ≥ 3.1.
- **Cache location**: set `export HF_HOME=/path/to/cache` to silence `TRANSFORMERS_CACHE` deprecation warnings.

---

## Licensing

- Repository code: MIT License (`LICENSE`).
- Apple FastVLM assets: governed by Apple’s licenses (`third_party/ml-fastvlm`).
- Model weights are not bundled; download from Apple’s Model Zoo and respect `LICENSE_MODEL`/`ACKNOWLEDGEMENTS`.

---

Pull requests, issues, and dataset findings are always welcome. Share checkpoints on the Hugging Face Hub if you adapt the policy to new tasks!
