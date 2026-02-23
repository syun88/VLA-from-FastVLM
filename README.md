# VLA-from-FastVLM

日本語版ガイドは [`README_JP.md`](README_JP.md) を参照してください。

End-to-end Vision-Language-Action (VLA) pipeline that fine-tunes Apple’s FastVLM models on the [lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) dataset. The project focuses on offline training/evaluation with a simple FastVLM adapter—no simulator dependency.

---

## Quick Start

```bash
# 1) Environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

# 2) (Optional) Cache the dataset locally
python - <<'PY'
from datasets import load_dataset
load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
PY

# 3) Fine-tune FastVLM (10 epochs by default)
python scripts/train.py \
  --output-dir outputs/train/fastvlm_aloha \
  --model-id "$FASTVLM_BACKBONE_PATH/llava-fastvithd_0.5b_stage3" \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --batch-size 2 \
  --num-workers 0 \
  --num-epochs 10 \
  --max-steps 1000 \
  --image-size 512 \
  --resize-with-padding true \
  --tokenizer-max-length 64

# 4) Evaluate MSE on the dataset (falls back to train split automatically)
python scripts/eval_dataset.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
  --split=validation
```

Each CLI has `--help` powered by `tyro`; all flags use kebab-case (`--checkpoint-dir`, `--num-epochs`, …).

---

## Requirements

- Python ≥ 3.10 with PyTorch ≥ 2.1 (CUDA, MPS, or CPU).
- No MuJoCo / simulator packages are required.
- Optional: Apple MPS backend (macOS 13+) or NVIDIA GPU.

---

## Project Layout

```
.
├── configs/                # Example YAML configs
├── scripts/                # CLI entry points: train / eval
├── src/vla_fastvlm/        # Python package
│   ├── data/               # Hugging Face dataset wrappers
│   ├── model/              # FastVLM adapter and policy head
│   ├── fastvla/            # FastVLM → VLA components (config / processor / model)
│   ├── training/           # Accelerate-based trainer & config dataclasses
│   └── utils/              # Logging, checkpoint, and device helpers
└── outputs/                # Default artifacts (checkpoints, logs)
```

---

## Dataset Notes

- The public Aloha dataset only exposes a `train` split. `eval_dataset.py` defaults to `--allow-missing-split` so `--split=validation` gracefully falls back to `train` and prints a notice. Disable with `--allow-missing-split=false` if you prefer hard failures.
- Use `--streaming` to iterate from the Hub without local downloads. Both `AlohaDataset` and `AlohaIterableDataset` accept `--limit-samples` (offline only) for quicker smoke tests.
- Set `HF_HOME` to relocate the Hugging Face cache. The legacy `TRANSFORMERS_CACHE` warning can be ignored or replaced with `HF_HOME`.

---

## FastVLM model format (`llava_qwen2`)

- The official FastVLM checkpoints use `model_type: llava_qwen2`. This is expected and should not be changed manually.
- Some local Stage2/Stage3 archives do not include enough metadata for direct `AutoModel` loading.
- This project now supports a bootstrap fallback: if your local checkpoint is `llava_qwen2` and missing `auto_map`, set:
  - `--model-id /path/to/your/local/checkpoint`
  - `--bootstrap-model-id apple/FastVLM-0.5B` (or another compatible FastVLM repo)
- The bootstrap repo is only used to import the correct config/model classes; your local weights remain the source of truth.

---

## Use with `lerobot-train`

This repository now exposes a LeRobot plugin policy type: `fastvla`.

1. Ensure your LeRobot environment has compatible deps (`transformers`, `timm`, etc.).
2. Add this repo to `PYTHONPATH` so LeRobot can import the plugin package.
3. Pass `--policy.discover_packages_path=vla_fastvlm.lerobot_fastvla`.

Example:

```bash
source /home/syun/lerobot/.venv/bin/activate
export PYTHONPATH=/home/syun/VLA-from-FastVLM/src:${PYTHONPATH}

lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.vlm_model_name=/home/syun/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
  --policy.bootstrap_model_name=apple/FastVLM-0.5B \
  --policy.repo_id=${HF_USER}/metaworld-fastvla-test \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs/ \
  --steps=100000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000
```

If you do not want Hub upload, add `--policy.push_to_hub=false` (or provide `--policy.repo_id`).

---

## Training (`scripts/train.py`)

- Hyperparameters live in the `TrainArgs` dataclass. Defaults: 10 epochs, batch size 4, streaming disabled.
- Image preprocessing offers letterboxing (`--resize-with-padding`) and tokenizer controls (`--tokenizer-max-length`, `--pad-to-max-length`, `--tokenizer-padding-side`).
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

## Preprocessing Reference

- Letterboxing is enabled in `FastVLMBackbone` by default. Disable with `--resize-with-padding=false` to revert to simple stretching.
- Tokenizer defaults (`--tokenizer-max-length 64`, right padding) keep prompts compact for FastVLM.

---

## CLI Reference

| Script | Purpose | Notable Flags |
| ------ | ------- | ------------- |
| `scripts/train.py` | Fine-tune FastVLM | `--output-dir`, `--model-id`, `--bootstrap-model-id`, `--dataset-repo-id`, `--num-epochs`, `--max-steps`, `--image-size`, `--resize-with-padding`, `--tokenizer-max-length` |
| `scripts/eval_dataset.py` | Offline dataset MSE | `--checkpoint-dir`, `--split`, `--allow-missing-split`, `--limit-samples`, `--streaming` |

Run any script with `--help` to view the full schema.

---

## Troubleshooting

- **Dataset split missing**: keep the default `--allow-missing-split` or use `--split=train`.
- **Slow image processor warnings**: upgrade to the latest fast processors or set `TRANSFORMERS_USE_FAST=1`. The fallback resizer keeps inference stable.
- **`pin_memory` warning on MPS**: expected; Apple’s backend ignores pinned memory for now.
- **OOM with large images**: reduce `--image-size` or disable padding.
- **Cache location**: set `export HF_HOME=/path/to/cache` to silence `TRANSFORMERS_CACHE` deprecation warnings.

---

## Licensing

- Repository code: MIT License (`LICENSE`).
- Apple FastVLM assets: governed by Apple’s licenses (`third_party/ml-fastvlm`).
- Model weights are not bundled; download from Apple’s Model Zoo and respect `LICENSE_MODEL`/`ACKNOWLEDGEMENTS`.

---

Pull requests, issues, and dataset findings are always welcome. Share checkpoints on the Hugging Face Hub if you adapt the policy to new tasks!
