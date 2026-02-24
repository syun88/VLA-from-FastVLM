# VLA-from-FastVLM

Japanese guide: [`README_JP.md`](README_JP.md)

`VLA-from-FastVLM` adapts Apple FastVLM checkpoints to a VLA policy and exposes:
- A LeRobot plugin policy: `policy.type=fastvla`
- Standalone training/eval scripts (`scripts/train.py`, `scripts/eval_dataset.py`)

This README is the current workflow. Old commands were removed.

## 1. Recommended workflow (LeRobot + custom FastVLM)

### 1.1 Prerequisites

- `lerobot` repo (with its own virtualenv)
- This repo: `VLA-from-FastVLM`
- A local FastVLM checkpoint directory, for example:
  - `/home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3`

### 1.2 Install compatible dependencies in LeRobot venv

```bash
source /home/<you>/lerobot/.venv/bin/activate
python -m pip install --upgrade pip

pip install -U \
  "transformers>=4.57.1,<5.0.0" \
  "timm>=1.0.0" \
  "tyro" \
  "metaworld" \
  "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0"
```

Why pin `huggingface-hub<0.36.0`?  
LeRobot `0.4.4` requires `<0.36.0`, and newer versions create dependency conflicts.

### 1.3 Make the plugin importable

```bash
export PYTHONPATH=/home/<you>/VLA-from-FastVLM/src:${PYTHONPATH}
```

### 1.4 Set HF user correctly

```bash
export HF_USER=your_hf_username
```

Do not include quotes in the value.  
`HF_USER="name"` can break CLI parsing for `--policy.repo_id=${HF_USER}/...`.

### 1.5 Download official Apple checkpoints

```bash
cd /home/<you>/VLA-from-FastVLM
bash scripts/download_fastvlm.sh
```

By default this script downloads `llava-fastvithd_0.5b_stage3`.

### 1.6 Smoke test (no Hub push)

Run from the LeRobot repo:

```bash
cd /home/<you>/lerobot

lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.vlm_model_name=/home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_0.5b_stage3 \
  --policy.bootstrap_model_name=apple/FastVLM-0.5B \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs_fastvla_smoke \
  --steps=10 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000
```

### 1.7 Full training example

```bash
cd /home/<you>/lerobot

lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.vlm_model_name=/home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
  --policy.bootstrap_model_name=apple/FastVLM-0.5B \
  --policy.repo_id=${HF_USER}/metaworld-fastvla-test \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs \
  --steps=100000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000
```

## 2. `llava_qwen2` and `LlavaQwen2ForCausalLM` (important)

`model_type=llava_qwen2` is correct for official FastVLM checkpoints.

You should not rewrite this in `config.json`.

Why `--policy.bootstrap_model_name` is still needed:
- Some local Stage2/Stage3 folders are missing metadata like `auto_map`.
- In that case, loading directly with `AutoModel...from_pretrained(local_path)` can fail.
- This project uses a bootstrap config from `apple/FastVLM-0.5B` to resolve classes, then loads your local weights.

So the common setup is:
- `--policy.vlm_model_name=/path/to/local/FastVLM`
- `--policy.bootstrap_model_name=apple/FastVLM-0.5B`

## 3. Standalone workflow (without LeRobot)

### 3.1 Install

```bash
cd /home/<you>/VLA-from-FastVLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 3.2 Train

```bash
python scripts/train.py \
  --output-dir outputs/train/aloha_fastvlm \
  --model-id /home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
  --bootstrap-model-id apple/FastVLM-0.5B \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --num-epochs 5 \
  --batch-size 4
```

### 3.3 Evaluate

```bash
python scripts/eval_dataset.py \
  --checkpoint-dir outputs/train/aloha_fastvlm/checkpoints/step-1000 \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --split validation
```

`eval_dataset.py` falls back to `train` when `validation` does not exist (default: `--allow-missing-split=true`).

## 4. Key flags

| Flag | Meaning |
| --- | --- |
| `--policy.discover_packages_path` | Imports plugin package (`vla_fastvlm.lerobot_fastvla`). |
| `--policy.type=fastvla` | Selects this repo's LeRobot policy wrapper. |
| `--policy.vlm_model_name` | Local or Hub FastVLM checkpoint to use as backbone. |
| `--policy.bootstrap_model_name` | Hub model used only to bootstrap missing `llava_qwen2` metadata. |
| `--policy.push_to_hub=false` | Disable Hub upload during smoke tests. |

## 5. Troubleshooting

- `No module named 'transformers'`
  - Install dependencies in the active LeRobot venv (`source .../lerobot/.venv/bin/activate`).
- `huggingface-hub ... incompatible`
  - Reinstall with `huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0`.
- `yaml.parser.ParserError` near `"user"/repo`
  - `HF_USER` likely contains quotes. Use `export HF_USER=user` (no quotes).
- `model type 'llava_qwen2' ... not recognize this architecture`
  - Keep `--policy.bootstrap_model_name=apple/FastVLM-0.5B` and confirm `config.json` exists in local checkpoint dir.
- `Split 'validation' not found`
  - Expected for some datasets. Use `--split train` or keep default fallback.

## 6. Repository layout

```text
.
├── scripts/
│   ├── download_fastvlm.sh
│   ├── train.py
│   └── eval_dataset.py
├── src/vla_fastvlm/
│   ├── lerobot_fastvla/     # LeRobot plugin entrypoint (policy.type=fastvla)
│   ├── fastvla/             # Core config/model for FastVLA
│   └── model/               # FastVLM adapter (llava_qwen2 bootstrap fallback)
└── USE_WITH_LEROBOT_FASTVLA.md
```

## 7. License

- This repository code: MIT (`LICENSE`)
- Apple FastVLM models/assets: Apple licenses in `third_party/ml-fastvlm`
- Model weights are not bundled; download and use under Apple license terms
