# VLA-from-FastVLM

Vision-Language-Action stack that extends Apple's FastVLM models with an action head for robotic manipulation. The pipeline fine-tunes FastVLM on the [lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) dataset and runs the resulting policy inside the `gym_aloha` simulator or on real Aloha hardware.

## Highlights
- ✅ **End-to-end training**: fine-tune FastVLM on Aloha insertion demonstrations with a custom PyTorch trainer built on top of `accelerate`.
- 🖥️ **Automatic device selection**: CUDA → MPS → CPU fallback with manual override via the `FASTVLM_FORCE_DEVICE` environment variable.
- 🕹️ **Simulation-ready**: roll out policies in MuJoCo-powered Aloha simulator with optional RGB video export.
- 🤖 **Real robot loop**: structured runner composed with a pluggable hardware interface for live control without ROS2.
- 🔁 **Future fine-tuning**: reproducible configs and checkpoint format (state dict + JSON config) for continued training or evaluation.

## Repository Layout

```
.
├── configs/                # Ready-to-use configuration files
├── scripts/                # CLI entry-points for training, evaluation, sim, and real control
├── src/vla_fastvlm/        # Core Python package
│   ├── data/               # Hugging Face dataset wrappers
│   ├── model/              # FastVLM backbone adapter + policy head
│   ├── training/           # Accelerate-based trainer
│   ├── sim/                # gym_aloha rollout helpers
│   ├── real/               # Real robot control loop abstractions
│   └── utils/              # Logging, checkpoint utilities, device helpers
└── pyproject.toml
```

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sim,real]"
```

> **Note**  
> - `sim` extra installs `gym-aloha` and video writers.  
> - `real` extra installs serial/HID dependencies for live hardware control.

## 2. Prepare the Dataset

The project trains on the authoritative dataset:

```bash
# Downloads and caches the dataset locally
python - <<'PY'
from datasets import load_dataset
load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
PY
```

No dummy data is created—the scripts stream real episodes directly from the Hugging Face Hub.

## 3. Fine-tune FastVLM

`scripts/train.py` exposes all relevant knobs. Reproduce Apple π₀-style command:

```bash
python scripts/train.py \
  --output_dir=outputs/train/fastvlm_aloha \
  --model_id=apple/FastVLM-base \
  --dataset_repo_id=lerobot/aloha_sim_insertion_human_image \
  --batch_size=2 \
  --num_workers=2 \
  --num_epochs=10 \
  --mixed_precision=bf16
```

Training artifacts:

- `training_config.json`: frozen trainer configuration.
- `checkpoints/step-*/policy_config.json`: serialized `FastVLMPolicyConfig`.
- `checkpoints/step-*/policy_state_dict.pt`: PyTorch weights.

Continue fine-tuning by pointing `--checkpoint_dir` to any previous checkpoint and using `--resume_from` in `TrainingConfig`.

## 4. Offline Evaluation (Dataset MSE)

```bash
python scripts/eval_dataset.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset_repo_id=lerobot/aloha_sim_insertion_human_image \
  --split=validation
```

## 5. Simulation Rollout

Ensure MuJoCo ≥ 3.1 is installed and `MUJOCO_GL=egl` (set automatically by the script). Then:

```bash
python scripts/run_sim.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --num_episodes=10 \
  --video_dir=outputs/eval/pi0_aloha
```

The command spins up `gym_aloha/AlohaInsertion-v0`, renders each episode, and writes MP4 files into the requested directory.

## 6. Real Robot Loop (Future Hardware Use)

Implement `AlohaHardwareInterface` with your I/O stack (e.g., Dynamixel bus or a custom bridge) and pass the dotted class path to the runner:

```python
# my_robot/interface.py
from vla_fastvlm.real import AlohaHardwareInterface
import numpy as np

class CustomAlohaInterface(AlohaHardwareInterface):
    ...
```

```bash
python scripts/run_real.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --interface_cls=my_robot.interface.CustomAlohaInterface \
  --interface_kwargs_json="{\"port\":\"/dev/ttyUSB0\",\"camera_topic\":\"camera/top\"}"
```

The runner opens the hardware connection, repeatedly queries the latest observation (`state` + RGB image), predicts the next joint command through the FastVLM policy, and streams it back to the robot at the configured control frequency.

## 7. Device Selection

Device choice is automatic:

1. CUDA GPU (if available).
2. Apple MPS backend.
3. CPU fallback.

Override by exporting `FASTVLM_FORCE_DEVICE=cpu` or `FASTVLM_FORCE_DEVICE=cuda`. Individual scripts also expose `--device_preference` to pin the runtime explicitly.

## 8. Configuration Files

- `configs/train_aloha.yaml`: baseline hyperparameters matching the README steps. Use it as a reference when overriding CLI flags (for example, load it with a short Python snippet to expand into `--key=value` pairs).

## 9. Roadmap

- ✅ Simulation control loop and dataset fine-tuning.
- 🔜 Integrate hardware-specific drivers for Aloha stations.
- 🔜 Add evaluation suites for success-rate scoring and rollout logging.
- 🔜 Extend to additional tasks (transfer cube, towel) and continuous learning.

---

Feel free to open issues for enhancements or share checkpoints through the Hugging Face Hub under your organization.
