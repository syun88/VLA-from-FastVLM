pip install lerobot 
pip install metaworld
pip install packaging
python - <<'PY'
import lerobot, pathlib
p = pathlib.Path(lerobot.__file__).parent / "envs" / "metaworld_config.json"
print(p)
PY

pip install num2words
pip install transformers

# wandb を使う場合（環境変数 WANDB_API_KEY を設定してから実行）
# 例: report_to=["wandb"] または ["tensorboard","wandb"]
# python scripts/train.py --report-to wandb

# FastVLA (this repo) fine-tuning via LeRobot CLI
# Replace ${HF_USER}/fastvla-metaworld with your Hugging Face repo for checkpoints.
./.venv/bin/lerobot-train \
  --policy.type=fastvla \
  --policy.repo_id=${HF_USER}/fastvla-metaworld \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs/ \
  --steps=100000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --wandb.enable=true\
