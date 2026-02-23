# Use FastVLA from LeRobot

このドキュメントは、`VLA-from-FastVLM` の `fastvla` ポリシーを `lerobot-train` から使うための最短手順をまとめたものです。

## 1. 何をしているか

`policy.type=fastvla` は、以下の構成で動作します。

- Backbone: FastVLM (`LlavaQwen2ForCausalLM`, `model_type=llava_qwen2`)
- Head: state + vision 特徴を結合する軽量 MLP action head
- LeRobot 側 integration: plugin (`--policy.discover_packages_path`) で動的登録

`llava_qwen2` は純正 FastVLM の正しい形式です。書き換えは不要です。

## 2. 前提

- LeRobot 環境: `/home/syun/lerobot/.venv`
- この repo: `/home/syun/VLA-from-FastVLM`
- 例のローカル checkpoint:
  - `/home/syun/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3`

## 3. 依存関係をそろえる（LeRobot 側 venv）

```bash
source /home/syun/lerobot/.venv/bin/activate

pip install -U \
  "transformers>=4.57.1,<5.0.0" \
  "timm>=1.0.0" \
  "tyro" \
  "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0"
```

メモ:

- `huggingface-hub` は `lerobot` 要件に合わせて `<0.36.0` に固定してください。
- `HF_USER` はクォート無しで設定してください。

```bash
export HF_USER=syun88
```

## 4. Plugin を見つけられるようにする

```bash
export PYTHONPATH=/home/syun/VLA-from-FastVLM/src:${PYTHONPATH}
```

## 5. まずは smoke test（10 steps）

Hub push を切って、最小で起動確認します。

```bash
lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.vlm_model_name=/home/syun/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
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

## 6. 本番学習コマンド例

```bash
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

## 7. 重要フラグの意味

- `--policy.discover_packages_path`
  - LeRobot に plugin パッケージを import させます。
- `--policy.type=fastvla`
  - plugin 側で登録した policy type を使用します。
- `--policy.vlm_model_name`
  - 実際に読み込む FastVLM checkpoint（ローカルパス可）。
- `--policy.bootstrap_model_name`
  - `llava_qwen2` ローカル checkpoint で `auto_map` が不足する場合のブートストラップ用 model id。
- `--policy.push_to_hub=false`
  - 検証時に Hub へ push しないための安全設定。

## 8. よくあるエラー

### `No module named 'transformers'`

LeRobot 側 venv に依存が入っていません。セクション 3 の `pip install` を実行してください。

### `yaml.parser.ParserError` with `"syun88"/repo`

`HF_USER` の中にクォートが入っています。以下で再設定してください。

```bash
export HF_USER=syun88
```

### `huggingface-hub ... is incompatible`

LeRobot 要件とのバージョン不一致です。`huggingface-hub<0.36.0` に戻してください。

### plugin import は通るが学習時に model load 失敗

- `--policy.vlm_model_name` が正しいディレクトリか確認
- `config.json` が存在するか確認
- `--policy.bootstrap_model_name` を `apple/FastVLM-0.5B` など有効な FastVLM repo に設定

## 9. 参考: plugin 実装の場所

- `src/vla_fastvlm/lerobot_fastvla/configuration_fastvla.py`
- `src/vla_fastvlm/lerobot_fastvla/modeling_fastvla.py`
- `src/vla_fastvlm/lerobot_fastvla/processor_fastvla.py`
- `src/vla_fastvlm/model/fastvlm_adapter.py` (llava_qwen2 bootstrap loader)

