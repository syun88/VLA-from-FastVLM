# VLA-from-FastVLM（日本語）

英語版: [`README.md`](README.md)

`VLA-from-FastVLM` は Apple FastVLM を VLA ポリシーとして使うための実装です。

- LeRobot 用プラグイン: `policy.type=fastvla`
- 単体スクリプト: `scripts/train.py` / `scripts/eval_dataset.py`

このREADMEは現在の手順に合わせて更新済みです。古い手順は使わないでください。

## 1. 推奨手順（LeRobotで自作FastVLMを使う）

### 1.1 前提

- `lerobot` リポジトリ（専用venvあり）
- `VLA-from-FastVLM` リポジトリ
- ローカルFastVLMチェックポイント（例）
  - `/home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3`

### 1.2 LeRobot側venvに依存を入れる 

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

`huggingface-hub<0.36.0` が重要です。  
LeRobot `0.4.4` では `0.36.0` 以上と依存衝突します。

### 1.3 プラグインをimport可能にする

```bash
export PYTHONPATH=/home/<you>/VLA-from-FastVLM/src:${PYTHONPATH}
```

### 1.4 HFユーザー名を設定（クォートなし）

```bash
export HF_USER=your_hf_username
```

`HF_USER="name"` のようにクォート付きで入れると、`--policy.repo_id=${HF_USER}/...` で YAML パースエラーになることがあります。

### 1.5 純正FastVLMをダウンロード

```bash
cd /home/<you>/VLA-from-FastVLM
bash scripts/download_fastvlm.sh checkpoints
```

デフォルトでは `llava-fastvithd_0.5b_stage3` を取得します。

### 1.6 スモークテスト（Hub pushなし）

LeRobotリポジトリで実行:

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

### 1.7 本番学習例

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

## 2. `llava_qwen2` と `LlavaQwen2ForCausalLM` は必要か？

結論: 必要です。純正FastVLMの正しい形式です。

- `model_type=llava_qwen2` は公式形式であり、`config.json` を手動変更しないでください。
- 一部のローカルStage2/Stage3チェックポイントは `auto_map` が不足しており、`AutoModel` 直読み込みで失敗します。
- そのため `--policy.bootstrap_model_name=apple/FastVLM-0.5B` を指定して、クラス定義だけブートストラップします。
- 実際の重みは `--policy.vlm_model_name` のローカルチェックポイントが使われます。

つまり通常は以下の組み合わせでOKです。

- `--policy.vlm_model_name=/path/to/local/FastVLM`
- `--policy.bootstrap_model_name=apple/FastVLM-0.5B`

## 3. 単体スクリプトで動かす（LeRobotなし）

### 3.1 インストール

```bash
cd /home/<you>/VLA-from-FastVLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 3.2 学習

```bash
python scripts/train.py \
  --output-dir outputs/train/aloha_fastvlm \
  --model-id /home/<you>/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
  --bootstrap-model-id apple/FastVLM-0.5B \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --num-epochs 5 \
  --batch-size 4
```

### 3.3 評価

```bash
python scripts/eval_dataset.py \
  --checkpoint-dir outputs/train/aloha_fastvlm/checkpoints/step-1000 \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --split validation
```

`validation` が存在しない場合は既定で `train` にフォールバックします（`--allow-missing-split=true`）。

## 4. 主要フラグ

| フラグ | 意味 |
| --- | --- |
| `--policy.discover_packages_path` | LeRobotにプラグイン (`vla_fastvlm.lerobot_fastvla`) を読み込ませる。 |
| `--policy.type=fastvla` | 本リポジトリのポリシーを選択。 |
| `--policy.vlm_model_name` | 使用するFastVLM本体（ローカル/Hub）。 |
| `--policy.bootstrap_model_name` | `llava_qwen2` の不足メタデータ補完に使うブートストラップ元。 |
| `--policy.push_to_hub=false` | スモークテスト時のHub push無効化。 |

## 5. トラブルシュート

- `No module named 'transformers'`
  - LeRobot側venvに依存が入っていません。`source .../lerobot/.venv/bin/activate` 後に再インストールしてください。
- `huggingface-hub ... incompatible`
  - `huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0` で入れ直してください。
- `"user"/repo` 付近の `yaml.parser.ParserError`
  - `HF_USER` にクォートが混ざっています。`export HF_USER=user` で再設定。
- `model type 'llava_qwen2' ... not recognize this architecture`
  - `--policy.bootstrap_model_name=apple/FastVLM-0.5B` を付ける。ローカルモデルの `config.json` 存在も確認。
- `Split 'validation' not found`
  - 想定内です。`--split train` を使うかフォールバックをそのまま利用してください。

## 6. リポジトリ構成

```text
.
├── scripts/
│   ├── download_fastvlm.sh
│   ├── train.py
│   └── eval_dataset.py
├── src/vla_fastvlm/
│   ├── lerobot_fastvla/     # LeRobotプラグイン (policy.type=fastvla)
│   ├── fastvla/             # FastVLAのコア実装
│   └── model/               # FastVLMアダプタ（llava_qwen2フォールバックあり）
└── USE_WITH_LEROBOT_FASTVLA.md
```

## 7. ライセンス

- 本リポジトリのコード: MIT（`LICENSE`）
- Apple FastVLM関連: `third_party/ml-fastvlm` のライセンスに従う
- モデル重みは同梱しないため、取得・配布時はAppleの条件を必ず確認する
