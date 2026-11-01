# LeRobotでFastVLAの使い方

このドキュメントは、`VLA-from-FastVLM` の `fastvla` ポリシーを `lerobot-train` で使用するための簡単な手順をまとめたものです。

## 1. 概要

`policy.type=fastvla` は、次のような構成で動作します：

* **バックボーン:** FastVLM (`LlavaQwen2ForCausalLM`, `model_type=llava_qwen2`)
* **ヘッド:** 状態と視覚特徴を結合する軽量MLPアクションヘッド
* **LeRobot統合:** プラグインによる動的登録 (`--policy.discover_packages_path`)

`llava_qwen2` は純正FastVLMの正しい形式で、変更は不要です。

## 2. 前提条件

* **LeRobot環境:** `/home/user/lerobot/.venv`
* **このリポジトリ:** `/home/user/VLA-from-FastVLM` ( `./scripts/download_fastvlm.sh` modelをダウンロードすること)
* **ローカルチェックポイント:**

  * `/home/user/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3`

## 3. 依存関係のインストール（LeRobot側venv）

LeRobot環境をアクティベートし、必要なパッケージをインストールします。

```bash
source /home/user/lerobot/.venv/bin/activate

pip install -U \
  "transformers>=4.57.1,<5.0.0" \
  "timm>=1.0.0" \
  "tyro" \
  "metaworld" \
  "huggingface-hub[cli,hf-transfer]>=0.34.2,<0.36.0"
```

**メモ:**

* `huggingface-hub` は `lerobot` の要件に合わせて `<0.36.0` に固定してください。
* `HF_USER` 環境変数を設定します（クォートなし）。

```bash
export HF_USER=your_huggingface_username
```

## 4. プラグインを認識させる

次に、プラグインのパスをPythonパスに追加します。

```bash
export PYTHONPATH=/home/user/VLA-from-FastVLM/src:${PYTHONPATH}
```

## 5. 最初の動作確認（Smoke Test）

Hubへのプッシュを無効化して、最小限の手順で動作確認を行います。

```bash
lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.vlm_model_name=/home/user/VLA-from-FastVLM/checkpoints/llava-fastvithd_7b_stage3 \
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

次に、本番用の学習コマンドです。

```bash

lerobot-train \
  --policy.discover_packages_path=vla_fastvlm.lerobot_fastvla \
  --policy.type=fastvla \
  --policy.device=cuda \
  --policy.vlm_model_name=/home/syun/VLA-from-FastVLM/checkpoints/llava-fastvithd_0.5b_stage3 \
  --policy.bootstrap_model_name=apple/FastVLM-0.5B \
  --policy.image_size=1024 \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/metaworld_mt50 \
  --env.type=metaworld \
  --env.task=assembly-v3,dial-turn-v3,handle-press-side-v3 \
  --output_dir=./outputs_fastvla_full_05b \
  --steps=100000 \
  --batch_size=1 \
  --num_workers=2 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --wandb.project=lerobot-fastvla \
  --wandb.entity=${WANDB_ENTITY} \
  --wandb.mode=online
```

## 7. 重要フラグの意味

* `--policy.discover_packages_path`: LeRobotにプラグインパッケージをインポートさせるパスです。
* `--policy.type=fastvla`: プラグインで登録されたポリシータイプを使用します。
* `--policy.vlm_model_name`: 使用するFastVLMのチェックポイント（ローカルパス可）。
* `--policy.bootstrap_model_name`: `llava_qwen2` のローカルチェックポイントで `auto_map` が不足する場合のブートストラップ用のモデルID。
* `--policy.push_to_hub=false`: 検証時にHubへプッシュしないようにする安全設定です。

## 8. よくあるエラー

### `No module named 'transformers'`

LeRobot側のvenvに依存パッケージがインストールされていません。セクション3のコマンドを実行してください。

### `yaml.parser.ParserError` with `"syun88"/repo`

`HF_USER` の設定にクォートが含まれています。以下のコマンドで再設定してください。

```bash
export HF_USER=syun88
```

### `huggingface-hub ... is incompatible`

LeRobotの要件に合わせて、`huggingface-hub<0.36.0` に戻してください。

### プラグインのインポートは成功するが、学習時にモデルのロードに失敗

* `--policy.vlm_model_name` が正しいディレクトリか確認
* `config.json` が存在するか確認
* `--policy.bootstrap_model_name` を `apple/FastVLM-0.5B` など有効なFastVLMリポジトリに設定

## 9. 参考: プラグインの実装場所

* `src/vla_fastvlm/lerobot_fastvla/configuration_fastvla.py`
* `src/vla_fastvlm/lerobot_fastvla/modeling_fastvla.py`
* `src/vla_fastvlm/lerobot_fastvla/processor_fastvla.py`
* `src/vla_fastvlm/model/fastvlm_adapter.py` (llava_qwen2ブートストラップローダー)