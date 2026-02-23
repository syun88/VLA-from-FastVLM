# VLA-from-FastVLM（日本語ガイド）

English version is available in [`README.md`](README.md).
LeRobot から `fastvla` を使う手順は [`USE_WITH_LEROBOT_FASTVLA.md`](USE_WITH_LEROBOT_FASTVLA.md) を参照してください。

Apple FastVLM モデルにアクションヘッドを追加し、[lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) データセットでファインチューニングする Vision-Language-Action (VLA) パイプラインです。シミュレータ依存を排除し、オフライン学習/評価に集中しています。

---

## クイックスタート

```bash
# 1) 環境構築
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

# 2) （任意）データセットをローカルにキャッシュ
python - <<'PY'
from datasets import load_dataset
load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
PY

# 3) FastVLM を 10 エポック学習
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

# 4) データセット上の MSE 評価（split が存在しない場合は自動フォールバック）
python scripts/eval_dataset.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
  --split=validation
```

すべてのスクリプトは `tyro` ベースの CLI で、フラグはハイフン区切りです（例: `--checkpoint-dir`）。

---

## 必要要件

- Python 3.10 以上 + PyTorch 2.1 以上（CUDA / MPS / CPU に対応）。
- MuJoCo や gym などのシミュレータ依存は不要。
- Apple MPS または NVIDIA GPU があると高速。

---

## ディレクトリ構成

```
.
├── configs/                # 参考設定
├── scripts/                # 学習 / 評価 CLI
├── src/vla_fastvlm/        # Python パッケージ本体
│   ├── data/               # データセットラッパー
│   ├── model/              # FastVLM アダプタ + ポリシーヘッド
│   ├── fastvla/            # FastVLM を VLA 化する構成 / プロセッサ / モデル
│   ├── training/           # Accelerate ベースのトレーナー
│   └── utils/              # ログ / チェックポイント / デバイス補助
└── outputs/                # 学習成果物（チェックポイント / ログ）
```

---

## FastVLM モデルの取得

- Apple の Model Zoo から `.safetensors` / `.pt` と付属ファイルをダウンロードしてください。
- 例: `bash scripts/download_fastvlm.sh ./models/apple-fastvlm` で一括取得。`FASTVLM_BACKBONE_PATH` 環境変数を設定すると CLI から参照しやすくなります。
- Transformers が必要とする Python モジュールが含まれない場合は、Hugging Face CLI で公式スナップショット（例: `apple/FastVLM-0.5B`）を同じディレクトリに展開してください。
- 重みの再配布時は Apple の `LICENSE` / `LICENSE_MODEL` / `ACKNOWLEDGEMENTS` を必ず同梱してください。

---

## データセットのポイント

- 公開データセットは `train` スプリットのみ。`scripts/eval_dataset.py` は既定で `--allow-missing-split` が有効になっており、`--split=validation` を指定しても `train` に切り替わって警告を表示します。明示的に失敗させたい場合は `--allow-missing-split=false` を指定してください。
- ローカルダウンロードを避けたい場合は `--streaming` を有効にします。
- 手早く検証するには `--limit-samples 256` や `--batch-size 1` などを組み合わせます。
- Hugging Face のキャッシュ場所は `HF_HOME` で変更できます。`TRANSFORMERS_CACHE` 非推奨の警告は `HF_HOME` を設定することで解消できます。

---

## FastVLM のモデル形式（`llava_qwen2`）

- 純正 FastVLM チェックポイントの `model_type: llava_qwen2` は正しい形式です。手動で書き換えないでください。
- 一部の Stage2/Stage3 ローカル配布アーカイブは、`AutoModel` で直接ロードするためのメタデータ（`auto_map` など）が不足しています。
- 本リポジトリではフォールバックを追加しており、`llava_qwen2` のローカルチェックポイントを以下で読み込めます:
  - `--model-id /path/to/your/local/checkpoint`
  - `--bootstrap-model-id apple/FastVLM-0.5B`（または互換 FastVLM リポジトリ）
- `bootstrap-model-id` はクラス定義の読み込みにのみ使用され、実際の重みはローカルチェックポイントを使います。

---

## `lerobot-train` で使う

このリポジトリは LeRobot 用プラグイン `policy.type=fastvla` を提供します。

1. LeRobot 側の環境に必要依存（`transformers` / `timm` など）があることを確認。
2. このリポジトリの `src` を `PYTHONPATH` に追加。
3. `--policy.discover_packages_path=vla_fastvlm.lerobot_fastvla` を指定。

実行例:

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

Hub に push しない場合は `--policy.push_to_hub=false` を追加してください（または `--policy.repo_id` を指定）。

---

## 学習（`scripts/train.py`）

- `TrainArgs` dataclass にすべてのハイパーパラメータが定義されています。既定値は 10 エポック、バッチサイズ 4、ストリーミング無効です。
- レターボックスによる画像前処理（`--resize-with-padding`）やトークナイザー長 (`--tokenizer-max-length` など) を制御できます。不要なら `--resize-with-padding=false` にします。
- `max_steps` が `null` の場合は `num_epochs` が終了条件になります。学習時間を短縮したい場合は `--num-epochs` を減らすか `--max-steps` を設定してください。
- 出力内容:
  - `outputs/train/<run_name>/training_config.json`: 実行時の `TrainingConfig`。
  - `outputs/train/<run_name>/checkpoints/step-*/`: `policy_state_dict.pt` と `policy_config.json`。
  - TensorBoard ログ（`tensorboard --logdir outputs/train/fastvlm_aloha` で閲覧）。
- 継続学習は `--checkpoint-dir` と `--resume-from` を併用して再開できます。

---

## 評価（`scripts/eval_dataset.py`）

- 学習済みポリシーを用いてデータセットの MSE を計測します。
- バッチサイズ・ワーカー数・ストリーミング有無は学習スクリプトと同じフラグで制御します。
- Apple MPS では `pin_memory` に関する警告が表示されますが、現状の PyTorch 仕様上は無視して問題ありません。
- 実行例（`eval_log.txt` より）:
  ```
  [eval_dataset] Split 'validation' not found; using 'train' instead.
  MSE on split 'train': 0.004554
  ```
- 手早い検証には `--limit-samples` と `--batch-size 1` の組み合わせが有効です。

---

## 前処理リファレンス

- `FastVLMBackbone` は既定でレターボックスを有効にしています。歪みを許容する場合は `--resize-with-padding=false` を指定してください。
- トークナイザーの長さ (`--tokenizer-max-length 64`) とパディング位置（右詰め）を短めに設定しています。

---

## CLI リファレンス

| Script | Purpose | Notable Flags |
| ------ | ------- | ------------- |
| `scripts/train.py` | FastVLM をファインチューニング | `--output-dir`, `--model-id`, `--bootstrap-model-id`, `--dataset-repo-id`, `--num-epochs`, `--max-steps`, `--image-size`, `--resize-with-padding`, `--tokenizer-max-length` |
| `scripts/eval_dataset.py` | データセットの MSE 評価 | `--checkpoint-dir`, `--split`, `--allow-missing-split`, `--limit-samples`, `--streaming` |

`--help` を付けて実行するとすべてのフラグが表示されます。

---

## トラブルシュート

- **データセット split が見つからない**: 既定の `--allow-missing-split` を維持するか、`--split=train` を指定してください。
- **画像プロセッサの警告**: `TRANSFORMERS_USE_FAST=1` を設定するか最新の fast processor に更新してください。フォールバックのリサイズでも推論は安定します。
- **MPS 上の `pin_memory` 警告**: 既知の仕様です。CPU メモリのみが対象なので無視して問題ありません。
- **画像サイズが大きく OOM になる**: `--image-size` を下げるかパディングを無効化してください。
- **キャッシュ場所**: `HF_HOME=/path/to/cache` で Hugging Face のキャッシュを移動できます。

---

## ライセンス

- リポジトリのコード: MIT License (`LICENSE`)
- Apple FastVLM のアセット: Apple のライセンス (`third_party/ml-fastvlm`)
- モデル重みは同梱していません。Apple Model Zoo から取得し、`LICENSE_MODEL` / `ACKNOWLEDGEMENTS` に従ってください。

---

フィードバック・プルリクエスト歓迎です。新しいタスクに適用したチェックポイントもぜひ共有してください。
