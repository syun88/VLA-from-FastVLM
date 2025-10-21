# VLA-from-FastVLM（日本語ガイド）

English version is available in [`README.md`](README.md).

Apple FastVLM モデルにアクションヘッドを追加し、[lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) データセットでファインチューニングして、`gym_aloha` シミュレータおよび実機 Aloha ロボットで動作可能な Vision-Language-Action (VLA) パイプラインを構築します。

---

## クイックスタート

```bash
# 1) 環境構築
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[sim,real]"

# 2) （任意）データセットをローカルにキャッシュ
python - <<'PY'
from datasets import load_dataset
load_dataset("lerobot/aloha_sim_insertion_human_image", split="train")
PY

# 3) FastVLM を 10 エポック学習
# python scripts/train.py \
#   --output-dir=outputs/train/fastvlm_aloha \
#   --model-id=apple/FastVLM-base \
#   --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
#   --batch-size=2 --num-workers=2 --num-epochs=10 --mixed-precision=bf16
# または
python scripts/train.py \
  --output-dir outputs/train/fastvlm_aloha \
  --model-id "$FASTVLM_BACKBONE_PATH/llava-fastvithd_0.5b_stage3" \
  --dataset-repo-id lerobot/aloha_sim_insertion_human_image \
  --batch-size 2 \
  --num-workers 0 \
  --num-epochs 10 \
  --max-steps 1000

# 4) データセット上の MSE 評価（split が存在しない場合は自動フォールバック）
python scripts/eval_dataset.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset-repo-id=lerobot/aloha_sim_insertion_human_image \
  --split=validation

# 5) MuJoCo シミュレーション＋動画保存
python scripts/run_sim.py \
  --checkpoint-dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --num-episodes=10 \
  --video-dir=outputs/eval/pi0_aloha
```

全スクリプトは `tyro` ベースの CLI を提供し、フラグはハイフン区切り（例: `--checkpoint-dir`）です。

---

## 必要要件

- Python 3.10 以上 + PyTorch 2.1 以上（CUDA / MPS / CPU に対応）。
- MuJoCo 3.1 以上（OS に応じて `MUJOCO_GL` を自動選択）。
- `gym-aloha` 0.1.3 以上（`[sim]` エキストラでインストール）。
- Apple MPS バックエンド（macOS 13+）または NVIDIA GPU を推奨。
- 実機制御では `AlohaHardwareInterface` を継承したクラスを自作し、`[real]` エキストラでシリアル・HID ドライバを追加。

---

## ディレクトリ構成

```
.
├── configs/                # 参考設定
├── scripts/                # 学習 / 評価 / シミュレーション / 実機 CLI
├── src/vla_fastvlm/        # Python パッケージ本体
│   ├── data/               # データセットラッパー
│   ├── model/              # FastVLM アダプタ + ポリシーヘッド
│   ├── training/           # Accelerate ベースのトレーナー
│   ├── sim/                # MuJoCo ロールアウト補助
│   ├── real/               # 実機制御インターフェース
│   └── utils/              # ログ / チェックポイント / デバイス補助
└── outputs/                # 学習成果物（チェックポイント / 動画 / ログ）
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

## 学習（`scripts/train.py`）

- `TrainArgs` dataclass にすべてのハイパーパラメータが定義されています。既定値は 10 エポック、バッチサイズ 4、ストリーミング無効です。
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

---

## シミュレーション（`scripts/run_sim.py`）

- `gym_aloha/AlohaInsertion-v0` を起動し、ポリシー推論と動画保存を行います。
- OS に応じてレンダリングバックエンドを自動選択します（Linux: `egl`、macOS: `glfw`、Windows: `d3d11`）。必要なら `MUJOCO_GL` を上書きしてください。
- FastVLM アダプタが入力画像を自動リサイズ（最短辺 512px 以上）し、必要に応じて torchvision にフォールバックします。フォールバック時は一度だけ警告が表示されます。
- 主なフラグ: `--num-episodes`, `--max-episode-steps`, `--task-instruction`, `--video-dir`, `--record-video=false`.

---

## 実機制御（`scripts/run_real.py`）

- `AlohaHardwareInterface` を継承したクラスを `--interface-cls` で指定し、必要なら `--interface-kwargs-json` でパラメータを渡します。
- ランナーは観測を取得してポリシーに入力し、推定アクションを設定周波数で送出します。
- `--device-preference` や `FASTVLM_FORCE_DEVICE` によるデバイス指定はシミュレーションと共通です。

---

## CLI 一覧

| スクリプト | 目的 | 代表的なフラグ |
| ---------- | ---- | -------------- |
| `scripts/train.py` | FastVLM の学習 | `--output-dir`, `--model-id`, `--dataset-repo-id`, `--num-epochs`, `--max-steps`, `--mixed-precision` |
| `scripts/eval_dataset.py` | データセット MSE 評価 | `--checkpoint-dir`, `--split`, `--allow-missing-split`, `--limit-samples`, `--streaming` |
| `scripts/run_sim.py` | MuJoCo ロールアウト + 動画保存 | `--checkpoint-dir`, `--num-episodes`, `--max-episode-steps`, `--video-dir`, `--device-preference` |
| `scripts/run_real.py` | 実機制御ループ | `--checkpoint-dir`, `--interface-cls`, `--interface-kwargs-json`, `--task-instruction` |

各スクリプトは `--help` で全パラメータを確認できます。

---

## トラブルシューティング

- **データセットのスプリットが無い**: 既定の `--allow-missing-split` を利用するか、`--split=train` を指定してください。
- **`TRANSFORMERS_CACHE` 非推奨警告**: `HF_HOME` を設定すると抑制できます。
- **画像プロセッサが遅い警告**: Transformers 4.52 以降で `use_fast=True` が既定になります。早めに `TRANSFORMERS_USE_FAST=1` を有効化するか、フォールバック警告は無視してください。
- **MPS 上の `pin_memory` 警告**: 仕様通り無視可能です。CPU / CUDA ではピン止めメモリが有効になります。
- **MuJoCo の EGL 読み込み失敗**: macOS では `glfw` が自動使用されるため問題は解消済みです。手動で変更する場合は `MUJOCO_GL=glfw` を指定してください。
- **`gym_aloha` import エラー**: `pip install -e ".[sim]"` が成功しているか、MuJoCo 3.1 以上がインストールされているか確認してください。

---

## ライセンス

- リポジトリ本体: MIT License（`LICENSE`）。
- Apple FastVLM 関連のコード / アセット: Apple Inc. のライセンスに従います（`third_party/ml-fastvlm/LICENSE`）。
- モデル重みは同梱していません。Model Zoo から取得し、ライセンス条件を遵守してください。

---

Issue や Pull Request、シミュレーションとまだ色々とできてるかどうかが怪しいので、結果の共有などIssuesを歓迎します。Aloha 以外のタスクへ適用した場合はぜひ情報提供をお願いします。
