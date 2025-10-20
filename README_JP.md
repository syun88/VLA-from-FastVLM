# VLA-from-FastVLM（日本語版ガイド）

このリポジトリは、Apple が公開した FastVLM モデルにアクションヘッドを追加し、LeRobot の Aloha Insertion データセットでファインチューニングする Vision-Language-Action（VLA）パイプラインです。学習済みポリシーは `gym_aloha` シミュレータおよび将来の実機制御で利用できます。

## FastVLM モデルの取得

Apple が公開する FastVLM のチェックポイントは本リポジトリには含まれていません。まずは公式 [Model Zoo](https://github.com/apple/ml-fastvlm?tab=readme-ov-file#model-zoo) から必要なバリアントをダウンロードしてください。

1. `FastVLM-Base` など目的の重みを選択します。
2. `.safetensors` / `.pt` と付随する tokenizer/config を入手し、`models/apple-fastvlm` など任意のフォルダへ保存します。
3. 付属のスクリプトでまとめて取得することも可能です。

    ```bash
    bash scripts/download_fastvlm.sh ./models/apple-fastvlm
    export FASTVLM_BACKBONE_PATH=$(pwd)/models/apple-fastvlm
    ```

Hugging Face のプライベートリポジトリにアップロードして `model_id` を指定する運用も可能です。取得した重みには Apple の `LICENSE`, `LICENSE_MODEL`, `ACKNOWLEDGEMENTS` を必ず同梱し、ライセンス条件を遵守してください。

## 特長
- ✅ **エンドツーエンド学習**：Accelerate ベースの軽量トレーナーで FastVLM を Aloha データセットに合わせて微調整。
- 🖥️ **デバイス自動切り替え**：CUDA → MPS → CPU の優先順位で自動選択。環境変数 `FASTVLM_FORCE_DEVICE` で強制指定も可能。
- 🕹️ **シミュレータ対応**：MuJoCo + `gym_aloha` によるシミュレーション実行と MP4 動画の保存に標準対応。
- 🤖 **実機制御の導線**：ハードウェアインターフェースを差し替え可能な制御ループを実装。ROS2 などは未導入。
- 🔁 **再学習しやすい構成**：チェックポイントを JSON 設定 + state dict で保存し、継続学習や推論に再利用可能。

## ディレクトリ構成

```
.
├── configs/                # 学習設定などのサンプル
├── scripts/                # 学習 / 評価 / シミュレーション / 実機制御用 CLI
├── src/vla_fastvlm/        # Python パッケージ本体
│   ├── data/               # Hugging Face データセットラッパー
│   ├── model/              # FastVLM バックボーン & アクションヘッド
│   ├── training/           # Accelerate ベースのトレーナー
│   ├── sim/                # `gym_aloha` 用シミュレーションユーティリティ
│   ├── real/               # 実機制御用インターフェース
│   └── utils/              # ログ / チェックポイント / デバイス補助関数
└── pyproject.toml
```

## 1. セットアップ手順

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sim,real]"
```

> **補足**  
> - `sim` エキストラで `gym-aloha` (0.1.3 以上) や動画出力系ライブラリをインストール。  
> - `real` エキストラでシリアル / HID 系ドライバ（`pyserial`, `hidapi` など）を追加。

## 2. データセット準備

本プロジェクトは Hugging Face Hub の [lerobot/aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image) を使用します。ローカルにキャッシュしたい場合は以下のスクリプトを実行してください。

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("lerobot/aloha_sim_insertion_human_image")
PY
```

##### もしダウンロードでCAS service error出たら

```bash
export HF_HUB_DISABLE_XET=1
```

## 3. FastVLM のファインチューニング

`scripts/train.py` が固定化されたトレーニング CLI です。Apple の公開手順と近い実行例は次のとおりです。

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

出力内容:
- `training_config.json`：実行時の `TrainingConfig` を JSON 化したもの。
- `checkpoints/step-*/policy_config.json`：`FastVLMPolicyConfig` を JSON 保存。
- `checkpoints/step-*/policy_state_dict.pt`：PyTorch の重み。

継続学習を行う場合は、保存済みのチェックポイントを `--resume_from` に指定してください。

## 4. データセット上での評価

```bash
python scripts/eval_dataset.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --dataset_repo_id=lerobot/aloha_sim_insertion_human_image \
  --split=validation
```

評価指標は MSE（Mean Squared Error）を標準出力に表示します。

## 5. シミュレーション実行

MuJoCo 3.1 以上をインストールし、`MUJOCO_GL=egl` を使用できる環境を整えてください（スクリプト内で自動設定）。その後、以下でシミュレーションを実行します。

```bash
python scripts/run_sim.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --num_episodes=10 \
  --video_dir=outputs/eval/pi0_aloha
```

各エピソードの成功可否・報酬などが標準出力に表示され、`video_dir` に MP4 ファイルが保存されます。

## 6. 実機制御（将来の実機運用向け）

`AlohaHardwareInterface` を継承したクラスを実装し、通信・センサ取得を独自ハードウェア向けに適合させます。

```python
# my_robot/interface.py
from vla_fastvlm.real import AlohaHardwareInterface
import numpy as np

class CustomAlohaInterface(AlohaHardwareInterface):
    ...
```

実行例:

```bash
python scripts/run_real.py \
  --checkpoint_dir=outputs/train/fastvlm_aloha/checkpoints/step-10000 \
  --interface_cls=my_robot.interface.CustomAlohaInterface \
  --interface_kwargs_json='{"port":"/dev/ttyUSB0","camera_topic":"camera/top"}'
```

制御ループは設定された周波数で観測を取得し、FastVLM ポリシーで推定した関節コマンドを送出します。

## 7. デバイス選択

デフォルトは以下の優先順位でデバイスを選択します。
1. CUDA GPU（利用可能な場合）
2. Apple MPS Backend
3. CPU

環境変数 `FASTVLM_FORCE_DEVICE=cpu` などを指定すると強制的に固定できます。また各スクリプトの `--device_preference` で上書きも可能です。

## 8. FastVLM モデルの入手

Apple が配布する FastVLM の重みは本リポジトリには含まれていません。公式リポジトリの [Model Zoo](https://github.com/apple/ml-fastvlm?tab=readme-ov-file#model-zoo) からダウンロードしてください。

1. 上記リンク先で必要なバリアント（例：`FastVLM-Base`）を選択。
2. Apple の利用規約に同意し、`.safetensors`（または `.pt`）や tokenizer/config など付随ファイルを取得。
3. 任意のパス（例：`models/apple-fastvlm`）に配置し、モデル読み込み時にそのパスを指定します。

    ```bash
    export FASTVLM_BACKBONE_PATH=/absolute/path/to/models/apple-fastvlm
    ```

    ```python
    from vla_fastvlm.model.fastvlm_adapter import FastVLMBackboneConfig

    cfg = FastVLMBackboneConfig(model_id=FASTVLM_BACKBONE_PATH)
    ```

   もしくは重みを自身のプライベート Hugging Face リポジトリへアップロードし、`model_id` にそのリポジトリ名を指定しても構いません。

取得した重みには Apple の `LICENSE`, `LICENSE_MODEL`, `ACKNOWLEDGEMENTS` を必ず同梱し、ライセンス条件を遵守してください。

公式スクリプト互換の `scripts/download_fastvlm.sh` を用意しています。以下のように実行するとチェックポイントをまとめて取得して解凍できます。

```bash
bash scripts/download_fastvlm.sh ./models/apple-fastvlm
```


## 9. 設定ファイル

## 8. 設定ファイル

`configs/train_aloha.yaml` は README の手順を再現するための参考設定です。CLI に直接 YAML を読み込ませる機能はありませんが、Python スニペット等で `--key=value` 形式に展開すると便利です。

## 9. ライセンス

- 本リポジトリのオリジナルコードは MIT License（`LICENSE`）で公開しています。
- FastVLM に関するコード/アセットは Apple Inc. が著作権を保有し、`third_party/ml-fastvlm/LICENSE` に記載された条件で再配布しています。
- モデル重み (`LICENSE_MODEL`) や追加の著作権表記 (`ACKNOWLEDGEMENTS`) については Apple のリポジトリ（[apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)）をご参照のうえ、ご自身で取得してください。本リポジトリには重みファイルを同梱していません。
- Apple の商標・ロゴはプロモーション用途で使用できません。本プロジェクトが Apple 公式とは無関係であることを README 等に明記しています。

---

質問や改善提案がありましたら、お気軽に Issue や Pull Request をお寄せください。*** End Patch
