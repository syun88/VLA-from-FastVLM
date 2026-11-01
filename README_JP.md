# VLA-from-FastVLM

[![CI](https://github.com/syun88/VLA-from-FastVLM/actions/workflows/ci.yml/badge.svg?branch=research/mlx-fastvla)](https://github.com/syun88/VLA-from-FastVLM/actions/workflows/ci.yml)

Apple FastVLM をロボットの Vision-Language-Action（VLA）バックボーンとして利用できるかを、
Apple Silicon 上で実際に学習・推論・閉ループ評価して検証する研究実装です。現在の主経路は
PyTorch ではなく、`mlx.core` / `mlx.nn` / `mlx.optimizers` による MLX ネイティブ実装です。

英語版の詳細: [`README.md`](README.md)

## 研究目的

FastVLM の効率性は主に VLM の Time-to-First-Token として報告されています。本研究では、
その速度と visual-token 圧縮が、行動チャンク予測を含む VLA でも有効かを検証します。
FastVLM が優れているとは仮定せず、同じデータ、Action Head、action horizon、評価 episode 数、
初期状態 seed を使い、以下を比較します。

- State-only
- FastViTHD visual feature only
- Frozen FastVLM（global-only / global + spatial）
- FastVLM language LoRA + Action Head
- SmolVLM2-500M の同等 Action Head

測る対象は、closed-loop task success、manipulation error、action stability、未見初期状態への
汎化、instruction counterfactual、推論 latency、memory、training throughput です。

## MLX ネイティブ構成

- Apple `FastVLM-0.5B` revision
  `16375720c2d673fa583e57e9876afde27549c7d0` を固定
- FastViTHD の 256 projected visual token から 4×4 spatial summary を作成
- FastVLM 最終 language hidden state + spatial summary + proprioceptive state を融合
- 8 step（80 Hz で 8 action）の Behavior Cloning action chunk を予測
- Frozen FastVLM では visual token と hidden state をキャッシュ
- FastVLM LoRA は Qwen2 attention / MLP projection に適用し、FastViTHD と multimodal projector は
  frozen のまま
- Action Head、LoRA、AdamW、forward/backward、checkpoint、推論は MLX で実行
- PyTorch は旧 `v0.1.0-alpha` の履歴保存のみ。`legacy/torch_v01/` に隔離し、active package からは
  import しません

Action Head は Meta-World 固有ではありません。camera 群、state 名、action 名、次元、
control frequency、action horizon、normalization は設定・task metadata で差し替えられます。
複数カメラと Dual Scorpion のような dual-arm schema のテストも含みます。

## 実測済みの結果

以下は 2026-11-01 時点で保存済みの結果です。未完了の実験を完成として扱っていません。

### 3 seed の state-only baseline

`button-press-topdown-v3`、`push-v3`、`reach-v3` の各 50 episode、合計 150 episode/seed。
初期状態 seed は 300000 以降で、学習 episode とは分離しています。

| training seed | Button | Push | Reach | Pooled success |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0/50 | 0/50 | 11/50 | 7.33% |
| 43 | 0/50 | 0/50 | 12/50 | 8.00% |
| 44 | 0/50 | 0/50 | 11/50 | 7.33% |

### Frozen FastVLM seed 42

同じ 3 task、各 50 episode、未見 seed。W&B run `1mweq435`。

| Task | Success |
| --- | ---: |
| button-press-topdown-v3 | 49/50 |
| push-v3 | 2/50 |
| reach-v3 | 10/50 |
| 合計 | 61/150（40.7%） |

これは seed 42 の結果であり、3 seed 全体の確定結論ではありません。9,075 frame の観測を
使った 5,000 optimizer update の低予算条件です。push と reach は低く、FastVLM が全 Meta-World
操作に有効という主張にはなりません。

### 単一タスク pilot

button-only、20 epoch の pilot では、Frozen FastVLM が 10/10、FastViTHD-only が 10/10、
state-only が 4/10 でした。これは実装の疎通確認用 pilot で、confirmatory 結果ではありません。

詳細な履歴、W&B run ID、未完了項目は
[`docs/research_log.md`](docs/research_log.md) に記録しています。

## データと split

- Dataset: `lerobot/metaworld_mt50`
- Dataset revision: `a59f742d218c903328164257ecf180f9b18018a1`
- 2,500 episode、204,806 frame、49 instruction、80 Hz
- 480×480 RGB、4 次元 state/action
- 使用 task: `button-press-topdown-v3`, `push-v3`, `reach-v3`
- task ごとに train 35 / validation 7 / test 8 episode
- split seed: `20261101`
- frame stride: 2
- action horizon: 8
- `push-v3` は 100 episode あるため、50 episode を deterministic に未使用として manifest に記録
- 学習と評価の episode は重複しません

Manifest: [`manifests/metaworld_mt50/v1`](manifests/metaworld_mt50/v1)

## セットアップ

macOS / Apple Silicon、Python 3.12、MLX 0.31 系を前提にしています。

```bash
cd /path/to/VLA-from-FastVLM
export FASTVLA_WORK_ROOT=/path/to/local/fastvla-storage
export HF_HOME=/path/to/huggingface-cache
export FASTVLA_FASTVLM_MLX="$FASTVLA_WORK_ROOT/models/apple-fastvlm-0.5b-mlx"
export FASTVLA_DATASET_ROOT="$HF_HOME/hub/datasets--lerobot--metaworld_mt50/snapshots/a59f742d218c903328164257ecf180f9b18018a1"
uv pip install --python python3 -e '.[dev,sim]'
wandb login --verify
hf auth whoami
```

W&B / Hugging Face の token は表示・保存・commit しません。W&B は
`fastvlm-vla-research` private project を使用します。

## 再現コマンド

### 1. Apple checkpoint を MLX 形式へ変換

既存のローカル checkpoint を再利用します。この変換は checkpoint-format compatibility のための
隔離処理で、学習・推論を PyTorch に戻すものではありません。

```bash
python3 -m mlx_vlm convert \
  --hf-path /path/to/apple/FastVLM-0.5B/snapshot \
  --mlx-path "$FASTVLA_FASTVLM_MLX" \
  --dtype bfloat16
```

### 2. FastVLM feature cache

```bash
python3 scripts/cache_features.py \
  --tasks button-press-topdown-v3,push-v3,reach-v3 \
  --wandb-mode online
```

10 sample ごとに resumable です。画像、weight、cache は Git に追加しません。

### 3. Frozen FastVLM Action Head

```bash
python3 scripts/train_action_head.py \
  --run-name fastvlm-frozen \
  --backbone-mode frozen \
  --seed 42 \
  --max-steps 5000 \
  --batch-size 1 \
  --wandb-mode online
```

validation loss で best checkpoint を選び、test split は checkpoint 選択に使いません。

### 4. FastVLM LoRA + Action Head

```bash
python3 scripts/train_lora.py \
  --run-name fastvlm-lora \
  --seed 42 \
  --max-steps 5000 \
  --wandb-mode online
```

MLX の trainable parameter は seed 42 で 5,654,184 でした。1 step の確認で LoRA gradient が
非ゼロで、optimizer update 後に adapter norm が変化することを確認しています。

### 5. Offline test

```bash
python3 scripts/evaluate_offline.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --wandb-mode online
```

Offline action error は closed-loop success の代替ではありません。

### 6. Closed-loop Meta-World

```bash
python3 scripts/evaluate_sim.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --episodes 50 \
  --seed-start 300000 \
  --run-name confirm-fastvlm-frozen-seed42 \
  --group fastvlm-frozen \
  --wandb-mode online
```

### ライブ推論画面

```bash
python3 scripts/live_demo.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-lora-seed42/best" \
  --task button-press-topdown-v3 --seed 300001
```

これは録画の再生ではありません。8 control step ごとに MuJoCo の現在の RGB frame と state を
FastVLM-LoRA checkpoint に実際に入力し、返った action chunk でシミュレータを進めます。
推論回数、推論時間、action、native success bit を画面に表示します。表示は 30 FPS ですが、
8 action chunk の推論に約 0.4 秒かかるため、80 Hz の実時間制御ではなく compute-paced な
ライブ demo です。

Counterfactual instruction の評価では `--counterfactual` を追加します。環境 seed と episode 数を
維持し、instruction だけを別 task の instruction に置き換えます。

### 7. Latency / memory benchmark

```bash
python3 scripts/benchmark_policy.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --task reach-v3 \
  --iterations 30 \
  --resolutions 256,512,768,1024 \
  --wandb-mode online
```

`mx.eval` で同期した image preprocessing、vision encoder、language model、Action Head、
end-to-end action chunk、amortized per-action time、cold load、peak memory を記録します。

### 8. 結果集計と paper artifact

```bash
python3 scripts/aggregate_results.py
python3 scripts/plot_results.py
python3 scripts/build_paper_results.py
```

生成先:

- `results/confirmatory/`
- `paper/figures/`
- `paper/generated/results.tex`

### 9. Checkpoint restore と release candidate

```bash
python3 scripts/validate_checkpoint.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --task reach-v3

python3 scripts/prepare_model_release.py \
  --checkpoint "$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-frozen-seed42/best" \
  --output-dir "$FASTVLA_WORK_ROOT/checkpoints/fastvla/release_candidates/fastvlm-vla"
```

`prepare_model_release.py` は `apple-amlr` の完全な license、attribution、派生であること、
変更内容、元 checkpoint revision、評価結果、MLX 推論手順を Model Card に含めます。upload は
`--upload --repo-id ...` を明示した場合だけ実行します。

## CI/CD

`.github/workflows/ci.yml` は、MLX が動作する ARM64 の `macos-15` runner を固定して使用します。

CI で実行するもの:

- `src`, `scripts`, `tests` の Ruff
- Python source の compile
- 13 個の unit test（公開リポジトリの secret/path hygiene test を含む）
- MLX の実 device と実計算
- sdist / wheel build
- `twine check`
- 新しい venv への wheel install と import
- build 成果物の artifact upload

`.github/workflows/release.yml` は `v*` tag で sdist / wheel を作り、GitHub Release に添付します。
Apple model weights や Hugging Face の認証情報は CI/CD に含めません。

CI/CD 修正 commit `4eff940` の GitHub Actions run は成功済みです。

## W&B / Hugging Face / storage

- W&B: `fastvlm-vla-research` private project。run ID、Git SHA、model/dataset revision、seed、
  config、checkpoint、latency、memory を記録
- Offline W&B run は sync 可能。local result JSON が正本
- Hugging Face: `syun88` の制約下で、license を満たす場合のみ研究用 checkpoint / adapter を公開
- Dataset bytes は license を確認し、無断で再配布しない
- HF Storage Bucket は不要と判断。外部 HDD に十分な空きがあり、artifact は数 GB。
  有料 storage の追加や plan 変更はしません

Apple FastVLM の weights は `apple-amlr` です。MIT の source-code license は weights に適用されません。
研究用派生モデルを公開する場合も、完全な Agreement、attribution、変更開示、非商用 Research
Purposes、Apple endorsement がないことを明記します。

## 既知の制約と未完了項目

- すべて simulation 結果です。実機 Dual Scorpion の性能ではありません。
- FastVLM LoRA は language layer の LoRA であり、FastViTHD / projector は frozen です。
  full-model fine-tuning ではありません。
- 低予算 5,000 update では push / reach の成功率が低く、FastVLM の一般的優位性は主張できません。
- 3 seed の frozen / Frozen LoRA / FastViTHD-only / global-only / SmolVLM2 比較、counterfactual、
  resolution ablation、最終 paper は実行中または未完了です。
- 未完了の測定値を推測値で埋めません。完了条件を満たすまでは結果を確定扱いしません。

## 公開リポジトリの安全性

公開する result には再現用の W&B run ID と checkpoint hash のみを含め、W&B entity 名、private
run URL、home directory path、machine 固有 mount path は含めません。W&B entity は local の
認証状態または `WANDB_ENTITY` から解決します。新しい result を公開する前に
`python3 scripts/sanitize_public_results.py` を実行してください。追跡対象 file に private path や
W&B entity 入り URL が再混入すると unit test が失敗します。

## License

- この repository の source code: MIT (`LICENSE`)
- Apple FastVLM code / weights: `third_party/ml-fastvlm/` と公式 checkpoint repository の
  Apple license。`third_party/apple-fastvlm-model-license.txt` も参照
- `lerobot/metaworld_mt50`: dataset card に従い Apache-2.0。dataset bytes は commit しません
