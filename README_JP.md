# VLA-from-FastVLM

Apple FastVLM をロボットの Vision-Language-Action バックボーンとして使う場合の、
Apple Silicon 上での MLX ネイティブ研究実装です。現在の主経路は PyTorch ではありません。
画像・言語の FastVLM 計算、Action Chunk Head、FastVLM LoRA、最適化、チェックポイント、
推論を `mlx.core` / `mlx.nn` / `mlx.optimizers` で実行します。

英語の詳細なセットアップ、再現コマンド、既知の制約は [`README.md`](README.md) を参照してください。

## 研究目的

FastVLM の効率性が、VLM の Time-to-First-Token だけでなく実ロボット操作の
Accuracy-Latency-Memory トレードオフでも維持されるかを、同一データ・同一 Action Head・
同一評価条件で検証します。FastVLM が優れているとは仮定しません。

## 現在の構成

- `apple/FastVLM-0.5B` revision
  `16375720c2d673fa583e57e9876afde27549c7d0` を固定
- FastViTHD の 4×4 空間要約 + FastVLM 最終 hidden state + 4 次元状態から
  8 step action chunk を予測
- Frozen FastVLM、FastVLM LoRA、FastViTHD のみ、State のみ、
  Global 特徴のみの比較
- `lerobot/metaworld_mt50` の episode 単位 train/validation/test 分割
- 実 MuJoCo Meta-World での closed-loop success、失敗分類、action stability、
  latency、memory 評価
- W&B に Git SHA、dataset/model revision、seed、checkpoint、run ID を記録
- checkpoint と feature cache は `/Volumes/data` に保存し、Git には含めません

旧 PyTorch 実装は履歴保持のため `legacy/torch_v01/` にあります。アクティブな
パッケージからは import しません。

## 再現の入口

```bash
uv pip install --python python3 -e '.[dev,sim]'
python3 scripts/cache_features.py
python3 scripts/train_action_head.py --run-name fastvlm-frozen --seed 42
python3 scripts/train_lora.py --run-name fastvlm-lora --seed 42
python3 scripts/evaluate_offline.py --checkpoint <checkpoint>/final
python3 scripts/evaluate_sim.py --checkpoint <checkpoint>/final --episodes 50
```

実験プロトコルは [`docs/experimental_protocol.md`](docs/experimental_protocol.md)、
upstream / license / local asset 監査は
[`docs/upstream_audit.md`](docs/upstream_audit.md) にあります。

## 制約

- 現時点の結果はシミュレーションのみで、実機の Dual Scorpion 性能ではありません。
- LoRA は FastVLM の language projection を適応し、FastViTHD と multimodal projector は
  frozen です。これを full-model fine-tuning とは表現しません。
- Apple FastVLM weights は `apple-amlr` ライセンスです。MIT のソースコードライセンスは
  weights には適用されません。派生モデルを公開する場合は非商用研究目的、元 Agreement の
  同梱、attribution、変更内容の開示、Apple endorsement がないことを明記します。
- 未検証の項目や結果を、計画・テスト件数・commit 数で完了扱いにはしません。
