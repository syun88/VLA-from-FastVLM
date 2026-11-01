# Pilot closed-loop result

This is a 10-episode engineering pilot on one task and one training seed. It is not a
confirmatory multi-seed result and must not be used to claim general FastVLM superiority.

| Condition | Successes / episodes | W&B run | Checkpoint |
| --- | ---: | --- | --- |
| FastVLM frozen, global + spatial + state | 10 / 10 | `0uke4wrc` | `/Volumes/data/checkpoints/fastvla/pilots/pilot-button-frozen-seed42-seed42/final` |
| FastViTHD spatial only | 10 / 10 | `7fxsmd2b` | `/Volumes/data/checkpoints/fastvla/pilots/pilot-button-fastvithd-seed42-seed42/final` |
| State only | 4 / 10 | `nnvg22rf` | `/Volumes/data/checkpoints/fastvla/pilots/pilot-button-state-seed42-seed42/final` |

All conditions used the same episode split, 8-step replanning, 500-step maximum, and held-out
initial-state seeds starting at 200000. Training used 20 epochs on only the button task. The
FastVLM and FastViTHD conditions therefore establish that the implementation can control the
simulator, not that those backbones outperform state input on the full task suite.

The `summary.json` and `episodes.jsonl` files were copied from the local rollout directories;
W&B contains the corresponding curves and summaries.
