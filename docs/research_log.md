# Research execution log

This log records completed work and provenance. It is not a substitute for raw metrics or
checkpoints.

## 2026-11-01

### Repository and environment audit

- Local branch: `research/mlx-fastvla`.
- Base commit before research work: `0409f0e3801c13953026d80921c4adb3e98a5487`.
- Foundation commit: `b6b81aa84225a054455fdb99cfe3e79b1042a912`; pushed to
  `origin/research/mlx-fastvla` and verified by `git ls-remote`.
- Unrelated `NemoClaw/` remained untracked and untouched.
- W&B viewer: `jimmy880510`; entity selected by W&B: `jimmy880510-`.
- W&B project `fastvlm-vla-research` created; GraphQL reports `access: PRIVATE`.
- Hugging Face CLI reports authenticated user `syun88` and organizations
  `mlx-community`, `LeRobot-worldwide-hackathon`, `robotics-course`.

### Dataset and model checks

- Dataset: 2,500 episodes, 204,806 frames, 49 instruction groups, 80 Hz, embedded 480 x 480
  images, 4-D state/action.
- `push-v3` is the only selected task with 100 episodes; the fixed protocol uses 50, matching
  the other tasks and recording the unused episodes.
- Split manifest seed: `20261101`.
- Selected cache samples after stride 2: button 1,659, push 6,228, reach 1,188.
- Apple `FastVLM-0.5B` converted from the existing local Transformers-cache checkpoint to MLX;
  a warm 1024-pixel frame plus text path measured roughly 0.32-0.41 s.
- Meta-World 3.0.0 failed under MuJoCo 3.17 with a model-equality-data broadcast error. Upgrade
  to Meta-World 3.1.1 / MuJoCo 3.3.0 resolved it; expert-policy runs on reach, button, and push
  reached native success.

### Feature cache run

- W&B run: `pq1rhmxv`,
  `https://wandb.ai/jimmy880510-/fastvlm-vla-research/runs/pq1rhmxv`.
- Run group: `feature-cache-v1`.
- Configuration corrected to source commit `b6b81aa` because the worker began on that exact
  working tree before the commit was created.
- Cache extraction is resumable and writes no remote model weights.

### Pilot training and closed-loop check

The first integrated run trained one **button-only, seed-42, 20-epoch** action-head policy for
each of three quick conditions. These are engineering pilots, not the pre-registered final
comparison.

| Condition | Train W&B | Eval W&B | Success, 10 held-out episodes |
| --- | --- | --- | ---: |
| FastVLM frozen, global + spatial + state | `v81n3b0y` | `0uke4wrc` | 10 / 10 |
| FastViTHD spatial only | `9hvvy66k` | `7fxsmd2b` | 10 / 10 |
| State only | `2t0wejum` | `nnvg22rf` | 4 / 10 |

The pilot demonstrates a genuine MLX-trained checkpoint driving the simulator. It does not
establish a task-suite, multi-seed, language-grounding, or efficiency conclusion. Result
manifests and per-episode records are under `results/pilot/button-press-topdown-v3/`.

The combined three-task cache is still in progress at this log update.
