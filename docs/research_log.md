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
- W&B project `fastvlm-vla-research` created; GraphQL reports `access: PRIVATE`. The entity is
  intentionally resolved from local authentication and is not committed.
- Hugging Face authentication was verified without recording the token or account metadata in
  the repository.

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

- W&B run ID: `pq1rhmxv` in the private `fastvlm-vla-research` project.
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

The combined three-task cache later completed; the continuation is recorded below.

### Three-task training update

The combined cache completed with 1,659 button, 6,228 push, and 1,188 reach samples after
stride-2 filtering. All confirmatory conditions use 5,000 optimizer updates with one sample per
update. State-only, FastViTHD-only, FastVLM global-only, frozen FastVLM, and frozen FastVLM
seed 42/43/44 checkpoints were produced. Best checkpoints are selected on the validation split.

State-only closed-loop evaluation completed with 50 test episodes per task and three seeds:

| Seed | Button | Push | Reach | Pooled success | W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0/50 | 0/50 | 11/50 | 7.33% | `4ntj2rrr` |
| 43 | 0/50 | 0/50 | 12/50 | 8.00% | `e4ar37u5` |
| 44 | 0/50 | 0/50 | 11/50 | 7.33% | `t5utyh8z` |

Frozen FastVLM seed 42 completed 50 fresh test episodes per task: 49/50 button, 2/50 push,
10/50 reach, and 61/150 pooled. W&B run ID: `1mweq435`. This is one seed of the confirmatory
comparison, not a cross-seed final result.

Frozen FastVLM seed 43 completed the same protocol: 29/50 button, 2/50 push, 11/50 reach, and
42/150 pooled (28.0%). W&B run ID: `nwv1lwcc`. The between-seed difference is material and is
part of the reported uncertainty, not a result to average away silently.

### Interactive random-spawn check

The live viewer's `R` key was exercised against the seed-42 LoRA checkpoint. It resets the same
Meta-World environment using the task's native random object-placement space and immediately
recomputes a policy action chunk. Three interactive random seeds were tried:
`366929013` (failure at 500 steps), `396323305` (failure at 47 steps), and `790126988`
(failure at 500 steps). This is a three-episode usability check, not a formal success-rate
estimate; it exposes a clear fragility to object placement that the controlled evaluation needed.

### Training-exposure audit

The 9,075 manifest count for the three-task cache is the full 50-episode pool per task, including
validation, test, and unused episodes. The actual LoRA training split contains 4,281 cached windows
across 105 episodes. With 5,000 optimizer updates at batch size one, the optimizer completed 1.168
passes over those cached windows: all 4,281 cached windows were visited once and 719 a second time.
All training episodes were touched, but `frame_stride=2` means only every second frame was cached,
so unique dense-frame coverage was approximately 50.6%. The LoRA `train/epoch` metric is schedule
progress, not a count of cache passes.

SmolVLA's official guide uses about 50 episodes per task, 20,000 steps at batch 64 (1.28 million
sampled windows); its paper reports 100,000 steps at batch 64 for simulation fine-tuning. The
pi0.5 openpi LIBERO example uses 1,693 episodes, 30,000 steps at batch 256 (7.68 million sampled
windows), and 50 evaluation trials per task. The local 5,000-window run is therefore not a
comparable training budget. The complete accounting and sources are in
`docs/training_exposure_audit.md` and
`manifests/training_exposure/fastvlm-lora-seed42.json`.

The FastVLM LoRA seed-42 run completed 5,000 updates with 5,654,184 trainable parameters. A
dedicated one-step check showed a nonzero gradient and a changed adapter norm, and the saved
adapter was materially changed after training. Its validation normalized MSE was 0.4902 versus
0.4236 for frozen FastVLM seed 42 in this low-data budget. A LoRA closed-loop run was stopped
after 10 exploratory button episodes (3 successes) to remove redundant per-control-step
rendering; that partial run is not a result. The 50-episode LoRA evaluation must be rerun with
the optimized rollout loop. LoRA seeds 43 and 44 also remain to be trained.

Offline held-out results for the completed cached-feature conditions were recorded under
`results/offline/`; all are test-split action errors and not substitutes for closed-loop success.

The first combined three-task frozen pilot (5,000 updates) used 10 episodes per task at
exploratory initial-state seeds from 100000: 9/10 button, 1/10 push, and 2/10 reach
(12/30 overall). This exploratory result is copied to `results/pilot/multitask-frozen-5k/` and
is distinct from the fresh-seed confirmatory evaluation. It indicates the 5,000-update action
head is under-trained for push and reach, so no final FastVLM conclusion is drawn from it.

### Open work at interruption boundary

- Fresh-seed 50-episode closed-loop evaluations for frozen FastVLM, FastViTHD-only, and
  global-only across all three training seeds.
- Rerun the cleaned-up 50-episode LoRA evaluation and train LoRA seeds 43/44.
- SmolVLM2-500M cache, training, and matched evaluation.
- Resolution/token-count benchmark, generated tables/figures, final paper text, and the
  license-compliant Hugging Face release candidate.

The resumable command sequence is in `README.md`; all completed checkpoints and W&B/run
identifiers remain under the path selected by `FASTVLA_WORK_ROOT` and in the private W&B project.
