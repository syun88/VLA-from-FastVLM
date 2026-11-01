# FastVLM-VLA experimental protocol

This document fixes the scientific comparisons before confirmatory outcomes are inspected.

## Units and data

- Dataset: `lerobot/metaworld_mt50`, revision
  `a59f742d218c903328164257ecf180f9b18018a1`, Apache-2.0.
- Tasks: `button-press-topdown-v3`, `push-v3`, and `reach-v3`.
- Episode split per task: 35 train, 7 validation, 8 test; split seed `20261101`.
- `push-v3` contains 100 episodes; the deterministic split uses 50 and records the unused
  episodes, so all three tasks contribute equally many demonstrations.
- Training observations use stride 2. No episode appears in more than one split.
- State is a 4-D gripper position/opening vector. Action is a 4-D
  delta-xyz + gripper vector. Both are normalized from train episodes only.
- Targets are 8-step action chunks. Chunks stop at episode boundaries and carry an explicit
  validity mask. Validation/test losses use those masks.

The action/state dimensions are task-adapter values in `config.json`; they are not constants
inside the model. Multi-camera, joint-space, and dual-arm schemas can be represented by
different adapters without changing FastVLM.

## Model conditions

1. State-only MLP: state -> 8-step action chunk.
2. FastViTHD spatial-only: projected FastViTHD grid -> action head.
3. FastVLM frozen: final FastVLM hidden state + 4 x 4 spatial FastViTHD grid + state.
4. FastVLM frozen, global-only: final hidden state + state, no spatial grid.
5. FastVLM LoRA: real language-layer LoRA weights updated jointly with the action head;
   FastViTHD and the multimodal projector stay frozen while projected visual tokens are cached.
6. A lightweight alternative VLM with the same action-head interface, if its runtime and
   checkpoint pass licensing and compatibility checks.

All confirmatory methods use the same 5,000 optimizer updates with one demonstration per
update, the same tasks, episode splits, stride, action horizon, optimizer family, and evaluation
seeds. This equal sample-exposure budget does not equalize wall-clock time: the FastVLM LoRA
condition must backpropagate through the language layers, while action-head conditions use
cached features. Training time, samples/second, and memory are reported rather than hidden.
Differences in pretrained history and architecture remain explicit limitations.

## Metrics

Offline test metrics are normalized action RMSE/MAE over valid chunk elements, separately by
task and aggregated with equal task weight.

Closed-loop metrics use Meta-World's native success definition:

- success count/rate and exact episode count;
- final task-specific manipulation error where a meaningful target exists;
- action-stability L1 (mean successive-action difference);
- failure categories;
- p50/p95 end-to-end observation-to-action-chunk latency;
- policy replan interval and amortized action latency;
- MLX peak memory and process memory;
- nominal instruction versus counterfactual instruction success.

Every confirmatory run records Git SHA, W&B run ID/URL, dataset revision, base-model
revision, config, seed, split manifest, and checkpoint identifier.

## Seeds and uncertainty

Primary comparisons use seeds 42, 43, and 44. The seed controls data order, action-head
initialization, LoRA initialization, and any optional sampling. Closed-loop evaluation uses
fixed, disjoint initial-state seed ranges and reports Wilson 95% intervals for success.
Training seeds are treated as independent replicates; no seed is selected based on test
performance.

Final target is 50 episodes per task per seed when runtime permits. Initial-state seeds beginning
at 200000 were used for the 10-episode engineering pilot, so confirmatory evaluation uses a fresh
range beginning at 300000. Any reduction is recorded
with the exact episode count and reason. Interim or smoke runs are not reported as
confirmatory evidence.

## Determinism and leakage controls

- Splits are chosen by episode index before model training and stored as JSON.
- The held-out test split is never used for optimizer updates or checkpoint selection.
- Validation chooses no hyperparameters in the confirmatory phase; the pilot configuration
  and its evidence are reported separately.
- Clarification/test seeds are distinct from calibration seeds.
- Data preprocessing uses the checkpoint's own 1024-pixel FastVLM processor configuration.
- Counterfactual tests change only the instruction text and keep the environment seed and
  observation sequence policy fixed as closely as possible.

## Evidence boundaries

Simulation success is not physical-robot performance. The project does not claim language
understanding from a constant prompt; it requires a paired counterfactual test. A token or
backbone is not called superior from one seed or from offline action loss alone.
