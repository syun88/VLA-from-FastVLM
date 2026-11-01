# Training exposure and reference-protocol audit

Audit date: 2026-11-01

This note separates episode coverage from frame coverage, optimizer updates, and sampled
observation windows for the completed FastVLM LoRA run. It also compares that budget with the
published SmolVLA and pi0.5/openpi training protocols. It is a diagnosis of the current evidence,
not a claim that training exposure is the only cause of the observed rollout failures.

## Local run: `fastvlm-lora-seed42`

Checkpoint provenance:

- Checkpoint: `$FASTVLA_WORK_ROOT/checkpoints/fastvla/runs/fastvlm-lora-seed42/best`
- Adapter SHA-256: `e85a402665e73be9a0a2965209b8175f4222d3beab367eda208618a5c71aaa8b`
- Action-head SHA-256: `0e37f0ddbca473dc0f5212af8804f4c0087943230805a3bfcfa52812fa1fdc1d`
- Training W&B run: `498nqciq`
- Base model: `apple/FastVLM-0.5B` at revision
  `16375720c2d673fa583e57e9876afde27549c7d0`
- Dataset: `lerobot/metaworld_mt50` at revision
  `a59f742d218c903328164257ecf180f9b18018a1`
- Trainable parameters: 5,654,184, comprising language-layer LoRA plus the action head
- Fixed FastVLM components: FastViTHD visual encoder and multimodal projector

The run used `max_steps=5,000`, `batch_size=1`, `gradient_accumulation_steps=1`, and an
8-action horizon. The LoRA trainer shuffled the training indices and began another permutation
whenever it reached the end. Consequently, 5,000 updates visited **1.168 passes of the 4,281
stride-2 training records**. All 4,281 cached training frames/chunks were seen at least once;
719 were seen a second time.

The larger `9,075` count stored under dataset manifests is the cache for the full 50-episode pool
per task. It includes validation, test, and deterministic unused episodes. It is not the number
of training observations used by the optimizer.

| Task | Train episodes | Train cache windows, stride 2 | Reconstructed full-frame opportunities | Optimizer samples | Cached-pass equivalent |
| --- | ---: | ---: | ---: | ---: | ---: |
| `button-press-topdown-v3` | 35 | 1,155 | 2,275 | 1,354 | 1.172 |
| `push-v3` | 35 | 2,299 | 4,563 | 2,681 | 1.166 |
| `reach-v3` | 35 | 827 | 1,619 | 965 | 1.167 |
| **Total** | **105** | **4,281** | **8,457** | **5,000** | **1.168** |

The full-frame count was reconstructed from each selected episode's maximum cached frame index
plus one; the cache stores every second frame. It is an estimate of the all-frame opportunity
count, not a count of independently visited image observations. Because every cached stride-2
record was visited at least once, unique dense-frame coverage was approximately
`4,281 / 8,457 = 50.6%`.

### What "full episode" means here

- **Episode selection:** complete episodes, not truncated episode fragments, were assigned to the
  train/validation/test splits.
- **Temporal supervision:** action chunks were constructed within episode boundaries and masked
  at the end; chunks did not cross episodes.
- **Observation sampling:** only every second frame was cached (`frame_stride=2`), and the
  optimizer consumed 5,000 action-window samples.
- **Evaluation:** closed-loop evaluation executes complete simulator episodes until native success
  or the step limit, rather than measuring offline batches.

Thus the run used every training episode and every cached training record, but it did **not** train
on an all-frame stream, and it had a much smaller sampled-window budget than the published systems
below. The LoRA trainer's logged `train/epoch=(step+1)/total_steps` is schedule progress, not a
count of passes over the cache.

## Published reference protocols

| System and source | Episodes used for training | Training updates | Global samples/windows | Trainable scope | Evaluation protocol |
| --- | --- | --- | --- | --- | --- |
| SmolVLA LeRobot guide | About 50 episodes per task; 10 demonstrations at each of 5 variations | 20,000 | 1,280,000 at batch 64 | Task fine-tuning of the released base model | Rollout in the target setup; dataset variation matters |
| SmolVLA paper, simulation fine-tuning | Meta-World dataset: 50 demonstrations for each of 50 tasks | 100,000 | 6,400,000 at batch 64 | Action expert trained; VLM frozen | 10 trials per task on Meta-World and LIBERO |
| pi0.5 openpi LIBERO example | `physical-intelligence/libero`: 1,693 episodes | 30,000 | 7,680,000 at batch 256 | No LoRA/freeze filter in `pi05_libero`; full-parameter fine-tuning | 50 trials per task; 10-action chunks, replanning every 5 executed actions |

The published systems also sample observation/action-chunk windows from episode data; "full
episode" is not a training sample unit. Their distinction from this study is that they expose the
optimizer to substantially more sampled windows, and SmolVLA explicitly reports that a 25-episode
dataset was insufficient while 50 episodes repeated across variations worked better.

At 5,000 sampled windows, the local LoRA run is **256 times smaller** than the SmolVLA documentation
budget, **1,280 times smaller** than the SmolVLA paper's simulation fine-tuning budget, and **1,536
times smaller** than the pi0.5 openpi LIBERO example. These ratios are not like-for-like model or
hardware comparisons, but they rule out describing the local budget as comparable to those official
recipes.

## Interpretation and next measurements

The evidence supports under-exposure as a **major plausible contributor** to the weak and volatile
closed-loop performance. It does not establish that longer training will fix the failures. The
completed seed-42 LoRA validation MSE (0.4902) was also worse than the frozen FastVLM seed-42 value
(0.4236), and a same-observation counterfactual test changed the action chunk by only about
`1e-6`, so language conditioning and the action-head architecture require separate investigation.

A defensible continuation must:

1. Train frozen and LoRA conditions with the same explicit sampled-window budget and report cache
   epochs, not just optimizer steps.
2. Run an all-frame (`frame_stride=1`) versus stride-2 ablation before attributing a causal effect
   to frame sampling.
3. Evaluate each training seed on the same fixed, fresh random-spawn episode set, separate from
   training and validation.
4. Report paired nominal versus unrelated-instruction rollouts rather than inferring language
   understanding from task identity.
5. Treat the four interactive `R`-key episodes as a usability check, not a success-rate estimate.

## Sources checked

- SmolVLA documentation: <https://github.com/huggingface/lerobot/blob/main/docs/source/smolvla.mdx>
- SmolVLA paper: <https://arxiv.org/abs/2506.01844>
- openpi README and LIBERO evaluation: <https://github.com/Physical-Intelligence/openpi>
- pi0.5 paper: <https://arxiv.org/abs/2504.16054>
- LIBERO dataset card: <https://huggingface.co/datasets/physical-intelligence/libero>
