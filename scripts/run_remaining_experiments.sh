#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="/Volumes/data/checkpoints/fastvla/runs"
eval_root="/Volumes/data/checkpoints/fastvla/evals"
benchmark_root="/Volumes/data/checkpoints/fastvla/benchmarks"
smol_cache="/Volumes/data/dataset/fastvla/features-smolvlm"
cd "$repo_root"

> /tmp/fastvla-queue-status
printf 'started %s\n' "$(date -Iseconds)" >> /tmp/fastvla-queue-status

wait_for_eval() {
  local run_name="$1"
  while pgrep -f "scripts/evaluate_sim.py .*--run-name ${run_name}" >/dev/null 2>&1; do
    sleep 30
  done
}

eval_run() {
  local checkpoint="$1" run_name="$2" group="$3" videos="$4"
  local counterfactual="${5:-false}"
  if find "$eval_root/$run_name" -name summary.json -print -quit 2>/dev/null | grep -q .; then
    printf 'skip existing eval %s\n' "$run_name" >> /tmp/fastvla-queue-status
    return
  fi
  printf 'eval %s %s\n' "$run_name" "$(date -Iseconds)" >> /tmp/fastvla-queue-status
  local extra=()
  if [[ "$counterfactual" == "true" ]]; then
    extra+=(--counterfactual)
  fi
  python3 scripts/evaluate_sim.py \
    --checkpoint "$checkpoint" \
    --episodes 50 \
    --seed-start 300000 \
    --run-name "$run_name" \
    --group "$group" \
    --save-videos "$videos" \
    "${extra[@]}" \
    --wandb-mode online
}

offline_run() {
  local checkpoint="$1"
  python3 scripts/evaluate_offline.py --checkpoint "$checkpoint" --wandb-mode online
}

train_lora() {
  local seed="$1"
  if [[ -f "$runs_root/fastvlm-lora-seed${seed}/best/metadata.json" ]]; then
    printf 'skip existing lora train %s\n' "$seed" >> /tmp/fastvla-queue-status
    return
  fi
  printf 'train lora seed %s %s\n' "$seed" "$(date -Iseconds)" >> /tmp/fastvla-queue-status
  python3 scripts/train_lora.py --run-name fastvlm-lora --seed "$seed" --max-steps 5000 --wandb-mode online
}

wait_for_eval "confirm-fastvlm-frozen-seed43"

# Finish the primary FastVLM frozen condition first.
eval_run "$runs_root/fastvlm-frozen-seed44/best" "confirm-fastvlm-frozen-seed44" "fastvlm-frozen" 0

# Complete the three-seed LoRA condition.
train_lora 43
train_lora 44
eval_run "$runs_root/fastvlm-lora-seed42/best" "confirm-fastvlm-lora-seed42" "fastvlm-lora" 1
eval_run "$runs_root/fastvlm-lora-seed43/best" "confirm-fastvlm-lora-seed43" "fastvlm-lora" 0
eval_run "$runs_root/fastvlm-lora-seed44/best" "confirm-fastvlm-lora-seed44" "fastvlm-lora" 0

# Visual-token and state-input controls.
for seed in 42 43 44; do
  eval_run "$runs_root/fastvithd-only-seed${seed}/best" "confirm-fastvithd-only-seed${seed}" "fastvithd-only" 0
done
for seed in 42 43 44; do
  eval_run "$runs_root/fastvlm-global-only-seed${seed}/best" "confirm-fastvlm-global-only-seed${seed}" "fastvlm-global-only" 0
done

# Held-out action metrics and representative performance benchmarks.
for run in fastvlm-frozen-seed42 fastvlm-frozen-seed43 fastvlm-frozen-seed44 \
           fastvithd-only-seed42 fastvithd-only-seed43 fastvithd-only-seed44 \
           fastvlm-global-only-seed42 fastvlm-global-only-seed43 fastvlm-global-only-seed44 \
           state-only-seed42 state-only-seed43 state-only-seed44 \
           fastvlm-lora-seed42 fastvlm-lora-seed43 fastvlm-lora-seed44; do
  if [[ "${run}" == fastvlm-lora-* && ! -f "$runs_root/${run}/best/metadata.json" ]]; then
    continue
  fi
  offline_run "$runs_root/${run}/best"
done

for run in fastvlm-frozen-seed42 fastvithd-only-seed42 fastvlm-global-only-seed42 \
           state-only-seed42 fastvlm-lora-seed42; do
  if [[ -f "$runs_root/${run}/best/metadata.json" ]]; then
    python3 scripts/benchmark_policy.py \
      --checkpoint "$runs_root/${run}/best" \
      --task reach-v3 \
      --iterations 30 \
      --resolutions 256,512,768,1024 \
      --output-root "$benchmark_root" \
      --wandb-mode online
  fi
done

# Materialize SmolVLM2 features only after the FastVLM comparisons so memory and
# thermal contention do not contaminate their runs.
python3 scripts/cache_features.py \
  --cache-root "$smol_cache" \
  --backbone-family smolvlm \
  --base-model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --base-model-path /Volumes/data/checkpoints/fastvla/models/smolvlm2-500m-mlx \
  --base-revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --visual-hidden-size 960 \
  --wandb-mode online

for seed in 42 43 44; do
  python3 scripts/train_action_head.py \
    --cache-root "$smol_cache" \
    --run-name smolvlm2-500m-frozen \
    --backbone-mode frozen \
    --backbone-family smolvlm \
    --base-model HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --base-model-path /Volumes/data/checkpoints/fastvla/models/smolvlm2-500m-mlx \
    --base-revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
    --visual-hidden-size 960 \
    --seed "$seed" \
    --max-steps 5000 \
    --batch-size 1 \
    --wandb-mode online
  eval_run "$runs_root/smolvlm2-500m-frozen-seed${seed}/best" "confirm-smolvlm2-500m-frozen-seed${seed}" "smolvlm2-500m-frozen" 0
done

# Paired instruction intervention for the two principal adapted/frozen conditions.
eval_run "$runs_root/fastvlm-frozen-seed42/best" "counterfactual-fastvlm-frozen-seed42" "fastvlm-frozen" 0 --counterfactual

printf 'complete %s\n' "$(date -Iseconds)" >> /tmp/fastvla-queue-status
