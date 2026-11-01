# Changelog

## v0.2.0-alpha — MLX-native research branch

### Added

- Native MLX FastVLM backbone, action-chunk head, frozen feature cache, and real LoRA training.
- Episode-disjoint Meta-World data adapter, action masking, per-task normalization, and checkpoints.
- Closed-loop Meta-World evaluation, latency/memory benchmarking, W&B provenance, and license audit.
- State-only, FastViTHD-only, global-only, frozen, LoRA, and SmolVLM comparison conditions.

## v0.1.0-alpha

Initial alpha release for researchers evaluating FastVLM-backed VLA policies.

### Added

- LeRobot plugin entrypoint with `policy.type=fastvla`.
- Standalone ALOHA training and evaluation scripts.
- Bootstrap loading path for local FastVLM `llava_qwen2` checkpoints with
  missing metadata.
- Lightweight CI, contribution guidelines, security policy, and project
  roadmap documentation.

### Known limitations

- Real robot rollout is not validated yet.
- Compatibility is currently documented around LeRobot `0.4.4`; newer releases
  need explicit tracking.
- Large FastVLM checkpoints and Apple model licenses are external to this
  repository.
