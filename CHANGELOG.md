# Changelog

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
