# Artifact and storage policy

## Local HDD first

`/Volumes/data` has approximately 2.6 TB free. Current active artifacts require:

- FastVLM MLX checkpoint: about 1.25 GB.
- SmolVLM comparison checkpoint: about 1 GB.
- FastVLM feature cache: expected roughly 4–5 GB for the selected 9,075 examples.
- Training checkpoints: action-head files are small; each LoRA adapter is tens of MB rather
  than a duplicated 0.5B base checkpoint.
- Selected rollout videos: a few hundred KB each.

This fits comfortably within local capacity. Model repositories and local checkpoints remain
the primary active storage.

## Hugging Face Storage Buckets

Decision: no bucket is created at the start of the study. The existing HDD has enough capacity,
the data are replicated by the GitHub/Hugging Face source revisions, and intermediate features
can be regenerated. A private bucket would add upload, recovery, privacy, and possible quota or
pricing considerations without a demonstrated current benefit.

The decision can be revisited only if measured local usage approaches capacity or a concrete
remote-recovery requirement appears. No paid plan, storage upgrade, or charge is authorized.

## Git

Committed: source, configs, split manifests, result manifests, plots/tables, paper sources, and
documentation.

Never committed: model weights, datasets, cached features, optimizer state, credentials,
high-volume rollouts, or W&B caches.

## Weights & Biases

Use the private `fastvlm-vla-research` project for scalar curves, summary metrics, provenance,
and a small number of selected videos. Do not repeatedly upload checkpoints or raw frames.
Local JSON/JSONL files remain the primary evidence if network sync fails.

## Hugging Face release boundary

Final model/adapter publication uses a versioned model repository and a Model Card only after
checkpoint restoration and inference validation. Restricted Apple-derived files carry
`apple-amlr`, the full agreement, attribution, derivative identification, modification
disclosure, and a non-commercial research statement. Dataset bytes are referenced from
`lerobot/metaworld_mt50` under Apache-2.0 and are not duplicated.
