# Meta-World split and cache manifest v1

- Dataset: `lerobot/metaworld_mt50`
- Dataset revision: `a59f742d218c903328164257ecf180f9b18018a1`
- License: Apache-2.0 (dataset card)
- Split seed: `20261101`
- Split policy: 35 train / 7 validation / 8 test episodes per task.
- `push-v3` has 100 episodes; the unselected 50 are listed in its JSON.
- Frame stride: 2.
- Action horizon: 8 consecutive 80-Hz actions.
- Feature cache: `/Volumes/data/dataset/fastvla/features`
- Base model: `apple/FastVLM-0.5B`, revision `16375720c2d673fa583e57e9876afde27549c7d0`.
- Extractors: W&B runs `pq1rhmxv` (button), `9xviempt` (push), `t1htwtfx` (reach), group `feature-cache-v1`.

The JSON files are the authoritative episode lists used by the cache and training code.
