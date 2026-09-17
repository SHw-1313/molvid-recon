# PVB multi-frame codec agent guide

## Mission

Implement the first, deliberately bounded version of the Scheme 1A multi-frame molecular codec. The codec must encode an ordered molecular clip, temporally compress it, and jointly reconstruct the clip. It is not a one-step PVB transition model wrapped in a loop.

The source of truth is:

- `docs/agent/multiframe_codec_v1/PLAN.md`
- `docs/agent/multiframe_codec_v1/TASKS.md`
- `docs/agent/multiframe_codec_v1/DECISIONS.md`
- `docs/agent/multiframe_codec_v1/HANDOFF.md`

Read all four files before editing code. Work in task order and update `TASKS.md` and `HANDOFF.md` as evidence is produced. Record any architectural deviation in `DECISIONS.md` before implementing it.

## Hard architectural invariants

- Preserve the existing `dyVAE`, PRETRAIN, MD, ADJ, DPO, and inference behavior. Add a new codec path; do not silently reinterpret `x0/x1`.
- The trajectory data unit is an ordered clip with one topology, not a collection of independent frame pairs.
- The canonical packed coordinate layout is `[T, N_total, 3]`; atom metadata is `[N_total, ...]` and `abid` maps atoms to samples.
- Frame indices are not physical time. Every trajectory clip must carry monotonic `time_ps` and positive `delta_time_ps`; never treat a 100 ps step and a 1 ns step as the same transition.
- Temporal attention, compression, latent metadata, and decoding must be conditioned on physical timestamps. Learned relative-time terms come from scalar functions of time differences and must not affect SE(3) transformation rules.
- v1 batches are homogeneous in task, temporal length, and sampling-interval bucket. Different buckets share model weights and alternate during training.
- Never upsample a coarse trajectory and label interpolated coordinates as fine-timescale ground truth. Fine trajectories may be downsampled as an explicitly labeled coarse-scale augmentation.
- The spatial graph encoder processes all frames in one batched call by assigning a distinct graph id to every `(sample, frame)` pair. No graph edge may cross frames or samples.
- Temporal compression and temporal decoding are causal. A prefix latent/reconstruction must not change when future input frames are appended.
- Invariant scalar features may produce temporal attention weights. Those same scalar weights may mix vector channels, but xyz axes must never be mixed by arbitrary learned matrices.
- The decoder receives the first-frame anchor, topology, masks, and compressed latent only. It must not receive target coordinates for later frames.
- Static samples use `T=1` plus coordinate corruption/denoising. Never replicate a static structure into a fake motion clip and train velocity losses on it.
- v1 is deterministic. Do not add KL, VQ, diffusion, a world-model trunk, or autoregressive rollout until the deterministic reconstruction gates pass.
- v1 stays atom-token based. Residue pooling, AF3/MSA conditioning, ligand interaction conditioning, and long-range chunk generation are follow-up work, not hidden additions.

## Repository and environment discipline

- Treat unrelated changes as user-owned. Never reset, discard, or overwrite them.
- Work on a dedicated branch or writable worktree. If this checkout is an archival/read-only upstream copy, stop and ask for a writable target.
- Before edits, record `git status --short`, `git rev-parse HEAD`, Python, PyTorch, CUDA, and GPU visibility in `HANDOFF.md`.
- All Python, test, and training commands must run through the local `enter-container` workflow and the `torch-ito` conda environment used by this project. First inspect the locally supported invocation. If either is unavailable, record the blocker; do not silently run in the host environment.
- Do not install packages or use the network without explicit user approval. Prefer the repository's current dependencies.
- Serialize GPU-heavy commands. Never launch competing training or evaluation jobs on the same GPUs.
- Use small CPU synthetic tests first, then one-GPU smoke tests, then DDP smoke tests. Full experiments are operator-run unless explicitly requested.

## Implementation style

- Prefer new focused modules over a broad rewrite:
  - `data/clip_dataset.py`
  - `data/collate_clip.py`
  - `module/multiframe_codec.py`
  - `module/temporal_codec.py`
  - `trainer/codec_trainer.py`
  - `config/codec.yaml`
  - `train_codec.py`
  - `eval_codec.py`
- If `TorchMD_VQ_ET` needs conditioning hooks, add optional constructor/forward arguments whose defaults execute the old path exactly. Add a checkpoint-load regression test.
- Keep tensors named with shape comments at module boundaries. Assert masks, topology consistency, temporal length, and graph-id ranges early.
- Avoid Python loops over frames in the spatial backbone. A small loop for preprocessing or metric reporting is acceptable; the model path must batch frames.
- Tests must cover shapes, masks, frame isolation, causality, SE(3) behavior, zero-valid-frame errors, static `T=1`, physical-time conditioning, 100 ps versus 1 ns buckets, and irregular finite-difference formulas.

## Definition of done for an agent run

An implementation run is complete only when it:

1. reports which task IDs were completed;
2. lists files changed and commands run;
3. records test/smoke evidence and unresolved blockers in `HANDOFF.md`;
4. leaves the worktree inspectable with no generated datasets, checkpoints, or large binary artifacts committed;
5. does not claim model-quality success without the pilot comparison specified in `PLAN.md`.
