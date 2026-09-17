# Multi-frame codec v1 handoff

## Prepared state

- Planning repository: official PVB clone.
- Inspected upstream commit: `c08e5e3cd49d45c6d748387e78224843bd356f50` (`submit version`).
- Remote inspected: `https://github.com/yaledeus/PVB.git`.
- No implementation code has been changed by the planning pass.
- Added only the agent guide, project-scoped Luna worker configuration, and the four planning files.
- Updated all six files so physical time is a first-class codec condition: continuous timestamps, relative-time attention bias, time-bucket batching, timestamped latents/decoder queries, and bucket-aware losses/evaluation are now binding.
- The planning environment did not expose a `codex` executable, so the command below is based on current official Codex CLI syntax and must be run in the user's Codex-enabled shell.

## What the next worker should do

1. Read `AGENTS.md`, `PLAN.md`, `TASKS.md`, and `DECISIONS.md` completely.
2. Run T00 and append the real environment/baseline evidence to this file.
3. Implement T01 through T08 sequentially in small commits or reviewable diffs.
4. Run only synthetic/CPU tests until the local container and GPU state are confirmed.
5. Stop before full training. T09 is a smoke gate; T10 prepares the operator-run pilot.

The worker must not collapse the task back to pair prediction. The implementation is wrong if the learned path accepts only `x0/x1`, loops over frames through `realization()`, lets the decoder see later target coordinates, treats frame index as physical time, equates 100 ps with 1 ns, or fabricates fine-timescale labels by interpolating coarse trajectories.

## Exact initial worker prompt

Use this in an interactive Codex session, or as the prompt to `codex exec`:

> Act as the sole implementation worker for Scheme 1A multi-frame codec v1. Read `AGENTS.md` and all four files in `docs/agent/multiframe_codec_v1/` before editing. Start with T00, then execute the first unblocked tasks in dependency order. Preserve all existing PVB behavior and implement an additive packed-clip path with batched PVB spatial encoding, causal SE(3)-safe temporal compression, and joint multi-frame decoding. Treat `time_ps`/`delta_time_ps` as first-class conditions: implement continuous relative-time attention bias, timestamped latents, target-time decoder queries, homogeneous sampling-interval buckets, correct physical finite differences, and bucket-stratified metrics. Never equate 100 ps with 1 ns or interpolate coarse trajectories into fine-timescale labels. Do not implement a generative trunk, rollout, KL/VQ, residue pooling, or AF3/MSA conditioning. Use `enter-container` and the `torch-ito` environment for Python/tests, make no dependency installs or network calls, and serialize GPU work. After every completed task, update `TASKS.md` with evidence and append commands, tests, changed files, and blockers to `HANDOFF.md`. Complete all CPU/synthetic implementation gates that are safe; stop and report precisely if real data, timestamps, checkpoint, container syntax, or GPU access blocks the next gate. Do not claim scientific success without the G5 pilot.

## Recommended non-interactive launch

From a shell where `codex` is installed, set `PVB_REPO` to the writable repository root and run:

```bash
PVB_REPO=/absolute/path/to/PVB
codex exec \
  -C "$PVB_REPO" \
  --model gpt-5.6-luna \
  -c 'model_reasoning_effort="max"' \
  --sandbox workspace-write \
  --ask-for-approval never \
  "Act as the sole implementation worker for Scheme 1A multi-frame codec v1. Read AGENTS.md and all files in docs/agent/multiframe_codec_v1/ completely, then execute the HANDOFF.md instructions and the first unblocked TASKS.md items in dependency order. Treat physical time as a first-class model condition: carry time_ps/delta_time_ps through temporal attention, compressed latents, target-time decoding, finite-difference losses, time-bucket batching, checkpoints, and stratified evaluation; never equate 100 ps and 1 ns or invent fine labels from coarse interpolation. Update TASKS.md, DECISIONS.md when needed, and HANDOFF.md with evidence. Preserve existing PVB behavior; do not install dependencies, use network access, or run full training."
```

Why these flags:

- `-C` sets the repository before Codex reads project instructions.
- the model and reasoning override pin the requested Luna Max worker;
- `workspace-write` permits repository edits while retaining a sandbox;
- `never` makes unapproved installs/escapes fail rather than block an unattended run.

Do not use `--dangerously-bypass-approvals-and-sandbox` for this task.

## Interactive-session instruction

The project contains `.codex/agents/worker.toml`, which overrides the built-in `worker` role with `gpt-5.6-luna` and max reasoning for this repository. In a Codex session opened at the repository root, say:

> Spawn/use the project-scoped `worker` agent as the sole writer. Give it the exact initial worker prompt in `docs/agent/multiframe_codec_v1/HANDOFF.md`. Wait for it to finish, then review its diff and report completed task IDs, tests, and blockers. Do not spawn additional write agents.

If the client does not expose custom agents, run the non-interactive command above; do not silently substitute another model.

## Inputs the operator may need to provide later

Do not block T01, T04, T05, T06, or synthetic T07 on these values. They are needed for later real-data/GPU gates:

```text
WRITABLE_PVB_REPO=/data4/users/sihao/workspace/PVB
PVB_ENCODER_CKPT=/data1/repo/PVB/ckpt
ATLAS_ROOT=/data1/repo/BioKinema/data_atlas
MISATO_ROOT=/data1/repo/PVB/download_data/MISATO
PROCESSED_DATASET_ROOT=/data4/users/sihao/data/pvb_cross_dataset_20260810
CODEC_GPU_IDS=0,1,2,3
ATLAS_NATIVE_DT_PS=10
MISATO_NATIVE_DT_PS=80
```

The worker must not commit these local paths.

## Evidence log template

The worker should append entries, not replace this prepared state.

```text
### YYYY-MM-DD — Txx

- Status:
- Files changed:
- Commands:
- Tests/results:
- Decisions added or changed:
- Blockers/next task:
```

## Pilot result template


| Time bucket | Clip span | Model             | Temporal ratio | Physical latent interval | Future dRMSD | Velocity RMSE | Acceleration RMSE | Bond RMSE | Clash rate | HF power retention | Peak GPU | Samples/s |
| ----------: | --------: | ----------------- | -------------: | -----------------------: | -----------: | ------------: | ----------------: | --------: | ---------: | -----------------: | -------: | --------: |
|      100 ps |    1.5 ns | framewise control |              1 |                   100 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|      100 ps |    1.5 ns | temporal codec    |              1 |                   100 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|      100 ps |    1.5 ns | temporal codec    |              4 |                  ~400 ps |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | framewise control |              1 |                     1 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | temporal codec    |              1 |                     1 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |
|        1 ns |     15 ns | temporal codec    |              4 |                    ~4 ns |          TBD |           TBD |               TBD |       TBD |        TBD |                TBD |      TBD |       TBD |

Final go/no-go decision: **TBD after G5; no result has been fabricated in this planning pass.**

### 2026-08-18 — T00/T01/T02 data worker

- Status: T00 complete; T01 core contract and T02 ATLAS/MISATO path implemented; full half-data materialization running next. mdCATH is blocked because no raw root or timestamp metadata was supplied in this task.
- Files changed: `utils/bio_utils.py` (NumPy 2.x dtype fix); `data/clip_dataset.py`; `data/trajectory_clips.py`; `scripts/preprocess_trajectory_clips.py`; `tests/test_clip_data.py`; this ledger.
- Baseline: branch `main`, upstream commit `c08e5e3cd49d45c6d748387e78224843bd356f50`, dirty state initially only untracked `agents/`. Valid invocation is `enter-container`, then `source /home/sihao/miniforge3/bin/activate torch-ito`. Environment: Python 3.11.15, NumPy 2.4.6, PyTorch 2.5.1+cu121, CUDA 12.1, 8 visible NVIDIA A100-SXM4-80GB GPUs. `python -c 'import torch; import data; from module.model import dyVAE; print(...)'` passed.
- Commands/tests: `python -m pytest -q tests/test_clip_data.py` — 6 passed; `python -m py_compile data/clip_dataset.py data/trajectory_clips.py scripts/preprocess_trajectory_clips.py utils/bio_utils.py tests/test_clip_data.py` — passed.
- Real-data evidence: one ATLAS system with 3 replicas and `clip_len=4`, `source_stride=1000` produced 6 clips; one MISATO system with `clip_len=4`, `source_stride=1` produced 25 clips. Both stores round-tripped through `ClipMMapDataset` and collated to packed `[4, N_total, 3]`; ATLAS native XTC time was verified as 10 ps, and MISATO uses the explicit 80 ps value from this handoff because the HDF5 itself has no timestamp field. A default-style ATLAS one-system run (`clip_len=16`, `window_stride=16`, `source_stride=10`) produced 186 clips, `dt_100ps`, `[16, 4508, 3]`, and 185,411,817 compressed data bytes.
- Decisions added or changed: no architectural decision changed. The new path is additive and does not alter the legacy `x0/x1` preprocessors. ATLAS 10 ps frames are decimated to an explicitly labeled 100 ps bucket; no coarse-to-fine interpolation is used. MISATO remains an explicitly labeled 80 ps bucket.
- Blockers/next task: write the requested half-system stores to `PROCESSED_DATASET_ROOT=/data4/users/sihao/data/pvb_cross_dataset_20260810` using `--fraction 0.5`, then validate manifests/counts and update this file. `/data4/scratch/sihao` is read-only inside the container; it was used only for disposable smoke output via `/tmp` instead.

### 2026-08-18 — T01/T02 full ATLAS and MISATO materialization

- Status: complete for the requested ATLAS and MISATO half-data; mdCATH remains blocked because this task supplied neither a raw root nor timestamp metadata. The static T=1 loading line remains pending as a separate integration task; the clip schema itself validates and serializes static T=1 records without frame replication.
- Output: `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps` and `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps`. Both use `schema_version=pvb.clip.v1`, `storage_format=npz-v1`, three non-overlapping system splits, and compressed per-record NumPy payloads with JSON metadata.
- ATLAS result: fraction `0.5`, seed `20260810`, selected `546/94/76` train/valid/test systems, `101556/17484/14136` records, `133176` total, `systems_skipped=0`. Native XTC timestamps were checked at 10 ps; this run uses source stride 10, producing an explicit 100 ps bucket with no interpolation.
- MISATO result: official split preserved, fraction `0.5`, seed `20260810`, selected `6985/790/822` train/valid/test systems, `41910/4740/4932` records, `51582` total, `systems_skipped=0`. MISATO HDF5 has no timestamp field, so the handoff-supplied verified native interval of 80 ps is recorded explicitly; source stride is 1 and no interpolation is used.
- Commands: full runs used `python scripts/preprocess_trajectory_clips.py atlas --root /data1/repo/BioKinema/data_atlas --output-root <staging> --fraction 0.5 --seed 20260810 --clip-len 16 --window-stride 16 --source-stride 10` and the corresponding MISATO command with `/data1/repo/PVB/download_data/MISATO` and `--source-stride 1`. After staging, every split was read from the final target with `ClipMMapDataset`; first/middle/last records were validated and two-record batches were collated.
- Validation: target-side readback passed for all six stores, including index lengths, `stats.json` storage format, `T=16`, time buckets, physical deltas, NumPy array fields, packed coordinates, and bond offsets. Final clip tree is approximately `84G`; no temporary staging directory remains.
- Files changed: `data/clip_dataset.py` (versioned npz-v1 writer/reader, canonical validation/collation; reader returns NumPy arrays), `data/trajectory_clips.py`, `scripts/preprocess_trajectory_clips.py`, `utils/bio_utils.py`, `tests/test_clip_data.py`, `agents/TASKS.md`, and this handoff.
- Tests: `python -m pytest -q tests/test_clip_data.py` — `9 passed`; `python -m py_compile data/clip_dataset.py data/trajectory_clips.py scripts/preprocess_trajectory_clips.py utils/bio_utils.py tests/test_clip_data.py` — passed. Existing legacy pair preprocessors were not changed except for the NumPy 2.x `np.compat.long` dtype fix.
- Next task: T03 task-aware dynamic batching, followed by the model codec gates. Do not treat the generated clip stores as evidence of model quality; no training or G5 result was run.

### 2026-08-18 — T02 static T=1 integration

- Status: complete for the existing ANI1x, PCQM4Mv2, and PDBBind static block stores. mdCATH was intentionally skipped per operator instruction and remains outside this handoff.
- Implementation: `legacy_static_record_to_clip()` and lazy `StaticClipDataset` in `data/clip_dataset.py` adapt legacy gzip-JSON `x0`/`b0` records to canonical static clips with `x=[x0]`, `bpos=[b0]`, `time_ps=[0]`, empty `delta_time_ps`, `task=STATIC`, and `time_bucket_id=static`. The adapter does not rewrite the old stores and uses NumPy view expansion rather than replicating frames to `T=16`.
- Metadata: existing atom order, coordinates, block positions, bond indices, masks, and PDBBind `edge_mask` are preserved. Legacy stores do not contain original atom names/indices, so the adapter records deterministic stored-order identities and labels their provenance as `legacy_block_order`; block ids are stably recovered from block-center position/type pairs.
- Files changed: `data/clip_dataset.py`, `data/__init__.py`, `tests/test_clip_data.py`, `agents/TASKS.md`, and this handoff.
- Unit tests: `python -m pytest -q tests/test_clip_data.py` — `11 passed`; `python -m py_compile data/clip_dataset.py data/__init__.py tests/test_clip_data.py` — passed; `import data` plus `StaticClipDataset` import — passed.
- Real-data validation against `/data4/users/sihao/data/pvb_cross_dataset_20260810/blocks`: all train/valid/test stores for all three sources loaded successfully. Counts/shapes were ANI1x `2050623/285072/161913` with first-record shapes `(1,10,3)/(1,13,3)/(1,12,3)`, PCQM4Mv2 `1351433/168929/168930` with `(1,18,3)/(1,17,3)/(1,17,3)`, and PDBBind `6413/367/167` with `(1,2169,3)/(1,2241,3)/(1,1333,3)`. Every checked record validated as `T=1`; representative batches had shape `[1, N_total, 3]` and static task id `0`.
- Next task: T03 task-aware dynamic batching. No model training or quality claim was made.

### 2026-08-18 — T03 task-aware dynamic batching

- Status: complete. The new clip path now forms deterministic homogeneous dynamic batches with an initial `T*N` bound, independent task/bucket mixture weights, DDP batch sharding, and batch diagnostics. No model training or quality claim was run.
- Files changed: `data/clip_batching.py`, `data/clip_dataset.py`, `data/__init__.py`, `tests/test_clip_batching.py`, `agents/TASKS.md`, and this handoff.
- Sampler: `TaskAwareClipBatchSampler` groups by `(task, frames, time_bucket_id)`, packs records while the sum of `frames * atoms` stays below `max_tokens`, and supports `task_weights`, `time_bucket_weights`, `batches_per_epoch`, `replacement`, `seed`, and `set_epoch`.
- DDP: one global deterministic batch schedule is generated and assigned by `global_batch_id % num_replicas`; ranks therefore do not receive the same logical batch and no padding duplicate is introduced.
- Metadata/logging: indexed trajectory stores avoid payload decompression for atom/frame/bucket specs; PDBBind static specs read actual atom counts because its legacy property zero is an adjacency budget. `summarize_clip_batch` and `ClipBatchLogger` report atoms, frames, effective tokens, task type, native delta time, and physical clip span.
- Commands/tests: inside `enter-container` with `torch-ito`, `python -m py_compile data/clip_batching.py data/clip_dataset.py data/__init__.py tests/test_clip_batching.py` passed; `python -m pytest -q` passed with `16 passed`. The T03 suite covers deterministic packing/bounds, zero-weight task and bucket exclusion, two explicit DDP ranks with disjoint batch sets, logs, indexed store readback, and DataLoader collation.
- Decisions added or changed: none. This is additive; legacy pair training and the existing `DynamicBatchWrapper` remain unchanged.
- Blockers/next task: no T03 blocker. Proceed to T04 batched PVB frame encoder; do not start full training before the later codec gates.

### 2026-08-18 — T04 batched PVB frame encoder

- Status: complete for the CPU/synthetic T04 gate. This is an additive spatial adapter; the legacy `dyVAE` path was not modified.
- Files changed: `module/multiframe_codec.py`, `module/__init__.py`, `tests/test_multiframe_codec.py`, `tests/test_multiframe_codec_equivariance.py`, `tests/test_multiframe_codec_optimized.py`, `agents/TASKS.md`, and this handoff.
- Implementation: `PVBFrameEncoder` expands per-atom metadata across time, assigns `graph_id = frame_id * B + abid`, vectorizes frame-offset covalent bond replication, builds distance edges in the same graph-id space, unions missing covalent edges, calls one shared `TorchMD_VQ_ET`, and restores `[T,N_total,C]`/`[T,N_total,3,C]` outputs. `FrameGraphBatch` exposes all graph/topology evidence. Checkpoint initialization returns matched/missing/unexpected/shape-mismatch keys.
- Commands/results: `python -m py_compile module/multiframe_codec.py module/__init__.py tests/test_multiframe_codec.py tests/test_multiframe_codec_equivariance.py tests/test_multiframe_codec_optimized.py` passed. Final `python -m pytest -q` passed with `21 passed` after the default-backend fallback fix.
- Coverage: T=1 and T=16 shapes, one-call behavior, frame/sample isolation, covalent offsets, scalar translation invariance, vector rotation equivariance, checkpoint key reporting, and default neighbor-backend behavior all pass in CPU synthetic tests.
- Environment/blocker: the current `torch-ito` installation exposes `TorchMD_VQ_ET.distance` but lacks `get_neighbor_pairs_kernel`, producing `NameError`. `neighbor_backend=auto` catches this and uses the dependency-free dense CPU finder; explicit `neighbor_backend=optimized` remains strict. No GPU smoke, real-data encoder run, or training was started.
- Decisions/next: no new architectural decision. Proceed to T05 causal temporal codec blocks; retain the optimized-kernel issue as an environment note for later GPU profiling.

### 2026-08-18 — neighbor backend repair and T05 temporal gate

- Status: the environment blocker found after T04 was repaired; T05 causal temporal feature gate is complete. T06 coordinate decoder and T07 trainer remain pending.
- Neighbor repair: `utils/torchmd_utils.py` now defines the missing `get_neighbor_pairs_kernel` through the installed non-periodic `torch_cluster.radius_graph`. `OptimizedDistance` returns the expected edge index/vector/weight/num-pairs tuple; periodic box requests fail explicitly. Explicit optimized neighbor construction passed a batched isolation check, and the full CPU suite passed.
- Original backbone: PVB's existing `TorchMD_VQ_ET` remains the spatial backbone; no replacement with an unrelated model was made. The original pair path still uses `module.graph.construct_edges`, while the new adapter uses the repaired optimized distance path.
- T05 implementation: `module/temporal_codec.py` adds physical-time RBF/MLP attention bias, local causal scalar/vector attention, SO(3)-safe vector channel normalization/FFN, ratio 1/2/4 causal compression, right-edge time/mask propagation, target-time causal upsampling, and refinement blocks.
- Tests: `python -m pytest -q tests/test_temporal_codec.py` — `7 passed`; full `python -m pytest -q` — `27 passed` with the existing `torch.load` FutureWarning only.
- Real GPU smoke: on A100 GPU 4, real ATLAS `atlas_5e3e_A_R1_w000000` (`T=16,N=887`) passed spatial optimized-neighbor encoding, ratio-4 temporal encoding, target-time decoding, backward, and AdamW update; latent `(4,887,16)`, decoded `(16,887,16)`, `0.756 s`, peak allocated `803.2 MiB`. This is a smoke result, not a quality or full-training claim.
- Next task: implement T06 joint coordinate decoder/refiner, then T07 losses and the isolated codec trainer before launching a real experiment.

### 2026-08-19 — T06 joint coordinate decoder and refiner

- Status: complete for the T06 CPU/synthetic acceptance gate; T07 is next. The decoder is additive and leaves the existing PVB `TorchMD_VQ_ET` class and legacy model/checkpoint path unchanged.
- Implementation: `module/coordinate_decoder.py` defines `CodecLatent`, `CoordinateDecoderOutput`, `JointMultiFrameDecoder`, and `LatentConditionedSpatialRefiner`. The decoder queries all target timestamps through `CausalTemporalDecoder`, predicts equivariant coordinate displacements from vector latents, and anchors them at explicit frame-0 `x_anchor`. It has no target-coordinate argument.
- Refiner: one existing `TorchMD_VQ_ET` call receives a flattened `[T*N_total]` graph. Static `z/b/batch/edge_index/bond_type` topology is replicated and frame-offset; `edge_weight_0/edge_vec_0` use repeated `x_anchor`, while `edge_weight_t/edge_vec_t` use `x_coarse`. The wrapper returns refined scalar/vector features and an equivariant displacement.
- Validation: `python -m pytest -q tests/test_coordinate_decoder.py` — 5 passed; full `python -m pytest -q` — 33 passed, with only the existing `torch.load(weights_only=False)` warning. Tests cover `[T,N,3]` shapes, target-time sensitivity, signature-level information-leak prevention, SE(3), gradients, one-call batching, a real small-graph TorchMD refiner, and two-clip tiny overfit.
- GPU note: a T06 real-ATLAS GPU attempt was made after the CPU gate, but this container currently reports `torch.cuda.is_available() == False` and NVML initialization failure. No T06 GPU result or coordinate-quality claim is made. The earlier T05 GPU smoke remains recorded above.
- Next action: implement T07 masked physical-unit losses, codec trainer/config/checkpoint round-trip, then continue through the remaining task ledger without pausing at a stage boundary.

### 2026-08-19 — T07 losses, trainer, and checkpoint gate

- Status: complete for the CPU/synthetic acceptance gate; T08 is next. The new path is isolated in `trainer/codec_losses.py`, `trainer/codec_trainer.py`, `train_codec.py`, and `config/codec.yaml`; existing `train.py` and legacy model types remain untouched.
- Losses: coordinate, local/contact pair distance, covalent bond length, explicit-`delta_time_ps` velocity, and centered nonuniform-grid acceleration. Temporal masks require consecutive valid trajectory frames and task id `TRAJECTORY`; static temporal losses return exact zero.
- Normalization/config: train-batch fitting produces per-bucket coordinate/motion scales with minimum-count fallback and epsilon guards. Canonical `time.unit=ps`, continuous time scale, bucket center/tolerance/weight, staged weights, and normalization settings are validated before training. Raw `velocity_raw`/`acceleration_raw` and normalized optimization terms are emitted per bucket, with task metrics.
- Checkpoint: schema `pvb.codec.checkpoint.v1` stores model/optimizer state, step/epoch, validated config, and normalization statistics; schema/missing-field failures are explicit. `PVBCodecModel` composes the existing PVB `TorchMD_VQ_ET` spatial path with T05/T06 modules.
- Tests: `python -m pytest -q tests/test_codec_training.py` — 5 passed; full suite — 38 passed with the existing `torch.load(weights_only=False)` warning. The PVB wrapper completed one CPU optimizer step on a distinct-atom synthetic clip, and checkpoint resume restored parameters and statistics.
- Next action: add T08 evaluation metrics and the ratio-1/no-temporal, ratio-1 temporal, and ratio-4 temporal controls, then proceed to GPU/DDP smoke gates.

### 2026-08-19 — T08 round-trip evaluation and controls

- Status: complete for the tiny-fixture evaluation gate; T09 GPU/DDP smoke is next.
- `evaluation/codec_evaluation.py` reports frame-0/future/all-frame RMSD and dRMSD, bond RMSE, contact error, clash rate, optional torsion change, velocity/acceleration RMSE, FFT frequency retention, latent token count, throughput, wall time, and peak GPU bytes. `eval_codec.py` constructs ratio-1/no-temporal, ratio-1 temporal, and ratio-4 temporal PVB controls from the codec config.
- Every control result is nested under native `time_bucket_id` with native delta, physical span, and ratio-derived latent interval. No cross-bucket mean is emitted; Markdown repeats that constraint.
- Tests: `python -m pytest -q tests/test_codec_evaluation.py` — 2 passed; full suite — 40 passed. A fixture report separates 80 ps, 100 ps, and static buckets and distinguishes anchor/no-temporal from a perfect temporal predictor.
- Next action: run reproducible one-device `T=16` ratio-4 forward/backward/optimizer/validation/checkpoint smoke, then attempt the two-rank DDP smoke if two idle GPUs are actually available.

### 2026-08-19 — T09 smoke gate

- Initial status before container device-cgroup repair: CPU/synthetic smoke was complete; GPU/NCCL acceptance was environment-blocked.
- `python smoke_codec.py --cpu-fallback --output /tmp/pvb_codec_smoke.json` passed on the actual `PVBCodecModel` with `T=16`, temporal ratio 4, `dt_100ps` and `dt_1ns` synthetic clips, optimizer step, validation, checkpoint save/load, `resumed_step=2`, `wall_time_s=0.1221`, and `peak_memory_bytes=0` (CPU).
- `python smoke_codec.py --ddp` passed with two CPU Gloo ranks. The script performs rank-specific input, one DDP backward/update, and an all-reduce. The smoke code uses one flattened PVB spatial call per batch; no per-frame spatial model loop is introduced.
- Initial GPU attempt before container repair: `python smoke_codec.py --device cuda:4` failed before model construction because `torch.cuda.is_available() == False`; NVML returned `Unknown Error`.
- Root-cause check: host-side `nvidia-smi -L` and `nvidia-container-cli info` both see all eight A100s, while inside `sihao-dev` opening `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia4` returns `EPERM` despite mode `0666`; direct `cuInit` returns `CUDA_ERROR_NO_DEVICE`. The container device-cgroup allow-list is missing.
- Pilot launch/config templates are in `agents/PILOT_COMMANDS.md`; after repair, the commands were executed on GPU 5 and the NCCL smoke used GPUs 0 and 5.

### 2026-08-19 — T10 pilot handoff

- Status: fixed-seed pilot execution complete after the container GPU repair; no final go/no-go claim is made from the pilot sample alone.
- `agents/PILOT_COMMANDS.md` contains fixed-seed commands for the three G5 controls, per-bucket runs (dt100/dt80/1ns), evaluation checkpoints, smoke checks, and the result table.
- `train_codec.py` supports dataset-root, seed, control, and checkpoint overrides; `eval_codec.py` accepts one checkpoint per control. Paths are rooted in `PROCESSED_DATASET_ROOT` from this handoff.
- Three checkpoints were trained for 1000 steps with 256-batch normalization fitting, and `outputs/g5/codec_eval.json`/`.md` contain the 32-batch native-bucket report (48 clips per control across dt100/dt80). Results are observations for threshold review, not a final go/no-go claim.
- Next action: review the pilot metrics against project quality thresholds and extend validation or launch the next experiment only after that decision.

### 2026-08-19 — T10 pilot execution

- `outputs/g5/ratio1_no_temporal/codec_step_00001000.pt` and `outputs/g5/ratio1_temporal/codec_step_00001000.pt` completed 1000 fixed-seed steps; `ratio4_temporal/codec_step_00001000.pt` completed likewise.
- `outputs/g5/codec_eval.json` and `outputs/g5/codec_eval.md` were generated on GPU 5 with 32 validation batches, native `dt_100ps`/`dt_80ps` stratification, 48 clips per control, and nonzero GPU peak accounting.
- A full quality/go-no-go decision is intentionally deferred until the report is compared with the project thresholds.

### 2026-08-20 - Training logs and visualization

- Status: complete for the requested training diagnostics; the final quality/go/no-go decision is still intentionally deferred to project thresholds.
- Files changed: trainer/codec_trainer.py, train_codec.py, tests/test_codec_training.py, scripts/plot_codec_results.py, agents/PILOT_COMMANDS.md, agents/TASKS.md, and this handoff.
- Logging: CodecTrainer.run(..., log_path=..., log_every=...) appends pvb.codec.train.v1 JSONL records. train_codec.py defaults to <save-dir>/train_metrics.jsonl and exposes --log-path/--log-every.
- Execution: the three fixed-seed controls were rerun on GPU 5 for 1,000 steps. Original outputs/g5/ checkpoints were preserved; logged rerun checkpoints are outputs/g5_logged/ratio*/codec_step_00001000.pt, with 1,000 log records in each outputs/g5/ratio*/train_metrics.jsonl.
- Evaluation/plots: outputs/g5_logged/codec_eval.json and .md were generated with 32 validation batches and 48 clips per control across native dt_100ps/dt_80ps. outputs/g5_logged/plots/ contains loss curves, reconstruction/structure plots, dynamics/PVB-style metric plots, and eval_metrics.csv.
- Metrics visualized: frame-0/future RMSD and dRMSD, future bond RMSE, contact error, clash rate, velocity/acceleration RMSE, and FFT frequency retention; the JSON retains the full bucket-stratified report.
- Tests: python -m pytest -q tests/test_codec_training.py - 6 passed; full python -m pytest -q - 41 passed with the existing torch.load warning; changed Python files compile successfully.
- Limitations/next action: loss curves are from same-seed/config diagnostic reruns rather than the original pilot checkpoints; there is no real 1 ns bucket. Compare the report against project quality thresholds before a go/no-go decision.


### 2026-08-20 — Unmodified PVB_origin four-condition baseline comparison

- Status: complete. The untouched reference repository at /data4/users/sihao/workspace/PVB_origin, commit c08e5e3cd49d45c6d748387e78224843bd356f50, was used for all four conditions and remains clean. PDB was excluded.
- Conditions: A static checkpoint direct; B dynamic checkpoint direct; C static checkpoint retrained; D dynamic checkpoint retrained. Checkpoints are the requested top entries under /data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt and /data1/repo/PVB/ckpt/misato_from_author_pretrain/version_0/checkpoint/epoch1_step4.ckpt, with C/D final checkpoints recorded in their output directories.
- Shared evaluation policy: seed 20260810, max_tokens 80000, max_batches 32, 48 selected validation clips (ATLAS dt_100ps: 20; MISATO dt_80ps: 28), rollout T=16 with 15 sequential transitions, and 10 SDE steps. Training policy uses ubound_per_batch=5000, max_batches=500 per source, max_epoch=1, and no PDB.
- C completed 1000 train batches with 8 original trainer OOM skips and global_step=992; its 1000-batch valid loss mean is 1.815241908967495. D completed 1000 train batches with global_step=1000; its 1000-batch valid loss mean is 1.3128217451274395.
- A/B/C/D structure and motion results are aggregated under outputs/pvb_origin_baselines/summary/summary.json and summary.md. loss_curves.{png,pdf}, eval_reconstruction.{png,pdf}, eval_dynamics.{png,pdf}, eval_metrics.csv, training_loss.csv, and validation_loss.csv are in the same directory.
- The training logs are static_retrain/train_pilot500.log and dynamic_retrain/baseline_d_train_gpu4_main.log. Raw direct/retrained reports, selected clip manifests, checkpoints, and valid-step reports remain next to them in outputs/pvb_origin_baselines/.
- Compatibility boundary: npz-v1 clips are not the original gzip-JSON MMAPDataset format, so the only adaptation is output-local streaming pair/clip reading. Original PVB dyVAE, DynamicTrainer, collator, and checkpoint serialization were not edited. The plotted values are PVB-style clip structure/motion metrics, not exact original eval_prot.py raw-trajectory TICA/MSM values.
- C and D retain the original checkpoint-specific fine-tuning settings from their respective configs: C lr=5e-5/warmup=1000, D lr=1e-4/warmup=100. The data, seed, split, batch bound, step budget, and evaluation policy are shared; this hyperparameter difference is explicitly retained rather than hidden.
- Validation: python -m py_compile scripts/aggregate_pvb_origin_baselines.py; python -m pytest -q -> 41 passed, 1 existing torch.load FutureWarning; aggregate finite-metric audit passed with 96 metric rows and 2,000 loss records.

### 2026-08-21 — Modified codec ATLAS-only retraining and evaluation

- Status: complete. This run uses the current modified multi-frame codec (`train_codec.py`/`eval_codec.py`), not the untouched `PVB_origin` baseline. PDB, MISATO, and all other datasets were excluded; only ATLAS `dt_100ps` train/valid stores were used.
- Fixed policy: seed `20260810`, `max_tokens=80000`, 256 train batches for normalization fitting, 1,000 optimizer steps per control, `log_every=1`, GPU 7, and 32 validation batches per control (64 ATLAS clips/control). The three controls were serialized to avoid GPU contention.
- Checkpoints/logs: `outputs/atlas_only_modified/ratio1_no_temporal/codec_step_00001000.pt` + `train_metrics.jsonl`; `ratio1_temporal/codec_step_00001000.pt` + log; `ratio4_temporal/codec_step_00001000.pt` + log. Each log contains exactly 1,000 records through step 1,000.
- Training totals decreased from step 1 to step 1,000 as follows: ratio1/no-temporal `1.1371 -> 0.8248` (last-100 mean `1.2241`), ratio1/temporal `1.1702 -> 0.5446` (last-100 mean `0.8728`), and ratio4/temporal `1.3259 -> 0.7067` (last-100 mean `1.0626`). The curves improve but remain minibatch-noisy rather than fully flat/converged.
- Validation reports: `outputs/atlas_only_modified/atlas_only_codec_eval.json` and `.md` include checkpoint-matched normalized weighted loss and all PVB-style metrics, stratified by native bucket. Validation total loss is ratio1/temporal `2.52509`, ratio4/temporal `2.83812`, and ratio1/no-temporal `3.16047`.
- ATLAS `dt_100ps` future metrics (ratio1/no-temporal; ratio1/temporal; ratio4/temporal): future RMSD `2.98163; 2.65981; 2.78176`, future dRMSD `2.01296; 1.78700; 1.87184`, velocity RMSE `0.0102119; 0.00880295; 0.0108576`, acceleration RMSE `0.000164202; 0.000140411; 0.000174638`, future bond RMSE `0.189274; 0.253299; 0.211263`, future contact error `0.000241982; 0.000209706; 0.000145346`, future clash rate `3.2407e-7; 7.0342e-7; 2.8288e-6`, and FFT frequency retention `0.016559; 0.250068; 0.067732`. Full frame-0/all-frame and loss components remain in the JSON.
- Visualization: `outputs/atlas_only_modified/plots/` contains `loss_curves`, `eval_loss`, `eval_reconstruction`, `eval_dynamics` in PNG/PDF plus `eval_metrics.csv`.
- Evaluation implementation now reports validation loss using each checkpoint's saved config, loss schedule, and normalization statistics; the plot/CSV path includes validation total and component losses.
- Validation: `python -m py_compile eval_codec.py evaluation/codec_evaluation.py scripts/plot_codec_results.py` passed; full `python -m pytest -q` passed (`41 passed`, one existing `torch.load(weights_only=False)` warning); output audit passed for checkpoints, 3,000 train records, 192 eval samples, loss fields, and plots.
- Boundary: the two mistakenly started `PVB_origin` ATLAS-only jobs were interrupted before this run and are not used as evidence; the reference repository was not modified.

### 2026-08-24 — ATLAS quarter-data 50-epoch workload probe

- Status: workload/convergence probe only; no final checkpoint or evaluation report was produced. The user requested three controls, 50 complete epochs, and 1/4 of the previously prepared ATLAS half-data.
- Implementation: `train_codec.py` now accepts `--max-epochs` and deterministic `--subset-fraction`, records subset size, batch count, normalization/training/total elapsed time, and supports sparse JSONL logging. `eval_codec.py` accepts the same validation subset fraction. `data/clip_batching.py` now derives cheap sampler metadata for `torch.utils.data.Subset` without decoding every selected NPZ payload.
- Dataset/workload: fraction `0.25`, seed `20260810`; ATLAS train `25,389/101,556` records, valid `4,371/17,484` records; `8,313` batches per complete train epoch; `415,650` optimizer steps per control for 50 epochs; `log_every=100`.
- Measurement: `ratio1_no_temporal` was started on GPU 7 with normalization fitting over 256 batches. Normalization took `9.47 s`. It was intentionally interrupted at step `600` after approximately `444.1 s` wall time, giving about `1.351 step/s`; projected remaining time is approximately `85.3 h` for one control and `10.6 days` for all three serialized controls.
- Loss evidence: six records were written at steps 100–600; the observed minibatch total loss remained noisy (roughly `0.56–1.68` in the first records), so this short probe is not a 50-epoch convergence result. No checkpoint was saved and no evaluation was run.
- The interruption was deliberate to avoid committing a ten-day GPU run without confirming the workload definition.


### 2026-08-25 — ATLAS three-system, nine-trajectory, 50-epoch run

- Status: complete for the requested selected-trajectory experiment. The untouched `PVB_origin` repository was not modified; PDB and MISATO were excluded.
- Candidate selection: `atlas_5e3e_A` (887 atoms), `atlas_1v7r_A` (1503 atoms), and `atlas_2wlt_A` (2499 atoms), each with R1/R2/R3. All three were retained because none was nearly static. Metrics and definitions are in `outputs/atlas_selected_trajectories/candidate_metrics.json` and the CSV companions. The metric run used within-clip Kabsch frame alignment, median adjacent-frame aligned RMSD, FFT high-pass power (bins k>=2, DC/k=1 removed), per-clip RMSF, and maximum valid-atom local displacement.
- Data selection: `outputs/atlas_selected_trajectories/clip_store/manifest.json` selects 9 trajectories, windows 0–48 for train (441 clips) and 49–61 for validation (117 clips). Train/validation sample-ID intersection is zero. The subset stores only contain new indexes and symlink to the original ATLAS `data.bin`; source payloads were not copied or changed.
- Training policy: seed `20260810`, GPU 0, `max_tokens=80000`, 256 normalization batches, 50 complete loader epochs, 144 train batches/epoch, 7,200 optimizer steps/control, `log_every=10`. Each control has 720 JSONL records and a `codec_step_00007200.pt` checkpoint. Elapsed training times: ratio1/no-temporal 5,133.43 s; ratio1/temporal 5,755.06 s; ratio4/temporal 6,132.27 s. The first incompatible `log_every=100` attempt was interrupted before a checkpoint and moved to `ratio1_no_temporal_aborted_log100/`.
- Full evaluation: after fixing the no-replacement sampler to cover all groups deterministically, the report evaluates all 117 validation clips for each control (`41` validation batches; no `max-batches` limit). Final report: `outputs/atlas_selected_trajectories/evaluation/codec_eval.json` and `.md`. The earlier 114-clip report is preserved as `codec_eval_partial_114.*`.
- Final dt_100ps validation summary (ratio1/no-temporal; ratio1/temporal; ratio4/temporal): total loss `0.698359; 0.300450; 0.516587`; frame-0 RMSD `0.476461; 0.144682; 0.067204`; future RMSD `1.355227; 0.865605; 1.105771`; future dRMSD `1.007538; 0.652273; 0.823784`; velocity RMSE `0.00566139; 0.00418887; 0.00615658`; acceleration RMSE `9.55481e-05; 6.98217e-05; 1.04060e-04`; frequency retention `0.122828; 0.707480; 0.329217`.
- Visualizations: `outputs/atlas_selected_trajectories/evaluation/plots/` contains `loss_curves`, `eval_loss`, `eval_reconstruction`, `eval_dynamics` in PNG/PDF and `eval_metrics.csv`. `summary.json` gathers selection, training, evaluation, timing, and validation metadata.
- Implementation/tests: `scripts/measure_atlas_dynamics.py`, `scripts/build_atlas_trajectory_manifest.py`, and the no-replacement sampler coverage fix in `data/clip_batching.py` were added/updated. Full `python -m pytest -q` passed: 42 tests, with the existing `torch.load(weights_only=False)` warning only. The training epoch count follows the existing replacement-sampling training policy; the validation path is now exact no-replacement coverage.


## 2026-08-25 — molvid graph engineering v1.2 Phase A handoff

The reviewed engineering packet was read in full. Phase A implementation is on branch `fix/graph-runtime-v1` and was accepted under an explicit CUDA graph-cap policy; Phase B is the next action after the Phase-A commit.

Command and evidence:

```text
PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py
outputs/engineering_v1/phase_a/acceptance.json
outputs/engineering_v1/phase_a/repaired_contract/
```

The runner fails immediately if `torch.cuda.is_available()` or the CUDA `torch_cluster.radius_graph` extension is unavailable. It does not select CPU. The fixed real batch is the recorded five ATLAS clips (`T=16,N=4435,B=5`); the graph tensors and bond union are CUDA-resident, the three historical control state dictionaries load with unchanged parameter names/shapes/counts, and the supplied directed bonds are marked exactly once. The same evidence contains BF16 finite/closeness results, a 200-step CUDA optimizer run with warmup-before-step and checkpoint sampler state, five exact no-replacement tiny epochs, pinned two-worker loading, explicit oversize failure, and throughput.

The historical baseline caveat is material and is not hidden: the old CPU `torch_cluster` implementation used nanoflann's unsorted radius-result traversal, while CUDA scans source indices. On the fixed capped batch, the repaired CUDA set differs by `12539` removed and `12512` added distance edges; covalent bond edges are identical. The code now makes the CUDA source-order cap explicit, writes a repaired CUDA reference contract, and records `old_cpu_numerical_identity_claimed=false`. Therefore no historical CPU bitwise output/loss/gradient identity or convergence/quality claim is made. This is the only non-identity boundary in the Phase-A report; it is an audited policy reconciliation, not a silent fallback.

Runtime recorded in the evidence: Python 3.11.15, torch 2.5.1+cu121, CUDA 12.1, `torch_cluster 1.6.3+pt25cu121`, NVIDIA A100-SXM4-80GB. The next worker action is to commit Phase A and run the required CUDA-only `distance_only` Phase-B diagnostics and short checks.


## 2026-08-25 — molvid graph engineering v1.2 Phase B handoff

Phase A was accepted and committed as `c2ad46d`; Phase B was then run on the same CUDA runtime without a CPU fallback.

Command and evidence:

```text
PYTHONPATH=/workspace/PVB python scripts/run_phase_b_acceptance.py
outputs/engineering_v1/phase_b/acceptance.json
```

The Phase-B implementation uses `bond_construction.mode: distance_only`. It scans the combined ATLAS selected-trajectory train/valid clip stores, chooses the lexicographically earliest sample per stable topology id, and registers only its canonical FP32 coordinates in a bounded CUDA cache. The inference API has no atom-type, block-type, supplied-bond, or atom-source-index argument. The graph-input mutation gate changed none of the distance edges, inferred bonds, union edges, distances, or bond flags.

The three reference diagnostics were:

- `atlas_1v7r_A`: inferred/supplied undirected edge counts 1600/1543; precision 0.964375, recall 1.0, F1 0.981864.
- `atlas_2wlt_A`: 2628/2534; precision 0.964231, recall 1.0, F1 0.981790.
- `atlas_5e3e_A`: 955/913; precision 0.956021, recall 1.0, F1 0.977516.

Every graph had one connected component and zero isolated atoms. The strict boundary test passed for `0.5 < d <= 2.2`. Identical initialization (86/86 state keys), two 200-step CUDA runs with checkpoint/resume, and two five-epoch exact-coverage runs passed under the same seed, AdamW, fp32, loader, and warmup contract. The short-run timing and logs are recorded in the JSON and output directory.

The precision below one is expected from the requested distance-only ablation: all pairs in the interval are inferred, including chemically near but non-covalent neighbors that are absent from the supplied bond list. No convergence, scientific quality, historical CPU identity, full-data, or long-run claim is made.


### 2026-08-26 — Selected three-system real-data overfit

The selected systems had not previously been overfit-tested on real data; the earlier overfit evidence was only the two-clip synthetic decoder unit test. I ran exactly three settings, not six: the three full-width modified-code topology controls, each from fresh initialization, with the same fixed train-only batch of `atlas_5e3e_A_R1_w000000`, `atlas_1v7r_A_R1_w000000`, and `atlas_2wlt_A_R1_w000000`. The batch has `T=16`, `N=4889` packed atoms, and `78224` atom-frame tokens. No validation samples were used.

Command:

```text
PYTHONPATH=/workspace/PVB python scripts/run_selected_three_system_overfit.py
```

Evidence is under `outputs/atlas_selected_trajectories/overfit_three_systems/run_20260826T094444/`: `protocol.json`, `summary.json`, per-control `result.json`, 50 loss records at steps 10 through 500, and step-500 checkpoints. The run was CUDA-only on an NVIDIA A100-SXM4-80GB with torch 2.5.1+cu121 and torch_cluster 1.6.3+pt25cu121.

| Control | Initial total | Final total | Reduction | Training s | Total s |
|---|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 1.604981 | 0.689677 | 57.03% | 900.655 | 907.993 |
| ratio1/temporal | 1.693397 | 0.320974 | 81.05% | 1025.914 | 1032.150 |
| ratio4/temporal | 1.697092 | 0.403721 | 76.21% | 1069.343 | 1074.834 |

All three controls decreased aggregate loss; the per-system final totals are retained in each `result.json`. Total wall time was `3016.762 s` (50.28 min). This test establishes that the current models can reduce training loss on the selected examples; it does not establish held-out performance or scientific quality.


### 2026-08-26 — Selected three-system distance-only real-data overfit

The three-setting real-data overfit was repeated with the Phase-B distance-only graph path. It used the exact same fixed train batch and controls as the topology run: atlas_5e3e_A_R1_w000000, atlas_1v7r_A_R1_w000000, and atlas_2wlt_A_R1_w000000; T=16, 4,889 packed atoms, 78,224 atom-frame tokens, no validation samples, fresh initialization, AdamW/fp32, 500 optimizer steps, and loss logging every 10 steps. This is three additional settings; the topology and distance-only rounds together form the six-setting bond-mode ablation.

Command:

~~~text
PYTHONPATH=/workspace/PVB python scripts/run_selected_three_system_overfit.py --bond-mode distance_only
~~~

Evidence:

~~~text
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/summary.json
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/<control>/result.json
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/<control>/train_metrics.jsonl
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/<control>/codec_step_00000500.pt
~~~

| Control | Initial total | Final total | Reduction | Training s | Total s |
|---|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 1.604809 | 0.725967 | 54.76% | 440.992 | 444.366 |
| ratio1/temporal | 1.698666 | 0.304935 | 82.05% | 631.662 | 636.098 |
| ratio4/temporal | 1.697084 | 0.396296 | 76.65% | 921.927 | 926.117 |

Whole-run elapsed time was 2,008.609 s (33.48 min). The protocol records three distance-only canonical references, each taken from the first frame of its selected train clip. Inference uses only those FP32 coordinates and the CUDA distance cache; supplied atom labels, block labels, atom-source indices, and supplied bond indices are not used to construct the inference graph. Runtime was CUDA-only on an NVIDIA A100-SXM4-80GB, torch 2.5.1+cu121, torch_cluster 1.6.3+pt25cu121.

All three aggregate training losses decreased. The result is a memorization/capacity diagnostic on the selected training examples; it is not held-out validation or a scientific-quality claim. The distance-only interval/extra-edge caveat from the Phase-B acceptance remains applicable.


### 2026-08-26 - Distance-only overfit evaluation and visualization

The missing visualization/evaluation deliverables were completed after the distance-only training run. The new evaluator is scripts/evaluate_selected_three_system_overfit.py; this round is invoked with --bond-mode distance_only (topology uses --bond-mode topology). It loads the three original step-500 distance-only checkpoints, registers the same three canonical first-frame references on CUDA, and evaluates the same three selected train clips as three one-clip batches with the existing PVB-style metrics.

Report files:

~~~text
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/codec_eval.json
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/codec_eval.md
~~~

Plot files:

~~~text
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/loss_curves.png
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/loss_curves.pdf
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_loss.png
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_loss.pdf
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_reconstruction.png
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_reconstruction.pdf
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_dynamics.png
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_dynamics.pdf
outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/plots/eval_metrics.csv
~~~

The report contains one dt_100ps row per control. Values are the unweighted mean over the three selected train systems, not held-out validation:

| Control | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Future bond RMSE | Velocity RMSE | Acceleration RMSE | Frequency retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 0.768195 | 0.944104 | 1.362365 | 0.986531 | 0.359487 | 0.00679080 | 0.000113960 | 0.428431 |
| ratio1/temporal | 0.315043 | 0.812106 | 0.815740 | 0.611587 | 0.286700 | 0.00513803 | 0.0000858913 | 0.656172 |
| ratio4/temporal | 0.417614 | 0.353831 | 0.915999 | 0.678755 | 0.168251 | 0.00705834 | 0.000120152 | 0.395787 |

PVB-style evaluation wall times were 608.821 s, 575.310 s, and 547.617 s; total metric-evaluation time was 1,731.747 s (28.86 min). The existing evaluator performs pairwise dRMSD/contact calculations on CPU after CUDA inference, which is the bottleneck. The loss plot uses all 50 training records (steps 10-500), with smoothing window 5.

The generic report and plot code now derives the displayed split label from evaluation_data.source_split. This report therefore says Train rather than Validation; no validation or generalization claim is made.


### 2026-08-26 - Topology overfit evaluation and visualization

- [x] Evaluate the three existing topology step-500 checkpoints with the same PVB-style round-trip metrics used for distance-only.
- [x] Generate the topology loss curve, evaluation-loss, reconstruction, dynamics plots, and metrics CSV.
- [x] Keep the topology and distance-only artifacts in separate run directories and label both reports as train-split diagnostics.

The topology training itself had already completed; its evaluation/visualization deliverables were accidentally omitted and were repaired without retraining. The first relaunch exposed a missing evaluator import before metric computation; it was fixed, rerun, and completed normally.

Evidence: `outputs/atlas_selected_trajectories/overfit_three_systems/run_20260826T094444/codec_eval.json` and `codec_eval.md`; plots are under the same run directory's `plots/` folder as `loss_curves`, `eval_loss`, `eval_reconstruction`, `eval_dynamics` in PNG/PDF and `eval_metrics.csv`.

Evaluation used the same three selected train clips as three one-clip batches; values are the unweighted mean over the three systems, not held-out validation. The `dt_100ps` results were:

| Control | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Future bond RMSE | Velocity RMSE | Acceleration RMSE | Frequency retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 0.738477 | 0.923680 | 1.328758 | 0.964085 | 0.360980 | 0.00672182 | 0.000112805 | 0.450899 |
| ratio1/temporal | 0.332049 | 0.812389 | 0.842627 | 0.630928 | 0.293975 | 0.00520169 | 0.0000868247 | 0.646337 |
| ratio4/temporal | 0.426532 | 0.378307 | 0.926879 | 0.691459 | 0.177938 | 0.00708774 | 0.000120524 | 0.389279 |

PVB-style evaluation wall times were 572.452 s, 57.610 s, and 9.465 s (sum 639.528 s, 10.66 min); GPU inference and CPU pairwise metric work are both included. The loss curve uses all 50 training records (steps 10 through 500) with smoothing window 5. Together with the distance-only run, all six settings now have training logs, checkpoints, eval reports, and plots.

## 2026-08-27 — Luna review repair handoff

The review prompt `LUNA_REVIEW_FIX_PROMPT.md` and all required agent documents were read before editing. Initial repository state was recorded as branch `fix/graph-runtime-v1`, HEAD `9ad0b7c99718839a9a20008bbb04152646c119e0`, with only the prompt untracked. Work was performed inside `enter-container` with the `torch-ito` environment. The host sandbox could not start the container directly because of a bubblewrap loopback setup error; the approved `enter-container` path was used. No packages, network, GPU process termination, destructive Git operation, or reference-repository edit was performed.

### Implemented repair areas

- P1-A: `trainer/codec_trainer.py::prepare_batch_then_to_device` is the shared ordering boundary. `eval_codec.py`, `smoke_codec.py`, trainer loss/evaluation, and selected-system evaluation prepare topology on the CPU batch before transferring tensors. `PVBCodecModel.forward` no longer hides a GPU preparation pass. Production `cuda_radius` remains CUDA-only; `dense_test` is explicit test infrastructure.
- P1-B/P1-C: `trainer/codec_contract.py` provides deterministic JSON-safe contract comparison. `trainer/codec_trainer.py` emits `pvb.codec.checkpoint.v2` with model/graph, distance-reference, optimizer, normalization, sampler, and config data; validates semantic compatibility before mutation; rolls back on apply failure; permits only a larger `max_steps`; and requires explicit flags for v1. `eval_codec.py` and `scripts/evaluate_selected_three_system_overfit.py` reconstruct/check the checkpoint contract and reject conflicting YAML/CLI bond settings.
- P1-D: `scripts/run_engineering_acceptance.py` now compares scalar/vector/decoded outputs, every configured loss component, all trainable parameter gradients, position gradients, and configured tolerances against an immutable external manifest. It also has true tensor-relative BF16 comparison, production-CUDA SE(3), and a cap-unsaturated CUDA-vs-dense synthetic gate. `scripts/build_engineering_reference.py` creates the external reference only as a separate explicit operation.
- P2-E/cache: `module/bond_sources.py`, `train_codec.py`, and `scripts/run_phase_b_acceptance.py` use train-only canonical references. Reference metadata includes sample/split/atom-count/identity/coordinate/inferred-bond hashes. `DistanceOnlyBondCache` keeps canonical CPU coordinates for rehydration after bounded GPU eviction and checks atom count/identity.
- P2-F: `data/clip_dataset.py` captures host atom counts, identity hashes, and task IDs; `trainer/codec_trainer.py` validates the immutable clock before transfer and uses host task metadata for per-task labels; `module/multiframe_codec.py` consumes host metadata; `module/neighbor_graph.py` removes CUDA-to-CPU graph construction and production scalar extraction; `scripts/profile_codec_runtime.py` records data wait, step latency, local scalar extraction, and synchronization events.
- P2-G: `scripts/check_tracked_artifacts.py` inventories tracked binary files and provides a CI guard; `agents/TRACKED_ARTIFACT_INVENTORY.json` stores the current inventory. Existing generated blobs were not removed or rewritten.

### Verification

Commands run inside the container:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_luna_review_contracts.py tests/test_codec_training.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q data module trainer eval_codec.py train_codec.py smoke_codec.py scripts
PYTHONDONTWRITEBYTECODE=1 python scripts/check_tracked_artifacts.py --output /tmp/pvb_tracked_artifacts.json
```

The targeted contract/codec run passed 17 tests, and the final full suite passed 53 tests with one pre-existing `torch.load(weights_only=False)` warning. The artifact inventory reports 30 tracked binary files totaling 2,043,945,238 bytes (~1.90 GiB); it includes `.pt/.pth/.ckpt` files and therefore the `--check` guard is expected to fail on the current historical tree. This is an explicit operator blocker, not a reason to delete files automatically.

### Acceptance boundaries

The visible A100 GPUs were occupied during this repair (GPU 0/1 near 71 GiB, GPU 3–7 at high utilization; compute processes were present). Consequently I did not start long or competing jobs. The repaired real-batch reference manifest is absent at `outputs/engineering_v1/baseline_contract/repaired_cuda_reference/manifest.json`. I invoked `PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py --allow-legacy-checkpoint --legacy-bond-mode topology`; it stopped at `_load_repaired_references` with `PHASE_A_HARD_FAIL` and did not start training or write acceptance evidence. The exact deferred commands are `PYTHONPATH=/workspace/PVB python scripts/build_engineering_reference.py --allow-legacy-checkpoint --legacy-bond-mode topology`, followed by the same acceptance command, after GPU availability and the external artifact destination are reviewed. No current before/after profiler pair exists: `PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode topology` or `--mode distance_only` can produce an after profile when a free GPU is available, but the script does not invent a pre-fix baseline. Production-CUDA evaluator, two-rank DDP, real fixed-batch FP32/BF16/SE(3), and current before/after synchronization results therefore remain unverified. The synthetic CUDA cache-capacity/identity test and all CPU/dense tests are covered by the suite.

The code-level and unit-test statuses are: P1-B fixed, P1-C fixed, P2-E fixed, cache scalability fixed; P1-A and P1-D partially verified pending real CUDA/reference evidence; P2-F partially verified pending profiler evidence; P2-G blocked pending operator-approved artifact migration/history cleanup. These are not convergence or scientific-quality claims.

### Files changed

Core changes: `data/clip_dataset.py`, `data/clip_batching.py`, `module/bond_sources.py`, `module/multiframe_codec.py`, `module/neighbor_graph.py`, `trainer/codec_contract.py`, `trainer/codec_trainer.py`, `eval_codec.py`, `train_codec.py`, and `smoke_codec.py`.

Acceptance/evaluation changes: `scripts/run_engineering_acceptance.py`, `scripts/build_engineering_reference.py`, `scripts/profile_codec_runtime.py`, `scripts/check_tracked_artifacts.py`, `scripts/run_phase_b_acceptance.py`, `scripts/run_selected_three_system_overfit.py`, and `scripts/evaluate_selected_three_system_overfit.py`. Tests: the existing dense/equivariance/optimized tests plus `tests/test_luna_review_contracts.py`. Metadata: `agents/TRACKED_ARTIFACT_INVENTORY.json`.

No unrelated pre-existing user changes were altered. The working tree intentionally remains uncommitted so the operator can review the complete diff and decide the artifact-history operation.

### 2026-08-27 — Continued GPU validation

GPU 5 was initially available, so the following commands were run inside enter-container with the torch-ito environment:

~~~text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/build_engineering_reference.py --allow-legacy-checkpoint --legacy-bond-mode topology
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py --allow-legacy-checkpoint --legacy-bond-mode topology
~~~

The first acceptance invocation exposed a real harness error in the synthetic CUDA gate. batch_cpu.x.grad was retained as the dense reference, then .to('cuda') preserved the autograd edge and the CUDA backward accumulated another gradient into the same CPU leaf. This made the reference exactly twice as large and produced the misleading 0.5 relative error. scripts/run_engineering_acceptance.py::_synthetic_cuda_gate now clones the dense position gradient and sets the CPU leaf gradient to None before transfer.

The corrected acceptance was rerun successfully:

~~~text
{"path": "outputs/engineering_v1/phase_a/acceptance.json", "status": "passed"}
~~~

Measured real fixed-batch details from that run:

- runtime: NVIDIA A100-SXM4-80GB, torch 2.5.1+cu121, torch_cluster 1.6.3+pt25cu121;
- fixed batch: [16, 4435, 3], five atlas_5e3e_A train clips;
- FP32 controls: ratio1_no_temporal, ratio1_temporal, and ratio4_temporal all passed output/loss/parameter-gradient/position-gradient comparisons;
- BF16 total-loss relative errors: 2.7426e-5, 2.3154e-3, and 5.3857e-5; coordinate relative errors were 2.3143e-5, 2.3063e-3, and 5.6907e-5;
- production CUDA SE(3) passed;
- optimizer 200-step test took 13.2603 s and the five tiny epochs took 0.6443 s;
- throughput multipliers against the recorded baseline were 2.099x, 2.287x, and 2.142x, above the required 1.2x.

The two after-only profiler commands also completed:

~~~text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode topology --warmup 2 --steps 5
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode distance_only --warmup 2 --steps 5
~~~

The JSONs were written to outputs/engineering_v1/profiling/topology.json and distance_only.json. Before the final explicit-cardinality repair, topology measured median step latency 2.04412 s and data wait 0.09299 ms/step; distance-only measured 1.73151 s and 0.05210 ms/step. The profiles are after-only and include the profiler's explicit per-step torch.cuda.synchronize(); they are not before/after improvement evidence. They recorded aten::_local_scalar_dense counts of 3185 and 3055 across five active steps, which led to tracing the dependency source: torch_cluster.radius_graph calls int(batch.max()) unless batch_size is supplied.

Follow-up repair: module/neighbor_graph.py::CudaRadiusNeighborList.__call__ now requires a positive host-derived batch_size and forwards it to torch_cluster.radius_graph; module/multiframe_codec.py::PVBFrameEncoder._distance_edges/build_graph supplies frames * batch_size; canonical distance registration supplies 1. The source audit now requires the explicit argument. The full suite after this repair is:

~~~text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
53 passed, 1 warning
~~~

The last GPU snapshot after profiling was:

~~~text
0: 71213 MiB / 90%
1: 71213 MiB / 89%
2: 7819 MiB / 89%
3: 8419 MiB / 71%
4: 50997 MiB / 98%
5: 6711 MiB / 85%
6: 36337 MiB / 95%
7: 533 MiB / 56%
~~~

Thus no safely idle GPU pair was available for the remaining two-rank NCCL DDP smoke or for replaying the full Phase-A/profile commands against the post-batch_size source. No process was killed or preempted. When a pair is genuinely idle, run:

~~~text
CUDA_VISIBLE_DEVICES=<host_gpu_a>,<host_gpu_b> PYTHONPATH=/workspace/PVB python smoke_codec.py --ddp --ddp-gpus 0,1 --output outputs/engineering_v1/ddp_smoke.json
CUDA_VISIBLE_DEVICES=<host_gpu_a> PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py --allow-legacy-checkpoint --legacy-bond-mode topology
CUDA_VISIBLE_DEVICES=<host_gpu_a> PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode topology --warmup 2 --steps 5
CUDA_VISIBLE_DEVICES=<host_gpu_a> PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode distance_only --warmup 2 --steps 5
~~~

The first two acceptance/profile reports must be regenerated after the explicit-cardinality patch before marking the current tree fully validated. The P2-G artifact migration/history operation remains operator-gated.


### 2026-08-27 — Post-patch GPU validation completed

Host GPUs 0 and 1 were idle enough for the independent two-rank smoke, and host GPU 5 was used for the serialized acceptance, profiling, and evaluator checks. No existing GPU process was killed or preempted.

DDP command and result:

~~~text
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/workspace/PVB python smoke_codec.py --ddp --ddp-gpus 0,1 --output outputs/engineering_v1/ddp_smoke.json
{"ddp":{"backend":"nccl","devices":[0,1],"loss":0.01264182198792696,"passed":true,"peak_memory_bytes":22441472,"world_size":2}}
~~~

The final acceptance replay used:

~~~text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py --allow-legacy-checkpoint --legacy-bond-mode topology
~~~

It completed with:

~~~text
{"path":"outputs/engineering_v1/phase_a/acceptance.json","status":"passed"}
~~~

The latest fixed-batch acceptance ran on an NVIDIA A100-SXM4-80GB with torch 2.5.1+cu121 and torch_cluster 1.6.3+pt25cu121. The batch shape was [16, 4435, 3] from five atlas_5e3e_A training clips. All three controls passed the FP32 output/loss/trainable-gradient/position-gradient comparisons, production CUDA SE(3), optimizer and epoch gates, source audit, and throughput gate. The latest throughput multipliers were 2.13184x, 2.43708x, and 2.39894x for ratio1/no-temporal, ratio1/temporal, and ratio4/temporal. The latest BF16 total-loss relative errors were 2.77953e-5, 2.32849e-3, and 3.45390e-5 in that same order.

Post-patch profiling commands:

~~~text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode topology --warmup 2 --steps 5
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python scripts/profile_codec_runtime.py --mode distance_only --warmup 2 --steps 5
~~~

Results:

| Mode | Median step | Mean step | Mean data wait | Local scalar events |
|---|---:|---:|---:|---:|
| topology | 0.910600657 s | 0.907951669 s | 0.05807 ms/step | 3175 |
| distance_only | 0.653270512 s | 0.654015613 s | 0.05285 ms/step | 3045 |

Both profiles include the profiler's explicit per-step torch.cuda.synchronize(), so the recorded synchronization self-CPU totals (about 2.372 s and 2.387 s across the profile) are not a claim of hidden forward synchronization. These are post-patch measurements only; no pre-fix baseline exists in the recorded artifacts, so no before/after speedup is reported. The source audit now reports auto_backend=false, cuda_graph_scalar_extraction=false, cuda_neighbor_cpu_round_trip=false, cuda_neighbor_scalar_extraction=false, explicit_radius_batch_size=true, and device_assertions=torch._assert_async.

Standard evaluator command:

~~~text
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/PVB python eval_codec.py --config config/codec.yaml --device cuda:0 --valid-root outputs/atlas_selected_trajectories/clip_store/valid --max-batches 1 --json outputs/engineering_v1/standard_eval_one_batch.json --markdown outputs/engineering_v1/standard_eval_one_batch.md --bond-mode topology
~~~

This completed for all three controls on one validation batch containing five selected records. The JSON/Markdown artifacts contain frame-0 RMSD, future RMSD/dRMSD, bond RMSE, velocity/acceleration RMSE, and frequency-retention fields. No checkpoint was supplied, so the values are a CUDA execution/metric-path diagnostic for freshly initialized models, not a trained-model quality or convergence result.

Final verification after the repair:

~~~text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
53 passed, 1 warning
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q data module trainer eval_codec.py train_codec.py smoke_codec.py scripts
~~~

Current boundary: P1-A and P1-D real-GPU acceptance is now passed; P2-F has after-only profiling but no before/after comparison; P2-G remains blocked pending operator approval for tracked-artifact migration/history cleanup. No long training or full-data evaluation was started by this validation pass.


### 2026-08-28 — Tracked binary cleanup completed

The operator approved cleanup of generated binaries in Git. The inventory identified 30 tracked binary files: 29 generated files under outputs/ and the small source resource module/equiformer_v2/Jd.pt. I removed exactly the 29 outputs paths from the Git index with git rm --cached. Their local working-tree copies remain available for reproducing or inspecting the previous experiments.

The removed set totals 2,043,923,541 bytes (approximately 1.90 GiB). It includes the atlas-only and selected-system checkpoints, topology/distance-only overfit checkpoints, G5 checkpoints, PVB-origin baseline checkpoints, and engineering acceptance/reference checkpoints. No local checkpoint was physically deleted.

The repository now has output-binary ignore rules for .pt, .pth, .ckpt, .safetensors, .pkl, and .pickle. After refreshing agents/TRACKED_ARTIFACT_INVENTORY.json, the guard reports:

~~~text
tracked_binary_count: 1
tracked_binary_bytes: 21697
policy.violations: []
~~~

The full test suite after cleanup is 53 passed with one pre-existing torch.load warning. Git history was not rewritten and no force-push was attempted. Therefore the current index is clean of generated output binaries, but old objects remain in historical commits until a separately reviewed history-purge operation is performed.


### 2026-08-31 — GitHub push rejection repaired

The attempted push was rejected because six engineering reference/checkpoint files in older branch commits exceeded GitHub's 100 MB per-file limit. Removing them only from the latest index was insufficient because GitHub validates all objects reachable from the pushed history.

The current branch was repaired without deleting the old history:

- old tip: 1b05e4f3f5fba864849ae8a34fecb7b2628659fa;
- clean tip: 3944e0404f71e4c0d9f98e85f984b8672ad5c086;
- clean tip parent: origin/main at 6459d3c0e823491b09295930bf6d5929a8f9961b;
- preserved old history: fix/graph-runtime-v1-history-with-binaries;
- candidate reachable blobs over 100 MB: 0.

The clean tip uses the current source/metadata tree, which already excludes generated output binaries. The local worktree remains clean. The docs-only follow-up commit records this cleanup before the normal push. No force-push is needed because git ls-remote reported no existing origin/fix/graph-runtime-v1 ref after the rejection.


### 2026-08-31 — Push authentication blocker

The requested normal push of fix/graph-runtime-v1 was attempted after the binary-free history repair. GitHub accepted the connection far enough to ask for the HTTPS username, but this container has no configured credential helper and no gh login. I aborted at the username prompt; no password or token was entered, stored, or exposed, and the remote branch remains absent.

Once the operator authenticates GitHub in the container or runs the command from an authenticated terminal, the candidate is ready for:

~~~text
git push -u origin fix/graph-runtime-v1
~~~

This is not a Git history or file-size blocker anymore.
