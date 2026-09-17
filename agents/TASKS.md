# Multi-frame codec v1 task ledger

Status notation: `[ ]` pending, `[-]` in progress, `[x]` complete, `[!]` blocked. A task is complete only when its evidence is written below the task.

## T00 — Baseline and environment capture

- [x] Confirm writable branch/worktree and record upstream commit plus dirty files.
- [x] Determine the valid `enter-container` invocation and activate `torch-ito` inside it.
- [x] Record Python, PyTorch, CUDA, GPU, and key dependency versions in `HANDOFF.md`.
- [x] Run the narrowest available upstream import/smoke check without changing data or installing packages.
- [x] Identify an optional PVB encoder checkpoint and a tiny local static/trajectory fixture; absence is not a blocker for synthetic tests.

Acceptance: baseline commands and outputs are summarized in `HANDOFF.md`; no user-owned changes were overwritten.

## T01 — Freeze the clip schema and tests

- [x] Add a typed/dataclass-like `ClipBatch` contract and validation helpers.
- [x] Add synthetic fixtures for variable atom counts, `T=1`, `T=16`, bonds, masks, 100 ps timestamps, 1 ns timestamps, and an irregular positive time grid.
- [x] Implement packed collation to `[T, N_total, 3]` with `atom_ptr` and `abid`.
- [x] Carry `time_ps`, `delta_time_ps`, and `time_bucket_id`; derive/check intervals in exactly one canonical helper.
- [x] Reject mixed task types/time buckets, inconsistent atom order, inconsistent topology, invalid masks, non-monotonic time, and non-positive intervals.
- [x] Test bond index offsets and sample/frame graph ids.

Acceptance: CPU tests cover every field and failure mode documented in `PLAN.md §4`.

## T02 — Ordered clip preprocessing and loading

- [x] Add clip generation for ATLAS without changing `preprocess_atlas()` pair output.
- [x] Add clip generation for MISATO without changing `preprocess_misato_pl()` pair output.
- [!] Add clip generation for mdCATH without changing `preprocess_mdcath()` pair output (intentionally skipped by operator; no raw path/timestamps supplied).
- [x] Implement deterministic non-overlapping windows with configurable `clip_len`, `window_stride`, and source stride; random training windows remain a loader follow-up.
- [x] Require native frame interval or per-frame timestamp metadata in every implemented trajectory configuration.
- [x] Align every trajectory frame to frame 0 using a protein `align_mask`, applying the same transform to all complex atoms.
- [x] Preserve atom identity/order, timestamps, sample origin, and transform/debug metadata.
- [x] Forbid coarse-to-fine target interpolation; ATLAS 10 ps data is explicitly decimated to 100 ps for this run.
- [x] Implement static `T=1` loading without frame replication.
- [x] Measure serialized clip size before deciding whether existing mmap JSON/gzip is sufficient.

Acceptance: a tiny real or locally generated trajectory round-trips through storage and collation with coordinates/topology unchanged within serialization tolerance.

## T03 — Task-aware dynamic batching

- [x] Add `T*N` as the initial complexity estimate for clip datasets.
- [x] Keep minibatches homogeneous in task, temporal length, and `delta_time_ps` bucket.
- [x] Add configurable static/trajectory and per-time-bucket sampling weights without coupling them to dataset size.
- [x] Verify DDP samplers do not duplicate the same batch on every rank.
- [x] Log atoms, frames, effective tokens, task type, native `delta_time_ps`, and physical clip span per batch.

Acceptance: deterministic tests show the configured mixture and batch bound; a two-rank dry/smoke test shows distinct rank samples.

### 2026-08-18 — T03 evidence

- `data/clip_batching.py` adds `ClipItemSpec.effective_tokens = T*N`, homogeneous group packing, configurable task/bucket weights, deterministic epoch seeding, and a direct DataLoader builder.
- `ClipMMapDataset` reads atom/frame/bucket metadata from `index.txt`; `StaticClipDataset` uses exact ANI1x/PCQM4Mv2 index counts and reads actual PDBBind atom counts because its legacy property zero is an adjacency budget.
- DDP safety is implemented by building one global batch schedule and sharding by global batch id across explicit `num_replicas`/`rank`; no padding duplicates are introduced.
- `summarize_clip_batch`/`ClipBatchLogger` emit atoms, frames, effective tokens, task type, native delta time, and physical clip span.
- Deterministic CPU coverage: `python -m pytest -q` — 16 passed; `python -m py_compile data/clip_batching.py data/clip_dataset.py data/__init__.py tests/test_clip_batching.py` — passed.

## T04 — Batched PVB frame encoder

- [x] Implement metadata expansion and `(sample, frame)` graph ids.
- [x] Replicate/offset covalent bonds by frame.
- [x] Construct nonbonded graph edges with no cross-frame or cross-sample edges.
- [x] Call the shared `TorchMD_VQ_ET` spatial encoder once for all frames and restore scalar/vector feature shapes.
- [x] Add optional checkpoint initialization with a matched/missing/unexpected-key report.
- [x] Add shape, frame-isolation, translation, and rotation tests.

Acceptance: `T=1` and `T=16` pass; scalar features are invariant and vector features rotate within configured fp32 tolerance.

### 2026-08-18 — T04 evidence

- Status: complete for the CPU/synthetic T04 gate. The additive frame encoder preserves the legacy PVB model path.
- Files changed: `module/multiframe_codec.py`, `module/__init__.py`, three `tests/test_multiframe_codec*.py` files, and this ledger.
- Graph adapter: metadata is expanded time-major; graph ids are `frame_id * batch_size + abid`; covalent bonds are replicated and offset by `frame_id * N_total`; distance and covalent edges are unioned and checked for cross-frame/sample violations.
- Spatial call: one shared `TorchMD_VQ_ET` invocation handles all flattened frames and restores `h=[T,N_total,C]` and `v=[T,N_total,3,C]`. Optional checkpoint loading reports matched, missing, unexpected, and shape-mismatched keys.
- Tests: `python -m py_compile module/multiframe_codec.py module/__init__.py tests/test_multiframe_codec.py tests/test_multiframe_codec_equivariance.py tests/test_multiframe_codec_optimized.py` — passed; `python -m pytest -q` — 21 passed after the final backend fix. T=1/T=16 shape, frame isolation, translation invariance, rotation equivariance, single-call, checkpoint-report, and default-backend tests passed.
- Environment note: the missing optimized neighbor symbol was repaired in `utils/torchmd_utils.py` using the installed `torch_cluster.radius_graph` backend. Explicit `neighbor_backend="optimized"` now works for the non-periodic PVB path; periodic boxes fail explicitly because no box metadata is present in the clip contract.
- Decisions added or changed: none. No GPU/real-data training or quality claim was made. Next task is T05 causal temporal codec blocks.

## T05 — Causal equivariant temporal codec blocks

- [x] Implement mask-aware local causal temporal attention for invariant and vector features.
- [x] Implement continuous relative-time RBF/MLP attention bias from physical `|time_i-time_j|` and configurable time units/scale.
- [x] Implement scalar/vector FFN and normalization without learned xyz-axis mixing.
- [x] Implement causal stride-2 downsampling for ratios 1/2/4.
- [x] Implement repeat/interpolate plus causal refinement upsampling.
- [x] Propagate/downsample frame masks and physical timestamps explicitly; define latent time as the right edge of each causal receptive field.
- [x] Make temporal decoding query requested `target_time_ps` and enforce the documented causal chunk-to-target mapping.
- [x] Add prefix-causality at complete chunk boundaries, padding, all-masked-error, `T=1`, gradient, SE(3), 100 ps versus 1 ns, timestamp-rescaling, and irregular-time tests.

Acceptance: appending future frames does not alter any valid prefix output in eval mode; ratio shapes are exact and documented.


### 2026-08-18 — T05 evidence

- Status: complete for the causal temporal feature gate; T06 remains the separate coordinate decoder/refiner task.
- Files changed: `module/temporal_codec.py`, `module/__init__.py`, `tests/test_temporal_codec.py`, and the neighbor backend in `utils/torchmd_utils.py`.
- Implementation: `CausalEquivariantTemporalBlock` performs local left-looking attention over the same atom, uses physical log-RBF/MLP relative-time bias, applies shared scalar attention weights to vector values, and keeps all linear maps off the xyz axis. Mask-aware scalar/vector FFN and SO(3) channel normalization are included.
- Compression/decoding: `CausalTemporalEncoder` supports ratios 1/2/4; stride-2 chunks propagate masks and store the latest valid timestamp at each causal receptive-field right edge. `CausalTemporalUpsample` accepts required `target_time_ps`, uses only already-available latent right edges plus a continuous time query, and `CausalTemporalDecoder` adds causal refinement blocks.
- Tests: `python -m pytest -q tests/test_temporal_codec.py` — 7 passed; full `python -m pytest -q` — 27 passed. Coverage includes prefix causality, masked padding, all-masked rejection, T=1, gradients, SE(3), 100 ps versus 1 ns conditioning, timestamp rescaling, irregular clocks, ratios, target-time queries, and finite decoder outputs.
- GPU evidence: real ATLAS `atlas_5e3e_A_R1_w000000` (`T=16,N=887`) passed spatial optimized-neighbor encoding, ratio-4 temporal encoding, target-time decoding, backward, and AdamW update on A100 GPU 4; latent shape `(4,887,16)`, decoded shape `(16,887,16)`, `0.756 s`, peak allocated `803.2 MiB`.
- Next task: T06 joint multi-frame coordinate decoder and shared TorchMD refiner; no coordinate quality claim has been made.

## T06 — Joint multi-frame decoder

- [x] Define `CodecLatent(z_h, z_v, latent_mask, latent_time_ps, x_anchor, topology metadata)`.
- [x] Decode all temporal positions jointly to coarse displacements from `x_anchor`.
- [x] Add a latent-conditioned shared TorchMD spatial refiner batched across frames.
- [x] TorchMD was not extended: the refiner wraps the existing PVB `TorchMD_VQ_ET`, so its legacy forward/checkpoint path remains unchanged.
- [x] Ensure decoder signatures cannot receive target `x[t>0]`.
- [x] Require `target_time_ps` in the trajectory decoder API and verify changing it changes temporal decoding without breaking SE(3) behavior.
- [x] Add an information-leak test and a two-clip overfit test.

Acceptance: the decoder outputs `[T, N_total, 3]`, uses no temporal rollout loop, and overfits a tiny synthetic dataset without NaN/Inf.

### 2026-08-19 — T06 evidence

- Status: complete for the joint coordinate-decoder gate. `CodecLatent` carries temporal scalar/vector latents, masks, physical latent clocks, frame-0 `x_anchor`, packed atom ids, and static topology metadata.
- `JointMultiFrameDecoder` queries all requested target times through the T05 causal decoder, predicts an equivariant displacement from vector features, and forms `x_coarse = x_anchor + displacement`; no target coordinates are accepted by its forward signature.
- `LatentConditionedSpatialRefiner` wraps the existing PVB `TorchMD_VQ_ET` with one latent-conditioned call over the flattened `[T*N_total]` graph. Static topology is replicated/offset per frame, while reference geometry is computed from `x_anchor` and decoded geometry from `x_coarse`.
- Tests: `python -m pytest -q tests/test_coordinate_decoder.py` — 5 passed; full `python -m pytest -q` — 33 passed with the existing `torch.load(weights_only=False)` warning only. Coverage includes output shapes, target-time sensitivity, information-leak signature checks, SE(3), gradients, one shared batched refiner call, a real small-graph TorchMD refiner, and two-clip tiny overfit.
- GPU note: a follow-up T06 real-ATLAS GPU smoke was attempted after the CPU gate, but the current container reports `torch.cuda.is_available() == False` and NVML initialization failure; no GPU claim is made for T06. The previously recorded T05 GPU smoke remains valid.
- Next task: T07 losses, isolated codec trainer, configuration, and checkpointing.

## T07 — Losses, trainer, config, and checkpointing

- [x] Implement masked coordinate, local/contact distance, bond, velocity, and acceleration losses.
- [x] Make `delta_time_ps` explicit in temporal losses; implement correct velocity and nonuniform-grid acceleration formulas.
- [x] Add train-split normalization statistics per time bucket with minimum-count/epsilon guards; checkpoint these statistics.
- [x] Set temporal losses exactly to zero for static samples.
- [x] Add staged loss-weight scheduling and per-task logging.
- [x] Log raw physical-unit temporal metrics and normalized optimization losses separately by time bucket.
- [x] Add isolated `train_codec.py`, `trainer/codec_trainer.py`, and `config/codec.yaml`.
- [x] Add explicit `time_unit`, time-bucket center/tolerance/weight, continuous time-embedding, and temporal-normalization configuration with validation.
- [x] Save a versioned config/schema with checkpoints; test save/resume and backward failure messages.
- [x] Keep existing `train.py` and model types functional.

Acceptance: one optimizer step and checkpoint round-trip pass on CPU/synthetic data; old PVB import/checkpoint regression still passes.

### 2026-08-19 — T07 evidence

- Status: complete for the isolated codec trainer gate. `trainer/codec_losses.py` implements masked coordinate, local/contact pair distance, covalent bond length, explicit-clock velocity, and centered nonuniform-grid acceleration losses. Temporal terms gate on trajectory task ids and are exactly zero for static samples.
- `fit_time_bucket_normalization` computes train-split coordinate/motion scales per bucket, applies minimum-count fallback and epsilon clamps, and serializes through `BucketNormalization`. `CodecTrainer` reports raw `velocity_raw`/`acceleration_raw` alongside normalized optimization terms, plus per-task and per-bucket metrics.
- `CodecTrainConfig` validates canonical `time.unit=ps`, continuous time scale, bucket center/tolerance/weight, normalization guards, and staged loss weights. `train_codec.py` is a separate entrypoint using only versioned clip stores; the legacy `train.py` path remains untouched.
- Checkpoints use schema `pvb.codec.checkpoint.v1` and contain model/optimizer state, step/epoch, validated config, and normalization statistics. Invalid schema and missing fields fail with explicit errors.
- Tests: `python -m pytest -q tests/test_codec_training.py` — 5 passed; full `python -m pytest -q` — 38 passed with the existing `torch.load(weights_only=False)` warning only. Coverage includes nonuniform formulas, static zeroing, masks, normalization fallbacks, config rejection, PVB CPU one-step, and checkpoint resume.
- Next task: T08 round-trip evaluation and controls.

## T08 — Round-trip evaluation and controls

- [x] Implement framewise ratio-1/no-temporal control.
- [x] Implement ratio-1 temporal and ratio-4 temporal codec evaluation.
- [x] Report RMSD/dRMSD, bond RMSE, contacts, clashes, velocity/acceleration, torsion change, and frequency retention.
- [x] Report latent token count, throughput, wall time, and peak memory.
- [x] Stratify every temporal metric by native `delta_time_ps`; include physical clip span and latent interval and forbid an unqualified cross-bucket mean.
- [x] Separate frame-0 reconstruction from future-frame metrics.
- [x] Export machine-readable JSON plus a concise Markdown summary.

Acceptance: evaluation runs on a tiny fixture and clearly distinguishes the three controls.

### 2026-08-19 — T08 evidence

- Status: complete for the fixture evaluation gate. `evaluation/codec_evaluation.py` and `eval_codec.py` provide three named controls: ratio-1/no-temporal framewise PVB, ratio-1 temporal, and ratio-4 temporal.
- Metrics include frame-0/future/all-frame RMSD and dRMSD, bond RMSE, contact error, nonbond clash rate, optional torsion change, velocity/acceleration RMSE, FFT frequency retention, latent token count, physical clip span, latent interval, wall time, samples/s, and peak GPU memory.
- Reports are keyed by native `time_bucket_id` and carry native delta, span, and latent interval; there is deliberately no unqualified cross-bucket aggregate. JSON schema is `pvb.codec.eval.v1`; Markdown output is concise and bucket-stratified.
- Tests: `python -m pytest -q tests/test_codec_evaluation.py` — 2 passed; full `python -m pytest -q` — 40 passed with the existing `torch.load(weights_only=False)` warning only. The fixture distinguishes anchor/no-temporal and perfect temporal controls across 80 ps, 100 ps, and static buckets.
- Next task: T09 one-GPU/DDP smoke gates; current container GPU availability must be rechecked before claiming a GPU result.

## T09 — GPU and DDP smoke gates

- [x] Run one-GPU `T=16`, ratio-4 forward/backward/optimizer/validation/checkpoint smoke on A100 GPU 4; CUDA/NVML is restored in the re-entered container.
- [x] Smoke-test synthetic 100 ps/1 ns batches on GPU and real ATLAS 100 ps plus MISATO 80 ps batches on GPU 5 through the full codec path.
- [x] Record GPU peak memory: synthetic smoke 23,566,848 bytes; real ATLAS 2,244,516,352 bytes; real MISATO 2,886,455,296 bytes.
- [x] Run two-GPU NCCL DDP smoke on GPUs 0 and 5 with the actual `PVBCodecModel`, optimizer update, and all-reduce.
- [x] Confirm no hidden per-frame model loop dominates runtime: the smoke uses one flattened PVB spatial call per batch and the T05/T06 temporal path; wall time is recorded.
- [x] Record commands and results in `HANDOFF.md`.

Acceptance: CPU diagnostics and real GPU/NCCL jobs exit cleanly and are reproducible from recorded commands; the three-control pilot and native-bucket report were executed after CUDA was restored.

### 2026-08-19 — T09 evidence

- Initial status before the container device-cgroup repair: CPU diagnostic gate complete; GPU-specific gate was blocked by the runtime environment.
- Command: `python smoke_codec.py --cpu-fallback --output /tmp/pvb_codec_smoke.json` passed. It ran the actual `PVBCodecModel` with `T=16`, ratio 4, one optimizer step and validation on both `dt_100ps` and `dt_1ns`, checkpoint save/load, and reported `wall_time_s=0.1221`, `peak_memory_bytes=0`, `resumed_step=2`.
- Command: `python smoke_codec.py --ddp` passed with two CPU Gloo ranks (`world_size=2`). It verifies process-group initialization, independent rank inputs, one DDP backward/update, and an all-reduce.
- Initial command before container repair: `python smoke_codec.py --device cuda:4` failed with `torch.cuda.is_available() == False` and NVML `Unknown Error`; this is retained as the historical failure that led to the device-cgroup repair.
- Root cause: host-side `nvidia-smi -L` and `nvidia-container-cli info` see all eight A100s, but inside `sihao-dev` opening `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia4` returns `EPERM` despite mode `0666`; direct `cuInit` returns `CUDA_ERROR_NO_DEVICE`. The container device-cgroup allow-list is missing.
- Reproducible commands and executed pilot artifacts are in `agents/PILOT_COMMANDS.md` and `outputs/g5/`; the earlier T05 real-ATLAS smoke remains historical evidence for that stage.

### 2026-08-19 — T09 GPU restored

- Status: complete after the container was re-entered with GPU device permissions.
- `python smoke_codec.py --device cuda:4` passed with CUDA visible; a real ATLAS GPU 5 smoke also passed (`dt_100ps`, `T=16`, peak `2,244,516,352` bytes), and a real MISATO GPU 5 smoke passed (`dt_80ps`, peak `2,886,455,296` bytes).
- `python smoke_codec.py --ddp --ddp-gpus 0,5` passed with NCCL, world size 2, actual PVB codec forward/backward/AdamW/all-reduce, and peak `22,440,960` bytes.
- The smoke path was extended with the `dt_80ps` bucket and real NCCL branch; no hidden per-frame spatial loop was introduced.

## T10 — Pilot experiment handoff

- [x] Provide exact commands for the three G5 controls with fixed seeds and data split.
- [x] Provide per-time-bucket commands/configs and separate go/no-go rows for 100 ps and 1 ns data when available.
- [x] Record dataset/checkpoint paths as operator-supplied placeholders rather than guessing.
- [x] Add a result table template and go/no-go checklist.
- [x] Update README usage after the real CUDA command was smoke-tested; README now points to the isolated pilot path.
- [x] Update `HANDOFF.md` with completed IDs, diffs, tests, limitations, and next action.

Acceptance: an operator can launch the pilot without reading implementation code; no quality claim is made before results exist.

### 2026-08-19 — T10 evidence

- Status: complete for the handoff/documentation gate. `agents/PILOT_COMMANDS.md` contains fixed-seed commands for all three controls, per-bucket commands, smoke gates, operator-supplied dataset/checkpoint placeholders, and a result/go-no-go table.
- The commands route through `train_codec.py`, `eval_codec.py`, and `smoke_codec.py`; each control has an explicit temporal ratio/layer setting and separate output directory. `config/codec.yaml` carries the canonical physical-time and normalization defaults.
- The real CUDA launch, ATLAS/MISATO smoke, three 1000-step controls, and 32-batch evaluation were smoke-tested and executed. No final go/no-go claim is made from the pilot sample alone.
- `HANDOFF.md` records T00–T10 implementation evidence, the repaired container diagnosis, GPU/NCCL evidence, checkpoint paths, evaluation outputs, and the remaining action: apply project-level quality thresholds before a go/no-go decision.

### 2026-08-20 - T10 training logs and visualization

- Status: complete for the requested reproducibility/diagnostics follow-up; the original T00-T10 implementation and pilot gates remain complete, while the final quality/go/no-go decision remains threshold-dependent.
- trainer/codec_trainer.py now appends schema pvb.codec.train.v1 JSONL records with step, epoch, learning rate, total/component losses, normalized/raw velocity and acceleration metrics, task totals, and time-bucket fields. train_codec.py exposes --log-path and --log-every, defaulting to <save-dir>/train_metrics.jsonl.
- All three fixed-seed 1000-step controls were rerun on GPU 5 without overwriting the original checkpoints; each produced 1,000 step records under outputs/g5/ratio*/train_metrics.jsonl and a new checkpoint under outputs/g5_logged/.
- scripts/plot_codec_results.py generated outputs/g5_logged/plots/loss_curves.{png,pdf}, eval_reconstruction.{png,pdf}, eval_dynamics.{png,pdf}, and eval_metrics.csv. The eval plots use the corresponding outputs/g5_logged/codec_eval.json report, stratified by native dt_80ps and dt_100ps.
- Validation: python -m pytest -q tests/test_codec_training.py - 6 passed; full python -m pytest -q - 41 passed with the existing torch.load warning; py_compile and git diff --check passed.
- Limitation: loss curves describe the same-seed/config logged reruns; they are not reconstructed from the original pilot checkpoints. No real 1 ns bucket exists, and no scientific go/no-go claim is made.

## Dependency order

```text
T00 -> T01 -> T02 -> T03
             T01 -> T04 -> T05 -> T06 -> T07 -> T08 -> T09 -> T10
```

T02 and T04 may be developed in parallel only in separate worktrees and only if the user explicitly requests multiple writers. The default Luna worker is the sole writer and executes sequentially.


## 2026-08-20 — Unmodified PVB_origin four-condition baseline

- [x] Run static-pretrain direct inference (A) and dynamic-pretrain direct inference (B).
- [x] Run static-pretrain retraining (C) and dynamic-pretrain retraining (D) with the same ATLAS dt_100ps + MISATO dt_80ps half-data, seed 20260810, ubound_per_batch=5000, 500 train batches/source, one epoch, and no PDB.
- [x] Evaluate all four conditions on the same 32 validation batches / 48 clips (20 ATLAS + 28 MISATO), T=16 sequential rollout, and 10 SDE steps.
- [x] Parse both retraining logs and generate loss curves, native-time reconstruction/dynamics plots, CSV, JSON, and Markdown summary.

Evidence: outputs/pvb_origin_baselines/summary/ contains the aggregate report and plots; A/B raw metrics are in static_direct/ and dynamic_direct/; C/D checkpoints, logs, validation loss, and raw metrics are in the corresponding *_retrain/ directories. The PVB_origin reference remains clean at commit c08e5e3cd49d45c6d748387e78224843bd356f50; full tests: 41 passed.

Important boundary: C/D use the untouched original dyVAE, DynamicTrainer, collator, and checkpoint serialization through output-local streaming adapters because the current npz-v1 clip stores are not original gzip-JSON MMAPDataset records. The structure/motion plots are the same PVB-style clip metrics as A/B, not the original raw-trajectory TICA/MSM evaluator. C uses the original static fine-tune hyperparameters (lr=5e-5, warmup 1000); D uses the original dynamic checkpoint policy (lr=1e-4, warmup 100), while data, seed, batch bound, step budget, split, and evaluation policy are shared.

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


### 2026-08-25 — ATLAS selected-trajectory 50-epoch training/evaluation

- [x] Compute motion diagnostics for the three atom-count candidates and retain only non-static systems.
- [x] Select 3 systems × 3 replicas; split clips 0–48 train and 49–61 validation for each trajectory.
- [x] Complete all three modified-code controls for 50 loader epochs / 7,200 steps each, logging every 10 steps.
- [x] Evaluate all 117 validation clips per control and generate loss/structure/dynamics plots.
- [x] Verify train/validation sample-ID disjointness and add a no-replacement sampler test preventing silent validation tail drops.

Evidence: `outputs/atlas_selected_trajectories/summary.json`, `clip_store/manifest.json`, `candidate_metrics.json`, each control's `train.log`/`train_metrics.jsonl`/`codec_step_00007200.pt`, and `evaluation/codec_eval.{json,md}` plus `evaluation/plots/`.

Final validation is stratified only by the available native `dt_100ps` bucket. The epoch count is the current training sampler's loader-epoch definition; validation uses deterministic no-replacement coverage of every selected clip.


## T11 — molvid graph engineering v1.2 Phase A/B

- [x] Read every file under `molvid_graph_engineering_v1_2_reviewed/` before implementation.
- [x] Replace the production graph path with an explicit CUDA-only radius backend; CPU construction remains only `dense_test`.
- [x] Add bounded CPU canonical/per-device topology caches and GPU bond-code-only union.
- [x] Add FP32 geometry/loss islands, BF16 autocast, pinned multi-worker loading, exact sampler/resume state, and explicit oversize errors.
- [x] Add complete `config/codec_engineering_full.yaml` and runnable Phase-A acceptance evidence.
- [x] Run Phase-A fixed real-batch, BF16, 200-step, five exact tiny-epoch, worker/pinned, warmup/resume, and throughput checks on CUDA.
- [x] Run Phase B only after the Phase-A commit; report distance-only precision/recall/F1/Jaccard, degree/isolated/component diagnostics, and its short checks.

### 2026-08-25 — Phase A evidence

- The acceptance runner is `scripts/run_engineering_acceptance.py`; command: `PYTHONPATH=/workspace/PVB python scripts/run_engineering_acceptance.py`.
- Evidence: `outputs/engineering_v1/phase_a/acceptance.json`, `phase_a/repaired_contract/*.pt`, `topology_200_steps.jsonl`, `topology_200_steps.pt`, and `topology_five_epochs.jsonl`.
- CUDA runtime: torch `2.5.1+cu121`, CUDA `12.1`, `torch_cluster 1.6.3+pt25cu121`, A100-SXM4-80GB. GPU absence remains a hard error in the runtime and acceptance runner.
- Fixed real batch: `T=16`, `N=4435`, `B=5`, `70960` atom-frame tokens. All three controls preserved parameter counts `593992`, `1974904`, and `3356074`; graph tensors stayed on CUDA; supplied directed bond flags were exact and unique.
- Real loader: 2 workers, pinned batch, 152 no-replacement batches, explicit oversize error for the lower-budget probe. BF16 total-loss relative errors were `2.60e-5`, `2.27e-3`, and `2.60e-4` for the three controls.
- The topology tiny run completed 200 optimizer steps with 20 records and checkpoint/resume; five exact tiny epochs covered all 8 clips exactly once per epoch. Repaired BF16 throughput multipliers versus T00 were `2.759x`, `2.319x`, and `2.088x`.
- Important honest boundary: the historical CPU nanoflann cap set is not bitwise identical to CUDA. The audit records `12539` removed and `12512` added distance edges, with `0` bond-edge differences. The repaired contract explicitly uses CUDA source-order capping and does not claim historical CPU numerical identity; no quality/convergence or full-data claim is made.


### 2026-08-25 — Phase B evidence

- Status: complete. Phase A was committed as `c2ad46d` before the Phase-B launch. The Phase-B path is CUDA-only and raises on missing CUDA or `torch_cluster.radius_graph`; no CPU fallback was used.
- Runner: `PYTHONPATH=/workspace/PVB python scripts/run_phase_b_acceptance.py`. Evidence: `outputs/engineering_v1/phase_b/acceptance.json`.
- Canonical references: all 558 train/valid clips were scanned; the lexicographically earliest `sample_id` per stable `topology_id` was selected for the three systems: `atlas_1v7r_A`, `atlas_2wlt_A`, and `atlas_5e3e_A`. The model receives only canonical FP32 coordinates for distance bond inference.
- Distance-only diagnostics (inferred versus supplied undirected bonds): 1v7r has 1600/1543 edges, precision 0.964375, recall 1.0, F1 0.981864; 2wlt has 2628/2534, precision 0.964231, recall 1.0, F1 0.981790; 5e3e has 955/913, precision 0.956021, recall 1.0, F1 0.977516. All inferred and supplied graphs have one connected component and zero isolated atoms; full distance, degree, and component statistics are in the JSON.
- Boundary and independence gates passed: `0.5 < d <= 2.2` including the exact tested upper boundary, and mutating `atype`, `btype`, `block_id`, and supplied `bond_index` did not change graph edges/bond flags.
- Common initialization matched all 86 state keys with seed 20260825. Both modes used the same AdamW/fp32 contract, completed 200 steps with 20 log records and checkpoint/resume, and completed five exact two-batch epochs covering all eight tiny clips each epoch.
- Honest boundary: the distance rule deliberately adds near-neighbor edges beyond the supplied covalent list; these are diagnosed rather than hidden. This is an architecture/graph acceptance result, not a convergence, quality, full-data, or long-run claim.


## 2026-08-26 — selected three-system real-data overfitting

- [x] Check whether the selected ATLAS systems already had a real-data overfit result; only the earlier two-clip synthetic decoder test existed.
- [x] Run the three full-width modified-code topology controls from fresh initialization on one fixed train clip per selected system: `atlas_5e3e_A_R1_w000000`, `atlas_1v7r_A_R1_w000000`, and `atlas_2wlt_A_R1_w000000`.
- [x] Complete 500 optimizer steps per control, log every 10 steps, evaluate the aggregate batch and each system separately, save checkpoints, and record synchronized CUDA wall time.

Evidence: `outputs/atlas_selected_trajectories/overfit_three_systems/run_20260826T094444/summary.json`, each control's `result.json`, `train_metrics.jsonl`, and `codec_step_00000500.pt`. Reproducible runner: `scripts/run_selected_three_system_overfit.py`.

The fixed batch contains 3 clips, 16 frames, 4,889 packed atoms, and 78,224 atom-frame tokens; it uses only the selected training split and no validation clip. This is three settings, one per control. It is not a 3-control x 2-bond-mode six-setting ablation.

Results (aggregate total loss, initial -> final; training time / total time): ratio1/no-temporal `1.604981 -> 0.689677`, 57.03% reduction, `900.655 s / 907.993 s`; ratio1/temporal `1.693397 -> 0.320974`, 81.05% reduction, `1025.914 s / 1032.150 s`; ratio4/temporal `1.697092 -> 0.403721`, 76.21% reduction, `1069.343 s / 1074.834 s`. Whole run wall time was `3016.762 s` (50.28 min), including selection and setup. All three controls have `fit_status=loss_decreased`.

This is a memorization/capacity diagnostic on three train clips, not a validation, generalization, convergence, or scientific-quality claim. GPU was required and used; no CPU fallback was enabled.


### 2026-08-26 — Selected three-system real-data distance-only overfit

- [x] Reuse the selected three-system train-only overfit batch and the three existing controls with bond_construction.mode=distance_only.
- [x] Use one canonical FP32 first-frame reference per selected system; do not use supplied bond/atom topology for distance-only inference.
- [x] Complete 500 optimizer steps per control, log every 10 steps, evaluate the aggregate batch and each system separately, save checkpoints, and record synchronized CUDA wall time.

Evidence: outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/summary.json, each control's result.json, train_metrics.jsonl, and codec_step_00000500.pt. The reproducible runner is scripts/run_selected_three_system_overfit.py --bond-mode distance_only.

The protocol matches the topology overfit gate: the three fixed train clips are atlas_5e3e_A_R1_w000000, atlas_1v7r_A_R1_w000000, and atlas_2wlt_A_R1_w000000; T=16, 4,889 packed atoms, 78,224 atom-frame tokens, no validation samples, fresh initialization, AdamW, fp32, and 10-step logging. This is three additional settings; topology plus distance-only together are the six-setting bond-mode ablation.

| Control | Initial total | Final total | Reduction | Training s | Total s |
|---|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 1.604809 | 0.725967 | 54.76% | 440.992 | 444.366 |
| ratio1/temporal | 1.698666 | 0.304935 | 82.05% | 631.662 | 636.098 |
| ratio4/temporal | 1.697084 | 0.396296 | 76.65% | 921.927 | 926.117 |

The whole distance-only run took 2,008.609 s (33.48 min). All three controls have fit_status=loss_decreased. The run used CUDA cuda:0 on NVIDIA A100-SXM4-80GB with torch 2.5.1+cu121 and torch_cluster 1.6.3+pt25cu121; no CPU fallback was enabled.

This remains a train-set memorization/capacity diagnostic, not a validation, generalization, convergence, or scientific-quality claim. The distance-only graph intentionally follows the Phase-B interval policy and can include near-neighbor edges beyond supplied covalent bonds; full loss components and per-system totals remain in each result.json.


### 2026-08-26 - Distance-only overfit evaluation and visualization

- [x] Evaluate all three distance-only step-500 checkpoints with the existing PVB-style round-trip metrics.
- [x] Generate the loss curve, loss-component/evaluation plot, reconstruction plot, dynamics plot, and CSV.
- [x] Make generic evaluation/report labels split-aware so this train-only overfit report is not mislabeled as validation.

The evaluator is scripts/evaluate_selected_three_system_overfit.py; run this distance-only round with --bond-mode distance_only (topology uses --bond-mode topology). Evidence: outputs/atlas_selected_trajectories/overfit_three_systems_distance_only/run_20260826T105636/codec_eval.json and codec_eval.md; plots are under the same run directory's plots/ folder as loss_curves, eval_loss, eval_reconstruction, eval_dynamics in PNG/PDF and eval_metrics.csv.

Evaluation used the same three selected train clips, as three one-clip batches; the reported metric values are the unweighted mean over the three systems. It is not held-out validation. The dt_100ps results were:

| Control | Train total loss | Frame-0 RMSD | Future RMSD | Future dRMSD | Future bond RMSE | Velocity RMSE | Acceleration RMSE | Frequency retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio1/no-temporal | 0.768195 | 0.944104 | 1.362365 | 0.986531 | 0.359487 | 0.00679080 | 0.000113960 | 0.428431 |
| ratio1/temporal | 0.315043 | 0.812106 | 0.815740 | 0.611587 | 0.286700 | 0.00513803 | 0.0000858913 | 0.656172 |
| ratio4/temporal | 0.417614 | 0.353831 | 0.915999 | 0.678755 | 0.168251 | 0.00705834 | 0.000120152 | 0.395787 |

Per-control PVB-style evaluation times were 608.821 s, 575.310 s, and 547.617 s (sum 1,731.747 s, 28.86 min). The evaluation was CUDA-enabled for model inference; the existing metric implementation moves predictions to CPU for pairwise dRMSD/contact statistics, which explains the long wall time. The loss curve uses the 50 records from steps 10 through 500 with smoothing window 5. The report and plot titles identify the source as Train.


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

## 2026-08-27 — Luna review repair (P1-A through P2-G)

- [~] P1-A: move topology registration before CUDA transfer through the shared `prepare_batch_then_to_device` helper; remove hidden forward preparation; update standard evaluation and DDP smoke callers. Real CUDA evaluator/DDP acceptance remains unverified while GPUs are occupied.
- [x] P1-B: persist the versioned v2 model/graph contract, distance-reference contract, and optimizer contract; reconstruct/check evaluator models; require explicit graph mode for v1 checkpoints.
- [x] P1-C: validate semantic training/sampler/model/normalization contracts before mutation, permit only a larger `max_steps`, and roll back state if application fails.
- [~] P1-D: implement executable FP32/BF16/SE(3) gates and immutable-reference manifest verification. The code is ready, but the external repaired FP32 reference manifest is not present and the GPUs are occupied, so the real fixed-batch gate is intentionally not marked passed.
- [x] P2-E: select distance-only references from the train split only and record sample IDs, split provenance, coordinate/identity hashes, and inferred-bond hashes.
- [~] P2-F: remove production forward-path CUDA scalar extraction/CPU graph copies and add a CUDA profiler. A current after-profile and before/after comparison remain unverified because no free GPU and no pre-fix profiler baseline are available.
- [~] P2-G: add the tracked-artifact inventory/guard and checked-in metadata. The existing branch still contains approximately 1.90 GiB of tracked binary data, including generated checkpoints; moving it to an approved store or reconstructing history requires operator approval and was not attempted.
- [x] Additional cache scalability: retain CPU canonical coordinates so bounded distance-only GPU-cache eviction can rehydrate entries, and validate exact atom count plus atom-identity hash.
- [x] Add contract/regression coverage and run the complete test suite.

Evidence and limits:

- Targeted Luna/codec tests: `17 passed`; final full-suite result must be recorded after the final verification command.
- No long training, full-data evaluation, GPU process termination, package installation, network access, checkpoint deletion, or Git history rewrite was performed.
- The hard-CUDA production policy remains unchanged: missing CUDA or the CUDA radius extension raises an error; `dense_test` is explicit CPU-only test infrastructure.
- `agents/TRACKED_ARTIFACT_INVENTORY.json` records path, size, purpose, and SHA-256 for all currently tracked binary artifacts. `scripts/check_tracked_artifacts.py --check` is expected to fail until the operator approves artifact migration/history cleanup.

### 2026-08-27 — Continued GPU validation

- [x] Generated the immutable repaired-CUDA reference on GPU 5; the manifest contains all three controls.
- [x] Ran the real Phase-A acceptance before the follow-up synchronization fix. It passed the fixed-batch FP32/BF16/SE(3), optimizer, epoch-coverage, source-audit, and throughput gates; evidence is outputs/engineering_v1/phase_a/acceptance.json.
- [x] Fixed an acceptance-harness bug found during the first run: the CUDA backward pass was accumulating into the CPU leaf used as the dense position-gradient reference. The reference is now cloned and the CPU leaf gradient cleared before transfer.
- [x] Ran topology and distance-only after-only profiles. Results are in outputs/engineering_v1/profiling/topology.json and distance_only.json; data wait was 0.093 ms/step and 0.052 ms/step respectively.
- [x] Fixed the remaining known CUDA scalar source in the graph path by passing host-derived batch_size explicitly to torch_cluster.radius_graph; the production neighbor API now rejects an omitted batch size instead of allowing the dependency to call int(batch.max()).
- [x] Re-ran the complete test suite after these changes: 53 passed, 1 pre-existing warning.
- [~] Replaying the full acceptance and after-only profiles on the post-batch_size code, plus the two-rank NCCL DDP smoke, remains pending because the GPU snapshot after profiling showed no safely idle GPU pair.
- [~] P2-G remains an operator-approved artifact migration/history-cleanup task.


### 2026-08-27 — Post-patch GPU validation completed

- [x] Replayed the real Phase-A acceptance after the explicit graph-cardinality repair. The current acceptance artifact is passed and covers the fixed-batch FP32, BF16, optimizer, epoch-coverage, SE(3), source-audit, and throughput gates.
- [x] Ran the two-rank NCCL DDP smoke on host GPUs 0 and 1. The run passed with world size 2, backend NCCL, and loss 0.01264182198792696.
- [x] Ran the post-patch topology and distance-only runtime profilers on host GPU 5. Median step latency was 0.910600657 s and 0.653270512 s, respectively; data-wait means were 0.05807 ms/step and 0.05285 ms/step.
- [x] Ran the standard CUDA evaluator on one validation batch (five selected records) for all three controls. The reports confirm the standard predictor ordering and metric path; because no trained checkpoint was supplied, this is an execution diagnostic, not a quality or convergence result.
- [x] Rechecked the full suite and compilation after the code repair: 53 tests passed with one pre-existing torch.load warning, and compileall passed.
- [~] P2-F has final after-only profiling evidence, but no before/after timing claim is made because no pre-fix profiler baseline was captured. P2-G still requires operator-approved artifact migration/history cleanup.

Evidence: outputs/engineering_v1/phase_a/acceptance.json, outputs/engineering_v1/ddp_smoke.json, outputs/engineering_v1/profiling/topology.json, outputs/engineering_v1/profiling/distance_only.json, outputs/engineering_v1/standard_eval_one_batch.json, and outputs/engineering_v1/standard_eval_one_batch.md.


### 2026-08-28 — Tracked binary artifact cleanup

- [x] Removed exactly 29 generated checkpoint/engineering binaries under outputs/ from the Git index with git rm --cached; local files were preserved.
- [x] Added ignore rules for generated outputs binaries with .pt, .pth, .ckpt, .safetensors, .pkl, and .pickle suffixes.
- [x] Refreshed the tracked-artifact inventory and ran the guard: one intentionally retained source asset remains, 21,697 bytes total, with zero policy violations.
- [~] Historical Git objects were not rewritten or force-pushed. The current index/worktree cleanup is complete; a separate reviewed history purge would be required if repository size must also be reduced retroactively.

Evidence: agents/TRACKED_ARTIFACT_INVENTORY.json, scripts/check_tracked_artifacts.py --check, and the staged index deletions shown by git diff --cached.


### 2026-08-31 — GitHub push cleanup

- [x] Diagnosed the rejected push: the current tree was binary-free, but the branch history still carried 29 large output blobs.
- [x] Built and verified a binary-free branch tip based on origin/main; the entire reachable history has zero blobs over 100 MB.
- [x] Preserved the former branch tip, including its historical binaries, at fix/graph-runtime-v1-history-with-binaries.
- [x] Kept the working branch name fix/graph-runtime-v1 so a normal non-force push can create the previously rejected remote branch.


### 2026-08-31 — Push authentication boundary

- [x] Attempted the requested normal push of the binary-free fix/graph-runtime-v1 tip.
- [~] The push reached GitHub authentication but the container has no HTTPS username/token, credential helper, or gh login. It was aborted without storing credentials; origin/fix/graph-runtime-v1 remains absent.
