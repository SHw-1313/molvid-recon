# Scheme 1A: PVB multi-frame codec v1 implementation plan

## 1. Outcome

Build a causal, SE(3)-equivariant multi-frame codec on top of PVB's TorchMD-Net/TorchNet-style spatial representation. Given an ordered clip with fixed topology,

\[
X_{0:T-1} \in \mathbb{R}^{T \times N \times 3},
\]

the encoder must create a temporally compressed latent clip, and the decoder must reconstruct all frames jointly from that latent plus a first-frame structural anchor. Physical timestamps are part of the representation: two adjacent frames separated by 100 ps and two separated by 1 ns are different conditions even when they occupy the same tensor indices. The model path must not reconstruct a trajectory by repeatedly calling the current one-step `dyVAE.realization()`.

The first release is an engineering and representation-learning proof, not the final molecular world model. It should establish that video-style temporal tokenization can preserve fast motion and local geometry while reducing temporal token count.

## 2. Current upstream diagnosis

Baseline inspected: upstream commit `c08e5e3cd49d45c6d748387e78224843bd356f50`.

The current repository is single-step/Markovian at four coupled layers:

1. `data/atlas_dataset.py` and `data/mdcath_dataset.py` turn trajectories into `(x0, x1)` pairs.
2. `data/collate.py` concatenates atoms and exposes only `x0` plus optional `x1`; there is no explicit time axis.
3. `module/model.py::dyVAE._train()` encodes one coordinate set and learns a bridge from `x0` to a single target structure.
4. `module/model.py::dyVAE.realization()` iterates a single state; its returned list is a rollout trace, not one-shot decoding of a compressed multi-frame token sequence.

`module/torchmd_et.py::TorchMD_VQ_ET` already returns the useful pair:

- invariant node features `h: [N, C]`;
- equivariant vector features `v: [N, 3, C]`.

Those features are the spatial backbone for the new codec. PVB also already provides atom type, block/residue type, per-atom CA/block reference coordinates, covalent bonds, local graph construction, dynamic atom batching, DDP training, and static/MD datasets. The plan reuses those strengths without keeping the pairwise trajectory assumption.

## 3. Scope boundary

### In scope

- ordered trajectory clips of a fixed topology;
- static structures through a nontrivial `T=1` denoising path;
- packed variable-atom batching;
- one batched spatial encoder call over all frames;
- causal scalar/vector temporal mixing;
- continuous physical-time conditioning for different and optionally irregular sampling intervals;
- temporal downsampling by 4 in the target configuration;
- joint temporal upsampling and multi-frame coordinate reconstruction;
- reconstruction, geometry, velocity, and acceleration losses;
- CPU unit tests, one-GPU smoke test, and DDP smoke test;
- pilot comparison against a framewise/no-temporal-compression baseline.

### Explicitly out of scope

- the latent diffusion/flow/DiT world-model trunk;
- rollout beyond the observed clip;
- language/MSA/AF3 conditioning;
- residue or patch token pooling;
- unordered conformer ensembles;
- KL/VAE posterior sampling or VQ codebooks;
- production trajectory generation or replacement of current PVB inference.

These exclusions prevent a failed experiment from being ambiguous: v1 answers whether the multi-frame representation and decoder work.

## 4. Data contract

### 4.1 Canonical `ClipBatch`

Each minibatch is homogeneous in task, temporal length, and sampling-interval bucket: all samples are either static (`T=1`) or ordered trajectory clips (`T=clip_len`) from one compatible `delta_time_ps` bucket. Atom counts may differ and are packed. Buckets alternate during training and share the same codec parameters.

| Field | Shape | Meaning |
|---|---:|---|
| `x` | `[T, N_total, 3]` | target atom coordinates after clip-consistent alignment |
| `bpos` | `[T, N_total, 3]` | PVB block/CA reference position repeated per atom |
| `atype` | `[N_total]` | atom types |
| `btype` | `[N_total]` | PVB block/residue types |
| `abid` | `[N_total]` | sample id for each packed atom |
| `atom_ptr` | `[B+1]` | packed atom offsets |
| `bond_index` | `[2, E_total]` | packed covalent bonds |
| `edge_mask` | `[N_total]` | existing complex/ligand edge semantics |
| `loss_mask` | `[N_total]` | atoms contributing to coordinate losses |
| `align_mask` | `[N_total]` | protein backbone/CA atoms used for Kabsch alignment |
| `frame_mask` | `[B, T]` | valid frames; all true in the initial fixed-length loader |
| `time_ps` | `[B, T]` | monotonic physical timestamps in picoseconds, relative to the clip start |
| `delta_time_ps` | `[B, T-1]` | positive physical interval between adjacent valid frames |
| `time_bucket_id` | `[B]` | batching/logging bucket such as 100 ps or 1 ns; not a replacement for continuous time |
| `task` | `[B]` | `STATIC=0`, `TRAJECTORY=1`; `ENSEMBLE=2` reserved only |
| `sample_id` | list of length `B` | traceability and split leakage checks |

Do not pad the atom axis. Reject a clip if atom identity/order or bonds change across its frames, a valid timestamp is non-monotonic, or `delta_time_ps` disagrees with `time_ps`. Static samples use `time_ps=[[0]]`, an empty `delta_time_ps`, and no temporal derivative loss.

### 4.2 Alignment and gauge

- For protein trajectories, align each frame to frame 0 using the protein backbone/CA `align_mask`, then apply the same rigid transform to every atom, including ligand atoms. This removes irrelevant global roto-translation without erasing protein-ligand relative motion.
- Center the aligned clip once using the frame-0 anchor center. Do not independently recenter ligand and protein.
- Save the transform metadata in preprocessing/debug output so alignment can be audited.
- Static inputs receive coordinate corruption; their target remains the clean structure. This prevents the first-frame anchor from making the static task an identity shortcut.

### 4.3 Clip creation

Add clip preprocessors/loaders rather than changing existing pair datasets in place.

- Initial target: `clip_len=16`, configurable source stride, random start for training, deterministic starts for validation.
- Every dataset configuration must declare its native frame interval in physical units or provide per-frame timestamps. Dataset A at 100 ps and dataset B at 1 ns remain distinct time buckets even when both use `clip_len=16`.
- v1 may require constant `delta_time_ps` within one clip and a narrow tolerance within one batch. The schema and finite-difference code must nevertheless support irregular positive intervals so this restriction can be relaxed without an API break.
- Do not interpolate a 1 ns trajectory into synthetic 100 ps targets. A 100 ps trajectory may be decimated every 10 frames into a separately labeled 1 ns augmentation, enabling a later cross-scale consistency experiment.
- Store one record per clip in the existing mmap format if size is acceptable. If JSON/gzip amplification is excessive, make a separate versioned binary array format only after measuring it; do not silently alter `MMAPDataset`.
- Compute dynamic batch cost from `T * N` initially. Record actual peak memory and revise the complexity formula only with profiling evidence.
- Keep static and trajectory data loaders/batches homogeneous, then alternate them using a task-aware mixture schedule. Do not pad a static item to 16 repeated frames.

## 5. Model architecture

### 5.1 High-level diagram

```text
ordered coordinates x[0:T] + topology + time_ps[0:T]
              |
              |  flatten (sample, frame) into graph batch
              v
   shared PVB/TorchMD spatial encoder (all frames in one call)
              |
       h[T,N,C] invariant
       v[T,N,3,C] equivariant
              |
   causal, physical-time-conditioned equivariant temporal blocks
   - q/k and gates from h
   - continuous relative-time bias from |time_i-time_j|
   - same scalar weights mix h and v
   - no learned xyz-axis mixing
              |
   stride-2 causal downsample x 2
              |
       z_h[L,N,Cz], z_v[L,N,3,Cvz], latent_time_ps[B,L]
                                             L = ceil(T/4)
              |
   target-time queries + causal temporal refinement x 2
              |
       h_dec[T,N,C], v_dec[T,N,3,C]
              |
   equivariant displacement head
              |
  x_coarse[t] = x_anchor + delta[t]
              |
  shared latent-conditioned TorchMD spatial refiner
  (all reconstructed frames batched; no target-frame coordinates)
              |
          x_hat[T,N,3] at requested target_time_ps
```

### 5.2 Spatial encoder adapter

Create a `PVBFrameEncoder` that:

1. expands static metadata across `T` without copying it in the stored dataset;
2. reshapes coordinates to `[T*N_total, 3]`;
3. maps `frame_graph_id = frame_id * B + abid`, which prevents cross-frame edges;
4. offsets/replicates covalent bonds per frame;
5. constructs local edges independently for each frame;
6. calls a shared `TorchMD_VQ_ET` once and restores `[T, N_total, ...]`.

Initialize from an existing PVB encoder checkpoint when supplied. Loading must report matched, missing, and unexpected keys. A missing checkpoint must not prevent from-scratch unit tests.

### 5.3 Causal equivariant temporal block

Implement a local-window temporal module per atom. For each atom trajectory:

- normalize and project invariant `h` to queries, keys, scalar values, and vector gates;
- apply a left-looking window and `frame_mask`;
- compute a continuous relative-time bias
  \[
  b_{ij}=\operatorname{MLP}\!\left(\operatorname{RBF}\!\left(\log\left(1+|t_i-t_j|/\tau\right)\right)\right)
  \]
  from `time_ps`, and add it to attention logits before the causal mask;
- use attention weights derived only from invariant quantities;
- apply the same scalar attention weights to vector value channels;
- use channel-linear maps on the last vector channel dimension while keeping the xyz dimension untouched;
- add scalar and vector residual/FFN paths with mask-aware normalization.

Absolute/elapsed-time embeddings may be added to invariant scalar features, but `time_bucket_id` must not be the only model input: continuous `time_ps`/`delta_time_ps` remain authoritative. Dataset identity must not stand in for time interval.

This is the molecular analogue of factorized video attention: TorchMD performs spatial mixing inside each frame, while temporal blocks mix the same atom across frames. Full neighbor-time tubes are a later extension.

Required tests:

- appending future frames does not change prefix outputs in evaluation mode;
- rotation/translation produces rotated/translated coordinates and invariant scalar latents;
- masked frames do not contribute;
- identical coordinates with 100 ps and 1 ns timestamps produce different temporal features when temporal conditioning is enabled;
- rescaling timestamps changes only invariant conditioning and preserves SE(3) behavior;
- `T=1` is valid and finite.

### 5.4 Temporal compression

Use two causal stride-2 stages for a target temporal ratio of 4.

- Left padding only; no future leakage.
- Downsample scalar and vector channels with scalar gates/weights.
- Downsample `frame_mask` and physical time metadata explicitly. Each latent token stores the right-edge/latest valid physical timestamp of its causal receptive field.
- Decoder uses target-time queries plus repeat/interpolation followed by causal refinement instead of transposed convolution, avoiding temporal checkerboard artifacts. A target frame may use only latent chunks whose causal receptive field is allowed by the documented chunk mapping; it must never use a later chunk.
- Support ratios 1, 2, and 4 from config. Ratio 1 is the required framewise/control baseline.

A ratio of 4 describes token compression, not a universal physical step: it is approximately 400 ps for 100 ps data and 4 ns for 1 ns data. Logs and checkpoints must record both the ratio and the physical time span.

### 5.5 Latent and decoder

The deterministic latent is:

- `z_h: [L, N_total, C_z]` invariant;
- `z_v: [L, N_total, 3, C_v]` equivariant;
- `latent_time_ps: [B, L]` and `latent_time_mask: [B, L]` describing every compressed token;
- first-frame `x_anchor: [N_total, 3]` as an explicit structural gauge/condition, not counted as a learned temporal token.

The temporal decoder accepts `latent_time_ps` and requested `target_time_ps`, then upsamples all requested frames jointly. An equivariant GVP-style head predicts displacement from `x_anchor`. A shared TorchMD spatial refiner then processes all coarse reconstructed frames in one batched call while conditioned on decoded scalar/vector features.

If conditioning requires changing `TorchMD_VQ_ET`, add optional `input_scalar`/`input_vector` dimensions and optional forward tensors. Defaults must reproduce the old zero-vector and embedding-only path exactly. Do not pass the ground-truth `x[t>0]` into the refiner.

## 6. Objectives

For trajectory clips:

\[
\mathcal{L}=w_x L_{coord}+w_d L_{dist}+w_v L_{vel}+w_a L_{acc}+w_b L_{bond}.
\]

- `L_coord`: masked coordinate error after the preprocessing alignment; report both all-atom and CA RMSD. Separate frame 0 from future frames.
- `L_dist`: robust error on covalent bonds plus a fixed union of local/contact pairs selected from the target clip without feeding their target distances to the decoder.
- `L_vel`: finite-difference error in physical units using `v_t=(x_{t+1}-x_t)/\Delta t_t`.
- `L_acc`: nonuniform-grid second finite difference
  \[
  a_t=\frac{2}{\Delta t_{t-1}+\Delta t_t}\left(\frac{x_{t+1}-x_t}{\Delta t_t}-\frac{x_t-x_{t-1}}{\Delta t_{t-1}}\right),
  \]
  enabled after the reconstruction-only warm-up.
- `L_bond`: bond-length regularizer and optional clash penalty.

For static samples, use corrupted input/anchor and only valid coordinate, distance, and bond terms. Set velocity/acceleration losses exactly to zero.

Dividing by `delta_time_ps` does not recover fast motion absent from coarse trajectories. Therefore, normalize temporal loss contributions using train-split statistics per time bucket (with epsilon and minimum-count safeguards), log both normalized losses and raw physical-unit metrics, and never collapse 100 ps and 1 ns velocity/acceleration metrics into one unstratified average. Dataset sampling weights must prevent a large coarse dataset from overwhelming fine-timescale batches.

Training schedule:

1. reconstruction warm-up: coordinate + bond/distance;
2. enable velocity loss;
3. enable acceleration loss at a small weight;
4. only after all gates pass, consider a separate decision for KL/VQ.

Minimum time-related configuration:

```yaml
data:
  time_unit: ps
  time_buckets:
    - {name: 100ps, center_ps: 100.0, tolerance_ps: 1.0, weight: 1.0}
    - {name: 1ns, center_ps: 1000.0, tolerance_ps: 10.0, weight: 1.0}
model:
  time_embedding:
    reference_scale_ps: 100.0
    num_rbf: 32
    use_log_delta: true
loss:
  temporal_normalization: per_time_bucket
```

Bucket centers/tolerances are data-loader policy, not categorical replacements for timestamps. Save the resolved bucket definitions, time-embedding configuration, and temporal normalization statistics in every codec checkpoint.

## 7. Training and evaluation plan

### Gate G0 — upstream safety

- Capture environment and baseline import/forward behavior.
- Add a checkpoint-load regression for the unchanged PVB path.
- Do not proceed if the target is not a writable clone/worktree.

### Gate G1 — clip data contract

- Synthetic and real tiny clip samples collate to the documented shapes.
- 100 ps, 1 ns, and irregular-time synthetic clips preserve correct timestamps and bucket assignment.
- Atom ordering/topology mismatch fails loudly.
- Static and trajectory batches both work; mixed-task items in one packed batch are rejected in v1.
- Complexity uses `T*N` and excludes batches above the configured bound.

### Gate G2 — framewise codec control

- `temporal_compression=1` and temporal block disabled.
- All frames are processed in one graph-batched call.
- A two-sample synthetic batch can overfit; no NaN/Inf.
- Equivariance and frame-isolation tests pass.

### Gate G3 — causal temporal codec

- Enable temporal blocks and compression 2, then 4.
- Latent length equals `ceil(T/ratio)` according to one documented padding rule.
- Causal-prefix tests pass.
- Relative-time conditioning, latent timestamp propagation, target-time decoding, and complete-chunk prefix tests pass for both 100 ps and 1 ns examples.
- Decoder reconstructs the full tensor without a temporal rollout loop.

### Gate G4 — mixed static/dynamics training

- Task-aware loader alternates `T=1` static and `T=16` trajectory batches.
- Within trajectory training, the loader alternates physical-time buckets without mixing incompatible intervals in one minibatch.
- Loss masks and logging are task-specific.
- One-GPU smoke run completes forward, backward, optimizer, checkpoint save/load, and validation.
- DDP smoke run completes on the available 2–4 A100 setup without each rank consuming identical samples.

### Gate G5 — pilot decision

Compare three models with the same spatial width and training subset. Run and report the comparison separately for every available physical-time bucket:

1. PVB-inspired framewise control, ratio 1, no temporal mixing;
2. temporal codec, ratio 1;
3. temporal codec, ratio 4.

Report:

- future-frame aligned RMSD and dRMSD;
- bond RMSE, contact precision/recall, and clash rate;
- velocity and acceleration RMSE;
- backbone torsion-change error;
- high-frequency motion power retention;
- latent token count, samples/s, and peak GPU memory.
- native `delta_time_ps`, physical clip span, and physical latent interval.

Suggested go/no-go criteria, to be confirmed after the first baseline run:

- ratio-4 future-frame dRMSD no worse than 1.10× the ratio-1 temporal codec;
- ratio-4 velocity RMSE better than the framewise control;
- no material degradation in bond RMSE or clash rate;
- `T=16` one-GPU training fits the selected A100 with headroom for DDP;
- causal and SE(3) tests remain within fp32 numerical tolerances.
- no bucket is hidden by an aggregate score; 100 ps and 1 ns results receive separate go/no-go rows.

Do not claim long-horizon superiority from this pilot; there is no rollout task here.

## 8. Proposed file changes

| Area | New/changed files | Purpose |
|---|---|---|
| data | `data/clip_dataset.py`, `data/collate_clip.py`, clip preprocessors | clip records, packed schema, validation |
| spatial adapter | `module/multiframe_codec.py` | flatten/restore graph batch, anchor handling, codec API |
| temporal | `module/temporal_codec.py` | causal attention, down/up sampling, masks |
| TorchMD hook | `module/torchmd_et.py` only if needed | optional latent conditioning with old defaults unchanged |
| loss/metrics | `module/codec_losses.py`, `utils/codec_metrics.py` | geometry and temporal objectives |
| training | `trainer/codec_trainer.py`, `train_codec.py`, `config/codec.yaml` | isolated new training path |
| evaluation | `eval_codec.py` | round-trip metrics and baselines |
| tests | `tests/codec/` | contract, causality, equivariance, regression, smoke |

## 9. Main risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Atom-time token count is too large | A100 OOM before useful clip length | `T*N` dynamic batching, local temporal windows, ratio-2 gate before ratio 4, profile before pooling |
| Temporal compression erases fast motion | good RMSD but poor kinetics | velocity/acceleration/torsion and frequency metrics; keep ratio configurable |
| Coarse sampling is mistaken for a unit conversion | 1 ns clips dominate loss or are interpreted as 100 ps motion | continuous physical-time conditioning, homogeneous time buckets, bucket-normalized losses, stratified metrics |
| Coarse trajectories are interpolated into fake fine labels | hallucinated high-frequency supervision | forbid coarse-to-fine label interpolation; permit only labeled fine-to-coarse decimation |
| Decoder leaks target frames | deceptively low reconstruction error | strict decoder signature and a test that zeroes/replaces target frames after encoding |
| Learned temporal vector mixing breaks equivariance | orientation-dependent output | attention weights only from invariants; never mix xyz axes; randomized rotation tests |
| Alignment erases ligand motion | false interaction quality | align on protein mask and transform entire complex together |
| Static anchor creates identity shortcut | static loss becomes meaningless | corrupt anchor/input and reconstruct clean structure |
| PVB checkpoint compatibility breaks | existing experiments unusable | additive entrypoint, optional hooks with old defaults, strict regression test |
| Codec task gets conflated with generation | unclear scientific result | no trunk or rollout in v1; evaluate round-trip only |

## 10. Completion definition

The code milestone is complete when G0–G4 pass and the operator can launch G5 from documented commands. The scientific milestone is complete only after G5 results are recorded in `HANDOFF.md` and the ratio-4 codec receives an explicit go/no-go decision.

## 11. Luna review repair status — 2026-08-27

The review repair was implemented in priority order. The code-level requirements are covered by the new contract/regression tests, while GPU- and artifact-store-dependent acceptance remains explicitly bounded:

| Review item | Status | Evidence / remaining boundary |
|---|---|---|
| P1-A standard CUDA ordering | partially verified | Shared CPU-prepare/transfer helper, evaluator/DDP fixes, CPU dense tests, and ordering regression are in place. A fresh real CUDA evaluator/DDP run is deferred while all visible GPUs are occupied. |
| P1-B model/graph checkpoint contract | fixed | v2 checkpoint fields, model reconstruction, semantic mismatch rejection, explicit v1 legacy path, and parameter-key/shape/count tests. |
| P1-C atomic resume | fixed | Preflight validation, exact optimizer/sampler/normalization checks, allowed `max_steps` extension, rollback, and next-step equality tests. |
| P1-D FP32/BF16/SE(3) gate | partially verified | Immutable external-reference loader/builder, executable FP32 gradient/output/loss comparisons, true BF16 relative error, synthetic CUDA cap-unsaturated comparison, and production-CUDA SE(3) code are present. Real fixed-batch execution is blocked by the absent external manifest and busy GPUs. |
| P2-E split isolation | fixed | Train-only canonical selection, provenance/hash manifest, evaluation contract check, and conflicting earlier-validation test. |
| P2-F synchronization | partially verified | Host metadata and async assertions remove known hot-path scalar/CPU graph work; profiler script exists. No before/after measurement can be claimed without a baseline and free GPU. |
| P2-G tracked artifacts | blocked pending operator action | Inventory and guard are present; the current branch still contains tracked checkpoint blobs. No deletion, LFS migration, or history rewrite was authorized. |
| cache scalability | fixed | CPU canonical source retention and exact atom-count/identity validation survive bounded GPU-cache eviction. |

The full test command and final count are recorded in the Luna handoff after the last verification run. These statuses do not claim convergence, quality, or full-data performance.

## 12. Continued GPU validation — 2026-08-27

The immutable reference was generated and the real Phase-A entry completed with status: passed before a final no-semantic-change synchronization repair. During that run, the synthetic gate exposed and the code repaired a CPU-leaf gradient accumulation bug in the acceptance harness. The subsequent profiler showed that the installed torch_cluster.radius_graph derives its batch size with int(batch.max()) when no explicit batch_size is supplied; CudaRadiusNeighborList now requires and forwards the host-derived graph count.

The complete CPU suite passes after the repair. The existing profiler JSONs are after-only measurements from immediately before the explicit batch_size patch, so they are evidence for the diagnosis, not final post-patch timing. A current GPU snapshot has no safely idle device pair. Final post-patch acceptance replay, post-patch topology/distance-only profiles, and two-rank NCCL DDP are therefore still open.


## 13. Post-patch GPU validation — 2026-08-27

The previously open real-GPU checks were completed after host GPUs 0, 1, and 5 became available. The current tree has:

| Check | Result | Evidence |
|---|---|---|
| Real fixed-batch Phase-A acceptance | passed | outputs/engineering_v1/phase_a/acceptance.json |
| Two-rank NCCL DDP smoke | passed | outputs/engineering_v1/ddp_smoke.json |
| Topology runtime profile | completed after-only | outputs/engineering_v1/profiling/topology.json |
| Distance-only runtime profile | completed after-only | outputs/engineering_v1/profiling/distance_only.json |
| Standard CUDA evaluator | completed for one validation batch and all three controls | outputs/engineering_v1/standard_eval_one_batch.json and .md |
| CPU regression suite and compileall | passed | 53 tests, one pre-existing warning; compileall clean |

The evaluator smoke intentionally used no trained checkpoint and must not be read as convergence evidence. P2-F remains partially verified because a pre-fix profiler baseline was not captured, so only post-patch latency/data-wait measurements are reported. P2-G remains operator-gated because tracked binary migration or history rewriting was not authorized.


## 14. Tracked binary cleanup — 2026-08-28

The operator-approved cleanup removed generated output binaries from the current Git index while preserving local experiment files.

| Item | Result |
|---|---|
| Generated tracked output binaries | 29 removed from the index |
| Local checkpoint/reference files | preserved in the worktree |
| Ignore policy | added for output checkpoint/data suffixes |
| Remaining tracked binary | module/equiformer_v2/Jd.pt, 21,697 bytes |
| Artifact guard | passed with zero violations |
| Git history rewrite | not performed |

The last item is deliberate: removing current tracking is reversible and reviewable, while filter-repo/history rewriting and force-push would change shared history. That separate operation should only be run with an explicit repository-history plan.


## 15. GitHub-compatible history cleanup — 2026-08-31

The rejected push was caused by large blobs in historical commits, not by the current tree. A new clean tip was created from origin/main using the already-clean current tree. The previous branch tip remains reachable through fix/graph-runtime-v1-history-with-binaries, so the original history was not deleted.

The candidate branch was checked with git rev-list --objects plus git cat-file --batch-check: zero reachable blobs exceed 100,000,000 bytes. The next operation is a normal push of fix/graph-runtime-v1; no force-push is required because origin has no branch with that name after the rejected attempt.


## 16. Push authentication — 2026-08-31

The binary/history issue is resolved locally and the candidate branch passes the 100 MB reachable-blob check. The normal push is currently blocked only by missing GitHub HTTPS credentials in the container. No token was entered or stored. After authentication is configured, rerun:

~~~text
git push -u origin fix/graph-runtime-v1
~~~
