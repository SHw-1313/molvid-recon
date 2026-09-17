# Multi-frame codec v1 decision log

Decisions here are binding until superseded by a new numbered decision with evidence. Do not silently edit an old decision after implementation begins.

## D001 — Additive codec path, not a `dyVAE` rewrite

**Decision:** Keep current PVB training/inference intact and introduce an isolated codec entrypoint and modules.

**Why:** Existing checkpoints and one-step experiments must remain reproducible. The new clip contract is semantically incompatible with silently reusing `x0/x1`.

## D002 — Packed time-major coordinate contract

**Decision:** Use `x: [T, N_total, 3]` with packed atoms, `abid`, and `atom_ptr`. Batches are homogeneous in task and temporal length.

**Why:** Padding proteins to a global `N_max` is wasteful; PVB already packs atoms. Time-major layout makes temporal shape explicit and lets the spatial adapter flatten frames into graph batches.

## D003 — True clips replace frame pairs

**Decision:** Add ordered clip preprocessing/loaders. Do not reconstruct clips by chaining independently sampled `(x0, x1)` records.

**Why:** Pair records lose temporal ordering, timestamps, and higher-order dynamics—the exact information the codec is intended to learn.

## D004 — Static and dynamics share weights, not fake time

**Decision:** Static data uses `T=1` corrupted-coordinate denoising; trajectory data uses ordered `T=16` clips. A task-aware loader alternates homogeneous batches.

**Why:** Repeating a static structure across time teaches zero motion as if it were a real trajectory and biases the temporal module.

## D005 — PVB scalar/vector features are the spatial latent basis

**Decision:** Reuse `TorchMD_VQ_ET` outputs `h` and `v`; do not replace the encoder with ViSNet or introduce CG-heavy irreps in v1.

**Why:** This preserves PVB's atom/residue type and local geometry strengths while providing an invariant/equivariant interface suitable for temporal mixing.

## D006 — SE(3)-safe temporal mixing without CG coefficients

**Decision:** Temporal attention/gates are computed from invariant scalar features. The resulting scalar weights mix scalar channels and vector channels identically across xyz; learned maps act on channels only.

**Why:** This retains equivariance while avoiding arbitrary coordinate-axis mixing and the implementation burden of CG tensor products.

## D007 — Causal encoder and decoder

**Decision:** All temporal attention/convolution/downsampling uses only the current and earlier frames. Upsampling uses repeat/interpolation plus causal refinement.

**Why:** The codec should later support streaming/chunk continuation. A bidirectional v1 would inflate reconstruction quality but create an incompatible latent interface for the planned world-model trunk.

## D008 — Deterministic codec before VAE/VQ

**Decision:** v1 has no KL term, posterior sampling, or vector quantization. Add such a bottleneck only after deterministic ratio-4 reconstruction and causality/equivariance gates pass.

**Why:** Otherwise posterior collapse or quantization error would be confounded with temporal architecture failure.

## D009 — Dual scalar/vector temporal latent

**Decision:** Preserve both invariant `z_h` and equivariant `z_v` through temporal compression.

**Why:** Scalar-only latents are easy to condition but force the decoder to recover directional motion from scratch. Vector latents provide a direct equivariant basis for displacements.

## D010 — First frame is an explicit anchor

**Decision:** Decode displacements relative to `x_anchor=x[0]` (or its corrupted form for static denoising). Report future-frame metrics separately and do not count the anchor as a learned temporal token.

**Why:** Molecular dynamics is naturally conditioned on an initial structure, and an anchor fixes the translation gauge. Static corruption prevents a trivial identity solution.

## D011 — Protein-based clip alignment

**Decision:** Kabsch-align every frame to frame 0 using a protein backbone/CA mask and apply the same transform to the whole system.

**Why:** It removes irrelevant global motion while preserving ligand motion relative to the protein. Independent ligand alignment is forbidden.

## D012 — Atom tokens only in v1

**Decision:** Do not add residue pooling or spatial patch tokens yet. Use `T*N` dynamic batching and measure memory first.

**Why:** The current data has per-atom `btype` and CA/block positions but no robust unique residue id in every dataset. Adding pooling now would couple data migration to the codec proof.

## D013 — Factorized video-style codec, no generative trunk

**Decision:** Spatial graph encoding/refinement and causal temporal blocks are factorized. The v1 output is a reconstructed observed clip, not a predicted future clip.

**Why:** This imports the useful video-codec principle—joint multi-frame tokens and temporal compression—without conflating it with diffusion/flow generation or rollout quality.

## D014 — Ratio 1 is a first-class control

**Decision:** Temporal ratios 1, 2, and 4 share the same API; ratio 1 supports both temporal-on and temporal-off controls.

**Why:** The pilot needs to separate gains from temporal mixing from losses caused by compression.

## D015 — No unapproved dependency changes

**Decision:** Implement with the current repository environment. Do not add FlashAttention, xFormers, PyTorch Lightning, Hydra, or another storage framework in v1.

**Why:** The immediate goal is a runnable patch in the user's existing PVB environment, not ecosystem migration.

## D016 — Physical time is a first-class model condition

**Decision:** Every trajectory clip carries `time_ps` and `delta_time_ps`. Temporal attention uses a continuous invariant relative-time bias derived from physical `|t_i-t_j|`; compressed latents carry `latent_time_ps`; the decoder receives `target_time_ps`.

**Why:** Frame index is not a physical clock. A 16-frame clip sampled every 100 ps spans 1.5 ns, while one sampled every 1 ns spans 15 ns. Treating them identically would make the shared codec learn contradictory motion semantics.

## D017 — Homogeneous sampling-interval buckets with shared weights

**Decision:** A minibatch is homogeneous in task, temporal length, and sampling-interval bucket. Static, 100 ps, 1 ns, and other configured buckets alternate under explicit sampling weights while sharing codec parameters. Continuous timestamps, not bucket id or dataset id, are the authoritative model input.

**Why:** Homogeneous buckets simplify masking, normalization, and profiling without splitting the model by dataset. Continuous conditioning preserves interpolation to unseen intervals better than a categorical dataset token alone.

## D018 — No coarse-to-fine label fabrication

**Decision:** Never interpolate a coarse trajectory into fine-timescale training targets. Fine trajectories may be decimated into explicitly labeled coarse clips; the reverse is forbidden.

**Why:** Dividing displacement by `dt` changes units but cannot reconstruct fast events already removed by 1 ns sampling. Interpolated 100 ps labels would teach visually smooth but physically unsupported motion.

## D019 — Temporal losses and metrics are time-bucket aware

**Decision:** Compute velocity and irregular-grid acceleration from physical `delta_time_ps`. Optimize with train-split normalization statistics per time bucket, checkpoint those statistics, log raw physical units, and report metrics separately by bucket. Do not use an unqualified cross-bucket mean for model selection.

**Why:** Fine and coarse sampling observe different frequency bands and produce different derivative distributions. A single aggregate can be dominated by dataset size or hide failure at one physical scale.

## D020 — Versioned binary clip storage after JSON measurement

**Decision:** New clip stores use `npz-v1`: numeric arrays are compressed NumPy members and scalar/list metadata is one JSON member. The reader accepts only this versioned payload; obsolete gzip+JSON compatibility is removed.

**Why:** A representative ATLAS record serialized as JSON was 1,726,089 bytes and gzip level 6 reduced it to 403,058 bytes while spending about 0.04 s on a single record in the local environment. `np.savez_compressed` produced 220,131 bytes in about 0.04 s and avoids Python float serialization; uncompressed NumPy storage was 844,984 bytes in under 1 ms. The versioned binary format is therefore both smaller and operationally safer for the half-data materialization. This is a storage implementation choice, not a change to the ClipBatch contract.

## Open questions requiring operator evidence

These are parameters, not permission to redesign the architecture:

1. Exact writable PVB repository path and branch convention.
2. Exact local `enter-container` command syntax and whether `torch-ito` already contains the PVB dependencies.
3. Local ATLAS/mdCATH raw paths, native frame timestamps/spacing, and desired 100 ps/1 ns (or other) bucket definitions.
4. PVB checkpoint path for encoder initialization.
5. Whether the first pilot is protein-only or includes a protein-ligand trajectory dataset.
6. Available A100 memory (40 GB or 80 GB), which sets the initial `T*N` bound.

The worker should implement synthetic/CPU-safe portions without these values, use explicit config placeholders, and stop before a data/GPU gate that genuinely needs them.


## D021 — Strict CUDA graph runtime and hard GPU failure

**Decision:** Production clip graph construction uses the explicit `cuda_radius` backend and the codec trainer's production `auto`/CUDA selections hard-fail when CUDA is unavailable. `dense_test` is the only CPU graph backend and must be selected explicitly.

**Why:** The graph is part of the model's numerical contract. A silent CPU fallback hides container/device failures and changes the execution path. Dataset registration occurs on the CPU batch before transfer; CUDA forward does not copy graph inputs back to host.

## D022 — Reconciled CUDA cap policy is explicit, historical CPU identity is not claimed

**Decision:** The repaired topology path uses the CUDA radius kernel's configured neighbor cap with source-order output. Distance-only inference uses a bounded CUDA over-query solely to prove its independent cap is not saturated. The old CPU nanoflann traversal is retained only as an audit reference; it is not emulated by a hidden CPU call or represented as numerically identical.

**Why:** The old CPU and CUDA extensions have different candidate traversal semantics when a target has more than the cap. The fixed real audit found only distance-edge differences (`12539` removed, `12512` added; zero bond-edge differences), but the difference is enough to invalidate a historical bitwise claim. A deterministic GPU policy plus a saved repaired reference is auditable and keeps the production path device-resident.

## D023 — Phase-A acceptance evidence is bounded, not a training-quality result

**Decision:** Phase A uses fixed-batch checks, 200 optimizer steps, and five exact tiny epochs only. It does not start a 50-epoch/full-data run and does not make convergence or scientific-quality claims. Phase B is a separately labeled distance-only ablation and starts only after the Phase-A commit.

**Why:** The reviewed acceptance packet explicitly separates engineering gates from model-quality experiments; short finite checks cannot support convergence conclusions.


## D024 — Canonical-coordinate CUDA distance-only ablation

**Decision:** Phase B uses `bond_construction.mode: distance_only`, selects one deterministic canonical reference per stable topology id, and infers bonds only from CUDA FP32 reference coordinates with the strict interval `0.5 < d <= 2.2 Å`. Supplied atom labels, block labels, atom-source indices, and supplied bond indices are not inputs to inference.

**Why:** This isolates the requested distance-only architecture from the stored topology graph while preserving a reproducible reference choice. The extra edges are a measurable consequence of the interval rule, so precision/recall/F1/Jaccard and graph diagnostics are mandatory and are stored with the acceptance evidence. Missing CUDA remains a hard error; there is no CPU fallback.


## D025 — Three-setting real-data overfit gate

**Decision:** The selected-system overfit gate is three settings: the three full-width topology controls, each trained from fresh initialization on one fixed training clip from each of the three selected systems. It is not six settings; adding distance-only would be a separate 3-by-2 bond-mode ablation.

**Why:** This isolates model memorization capacity on the selected systems while keeping the comparison aligned with the existing three-control experiment. The gate reports synchronized CUDA time and per-system loss but does not convert training-set loss reduction into a validation or quality claim.


## D026 — Three-setting distance-only real-data overfit ablation

**Decision:** Repeat the selected-system overfit gate with the same three controls, data batch, optimizer, and 500-step logging policy under bond_construction.mode: distance_only. Use one deterministic canonical FP32 first-frame reference per selected system and keep this round separate from the topology round.

**Why:** This makes the previously discussed six-setting comparison explicit as three topology controls plus three distance-only controls. It isolates the effect of graph construction while preserving the model/control and data protocol. Supplied bond/atom topology is excluded from distance-only inference, CUDA absence remains a hard error, and the resulting train loss is not treated as validation or scientific-quality evidence.


## D027 - Split-aware evaluation reports and plots

**Decision:** Evaluation Markdown and plot titles derive their split label from evaluation_data.source_split. Train-only overfit reports must be labeled Train, while ordinary validation reports retain the Validation label.

**Why:** The same PVB-style metric code is used for train diagnostics and held-out evaluation. A fixed Validation label would misrepresent a train-clip memorization test. The split-aware label preserves the shared evaluator while making the evidence boundary explicit.


## D028 - Both bond-mode rounds share the evaluation and visualization contract

**Decision:** Every selected-system overfit bond-mode round must provide its own checkpoint-matched PVB-style report and plots: all logged loss records, train-split total/component loss, reconstruction metrics, dynamics metrics, and time-bucket CSV. Topology and distance-only outputs remain in separate run directories.

**Why:** The six-setting ablation consists of three topology controls plus three distance-only controls. Training completion alone is not a complete deliverable; omitting evaluation or plots for one bond mode makes the comparison incomplete and can hide which graph path was actually evaluated. The evaluator records the bond mode explicitly and registers topology from the CPU batch before CUDA inference.

## D029 — Versioned codec model and resume contracts

**Decision:** New codec checkpoints use `pvb.codec.checkpoint.v2` and persist JSON-safe model/graph, distance-reference, optimizer, normalization, and sampler contracts. Resume compares all semantic fields exactly and permits only an explicit `max_steps` extension. `pvb.codec.checkpoint.v1` is accepted only through an explicit legacy flag; PVB v1 loads also require an explicit bond mode.

**Why:** Parameter keys and shapes are intentionally shared by topology and distance-only models, so `load_state_dict()` cannot detect graph-semantic drift. A checkpoint must therefore describe the complete forward and optimizer contract independently of parameter tensors.

## D030 — Immutable external FP32 reference gate

**Decision:** The Phase-A real-batch FP32 regression uses an externally supplied, hashed repaired-CUDA reference manifest. The acceptance runner fails if that manifest is absent and never regenerates it during comparison. A separate operator-run builder creates the manifest only when the GPU is available and the artifact destination has been reviewed.

**Why:** Generating a reference and comparing against it in one run would make the regression tautological. The historical CPU and repaired CUDA cap policies also differ, so the real reference must be versioned as the repaired CUDA contract; cap-unsaturated synthetic CUDA-vs-dense checks remain a separate equivalence gate.

## D031 — Train-only distance-reference provenance

**Decision:** Training distance-only graph references are selected from the training records only. Each frozen reference records the canonical sample ID, source split, atom count, atom-identity hash, coordinate hash, and inferred-bond hash; evaluation compares the checkpoint reference contract instead of deriving a graph from validation targets.

**Why:** A validation coordinate must never determine a graph used by training. Exact identity and provenance checks also prevent an evicted/reloaded reference from being silently reused for a different topology.

## D032 — Host metadata plus device-resident CUDA hot path

**Decision:** Immutable atom counts/offsets and atom-identity hashes are captured during CPU collation/registration. Production CUDA graph construction uses device-resident geometry and async device assertions; one-time scalar/hash work is confined to canonical cache registration. A profiler records after-only measurements unless an explicit pre-fix baseline is supplied.

**Why:** Repeated `.item()`, Python branching on CUDA scalars, or CUDA-to-CPU graph copies in every forward introduce synchronization proportional to batch size. Moving static checks to the CPU boundary preserves correctness while keeping the forward path asynchronous.

## D033 — Artifact guard without unapproved history mutation

**Decision:** Add a read-only tracked-binary inventory and CI guard, retain small manifests/metadata in Git, and stop before deleting blobs or rewriting history. The current inventory is evidence of the existing violation; artifact-store migration, Git LFS adoption, or clean-history reconstruction requires an operator-approved exact plan.

**Why:** Removing files in a later commit would not remove their historical Git objects, while deleting or rewriting user artifacts without an approved preservation path is unsafe.

### D034 — Supply graph cardinality to the CUDA radius wrapper

torch_cluster.radius_graph computes batch_size with Python int(batch.max()) when its optional argument is omitted. The repaired production graph path now receives the graph count from host-known frame/sample shapes and forwards it explicitly. CudaRadiusNeighborList rejects omitted cardinality; no CUDA-to-CPU inference or fallback is permitted.

### D035 — Freeze dense reference gradients before CUDA transfer

A tensor transfer can preserve the autograd edge from a CPU leaf to its CUDA copy. The synthetic CUDA-vs-dense acceptance fixture therefore clones the dense position gradient and clears the CPU leaf gradient before the CUDA backward pass. This prevents the candidate gradient from being accumulated into the reference and falsely failing the regression.


### D036 — Serialize GPU validation and report evaluator scope explicitly

Use only GPUs observed to be idle, run the independent NCCL smoke on a pair, and serialize the heavier acceptance/profile/evaluator jobs on one card. Do not terminate or preempt existing processes. Record the standard evaluator's checkpoint and batch scope; a one-batch run with freshly initialized models validates CUDA execution and metric plumbing only, not training quality or convergence. Post-patch profiler results are reported as after-only unless a real pre-fix baseline exists.

Why: the validation request concerns the repaired runtime, while other users' jobs remain out of scope. Clear scope labels prevent an execution smoke from being mistaken for a scientific result, and an after-only measurement cannot support a before/after performance claim.


### D037 — Remove generated binaries from the current index, preserve local results

For the operator-approved cleanup, remove the 29 generated binary artifacts under outputs/ with git rm --cached, retain their local working-tree copies, and add suffix-specific outputs ignore rules. Keep module/equiformer_v2/Jd.pt because it is a small source dependency. Do not run filter-repo, delete historical objects, or force-push as part of this working-tree cleanup.

Why: this removes generated checkpoints from the repository's current tracked state without destroying experiment results or rewriting shared history. A historical size purge is a separate destructive operation requiring its own reviewed target list and recovery plan.


### D038 — Publish a binary-free squashed tip while preserving the old branch

When a remote rejects historical generated binaries, create a new tip from origin/main and the already-clean current tree, verify every reachable blob is below 100 MB, and retain the former branch tip under a clearly named local backup branch. Publish the clean branch with a normal push when the remote branch was never created.

Why: deleting binaries only in the latest commit does not remove historical objects. This approach avoids losing the old experiment history, avoids force-push against an existing remote branch, and makes the exact GitHub compatibility check auditable.


### D039 — Do not handle GitHub credentials inside the agent session

If a verified push reaches an HTTPS username prompt and no credential helper or gh login is available, abort without accepting or logging a password/token. Leave the already-verified local branch ready for the operator's authenticated push.

Why: the repository history is now GitHub-compatible, while credential collection is an external authorization boundary and secrets must not enter the agent transcript or workspace.
