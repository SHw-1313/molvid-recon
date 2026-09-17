# Luna/Codex execution prompt — molvid graph/runtime engineering v1.2

Repository: `SHw-1313/molvid`  
Target branch: `fix/graph-runtime-v1`  
Execution environment: run all commands inside `enter-container` with the `torch-ito` conda environment active.

## Mission

Repair the existing multi-frame codec's graph/runtime implementation **without changing the scientific architecture**.

This phase must preserve:

- hidden dimensions, layer counts, temporal ratio, temporal blocks, decoder equations, losses, cutoffs, and neighbor count;
- default graph semantics: `unique(radius/KNN edges ∪ directed covalent edges)` with a binary bond flag;
- the distance-only bond variant must be explicitly labeled as an ablation and must never silently replace the default topology-bond path;
- static/dynamic task semantics;
- checkpoint parameter names, shapes, and count;
- outputs and gradients up to explicitly stated floating-point tolerances.

Do not implement state-detail packing, teacher training, probability learning, decoder redesign, or ViSNet integration in this branch.

## Mandatory deliverables

1. Strict device-resident CUDA neighbor-list path.
2. Coordinate-independent topology cache and efficient bond/radius union without full-edge sort/`torch.isin`.
3. BF16 autocast with FP32 geometry and loss islands.
4. Pinned, multi-worker, non-blocking input pipeline with worker-safe mmap handling.
5. Correct epoch/sampler handling for engineering tests.
6. A repaired full-graph execution path with compatibility and short-run acceptance evidence.
7. A controlled static-bond-source ablation: supplied topology bonds versus canonical-reference distance-only inferred bonds.

## Optional stretch deliverable

An exact core-plus-halo single-graph execution backend. Start it **only after the mandatory full-graph path passes all gates**. Failure of the optional backend must not block merging the mandatory fixes.

## Hard prohibitions

The production CUDA path must contain none of the following:

- `.cpu()`, `.numpy()`, host-side neighbor search, or a graph-related CPU round trip;
- `neighbor_backend=auto`;
- silent fallback to dense `torch.cdist`;
- full-edge `torch.sort`, `torch.unique`, `torch.isin`, or coalescing over all radius-plus-bond edges;
- computing or transferring `edge_vec`/`edge_weight` and then discarding/recomputing them;
- an unbounded GPU topology cache;
- persistent cache/partition buffers that add state-dict keys;
- silent dropping of oversize samples;
- a naive no-halo atom split;
- any 50-epoch, full-data, or architecture-ranking run;
- using atom types, residue identities, supplied bond topology, or bond order to infer bonds in the `distance_only` ablation.

If CUDA radius-graph support is unavailable, fail immediately with an actionable error.

---

## Phase A — mandatory engineering repair

### T00 — baseline contract

Before editing:

1. Record the exact commit and environment versions.
2. Run the existing unit suite.
3. For all three existing controls, save on one fixed real batch:
   - parameter names/shapes/count;
   - FP32 decoded coordinates and scalar/vector latent outputs;
   - total and component losses;
   - parameter gradients and a position-gradient probe;
   - canonicalized unordered edge codes and bond flags;
   - wall time, step/s, atom-frame tokens/s, and peak memory after warmup.
4. Store evidence under `outputs/engineering_v1/baseline_contract/`.
5. Do not rerun the previous 50-epoch experiment.

### T01 — strict CUDA neighbor backend

Prefer a dedicated module such as `module/neighbor_graph.py`.

Production behavior:

```python
with torch.no_grad():
    edge_index = radius_graph(
        pos_fp32,
        r=cutoff_upper,
        batch=graph_id,
        loop=True,
        max_num_neighbors=max_num_neighbors,
    )

src, dst = edge_index
edge_vec = pos_fp32[src] - pos_fp32[dst]
edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
```

Requirements:

- `pos_fp32`, `graph_id`, `edge_index`, `edge_vec`, and `edge_weight` stay on CUDA;
- compute differentiable geometry exactly once from the original FP32 CUDA positions;
- preserve cutoff, loops, direction/flow, graph isolation, and neighbor cap;
- expose and log `backend_used="cuda_radius"`;
- keep a small `dense_test` backend for CPU unit tests only;
- remove exception-based production fallback;
- assert no cross-sample or cross-frame edges.

Tests:

- production graph tensors are CUDA tensors;
- no `.cpu()`/host neighbor call is reached;
- unavailable CUDA extension raises;
- synthetic no-tie graphs reproduce the legacy unordered edge set exactly;
- real batches satisfy the same cutoff, cap, loop, and graph-isolation invariants; any capped-neighbor difference must be audited rather than ignored.

### T02 — topology cache and efficient bond union

Add an explicit graph-input bond source:

```yaml
model:
  bond_construction:
    mode: topology  # topology | distance_only
```

The mandatory/default `topology` mode must preserve:

```text
final edges = unique(radius edges ∪ directed covalent edges)
bond_type = 1 exactly for supplied covalent edges
```

The `distance_only` mode is implemented for the later controlled ablation; it must not alter the default acceptance baseline.

Use two cache levels:

1. bounded canonical CPU cache keyed by a stable topology identifier/hash;
2. bounded per-device materialization cache, invalidated on device change.

Cache only coordinate-independent data:

- atom/block/component metadata;
- normalized, directed, bond-only-deduplicated local `bond_index`;
- local bond codes;
- frame/batch offset templates where useful;
- optional partition metadata.

Do not cache radius edges, distances, or directions by default.

Efficient per-batch union:

1. Materialize directed global bond edges for the packed samples and frames.
2. Sort **bond codes only**; do not sort all geometric edges.
3. Build radius edges on GPU.
4. Encode radius edges as 64-bit codes.
5. Use GPU `searchsorted` against sorted global bond codes.
6. Remove radius edges already represented by bonds.
7. Append every directed bond edge once.
8. Create `bond_type` directly as zeros then ones.
9. Compute final edge vectors and distances once.

Tests:

- exact union edge set on synthetic and fixed real batches after canonicalization in test code;
- no duplicate final edges;
- every covalent edge appears exactly once with flag 1;
- all nonbond edges have flag 0;
- repeated windows sharing topology produce cache hits;
- bounded LRU eviction works;
- caches create no persistent state-dict entries and no stale device tensors.

### T03 — BF16 with FP32 geometry/loss islands

Add:

```yaml
training:
  precision: bf16  # fp32 | bf16
```

Requirements:

- parameters and optimizer states remain FP32;
- use CUDA BF16 autocast for embeddings, linear layers, attention, and FFNs;
- retain FP32 for positions, neighbor construction, edge vectors/distances, sensitive vector normalization, coordinate accumulation, velocity/acceleration, and all loss reductions;
- cast decoded coordinate residuals to FP32 before adding anchors and computing losses;
- do not use GradScaler for BF16;
- save precision in checkpoint/run metadata;
- CPU remains FP32.

Acceptance on the fixed batch:

- all outputs, losses, and gradients are finite;
- BF16 total-loss relative difference versus repaired FP32 is at most `2e-2`;
- graph edge codes are identical because geometry is built from FP32 coordinates;
- resume preserves precision configuration.

### T04 — normal data pipeline

Implement:

- `ClipBatch.pin_memory()`;
- `ClipBatch.to(device, non_blocking=True)` or an equivalent typed transfer path;
- non-blocking host-to-device copies;
- configurable defaults:
  - `num_workers: 8`;
  - `pin_memory: true`;
  - `persistent_workers: true` when workers > 0;
  - `prefetch_factor: 2`;
- worker-safe lazy/reopened mmap handles (`__getstate__/__setstate__` or equivalent);
- one-time store schema/manifest validation at dataset open;
- trusted-store fast collation that skips repeated full record validation in the hot path;
- strict/debug mode that retains full validation.

Tests:

- 0-, 2-, and 8-worker loaders produce tensor-equal batches for fixed sample IDs;
- no deadlock or stale mmap handle across two epochs;
- tensors are actually pinned before transfer;
- deterministic sample order for a fixed seed;
- no sample-content, mask, or ordering change.

### T05 — sampler/trainer correctness for acceptance runs

The previous tiny run showed epoch-like periodicity, so the engineering acceptance must have unambiguous epochs.

Required changes:

- call `batch_sampler.set_epoch(epoch)` whenever a new epoch/iterator begins;
- provide a no-replacement exact-pass mode for tiny acceptance;
- assert each tiny-train clip appears exactly once per acceptance epoch;
- report duplicates, missing clips, and unique clip count;
- apply warmup LR before the corresponding optimizer step, not after it;
- save/restore sampler epoch, optimizer step, and scheduler/warmup state.

Do not change the large-scale sampling policy by default; add an explicit acceptance-mode configuration.

### T06 — explicit oversize policy

Add:

```yaml
data:
  oversize_policy: error  # error | partition
```

Requirements:

- never silently `continue` or drop a sample;
- full mode with `error` lists the ineligible samples and fails clearly;
- partition mode may place an oversize sample alone and route it to the partition backend;
- the sampler must not discard it before the model sees it;
- write eligibility/exclusion manifests and report systems, trajectories, clips, and atom-frame tokens.

### T07 — repaired full-graph acceptance

Create a **complete runnable** `config/codec_engineering_full.yaml`, not a partial overlay.

Before the optional partition work, the full path must pass:

1. state-dict keys/shapes/count unchanged;
2. synthetic exact edge-set and bond-flag equivalence;
3. fixed real-batch FP32 output/loss/gradient regression;
4. BF16 finite/closeness regression;
5. 200 optimizer steps;
6. five exact tiny-set epochs with no replacement.

The 200-step run and five-epoch run are two separate acceptance commands. Read their budgets from the config `acceptance` section; do not interpret `max_steps: 200` as the five-epoch budget.

FP32 regression tolerances against the recorded baseline:

- scalar output: `rtol=2e-4`, `atol=2e-5`;
- vector output: `rtol=5e-4`, `atol=5e-5`;
- total-loss relative difference: `<=1e-4` when the canonical edge set is identical;
- parameter-gradient relative error: `<=1e-3`;
- position-gradient relative error: `<=1e-3`.

If the GPU backend changes the capped-neighbor set, record the exact differing edges and do not claim strict numerical equivalence until the selection policy is reconciled.

Runtime merge gate on the same machine, batch, and model:

- zero graph-related CPU round trips;
- no silent fallback;
- repaired BF16 full path reaches at least `1.20x` the T00 baseline atom-frame throughput;
- otherwise the phase is not complete and the remaining bottleneck must be identified.

---

## Phase B — required short static-bond-source ablation

Start only after the mandatory topology-bond full-graph path passes and is committed. This phase changes graph-input semantics deliberately, so it is **not** subject to topology-path output equivalence.

### T08 — distance-only static bond inference

Create:

```yaml
model:
  bond_construction:
    mode: distance_only
    source_frame: canonical_reference
    canonical_reference_policy: manifest_reference_or_earliest_clip
    min_distance_angstrom: 0.5
    max_distance_angstrom: 2.2
    max_num_neighbors: 64
    directed: true
    freeze_scope: topology_id
    use_atom_types: false
    use_residue_or_topology: false
```

Definition:

1. Require a stable `topology_id`/system identifier for every sample.
2. Select one deterministic canonical reference coordinate set per `topology_id`: prefer an explicit manifest reference; otherwise use the lexicographically earliest clip/replica/window anchor. Never use whichever clip happens to be loaded first.
3. Use only that canonical reference's FP32 coordinates.
4. Consider atom pairs within that topology/system only.
5. Infer an undirected bond when `0.5 Å < ||x_i-x_j||_2 <= 2.2 Å`.
6. Convert every inferred pair to two directed edges and exclude self edges.
7. Freeze and reuse the inferred bond set for every frame, window, and replica sharing the same `topology_id`. Do not infer bonds independently from each clip, because that would leak coordinate fluctuations into the supposedly static topology.
8. Do not read or use supplied `bond_index`, atom type, block/residue identity, component chemistry, or bond order to construct graph-input bonds.
9. Build the final graph with the same efficient union algorithm: `unique(radius edges ∪ inferred distance bonds)`.
10. Infer bonds on CUDA under `torch.no_grad()`; compute final differentiable edge geometry once from original FP32 positions.
11. Assert that the distance-bond neighbor cap is not saturated; otherwise fail and report the topology.

Store the inferred result in a separate bounded cache keyed by `(topology_id, canonical_reference_hash, cutoff settings)`. It is coordinate-derived during construction but static after construction.

Controlled-variable rule:

- `bond_construction.mode` changes only the bond edges and `bond_type` used by the spatial graph.
- Keep `bond_length_loss` and bond-quality evaluation based on the original supplied topology for both variants. This isolates graph-input bond construction rather than changing supervision at the same time.
- The supplied topology may be used only for evaluation diagnostics and the unchanged bond loss; it must not enter graph construction in `distance_only` mode.

### T09 — bond-source ablation protocol

Create a complete runnable `config/codec_engineering_full_distance_bond.yaml`.

It must be identical to the repaired full topology-bond config except for `model.bond_construction` and run labels/output paths.

Fairness requirements:

- initialize topology and distance-only runs from the same saved initial state;
- use the same seed, exact sample IDs, batch schedule, optimizer, precision, and data exposure;
- run only the fixed-batch checks, 200 optimizer steps, and five exact tiny-set epochs;
- label the result `bond_source_ablation_development`;
- do not make a convergence or architecture-quality claim from this short run.

Required diagnostics, aggregated per system rather than only per clip:

- inferred/topology bond precision, recall, F1, and Jaccard;
- inferred and topology directed-edge counts;
- missed topology-bond distance distribution;
- distance-only false-positive distance distribution;
- degree distribution, isolated atoms, and connected-component count;
- atom-frame throughput, peak memory, cache hit rate;
- train/validation loss components and existing reconstruction metrics.

The distance-only variant does not gate merging the mandatory engineering repair. Its purpose is to quantify whether explicit topology bonds are necessary and how much graph behavior changes when static bonds are inferred from geometry alone.

---

## Phase C — optional exact halo-subgraph backend

Start only after Phase A passes and is committed.

### T10 — execution-only partition backend

Add:

```yaml
model:
  spatial_execution:
    mode: full              # full | halo_subgraph
    max_core_atoms: 2000
    halo_hops: auto
    partition_method: spatial_morton
    subgraph_node_budget: 24000
```

Important scope:

- this backend still builds the full CUDA edge index first;
- it is intended to reduce message-passing activation memory and permit larger single systems, not to eliminate full neighbor-list construction;
- it must not add parameters or persistent state-dict buffers.

The current receptive field is:

```text
1 initial NeighborEmbedding aggregation + spatial_layers message-passing layers
```

Therefore:

```text
halo_hops = 1 + spatial_layers
```

Implementation:

1. Partition each molecular system into spatially contiguous core atoms from frame-0/anchor coordinates.
2. Reuse core ownership across all frames of a clip.
3. For each frame, expand each core through the full graph by `halo_hops`.
4. Build induced halo subgraphs with correct local remapping and bond flags.
5. Microbatch disconnected halo subgraphs under `subgraph_node_budget`.
6. Run the unchanged spatial encoder.
7. Write back only core outputs.
8. Cover every original atom as core exactly once; halo atoms may repeat.
9. Reassemble full `[T,N,C]` and `[T,N,3,C]` outputs before temporal processing.

Do not require each partition invocation to contain the full graph edge set. The correct gate is that every core node has its complete receptive-field dependency closure and that merged outputs/gradients match full execution.

Acceptance against the repaired full path:

- scalar max abs difference `<=2e-5`;
- vector max abs difference `<=5e-5`;
- total-loss relative difference `<=1e-4`;
- parameter-gradient relative error `<=1e-3`;
- position-gradient relative error `<=1e-3`;
- each core atom covered exactly once;
- cross-core covalent-bond case passes;
- SE(3) equivariance passes;
- state-dict keys/shapes/count exactly unchanged.

Recommendation gate:

- pass every equivalence test; and
- either reduce peak GPU memory by at least 20% on the fixed large batch, or process an oversize sample that full mode cannot process;
- otherwise keep it experimental and do not recommend it as default.

### T11 — short comparison only

Create a complete `config/codec_engineering_partition.yaml` identical to the full config except partition-specific fields and oversize policy.

Run only:

- fixed-batch FP32 equivalence;
- fixed-batch BF16 finite regression;
- 200 optimizer steps;
- five exact tiny-set epochs.

Record wall time, step/s, atom-frame tokens/s, peak allocated/reserved memory, data-wait time, cache hit rate, radius/bond edge counts, partition count, and halo/core ratio.

Do not run 50 epochs.

### T12 — documentation and stop

Update repository `PLAN.md`, `TASKS.md`, `DECISIONS.md`, and `HANDOFF.md` with:

- files changed and exact commands;
- baseline and final commit IDs;
- all test outputs;
- state-dict compatibility;
- graph/bond equivalence;
- BF16 and loader evidence;
- sampler exact-pass evidence;
- full-path throughput result;
- topology-versus-distance-only bond ablation diagnostics and short-run result;
- partition equivalence/memory result;
- unresolved risks.

Stop after this handoff. Do not begin architecture work.
