# TASKS — graph/runtime engineering v1.2

## Phase A — mandatory full topology-bond repair

- [ ] A00 create `fix/graph-runtime-v1`; record commit, environment, baseline graphs/outputs/losses/gradients/runtime
- [ ] A01 implement strict CUDA radius backend; remove graph-related CPU round trips and silent fallback
- [ ] A02 compute final edge vectors/distances once from FP32 CUDA coordinates
- [ ] A03 implement bounded canonical topology cache and bounded per-device materialization
- [ ] A04 replace full-edge sort/unique/isin with bond-only sorted codes, GPU search, and direct union
- [ ] A05 prove edge-union, duplicate, bond-presence, and bond-flag correctness
- [ ] A06 add BF16 autocast with FP32 geometry/loss islands
- [ ] A07 implement `ClipBatch.pin_memory()` and non-blocking typed transfer
- [ ] A08 make mmap loading worker-safe and add trusted-store one-time validation/fast collate
- [ ] A09 enable configurable workers, persistent workers, and prefetch
- [ ] A10 fix sampler `set_epoch`, exact no-replacement acceptance epochs, warmup ordering, and resume state
- [ ] A11 add explicit oversize policy and manifests; prohibit silent skipping
- [ ] A12 create complete runnable full-graph topology-bond config
- [ ] A13 pass fixed-batch FP32/BF16 regressions
- [ ] A14 pass 200 steps and five exact tiny-set epochs
- [ ] A15 meet the no-CPU-round-trip and >=1.20x atom-frame-throughput merge gate
- [ ] A16 commit Phase A and update handoff evidence

## Phase B — required short distance-only bond ablation after A16

- [ ] B00 add explicit `bond_construction.mode: topology | distance_only`
- [ ] B01 infer bonds once from a deterministic canonical reference per `topology_id` on CUDA using only `0.5 Å < distance <= 2.2 Å`
- [ ] B02 create two directed edges per inferred pair; exclude self edges; reuse identically across all frames/windows/replicas of the topology
- [ ] B03 prove distance-only graph construction does not read supplied topology/atom/residue/bond-order inputs
- [ ] B04 add bounded cache keyed by topology_id, canonical-reference hash, and cutoff; prohibit per-clip re-inference
- [ ] B05 keep topology-based bond loss/evaluation unchanged to isolate graph-input bond source
- [ ] B06 create complete `codec_engineering_full_distance_bond.yaml`
- [ ] B07 save one common initialization and enforce identical samples/batches/seeds/optimizer/precision
- [ ] B08 report precision/recall/F1/Jaccard, bond counts, distance distributions, degrees, isolated atoms, and components
- [ ] B09 run only fixed-batch checks, 200 steps, and five exact tiny epochs
- [ ] B10 label results development-only; do not make convergence claims

## Phase C — optional exact halo-subgraph backend

- [ ] C00 implement anchor-defined spatial core partitions
- [ ] C01 derive exact `halo_hops=1+spatial_layers`
- [ ] C02 implement induced halo subgraphs, local remapping, and bounded spatial microbatching
- [ ] C03 cover every atom as core exactly once and merge full h/v outputs
- [ ] C04 prove full/partition forward, loss, parameter-gradient, and position-gradient equivalence
- [ ] C05 prove cross-boundary bond correctness and SE(3) equivariance
- [ ] C06 create complete partition config
- [ ] C07 run only 200 steps and five exact tiny epochs
- [ ] C08 recommend only if equivalence passes and memory/eligibility benefit is material
- [ ] C09 update final HANDOFF and stop; do not start architecture work
