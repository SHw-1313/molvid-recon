# PLAN — graph/runtime engineering v1.2

## Goal

Repair the existing deterministic codec's graph construction, topology handling, precision, loading, and acceptance workflow before any architecture work resumes.

## Invariants

The mandatory production path preserves the model, losses, topology-bond graph semantics, parameter state dict, and scientific task. Only execution/storage/batching/precision may change.

The distance-only static-bond variant is a separately labeled ablation. It must never silently replace the topology-bond default.

## Mandatory Phase A — full topology-bond repair

1. Record a fixed-batch baseline contract.
2. Replace the CPU-backed neighbor path with strict CUDA radius graph construction.
3. Cache static supplied topology and replace full-edge sort/`isin` with bond-only code lookup plus direct concatenation.
4. Add BF16 autocast with FP32 geometry/loss islands.
5. Add worker-safe mmap loading, true pinning, non-blocking transfer, and multi-worker prefetch.
6. Fix acceptance-epoch handling (`set_epoch`, no-replacement exact passes, warmup ordering, resume state).
7. Prohibit silent oversize filtering.
8. Pass fixed-batch regression, 200 steps, and five exact tiny epochs.

## Required Phase B — static-bond-source ablation

Add `bond_construction.mode: topology | distance_only`.

The distance-only variant:

- infers bonds once from a deterministic canonical reference per `topology_id`;
- uses a fixed `0.5 Å < d <= 2.2 Å` interval;
- uses no atom type, residue identity, supplied topology, or bond order for graph construction;
- reuses the same inferred bond set across every frame, window, and replica sharing that `topology_id`;
- keeps the original topology-based bond loss and evaluation so only graph-input bond construction changes;
- uses the same initialization, batches, seed, optimizer, precision, and short acceptance budget as the topology variant.

Report bond overlap/connectivity diagnostics plus the existing reconstruction metrics. Do not treat the short run as a convergence comparison.

## Optional Phase C — exact halo-subgraph execution

Implement a core-plus-exact-halo spatial execution backend after Phase A is complete. It must reproduce full outputs/gradients and provide a material memory/eligibility benefit; otherwise it remains experimental.

## Non-goals

State-detail packing, decoder changes, probability learning, ViSNet integration, DDP scaling, scientific architecture ranking, and 50-epoch/full-data training.
