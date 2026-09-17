# DECISIONS — graph/runtime engineering v1.2

## D1 — topology bonds remain the production default

Keep `unique(radius edges ∪ directed supplied covalent edges)` with binary bond flags. Do not regress the production path to PVB's weaker bond-label-only behavior.

## D2 — distance-only bonds are an explicit ablation

Add `bond_construction.mode: distance_only` only as a separately labeled experiment. It may not silently replace topology bonds or be used to satisfy topology-path equivalence gates.

## D3 — distance-only means distance only

Infer a static directed bond set once from a deterministic canonical reference per `topology_id` using `0.5 Å < d <= 2.2 Å`. Do not use atom types, residue/block identities, component chemistry, supplied topology, or bond order for graph construction.

## D4 — isolate graph input from supervision

The distance-only ablation changes graph-input bond edges and bond flags only. The existing supplied-topology bond loss and evaluation remain unchanged, preventing two experimental variables from changing together.

## D5 — strict CUDA production backend

CPU/dense neighbor construction is test-only. Missing CUDA support is a hard error.

## D6 — sort only small bond topology

Sorting/deduplicating bond-only topology is allowed. Sorting/coalescing the complete geometric edge list is forbidden.

## D7 — cache topology, not ordinary geometry

Use a bounded canonical CPU cache plus bounded per-device materialization for supplied topology. Distance-inferred bonds use a separate bounded cache keyed by `topology_id`, canonical-reference hash, and cutoff. Per-clip bond re-inference is forbidden because it would make static topology depend on trajectory fluctuations.

## D8 — selective BF16

Neural operations use BF16 autocast; geometry, coordinates, finite differences, and loss reductions remain FP32.

## D9 — engineering epochs must be exact

Tiny acceptance uses no replacement and full clip coverage. The production large-data sampler remains separately configurable.

## D10 — no silent data loss

Oversize samples are reported and either rejected explicitly or sent to the partition backend.

## D11 — partition is optional and execution-only

Only a core-plus-complete-receptive-field halo implementation is admissible. It must preserve state dict and numerical behavior; otherwise it stays experimental.

## D12 — architecture work is out of scope

No state-detail, decoder, probability, or ViSNet changes in this branch.
