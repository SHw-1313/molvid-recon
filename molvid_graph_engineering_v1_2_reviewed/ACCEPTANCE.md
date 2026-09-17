# Engineering acceptance matrix — v1.2

| Gate | Repaired topology-bond full graph | Distance-only bond ablation | Optional halo subgraph |
|---|---:|---:|---:|
| Same parameter keys/shapes/count; no new persistent cache buffers | required | required | required |
| Production graph tensors remain on CUDA | required | required | required |
| No graph-related CPU/dense fallback | required | required | required |
| Synthetic exact legacy topology-edge semantics | required | n/a by design | inherited from full builder |
| Every directed graph-input bond exactly once, correct flags | supplied bonds | inferred bonds | required in induced subgraphs |
| No full-edge sort/unique/isin | required | required | required |
| No topology/atom/residue information used for distance inference | n/a | required | n/a |
| One deterministic canonical reference; identical bonds across topology_id | n/a | required | n/a |
| Distance-bond neighbor cap not saturated | n/a | required | n/a |
| FP32 output/loss/gradient regression | required | finite only; no equivalence expected | against repaired full |
| BF16 finite; total-loss relative difference <= 2% | required | required versus its own FP32 run | required |
| True pinned/non-blocking multi-worker loading | required | required | required |
| Exact no-replacement tiny epochs; no missing/duplicate clips | required | required | required |
| No silent oversize exclusion | required | required | required |
| Atom-frame throughput >= 1.20x baseline | required | report only | report only |
| Bond precision/recall/F1/Jaccard versus supplied topology | report | required | report |
| Degree, isolated-atom, and connected-component audit | report | required | report |
| Every original atom covered as core exactly once | n/a | n/a | required |
| Complete `1 + spatial_layers` receptive-field halo | n/a | n/a | required |
| Cross-core covalent-bond test | n/a | n/a | required |
| SE(3) test | required | required | required |
| Peak-memory reduction >=20% or enables an otherwise ineligible sample | n/a | n/a | required for recommendation |
| 200-step acceptance | required | required | required |
| Five exact tiny-set epochs | required | required | required |
| Convergence/quality claim from short run | forbidden | forbidden | forbidden |
| 50-epoch/full-data run | forbidden | forbidden | forbidden |
