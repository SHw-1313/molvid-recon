# PLAN — ViSNet spatial backbone v1

## 1. Objective

Add ViSNet as an interchangeable atom-level spatial encoder for the existing deterministic
multi-frame codec and decide, through a reproducible tiny-overfit experiment, which spatial
backend should be carried into the later Scheme 1A state-detail codec.

This phase answers:

1. Can ViSNet stably replace the current TorchMD-ET frame encoder?
2. Does native geometric radius-graph ViSNet provide the best quality/speed/scalability tradeoff?
3. Does explicitly retaining topology bond edges and a binary bond feature materially improve ViSNet?

It does not attempt a publication-grade controlled comparison. Parameter count and graph edge
sets may differ naturally, but data, temporal path, decoder, losses, optimizer, precision,
and data exposure must remain fixed.

## 2. Model matrix

### A. `torchmd_et`

- Current repaired external CUDA radius graph.
- Current union with supplied topology bond edges.
- Current binary bond indicator.
- Existing TorchMD-ET spatial encoder.
- Serves as the development baseline.

### B. `visnet_radius`

- ViSNet representation-only core.
- Native CUDA radius graph constructed inside the ViSNet path exactly once.
- No topology registration, topology cache materialization, bond replication, bond union,
  or bond feature in the forward path.
- Intended as the speed/scalability-first candidate.

### C. `visnet_bonded`

- The same ViSNet representation core and parameterization as `visnet_radius`.
- Consumes the current repaired external graph:
  `unique(radius edges ∪ supplied directed covalent edges)`.
- Injects the current binary covalent indicator into ViSNet edge features.
- Intended as the chemistry-aware candidate.

All three use:

```yaml
temporal_layers: 1
temporal_ratio: 1
use_spatial_refiner: false
```

The temporal codec and decoder are unchanged.

## 3. Common spatial output contract

Every backend returns atom-level:

```python
h: Tensor  # [T, N_total, C], SE(3)-invariant scalar features
v: Tensor  # [T, N_total, 3, C], SE(3)-equivariant vector features
```

Recommended internal interface:

```python
@dataclass
class SpatialEncoderOutput:
    h: Tensor
    v: Tensor
    edge_index: Tensor | None
    edge_weight: Tensor | None
    edge_vec: Tensor | None
    edge_type: Tensor | None
    backend_used: str
    graph_mode: str
```

Frame-node packing must be independent of external graph construction, so native ViSNet does
not accidentally build an unused external graph before building its own radius graph.

## 4. ViSNet v1 design

Use a representation-only ViSNet core, not the full energy/force model.

Initial atom scalar state:

\[
h_i^0 = E_{\mathrm{atom}}(a_i) + E_{\mathrm{block}}(b_i)
\]

Recommended defaults:

```yaml
hidden_channels: 128
num_layers: 2
num_heads: 8
num_rbf: 32
lmax: 1
vertex: true
cutoff: 5.0
max_num_neighbors: 32
trainable_rbf: false
vecnorm_type: null
```

`lmax=1` is mandatory in this phase because the current temporal codec expects a Cartesian
three-axis vector representation. Reject `lmax=2` with an actionable configuration error.

Preserve upstream ViSNet attribution and license notices in adapted code.

### Bonded edge injection

For `visnet_bonded`, use the current binary edge label:

```text
0 = geometric/nonbonded edge
1 = supplied covalent edge
```

A suitable first implementation is:

\[
e_{ij}^{\mathrm{input}}
=
\operatorname{RBF}(d_{ij})
+
E_{\mathrm{bond}}(b_{ij})
\]

followed by the normal ViSNet edge embedding/update path.

Do not add bond order, angle graphs, torsion graphs, residue pooling, or new supervision in this phase.

## 5. Refactor boundary

Split the current frame encoder into:

```python
pack_frame_nodes(batch) -> FrameNodeBatch

build_external_graph(
    nodes: FrameNodeBatch,
    batch: ClipBatch,
) -> FrameGraphBatch
```

Dispatch:

```python
nodes = pack_frame_nodes(batch)

if graph_mode == "native_radius":
    output = backbone.forward_native(nodes)
else:
    graph = build_external_graph(nodes, batch)
    output = backbone.forward_external(nodes, graph)
```

Keep `build_graph(batch)` as a compatibility wrapper for current TorchMD tests and call sites.

## 6. Testing sequence

### Gate 0 — unit and contract tests

- factory/config parsing;
- output shapes;
- static `T=1` and trajectory `T=16`;
- no cross-frame or cross-sample native edges;
- scalar invariance and vector equivariance;
- finite/nonzero position and parameter gradients;
- `visnet_radius` proves external graph construction was not called;
- changing the binary edge type changes `visnet_bonded` output;
- checkpoint save/resume;
- legacy TorchMD config remains the default.

### Gate 1 — three-clip micro-overfit

Use exactly:

```text
atlas_5e3e_A_R1_w000000
atlas_1v7r_A_R1_w000000
atlas_2wlt_A_R1_w000000
```

Run 500 optimizer steps for the three spatial backbones, all with ratio-1 temporal.

Purpose: catch broken wiring, gradients, NaN/OOM, and checkpoint issues.
This is not the final backend decision.

### Gate 2 — final tiny-overfit

Systems:

```text
atlas_5e3e_A
atlas_1v7r_A
atlas_2wlt_A
```

Replicas:

```text
R1, R2, R3
```

Per replica:

```text
train:        w000000–w000048  (49 clips)
late_holdout: w000049–w000061  (13 clips)
```

Totals:

```text
9 trajectories
441 train clips
117 late-holdout clips
16 frames per clip
dt_100ps
```

Training:

```yaml
epochs: 30
max_steps_safety: 6000
precision: fp32
lr: 1.0e-4
weight_decay: 1.0e-6
warmup_steps: 20
grad_clip: 1.0
replacement: false
shuffle: true
max_tokens: 80000
eval_every_epochs: 1
```

Each train epoch must cover every one of the 441 train sample IDs exactly once.
Each evaluation must cover all 117 holdout clips.

## 7. Reported outputs

For every backend:

- resolved configuration;
- parameter count;
- graph mode and edge count;
- train loss components by epoch;
- train and late-holdout future RMSD/dRMSD;
- velocity and acceleration RMSE;
- bond RMSE;
- frequency retention;
- atom-frame tokens/s;
- wall time per epoch;
- peak allocated/reserved GPU memory;
- native-radius coverage of supplied topology bonds as a diagnostic only;
- checkpoint paths and resume result.

Aggregate results into JSON, CSV, Markdown, and plots.

## 8. Recommendation rule

This run selects a development default, not a final scientific winner.

Prefer `visnet_radius` when its late-holdout future dRMSD is within 10% of the best backend,
because it has the simplest graph/data requirements.

Prefer `visnet_bonded` when it shows a meaningful, coherent improvement in dynamic or geometric
metrics over `visnet_radius` and its step time is no more than 1.5 times the native version.

Retain `torchmd_et` when both ViSNet variants fail the implementation or tiny-overfit quality gates.

## 9. Explicit non-goals

- parameter matching;
- identical spatial edge sets;
- multi-seed statistical claims;
- state-motion-observation decomposition;
- state-detail temporal packing;
- residue-level temporal tokens;
- decoder changes;
- AF3-like conditions;
- probability learning;
- VideoDiT;
- full-dataset training.
