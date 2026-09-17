# PLAN — ViSNet spatial backbone v2 reference-faithful repair

## 1. Why this phase exists

The v1 phase established a stable interchangeable-backbone scaffold, but its
`_MolViSNetLayer` is a custom scalar/vector attention layer rather than a faithful ViSNet
ViS-MP block. The v1 experiment therefore answers whether that lightweight approximation works
inside the MolViD codec; it does not answer whether the original ViSNet geometry is useful.

Known v1 tiny-overfit facts, which must remain immutable:

| Backbone | Final future dRMSD | Final future RMSD | Velocity RMSE | Frequency retention |
|---|---:|---:|---:|---:|
| `torchmd_et` | 0.644799669 | 0.873626672 | 0.004180777 | 0.670033 |
| `visnet_radius` v1 | 0.682236567 | 0.932992843 | 0.004453239 | 0.636193 |
| `visnet_bonded` v1 | 0.679827729 | 0.928350341 | 0.004442983 | 0.619523 |

The v1 bonded model was better than TorchMD on frame-zero dRMSD and future bond RMSE, but worse
on future global geometry and dynamic metrics. This motivates a geometry-faithful repair while
keeping the temporal codec and decoder fixed.

## 2. Scientific question

This phase asks one primary question:

> When the original ViSNet geometric operations are restored, does the current MolViD codec
> retain the local-geometry advantages of v1 while recovering future structure and motion
> quality relative to TorchMD-ET?

It separately answers:

1. Can the original ViS-MP operators be reproduced numerically from pinned reference code?
2. Does internal `lmax=2` geometry add value when the downstream codec consumes only an `l=1`
   Cartesian vector representation?
3. Is any remaining deficit caused by ViSNet geometry, rather than by the v1 approximation?

## 3. Reference sources and provenance

Primary source:

```text
repository: https://github.com/microsoft/AI2BMD.git
branch: ViSNet
commit: 497efaa190ee6f6cbc6030710c44208a01ece52d
local reference: /data4/users/sihao/workspace/AI2BMD_visnet_ref
primary files:
  visnet/models/visnet_block.py
  visnet/models/utils.py
```

Independent implementation cross-check:

```text
repository: https://github.com/pyg-team/pytorch_geometric.git
commit: 79d33965a40b7fa83616a9f598a0f8619f25d939
local reference: /data4/users/sihao/workspace/pytorch_geometric_visnet_ref
primary file:
  torch_geometric/nn/models/visnet.py
```

The worker must record the checked-out hashes. Reference repositories are read-only scientific
specifications. Do not modify them, import them as production runtime dependencies, or copy
unreviewed files wholesale. Preserve the MIT attribution for adapted code.

The old comment claiming a source at `torchmd-net v2.0.0/torchmdnet/models/visnet.py` must be
corrected: that path is not the source of the v2 implementation.

## 4. Versioning and backend matrix

Do not change the semantics of the checked-in v1 names or checkpoints.

Retain:

```text
torchmd_et
visnet_radius          # v1 legacy
visnet_bonded          # v1 legacy
```

Add:

```text
visnet_v2_radius       # reference-faithful core, native radius graph
visnet_v2_bonded       # same core, external radius + topology-bond graph
```

The default remains `torchmd_et` until the v2 evidence is reviewed. Backend names, resolved
geometry configuration, reference provenance, graph mode, and public/internal vector layouts
must be stored in the model contract and checkpoint.

Prefer a new focused module such as `module/visnet_v2.py`. Keep `module/visnet.py` as the v1
implementation except for minimal factory/export wiring needed to expose v2.

## 5. Required reference-faithful geometry

### 5.1 Input and radial geometry

For atom (i):

\[
x_i^0 = E_{\mathrm{atom}}(a_i) + E_{\mathrm{blocktype}}(b_i).
\]

`b` remains residue/block **type**, not a unique block identity. This is a project extension to
the atom embedding, not EPT-style hierarchy.

Implement the reference operations:

- `Sphere` for `lmax in {1, 2}`;
- normalized edge directions with self-loop-safe handling;
- `ExpNormalSmearing` as the canonical RBF;
- optional Gaussian RBF only as an explicit ablation/compatibility mode;
- cosine cutoff;
- reference `NeighborEmbedding`;
- reference node-conditioned `EdgeEmbedding`.

For `lmax=1`, the directional basis has 3 components. For `lmax=2`, it has 8 components:

\[
K=(l_{\max}+1)^2-1.
\]

### 5.2 ViS-MP message block

The implementation must follow the reference computation, including:

- scalar `LayerNorm` and vector `VecLayerNorm`;
- scalar Q/K/V projections;
- edge-dependent `dk` and `dv` modulation;
- reference SiLU attention times cosine cutoff;
- **no incoming-edge softmax** in the faithful path;
- scalar-to-vector directional message construction;
- scalar/vector aggregation;
- vector-vector invariant dot products feeding scalar updates;
- residual scalar and vector updates.

The v1 behavior of averaging attention over heads before value aggregation must not appear in
the v2 faithful path.

### 5.3 Edge update and higher-order geometry

Every non-final ViS-MP layer must update edge features using vector rejection:

\[
\operatorname{reject}(v,d)=v-(v\cdot d)d.
\]

The following variants must exist because they are part of the reference implementation:

```text
vertex_type: none
vertex_type: edge
vertex_type: node
```

The canonical experiment uses `vertex_type=edge`. It includes both the source/target rejection
term and the additional vertex term used by `ViS_MP_Vertex_Edge`. `none` and `node` are operator
parity/ablation modes; they are not separate full experimental lanes unless evidence requires it.

The final ViS-MP layer does not update edge features, matching the reference code.

### 5.4 Output normalization and codec adapter

Apply both scalar output normalization and reference vector output normalization.

Internal representation:

```python
h: Tensor       # [M, C]
v_full: Tensor  # [M, K, C], K=3 for lmax=1, K=8 for lmax=2
```

Public codec representation:

```python
h: Tensor  # [M, C]
v: Tensor  # [M, 3, C]
```

For `lmax=1`, `v is v_full`. For `lmax=2`, export the first three reference-basis components,
which are the `l=1` Cartesian components, while retaining `v_full` only as optional diagnostic
metadata. The full `l<=2` state must participate in every ViS-MP layer before this final adapter;
do not truncate to `l=1` between layers.

Tests must prove scalar invariance and public `l=1` vector equivariance for both `lmax=1` and
`lmax=2` configurations.

### 5.5 Bonded extension

The reference ViSNet is distance-graph based. The project-specific bonded extension remains:

\[
e^{\mathrm{radial}}_{ij}=\operatorname{RBF}(d_{ij})+E_{\mathrm{bond}}(b_{ij}),
\]

before the reference edge embedding. This extension must be isolated so that setting the bond
embedding to zero recovers the reference-faithful distance-only path.

`visnet_v2_radius` owns one native radius construction and never prepares the external graph.
`visnet_v2_bonded` consumes the existing external graph and never constructs a second one.

## 6. Configuration contract

Add explicit, non-no-op fields:

```yaml
model:
  spatial_backbone: visnet_v2_bonded
  lmax: 2
  vertex_type: edge
  rbf_type: expnorm
  trainable_rbf: false
  vecnorm_type: max_min
  trainable_vecnorm: false
```

Backward compatibility:

- old `torchmd_et`, `visnet_radius`, and `visnet_bonded` contracts load unchanged;
- legacy `vertex: true/false` remains valid for v1;
- v2 requires or resolves an explicit `vertex_type` and records the resolved value;
- unsupported combinations fail before training with an actionable message;
- no configuration field may be accepted and then ignored.

If the existing checkpoint schema cannot express these semantics unambiguously, increment the
schema version while retaining a loader for v1 contracts. Do not rewrite old artifacts.

## 7. Parity and geometry tests before training

`REFERENCE_PARITY.md` is a required audit table, not optional documentation. Before substantive
implementation, fill its reference symbol and target mapping. After implementation, attach a
test or fixture to every required row.

Required tests include:

1. deterministic operator parity against the pinned reference for Sphere, RBF, vector norm,
   neighbor embedding, edge embedding, and one-layer ViS-MP;
2. deterministic full-block parity with project extensions disabled or zeroed;
3. `vertex_type` changes real modules and calculations, not metadata only;
4. `vecnorm_type` changes real normalization, not metadata only;
5. lmax-1 and lmax-2 shape and SE(3) tests;
6. explicit-chain angle and dihedral sensitivity with unchanged graph-edge distances;
7. final-layer edge-update absence;
8. no cross-frame/sample edges and no duplicate graph construction;
9. finite, nonzero coordinate and parameter gradients;
10. v1 checkpoint and output-regression tests.

Reference parity fixtures must be small, deterministic, versioned, and record source commit,
configuration, dtype, seed, and tolerances. They may be generated from the pinned checkout, but
production code and normal tests must not require `/data4/...` to exist.

## 8. Fixed downstream scaffold

Keep unchanged during the scientific comparison:

```yaml
temporal_layers: 1
temporal_ratio: 1
use_spatial_refiner: false
```

Also keep unchanged:

- clip schema and 16-frame ordering;
- time conditioning;
- temporal encoder and decoder implementation;
- coordinate head;
- loss definitions and weights;
- optimizer, warmup, precision, and data exposure;
- graph-runtime repairs.

This phase does not add EPT residue tokens, AF3/MSA context, decoder redesign, flow matching,
diffusion, state-detail packing, rollout, or full-data training.

## 9. Training gates

### Gate 0 — implementation and parity

All parity, geometry, shape, graph, SE(3), gradient, config, and checkpoint tests pass. No GPU
training begins before this gate is green.

### Gate 1 — 500-step micro-overfit

Use the existing three fixed clips and compare:

```text
torchmd_et
visnet_bonded              # v1 legacy diagnostic
visnet_v2_bonded lmax=1, vertex_type=edge
visnet_v2_bonded lmax=2, vertex_type=edge
```

Every run must complete 500 steps, remain finite, exercise all principal geometry modules, reduce
total loss by at least 30%, save a checkpoint, and resume for one additional update.

Use this gate to choose `lmax=1` or `lmax=2` for the final v2 tiny run. Selection is based first
on final absolute loss and dynamic metrics, then runtime/memory. Do not select by relative loss
reduction from a different initialization alone.

### Gate 2 — exact 441/117 tiny comparison

Use the existing immutable v1 manifest/protocol:

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
replicas: R1, R2, R3
train: w000000-w000048, 441 clips
late holdout: w000049-w000061, 117 clips
T=16, dt=100 ps, zero overlap
FP32, 30 epochs, max_tokens=80000, replacement=false
```

Required comparison:

```text
stored immutable torchmd_et result
stored immutable visnet_bonded v1 result
new selected visnet_v2_bonded result
```

Reuse stored baselines only after checking commit, manifest, split, protocol, evaluator, and
metric semantics. If any scientific input differs, rerun the affected baseline rather than
mixing incomparable numbers.

`visnet_v2_radius` receives a micro/performance run after bonded v2 is correct; a second 30-epoch
radius run is optional unless bonded-versus-radius remains scientifically unresolved.

## 10. Decision and reporting

Report absolute final metrics, not just pass/fail or improvement from initialization:

- frame-zero and future RMSD/dRMSD;
- future bond and contact error;
- velocity and acceleration RMSE;
- frequency retention;
- train and holdout total loss;
- parameter count, wall time, tokens/s, allocated/reserved memory;
- graph and topology diagnostics.

Also report gap recovery relative to v1 and TorchMD as a descriptive statistic:

\[
R_{\mathrm{gap}} =
\frac{m_{\mathrm{v1}}-m_{\mathrm{v2}}}
     {m_{\mathrm{v1}}-m_{\mathrm{TorchMD}}},
\]

for lower-is-better metrics when the denominator is meaningful. Do not turn this statistic into
a universal quality threshold.

Classify the outcome as:

```text
PARITY_FAILED          reference-faithful claim is invalid
IMPLEMENTED            parity/correctness passed, training not yet decisive
V2_NO_GAIN             v2 does not improve v1 dynamic quality
V2_PARTIAL_RECOVERY    v2 improves v1 but does not match TorchMD
V2_COMPETITIVE         v2 is within run-to-run uncertainty of TorchMD
V2_PREFERRED           v2 coherently improves the locked primary and guardrail metrics
```

One-seed tiny-overfit can establish `V2_NO_GAIN` or `V2_PARTIAL_RECOVERY`; it cannot by itself
justify a strong model-family claim. A `V2_COMPETITIVE` or `V2_PREFERRED` decision must identify
which points require multi-seed confirmation.

## 11. Stop condition

Stop after the v2 comparison report and final handoff. Do not begin AF3 fusion, state-detail
packing, a new trunk, rollout, scaling-law training, or full-dataset training in this phase.
