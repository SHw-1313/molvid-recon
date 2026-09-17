# ACCEPTANCE — ViSNet spatial backbone v2

## A. Source and audit gate

All are required:

- pinned AI2BMD and PyG commits are checked out and recorded;
- licenses and exact adapted symbols are recorded;
- `REFERENCE_PARITY.md` maps every required geometry operation to local code and a test;
- the incorrect TorchMD-Net v2.0.0 ViSNet source attribution is removed or corrected;
- production code does not depend on `/data4/...` reference paths.

## B. Operator parity gate

Using deterministic FP32 inputs and copied/mapped reference weights:

- Sphere lmax-1 and lmax-2 outputs match the pinned reference;
- ExpNormalSmearing and cosine cutoff match;
- VecLayerNorm `none` and `max_min` match;
- NeighborEmbedding matches;
- EdgeEmbedding matches;
- ViS-MP `none` matches;
- ViS-MP vertex-edge matches;
- ViS-MP vertex-node matches;
- a full multi-layer representation block matches with block/bond extensions disabled.

Default numerical target is `rtol <= 2e-5`, `atol <= 2e-6` in FP32. If a CUDA scatter ordering
requires a wider tolerance, the worker must demonstrate CPU parity first, quantify the CUDA
difference, and record the justified tolerance instead of silently weakening the gate.

Fixtures must record source commit, seed, dtype, configuration, input hash, state-dict hash, and
expected-output hash. Normal tests must run without the external reference checkout.

## C. Geometry-operation gate

All are required:

- edge features change across non-final layers and stay unchanged after the final layer;
- vector rejection is exercised by non-collinear synthetic geometry;
- `vertex_type=edge` exercises both rejection terms;
- `vertex_type=node` exercises its node-level vertex term;
- an explicit three-atom chain is angle-sensitive while its graph-edge lengths are unchanged;
- an explicit four-atom chain is dihedral-sensitive while bond lengths and bond angles are fixed;
- disabling edge/vertex updates produces a measurably different result on these fixtures;
- every geometry-specific parameter receives a finite, nonzero gradient.

These tests must fail against the v1 `_MolViSNetLayer`; otherwise they do not discriminate the
missing geometry that motivated v2.

## D. Representation and SE(3) gate

- lmax-1 internal/public shapes are `[M,3,C]`;
- lmax-2 internal shape is `[M,8,C]` and public shape is `[M,3,C]`;
- lmax-2 truncation occurs only once after all ViS-MP layers and output normalization;
- scalar output is translation/rotation invariant within declared tolerance;
- public l1 output is translation invariant and rotation equivariant;
- T=1 and T=16 work;
- no edge crosses frame or sample boundaries;
- native v2 constructs exactly one graph and never prepares external topology;
- bonded v2 consumes the supplied graph and never constructs another graph;
- coordinate and all principal parameter gradients are finite and nonzero.

## E. Configuration and compatibility gate

- `visnet_v2_radius` and `visnet_v2_bonded` are selectable;
- legacy `torchmd_et`, `visnet_radius`, and `visnet_bonded` outputs/checkpoints remain valid;
- `vertex_type`, `lmax`, `rbf_type`, `vecnorm_type`, and `trainable_vecnorm` alter real modules;
- no accepted option is a no-op;
- resolved geometry configuration and reference provenance round-trip through checkpoints;
- unsupported combinations fail before optimizer construction;
- old v1 aggregate results and files are not overwritten.

## F. Regression gate

- the full pre-v2 test suite passes;
- a frozen v1 synthetic fixture reproduces its previous output within tolerance;
- TorchMD model contract and forward path are unchanged;
- temporal encoder, coordinate decoder, loss code, data schema, and evaluator have no semantic
  diff unless an implementation blocker is documented and approved;
- no new dependency is installed.

## G. 500-step micro-overfit gate

For TorchMD, v1 bonded, v2 bonded lmax-1, and v2 bonded lmax-2:

- 500 steps complete in FP32;
- no NaN, Inf, OOM, CPU production fallback, or hidden graph rebuild occurs;
- total loss falls by at least 30%;
- all principal v2 geometry modules receive gradients during real training;
- checkpoint saves at step 500 and resumes to step 501;
- final absolute losses, future geometry, dynamic metrics, runtime, and memory are reported.

Failure of one v2 configuration does not authorize changing the dataset, loss, or evaluator.

## H. 441/117 tiny gate

- manifest and protocol hashes match the immutable v1 experiment or affected baselines are rerun;
- every epoch covers 441 train clips exactly once;
- all 117 late-holdout clips are evaluated every epoch;
- the selected v2 configuration runs 30 epochs under the locked protocol;
- final absolute train/holdout loss, RMSD/dRMSD, bond/contact, velocity/acceleration, frequency,
  throughput, memory, and graph diagnostics are reported;
- the report separates implementation parity from empirical model quality;
- no decision is based only on percentage improvement from untrained initialization.

## I. Final classification gate

The final report uses exactly one of:

```text
PARITY_FAILED
IMPLEMENTED
V2_NO_GAIN
V2_PARTIAL_RECOVERY
V2_COMPETITIVE
V2_PREFERRED
```

It states the evidence, limitations, remaining uncertainty, and whether multi-seed confirmation
is required. It does not claim that ViSNet as a model family wins or loses from one tiny run.
