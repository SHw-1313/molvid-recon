# DECISIONS — ViSNet spatial backbone v2

## D1 — v2 is a reference-faithful repair, not another inspired approximation

The implementation must reproduce pinned AI2BMD/PyG ViSNet operators before project extensions
are added. Shape/equivariance tests alone are not sufficient evidence of ViSNet fidelity.

## D2 — preserve v1 names and artifacts

`visnet_radius` and `visnet_bonded` keep their v1 semantics. Add `visnet_v2_radius` and
`visnet_v2_bonded`; do not silently change old checkpoint behavior or relabel v1 results.

## D3 — pinned references

AI2BMD branch `ViSNet` commit `497efaa190ee6f6cbc6030710c44208a01ece52d` is primary.
PyTorch Geometric commit `79d33965a40b7fa83616a9f598a0f8619f25d939` is an independent
cross-check. Exact hashes and licenses must be recorded in the handoff.

## D4 — limited network exception

The user authorizes Git clone/fetch only for the pinned ViSNet reference repositories under
`/data4/users/sihao/workspace`. This is not permission to install packages, download datasets,
or use arbitrary network resources.

## D5 — restore the actual ViS-MP attention rule

The faithful path uses edge-modulated Q/K/V, SiLU attention, and cosine cutoff. It does not use
the v1 incoming-edge softmax or average head attention before value aggregation.

## D6 — restore persistent edge geometry

Reference neighbor embedding, node-conditioned edge embedding, vector rejection, and edge-state
updates are mandatory. `vertex_type=edge` is the canonical experiment; `none` and `node` remain
available for parity and bounded ablation.

## D7 — lmax-2 internal, l1 public adapter

The canonical full-geometry candidate may keep eight `l<=2` components internally. Only the
first three `l=1` components enter the existing temporal codec. Truncation is allowed only after
all ViS-MP layers and output normalization.

## D8 — both scalar and vector normalization are real operations

`vecnorm_type` and `trainable_vecnorm` must instantiate and control reference `VecLayerNorm`.
Configuration fields that are stored but unused are acceptance failures.

## D9 — block awareness remains type-only

Keep `atom_embedding + block_type_embedding` for compatibility with the project and TorchMD
baseline. Do not call it EPT hierarchy and do not introduce residue tokens, pooling, or block
message passing in this phase.

## D10 — bond information is an isolated project extension

The binary covalent feature may modify radial features before the reference edge embedding.
Zeroing the bond embedding must recover the distance-only reference path on the same graph.

## D11 — fixed temporal/decoder scaffold

The temporal codec, coordinate decoder, losses, optimizer, data split, and ratio-1 settings stay
fixed. This phase diagnoses spatial geometry only.

## D12 — absolute metrics replace the v1 relative-initialization gate

Do not reject v2 only because it fails a percentage loss reduction from an unusually good
initialization. Report final absolute train/holdout metrics and direct comparisons with locked
v1/TorchMD results.

## D13 — correctness before GPU work

No micro-overfit begins until operator parity, geometry sensitivity, SE(3), gradients, graph
isolation, config semantics, and checkpoint regression all pass.

## D14 — bounded experiment matrix

The required micro matrix is TorchMD, legacy v1 bonded, v2 bonded lmax-1, and v2 bonded lmax-2.
Only the selected v2 configuration requires a new 441/117 tiny run. A full radius-v2 tiny run is
optional unless it is needed to resolve a remaining question.

## D15 — checkpoint semantics must be versioned

If new fields cannot be represented by the existing contract without ambiguity, bump the model
contract schema and provide explicit backward loading for v1. Never mutate old checkpoint files.

## D16 — no architecture spillover

No EPT hierarchy, AF3/MSA conditioning, temporal redesign, spatial refiner, diffusion, flow,
rollout, scaling law, or full-data experiment belongs to v2.
