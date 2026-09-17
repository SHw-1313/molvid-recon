# DECISIONS — ViSNet spatial backbone v1

## D1 — three natural backends, not a strict scientific ablation

Compare `torchmd_et`, `visnet_radius`, and `visnet_bonded` under the same downstream codec and
data protocol. Parameter count and spatial edge sets need not match.

## D2 — ratio-1 temporal is the fixed scaffold

Use `temporal_layers=1`, `temporal_ratio=1` for every run. Legacy ratio-4 would confound
the spatial comparison with its known temporal information bottleneck.

## D3 — native ViSNet must not build two graphs

`visnet_radius` bypasses the external graph builder completely and invokes its native CUDA
radius graph exactly once.

## D4 — bonded and radius ViSNet share one representation core

The ViSNet architecture and parameterization are shared. Only graph provision and binary
covalent edge information differ.

## D5 — representation-only ViSNet

Do not instantiate energy, force, atom-reference, or graph-reduction heads. Return atom-level
scalar and vector representations to the existing temporal codec.

## D6 — block-aware scalar initialization

Use both project atom type and block/residue type embeddings. Optional block representations
may be exposed as diagnostics but do not enter the temporal path in this phase.

## D7 — `lmax=1` only

The current temporal vector contract is `[N,3,C]`. Higher irreps are deferred.

## D8 — binary bond information only

Use the existing `bond/nonbond` label. Bond order and richer chemical edge types are future work.

## D9 — tiny-overfit is the final acceptance experiment

The final comparison uses 441 early-time training clips and 117 late-time holdout clips from
the same three systems and nine trajectories. It is a development selection, not unseen-protein
generalization.

## D10 — no planning delegation to the implementation worker

The worker executes these prepared documents. It may update task status and evidence, but it
must not rewrite the plan or silently broaden scope.
