# HANDOFF — graph/runtime engineering v1.2

Status: not started

## Branch

`fix/graph-runtime-v1`

## Baseline commit and environment

To be filled before editing.

## Mandatory Phase A evidence

- parameter/state-dict compatibility;
- CUDA device-residency and no-fallback evidence;
- legacy/new edge-union and bond-flag checks;
- topology-cache hit/eviction and state-dict cleanliness;
- FP32 regression and BF16-closeness results;
- pinned multi-worker loader result;
- exact no-replacement epoch coverage and warmup/resume checks;
- oversize eligibility manifests;
- 200-step and five-epoch results;
- atom-frame throughput improvement versus baseline.

## Required Phase B bond-source ablation evidence

- exact distance-only definition and configuration;
- proof that graph construction does not consume supplied topology, atom type, residue/block identity, or bond order;
- canonical-reference selection and identical-bonds-across-topology_id test;
- inferred/topology precision, recall, F1, Jaccard, and directed-edge counts;
- missed/false-positive distance distributions;
- degree, isolated-atom, and connected-component diagnostics;
- common initialization, sample IDs, batch schedule, seed, optimizer, and precision evidence;
- 200-step and five-exact-epoch development results;
- explicit statement that no convergence claim was made.

## Optional Phase C evidence

- core/halo statistics and exact coverage;
- full/partition output/loss/gradient equivalence;
- cross-boundary bond and SE(3) tests;
- memory or oversize-eligibility benefit;
- recommendation: default, experimental, or rejected.

## Stop condition

Do not proceed to state-detail or other architecture work from this branch.
