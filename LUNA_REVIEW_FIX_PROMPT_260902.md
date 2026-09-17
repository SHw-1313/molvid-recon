# Luna review/fix prompt — ViSNet spatial backbone v2 — 2026-09-02

You are the sole review-and-fix worker for MolViD's ViSNet spatial backbone v2.

The v1 implementation is an implementation-valid scalar/vector network, but it is not a faithful
implementation of original ViSNet ViS-MP geometry. Your job is first to prove the discrepancy
against pinned reference code, then add new reference-faithful v2 backends and run the bounded
comparison. Do not begin by rewriting the plan.

## 1. Repository and branch

The working repository is expected under `/data4/users/sihao/workspace`. Locate the existing
MolViD checkout; do not create a second target checkout if one already contains the v1 branch.

Before any mutation:

```bash
pwd
git status --short --branch
git rev-parse HEAD
git remote -v
```

The source is:

```text
branch: feat/visnet-spatial-v1
commit: bca9ea98df643f4965b009688f2967bd669e7fe2
```

Create/use `feat/visnet-spatial-v2` from that source. If the worktree is dirty, preserve all
user-owned changes and record them; do not reset, checkout over, clean, or delete them. If the
target branch already exists, inspect it rather than recreating it.

## 2. Read before editing

Read completely, in this order:

1. root `AGENTS.md`;
2. all six files required by the current v1 root rule, in its stated order, as immutable
   historical context;
3. this prompt;
4. `agents/visnet_spatial_v2/PLAN.md`;
5. `agents/visnet_spatial_v2/DECISIONS.md`;
6. `agents/visnet_spatial_v2/ACCEPTANCE.md`;
7. `agents/visnet_spatial_v2/REFERENCE_PARITY.md`;
8. `agents/visnet_spatial_v2/TASKS.md`;
9. `agents/visnet_spatial_v2/HANDOFF.md`.

Do not overwrite or repurpose the v1 phase files.

## 3. Required root AGENTS.md update

The current root file still says the active phase is v1 and forbids all network use. The user has
authorized the v2 phase and two pinned reference checkouts. As task V201, update root `AGENTS.md`
before source retrieval so that it states:

- active phase is **ViSNet spatial backbone v2 reference-faithful repair**;
- the v2 files above are authoritative and must be read before code changes;
- v1 phase documents/artifacts are read-only historical evidence;
- work occurs on `feat/visnet-spatial-v2` based on `feat/visnet-spatial-v1` commit
  `bca9ea98df643f4965b009688f2967bd669e7fe2`;
- all Python/tests/fixture generation/training still run through `enter-container` and `torch-ito`;
- no package installation or dependency upgrade is allowed;
- the only network exception is Git clone/fetch of the two repositories and exact commits listed
  below into `/data4/users/sihao/workspace`;
- reference checkouts are read-only and are not production runtime dependencies;
- v1 backends, contracts, checkpoints, and result files must remain reproducible;
- no temporal/decoder/loss/data/evaluator redesign, EPT hierarchy, AF3, diffusion, flow, rollout,
  scaling-law, or full-data work is allowed in this phase.

Preserve all unrelated repository and safety rules in `AGENTS.md`.

## 4. Authorized reference retrieval

Only these external Git operations are authorized:

```text
AI2BMD primary reference
  remote: https://github.com/microsoft/AI2BMD.git
  branch: ViSNet
  commit: 497efaa190ee6f6cbc6030710c44208a01ece52d
  destination: /data4/users/sihao/workspace/AI2BMD_visnet_ref

PyG independent cross-check
  remote: https://github.com/pyg-team/pytorch_geometric.git
  commit: 79d33965a40b7fa83616a9f598a0f8619f25d939
  destination: /data4/users/sihao/workspace/pytorch_geometric_visnet_ref
```

If a destination exists, verify its remote and state before fetching. Do not overwrite an
unrelated directory. Checkout the exact commit and record `git rev-parse HEAD`, status, license,
and relevant source-file hashes in `HANDOFF.md`.

Do not clone TorchMD-Net as the ViSNet source. The v1 header's TorchMD-Net v2.0.0 `visnet.py`
attribution is incorrect and must be replaced with the actual source provenance.

No dataset download, package install, pip/conda mutation, or unrelated web access is authorized.

## 5. Review before fix

Before substantive code changes, compare current `module/visnet.py` line by line with:

```text
AI2BMD_visnet_ref/visnet/models/visnet_block.py
AI2BMD_visnet_ref/visnet/models/utils.py
pytorch_geometric_visnet_ref/torch_geometric/nn/models/visnet.py
```

Fill `agents/visnet_spatial_v2/REFERENCE_PARITY.md` with:

- exact reference symbol;
- current v1 behavior/missing operation;
- planned local v2 symbol;
- parity test or fixture;
- deliberate project extension, if any.

Your review must explicitly verify the known discrepancies:

- no reference Sphere path in v1;
- no NeighborEmbedding or node-conditioned EdgeEmbedding;
- no persistent non-final edge-state update;
- no vector rejection;
- `vertex=True` is metadata/validation rather than vertex geometry;
- `vecnorm_type` does not instantiate reference VecLayerNorm;
- v1 uses incoming-edge softmax and averages heads, unlike ViS-MP;
- v1 does not apply reference vector output normalization.

If any diagnosis is wrong, record evidence and amend `DECISIONS.md` before implementation.

## 6. Fix mission

Preserve:

```text
torchmd_et
visnet_radius       # v1
visnet_bonded       # v1
```

Add:

```text
visnet_v2_radius
visnet_v2_bonded
```

Prefer a new `module/visnet_v2.py` so v1 checkpoints and outputs remain stable.

Implement the pinned reference geometry, not a functional approximation:

1. Sphere with `lmax=1` and `lmax=2`;
2. ExpNormalSmearing canonical RBF and cosine cutoff;
3. VecLayerNorm including `max_min`;
4. NeighborEmbedding;
5. node-conditioned EdgeEmbedding;
6. edge-modulated Q/K/V and original SiLU attention/cutoff rule;
7. scalar/vector ViS-MP messages and invariant vector-dot scalar update;
8. vector rejection;
9. persistent non-final edge updates;
10. `vertex_type=none|edge|node`, canonical `edge`;
11. final-layer no-edge-update behavior;
12. scalar and vector output normalization.

Project extensions are limited to:

- atom type plus block/residue **type** embedding;
- optional binary covalent edge embedding;
- native/external graph adapters;
- final public codec adapter.

For `lmax=2`, keep `[M,8,C]` throughout all ViS-MP layers. Only after final output normalization,
export the first three `l=1` components as public `[M,3,C]`. Do not claim that block-type
embedding is EPT atom-block hierarchy.

Every new configuration field must control a real operation and round-trip through model
contracts/checkpoints. Add a schema version only if required, with explicit v1 loading.

Keep unchanged:

- temporal layers=1 and ratio=1;
- temporal encoder/decoder;
- coordinate head and disabled spatial refiner;
- data schema, split, physical-time handling, losses, optimizer, evaluator;
- graph-runtime repairs.

## 7. Required evidence sequence

Execute `TASKS.md` in order.

### Gate 0: reference parity and discriminating tests

Before GPU training, pass:

- deterministic operator/full-block parity against pinned reference fixtures;
- lmax-1/lmax-2 shape and SE(3) tests;
- angle and dihedral sensitivity with fixed explicit graph-edge distances;
- edge evolution and final-layer no-update tests;
- real `vertex_type`/`vecnorm_type` option tests;
- native one-graph and external no-rebuild tests;
- finite/nonzero geometry parameter and coordinate gradients;
- v1/TorchMD checkpoint and output regression;
- full relevant suite, py_compile, and `git diff --check`.

Parity target is FP32 `rtol <= 2e-5`, `atol <= 2e-6`. Do not weaken it without CPU evidence and a
recorded, justified CUDA scatter-order analysis.

### Gate 1: fixed 500-step micro-overfit

Compare:

```text
torchmd_et
visnet_bonded v1
visnet_v2_bonded lmax=1 vertex=edge
visnet_v2_bonded lmax=2 vertex=edge
```

Freeze protocol first. Require finite execution, at least 30% total-loss reduction, real
geometry-module gradients, checkpoint save/resume, absolute final metrics, runtime, and memory.
Select the final v2 config by absolute metrics, not relative improvement from different initial
losses.

### Gate 2: exact tiny comparison

Run the selected v2 bonded configuration on the immutable 441/117, T=16, dt=100 ps, FP32,
30-epoch protocol. Reuse stored TorchMD/v1 results only after hash/semantic comparability checks;
otherwise rerun the affected baseline. Do not overwrite v1 artifacts.

Run v2 radius as a bounded micro/performance comparison. A second 30-epoch radius run is not
required unless it resolves a material scientific ambiguity.

## 8. Documentation and stop rules

- Update task status and concise evidence in `TASKS.md`.
- Append dated evidence to `HANDOFF.md`; never rewrite earlier evidence.
- Update `REFERENCE_PARITY.md` as parity is proven.
- Append to `DECISIONS.md` only for a real forced deviation.
- Never alter `PLAN.md` or `ACCEPTANCE.md` merely to make a failing result pass.
- Keep generated checkpoints, datasets, large plots, and reference repositories out of Git.
- Do not push unless the operator explicitly requests it.

If reference parity, the required environment, writable branch, data, or GPU is unavailable,
complete all safe earlier tasks, record the exact blocker, and stop. Do not invent results, install
dependencies, silently fall back to host Python/CPU, or broaden scope.

Finish with:

- completed task IDs;
- changed files;
- reference and fixture hashes;
- exact commands/tests/tolerances;
- micro/tiny absolute metrics and gap recovery;
- outcome classification;
- limitations and multi-seed needs;
- explicit statement that no later architecture phase began.
