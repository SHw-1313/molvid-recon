# Luna coding prompt — zero-preserving state/detail temporal codec v2 — 2026-09-03

You are the sole implementation worker for MolViD's next temporal-codec phase. Implement the
approved deterministic state/detail codec, prove its algebraic and geometric contracts, run the
bounded three-system/nine-trajectory T0 comparison, prepare an operator review packet, and then
stop. Do not begin the 64-system T1 experiment without a later explicit operator approval.

## 1. Repository, source, and target branch

The repository is expected at:

```text
/data4/users/sihao/workspace/PVB
```

Locate the existing checkout rather than cloning another target repository. Before mutation run:

```bash
pwd
git status --short --branch
git rev-parse HEAD
git remote -v
```

Approved source:

```text
branch: feat/visnet-spatial-v2
commit: 045dbb8809e9d7aee418eb355b6477e05088d4fd
target branch: feat/state-detail-codec-v2
```

The ViSNet v2 source and evidence remain historical and reproducible, but every new codec run
uses `spatial_backbone=torchmd_et`. Do not delete ViSNet code, rename its backends, or overwrite
its artifacts.

If the source worktree contains user changes, preserve and record them. Never reset, clean,
checkout over, delete, or rewrite history. If the target branch already exists, inspect it before
continuing. If the repository is not at the approved source and the difference cannot be safely
explained, stop and report it.

## 2. Read before editing

Read completely, in order:

1. root `AGENTS.md`;
2. the v1 and v2 phase documents required by the current root file, as read-only history;
3. this prompt;
4. `agents/state_detail_codec_v2/PLAN.md`;
5. `agents/state_detail_codec_v2/DECISIONS.md`;
6. `agents/state_detail_codec_v2/ACCEPTANCE.md`;
7. `agents/state_detail_codec_v2/TASKS.md`;
8. `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`;
9. `agents/state_detail_codec_v2/HANDOFF.md`.

Do not begin by rewriting the plan. The prepared plan, decisions, and acceptance gates are
binding. Update task status and append evidence as work proceeds.

## 3. Required root AGENTS.md transition

The current root file says the active phase is ViSNet v2 and forbids state/detail work. As task
S201, update it before implementation so that it records:

- active phase: **zero-preserving state/detail temporal codec v2**;
- source/target branch and exact source commit above;
- the new phase files and read order;
- all older phase files and results are read-only history;
- all Python, tests, training, evaluation, and plots run through `enter-container` with the
  `torch-ito` environment;
- no package installation, dependency upgrade, arbitrary network access, dataset download,
  destructive Git operation, or generated-checkpoint commit;
- new experiments use `torchmd_et`; ViSNet backends remain compatible but are out of the matrix;
- scope is codec implementation, tests, and T0 only;
- no T1 manifest generation, T1 training, static/dynamic large-data training, DiT, observation
  adapter, forecasting, rollout, AF3/MSA, VAE/KL/VQ, scaling-law, or full-data work;
- after the T0 review packet is complete, status must be `WAITING_FOR_OPERATOR_REVIEW`, and no
  later task may start without explicit operator approval.

Preserve unrelated repository and safety rules.

## 4. Scientific and architectural contract

This phase compares temporal compression before a future molecular Video DiT. The codec sees and
reconstructs the complete 16-frame clip. It does not decide which frames are observed or
predicted. Do not add 18-frame windows or history/future splitting.

Use deterministic AE training. There is no KL, posterior sampling, VQ, adversarial prior, MMD,
or latent-distribution objective.

### 4.1 Fixed spatial path

Use the existing `torchmd_et` frame encoder with the selected 128-channel configuration. Do not
modify its geometry operations. T0 must load the same recorded TorchMD frame-encoder weights for
every control and freeze them during the locked 30-epoch comparison. A separate gradient smoke
must prove that end-to-end unfreezing remains possible; it is not a fifth experiment.

### 4.2 No per-atom x0 coordinate bypass

The current code stores `batch.x[0]` as `CodecLatent.x_anchor` and adds it to every decoded frame.
Remove that behavior from every new v2 control. No complete clean frame may bypass the latent.

Translation handling may retain only one origin vector per sample, computed by an explicit,
masked centroid rule and recorded in the contract. The origin contains no per-atom structure.
The decoder reconstructs centered coordinates from equivariant latent vectors and then restores
the one-vector origin. Topology, atom types, masks, and timestamps remain allowed metadata.

Legacy checkpoint loading must keep the old per-atom-anchor behavior under the old schema. Do not
silently reinterpret old checkpoints.

### 4.3 Ratio and capacity contract

For spatial width `C=128`:

```text
R1-SD: 16 tokens x state(C)                         = 16C active features
R2-SD:  8 tokens x [state(C) + detail(C)]           = 16C active features
R4-SD:  4 tokens x [state(C) + compressed detail(C)] = 8C active features
R4-Matched-Pooling: 4 tokens x two unstructured C banks = 8C active features
```

Ratio denotes temporal-token reduction. Reports must also state active latent elements and raw
bank widths; do not claim that R2 halves total latent feature volume.

### 4.4 Fixed block-local lifting

For a generic scalar or vector frame feature `F`, use orthonormal Haar lifting.

R1:

```text
S = F0
```

R2:

```text
S = (F0 + F1) / sqrt(2)
D = (F1 - F0) / sqrt(2)
```

R4:

```text
S01  = (F0 + F1) / sqrt(2)
D01  = (F1 - F0) / sqrt(2)
S23  = (F2 + F3) / sqrt(2)
D23  = (F3 - F2) / sqrt(2)
S    = (S01 + S23) / sqrt(2)
Dmid = (S23 - S01) / sqrt(2)
detail coefficients = [Dmid, D01, D23]
```

R4 encodes the three detail coefficients into one C-wide detail bank and decodes that bank back
to three coefficients before the exact inverse lifting. Do not call these slow and fast motion.
Do not use residuals relative to assumed constant velocity.

The core is block-local. Do not add cross-block temporal attention; the later DiT owns long-range
token interaction.

### 4.5 Zero-preserving detail path

For every state/detail control, construction must enforce:

```text
E_detail(0) = 0
D_detail(0) = 0
```

Required consequences:

- repeated identical frames produce exactly zero detail before and after the learned bottleneck;
- zero detail decodes to identical within-block frame features;
- a repeated-static 16-frame input cannot acquire coordinate motion from time, state, bias,
  normalization, dropout, or a refiner;
- state and relative time may influence detail only multiplicatively;
- detail linear maps have no bias;
- detail normalizers have no additive affine offset;
- zero-preserving paths use no dropout;
- time conditioning cannot independently create vector displacement.

Static `T=1` has `detail_valid=false`; a synthetic repeated-static `T=16` has valid detail whose
value is zero. Never train by copying static structures into fake trajectories.

### 4.6 Matched pooling control

`R4-Matched-Pooling` is a new-framework capacity control, not the old `4 x C` codec. Use two
independent learned block-pooling heads, each producing one C-wide scalar/vector bank from four
frames. Its total token capacity is `4 x 2C = 8C`, it uses the new no-anchor coordinate decoder,
and it receives the same TorchMD features, data, losses, and optimization budget as R4-SD.

It has no state/detail semantics and no zero-motion requirement. Report its natural parameter
count; do not secretly enlarge R4-SD to match it. The existing old R4 result is historical only
and must not be retrained or placed in the locked four-way ranking.

## 5. Implementation requirements

Prefer new, focused v2 classes and an explicit config mode rather than mutating the semantics of
`CausalTemporalEncoder`, `CodecLatent`, or old checkpoint schemas in place. Suggested public
control names are:

```text
ratio1_state_detail
ratio2_state_detail
ratio4_state_detail
ratio4_matched_pooling
```

The structured latent contract must carry scalar/vector banks, token and component masks,
per-block original clocks, atom-to-sample assignment, topology, and only the allowed sample
origin. Every semantic config field must affect computation and round-trip through a versioned
model/checkpoint contract.

Implement variable masks without allowing invalid/padded frames to affect valid coefficients.
The required production clip is `T=16`; `T=1` static support and masked/partial-block behavior
must be explicit and tested rather than guessed.

Do not use the optional spatial refiner in any locked comparison.

## 6. Required evidence before GPU training

Pass all gates in `ACCEPTANCE.md`, including:

- analytical lifting/inverse-lifting round trips for h and v;
- exact coefficient ordering and masks for R1/R2/R4;
- shape and capacity accounting;
- T=1/T=16 and irregular-clock tests;
- repeated-static zero-detail and zero-coordinate-motion tests;
- dynamic-detail and every principal parameter gradient tests;
- SE(3) translation/rotation behavior;
- proof that no per-atom target/reference coordinate enters the new decoder;
- decoder target-mutation/no-hidden-target tests;
- old checkpoint and legacy-code regression;
- checkpoint/config round trip;
- full suite, `py_compile`, and `git diff --check`.

If a zero-motion test fails, fix the algebra. Do not add a consistency loss or relax the gate.

## 7. T0 training and evaluation

First run a bounded one-batch/three-clip overfit for all four controls. Then run the exact prior
three-system/nine-trajectory ATLAS protocol:

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
replicates: three per system
clip length: 16
time bucket: native dt_100ps
expected existing split: 441 train clips / 117 fixed holdout clips
epochs: 30 complete epochs
seed: 20260903
precision: FP32
spatial backbone: torchmd_et, common weights, frozen
spatial refiner: disabled
```

Before launching, independently verify sample IDs, all three replicates, counts, windows, split,
source paths, and manifest hashes. If the existing data do not match, stop instead of silently
constructing a replacement protocol.

Resolve the loss schedule from epoch fractions after the loader length is frozen:

```text
0%-10%:   coordinate 1.0, local 0.1, bond 0.1
10%-30%:  add velocity 0.1
30%-100%: add acceleration 0.05
```

Run the four controls with identical data exposure and one seed. With four free GPUs they may run
one per GPU; with two, use two waves. Never kill or occupy another user's process. Record actual
device mapping, wall time, peak memory, samples/s, and temporal-only versus end-to-end timing.

Evaluation must include rate/capacity, per-frame and per-block-offset RMSD/dRMSD, bond/contact/
clash, velocity/acceleration, RMSF, lagged correlation or ACF, frequency retention, block-boundary
jump, state/detail norms, full versus detail-zero decoding, and system-level results. Do not
interpret the three systems as a scientific architecture selection.

## 8. Mandatory operator pause before T1

After T0:

1. generate immutable JSON/CSV/Markdown reports and readable loss/evaluation plots;
2. complete `agents/state_detail_codec_v2/OPERATOR_REVIEW.md` with exact paths and a concise
   checklist;
3. append all commands, hashes, changed files, tests, metrics, runtime, failures, and limitations
   to `HANDOFF.md`;
4. set phase status to `WAITING_FOR_OPERATOR_REVIEW`;
5. stop.

Do not create the 64-system split or T1 manifest. Do not launch T1, even if all gates pass and
GPUs are idle. The only phrase that authorizes the next phase is a later explicit user message
approving T1 after code and T0 review.

## 9. Documentation discipline

- `PLAN.md`, initial `DECISIONS.md`, and `ACCEPTANCE.md` are read-only.
- Update `TASKS.md` status with concise evidence.
- Append implementation-forced decisions to `DECISIONS.md`; never rewrite approved decisions.
- Append dated work records to `HANDOFF.md`.
- Fill `OPERATOR_REVIEW.md` only with observed evidence; leave unknown items pending.
- Preserve all older agent files and outputs.

Do not mark a training result normal merely because loss decreased. Show complete curves,
per-control absolute metrics, detail utilization, zero-motion evidence, and checkpoint resume.
