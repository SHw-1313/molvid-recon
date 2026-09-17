# Luna coding prompt — R2/R4 state-detail latent DiT probe v1 — 2026-09-07

You are the sole implementation worker for MolViD's first state/detail latent DiT probe. Build one
shared, SE(3)-equivariant, physical-time-aware DiT path for the frozen R2-SD and R4-SD codecs,
prove its interfaces with tests, run only the bounded smoke authorized below, prepare an operator
review packet, and stop.

This is an implementation-readiness phase. It is not the real R2/R4 scientific pilot and must not
be expanded into a full molecular world model.

## 1. Repository, source, and isolated target

Expected existing repository:

```text
/data4/users/sihao/workspace/PVB
```

Audited source:

```text
remote: https://github.com/SHw-1313/molvid.git
source branch: fix/state-detail-codec-v2-t1-gates
source commit: 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541
target branch: feat/dit-state-detail-probe-v1
recommended worktree: /data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1
```

The source checkout is running the authorized codec T1 experiment. Do not edit or run DiT work in
that checkout. Audit it read-only, then create/use the dedicated target worktree. Before mutation,
record:

```bash
pwd
git status --short --branch
git rev-parse HEAD
git remote -v
git worktree list --porcelain
nvidia-smi
```

If the source branch has advanced beyond the audited commit, inspect and record the difference.
Do not silently rebase onto new codec semantics. If the target branch/worktree already exists,
inspect it and preserve its contents. Never reset, clean, stash, checkout over, or delete user work.

## 2. Prepared files and read order

The following prepared files must exist in the target branch:

```text
LUNA_DIT_STATE_DETAIL_PROBE_V1_PROMPT_260907.md
agents/dit_state_detail_probe_v1/PLAN.md
agents/dit_state_detail_probe_v1/DECISIONS.md
agents/dit_state_detail_probe_v1/ACCEPTANCE.md
agents/dit_state_detail_probe_v1/TASKS.md
agents/dit_state_detail_probe_v1/HANDOFF.md
agents/dit_state_detail_probe_v1/OPERATOR_REVIEW.md
agents/dit_state_detail_probe_v1/worker.toml
```

Read completely, in order:

1. root `AGENTS.md`;
2. the state/detail codec files named by root `AGENTS.md`, as read-only history;
3. this prompt;
4. `agents/dit_state_detail_probe_v1/PLAN.md`;
5. `agents/dit_state_detail_probe_v1/DECISIONS.md`;
6. `agents/dit_state_detail_probe_v1/ACCEPTANCE.md`;
7. `agents/dit_state_detail_probe_v1/TASKS.md`;
8. `agents/dit_state_detail_probe_v1/HANDOFF.md`;
9. `agents/dit_state_detail_probe_v1/OPERATOR_REVIEW.md`.

PLAN, prepared DECISIONS, and ACCEPTANCE are binding. Do not begin by rewriting them. Update task
status only after evidence exists, append real implementation-driven decisions, append evidence to
HANDOFF, and fill OPERATOR_REVIEW at the end.

## 3. Required root AGENTS.md transition

On the new DiT branch only, update root `AGENTS.md` before source implementation so it records:

- active phase: **R2/R4 state-detail latent DiT probe v1**;
- source commit and dedicated branch/worktree;
- the read order above;
- active T1 checkout/processes/outputs and all preceding phase artifacts are read-only;
- all Python, tests, smoke, evaluation, and plotting run through `enter-container` with
  `conda activate torch-ito`;
- no package installation, dependency upgrade, arbitrary download, destructive Git operation,
  or generated-checkpoint commit;
- only R2-SD and R4-SD are candidates; R1 and matched pooling are historical controls;
- codec/frame encoder are frozen and no codec/T1 script is rewritten;
- current scope is implementation, correctness tests, and bounded smoke only;
- no real pilot, full-T1-data DiT training, T1 test access, scaling, static mixing, AF3/MSA,
  VAE/KL/VQ, long rollout, H=2, energy, ensemble, or later architecture work;
- final status is `WAITING_FOR_T1_AND_OPERATOR_REVIEW`.

Preserve relevant repository-wide safety and compatibility rules. Do not edit root `AGENTS.md` in
the running T1 checkout.

## 4. Scientific boundary

The repaired deterministic codec already answers whether true encoder latents can reconstruct a
16-frame trajectory. This phase asks a different question: whether a generative DiT can model the
encoded latent distribution without destroying the codec's structural and dynamic information.

Do not select a codec by autoencoder RMSD alone. Conversely, a 100-step DiT smoke is not evidence
that either latent distribution is scientifically better. Its purpose is only to prove that the
same implementation executes correctly for both ratios.

Provisional candidates are fixed:

```text
primary scalable candidate: ratio4_state_detail
safety candidate:          ratio2_state_detail
not DiT candidates:        ratio1_state_detail, ratio4_matched_pooling
```

## 5. Frozen tensor contract

Use and assert the notation from PLAN. For T=16 and codec width C=128:

```text
R2: K=8
R4: K=4

state_h, detail_h  [K, N, C]
state_v, detail_v  [K, N, 3, C]

h_codec = concat(state_h, detail_h, -1)  [K, N, 2C]
v_codec = concat(state_v, detail_v, -1)  [K, N, 3, 2C]
```

State/detail share one sequence location. Never double the sequence length by making them separate
tokens. Scalar projections may use ordinary channel linear maps. Vector projections are bias-free
channel maps and never mix xyz.

`raw_detail_h/v` must be absent from the DiT batch and may be `None` in generated latents. Static
N-axis topology, masks, real block times, ratio id, and `[B,3]` sample origin are metadata or
conditions. Target coordinates, raw Haar coefficients, frame-expanded radius graphs, distances,
edge vectors, contacts, and future-derived metadata are prohibited.

For H>0, derive `sample_origin` from the clean observed frame-0 centroid. For H=0, use a fixed zero
origin. Never derive it from target-only future coordinates. State field masks use valid tokens;
detail field masks also intersect the codec's `detail_valid` field.

## 6. Required modules

Implement focused modules, with small compatibility exports only when needed:

```text
module/state_detail_latent_adapter.py
module/latent_rectified_flow.py
module/molecular_dit.py
trainer/dit_trainer.py
evaluation/dit_evaluation.py
config/dit_state_detail_probe.yaml
scripts/run_state_detail_dit_smoke.py
tests/test_state_detail_latent_adapter.py
tests/test_latent_rectified_flow.py
tests/test_molecular_dit.py
tests/test_dit_trainer.py
```

Do not rewrite `module/model.py`, `module/state_detail_codec_v2.py`, codec trainer/evaluator
semantics, or T1 scripts. Reuse small proven scalar/vector utilities only when their masking and
attention semantics match this phase. The deterministic codec's causal temporal block is not the
DiT temporal block.

### 6.1 Adapter and statistics

Implement:

- R2/R4 validation and packing;
- scalar/vector projections and output heads;
- reconstruction of generated `StateDetailLatent` with raw diagnostics absent;
- train-only, ratio-specific, mask-aware latent statistics;
- scalar per-channel mean/std;
- vector per-channel RMS without directional mean;
- inverse normalization;
- versioned contracts and hashes.

Statistics may use synthetic or explicitly labeled T0 data for the bounded smoke. Do not claim
they are production statistics. The later pilot must recompute/freeze them from approved T1 train
data only.

### 6.2 Rectified flow

Implement arbitrary-rank broadcasting for:

```text
z_tau = (1 - tau) * eps + tau * z_data
u_target = z_data - eps
```

Use one `tau` per sample across all its atom/time/scalar/vector elements. Vector Gaussian noise is
isotropic. Compute four independently mask-normalized MSE terms for state_h, detail_h, state_v,
and detail_v, then average them equally. Do not add coordinate loss, KL, VQ, adversarial loss,
SNR weighting, or learned loss weights.

### 6.3 Observation policy

Support frame prefix lengths H=0,4,8. Convert the frame mask to fully observed latent blocks for
R2 and R4. A block is clean-observed only if every valid source frame is observed. Reject any
partial block with an actionable error.

Noise and loss apply only to unobserved valid tokens. Observed state/detail fields remain clean and
are clamped after every Euler update. Invalid elements remain zero. Mutating future coordinates
after the observation condition has been built must not change that condition.

Do not implement H=2 in this phase. R4's compressed detail bank mixes three four-frame Haar detail
coefficients, so exposing a partially observed block would be ambiguous and may leak future input.

### 6.4 Molecular DiT

Implement one factorized scalar/vector DiT with:

1. atom-to-block pooling;
2. per-sample block/residue spatial attention at each latent time;
3. block-to-atom broadcast;
4. bidirectional per-atom temporal attention over K;
5. scalar/vector FFN;
6. AdaLN-Zero modulation;
7. four scalar/vector flow-output fields.

Queries and keys are invariant scalar features. Scalar attention weights may mix scalar and vector
values. Every vector channel transform is bias-free and axis-preserving. Dense attention over all
`K*N` atom-time tokens or all atoms is prohibited. The initial dense block-level backend is
permitted but must be named in the model contract and report its complexity.

AdaLN may apply scalar shifts. Vector conditioning is scale/gate only and must not create an
additive vector from invariant conditioning.

Required conditions:

```text
flow time tau
physical block timestamps and span
ratio id
observed/unobserved flag
atom type
block type
component id
```

Temporal denoising attention is bidirectional. Causality comes from not supplying unknown clean
latents and from exact observation clamping, not from a causal attention triangle.

Smoke defaults:

```text
scalar width: 256
vector width: 128
depth: 4
heads: 8
FFN multiplier: 4
dropout: 0
```

Both candidates use the identical size policy. Do not tune R2 and R4 separately.

### 6.5 Sampling

Implement deterministic normalized-latent ODE generation with fixed Euler solvers at 8 and 16
steps. At each step, restore observed latents exactly and zero invalid values. Inverse-normalize,
construct the codec latent without raw detail, decode, and record all seeds and hashes. Do not add
an SDE, CFG, guidance, rejection, or energy refinement.

## 7. Trainer and checkpoint requirements

The trainer must:

- force frozen codec and frame encoder into eval mode;
- fail if a codec parameter is trainable;
- apply field masks before reduction;
- log total and per-field RF losses;
- record grad norm, learning rate, tau distribution, observation mixture, and valid elements;
- support deterministic seed and checkpoint resume;
- serialize model config, ratio, adapter/stats/codec/data hashes, optimizer, scheduler, step, and
  RNG state;
- refuse a checkpoint whose ratio, codec, statistics, data, or model contract differs.

Use a new output root such as:

```text
outputs/dit_state_detail_probe_v1/
```

Never write under the running T1 output root.

## 8. Evaluation boundary

Implement report plumbing for the later pilot, but do not claim scientific results from smoke.
The evaluator must separate:

```text
codec oracle:       decode(E(x)) versus x
generated result:   decode(z_generated) versus x
generation gap:     generated metric minus codec-oracle metric
```

Reuse the corrected codec evaluator for aligned RMSD, raw RMSD, dRMSD, bonds, contacts, clashes,
velocity/acceleration, dynamic correlation, RMSF, frequency, and boundary metrics. Add latent
field loss and DiT trunk-only runtime. Do not use best-of-N RMSD as the primary result.

## 9. Mandatory tests before CUDA

Write focused tests for every item in ACCEPTANCE, especially:

- exact R2/R4 shapes and K=8/K=4;
- no state/detail token doubling;
- raw detail and target-coordinate exclusion;
- normalization and inverse normalization;
- arbitrary-rank rectified-flow equations;
- equal four-field loss weighting;
- ragged samples and blocks;
- physical-time conditioning;
- H=0/4/8 conversion and partial-block rejection;
- future mutation isolation and observed clamping;
- scalar invariance, vector equivariance, and same-noise rotation fixture;
- no learned xyz mixing;
- deterministic seeded 8/16-step sampling;
- checkpoint/config/statistics/codec contract round-trip;
- generated-latent decoding with `raw_detail_h/v=None`.

Then run the relevant legacy/state-detail/T1 tests and full suite. Run `py_compile`, source audit,
and `git diff --check`. All Python commands must be inside `enter-container` with `torch-ito`.

## 10. Bounded CUDA smoke only

After CPU tests pass, audit GPU processes. Use only an idle device. Do not kill, suspend, migrate,
or preempt anything. If no GPU is safely idle, record `CUDA_SMOKE_NOT_RUN` and stop cleanly.

For each candidate, the maximum authorized execution is:

```text
optimizer steps: <=100
wall time: <=15 minutes
data: one bounded real batch/subset; never complete T1 train
test split: forbidden
```

The smoke must record:

- exact data/sample ids and provenance;
- codec and statistics hashes;
- initial/final total and four field losses;
- finite/nonzero gradients in every intended module;
- unchanged frozen codec/frame-encoder hashes;
- checkpoint resume for one additional step;
- 8-step and 16-step finite decode;
- trunk tokens/s, end-to-end samples/s, parameters, and peak allocated/reserved memory.

Smoke results are execution evidence only. Do not rank R2 versus R4 from 100 steps.

## 11. Review packet and stop

Complete all task/evidence records. `OPERATOR_REVIEW.md` must state exact code state, test commands,
scope audit, R2/R4 shapes, smoke results, failures, warnings, and untested scientific claims.

End with:

```text
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
scientific DiT pilot: NOT_STARTED
T1 test accessed: NO
```

Then stop. Do not continue merely because T1 finishes, checkpoints appear, GPUs become idle, or
the bounded smoke looks good. A later operator instruction must authorize the real matched R2/R4
DiT pilot.
