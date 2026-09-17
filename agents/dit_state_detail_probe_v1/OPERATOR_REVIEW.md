# OPERATOR REVIEW — R2/R4 state-detail latent DiT probe v1

Status: DRAFT

This packet is for implementation and bounded-smoke review. It must not claim a scientific codec
winner because the real R2/R4 DiT pilot is outside the current authorization.

## 1. Exact source state

```text
branch:
base commit:
review commit or worktree hash:
git status:
diff summary:
```

## 2. Scope audit

```text
active T1 checkout modified: NO/YES
T1 outputs modified: NO/YES
T1 test opened: NO/YES
real DiT pilot started: NO/YES
longest CUDA smoke steps/time:
```

## 3. Architecture contract

Report:

- R2/R4 exact input/output shapes;
- packed state/detail and scalar/vector widths;
- spatial and temporal interaction paths;
- flow-time and physical-time conditioning;
- observation-mask/clamping semantics;
- static topology fields and prohibited fields;
- model, codec, stats, and data contract hashes.

## 4. Correctness results

Include exact test commands and results for:

- normalization and RF equations;
- observation/future isolation;
- SE(3) and isotropic vector noise;
- ragged sample/block isolation;
- checkpoint/config/statistics round-trip;
- existing codec/T1 regression suite.

## 5. Bounded smoke results

| Metric | R2 | R4 |
|---|---:|---:|
| Steps / wall time | | |
| Initial/final total RF loss | | |
| state_h/detail_h/state_v/detail_v loss | | |
| Parameters | | |
| Trunk tokens/s | | |
| End-to-end samples/s | | |
| Peak allocated/reserved GiB | | |
| Frozen codec unchanged | | |
| Resume | | |
| 8/16-step decode finite | | |

## 6. Known limitations

At minimum state whether:

- T1 codec checkpoints were unavailable or provisional;
- latent statistics came from synthetic/T0 rather than approved T1 train data;
- smoke loss is not a generated-trajectory quality comparison;
- block-level dense attention remains a future scaling constraint;
- H=2 and partial-block observation remain unsupported.

## 7. Decision

Leave exactly one proposed status:

```text
APPROVE_PILOT
REQUEST_FIX
REJECT_DESIGN
```

Proposed status: PENDING

Operator notes:

```text
pending review
```

## Observed review packet — 2026-09-07

Exact source: feat/dit-state-detail-probe-v1 at HEAD 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541.
The audited T1 source checkout stayed read-only and its pre-existing dirty files were preserved.

### Observed implementation state

The target branch is uncommitted and contains the focused adapter, rectified-flow objective,
factorized molecular DiT, trainer/checkpoint protocol, evaluation report plumbing, configuration,
smoke runner, and focused tests. No generated checkpoint is intended for commit. The source T1
checkout, T1 scripts, codec implementation, codec checkpoints, and T1 output root were not
modified.

The shared model contract is:

- R2-SD: K=8, state_h/detail_h [8,N,128], state_v/detail_v [8,N,3,128], packed scalar
  [8,N,256] and vector [8,N,3,256].
- R4-SD: K=4, state_h/detail_h [4,N,128], state_v/detail_v [4,N,3,128], packed scalar
  [4,N,256] and vector [4,N,3,256].
- State and detail share one sequence location; neither candidate doubles the sequence length.
- The smoke model uses scalar width 256, vector width 128, depth 4, 8 heads, FFN multiplier 4,
  and dropout 0. The backend is factorized block spatial attention plus bidirectional per-atom
  temporal attention, with complexity sum_s K*M_s^2 + sum_n K^2.
- Conditions include flow time, physical block time/span, ratio, observation flag, atom/block/
  component IDs, and coordinate-derived frame-0 origin only for H>0. Raw detail, target
  coordinates, future-derived metadata, dense K*N attention, and learned xyz mixing are excluded.
- H=0, H=4, and H=8 are implemented. Partial blocks and H=2 are rejected.

### Verification evidence

Commands were run inside enter-container with the torch-ito environment:

- Baseline before implementation: 115 passed, one pre-existing torch.load FutureWarning.
- Focused DiT tests: 25 passed.
- Legacy/state-detail/evaluator regression selection: 45 passed.
- Final full suite: 140 passed, one pre-existing torch.load FutureWarning.
- py_compile over all new Python files passed.
- git diff --check passed.

The focused tests cover R2/R4 packing and exact K values, no token doubling, raw-detail exclusion,
mask-aware normalization and inverse normalization, arbitrary-rank flow equations, equal four-field
loss weighting, ragged inputs, physical-time conditioning, H-prefix conversion and partial-block
rejection, future-mutation isolation, observed clamping, scalar invariance, vector equivariance
including a same-noise rotation fixture, axis-preserving projections, deterministic 8/16-step
sampling, generated-latent decoding with raw detail absent, and checkpoint contract rejection.

### Bounded CUDA smoke

GPU 7 was idle before both launches; no running process was killed, suspended, migrated, or
preempted. Both runs used the immutable audited-source T0-valid payload, records 2, H=4, and the
same model-size policy. The payload is not a T1 test split.

| candidate | K | steps | initial total | final total | parameters | 8/16 decode | report |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ratio2_state_detail | 8 | 1 plus resume to 2 | 3.351095 | 3.351095 | 8,679,680 | finite/finite | outputs/dit_state_detail_probe_v1/ratio2_state_detail/smoke_report.json |
| ratio4_state_detail | 4 | 2 plus resume check | 3.480752 | 3.151710 | 8,679,680 | finite/finite | outputs/dit_state_detail_probe_v1/ratio4_state_detail/smoke_report.json |

The R2 report records the first one-step training result and a successful one-step checkpoint
resume to step 2; its final-loss entry is therefore the recorded step-1 value, not an unrecorded
post-resume loss. The R4 report records two training steps and a successful resume check. This is
an execution-record caveat, not a changed acceptance threshold.

R2 field losses (state_h, detail_h, state_v, detail_v) were
3.964561, 3.873134, 2.842202, 2.724482. R4 field losses were initially
4.173809, 4.475360, 2.626467, 2.647372 and finally
3.633723, 3.836304, 2.461705, 2.675109. Gradients were finite and nonzero in every intended
trainable group; frozen codec and frame-encoder hashes were unchanged; checkpoint resume and
finite 8/16-step decoding succeeded; generated latents had raw detail absent and observed values
were clamped.

R2 measured trunk throughput was 13,775.39 tokens/s and end-to-end throughput 0.09234 samples/s;
peak allocated/reserved memory was 4.54/10.12 GiB. R4 measured 14,339.10 tokens/s and
0.19218 samples/s; peak allocated/reserved memory was 2.37/10.12 GiB. These are bounded smoke
measurements only and do not rank R2 versus R4.

The reports contain exact sample IDs, topology IDs, codec/statistics/adapter/model hashes, losses,
gradients, resume evidence, memory, throughput, decode finiteness, and provenance. The R2 report
topology IDs were corrected in the report to the exact source-batch IDs after the first successful
run; the runner was then corrected before the R4 run so it records them directly.

### Scope audit and limitations

No approved trained T1 codec checkpoint was available in the target worktree, so the smoke used
randomly initialized torchmd_et codec instances only to exercise interfaces and decoding. Latent
statistics are explicitly labeled T0 smoke statistics and are not production train-only
statistics. No scientific generated-trajectory comparison, codec selection, ratio selection,
test access, real T1 pilot, best-of-N result, long rollout, H=2, energy, ensemble, static mixing,
AF3/MSA conditioning, VAE/KL/VQ, scaling study, or later architecture work was run. Evaluation
plumbing separates codec oracle, generated result, and generation gap and includes latent loss and
runtime fields, but no scientific evaluation claim is made from this smoke.

The initial smoke attempts exposed and repaired only execution issues: direct-script import path,
missing target-worktree T0 payload, topology-cache preparation, and CPU/CUDA RNG restoration.
One host-shell Python inspection was attempted and failed immediately because host Python had no
torch; it read no project data and caused no mutation. All substantive Python work remained inside
enter-container.

Proposed status: PENDING

Implementation gate: PASS.
CPU correctness gate: PASS.
Bounded CUDA execution gate: PASS WITH THE RECORDED R2 REPORT CAVEAT.
Scientific pilot gate: NOT RUN; operator authorization is still required.

phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
scientific DiT pilot: NOT_STARTED
T1 test accessed: NO
## Review-fix record — 2026-09-07

### Exact code state

Reviewed starting commit: `14f0f422a651126eec42d525dae55dbefb7bdd38`.
Branch: `feat/dit-state-detail-probe-v1`.
Base codec commit: `23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541`.
Final repair commit: the focused commit created from this review-fix record; the exact hash is
printed by the post-commit `git rev-parse HEAD` handoff. The original packet above is preserved;
only claims invalidated by the review are superseded here.

Files changed by the repair are `module/molecular_dit.py`,
`module/state_detail_latent_adapter.py`, `module/latent_rectified_flow.py`, `module/__init__.py`,
`trainer/dit_trainer.py`, `evaluation/codec_evaluation.py`, `evaluation/dit_evaluation.py`,
`scripts/run_state_detail_dit_smoke.py`, `tests/dit_test_utils.py`,
`tests/test_molecular_dit.py`, `tests/test_state_detail_latent_adapter.py`,
`tests/test_dit_trainer.py`, `tests/test_dit_evaluation.py`, and
`tests/test_dit_cuda_correctness.py`, plus appended phase records.

### Three corrected scientific-contract defects

- SO(3): the repaired model uses xyz-contracted equivariant vector normalization, no
  component-wise vector activation, bias-free axis-preserving vector FFN maps, invariant vector
  norms into scalar FFN features, scalar gates into vector channels, and explicit nonzero-gate
  full-model tests. The previous rotation test could pass only because AdaLN-Zero gates were all
  zero; the new fixture warms the complete block and checks scalar invariance, vector rotation,
  same-noise flow equivariance, exact attention/FFN gradient groups, and negative component-wise
  LayerNorm/SiLU fixtures.
- Origin: H>0 now uses the frozen codec's loss-masked frame-0 centroid and H=0 is exactly zero.
  The tests include a loss-masked-out extreme atom, future-coordinate mutation, H=0, and
  clamped decode in the codec gauge. The old all-atom average could create a translation error.
- Evaluation history: conditional reports now pass explicit H and use observed `[0,H)`, future
  `[H,T)`, boundary `[H-1,H)`, and full diagnostic `[0,T)` intervals. Future spatial and sliced
  RMSF/dynamic metrics cannot be improved by changing observed frames; H=4 and H=8 are distinct;
  H=0 has no invented boundary. The previous hard-coded `1..T-1` future section was superseded.

### Gate evidence

Focused repaired tests passed: `39 passed in 18.95s`. The complete suite passed with
`154 passed, 1 skipped, 1 warning in 15.40s`; the only skip is the explicit CUDA guard when its
environment variable is absent, and the warning is the pre-existing codec `torch.load` future
warning. `py_compile` over all changed Python files and `git diff --check` passed. The final source
audit found no prohibited target-coordinate/raw-detail/coordinate-graph inputs, learned xyz
mixing, or unintended CPU conversions.

The explicit CUDA test on idle GPU 7 passed: `1 passed in 17.41s`. It covered real CUDA FP32 and
BF16 autocast, nonzero-gate rotation/backward, all latent fields/masks/topology/origin, six
statistics tensors after fresh checkpoint recovery, deterministic 8/16-step sampling, H=4/H=8
evaluation, and finite/nonzero gradients in every intended group. GPU 7 was idle before and after;
no other process was interrupted.

Final two-step smokes used the new output root
`outputs/dit_state_detail_probe_v1/review_fix_260907`, serially on GPU 7, with H=4, CUDA BF16,
fresh statistics recovery, exact observed clamping, unchanged frozen hashes, and 24 model
evaluations for combined 8/16-step sampling. R2 wall/train/sampling values were
`43.6597255s / 5.3817530s per step / 18.6488833s`; R4 values were
`29.7499437s / 2.6369080s per step / 9.8733035s`. Train and sampling throughput were reported
separately, with synchronized CUDA timing. Reports contain exact IDs and hashes and declare
observed `[0,4)`, future `[4,16)`, boundary `[3,4)`, and full diagnostic `[0,16)` intervals.
The target-relative T0 payload was missing `data.bin`, so the already-recorded immutable source
T0-valid payload was used read-only; this is execution evidence only. No T1 test was accessed.

### Warnings and untested scientific claims

The reports are not production statistics, not T1 codec results, and do not rank R2 versus R4.
No scientific pilot, approved T1 checkpoint load, train/validation pilot selection, or test
evaluation was performed. The production pilot runner and real-data schedule remain future work.
H=2/partial blocks, long rollout, DDP, static mixing, AF3/MSA, VAE/KL/VQ, CFG, energy guidance,
ensemble control, and later architecture work remain out of scope.

implementation repair gate: PASS
nonzero-gate SO(3) gate: PASS
history-aware evaluation gate: PASS
bounded CUDA gate: PASS
scientific R2/R4 pilot: NOT_STARTED
T1 test accessed: NO
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
