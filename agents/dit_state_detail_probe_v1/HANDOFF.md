# HANDOFF — R2/R4 state-detail latent DiT probe v1

Append evidence; do not delete prior entries. Do not mark a task complete without a path, command,
or observed result.

## Initial state

```text
date:
repository:
worktree:
branch:
HEAD:
remote:
git status:
source commit comparison:
active T1 checkout/processes:
Python/PyTorch/CUDA:
visible and occupied GPUs:
```

## Required implementation record

```text
completed task IDs:
changed source files:
changed tests/configs/scripts:
model contract version/hash:
R2 codec checkpoint/hash used for smoke:
R4 codec checkpoint/hash used for smoke:
latent-stat source/hash:
data manifest/hash:
```

## Required commands

Record exact commands, including entry into `enter-container`, activation of `torch-ito`, working
directory, CUDA mapping, configuration, seed, output root, and exit status.

```text
preflight:
focused tests:
full tests:
compile/source checks:
R2 smoke:
R4 smoke:
checkpoint resume:
report generation:
```

## Required evidence table

| Evidence | R2 | R4 |
|---|---|---|
| K and packed shapes | | |
| total/trainable parameters | | |
| initial/final smoke loss | | |
| per-field RF losses | | |
| finite/nonzero gradients | | |
| frozen codec hash before/after | | |
| resume step | | |
| trunk tokens/s | | |
| end-to-end samples/s | | |
| peak allocated/reserved GiB | | |
| 8/16-step decode finite | | |
| observed clamp exact | | |

## Deviations and failures

For every deviation, failed attempt, warning, or unavailable dependency, record:

```text
task ID:
observed fact:
impact:
decision:
artifact/log path:
whether acceptance changed: no/yes (yes requires operator approval)
```

Do not weaken thresholds or silently replace a failed backend.

## Final handoff

```text
implementation gate: PASS / FAIL / BLOCKED
CPU correctness gate: PASS / FAIL / BLOCKED
bounded CUDA gate: PASS / FAIL / NOT_RUN
scientific R2/R4 comparison: NOT_RUN
T1 test accessed: NO
## Review-fix continuation — 2026-09-07

### Code state and scope

The repair started from reviewed commit
`14f0f422a651126eec42d525dae55dbefb7bdd38` on `feat/dit-state-detail-probe-v1`, with base
codec commit `23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541`. The final repair is the focused commit
created from this record; its exact hash is emitted by post-commit verification. The active T1
checkout, processes, outputs, test split, and codec implementation remained read-only.

Changed implementation and test files:

- `module/molecular_dit.py`, `module/state_detail_latent_adapter.py`,
  `module/latent_rectified_flow.py`, `module/__init__.py`
- `trainer/dit_trainer.py`
- `evaluation/codec_evaluation.py`, `evaluation/dit_evaluation.py`
- `scripts/run_state_detail_dit_smoke.py`
- `tests/dit_test_utils.py`, `tests/test_molecular_dit.py`,
  `tests/test_state_detail_latent_adapter.py`, `tests/test_dit_trainer.py`,
  `tests/test_dit_evaluation.py`, `tests/test_dit_cuda_correctness.py`
- this phase's `DECISIONS.md`, `TASKS.md`, `HANDOFF.md`, and `OPERATOR_REVIEW.md`

The original implementation packet and its evidence remain above unchanged. This section
supersedes only old equivariance, origin, evaluator-history, AMP, statistics-persistence, and
smoke-bookkeeping claims that the review invalidated.

### Corrected contracts

1. Vector normalization now contracts xyz into FP32 norms and the vector FFN uses only
   axis-preserving bias-free maps with invariant scalar gates. Vector norms inform the scalar FFN,
   scalars gate vector channels, and scalar q/k attention weights remain shared for scalar/vector
   values. Nonzero spatial, temporal, and FFN AdaLN gates are exercised by the tests.
2. H>0 observation origins use the frozen codec's `compute_masked_centroid_origin` over the
   frame-0 `loss_mask`; H=0 uses an exact zero origin. Future coordinates and masked-out extreme
   atoms cannot alter the condition, and observed decoded blocks remain in the codec gauge.
3. Conditional evaluation explicitly reports observed `[0,H)`, future `[H,T)`, boundary
   `[H-1,H)`, and full diagnostic `[0,T)` intervals. Future spatial and temporal metrics use
   sliced trajectories, masks, and times; H=0 has no fabricated boundary.

### Verification commands and results

All successful Python commands below ran through `enter-container` with `conda activate torch-ito`.

Focused repaired suite:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py tests/test_molecular_dit.py tests/test_dit_trainer.py tests/test_dit_evaluation.py
```

Result after the final source edit and after smoke generation: `39 passed in 18.95s`.

Complete suite:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider
```

Result: `154 passed, 1 skipped, 1 warning in 15.40s`. The sole skip is the explicit CUDA-only
test guard when `DIT_RUN_CUDA_CORRECTNESS` is unset; the pre-existing warning is the codec
checkpoint test's `torch.load(weights_only=False)` FutureWarning. The CUDA test was run explicitly
and passed, so this skip is not treated as CUDA evidence.

Static checks passed: `py_compile` over all changed Python files, `git diff --check`, and a
second compile of the two new evaluator/CUDA test files. The source audit found no
target-coordinate, raw-Haar-detail, coordinate-derived graph, learned xyz-mixing, or unintended
`.cpu()` path in the DiT input/trunk. CPU conversion remains limited to hashing, checkpoint
serialization, and final scalar/JSON reporting.

Explicit CUDA correctness on audited idle GPU 7:

```text
CUDA_VISIBLE_DEVICES=7 DIT_RUN_CUDA_CORRECTNESS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider -rs tests/test_dit_cuda_correctness.py
```

Result: `1 passed in 17.41s`. It exercised CUDA FP32 forward/backward, nonzero-gate rotation and
same-noise fixtures, masks/topology/origin/all fields, real BF16 autocast with activation dtype
evidence, fresh checkpoint/statistics reconstruction, deterministic 8/16-step sampling, H=4/H=8
evaluation, and nonzero finite gradients in all intended groups. GPU 7 was idle before and after;
other active GPUs were not touched.

### Final repaired T0 smoke evidence

Exactly one two-optimizer-step smoke was run serially per candidate on GPU 7, with H=4 and the
immutable T0-valid payload
`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid`.
The target-worktree-relative payload was first checked and failed before model execution because
its `data.bin` was absent; the existing source-worktree T0 payload was used read-only as already
authorized by D017. No T1 data or test split was opened. These are execution evidence only, not
scientific comparisons.

| field | ratio2_state_detail | ratio4_state_detail |
|---|---:|---:|
| report | `outputs/dit_state_detail_probe_v1/review_fix_260907/ratio2_state_detail/smoke_report.json` | `outputs/dit_state_detail_probe_v1/review_fix_260907/ratio4_state_detail/smoke_report.json` |
| wall seconds | 43.6597255 | 29.7499437 |
| optimizer steps | 2 (including resume) | 2 (including resume) |
| train step mean seconds | 5.3817530 | 2.6369080 |
| train steps/s | 0.1858131 | 0.3792320 |
| sampling seconds | 18.6488833 | 9.8733035 |
| sampling samples/s | 0.1072450 | 0.2025664 |
| combined sampling model evaluations | 24 | 24 |
| trunk tokens/s | 18264.2572 | 17248.9380 |
| end-to-end samples/s | 0.0458088 | 0.0672270 |
| parameters | 11173120 | 11173120 |
| peak allocated/reserved bytes | 3791333376 / 10856955904 | 2032064000 / 10856955904 |
| autocast | CUDA BF16 | CUDA BF16 |
| observed clamping | exact | exact |
| 8/16-step decode | finite / finite | finite / finite |

Both reports record exact sample IDs `atlas_5e3e_A_R1_w000049` and
`atlas_5e3e_A_R1_w000050`, the same topology ID
`5e3e_A::75184f3b9f0f379e56ff670b2f368df8ac39ee225ec62a1d8731a70f5dec1b80`, fresh six-tensor
statistics recovery, unchanged frozen codec/frame-encoder hashes, and finite/nonzero gradients
for spatial attention, temporal attention, scalar/vector FFN, AdaLN, adapter, and all four output
heads. Both report H=4 intervals as observed `[0,4)`, future `[4,16)`, boundary `[3,4)`, and full
diagnostic `[0,16)`.

R2 hashes: codec
`6d0b242a2cf7b4bf1b92c443fbde65dc43e6086f107bbe125920cfba64758d98`, frame
`35b0cc04f1f71d12870ed6b3c27b6dd94cd1ffb29364cfc90df82293465c32aa`, statistics
`2dfe02c0f8db7ead00441fd7d5ee311a8b2fad9d52cd7724fd27c34ec4e19bce`, adapter
`3570a83ac0ca08d10af875ea7dc341d1e6ccee5783d78df6cce5e1f502a148dd`, model
`c0621a4a8732d7c2f3f812e23b008b9d09216642a98eaf5b10d6cb9013cac44c`.

R4 hashes: codec
`e6706b3d38daa221a7e9458760999ff9d652c3bd5647f886d2cf89cfab868d54`, frame
`8cd27004619765d4f2c3e6b954d80e7dcdfafa19246757dcea076408bbba4537`, statistics
`a11f149d076e4e38082798c3887d95394a39f2033b75251a8472c10906147a5a`, adapter
`0a63d96d401a721e50825d4272294e15fb8c125d901b00da589b8f44403b0970`, model
`5a252516da5eab27569b542ba04f4c1dedf86a7a65e59fd738bb07603a964918`.

The first/final total losses were R2 `3.4083285` / `3.0662241` and R4 `3.5162063` /
`3.1409760`. Per-field initial/final losses were R2
`(4.3647718, 4.0679617, 2.5738816, 2.6266994)` /
`(3.7195108, 3.5214915, 2.4415779, 2.5823169)` and R4
`(4.2918005, 4.2610207, 2.8383884, 2.6736155)` /
`(3.6766608, 3.6511056, 2.5582311, 2.6779058)` in
`(state_h, detail_h, state_v, detail_v)` order. These values are not a ratio ranking.

### Remaining limitations and stop

The smoke uses random T0 codec instances and T0-derived statistics solely for execution. There is
no approved T1 codec checkpoint/statistics artifact, production pilot runner, train/validation
selection, scientific R2/R4 result, or test evaluation. H=2 and partial-block observation remain
unsupported. Dense block-level attention is still the declared initial backend. No full-T1
training, T1 test access, DDP, static mixing, AF3/MSA, VAE/KL/VQ, CFG, energy guidance, ensemble,
long rollout, or later architecture phase was started.

implementation repair gate: PASS
nonzero-gate SO(3) gate: PASS
history-aware evaluation gate: PASS
bounded CUDA gate: PASS
scientific R2/R4 pilot: NOT_STARTED
T1 test accessed: NO
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
real DiT pilot started: NO
phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
operator decision requested: APPROVE_PILOT / REQUEST_FIX / REJECT_DESIGN
```

## Observed execution record — 2026-09-07

Repository isolation:

- Target: /data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1, branch
  feat/dit-state-detail-probe-v1, HEAD 23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541.
- Audited source: /data4/users/sihao/workspace/PVB, branch fix/state-detail-codec-v2-t1-gates,
  same HEAD. Its pre-existing dirty files remained untouched:
  scripts/report_state_detail_codec_v2_t1.py and the two untracked T1 packet scripts.
- Remote: https://github.com/SHw-1313/molvid.git. The source and target worktrees stayed
  separate. No Git reset, checkout-over, clean, rebase, push, or commit was used.
- GPU 7 was idle before each successful CUDA launch; it was not killed, suspended, migrated, or
  preempted. Final audit still showed GPU 7 at 0 MiB and 0 percent utilization.

Completed evidence-backed tasks: D000-D004, D010-D064. D100-D104 remain blocked pending later
operator authorization. Source files added or changed:

- module/state_detail_latent_adapter.py
- module/latent_rectified_flow.py
- module/molecular_dit.py
- trainer/dit_trainer.py
- evaluation/dit_evaluation.py
- config/dit_state_detail_probe.yaml
- scripts/run_state_detail_dit_smoke.py
- package exports in module/__init__.py, trainer/__init__.py, evaluation/__init__.py
- focused tests in tests/test_state_detail_latent_adapter.py, tests/test_latent_rectified_flow.py,
  tests/test_molecular_dit.py, tests/test_dit_trainer.py, and tests/dit_test_utils.py

Contract record:

- R2: K=8, state/detail scalar [8,N,128], vector [8,N,3,128], packed scalar [8,N,256],
  packed vector [8,N,3,256].
- R4: K=4 with the same C=128 and packed channel policy.
- Smoke model policy was identical for both ratios: Dh=256, Dv=128, depth=4, heads=8,
  FFN multiplier=4, dropout=0. Parameter count was 8,679,680 for both.
- Model backend contract: dense block-level spatial attention plus bidirectional per-atom
  temporal attention; reported complexity is sum_s K*M_s^2 + sum_n K^2. No all-atom/time dense
  attention, coordinate metadata, radius graph, distance, contact, or raw Haar input enters the
  DiT batch.

Commands and results, all successful Python commands run after entering enter-container and
activating torch-ito from the target directory:

- Baseline before implementation: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
  -> 115 passed, 1 pre-existing FutureWarning.
- Focused final DiT suite: the four new test files -> 25 passed.
- Legacy/state-detail/evaluator regression selection:
  tests/test_state_detail_codec_v2.py tests/test_codec_training.py
  tests/test_codec_evaluation.py tests/test_luna_review_contracts.py -> 45 passed.
- Final full suite: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
  -> 140 passed, 1 pre-existing torch.load FutureWarning.
- Final compile: python -m py_compile module/state_detail_latent_adapter.py
  module/latent_rectified_flow.py module/molecular_dit.py trainer/dit_trainer.py
  evaluation/dit_evaluation.py scripts/run_state_detail_dit_smoke.py tests/dit_test_utils.py
  tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py
  tests/test_molecular_dit.py tests/test_dit_trainer.py -> passed.
- Source audit covered forbidden raw-detail/target-coordinate/radius-graph paths and dense
  attention; git diff --check -> passed.

Smoke commands:

    CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 python scripts/run_state_detail_dit_smoke.py --candidate ratio2_state_detail --device cuda --max-steps 1 --records 2 --data-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid --output-root outputs/dit_state_detail_probe_v1

    CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 python scripts/run_state_detail_dit_smoke.py --candidate ratio4_state_detail --device cuda --max-steps 2 --records 2 --data-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t0_data/clip_store/valid --output-root outputs/dit_state_detail_probe_v1

The target worktree had the T0 manifest/index but not data.bin. Per D017, both runs read the
immutable T0-valid data.bin in the audited source worktree through target code; no T1 test split
was opened. The two sample IDs were atlas_5e3e_A_R1_w000049 and atlas_5e3e_A_R1_w000050, with
topology ID 5e3e_A::75184f3b9f0f379e56ff670b2f368df8ac39ee225ec62a1d8731a70f5dec1b80.

Smoke evidence:

| Evidence | R2 | R4 |
|---|---:|---:|
| K / packed shapes | 8 / [8,N,256], [8,N,3,256] | 4 / [4,N,256], [4,N,3,256] |
| steps / report wall time | 1 logged + resume to 2 / 21.660 s | 2 total / 10.407 s |
| initial total loss | 3.351095 | 3.480752 |
| final total loss | 3.351095 (report generated before bookkeeping fix; step-2 resume was finite) | 3.151710 |
| state_h / detail_h / state_v / detail_v initial | 3.964561 / 3.873134 / 2.842202 / 2.724482 | 4.173808 / 4.475360 / 2.626466 / 2.647372 |
| state_h / detail_h / state_v / detail_v final | same logged step-1 values; see caveat above | 3.633723 / 3.836304 / 2.461705 / 2.675109 |
| parameters | 8,679,680 | 8,679,680 |
| trunk tokens/s | 13,775.39 | 14,339.10 |
| end-to-end samples/s | 0.09234 | 0.19218 |
| peak allocated / reserved GiB | 4.544 / 10.117 | 2.368 / 10.117 |
| gradients | finite and nonzero in adapter, flow time, all embeddings, and blocks | finite and nonzero in adapter, flow time, all embeddings, and blocks |
| frozen codec / frame encoder | unchanged / unchanged | unchanged / unchanged |
| checkpoint resume | step 1 to step 2 | step 1 to step 2 |
| 8/16-step decode | finite / finite; raw detail absent | finite / finite; raw detail absent |
| observed clamp | exact | exact |

Artifacts are under outputs/dit_state_detail_probe_v1/ratio2_state_detail and
outputs/dit_state_detail_probe_v1/ratio4_state_detail. Generated checkpoints are intentionally
uncommitted. Codec hashes, statistics hashes, adapter hashes, model hashes, exact losses, and
runtime values are in each smoke_report.json.

Deviations and repairs:

1. The first target-launched smoke stopped before data access because direct script execution did
   not put the repository root on sys.path. The runner now resolves its own root.
2. The next attempt reached the frozen encoder and correctly failed because the topology cache had
   not been registered. The runner now calls the existing CPU prepare_batch API before CUDA.
3. Two checkpoint-resume attempts exposed CPU and CUDA RNG-device restoration issues. The trainer
   now restores CPU RNG as CPU bytes and each CUDA state explicitly; the successful R2/R4 runs
   verified resume.
4. The successful R2 report predates the final smoke bookkeeping change, so its final loss field
   is explicitly classified as the logged step-1 value, while its step-2 resume and finite decode
   evidence remain recorded. R4 uses the corrected max-step accounting.
5. One diagnostic host-shell Python invocation was attempted outside enter-container and failed
   immediately with ModuleNotFoundError before importing torch or touching project data. All
   successful Python, test, compile, smoke, and evaluation commands ran inside enter-container
   with torch-ito. Acceptance criteria were not changed.

Final gate classification:

- implementation gate: PASS
- CPU correctness gate: PASS
- bounded CUDA gate: PASS WITH THE RECORDED R2 REPORT CAVEAT
- scientific R2/R4 comparison: NOT RUN
- real DiT pilot: NOT STARTED
- T1 test accessed: NO
- phase status: WAITING_FOR_T1_AND_OPERATOR_REVIEW
```
