# OPERATOR REVIEW — state/detail codec v2 T0

Status: WAITING_FOR_OPERATOR_REVIEW

The implementation gates and the bounded four-control T0 are complete. This packet is awaiting
operator review; no T1 work is authorized by this document.

## 1. Exact code state

```text
branch: feat/state-detail-codec-v2
base commit: 045dbb8809e9d7aee418eb355b6477e05088d4fd
review commit or working-tree hash: HEAD=045dbb8809e9d7aee418eb355b6477e05088d4fd; no review commit; dirty worktree
git status: tracked edits in AGENTS.md, config/codec.yaml, module/__init__.py, trainer/codec_trainer.py;
  untracked prompt, state/detail phase files, new module/tests/scripts, and T0 outputs
diff summary: tracked diff is 334 insertions / 98 deletions across 4 files; generated artifacts are untracked
```

Changed source files:

- `AGENTS.md`
- `config/codec.yaml`
- `module/__init__.py`
- `module/state_detail_codec_v2.py`
- `trainer/codec_trainer.py`

Changed tests/scripts/configs and phase records:

- `tests/test_state_detail_codec_v2.py`
- `scripts/pack_t0_clip_store.py`
- `scripts/run_state_detail_codec_v2_t0.py`
- `agents/state_detail_codec_v2/TASKS.md`
- `agents/state_detail_codec_v2/HANDOFF.md`
- `agents/state_detail_codec_v2/OPERATOR_REVIEW.md`

## 2. Commands and environment

```text
test/compile command: enter-container; conda activate torch-ito; cd /workspace/PVB
conda environment: torch-ito
Python: 3.11.15
torch/cuda/torch_cluster: torch 2.5.1+cu121 / CUDA 12.1 / torch_cluster 1.6.3+pt25cu121
GPU mapping: `neibu`, CUDA_VISIBLE_DEVICES=0, NVIDIA A100-SXM4-80GB (80 GiB)
```

Final T0 command, run in the isolated `neibu` checkout and copied back to this checkout:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_state_detail_codec_v2_t0.py --mode all \
  --store-root outputs/state_detail_codec_v2/t0_data/clip_store \
  --output-root outputs/state_detail_codec_v2/t0_remote_final
```

Validation commands/results:

```text
PYTHONPATH=. python -m pytest -q -> 107 passed, 1 pre-existing FutureWarning, 12.81 s
python -m py_compile module/state_detail_codec_v2.py module/__init__.py trainer/codec_trainer.py
  scripts/pack_t0_clip_store.py scripts/run_state_detail_codec_v2_t0.py
  tests/test_state_detail_codec_v2.py -> passed
git diff --check -> passed
```

## 3. Architecture checks

| Check | Evidence path | Result |
|---|---|---|
| R1/R2/R4 lifting and inverse mapping | `tests/test_state_detail_codec_v2.py`; S210/S240 | Passed in strict FP32 at `rtol=1e-6`, `atol=1e-7`; R4 order is `Dmid,D01,D23`. |
| latent shapes and 16C/16C/8C accounting | `module/state_detail_codec_v2.py`; S211/S220-S225 | Passed: C=128 gives R1=2048 active elements, R2=2048, R4-SD=1024. |
| R4 matched-pooling 8C accounting | `module/state_detail_codec_v2.py`; S230-S231 | Passed: two independent C-wide banks, 1024 active elements, natural parameter count retained. |
| no per-atom x0/reference bypass | `tests/test_state_detail_codec_v2.py`; source audit; S212/S244 | Passed; new decoder has no target-coordinate argument and `state_detail_codec_v2.py` has no `x_anchor`. |
| only per-sample origin bypasses latent | `module/state_detail_codec_v2.py`; model contract; S212/S224 | Passed; origin is one `[B,3]` `frame0_loss_masked_centroid` vector. |
| repeated-static zero detail | `tests/test_state_detail_codec_v2.py`; S241 | Passed exactly before and after the learned detail bottleneck for R2/R4. |
| repeated-static zero coordinate motion | `tests/test_state_detail_codec_v2.py`; S241/S224 | Passed; decoded within-block feature/coordinate motion is zero under zero detail. |
| dynamic detail nonzero and receives gradients | `tests/test_state_detail_codec_v2.py`; final `micro_summary.json`; S242/S252 | Passed; all four micro runs have finite nonzero principal gradients; R2/R4 detail is utilized. |
| T=1 versus repeated-T16 semantics | `tests/test_state_detail_codec_v2.py`; S225/S240 | Passed; T=1 has `detail_valid=false`, repeated T=16 retains valid zero detail. |
| SE(3), masks, irregular clocks, isolation | `tests/test_state_detail_codec_v2.py`; S240/S243 | Passed for rotation/translation, partial blocks, invalid-frame isolation, clocks, and samples. |
| old checkpoint/legacy regression | `tests/test_state_detail_codec_v2.py`; `outputs/state_detail_codec_v2/preflight/legacy_audit.json`; S213/S245 | Passed; legacy loading retains old schema and old per-atom-anchor behavior. |

## 4. T0 data identity

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
replicates: R1, R2, R3 for each system (nine trajectories)
train clips: 441, windows w000000 through w000048
holdout clips: 117 late holdout, windows w000049 through w000061
T / dt: T=16 / native dt_100ps (100 ps)
manifest contract hash: 2dc3790c22ddb2b9606d3e4087a7204e9b91be20252dd6dc6dc7a3126550bca4
source manifest hash: 700f01e40f4e0fda697191cd161bb8161b97493a994415cee286354a9809450e
train-index hash: 04806744073a7df56af2fd5671849e1147a7c6492e956d389b6830aa136d9b0a
holdout-index hash: 5d247fcb5f21ba2780d6b93880afbc01675ae1a42a935292f04161b7ff1ea0d1
common TorchMD encoder hash: e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a
```

The manifest independently reports `intersection_count=0`, lazy loading, `replacement=false`,
`max_tokens=80000`, FP32, 30 complete epochs, and a 6000-step safety cap. There is no semantic
protocol discrepancy. The original source root is not present inside the portable copied
container (`source_roots_exist_at_audit=false`); the exact-byte compact clip store and its
provenance are present at `outputs/state_detail_codec_v2/t0_data/clip_store/`.

## 5. Training completion

| Control | 30 epochs | Best/final checkpoints | Resume | Finite | Detail/bank utilized |
|---|---|---|---|---|---|
| R1-SD (`ratio1_state_detail`) | yes; 4,976 steps | `ratio1_state_detail/codec_best.pt` (`7e98e51b...`); `codec_step_00004976.pt` (`26cfc054...`) | `4976 -> 4977`, passed | yes | state only; detail absent by contract |
| R2-SD (`ratio2_state_detail`) | yes; 4,976 steps | `ratio2_state_detail/codec_best.pt` (`0425288a...`); `codec_step_00004976.pt` (`32466181...`) | `4976 -> 4977`, passed | yes | detail valid fraction 1.0; encoded h norm 0.191008 |
| R4-SD (`ratio4_state_detail`) | yes; 4,976 steps | `ratio4_state_detail/codec_best.pt` (`51bb7f5a...`); `codec_step_00004976.pt` (`d8c381e2...`) | `4976 -> 4977`, passed | yes | detail valid fraction 1.0; encoded h norm 0.238345 |
| R4-Matched-Pooling (`ratio4_matched_pooling`) | yes; 4,976 steps | `ratio4_matched_pooling/codec_best.pt` (`ba807ff9...`); `codec_step_00004976.pt` (`4fd3b29f...`) | `4976 -> 4977`, passed | yes | bank A h norm 1.618426; bank B h norm 1.584718 |

All four runs used common frozen encoder source hash above; before/after frozen-state hash was
`c8e3fe219b4fc100774a03c74150dc8d2b8d12299d334f59102d44be2e2c377f` in the micro evidence.
The separate one-step unfreeze smoke passed with 41 nonzero spatial gradients and changed
frame-encoder state; it was not a fifth comparison run.

Loss-curve report:

```text
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curves.png/.pdf
per-control loss_curve_ratio1_state_detail.{png,pdf}, loss_curve_ratio2_state_detail.{png,pdf},
loss_curve_ratio4_state_detail.{png,pdf}, loss_curve_ratio4_matched_pooling.{png,pdf}
```

The operator should inspect the complete curves and plateau behavior; fixed step count alone is
not evidence of convergence.

## 6. Core evaluation

Aggregate report paths:

```text
JSON:     outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.json
CSV:      outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.csv
Markdown: outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.md
plots:    loss_curves, evaluation_curves, block_detail_metrics, performance_summary, and all four
          per-control loss curves, each in PNG and PDF form in the same directory
```

The following are final late-holdout future metrics. ACF is prediction followed by absolute error;
boundary is predicted jump followed by absolute error. Peak memory is allocated bytes.

| Metric | R1-SD | R2-SD | R4-SD | R4-Matched-Pooling |
|---|---:|---:|---:|---:|
| Holdout RMSD | 11.424228 | 11.241862 | 11.138619 | 10.544774 |
| Holdout dRMSD | 10.154870 | 9.900528 | 9.767301 | 9.306050 |
| Velocity RMSE | 0.0209102 | 0.0128179 | 0.00817435 | 0.0106722 |
| Acceleration RMSE | 0.000358905 | 0.000192065 | 0.000126111 | 0.000170632 |
| RMSF correlation | 0.766342 | 0.784969 | 0.782398 | 0.714334 |
| Lagged/ACF metric | 0.958019 / 0.039499 | 0.984965 / 0.012553 | 0.994066 / 0.003452 | 0.990999 / 0.006518 |
| Frequency retention | 9.382636 | 6.367882 | 4.089264 | 4.967844 |
| Boundary jump | 3.297411 / 2.378538 | 2.793389 / 1.875074 | 2.323409 / 1.404246 | 2.670770 / 1.751607 |
| Peak memory (bytes) | 25,781,457,408 | 26,321,821,696 | 26,322,873,344 | 25,788,809,728 |
| End-to-end samples/s | 15.6238 | 15.3303 | 15.1350 | 15.6286 |

Capacity and parameter accounting for C=128: R1 is 16C active elements with 643,016 total /
82,432 trainable parameters; R2 is 16C with 724,936 / 164,352; R4-SD is 8C with 856,008 /
295,424; matched pooling is 8C with 1,102,536 / 541,952. R2 does not halve total active
feature volume; it preserves 16C using state plus detail banks.

Per-system and block-offset reports:

```text
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio1_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio2_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio4_state_detail/evaluation.json
outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/ratio4_matched_pooling/evaluation.json
```

These files contain per-frame, per-block-offset, per-system, bond/contact/clash, velocity/
acceleration, RMSF, ACF, frequency, and boundary metrics. Runtime JSON files contain spatial-only,
temporal-only, end-to-end, samples/s, token/s, wall time, and peak allocated/reserved memory.

## 7. State/detail diagnostics

```text
repeated-static detail norm: exactly zero before and after the learned bottleneck (unit tests)
repeated-static maximum coordinate motion: exactly zero under zero-detail decode (unit tests)
dynamic R2 detail norm/distribution: holdout raw h=0.160897, encoded h=0.191008, decoded h=0.680709;
  valid fraction=1.0 and zero fraction=0.0 (norm summaries, not a fitted distribution)
dynamic R4 detail norm/distribution: holdout raw h=0.163351, encoded h=0.238345, decoded h=0.623712;
  valid fraction=1.0 and zero fraction=2.96e-08 (norm summaries, not a fitted distribution)
R2 full versus detail-zero decoded metrics: future RMSD 12.228289 / 12.275951 and dRMSD
  11.139406 / 11.230306 (full / detail-zero)
R4 full versus detail-zero decoded metrics: future RMSD 12.113563 / 12.166277 and dRMSD
  10.978540 / 11.085510 (full / detail-zero)
matched-pooling bank utilization: bank A h=1.618426, bank B h=1.584718; no detail semantics
```

## 8. Failures, warnings, and interpretation limits

- Two early remote attempts were not accepted as T0 evidence: the first exposed a pre-training
  one-step unfreeze schedule-boundary bug; the second completed R1 but exposed resume mismatch
  caused by dynamic per-epoch batch counts. The runner was corrected to precompute all 30 epoch
  batch counts (total 4,976), then the final run above completed all controls and resume checks.
- The full suite retains one pre-existing `torch.load(weights_only=False)` FutureWarning in the
  legacy multiframe test; no new test failure occurred.
- T0 uses one seed and three selected systems/nine trajectories. The apparent matched-pooling
  advantage in future RMSD/dRMSD is not a production recommendation and is confounded by its
  natural larger parameter count.
- The source data root is absent in the portable copied container, but exact clip bytes,
  provenance, indexes, and audit hashes are present; this is a path portability note, not a
  semantic split mismatch.

Do not use these three systems to choose the production ratio. The review decision concerns code
correctness and whether T0 behavior is sane enough to justify T1.

## 9. Operator decision

Leave exactly one status after review:

```text
APPROVE_T1
REQUEST_FIX_AND_REPEAT_T0
REJECT_CURRENT_CODEC_DESIGN
```

Operator status: PENDING

Operator notes:

```text
pending operator review
```
## 10. Repair repeat packet — 2026-09-04

This section is the evidence packet for the requested repair repeat. Sections 1–9 above are
historical evidence from the earlier operator review and are not rewritten.

Status: **WAITING_FOR_OPERATOR_REVIEW**

T1 status: **NOT_STARTED**

Repair operator decision being addressed: REQUEST_FIX_AND_REPEAT_T0.

### 10.1 Identity and reproducibility

- Repository: /data4/users/sihao/workspace/PVB
- Branch: fix/state-detail-codec-v2-t1-gates
- Verified source/base commit: 48bbff992e66cc5f351911e23f31750325ef3726
- Repair implementation commit: 7878df7286350b6b46cb2198d942fdcda2451ba1
- Remote: https://github.com/SHw-1313/molvid.git
- The implementation commit worktree was clean. The packet append is the only subsequent
  documentation change; its final commit and clean worktree are recorded in the final handoff.
- No historical v1/v2 source, checkpoint, report, or result artifact was overwritten.

The implementation repair commit source SHA256 set is recorded below. Generated checkpoints, plots,
and packed data remain external artifacts under the run directory.

~~~text
evaluation/codec_evaluation.py                 1c7ac1b70146a933c88fdd790ff421e52eba85af3ce6bff5e1ac0437ad8ae544
evaluation/__init__.py                          332c46b68cdff9ba80062352ba4b86aefb8f2d4a5b03d655abbb4762e5f34b28
module/state_detail_codec_v2.py                5ff5fd0f08b772e59c0067fbcc0a3c8a3881d7e921084a92bd97d8f418080928
module/__init__.py                             6f4f23a9966e88c8738bfc2e06120a43c9182ec8965fd6c318ac903b0f8924ef
trainer/codec_trainer.py                        39e02bbc68858d778a533c80aabf32b5f05769f542aa3b8366de396dc5124694
scripts/run_state_detail_codec_v2_t0.py         03211c6c94de6a9215cdbc9cfcc975ce787226731bb73b9e9d5140b162b5fd6e
scripts/run_state_detail_codec_v2_r1_diagnostic.py dd9b162c8022b43ec1958bccea927ebfbab633729771880bfcc16edafb86c477
scripts/run_state_detail_codec_v2_r1_single_clip.py e92b530cc4f757b4c995c71d228c7b8b497d331506c69488a594bd4c73d31d12
scripts/run_state_detail_codec_v2_ratio_smoke.py 625588ba6d160d3cd445eba5eaf77d3718d2b0f0421a42637be6582178efab46
tests/test_codec_evaluation.py                  eac51038968acfeb125cf89ee8539f9dc39ce277097301c579425a67c1d0a52f
tests/test_state_detail_codec_v2.py             b76fc0001077e1fb9b4e9856ea03864e368a25fe563fbafbea7979f321b4978c
~~~

### 10.2 Repairs and gate evidence

Topology metadata now uses StaticTopologyMetadata, an N-axis, coordinate-independent schema
containing atom/block/component identifiers, sample pointers, and bounded covalent bond pairs.
It no longer serializes FrameGraphBatch.edge_index, radius edges, positions, edge vectors,
distances, frame-expanded indices, or target coordinates. Tests cover N-axis bounds, longer-T
shape invariance, coordinate independence, future-frame independence, and exclusion of radius
and distance data.

The evaluator now reports per-frame Kabsch aligned_rmsd using align_mask and scores it on
loss_mask, alongside explicitly named centroid_gauge_raw_rmsd. Contacts preserve pair identity
and report precision, recall, F1/Jaccard, FP/FN, and occupancy MAE with a 4.5 Å cutoff and
covalent-pair exclusion. Dynamic correlation/ACF uses mean-removed, per-trajectory Kabsch-aligned
frame-to-frame velocity in Å/ps. RMSF is aligned to each trajectory's own first frame and
aggregated with sample-equal weighting. Frequency power above one is recorded as excessive
predicted motion.

The no-anchor reconstruction repair is a shared, bias-free CenteredCoordinateVectorStem.
Centered coordinates enter the equivariant latent before temporal packing; no per-atom x0 metadata
is added after decoding. The same centered stem and framewise equivariant decoder are used by
R1/R2/R4-SD. ratio4_matched_pooling is truthfully documented and reported as linear two-bank
pooling, latent-volume-matched but not parameter-matched.

### 10.3 Required pre-T0 diagnostics

The authoritative cached-feature diagnostic is:

~~~text
outputs/state_detail_codec_v2/repair/r1_cached_feature_diagnostic/run_20260904T_stage2c/
~~~

It contains one real clip, atlas_5e3e_A_R1_w000000, frozen TorchMD features
h=[16,887,128], v=[16,887,3,128], and centered targets [16,887,3].

| Diagnostic path | Aligned RMSD | Raw RMSD | dRMSD | Bond RMSE |
|---|---:|---:|---:|---:|
| No-stem pointwise initial | 13.652314 | 13.798742 | — | — |
| Detached pointwise, 500 steps | 5.524466 | 5.615697 | 4.559049 | 2.675140 |
| Linear equivariant vector oracle | 7.392781 | 7.466507 | 6.587139 | 3.116368 |
| Origin-only baseline | 13.701599 | — | — | — |

The diagnostic supports a pointwise reconstruction bottleneck and motivated the centered-vector
stem. It is a diagnostic, not production training evidence. Its JSON SHA256 is
1af48019a526fd12748b339bd544081b0210d0917557cd4070e25efda933ac19; the cached feature SHA256
is 6e122621df08c6447c0906a8114c1ffe174fc5c41f35f36c22e28578fc6f8506.

The formal true single-clip R1 run is:

~~~text
outputs/state_detail_codec_v2/repair/r1_single_clip/run_20260904T_stage2c/
~~~

It uses exactly one sample ID, 1,000 steps, and logs every 25 steps. The threshold was frozen
before inspecting repaired training results: aligned/raw RMSD ≤ 1.5 Å, dRMSD ≤ 1.5 Å, bond RMSE
≤ 0.5 Å, origin-only aligned improvement ≥ 0.5 Å, finite curve, and a converged final window.
All thresholds passed. Final all-frame aligned/raw/dRMSD/bond are
0.084951/0.085038/0.111515/0.024777 Å; future-frame values are
0.085115/0.085196/0.111787/0.024745 Å. Origin-only aligned improvement is 0.993800 Å.
All intended stem/codec gradients are finite and nonzero, the frozen encoder is unchanged, and
checkpoint resume passed 1000 -> 1001. Result SHA256 is
921df11a11c4418b18c61d42785b7b0793df99f7c80fbb08528bf6174d652ea1; checkpoint SHA256 is
5654bb4c37fa51976421cf4a23420fb536cc49106bbd04afa95d5dd4b8ae1dd0.

The post-R1 ratio smoke is:

~~~text
outputs/state_detail_codec_v2/repair/ratio_smoke/run_20260904T_stage3/
~~~

Each of R2, R4-SD, and matched pooling ran 200 steps on the same one-clip input, passed
zero-preserving detail, finite/nonzero gradients, unchanged frozen encoder, shared decoder
contract, and resume 200 -> 201. Final aligned/raw/dRMSD/bond/contact-F1 are R2
0.066825/0.067810/0.055249/0.040102/0.981242, R4-SD
0.072682/0.073069/0.059461/0.041076/0.981053, and matched pooling
0.193294/0.193816/0.223682/0.079457/0.960623. Summary SHA256 is
d01d56bf0dd6387640837764d9176375f49bcb48e601ccb0754f23987327b7a0.

### 10.4 Exact bounded T0 protocol and manifest

The authoritative repaired T0 run is:

~~~text
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/
~~~

The command, run inside enter-container after activating torch-ito, was:

~~~bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m scripts.run_state_detail_codec_v2_t0 --mode all \
  --store-root outputs/atlas_selected_trajectories/clip_store \
  --output-root outputs/state_detail_codec_v2/repair/t0_repeat
~~~

The run uses systems atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A, replicas R1/R2/R3,
windows w000000–w000048 for train and w000049–w000061 for the fixed late holdout,
T=16, dt_100ps, lazy loading, replacement=false, FP32, seed 20260903,
max_tokens=80000, 30 complete epochs, and a 6000-step safety cap. It independently verified
the canonical store against the historical portable store: all 558 sample IDs and per-sample
NPZ payload hashes matched. Counts are exactly 441 train, 117 holdout, and zero overlap.

Manifest/protocol hashes:

~~~text
source manifest             700f01e40f4e0fda697191cd161bb8161b97493a994415cee286354a9809450e
train index                 f8a05f905f1467864993291e64008db9d26109261f4a007f11cc15fe4ae8c3f9
late holdout index          c02d45d2cf77bfbbce787f1d8b747556d52d91e7c7de7f61baa26e6ab62a2a61
built manifest              01c7bcb0cc463eee0ff8afecc7dbaad211156d3793c0b017afb8549f70114a2f
manifest_contract.json      e0bc84c98c5a955fe85fc5b663a34af9abe28af93860603e58a11106c754a73b
protocol.json               8c14c67b62b504c74d1ade800fd354193b89f1654214bdf192e232b27483b0a6
~~~

Runtime was container cuda:0 on physical GPU mapping CUDA_VISIBLE_DEVICES=1, NVIDIA
A100-SXM4-80GB, Torch 2.5.1+cu121, CUDA 12.1, and torch_cluster 1.6.3+pt25cu121. The
protocol records coordinate_stem=centered_vector, the shared framewise equivariant decoder,
linear two-bank pooling semantics, and all evaluator semantics.

### 10.5 Three-system T0 results

All four controls completed 30 epochs and 4,976 scheduled steps. Every control has complete
holdout evaluation, best/final checkpoints, and resume 4976 -> 4977; the common TorchMD frame
encoder remained frozen. Metrics below are final late-holdout future metrics. RMSD values are Å,
velocity is Å/ps, and acceleration is Å/ps². Frequency power above one is excessive motion.

| Control | Active elements/atom | Total/trainable params | Aligned RMSD | Centroid-gauge raw RMSD | dRMSD | Bond RMSE | Contact F1 | Velocity RMSE | Accel RMSE | Dynamic corr | RMSF corr | Power ratio | Boundary error | Train s | Tok/s | E2E s/batch | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1-SD | 2048 | 643,144 / 82,560 | 0.105395 | 0.105435 | 0.143339 | 0.022667 | 0.986426 | 0.0001221 | 2.093e-6 | 0.999850 | 0.999968 | 1.012466 | 0.005946 | 4328.4 | 79,699 | 0.193 | 24.01 |
| R2-SD | 2048 | 725,064 / 164,480 | 0.078981 | 0.079039 | 0.103156 | 0.022113 | 0.988454 | 8.297e-5 | 1.323e-6 | 0.999923 | 0.999977 | 1.005878 | 0.002846 | 5971.0 | 57,773 | 0.197 | 24.47 |
| R4-SD | 1024 | 856,136 / 295,552 | 0.064856 | 0.064934 | 0.082521 | 0.021026 | 0.988925 | 7.330e-5 | 1.231e-6 | 0.999939 | 0.999983 | 1.003593 | 0.001552 | 5983.2 | 57,656 | 0.200 | 24.47 |
| R4-Matched-Pooling | 1024 | 1,102,664 / 542,080 | 0.090435 | 0.090519 | 0.115388 | 0.028109 | 0.986176 | 0.0006986 | 1.355e-5 | 0.994320 | 0.999795 | 1.004576 | 0.005997 | 4298.5 | 80,254 | 0.193 | 24.02 |

R1 is now a useful no-compression reconstruction upper bound: it reconstructs at approximately
0.105 Å raw/aligned RMSD and is worse than R2/R4 on this one-seed repeat, rather than failing
at the former approximately 11 Å scale. Structure and dynamics metrics improve together, with
power ratios close to one and no large generated-motion excess. This is bounded development
evidence, not a ratio selection claim. R2 preserves 16C active capacity; R4-SD is 8C. Matched
pooling is latent-volume-matched but has a naturally larger parameter count and is not a
parameter-matched comparator.

The aggregate report also records contact precision/recall/Jaccard, per-frame/per-block-offset
metrics, per-system metrics, velocity/acceleration, aligned RMSF, dynamic ACF, frequency power,
boundary jumps, and full versus detail-zero counterfactuals. In the latter, R2 full/detail-zero
future RMSD is 0.078981/12.275951 Å and dRMSD is 0.103156/11.230306 Å; R4-SD is
0.064856/12.166277 Å and 0.082521/11.085510 Å. Detail is therefore being used, while the
ablation is not a claim that the detail-zero path is a trained model.

Gap recovery against the prior raw historical rows is:

~~~text
R1 raw RMSD 11.424228 -> 0.105435; dRMSD 10.154870 -> 0.143339; bond 3.083183 -> 0.022667
R2 raw RMSD 11.241862 -> 0.079039; dRMSD  9.900528 -> 0.103156; bond 3.122913 -> 0.022113
R4 raw RMSD 11.138619 -> 0.064934; dRMSD  9.767301 -> 0.082521; bond 3.107002 -> 0.021026
MP raw RMSD 10.544774 -> 0.090519; dRMSD 9.306050 -> 0.115388; bond 3.344215 -> 0.028109
~~~

### 10.6 Checkpoints, curves, and artifact hashes

The report files are:

~~~text
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/aggregate_comparison.json
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/aggregate_comparison.csv
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/aggregate_comparison.md
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/loss_curves.png
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/loss_curves.pdf
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/evaluation_curves.png
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/evaluation_curves.pdf
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/block_detail_metrics.png
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/block_detail_metrics.pdf
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/performance_summary.png
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210/performance_summary.pdf
~~~

The four per-control loss curves are loss_curve_<mode>.png/.pdf in the same directory. Each
plot includes runtime/performance annotations where applicable. Aggregate/report hashes are:

~~~text
aggregate_comparison.json   e7d9ca83b280725dfac7a87c574eb02fb511bc75a6bde906f4a81904c95b6626
aggregate_comparison.csv    5b93e8f1eea6eb07828d3ea64f0032c9f4bf6a0a06d404ac5bb1317ef7ce2872
aggregate_comparison.md     db3a823b2c9cdd0e4983c328028648e23df987ebe89bb6597e0836bc713b02b1
micro_summary.json           179b5c111ffc565a5518457d02d1238e38923b063f521a1e3b2dfef7a219a45a
unfreeze_smoke.json          f609356e9e98bf807f9cc2f7dad81adb4417200a00776423d27b702302e21fe9
~~~

Final/best checkpoint SHA256 values are:

~~~text
R1 final 02434947a68b30b9ecf585a4ac86f5d68e653304861fad0c7b42de5b0db9113d
R1 best  13955f92217a291df381b3ebfb9ab1450648a7083148db9c5cbe1aad8b8e763c
R2 final e3e509b8cb5c16e7f3e9d535c47967ac5c4918cbf2ebb9fad44feb509328d9a5
R2 best  8f00848dbe6816af22f7a97f59cf1f6dd4f4fb16c9d2e573f0726a70353e8bf0
R4 final 5ed009cee98e62cb6f229d5a55ab2a4d0997453713879de7e56c21f2873b2e68
R4 best  0cb07483b89c05060be03f7210ed5373cd4ef2de2ba28e4e8242e0bf5ca608ce
MP final  b778d2532e516dab23d2e9887dec936d922a0dcbfa3d4756480644ce6803a157
MP best   0f1528ea2837fb12422a45663d3f46f68cdfc34ee33e97bb3f7b3e4d8884d0f2
~~~

### 10.7 Tests and classification

All commands were run in the active enter-container shell with conda activate torch-ito:

~~~text
python -m pytest -q tests/test_codec_evaluation.py tests/test_state_detail_codec_v2.py
28 passed

python -m pytest -q
115 passed, 1 pre-existing FutureWarning

python -m py_compile <repaired modules, all three repair scripts, and relevant tests>
passed

git diff --check 48bbff992e66cc5f351911e23f31750325ef3726 7878df7286350b6b46cb2198d942fdcda2451ba1
passed
~~~

Classification:

- implementation pass: yes;
- evaluator/topology contract pass: yes;
- genuine single-clip R1 pass: yes, against the predeclared threshold;
- bounded ratio smoke and three-clip micro-overfit pass: yes;
- bounded three-system T0 repeat: complete and finite, with a meaningful R1 upper bound and
  repaired evaluator evidence;
- development-backend recommendation: torchmd_et remains the only backend in this phase;
  no production ratio selection is made;
- multi-seed/general-system confirmation: still required before a scientific architecture
  decision.

Known limitations are one seed, three systems/nine trajectories, external binary evidence rather
than Git-tracked checkpoints, and the need for broader/multi-seed validation. The canonical
source root was available during the audit and matches the historical portable clip payloads.

The required stop has been honored: no T1 manifest, 64-system split, T1 benchmark, T1 training,
static/dynamic large-data run, DiT, observation adapter, forecasting, rollout, AF3/MSA, VAE/KL/VQ,
scaling-law, or later architecture phase began.
### 10.8 Loss-curve display addendum — 2026-09-04

The existing T0 evidence was re-plotted without retraining. The aggregate loss plot and all four
per-control loss plots now use a logarithmic y-axis. Training total loss is a solid line; the
same-control late-holdout/test total loss is a same-color dashed line, aligned to cumulative
optimizer steps including initial step zero. This makes train/holdout divergence and overfit
visible.

Plot-only command, run through enter-container with torch-ito:

~~~bash
python -m scripts.run_state_detail_codec_v2_t0 --plot-only-run \
  outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210
~~~

Updated files are loss_curves.png/.pdf and loss_curve_<mode>.png/.pdf in
outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210. No model, data, loss,
checkpoint, or evaluation value changed. Plotting source SHA256 is
bf9906d25639d93ae3d8ad1666d373a762f77c8934407999e40d3771d774779f. The updated aggregate
Markdown report SHA256 is 46eda7ede9f9abaaf305367aec7797a27c693f4556c85446c7987c4d2ea14150.

Status remains: **WAITING_FOR_OPERATOR_REVIEW**.

T1 status remains: **NOT_STARTED**.
