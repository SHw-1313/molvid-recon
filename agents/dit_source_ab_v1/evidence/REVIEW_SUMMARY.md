# Source A/B v1 review evidence

Date: 2026-09-10. This is a small text index; generated checkpoints, JSONL histories,
and large profile artifacts remain outside Git under the container output root.

## Scope

- Worktree: `/data4/users/sihao/workspace/molvid-dit-factorized-backend-v2`.
- Branch: `perf/dit-factorized-backend-v2`; baseline was `557a9042cb545750706e37040b26fc000e39b9dc`.
- A source implementation was read from fixed commit `a25e6bf66214a82adf9a56d492a85a28207ef248`; no merge or cherry-pick was used.
- No push, remote merge, test-payload access, codec edit, or conditional-prior model generation.

## Imported diagnostics

Imported exactly: `evaluation/dit_diagnostics.py`, `scripts/evaluate_dit_pilot_diagnostics.py`,
`config/dit_pilot_diagnostics_v1.yaml`, and `tests/test_dit_diagnostics.py`.
The fixed A worktree was not modified.

## Frozen inputs

- Manifest: `manifest_20260904_token80000`; valid count 1488; preflight reported test unopened.
- R4 codec SHA256: `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
- Statistics SHA256: `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`.
- Device: A100-SXM4-80GB UUID `f0bea51c-913e-fe29-1cdd-4cde2adb3ff3`, visible GPU 5,
  torch 2.5.1+cu121, CUDA 12.1, bfloat16, TF32 disabled.

## Checks

- Focused CUDA source tests: 3 passed; affected backend/RF/trainer/model/source tests: 24 passed;
  imported diagnostic tests: 10 passed. No full-repository pytest was run.
- Real verify passed for R4, H4/H8, 8 Euler steps: finite decode, exact observed clamp,
  source diff <=4.77e-7, target-plus-center diff <=1.83e-6, test unopened.
- Real source check passed on 8 fixed R1/w30 clips; repeat center passed all implementation
  checks and was selected without future-coordinate ranking. Block-state was retained as fallback.

## Profile and contract

- Source runner profile: Gaussian p50/p90 0.4928/0.5879 s; conditional p50/p90
  0.6737/0.7099 s; 5 warmup and 20 measured updates per arm.
- RAM cache equivalence passed; source center was 72,581,120 bytes, but science cache was
  frozen disabled under the 8 GiB bound (direct 0.1845 s, RAM hit 0.0422 s, miss 0.2361 s).
- Candidate 20,000/10,000/4,500 were feasible under the 32 GPU-hour and 36 wall-hour
  limits; selected common endpoint: 20,000 updates.
- Shared initialization hash: `be57b911c473805a7995e3ea33803e4947cacd8275ff44bb80f6b9c48aeac7ae`.
  Backend: `factorized_v2`; source sigma/noise std: 1.0; history: H4/H8.

## Training and evaluation

- Gaussian and conditional each completed 20,000 optimizer updates, 20,000 history rows,
  and shared initialization; train summary status was PASS and test remained unopened.
- Real evaluation used both common step-20,000 checkpoints, H4/H8, final and subset steps,
  and 4 subset draws. Output root: `/tmp/molvid_dit_source_ab_v1/20260910_r4_source_ab_v1`.
- Existing evaluator wrote 52 main rows per arm because some dynamic validation batches contain
  multiple clips; it also omitted system identity, so reported `system_count=1`. These results
  are descriptive only, not a strict 72-clip/system-balanced paired claim.
- Descriptive main aligned RMSD/bond RMSE: Gaussian 2.1646/0.8945 A; conditional 2.0777/0.7049 A.
  Main contact F1: Gaussian 0.7124; conditional 0.7379. Subset rows: 128 per arm.

## Known limitations

- The outer change review previously rejected the narrow repair to the older
  `scripts/profile_dit_backend.py`; it was restored clean and is not claimed as F1/F2 PASS.
- Imported diagnostic unit tests pass, but the old diagnostic evaluator still has known
  occupancy/row-grouping limitations; no full A diagnostics run was started.
- Strict paired clip/system aggregation remains a follow-up review item. No quality selection
  or scientific conclusion is made here.

## Reproduction and final state
