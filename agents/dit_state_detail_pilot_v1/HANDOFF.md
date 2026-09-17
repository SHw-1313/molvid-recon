# Pilot handoff

## 2026-09-08 — pilot start and preflight

The repair branch was verified remotely at
`28a499f4c03deb4647a6468a5c477bd8eda8d62f`. The isolated worktree is
`/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1` on
`exp/dit-state-detail-t1-pilot-v1`. The repair worktree remains separate and unchanged.

The frozen manifest preflight passed: 64 systems split 48/8/8, clips 8,928/1,488/1,488,
all three system-disjointness flags true, and `test_sampling.opened=false`. Manifest content hash
is `f8a764eb38e90c7485bf9799115d570bb868f34584ebbafab8c799fec03df4d2`; materialization hash is
`5b622ae6d0ac2a6b498f3a9388bd2053dbb29cc8aca83eecf5723987c94d49c7`.

Approved T1 result files are the full 20260904 seed-20260903 R2/R4 results. Both report
`status=passed` and schema `pvb.codec.state_detail.t1_result.v1`. Verified best-checkpoint hashes:

- R2: `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9`;
- R4: `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.

No clip payload was opened during preflight, and no test store was constructed or read. GPU 5 and
GPU 7 were idle at audit; active processes on other GPUs were left untouched.

Next action is P003: inspect the existing data/trainer interfaces and implement only the focused
resumable pilot runner described in `PILOT_PROTOCOL.md`. Do not launch training before the runner
has train/validation-only and hash/contract tests.

## 2026-09-08 — repaired runner and completed T1 profiles

The first profile launch was intentionally treated as a failed execution check, not as pilot
evidence. Both R2 and R4 reached statistics completion and then raised
`RuntimeError: training schedule hash changed` at the first train batch. The defect was in the
new pilot runner: `_batch_schedule_hash()` included sampler metadata and selected ids, whereas
`_next_train_batch()` returned only a batch-list hash. The repair adds
`_current_batch_schedule_hash()` and makes both paths use the identical materialized schedule;
the exact regression is
`tests/test_dit_pilot_runner.py::test_consumed_training_schedule_hash_matches_contract_hash`.
The failed logs remain at `outputs/dit_state_detail_pilot_v1/profiles_20260908` and
`outputs/dit_state_detail_pilot_v1/profiles_20260908_fix`; neither is valid result data.

Post-repair validation, all run inside the `torch-ito` container:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m py_compile scripts/run_state_detail_dit_pilot.py tests/test_dit_pilot_runner.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/test_dit_pilot_runner.py tests/test_state_detail_latent_adapter.py tests/test_latent_rectified_flow.py tests/test_molecular_dit.py tests/test_dit_trainer.py tests/test_dit_evaluation.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES=5 DIT_RUN_CUDA_CORRECTNESS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q -p no:cacheprovider -rs tests/test_dit_cuda_correctness.py
```

Results after the repair were respectively `44 passed` (focused), `159 passed, 1 skipped, 1
warning` (complete suite), and `1 passed` (actual CUDA correctness; the explicit CUDA opt-in was
set, so this was not the earlier skip). The one complete-suite skip is the pre-existing unrelated
CUDA-gated test. No test split was opened.

The final profile commands were the following, launched on separately audited idle GPU 5 (R2)
and GPU 7 (R4), with a new output root:

```text
CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/run_state_detail_dit_pilot.py --candidate ratio2_state_detail --manifest-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000 --codec-result /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/result.json --codec-checkpoint /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/codec_best.pt --output-root outputs/dit_state_detail_pilot_v1/profiles_20260908_fix2 --run-name profile_200_seed20260907 --device cuda:0 --seed 20260907 --steps 200 --profile --validation-interval 100
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/run_state_detail_dit_pilot.py --candidate ratio4_state_detail --manifest-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000 --codec-result /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/result.json --codec-checkpoint /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt --output-root outputs/dit_state_detail_pilot_v1/profiles_20260908_fix2 --run-name profile_200_seed20260907 --device cuda:0 --seed 20260907 --steps 200 --profile --validation-interval 100
```

Final profile summaries are:

| candidate | status | completed steps | wall seconds | train seconds/step | optimizer seconds/step | peak allocated / reserved | statistics hash | resume |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `ratio2_state_detail` | PASS | 200 | 2328.9523648791946 | 9.969244391846004 | 9.651530484342947 | 26601327104 / 84477476864 | `2dc3541483afb823fd264f48276b7a7e39cba253d7db36cac566cd38729c0e1c` | loaded 200, resumed 201 |
| `ratio4_state_detail` | PASS | 200 | 1198.268539188895 | 5.008413683730177 | 4.691751987158787 | 26356072960 / 84477476864 | `34733304618c1ffdd009bc6d78d3a8d83ea11b4d0625919d79dbfee8a1aee993` | loaded 200, resumed 201 |

Both profiles used the identical seed `20260907`, max token bound `80000`, 24 train clips per
trajectory, H schedule `[4,8]`, and validation windows `[0,30,61]` over all eight validation
systems (72 clips; 144 history evaluations). Both used BF16 autocast on NVIDIA A100-SXM4-80GB,
reported `test_opened=false`, and retained the approved frozen codec/frame-encoder hashes. The
common train data hash is
`9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304ebfe184`; the approved codec state
hashes are R2 `8473e5c8ff6d13567d73d868b3a569499a7257069dad1c8d153b610aa6e6abc0` and R4
`9a30e3838403cbd9f2cfa7344276cbdc7a8176d75ce5d4da028a3ea39ec390b9`.

Step-200 validation RF totals were R2 `1.2945222069915887` and R4 `1.3085407876826318`.
These values and the profile throughput are execution evidence only; they do not select a ratio.
The fixed validation protocol explicitly records observed `[0,H)`, future `[H,16)`, and H=4/H=8;
there is no boundary/scientific generation evaluation in this profile-only stage yet.

The next gate is P005: audit the now-idle devices and freeze one common `pilot_budget.json` from
the measured profile throughput and declared remaining overnight window. Only after that frozen
budget may matched R2/R4 pilot training begin; the test split remains unopened.

## 2026-09-08 — frozen common budget

The post-profile GPU audit at 12:03:27 +08:00 found GPU 5 and GPU 7 idle; active jobs on GPUs
0–4 and 6 were left untouched. The required budget file is
`outputs/dit_state_detail_pilot_v1/pilot_budget.json` and was validated as JSON before any
matched pilot step.

The declared overnight window is 2026-09-08 12:03:27 through 2026-09-09 08:00 local time:
71,793 seconds remain and the 80% usable limit is 57,434.4 seconds. Using measured profile
throughput, validation/checkpoint overhead, a 1,800-second train-statistics reserve, and a
5,400-second generated-validation reserve, the common 100-step grid freezes
`S=4500`. Estimated totals are R2 57,199.853 seconds and R4 32,752.177 seconds; the next
100-step budget is estimated at 58,308.324 seconds and does not fit the usable limit. Both
candidates therefore receive exactly 4,500 steps, seed `20260907`, the same ordered schedule,
H=4/H=8 mixture, and validation cadence. `S>=1000` and two idle GPUs are confirmed.

P005 is complete. The next action is the matched two-GPU pilot: R2 on GPU 5 and R4 on GPU 7,
with train/validation stores only, no test construction, validation RF loss checkpoint selection,
and final fixed validation generation evaluation. No ratio will be declared the winner.

## 2026-09-09 — matched T1 pilot completion and operator handoff

The final matched run used the frozen pilot implementation commit
`761e6fb634a6eff916f9ee3386f91bffa7fc14ce` and output root
`outputs/dit_state_detail_pilot_v1/full_20260908_S4500`. No source code was changed during the
run. The exact commands were:

```text
CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/run_state_detail_dit_pilot.py --candidate ratio2_state_detail --manifest-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000 --codec-result /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/result.json --codec-checkpoint /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/codec_best.pt --output-root outputs/dit_state_detail_pilot_v1/full_20260908_S4500 --run-name seed20260907_S4500 --device cuda:0 --seed 20260907 --steps 4500 --validation-interval 100 --evaluate-generated
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python scripts/run_state_detail_dit_pilot.py --candidate ratio4_state_detail --manifest-root /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000 --codec-result /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/result.json --codec-checkpoint /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt --output-root outputs/dit_state_detail_pilot_v1/full_20260908_S4500 --run-name seed20260907_S4500 --device cuda:0 --seed 20260907 --steps 4500 --validation-interval 100 --evaluate-generated
```

Both commands completed with `status=PASS`, `target_steps=4500`, train/validation-only access,
and `test_opened=false`. The run used NVIDIA A100-SXM4-80GB devices (R2 visible GPU 5, R4 visible
GPU 7), BF16 autocast, seed `20260907`, and the identical deterministic H=4/H=8 schedule. The
validation subset was fixed to all eight validation systems, all three replicas, windows 0/30/61:
72 clips and 144 H evaluations. The reports are:

```text
outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio2_state_detail/seed20260907_S4500/pilot_summary.json
outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio4_state_detail/seed20260907_S4500/pilot_summary.json
```

Each report has a fresh checkpoint/statistics reconstruction and resume from 4500 to 4501. The
selected checkpoint is `best_validation.pt` in each candidate directory and selection is by
validation RF loss only. The summary field `completed_steps=4100` is retained as selected/best
checkpoint metadata; the validation history and resume check reached step 4500/4501.

Frozen provenance and contracts:

- manifest content hash:
  `f8a764eb38e90c7485bf9799115d570bb868f34584ebbafab8c799fec03df4d2`;
- manifest materialization hash:
  `5b622ae6d0ac2a6b498f3a9388bd2053dbb29cc8aca83eecf5723987c94d49c7`;
- common T1 train data hash:
  `9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304ebfe184`;
- R2 codec checkpoint/result/state hashes:
  `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9`,
  `0b354280405f3001c443ed18378b9181caef1e4bf0d1cb65f5f4c8ed3dd47e02`,
  `8473e5c8ff6d13567d73d868b3a569499a7257069dad1c8d153b610aa6e6abc0`;
- R4 codec checkpoint/result/state hashes:
  `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`,
  `1259299fa1e23f9dfd4915ae95a7cc4fc9dd2f3f4d1683b4adda5d677a2b981c`,
  `9a30e3838403cbd9f2cfa7344276cbdc7a8176d75ce5d4da028a3ea39ec390b9`;
- train-only statistics hashes: R2
  `2dc3541483afb823fd264f48276b7a7e39cba253d7db36cac566cd38729c0e1c`, R4
  `34733304618c1ffdd009bc6d78d3a8d83ea11b4d0625919d79dbfee8a1aee993`;
- frame encoder source/state hashes:
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a` and
  `c8e3fe219b4fc100774a03c74150dc8d2b8d12299d334f59102d44be2e2c377f`.

Generated validation protocol and results:

```text
observed: [0,H)       future: [H,16)
boundary: [H-1,H)     full: [0,16) diagnostic only
```

| candidate | H | future aligned RMSD | future dRMSD | generation-gap aligned RMSD | generation-gap dRMSD |
|---|---:|---:|---:|---:|---:|
| ratio2_state_detail | 4 | 3.1695463526 | 2.5373122834 | 3.1470176881 | 2.5150342596 |
| ratio2_state_detail | 8 | 3.0175429412 | 2.3996681333 | 2.9950194820 | 2.3774038546 |
| ratio4_state_detail | 4 | 3.1684368944 | 2.6287575211 | 3.1470976002 | 2.6099774709 |
| ratio4_state_detail | 8 | 3.0011343126 | 2.4643223515 | 2.9797902245 | 2.4455289099 |

| candidate | validation RF total at step 200 | wall seconds | optimizer seconds/step | train steps/s | peak allocated/reserved bytes |
|---|---:|---:|---:|---:|---:|
| ratio2_state_detail | 1.2945222069915887 (profile only) | 58444.19946962781 | 11.128112767636466 | 0.08649004534495383 | 26781804032 / 84477476864 |
| ratio4_state_detail | 1.3085407876826318 (profile only) | 30622.021786798257 | 5.482367828324851 | 0.1688784381198368 | 26356072960 / 84477476864 |

The generated validation values above are execution evidence only. They must not be used to rank
R2 versus R4 or to declare a winner. RMSF/ACF/contact outputs, where present in the full report,
are secondary diagnostics only. The test dataset was never constructed or read; the manifest
continues to report `test_sampling.opened=false`.

The final full output tree is intentionally untracked: no checkpoints, statistics tensors, logs,
or large `pilot_summary.json` files were staged. A single push attempt for
`exp/dit-state-detail-t1-pilot-v1` was blocked by the environment network-export safety review;
no destructive operation, retry, or workaround was performed. The local branch is ready for
independent operator review at the implementation commit plus this evidence update.

Final status: `WAITING_FOR_OPERATOR_REVIEW`.
