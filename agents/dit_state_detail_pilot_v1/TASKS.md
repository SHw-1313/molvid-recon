# Pilot tasks

- [x] P001 verify the pushed repair commit, isolated branch/worktree, frozen manifest counts,
  system disjointness, unopened test sampling, and approved T1 codec checkpoints
- [x] P002 create the pilot protocol and preserve the branch-level safety transition in `AGENTS.md`
- [x] P003 implement the focused resumable real-T1 pilot runner with train/validation-only data
  access, frozen codec/frame encoder, train-only statistics, deterministic H=4/H=8 schedule,
  atomic checkpoints, and hash/contract refusal; verified by runner tests and T1 preflight
- [x] P004 run one 200-step profile for each candidate with validation RF loss and checkpoint resume;
  repaired schedule-hash contract and completed both candidates under `profiles_20260908_fix2`
- [x] P005 freeze and record one common `pilot_budget.json`; two idle GPUs were available and the
  common budget is `S=4500` (>=1000)
- [x] P006 run the matched R2/R4 pilot with the common seed, schedule, and step budget without
  opening test; both candidates reached the frozen target `S=4500` and passed checkpoint resume
- [x] P007 select checkpoints by validation RF loss and evaluate the fixed all-eight-system
  validation subset for H=4/H=8 future metrics and codec-oracle generation gaps
- [x] P008 append exact commands, hashes, profiles, losses, checkpoints, warnings, and final
  operator-review status; do not declare a ratio winner

## 2026-09-09 — P006/P007/P008 matched pilot completion

The matched pilot completed for both candidates from implementation commit
`761e6fb634a6eff916f9ee3386f91bffa7fc14ce`, using seed `20260907`, the frozen common budget
`S=4500`, identical ordered H=4/H=8 training and validation schedules, and train/validation-only
stores. Both reports have `status=PASS`, `target_steps=4500`, `test_opened=false`, and a fresh
resume check from step 4500 to step 4501. The report field `completed_steps=4100` is the selected
best-checkpoint metadata; validation history and the resume check reached the frozen 4500-step
target. Both selected checkpoints were chosen by validation RF loss only:

- R2: `outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio2_state_detail/seed20260907_S4500/best_validation.pt`
- R4: `outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio4_state_detail/seed20260907_S4500/best_validation.pt`

Final report paths are the corresponding `pilot_summary.json` files under the same two candidate
directories. Their approved codec checkpoint hashes are R2
`b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9` and R4
`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`; codec state hashes are R2
`8473e5c8ff6d13567d73d868b3a569499a7257069dad1c8d153b610aa6e6abc0` and R4
`9a30e3838403cbd9f2cfa7344276cbdc7a8176d75ce5d4da028a3ea39ec390b9`. The common T1 train data
hash is `9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304ebfe184`. Statistics were
reported as `production_t1_train_only`; their hashes are R2
`2dc3541483afb823fd264f48276b7a7e39cba253d7db36cac566cd38729c0e1c` and R4
`34733304618c1ffdd009bc6d78d3a8d83ea11b4d0625919d79dbfee8a1aee993`.

The fixed validation subset contains all eight validation systems, all three replicas, windows
0/30/61, 72 clips, and 144 H=4/H=8 history evaluations. The report protocol is explicit for
each conditional evaluation: observed `[0,H)`, future `[H,16)`, boundary `[H-1,H)`, and full
`[0,16)` diagnostic. Future metrics and generation gaps were:

| candidate | H | future aligned RMSD | future dRMSD | generation gap aligned RMSD | generation gap dRMSD |
|---|---:|---:|---:|---:|---:|
| R2 | 4 | 3.1695463526 | 2.5373122834 | 3.1470176881 | 2.5150342596 |
| R2 | 8 | 3.0175429412 | 2.3996681333 | 2.9950194820 | 2.3774038546 |
| R4 | 4 | 3.1684368944 | 2.6287575211 | 3.1470976002 | 2.6099774709 |
| R4 | 8 | 3.0011343126 | 2.4643223515 | 2.9797902245 | 2.4455289090 |

Runtime evidence: R2 wall `58444.19946962781` s, optimizer mean `11.128112767636466` s/step,
train throughput `0.08649004534495383` steps/s, peak allocated/reserved
`26781804032/84477476864` bytes; R4 wall `30622.021786798257` s, optimizer mean
`5.482367828324851` s/step, train throughput `0.1688784381198368` steps/s, peak
allocated/reserved `26356072960/84477476864` bytes. Both ran on NVIDIA A100-SXM4-80GB with
BF16 autocast, and the generated reports include fresh statistics/checkpoint recovery and the
frozen codec/frame-encoder hash checks.

These are real-T1 execution and validation evidence only. They do not rank R2 versus R4 or select
a scientific winner. The full output directories, checkpoints, statistics artifacts, and large
JSON reports remain untracked and were not staged. A push of the pilot branch was attempted once
but was blocked by the environment's network-export safety review; no workaround or repeat was
performed. Final phase status remains `WAITING_FOR_OPERATOR_REVIEW`.

## 2026-09-08 — P003/P004 execution evidence and repair

The first two profile attempts exposed a real runner defect before any valid training result:
`_batch_schedule_hash()` hashed the full sampler contract, while `_next_train_batch()` returned a
hash of only the batch lists. Both candidates therefore stopped at the first training batch with
`RuntimeError: training schedule hash changed`. The runner now hashes the same materialized
epoch-specific sampler contract consumed by `_next_train_batch()`, and
`tests/test_dit_pilot_runner.py::test_consumed_training_schedule_hash_matches_contract_hash`
guards the behavior. The failed outputs are retained under `profiles_20260908` and
`profiles_20260908_fix`; they are not valid pilot evidence.

The repaired run completed both candidates in `outputs/dit_state_detail_pilot_v1/profiles_20260908_fix2`:

| candidate | status | steps | wall seconds | train seconds/step | optimizer seconds/step | peak allocated/reserved bytes | resume |
|---|---:|---:|---:|---:|---:|---:|---|
| `ratio2_state_detail` | PASS | 200 | 2328.952 | 9.969244 | 9.651530 | 26601327104 / 84477476864 | 200 → 201 |
| `ratio4_state_detail` | PASS | 200 | 1198.269 | 5.008414 | 4.691752 | 26356072960 / 84477476864 | 200 → 201 |

Both used NVIDIA A100-SXM4-80GB, BF16 autocast, seed `20260907`, GPU 5/7, train-only statistics,
the same H=4/H=8 schedule, and the same validation plan (all 8 validation systems, 72 clips,
windows 0/30/61). Data hash is
`9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304ebfe184`; both report
`test_opened=false`. Statistics hashes are R2
`2dc3541483afb823fd264f48276b7a7e39cba253d7db36cac566cd38729c0e1c` and R4
`34733304618c1ffdd009bc6d78d3a8d83ea11b4d0625919d79dbfee8a1aee993`.

Step-200 validation totals were R2 `1.2945222069915887` and R4 `1.3085407876826318`.
These are execution/profile observations only and are not a scientific ranking.

## 2026-09-08 — P005 frozen budget

At the post-profile audit, GPU 5 and GPU 7 were idle; no other GPU process was touched. The common
budget is frozen in `outputs/dit_state_detail_pilot_v1/pilot_budget.json` before matched training:
`S=4500`, seed `20260907`, validation interval 100, identical H=4/H=8 schedule, and no test
access. The declared local overnight cutoff is 2026-09-09 08:00, with 57,434.4 seconds available
at the required 80% fraction. The conservative R2 estimate including 30 minutes train-statistics
reserve and 90 minutes generated-validation reserve is 57,199.85 seconds; the next 100-step
budget estimates 58,308.32 seconds and therefore exceeds the usable window. This makes 4,500 the
largest checked 100-step common budget under the recorded assumptions. The R4 estimate is
32,752.18 seconds. The budget is frozen before any matched pilot step.
