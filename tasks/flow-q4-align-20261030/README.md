# Q4 training alignment with matched frozen adapters

2026-11-02 final: all three arms completed schedule slot512 (A256 real optimizer updates; B/C512). Pair checks32/512 passed. All108 new trajectories plus36 exact parent and72 oldT trajectories are scored, full_study_complete=true, missing/errors empty. GPU cost including checks/failures is29,711.239107s =8.253122 A100h of24h (checks123.224768s, training9,952.036907s, generation19,635.977432s). CPU wall1,480.915s /CPU1,503.959s. No new computation was run for this closeout; unused budget does not authorize another experiment. L4096 remains the default.

Result: Q1 training helped the old shared adapter's Q4 rollout under this recipe, but did not improve every sampling protocol. With identical joint windows and LR slots, oldTshared versus A has lower protein bond/angle error in6/6 cases and Q4 CA-ACF error in5/6; the same addition worsens Q12 and joint CA-ACF in6/6. A omits256 Q1 updates, so the comparison includes the additional Q1 optimizer/loss exposure and is not proof of a particular Q1 mechanism independent of dose.

B's explicit Q4 training versus oldTshared improves Q4 protein bond/angle errors in6/6, ligand angle in3/3 and CA-ACF in4/6. Average amplitude decreases further. C versus B restores average amplitude towardsMD, but Q4 protein bonds/angles worsen6/6, ligand angles3/3 and CA-ACF4/6. Thus there is a geometry–motion tradeoff, not a uniformly superior new default.

Q4 rollout below: first seed means within each system, then equal means of three systems per dataset/physical clock. Amplitude ratio ideal1; CA-ACF is RMSE to same MD. Geometry columns are percentages, not fractions. ATLAS100ps and MISATO80ps are never pooled.

| Dataset / clock | Arm | CA amplitude | CA ACF error | G8 | Protein bond % | Protein angle % | Ligand angle % |
|---|---|---:|---:|---:|---:|---:|---:|
| ATLAS 100ps | MD | 1.00000 | 0.00000 | 0.17338 | 0.00000 | 0.00079 | NA |
| ATLAS 100ps | A joint only | 1.33402 | 0.19682 | 0.37760 | 2.03688 | 2.77008 | NA |
| ATLAS 100ps | old T shared | 0.84666 | 0.14190 | 0.34567 | 0.24185 | 0.44916 | NA |
| ATLAS 100ps | B shared Q4+Q1 | 0.79925 | 0.13324 | 0.33302 | 0.17555 | 0.35055 | NA |
| ATLAS 100ps | C routed Q4+Q1 | 0.94257 | 0.15446 | 0.35654 | 0.38834 | 0.63428 | NA |
| MISATO 80ps | MD | 1.00000 | 0.00000 | 0.10987 | 0.00086 | 0.00050 | 0.00000 |
| MISATO 80ps | A joint only | 1.31854 | 0.21593 | 0.40782 | 0.84841 | 1.54500 | 30.23690 |
| MISATO 80ps | old T shared | 0.91615 | 0.14691 | 0.36981 | 0.06852 | 0.23454 | 22.88434 |
| MISATO 80ps | B shared Q4+Q1 | 0.87538 | 0.14021 | 0.36473 | 0.05026 | 0.18498 | 21.86168 |
| MISATO 80ps | C routed Q4+Q1 | 0.99459 | 0.15689 | 0.37487 | 0.12059 | 0.35898 | 24.23547 |

The large-motion6e5y case remains underpowered: oldTshared→B→C Q4 amplitude0.4540→0.4298→0.4958, fixed-MD-flexible-region amplitude0.3806→0.3535→0.4255, CA-ACF error0.11390→0.12712→0.09544. Better average ACF does not remove this failure. Q4 G8 remains too high: B0.3330/0.3647 versus MD0.1734/0.1099 (ATLAS/MISATO). JointQ50 G remains near zero. Q12 B's ACF worsens versus oldTshared in6/6, despite better geometry.

B reduces the seam/within-block step contrast: ATLAS0.681/0.691Å, MISATO0.618/0.621Å, compared with MD0.994/0.987 and0.795/0.779Å. Both sides are smaller thanMD, so smoother seams alone are not evidence of restored dynamics. MISATO Q4 ligand chirality anomaly remains46.53% inB versus46.32% oldTshared (two supported cases;5TV1 NA), despite bond/angle/clash improvements.

Existing evidence: [comparison](../../runs/flow_q4_align_20261030/evaluation/comparison/comparison_summary.json), [all draw metrics](../../runs/flow_q4_align_20261030/evaluation/trajectories/draw_metrics.csv), [seams](../../runs/flow_q4_align_20261030/evaluation/trajectories/seam_frames.csv), [D curves](../../runs/flow_q4_align_20261030/evaluation/trajectories/distance.csv), [raw viewer](../../runs/flow_q4_align_20261030/evaluation/viewer/index.html). Viewer19HTML/234NPZ exist; this is a file verification, not manual WebGL review.

## Frozen design and historical execution record

The dated launch descriptions below are retained as history and superseded by the completed result above. Their PIDs are historical, not live jobs.

2026-10-30: formal training is running on physical GPU7 (32GiB allocator cap), science commit50950be. A shared_joint_only completed its first actual optimizer update: RF loss0.4038921595, gradient norm0.0295478422, both finite; slot2 deliberately skips Q1. B/C are queued in the same finite lane, not yet formally trained. No new scientific evaluation result is available. Independent cap86400 A100 GPU seconds (24h) includes all checks/failures/three trainings/108 generations; CPU scoring is separate. Existing M memory evaluation on GPU4 continues unchanged.

The central `/workspace/molvid_recon/{AGENTS.md,HANDOFF.md,TASKS.md}` are the sole management authority. This isolated worktree uses task-routing science1e8598e and warm-starts the same L4096 weights, not old trained adapters. Frozen backbone10,009,356, trainable adapters166,528, total10,175,884. Existing torch-ito only, no dependency change.

A `shared_joint_only` follows the original512 task slots but skips all256 Q1 optimization slots. Its256 joint updates use the exact original H4Q12/H8Q8 windows, target sets, RF epsilon/progress stream, original slot LR and joint auxiliary schedule. No replacement joint data. The omission of Q1 changes total AdamW updates and total supervised targets; this is the treatment, not complete matching. Final step means schedule slot512, while actual_optimizer_updates is256. B/C actual optimizer updates are512.

B `shared_q4_q1` and C `routed_q4_q1` retain256 original joint and256 Q1 slots. H4Q12 splits into three H4Q4 blocks; H8Q8 into two H4Q4 blocks with the immediately preceding real4 frames. All original target atoms/frames occur exactly once per original slot, with4/Q accumulation and one clip/update. This is teacher-history supervision, not generated-history training. Draw full-window progress and standard epsilon once, then slice to the same physical targets; each chunk source is centered at its own last observed frame. B/C differ only in adapter mixing. Relative to oldT, query grouping and visible history change.

Shared mixing(.5,.5); routed joint(.75,.25)/Q1(.25,.75). Both adapter branches always compute. Seed71/newAdamW/original512-slot LR with128warmup; save first32slots and exact resume. Source rho1.1397942064485225Å, ligand lambda.09840946099484672, every8complex×original-H visits preserved. Joint auxiliary counts once per original slot; split contributions use4/Q and normal bands from the complete original target window. No shorter-window band or increased loss dose. Original joint/Q1 RNG streams remain independent. Full exposure reference is `runs/flow_q4_align_20261030/exposure_plan.json`.

Fixed108 new trajectories: sixDEV×seed17/29×three arms×Q4,Q12,jointQ50. Priority complete matched Q4 groups, thenQ12, thenjoint ability checks; original35NFE per block and originalT final-block semantics. OldT shared/routed and L4096 exact protocol rows are reused. ATLAS100ps/MISATO80ps and history/query/protocol/window remain separate. Existing geometry, ligand internal/pocket motion, CA RMSF amplitude/spatial correlation, ACF/G1,4,8/D, seams/individual-block amplitudes and raw viewer are retained. No extra model/loss or evaluation framework.

Training each has10800GPU-second cap, including first32/resume/failures. Checks at most3600seconds; generation subtracts all actual checks/training from the one86400 cap, then its own accumulated process time. Expected generation from oldT per-protocol measured costs is7.77GPUh for108draw; this is an estimate, not a shared-resource speed guarantee. Insufficient budget produces explicit partial coverage in fixed priority order.


Actual execution: host lane1719402 / runner1719407 / first32 trainer1719413 correspond to container1464159 /1464163 /1464165. Each arm saves first32 slots and resumes the same optimizer/data/RNG/slot schedule to512; A performs256 actual updates, B/C512. The ordered lane proceeds A→B→C without another chat confirmation. Authoritative execution receipts are `runs/flow_q4_align_20261030/training_lane.json` and each `{arm}_all_execution.json`; live optimization evidence is each `updates.jsonl`.

The generation waiter is host1719404/container1464160, physical GPU7; receipt `generation_execution.json` is waiting_for_training. The CPU tail is host1719405/container1464161; `cpu_wait.json` is waiting_for_generation_terminal with100800-second finite wait. It then checks actual32/512 checkpoints, imports available real draws, scores with the existing common definitions and exports `evaluation/viewer/index.html`. Failed/missing arms remain NA; no parent output is relabeled as a new draw. Both waiters are already running.

Necessary checks completed: identical initial states across three modes; real two-slot training/optimizer checkpoints; B exact resume toslot4; full sample/epsilon/progress/LR identity against the oldT slot stream. Real maximum17002-atom original H8Q8 split-Q4 backward took15.5676s, reserved14.584GiB, with finite nonzero adapter gradients and no base gradients. One real ligand event accumulated three Q4 native35NFE tails in54.8537s, full original12-frame normal band and full-window epsilon consumption preserved. Check-only optimized models are discarded and never used for formal training. Two test-entry failures (wrong import, shared output directory collision) are retained; combined necessary checks/failures cost123.224768 GPU seconds. Remaining total allowance is86276.775232seconds minus formal training/generation elapsed time; it is not reset between arms.

Reproduction entries (existing torch-ito, this worktree): `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python -u tasks/flow-q4-align-20261030/run_training.py`; `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python -u tasks/flow-q4-align-20261030/run_generation.py --wait --allocator-GiB 32`; `PYTHONPATH=$PWD bash tasks/flow-q4-align-20261030/run_cpu.sh --wait --allow-partial --max-wait-seconds 100800`. These are already launched: do not execute them again over existing receipts. Check evidence is `runs/flow_q4_align_20261030/checks/cuda/results.json`; all costs are in `checks_execution.json` and subsequent finite process receipts.
