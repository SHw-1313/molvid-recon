# Lag-conditioned dynamic Q1 and joint Flow

2026-10-28: all five arms completed 512 updates with finite final losses/gradients. The final pairing check passed for all arms: mixed-arm joint inputs/source/default RNG/auxiliary state match the C256 joint-dose reference; no differences were reported. The fixed generation list is still running (checked 714/800 jobs, 2588/2760 draws); priorities 1–6 are complete and the remaining high-NFE Q12/Q4/coarse-fine jobs continue within the original 172800 GPU-second total. A separate CPU preview is scoring the completed core trajectories and endpoint/compatibility results without interrupting generation or the final CPU tail. The first preview exposed a task-adapter protocol-string mismatch (132 import errors); that failed output is preserved, two metadata/interface fixes passed real generated joint/endpoint/H1-rollout checks, and the corrected previews imported 132/132 and 888/888 new draws with errors empty. The completed motion preview and first-origin endpoint results are summarized below; trajectory geometry, compatibility and the full-study conclusion remain pending. Results are not inferred from process exit codes. Central molvid_recon/AGENTS.md, HANDOFF.md and TASKS.md remain the sole authority.

Five arms start from `/workspace/MOLVID-FLOW-LIGAND-4096-20261026/runs/flow_ligand_4096_20261026/ligand/coordinate_final.pt`: `control`, `h1`, `history_mix`, `dense_history`, `lag1`. All use the same 10,009,356-parameter group-QK/atom-V model, ordinary bidirectional query attention, fixed RF source rho1.1397942064485225 Å and native18interval/35NFE. Static observation corruption0.2 Å is unchanged and is not applied to clean dynamic histories. No bridge training, new loss, time encoding or noise model.

All arms warm-start the same completed weight and reset AdamW, data visits and stage RNG; seed71,512 updates,lr1e-4,warmup128,cosine to1e-5. This is a new task stage, not exact continuation of the old4096 LR trajectory. Each mixed arm alternates256 joint and256 actual dynamic Q1 updates, with no additional factor0.5 on each sampled task loss. Joint uses H4Q12/H8Q8 alternating by independent joint ordinal (not global step). All four B arms share joint windows/source/progress and Q1 system/anchor/standard-normal draws. `control` has512 joint updates; its256 checkpoint matches the mixed arms' total joint exposure and first256 joint random stream. Its earlier optimizer step/LR position remains different, so it is a dose reference, not a complete causal isolation of task conflict. C512 is the same-update-budget reference. Primary history/lag attribution compares the four mixed arms at the same256+256 exposure.

Q1 target lag is delta/2delta/4delta in50/30/20 proportions (ATLAS delta100ps, MISATO80ps). `h1` only observes the last frame; `history_mix` uses20%H1/80%H4, with H4 half rollout-compatible timestamp transitions, one quarter dense ages[3,2,1,0]delta and one quarter multiscale ages[8,4,1,0]delta. `dense_history` changes only H4 placement to dense, retaining the same H1 indicator. `lag1` changes only target lag to delta while retaining the original sampled history times. Endpoint and history sources must be real within one TRAIN replica. Old16-frame caches are not reinterpreted as arbitrary continuous trajectories. Actual eligible scope and crossings are recorded in the data manifest.

Ligand auxiliary retains lambda0.09840946099484672, every8complex/history visits,35NFE source endpoint with only the last4intervals differentiated. Joint and kernel visit counters are separate to retain the joint auxiliary stream across arms. Normal-band settings are unchanged; a Q1 prediction is compared with detached multi-frame TRAIN support from the same system, never a silently collapsed one-frame MAD. Actual task/domain/H/lag/history counts and geometry events are logged. Target truth cannot enter observed conditions.

Budget: independent 172800 GPU process-wall seconds total. Initial planning division was five training lanes 21600 s each, auxiliary checks 7200 s and generation 57600 s. PM approved transferring unused training/check allocation to generation within the same total; the one generation runner waits for all training receipts to settle and subtracts actual recorded training/check seconds before setting its available budget. No automatic retry or expansion. CPU data preparation/scoring is separately timed. Final results separate dataset/actual dt/history/Q/protocol/reference window; independent endpoint draws are not assembled into trajectories or scored for ACF. StandardQ50 keeps the existing initial index3/MD window; endpoint experiments use separate fixed real initial states. Energy scores use held-out MD outcomes, not a claim of known conditional distributions. E/bridge is only a reusable kernel/history interface; existing coarse-fine remains joint-anchor pure-clamp sampling.

## Actual data, checks and continuous execution

TRAIN inventory512ATLAS/1022MISATO was read in the existing container. The fixed256 Q1 visits select128 distinct systems per domain; lag counts128/77/51, H1/H4=51/205, H4 rolling/dense/multiscale103/51/51. Four windows require17real frames; original raw data supplied the extra frame with original16 x/bpos exactly matching, unchanged replica/atom/topology, and all complex windows satisfy the existing5Å rule (maximum3.882019Å). No DEV/TEST entered training. Manifest: `../../runs/flow_kernel_20261027/data/data_manifest.json`; CPU checks passed.

Necessary CUDA checks passed: actual3M35 dynamicH4Q1 finite RF/temporal gradient, H1 native35NFE4.63s, checkpoint strict load and no new parameters; Q1 penalty uses detached16frame support and zero protein-endpoint gradient. Maximum selectedH4 TRAIN3KWF11955atoms RF FB finite, peak11.23GiB; small5A5Q nonuniformH4 native35NFE/tail4 backward finite (21bonds/29angles), peak1.14GiB. Auxiliary62.861541s includes the first17.895623s failed fixture check: its almost-common translation did not force abnormal internal geometry, fixed by perturbing one ligand atom; actual training geometry or thresholds were not changed.

Finite commands are `run_train.py --arm ...` (first32 saved and automatically resumed through256/512) and `continue_lane.sh 4|2`. Joint and kernel ligand auxiliaries also have independent saved source/default RNG instances. `check_pair.py` reads actual logs/checkpoints on CPU. Generation uses one `run_generation.py --wait` receipt after all five training receipts settle; it deducts actual training and auxiliary seconds from172800 before each subprocess. This avoids parallel generation lanes each claiming the remaining budget. Parent L4096 standard48draw is reused read-only. Endpoint/direct, generated-history rollouts and same-condition joint/composed-kernel compatibility are distinct protocols.

Predeclared genplan requests2760newdraw:2160independent endpoints,240standardQ50,180native-history delta/2delta/4delta rollouts,160small compatibility draws,8cross-history rollouts and12C256 joint draws. Priorities are t0=12 endpoints → standardjoint/C256 → delta rollouts → small compatibility/history cross → t0=28/44 endpoints → coarse-lagrollouts → other Q50 protocols. Partial budget coverage is reported, never selected by results. Standard reference remains originalidx3; endpoint origins12/28/44 are separate. Coarse4delta gives12uniform points through48delta, with2delta uncovered; not13regular points or an interpolated50-frame path. H1/H4 cross in6e5y/4J8T separates native-policy differences; differentlags/history/windows stay in separate tables. No new bridge training; oldCF is still joint-anchor pure clamp.


## Save/resume launch history and current evaluation tail

Historical startup snapshot (superseded by the completed five-arm check_pair512 above): `runs/flow_kernel_20261027/check_pair32.json` passed the available C/Hmix pair, errors empty: the first 16 Hmix joint updates exactly match C's first 16 joint windows, history/query settings, source epsilon summaries, progress and auxiliary triggers. Coverage is explicitly C32/Hmix32; the other arms are still queued and are not claimed as checked. The final CPU tail repeats this read-only pairing for all available arms at 32/512, with C256 as the joint-dose reference.

Historical startup costs at that snapshot were C first32 1111.828959 s, Hmix first32 336.741593 s and all auxiliary checks/failures 62.861541 s: 1511.432093 GPU seconds (0.419842 h) settled, plus the currently running resumed training processes. The receipts add each complete or failed process duration; this settled amount is not the final total. No old experiment budget is borrowed.

The single CPU tail was actually started with `bash tasks/flow-kernel-20261027/run_cpu.sh --wait --allow-partial`. Its current receipt is `runs/flow_kernel_20261027/cpu_wait.json` (PID 1282650, waiting_for_generation_terminal), log `cpu_tail.log`; final products go to `runs/flow_kernel_20261027/evaluation/`. It performs available-pair checks, imports only actual raw outputs, scores independent endpoint distributions separately from trajectories, evaluates compatible same-origin endpoints, and exports the existing raw viewer. Missing draws remain NA; parent L4096 results are labeled read-only reuse and never replace a missing new draw.

The existing CPU adapter smoke checks passed without model calls or CUDA: D/G identity on a real MD snippet, same-MD motion sanity, all evaluator module syntax, and partial parent-only import/export. The compatibility cross-ensemble distance excludes equal-seed pairs (four draws give 12 cross pairs, excluding 4); the finite unbiased energy-distance estimate may be negative. The maximum predeclared request remains 800 jobs / 2760 new draws in the fixed priority order above; actual completed coverage and remaining budget, not the request count, determine the final result.


2026-10-28 CPU preview outputs: `runs/flow_kernel_20261027/evaluation_preview_trajectories_fixed_20261028/` (priorities2/3, joint+C256+native-delta rollout) and `evaluation_preview_endpoints_20261028/` (priorities1/4, t0=12 endpoint and predeclared compatibility/history-cross diagnostics). These are partial snapshots, separate from final `evaluation/`; the original erroneous `evaluation_preview_trajectories_20261028/` remains failed evidence and must not be presented as new-model results.


## 2026-10-28 completed core preview (not final study)

The unchanged motion and temporal functions scored all 132 new joint/delta-rollout trajectories, 12 reused parent joint trajectories and 18 protocol-matched MD references; all 162 rows are ok. The real 1whz/C256 smoke reproduces the complete scorer motion values exactly. Each row below averages two seeds within a case, then the three cases within the stated dataset/clock/history/protocol. CA amplitude is mean CA RMSF divided by its matching MD value; ACF error is CA vector ACF RMSE.

| Dataset / step | Protocol | Arm | CA amplitude / MD | CA ACF error | G1 | Mean absolute G1 error to MD |
|---|---|---|---:|---:|---:|---:|
| ATLAS / 100 ps | joint | MD | 1.00000 | 0.00000 | 0.59588 | 0.00000 |
| ATLAS / 100 ps | joint | parent_L4096 | 0.60612 | 0.44379 | 0.05231 | 0.54357 |
| ATLAS / 100 ps | joint | control256 | 0.62457 | 0.39485 | 0.10509 | 0.49079 |
| ATLAS / 100 ps | joint | control | 0.61287 | 0.44133 | 0.05574 | 0.54014 |
| ATLAS / 100 ps | joint | h1 | 0.63796 | 0.42966 | 0.07011 | 0.52577 |
| ATLAS / 100 ps | joint | history_mix | 0.69150 | 0.42433 | 0.07306 | 0.52282 |
| ATLAS / 100 ps | joint | dense_history | 0.66944 | 0.42684 | 0.07055 | 0.52533 |
| ATLAS / 100 ps | joint | lag1 | 0.70723 | 0.42830 | 0.06837 | 0.52751 |
| ATLAS / 100 ps | rollout_h4_lag1 | MD | 1.00000 | 0.00000 | 0.59588 | 0.00000 |
| ATLAS / 100 ps | rollout_h4_lag1 | control | 13.66121 | 0.28114 | 0.85950 | 0.26362 |
| ATLAS / 100 ps | rollout_h4_lag1 | history_mix | 1.12765 | 0.19132 | 0.70943 | 0.13393 |
| ATLAS / 100 ps | rollout_h4_lag1 | dense_history | 1.12964 | 0.18932 | 0.70908 | 0.12564 |
| ATLAS / 100 ps | rollout_h4_lag1 | lag1 | 1.14658 | 0.18676 | 0.70938 | 0.12653 |
| ATLAS / 100 ps | rollout_h1_lag1 | MD | 1.00000 | 0.00000 | 0.59588 | 0.00000 |
| ATLAS / 100 ps | rollout_h1_lag1 | h1 | 2.36732 | 0.35732 | 0.92277 | 0.32689 |
| MISATO / 80 ps | joint | MD | 1.00000 | 0.00000 | 0.51320 | 0.00000 |
| MISATO / 80 ps | joint | parent_L4096 | 0.65685 | 0.39363 | 0.04390 | 0.46931 |
| MISATO / 80 ps | joint | control256 | 0.67389 | 0.37070 | 0.07070 | 0.44250 |
| MISATO / 80 ps | joint | control | 0.66757 | 0.39247 | 0.04456 | 0.46864 |
| MISATO / 80 ps | joint | h1 | 0.67093 | 0.38919 | 0.04899 | 0.46421 |
| MISATO / 80 ps | joint | history_mix | 0.71911 | 0.38413 | 0.05266 | 0.46055 |
| MISATO / 80 ps | joint | dense_history | 0.70057 | 0.38585 | 0.05096 | 0.46225 |
| MISATO / 80 ps | joint | lag1 | 0.74085 | 0.38454 | 0.05206 | 0.46114 |
| MISATO / 80 ps | rollout_h4_lag1 | MD | 1.00000 | 0.00000 | 0.51320 | 0.00000 |
| MISATO / 80 ps | rollout_h4_lag1 | control | 11.71588 | 0.36772 | 0.87500 | 0.36180 |
| MISATO / 80 ps | rollout_h4_lag1 | history_mix | 1.12929 | 0.23216 | 0.69175 | 0.17854 |
| MISATO / 80 ps | rollout_h4_lag1 | dense_history | 1.13958 | 0.23945 | 0.70036 | 0.18715 |
| MISATO / 80 ps | rollout_h4_lag1 | lag1 | 1.18578 | 0.24183 | 0.70724 | 0.19403 |
| MISATO / 80 ps | rollout_h1_lag1 | MD | 1.00000 | 0.00000 | 0.51320 | 0.00000 |
| MISATO / 80 ps | rollout_h1_lag1 | h1 | 2.46705 | 0.43027 | 0.93378 | 0.42057 |

The mixed-task H4 rollouts have much smaller amplitude error than the joint-only control used as a Q1 kernel, but long-lag correlation remains too high. Hmix G4/G8 are 0.61555/0.41708 versus MD 0.34322/0.17338 for ATLAS100ps, and 0.62970/0.42947 versus 0.28933/0.10987 for MISATO80ps. The large-motion 6e5y Hmix rollout still has CA amplitude 0.60481 and fixed-MD flexible-region amplitude 0.52396; a near-MD G1 does not resolve that deficit. Joint generation improves slightly against C512 on amplitude and ACF, but C256 has lower ACF error than the mixed arms. C256 differs in optimizer/LR position; neither comparison alone isolates task conflict. H1-native rollout is separately labeled because inference history also changes. No control H1 rollout was generated; the predeclared H1/H4 cross in 6e5y and 4J8T is being scored separately.

Independent endpoints: 180 complete groups x 4 draws = 720 draws, errors empty, currently only origin12 (one of three predeclared origins). The table below is H4 and the shortest target lag, not joint trajectory geometry. Energy score uses one held-out MD future per condition; lower is better but it does not reveal the complete true conditional distribution.

| Dataset / target lag | Arm | CA energy score (A) | Internal-distance energy score (A) | Protein bad bonds (%) | Protein bad angles (%) | Ligand bad bonds (%) | Ligand bad angles (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| ATLAS / 100 ps | control | 1.16443 | 0.82594 | 21.24900 | 16.66012 | NA | NA |
| ATLAS / 100 ps | h1 | 0.99513 | 0.62078 | 2.79286 | 2.43091 | NA | NA |
| ATLAS / 100 ps | history_mix | 0.99071 | 0.61021 | 1.26085 | 1.20852 | NA | NA |
| ATLAS / 100 ps | dense_history | 0.98613 | 0.61080 | 1.28203 | 1.23190 | NA | NA |
| ATLAS / 100 ps | lag1 | 0.96579 | 0.60637 | 1.33871 | 1.18066 | NA | NA |
| MISATO / 80 ps | control | 0.86269 | 0.69714 | 16.31851 | 13.95465 | 25.73495 | 39.55975 |
| MISATO / 80 ps | h1 | 0.59617 | 0.49605 | 0.64361 | 0.78760 | 12.99769 | 19.00869 |
| MISATO / 80 ps | history_mix | 0.59673 | 0.50075 | 0.20652 | 0.33033 | 8.85995 | 11.84561 |
| MISATO / 80 ps | dense_history | 0.59620 | 0.50073 | 0.19166 | 0.30925 | 8.13657 | 12.38282 |
| MISATO / 80 ps | lag1 | 0.59170 | 0.49808 | 0.22410 | 0.32229 | 8.36806 | 11.42895 |

Actual products: [motion/G preview](../../runs/flow_kernel_20261027/evaluation_preview_trajectories_fixed_20261028/motion_G_preview.json), [first-origin endpoint scores](../../runs/flow_kernel_20261027/evaluation_preview_endpoints_20261028/endpoints/summary.json), and [raw preview player](../../runs/flow_kernel_20261027/evaluation_preview_trajectories_fixed_20261028/preview_viewer/index.html). The player contains original trajectories and MD; its geometry scoring is explicitly not yet complete. Full common/geometry scoring and compatibility are continuing in the two independent CPU previews, while the original GPU list and final CPU tail continue. The latest settled cost is about 21.48 A100 process-hours plus the currently running job, within the unchanged 48-hour ceiling. The remaining high-NFE Q12/Q4/CF jobs are roughly 10–12 hours at observed shared-GPU throughput, not a guaranteed ETA. No new experiment or retry has been started for this preview.
