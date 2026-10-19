# Flow A future attention: chemical-unit Q/K

2026-10-20 authorized implementation and paired study. Worktree branch research/flow-groupqk-20261020 starts at 2f50820. Only future query-attention routing changes: masked mean scalar features by sample/component/protein residue before Q/K; atom h/v Values and residual outputs remain. Ligand routing is per atom because its block may cover a whole flexible ligand. Default atom behavior is preserved.

Same static RF parent /data1/sihao/molvid-derived/coordinate-static-pretrain-v1/seed71/flow/static/coordinate_final.pt; atom/group each seed71, 512 updates, lr1e-4/warmup128, original 128 ATLAS+128 MISATO and H4/Q12 or H8/Q8, unchanged loss/source/35-NFE sampler. Paired initialization and data/noise exposure will be checked. This starts from static pretraining; historical 2048-step A plus continuation results are not a matched control.

Budget is the explicitly reassigned old RF remainder: 21600-7274.841=14325.159 seconds; hard cap14325 seconds for all CUDA checks, training, 24 Q50 draws and retries. EDM low-noise remains independent. Run outputs go to runs/flow_groupqk_20261020. CUDA checks passed:4 tests,26.513 full process seconds. Actual source check used TRAIN3M35 H4/Q12,35NFE,1629 protein valid atoms,260 groups and37 singleton ligand routes. GPU6 was checked idle on host/container and now runs atom training; first hostPID1884702, step1 loss0.67689425,4-second update. Availability is rechecked before each next command.

Model worker owns bridge_history.py, flow/coordinate_state.py and focused tests; scoring worker owns the two thin task adapters. Engineer owns trainer plumbing, configs and the end-to-end run. Results will include backbone and sidechain geometry, clashes and motion protection (CA amplitude/fixed-flex RMSF/CA-vector and PCA ACF, especially atlas_6e5y_A), plus original raw viewer.

Verified parent: step2048, 2048 static examples, no dynamic examples; both query output maps are exactly zero. Initial upstream Q/K gradients may therefore be zero; the focused fixture opens its own test maps to check reachable gradients without changing real initialization.

Actual block check: TRAIN 3M35 has 1629 protein heavy atoms in223 residue blocks and37 ligand atoms in1 block; DEV 4J8T has1046 protein atoms in128 residue blocks and28 ligand atoms in1 block. Protein blocks match residues; ligand blocks do not justify shared routing. Detailed source identity is recorded in the run checks.

Execution uses the existing trainer, with only the already-established operational --stop-after option copied into this independent branch. Both recipes remain updates512 when saving/loading at32. The paired check verifies data/noise records and final generator/global RNG/mixture/exposure states; no new objective or scheduler.
