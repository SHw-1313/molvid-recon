# Generated C-alpha -> residue detail, registered pilot

Parent db5a797; independent branch exp/ca-detail-20261005. No default model
change, sealed test access, external baseline mutation, or Mamba transplant.

Question: can fixed generated CA motion support better all-heavy-atom geometry,
and how much of remaining error disappears with real future CA (oracle only)?
This is NOT a newly trained CA generator or complete hierarchical pretraining.
Frozen A1 full-atom inference supplies CA, then all generated non-CA coordinates
are discarded by the detail predictor. Only the unmodified A1 baseline scores
its full output. The detail network receives observed coordinates/topology,
query physical times and declared CA condition; no future latent or atom target.

Data: deterministic hash-selected 32 existing train systems, R1 two separated
windows each, H4/100ps and future Q12/100ps. Existing valid8 systems, R1 two
separated windows each, source seeds17/29. Train CA source seed11. Exact source
schedule, timestamps and hashes saved before inference. All atoms, not the
64-atom U/K cache. No tuned new holdout or unseen-family claim.

Geometry: residue-index CA mapping verified against atom identity and observed
bpos; reject unsupported ligands/missing CA explicitly. Start with rigid
translation of last-observed residue offsets to each conditioned CA. A small
rotation/translation-equivariant covalent+atom/CA message network predicts
detail refinements, with CA hard-clamped at every layer. Chemical types are
fixed. Same interface for observed structures across datasets; initial pilot
is explicitly protein-only and does not claim complex support.

Controls: original A1; persistence; rigid detail transport under generated CA
and oracle CA; learned recovery under generated CA and oracle CA. Two training
arms: clean-CA only vs alternate clean/generated CA, matched initialization,
sample order and 512 updates, seeds71/72, same optimizer/LR. In generated-CA
training, residue-relative target offsets (true atom minus true CA) are loss
targets only; this avoids a constant uncorrectable CA translation term. Bond
loss is measured on actual predicted coordinates, so bad generated CA cannot
be hidden by residue-wise metric alignment. No future-condition leakage.

Loss fixed before results: non-CA detail coordinate MSE + covalent bond-length
MSE, weights1/1 (A^2); no post-hoc weight selection. Width64, three message
layers, AdamW3e-4, weight_decay1e-4, clip1. FP32 CUDA. Train cap1 GPUh per arm,
cache+evaluation cap2 GPUh total, <=2GiB artifacts and >=6GiB disk reserve.
All final checkpoints scored; no validation checkpoint selection.

Report system-equal raw coordinate/detail/CA error, bond/angle/torsion/clash,
CA and all-atom MSD/RMSF, paired system bootstrap separately per training seed.
Stochastic generated pointwise error is descriptive, not distribution matching.
Success needs geometry improvement without simply suppressing motion. True CA
is an assisted bound, never called a deployable result. If oracle helps but
generated CA does not, prioritize coarse motion/geometry; if neither helps,
do not expand the hierarchy from this evidence. Training on source-generated
CA is exposure adaptation, not proof of end-to-end generator improvement.

Proof of execution: immutable source/input manifest; GPU model/UUID/PID/UTC;
nvidia-smi periodic log; synchronized CUDA event time, peak allocated bytes,
per-step loss/gradient and optimizer step, initial/final parameter comparison,
checkpoint SHA, strict reload/held-out metric reproduction. Prior-run audit
checks old artifacts without inventing retroactive GPU utilization telemetry.
