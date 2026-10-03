# Static decoder probe v1 handoff

Status: **complete bounded pilot; no release promotion**.

- Worktree/branch: `/data4/users/sihao/workspace/molvid-recon-static-decoder-probe-v1`,
  `feat/static-decoder-probe-v1`.
- Training implementation commit: `07b6ea76a1460eb37f85154cb3770983ad64c490`.
  Dynamic evaluator: `e64bd525d87a85e3c53a6288cce061808b455210`.
  Final held-out evaluator: `abd8f7bbc7fc9047692aa72796aaa14a888937f6`.
- Released-A1 parent SHA256: `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.
  Gamma is `0.00017325410836320802`; its calibration SHA256 is
  `2e09ed560c50d5938ffce1b8fe6c2080d001972fba79400154b2a9a966351478`.
- Formal checkpoints: baseline `f4d0ccd956a...`, internal MD `90e80ba76d9...`,
  external PDB `9004d3e02ae...`. Each has 500 matched dynamic updates; the
  auxiliary arms have 125 static updates.
- External data: 3,000 fully supervised 303--998-atom records from the processed
  PDB train store derived from `/data5/PVB/pdb`. Eligibility SHA256 is
  `971aee0b385fa50e28f35d63a3890adfe3d61734814a909510367841781e1e69`.
  Exact split overlaps are excluded. Strict near-homology filtering was
  explicitly waived for this version and remains a limitation.
- Dynamic result: internal MD reduces ordinary clashes 8.38% and severity
  10.11% with small bond/motion costs. External PDB does not show consistent
  ordinary geometry improvement; it gives small fixed-history bond/MSD gains.
- Static result: external PDB lowers held-out output coordinate MSE 7.75--12.40%
  versus internal MD across clean and three perturbation levels. Neither arm is
  a net denoiser relative to raw perturbed input coordinates.
- Dynamic artifacts: `runs/static_decoder_probe_v1/comparison_seed0/`; static
  artifacts: `runs/static_decoder_probe_v1/static_eval_formal/`. All validation
  runs record `test_opened=false`.
- Independent review chain is complete: implementation initial FIX_REQUIRED,
  fixed PASS, 500-step delta PASS, dynamic evaluator PASS, static evaluator
  initial FIX_REQUIRED, fixed PASS, legacy-record compatibility fix PASS.
- Known measured GPU time is about 4.21 hours; conservative all-in accounting is
  4.73 hours, below the 8-hour cap. A/B/C were not stopped, restarted or
  modified.

Read `report.md` first. Machine-readable identities, loss ledger, training
manifest, commands, diagnostics and costs are in `artifacts/`.
