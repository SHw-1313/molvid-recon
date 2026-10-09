# Final pretrained-geometry-bridge evaluation records

These completed final-checkpoint evaluation reports were produced in the separate `molvid-complex-validation64-20261007` worktree and copied verbatim here so the model branch carries its own final evidence.

Source worktree:
`/data4/users/sihao/workspace/molvid-complex-validation64-20261007`

The local generated-coordinate summaries and evaluation outputs for the pretrained bridge live in this branch under:
`runs/pretrained_geometry_bridge_20261007/evaluation_seed71/`
`runs/pretrained_geometry_bridge_20261007/evaluation_seed72/`

## Included records

- `bridge72_final16384_joint_result.json`: seed72, fixed 16,384 update final checkpoint; final protein/complex forecasts, ligand chemistry, and internal/CA path comparisons.
- `final_two_seed_and_time71_result.json`: fixed-final base and time71 comparisons, including the seed71 learning-curve/follow-up evidence.
- `final_two_seed_clock_result.json`: matched-history 80/100 ps clock-response evaluation.
- `bridge72_final64_result.json`: seed72 final 64-system complex evaluation; its measurements are useful, while its decision/status text predates the completed aggregate reports.
- `generalization_result.json`: completed two-seed base and physical-time-FiLM 64-system evaluation, including old16/new48/all64 and metadata strata.

The reports are unchanged copies of their source JSON. Their embedded source paths describe the originating evaluation worktree. For current status, use the reports above together with the completed summaries in this branch; do not use stale “inference pending” text from an earlier status snapshot.

## Final scientific reading

The 16,384-update pretrained bridge learns motion beyond static persistence, but it does not reach joint geometry-and-motion quality against released PVB. Across the completed validation64 panel, both base seeds retain worse ligand bond error than PVB on new48. Whole-path superiority is not stable; contact occupancy improves, with Brier improvement less consistent. Seed72 final protein results show lower MSD/RMSF errors than its fixed 11,264 checkpoint but worse bonds, angles, clashes, and torsion increments. Internal ligand clashes are especially high in seed72. Therefore the final checkpoint is evidence of useful motion learning with geometric costs, not a promotion candidate.

All comparisons reuse development systems, use marginal system bootstrap intervals, and do not establish sealed-test generalization, equilibrium thermodynamics, or kinetic rates.
