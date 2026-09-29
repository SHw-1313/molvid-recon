# Static decoder probe v1 decisions

## 2026-09-29

- Base implementation on A's published common-only release.  Its CUDA status
  remains pending and it is not treated as an A recipe/capacity release.
- Add a dedicated `T=1` preparation path.  Keep
  `prepare_frame_joint_batch`'s requirement for at least one history and one
  future frame unchanged.
- Treat coordinate noise sigma as per-axis Angstrom; record the corresponding
  isotropic three-dimensional RMS displacement (`sqrt(3) * sigma`).
- Rebuild every coordinate-derived input from the perturbed coordinates.  A
  static batch stores clean coordinates only in an explicitly named target
  field that is never passed to the decoder.
- Use coordinate MSE and clean-target covalent bond-length loss initially.
  The common release has no validated differentiable clash implementation, so
  clash is disabled and reported, not reimplemented with an unreviewed rule.
- Do not claim or run the external-static arm without an authorized compatible
  data path and overlap exclusions.

