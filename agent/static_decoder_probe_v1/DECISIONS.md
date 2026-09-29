# Static decoder probe v1 decisions

## 2026-09-29

- Start from A's common release, then absorb the reviewed A stable source only
  after its immutable evaluated A1 recipe release was published.
- Add a dedicated `T=1` preparation path.  Keep
  `prepare_frame_joint_batch`'s requirement for at least one history and one
  future frame unchanged.
- Treat coordinate noise sigma as per-axis Angstrom; record the corresponding
  isotropic three-dimensional RMS displacement (`sqrt(3) * sigma`).
- Rebuild every coordinate-derived input from the perturbed coordinates.  A
  static batch stores clean coordinates only in an explicitly named target
  field that is never passed to the decoder.
- Use coordinate MSE, clean-target covalent bond length at relative weight 0.1,
  and A1's reviewed topology-aware clash at its released weight.  Do not invent
  a second clash rule.
- Calibrate one shared static gamma from the released A1 parent using the full
  scheduled decoder-gradient objective, including an actual-source sampled
  clash update, and the same 20% clean mixture used in training.
- Reuse the full dynamic AdamW so decoder moments evolve naturally, but make
  every static step fail closed if any inactive parameter or pre-existing
  inactive optimizer tensor/version changes.  Static steps do not advance the
  dynamic step, LR schedule, scheduler, data cursor, or RNG streams.
- Select exactly one midpoint frame from the first indexed train clip for each
  of 192 systems x 3 replicas.  The independent schedule and RNG state are
  checkpointed rank-locally; formal static training is restricted to one CUDA
  rank until a synchronized multi-rank implementation exists.
- Do not claim or run the external-static arm without an authorized compatible
  data path plus direct-overlap and near-homology exclusions.  The visible PDB
  stores fail this metadata gate, so their existence is not eligibility.
