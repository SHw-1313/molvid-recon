# Fix response to independent review of `73ba705`

Status: all known findings were implemented in immutable training candidate
`07b6ea76a1460eb37f85154cb3770983ad64c490`. The fixed review and uniform
500-step delta review both passed; all three formal arms and their evaluations
are complete.

## Responses

1. Added released-A1 three-arm orchestration with a config-enforced parent for
   baseline and both static arms, one dynamic schedule, 4:1 cadence, independent
   static RNG, exposure manifests, and strict SHA-bound resume.
2. Gamma calibration now uses the complete released-A1 decoder-gradient
   objective, including actual-source sampled clash, the training 20% clean
   mixture, and raw/weighted/zero-gradient diagnostics.
3. Reused the authorized processed PDB train store. Runtime binds index and
   `data.bin` hashes, selection report, source/split, code/data/normalization
   identities, selected count/index hash, and the entire eligibility manifest.
   Exact A split overlap is excluded without opening test trajectories. The
   user's explicit waiver replaces strict near-homology filtering for this
   bounded version and is recorded as a limitation.
4. Made full-optimizer reuse fail closed across inactive parameters and all
   pre-existing inactive optimizer state; scheduler advancement remains off.
5. Added deterministic rank-local static schedule/RNG/exposure checkpoint state,
   required resume SHA256, and verified step-4 to step-8 exact resume.
6. Corrected exposure accounting to include `loss_mask`-supervised atoms. Both
   arms use 303--1,000-atom records, at most six samples per update, and one
   common tight target for sample, total-atom, and effective-atom exposure.
7. Excluded external fragments below the internal pool minimum. All 3,000 PDB
   records are fully supervised; independent audit found compatible atom/block
   types, shapes, topology indices, and unique dataset indices.
8. Updated interrupted-run, peak-memory, data-card, task, config, and review
   identities so no old eligibility or candidate claim is used for launch.

## Final evidence

- Focused tests: 29 passed; final GPU1 CUDA tests: 4 passed; compile checks pass.
- A1 smoke result SHA256:
  `e76611870ebe1bd7c7d8db6f242ee80492a0fb36a4b7b98fb98b0ce47778ca79`.
- Gamma report SHA256:
  `2e09ed560c50d5938ffce1b8fe6c2080d001972fba79400154b2a9a966351478`.
- External eligibility SHA256:
  `971aee0b385fa50e28f35d63a3890adfe3d61734814a909510367841781e1e69`.
- Resolved experiment SHA256:
  `49bfdecd6162e9e752b0e7d106afbb48bc6d795ced0733968b47d9c091cc6718`.
- Config SHA256: baseline `b6370b09...`, internal `2ab78281...`, external
  `ef990ae8...` (full values in `REVIEW_REQUEST_FIXED.md`).
- The reviewed 1,000-step plan was uniformly budget-reduced before observing
  results. Current 500-step planned internal/external exposure is 737/742
  samples and 531,450/523,904 total/effective atoms over 125 static updates.
- Three final 8-step preflights share execution hash `abbae4b...` and peak near
  48.0 GB. Internal strict resume and external real-store updates both pass.
- Sealed test trajectories opened: no.

The later 500-step launch identities are the separately reviewed configs and
resolved experiment recorded in `REVIEW_FIXED_500.md`; the 1,000-step hashes
above remain historical evidence and were not launched.
