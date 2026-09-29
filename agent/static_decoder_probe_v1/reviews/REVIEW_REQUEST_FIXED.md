# Static decoder final-candidate independent re-review request

## Immutable subject and launch gate

- Candidate commit: `07b6ea76a1460eb37f85154cb3770983ad64c490`.
- Prior reviewed commit/report: `73ba705ab4e506e3274d54b62c9a351e84127125`
  / `reviews/REVIEW.md` (`FIX_REQUIRED`).
- Specification:
  `/data4/users/sihao/workspace/SESSION_D_STATIC_DECODER_PILOT_PROMPT_260929.md`.
- Reviewer is read-only and must not edit model code or start formal training.
- Requested verdict: explicit `PASS` or `FIX_REQUIRED` for the frozen three-arm
  1,000-step released-A1 screen. Preserve severity-ranked findings.

## Released and candidate identities

- A recipe release: `38702552fd847f79cf5a6e9c3f0d4b2a104ef7c590316ac6c5395ec2c94626f2`.
- A1 parent: `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.
- A1 config: `15d55300b0bbab366f8ddd984b5c8f94b6ee94b2f6ad1d0264506e3018db1cce`.
- A auxiliary: `625d11cff4a579e96487029f58d88e8e3d6129c3123e80847cc6edb9c8a0e6f3`.
- Teacher: `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
- Dynamic data / normalization:
  `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1` /
  `54bbdc198373ae9f477bf5803643aab77f960fe4498da8db1fcf9c5bfdd770c0`.
- Baseline/internal/external config SHA256:
  `b6370b09aa70e390bc9d56b3459ab360eb3900e3868aeaf0735169bbbcb38ee4`,
  `2ab78281a306f18c2fa6041c46a789e89b48c4b699818a677d1e7a43d2c1a4ae`,
  `ef990ae8f06fc1f8cc8cabf36897a5441eee4dcf1ca6e4eda99f345d87441148`.
- Resolved experiment SHA256:
  `49bfdecd6162e9e752b0e7d106afbb48bc6d795ced0733968b47d9c091cc6718`.

## CUDA and numerical evidence

- Directory: `runs/static_decoder_probe_v1/a1_smoke_final_07b6ea7/`.
- Result/gamma SHA256:
  `e76611870ebe1bd7c7d8db6f242ee80492a0fb36a4b7b98fb98b0ce47778ca79` /
  `2e09ed560c50d5938ffce1b8fe6c2080d001972fba79400154b2a9a966351478`.
- Gamma `0.00017325410836320802`; nine A1 train batches; 0.075 median
  static/dynamic decoder-gradient ratio; actual-source Euler8/K2 clash covered;
  clean probability 0.20; raw/weighted/zero-gradient diagnostics recorded.
- FP32/BF16 target-mutation, disabled-static output, inactive parameter, and
  teacher-state deltas are zero. Largest 4,975-atom update peak is 802,621,440
  allocated bytes. Final focused tests 29 passed; GPU1 tests 4 passed.

## Data and exposure evidence

- Internal: 62 A train systems x R1/R2/R3 = 186 midpoint structures,
  303--1,000 atoms, clean target only.
- External eligibility:
  `agent/static_decoder_probe_v1/artifacts/external_pdb_3000_eligibility.json`,
  SHA256 `971aee0b385fa50e28f35d63a3890adfe3d61734814a909510367841781e1e69`.
- External store index/data SHA256:
  `01075c0c3c3b9600f42537282538231ae87fc572808b69ed18f7283d6587f732` /
  `4bc777a52643e2d34dac1d9e0b5af66ae52a85657d1303660c7fb617ebcc2513`.
- 3,000 unique 303--998-atom records; all atoms supervised; atom/block types,
  shapes, bonds and topology indices independently audited compatible.
- Exact PDB-chain overlap with A train/valid/sealed-test manifest IDs excluded;
  test payloads not opened. Strict sequence/structure near-homology filtering is
  explicitly waived by the user for this bounded version and remains a stated
  limitation, not an inferred exclusion.
- Shared target: 250 updates, 1,475 +/- 7 samples, 1,056,299 +/- 1,399 total and
  supervised atoms. Planned internal/external: 1,468/1,482 samples and
  1,057,698/1,054,901 atoms.

## Formal-path preflight evidence

Directory: `runs/static_decoder_probe_v1/preflight_07b6ea7/`.

- All arms reached step 8 with execution hash
  `abbae4b784aaef17578e9e36fd423fc5bf563f769acc387c2b040a3735849bbf`.
- Peak allocated bytes: baseline 47,998,322,176; internal 47,993,743,872;
  external 48,003,507,200.
- Step-8 checkpoint SHA256: baseline
  `860756e1abbd4548704493579ee2684a605aacf11b06dc724d0560b98bbf5451`,
  internal `637484644e65b1c62b2a7a044b02e0b99451053293311964071fa886d207d9a7`,
  external `15d7f28178fd45324e5f5fc70a34ab27402e43748814a13549d86539ddf9eb60`.
- Internal step-4 checkpoint SHA256
  `968b2c98226cd91b41c978ca7fa0dbb0c7ee2c405f4c3ebc19c60dc6d2564b57`
  resumed with its required SHA and restored schedule/RNG/exposure to two static
  updates at step 8.
- External runtime revalidated all manifest/store identities and executed two
  real PDB static updates.

Please return the exact final report text for verbatim preservation as
`reviews/REVIEW_FIXED.md`. Do not treat the explicitly authorized homology
waiver as proof of homology exclusion or as an unstated blocker.
