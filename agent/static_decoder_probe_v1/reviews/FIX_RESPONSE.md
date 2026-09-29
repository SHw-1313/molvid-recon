# Fix response to independent review of `73ba705`

Status: code responses implemented and focused checks pass.  Formal comparison
training remains gated on released-A1 CUDA calibration/smoke and fresh
independent re-review of the fixed commit.

## Responses

1. Added released-A1 continuation orchestration with one shared dynamic schedule,
   explicit 4:1 static cadence, independent and resumable static data/noise/model
   RNG state, exposure/cost manifests, and identical model-only warm starts.
2. Changed gamma calibration to use released A1 and the complete scheduled dynamic
   objective, including actual-source sampled clash on its released cadence;
   use the same 20% clean mixture as training and record per-term/zero-gradient
   diagnostics.
3. Marked the existing external PDB store ineligible unless existing authorized
   metadata proves direct-overlap and near-homology exclusion against frozen A
   validation/test identities. Do not synthesize the third arm.
4. Made full-optimizer reuse fail-closed by checking inactive parameters and
   all pre-existing inactive optimizer state across every static step; add a
   post-dynamic-moment CUDA test and keep scheduler advancement disabled.
5. Added an explicit interrupted-run record, corrected peak/live memory naming,
   update task documents, and commit the review request/report/fix response.

## Focused verification before fixed-candidate commit

- `pytest -q tests/test_static_schedule.py tests/test_gm_common_schedule.py`:
  8 passed.
- `pytest -q tests/test_frame_joint_contracts.py`: 19 passed.
- `CUDA_VISIBLE_DEVICES=1 pytest -q tests/test_static_decoder.py`: 4 passed.
- Python compilation passed for all changed implementation/CLI/tool/test files.
- Sealed test trajectories opened: no.

The A1 gamma/smoke artifact, its hashes, and the fixed/reviewed commit identity
will be appended after the code is committed and CUDA evidence completes.
