# Held-out static evaluator independent re-review

## Scope and immutable identity

- Parent: `8792b17b6b8bd0044c2c25a3cd7d541d59684abd`
- Fixed candidate: `080511eeb0372409fa3cc09ba9a2a4a1fc429739`
- Reviewed files only:
  - `tools/evaluate_static_decoder.py`
  - `tests/test_static_decoder_evaluator.py`
- Evaluator SHA-256:
  `cce7232189f9f8ae1d6da7d891811c3c162ff81fec7ada7fed2de4880869a64e`
- Test SHA-256:
  `520037d59cf954b16bf1434d979cb7b6bfe9f8d811f6baa51dd5620aa35726fc`

This was a read-only independent delta review. No model/evaluator code was
edited and no sealed-test trajectory was opened.

## Repair verification

The fixed evaluator:

- freezes sigmas `(0.0, 0.05, 0.10, 0.20)`, anchor window 0, 8 unique
  systems, replicas R1-R3, 24 unique selected source IDs and 96 exact grid
  rows (`tools/evaluate_static_decoder.py:50-60`, `71-138`, `193-216`);
- checks the exact manifest, valid-store index, released physical recipe,
  shared static config and model config hashes before CUDA/model loading
  (`239-247`);
- requires the manifest's validation index identity and declared valid-store
  path, requires the opened path to match, and derives `split=valid` from that
  verified path (`282-293`);
- verifies the gamma calibration's auxiliary-recipe SHA against the frozen
  released recipe (`264-268`);
- preserves the previously reviewed middle-frame conversion, clean-target
  boundary, record-local perturbation, shared static loss/clash,
  system-equal aggregation and CUDA-only execution.

The hard-coded identities matched the live approved files:

- manifest: `af91cd50aa2db1a6501ea0a1fa522a421a276b446f434b73f5a41c906bd066cd`
- valid-store index: `c89f870d9bf087ee7bf983613012210af35c30199309dd1e63c0496887eef71e`
- physical recipe: `625d11cff4a579e96487029f58d88e8e3d6129c3123e80847cc6edb9c8a0e6f3`
- shared static config: `be6fefec137aca42e65858f286353c164c6ce70a706f8b1e1adffd2350e09f39`
- model config: `7c67c5e3da8e4cb29da5560c4c8ddcb53683ae9463357a6b077848ea552a79df`

The manifest itself contained 8 systems, replicas R1-R3, 48 selected records
across the two declared windows, and exactly 24 unique window-0 source IDs,
with `test_opened=false`.

## Verification

- `git diff --check` passed.
- In `enter-container` / `torch-ito`, with bytecode and pytest-cache writes
  disabled: `8 passed in 6.02s`.
- Added negative coverage rejected missing/wrong validation identities,
  nonzero anchor selection, wrong sigma sets, missing/duplicate grid rows and
  duplicate row keys.

## Verdict at fixed-code gate

**PASS**

One exact GPU1 validation invocation remained required before launching all
three diagnostic arms. A later preflight against the approved legacy valid
store exposed that those serialized records omit the optional `frame_mask`
field. The `080511e` preflight wrote only
`baseline_failed_missing_frame_mask_080511e/protocol.json` (protocol SHA-256
`ba5ff7c379d960edf2e9ceae20f05171d8bfd4d7438a05456313c3e930ef83c0`)
and produced no rows or summary. Therefore `080511e` was a valid repair of the
reviewed fail-open issues but was not the final runtime-compatible launch
identity; the compatibility-only follow-up is reviewed separately.
