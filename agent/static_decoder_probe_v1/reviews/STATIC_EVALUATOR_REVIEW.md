# Held-out static evaluator independent review

## Scope and immutable identity

- Parent: `e64bd525d87a85e3c53a6288cce061808b455210`
- Candidate: `8792b17b6b8bd0044c2c25a3cd7d541d59684abd`
- Reviewed files only:
  - `tools/evaluate_static_decoder.py`
  - `tests/test_static_decoder_evaluator.py`
- Candidate evaluator SHA-256:
  `0abbba218627b337895934da7ceff9002866d829bf72f949bd239da30caa8392`
- Candidate test SHA-256:
  `9dfa584c82f5983ede40cb3f610f546c9c697c647b68816f2ea71d786ba2de7c`

This was a read-only independent review. No model/evaluator code was edited and
no sealed-test trajectory was opened.

## Findings

### HIGH — the frozen 8 x 3 x 4 protocol was not enforced

At candidate lines `59-67`, arbitrary sigma sets were accepted; line `163`
accepted an arbitrary anchor window; lines `70-98` accepted an arbitrary
nonempty systems/replicas cross-product. Lines `313-315` derived the expected
row count from those inputs instead of requiring 24 selected records and 96
rows. The candidate test at lines `13-42` used a 2-system x 2-replica manifest
and treated it as valid.

Required repair: require anchor window 0, exact sigmas
`(0.0, 0.05, 0.10, 0.20)`, 8 unique systems, replicas R1-R3, 24 unique
system/replica sources, and an exact 96-row 8 x 3 x 4 grid.

### HIGH — validation-only provenance was fail-open

At candidate lines `203-206`, the declared validation-store index hash was
checked only when the field was present. A manifest could omit
`source_valid_index_sha256`, point the CLI at an arbitrary store, and still
produce a protocol that unconditionally claimed `split=valid` and
`test_opened=false` at lines `230-231`. The test manifest omitted the store
identity and passed.

Required repair: require and verify the declared validation-store hash and
path, derive the split from the verified source contract, and fail closed on
the already-approved validation manifest/store identities.

### MEDIUM — the scientific input identities were recorded but not frozen

Candidate lines `184-191` accepted arbitrary recipe and static-loss config
files. Lines `215-219` recorded hashes after loading but did not require the
approved hashes or cross-check the gamma calibration's auxiliary-recipe hash.
Thus a wrong physical recipe or loss configuration could complete with a
plausible, self-described protocol.

Required repair: freeze and verify the manifest, store index, released
physical recipe, shared static config and model config identities, and verify
the gamma recipe link.

### MEDIUM — tests did not exercise the emitted numerical path

The five tests covered selection helpers, seed stability, sigma validation and
aggregation, but not exact 96-row coverage, fail-closed identities, middle
frame conversion, clean-target isolation, CUDA execution or artifact output.
A bounded real CUDA preflight remained required after repair.

## Correct behavior already present

The candidate did correctly use the middle valid frame and the public T=1
static conversion, kept the clean coordinates in the target boundary, derived
record/frame/sigma-local noise seeds, invoked the shared static training loss
and released physical clash implementation, and aggregated replica means
before a system-equal mean. These correct pieces did not close the fail-open
protocol findings above.

## Verification

- `git diff --check` passed for the two-file candidate delta.
- In `enter-container` / `torch-ito`, with bytecode and pytest-cache writes
  disabled: `5 passed in 5.72s`.
- The intended validation metadata was independently checked: the ordinary
  paired-view manifest contained 8 systems, replicas R1-R3, and 24 unique
  anchor-window-0 source IDs; it declared `test_opened=false` and the matching
  valid-store index hash. The issue was that the candidate did not require
  these frozen facts.

## Verdict

**FIX_REQUIRED**
