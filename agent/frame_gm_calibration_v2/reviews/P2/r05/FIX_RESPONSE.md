# P2 r05 fix response

The remaining R4 coverage blocker is closed by a file-level four-arm fixture.

- It constructs complete legacy (8 views/system) and fixed-history
  (4 views/system) metric/protocol/store/manifest artifacts for all four arms.
- The passing case invokes the production CLI `main()`, which traverses formal
  provenance, `load_arm_values()`, cross-arm protocol/target pairing, bootstrap,
  and selection output.
- The failing case changes only G's legacy 100 ps/H4 target while leaving its
  fixed 100 ps/H4 target unchanged, updates the generated files and hashes, and
  verifies that the production entry point fails with
  `per-view RMSF targets are not paired for G`.
- A separate case changes only GM's sampler manifest and verifies the symmetric
  formal sampler/exposure guard.

Required-environment verification:

```text
PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py
14 passed in 0.33s
```
