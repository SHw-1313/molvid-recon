# P2 r04 fix response

All r04 blockers were addressed in the P2 parent selector and its focused tests.

- R1a: each formal runtime `resolved_config.json` is compared exactly, excluding
  only the explicit `resolved` section, with the hash-bound reviewed config.
  Complete exposure and sampler manifests must also be identical across all four
  formal arms.
- R1b: metric, protocol, and current paired-store `index.txt` hashes must agree.
  The paired-manifest file hash is recomputed, `test_opened=false` is required,
  and legacy versus fixed-history family contracts are validated explicitly.
  Actual paired-data identities are retained in selection provenance.
- R2: per-view RMSF target keys now include `legacy` or `fixed`, so all twelve
  views per system remain distinct during cross-arm pairing checks.
- R4: focused tests now cover store replacement, sealed-test manifests, distinct
  legacy/fixed H4 targets, exact runtime config binding, and symmetric formal
  sampler/exposure artifacts.

Verification in the required container and `torch-ito` environment:

```text
PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py tests/test_frame_gm_distribution.py
17 passed in 3.02s
```

Only `tests/test_frame_gm_selection.py` is relevant to the P2 review. The
distribution test was run in the same invocation but belongs to uncommitted P3
work and is outside the requested review scope.
