# Targeted evaluator review request

Review commit `e64bd525d87a85e3c53a6288cce061808b455210` read-only against parent
`07b6ea76a1460eb37f85154cb3770983ad64c490`.

Scope is deliberately limited to `tools/evaluate_gm_a.py` and
`tests/test_gm_a_evaluator.py`. Training has completed at the reviewed parent;
this commit cannot change any trained checkpoint.

The task budget requires one common generated draw while retaining every
ordinary/fixed-history record, system, replica, anchor, H/lag bucket,
persistence path and oracle path. The intended invocation is `--seeds 0` for
every arm. Default omission of `--seeds` must retain legacy `[0, 1, 2]`
behavior. Empty, negative and duplicate seed sets must fail. Resume identity
must include the explicit seed list, and row coverage must be
`record_count * (2 + generated_seed_count)`.

Evidence:

- evaluator SHA256: `b0060e7990921cb0fba55dfb63db7f819a76ed6e1dab2eec49cda658c3d2f188`
- targeted test SHA256: `95cf9adcbadfe87fe90657d0098adc22385feb95ff2c20e55f6ad06fe9fd0e67`
- `python -m pytest -q tests/test_gm_a_evaluator.py`: 5 passed in 4.43 s,
  inside `enter-container` with `torch-ito` active.

Please return explicit `PASS` or `FIX_REQUIRED` and check protocol identity,
resume safety, exact path/row coverage, preservation of legacy defaults, and
whether a `--limit 1 --seeds 0` CUDA timing preflight is sufficient before six
full ordinary/fixed-history evaluations. Do not edit code or open sealed test
trajectories.
