# Targeted review of evaluator commit `e64bd52`

Verdict: **PASS** for evaluator commit `e64bd525d87a85e3c53a6288cce061808b455210`.

This approval is limited to the evaluator delta from parent `07b6ea76a1460eb37f85154cb3770983ad64c490`.

## Severity-ranked findings

### BLOCKER / HIGH / MEDIUM

None.

### LOW — focused tests do not exercise CLI integration

The five tests correctly cover path construction and rejection of empty, negative and duplicate seed lists (`tests/test_gm_a_evaluator.py:8-29`), but they do not execute argparse defaults, protocol identity, resume rejection, or final row-count validation.

Static review finds those implementations correct, and the planned CUDA CLI preflights will close this integration gap. This is not a code-fix requirement.

### LOW — one record is a smoke, not a precise timing model

A `--limit 1` invocation includes checkpoint/model loading and only one record size/history/lag. Its wall time should not be extrapolated directly to the full manifests. It is sufficient as a functional and coverage smoke when combined with the historical full-evaluation ledger and continued stage-level GPU-hour accounting.

## Findings

### Seed contract — pass

`--seeds` accepts one or more integers and defaults to legacy `[0,1,2]` behavior (`tools/evaluate_gm_a.py:49-55`). `_evaluation_paths` always retains persistence and oracle, appends exactly one generated path per requested seed, and rejects empty, negative and duplicate lists (`tools/evaluate_gm_a.py:35-46`).

For `--seeds 0`, the exact paths are:

1. `persistence`, seed `null`;
2. `oracle`, seed `null`;
3. `generated`, seed `0`.

Omitting `--seeds` retains the previous five-path ordering with generated seeds 0, 1 and 2. Existing aggregation accepts a single generated draw without assuming three draws (`tools/recompute_frame_gm_metrics_v3.py:125-168`).

### Protocol identity and resume safety — pass

The ordered seed list is serialized directly into `protocol.json` (`tools/evaluate_gm_a.py:63-66`). Resume requires exact equality of the complete protocol, so changing the seed list, its order, limit, checkpoint, recipe, store or manifest rejects reuse of an existing output (`tools/evaluate_gm_a.py:67-73`).

Legacy three-seed outputs remain resumable because omission of `--seeds` recreates the old `[0,1,2]` protocol value.

A timing preflight must use a separate scratch output directory. Its protocol contains `limit=1`, so it intentionally cannot be resumed as the unlimited formal evaluation.

### Path and row coverage — pass

Every selected record iterates over the resolved path tuple, and row keys retain path and seed identity (`tools/evaluate_gm_a.py:78-110`). Final coverage is now correctly:

`record_count × (2 + generated_seed_count)`

through `len(ids) * len(paths)` (`tools/evaluate_gm_a.py:112-113`). Thus one seed yields exactly three rows per record and the legacy default yields five.

Persistence and oracle are not removed or replicated per generated seed. System/replica/anchor/H/lag coverage continues to come from the unchanged complete manifest (`tools/evaluate_gm_a.py:74-85`).

### Output traceability — pass

The summary retains checkpoint identity, protocol SHA256, record count, row count, measured invocation GPU-hours and `test_opened=false` (`tools/evaluate_gm_a.py:114-119`). The protocol itself binds seed order, Euler steps, precision, store and manifest hashes.

### Scope integrity — pass

The commit changes only:

- `tools/evaluate_gm_a.py`;
- `tests/test_gm_a_evaluator.py`.

It cannot alter trained checkpoints. Independently verified hashes are:

- Evaluator: `b0060e7990921cb0fba55dfb63db7f819a76ed6e1dab2eec49cda658c3d2f188`.
- Test: `95cf9adcbadfe87fe90657d0098adc22385feb95ff2c20e55f6ad06fe9fd0e67`.

I independently reran the focused tests inside `enter-container` with `torch-ito`: **5 passed in 3.73 seconds**.

## CUDA preflight decision

`--limit 1 --seeds 0` is a sufficient minimum functional gate before the full evaluations, with these conditions:

- Run the exact limited command for each of the three checkpoints against both ordinary and fixed-history stores—six small preflight invocations—not merely once against one arm/view.
- Use six fresh scratch output directories, separate from the formal output directories.
- For every preflight, verify:
  - protocol seed list is exactly `[0]`;
  - `limit` is `1`;
  - checkpoint, recipe, store and manifest hashes are correct;
  - exactly three unique rows exist;
  - paths are persistence, oracle and generated seed 0;
  - summary reports `records=1`, `rows=3`, the matching protocol SHA and `test_opened=false`;
  - coordinate/profile artifacts and reported metrics are finite;
  - measured GPU time is recorded.
- Treat timing as a sanity check, not a direct one-record linear extrapolation. Continue cumulative accounting after each full stage and stop optional diagnostics before the eight-GPU-hour cap.

The full runs must use fresh outputs without `--limit`, the exact common `--seeds 0` argument, and unchanged complete validation manifests. With one generated draw, reports must retain the previously recorded limitation that between-generation-seed variance is not estimable.

## Final decision

**PASS.**

Commit `e64bd525d87a85e3c53a6288cce061808b455210` correctly implements the explicit single-seed evaluation contract while preserving legacy defaults, fixed reference paths, strict resume identity and exact row coverage. No sealed-test trajectory was opened during this review.
