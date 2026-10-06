# Geometry activation recomputation — isolated execution experiment

Parent db7ee37; independent branch exp/query-geometry-recompute-20261006.
The already running/queued source trees remain unchanged. This is not a
new scientific model, accuracy gain, capacity increase, or default promotion.

## Motivation and boundaries

The trainable current-coordinate geometry encoder retains its per-frame
TorchMD intermediate activations even when the downstream DiT blocks are
checkpointed. Optional non-reentrant frame recomputation trades work for
activation storage. Keep every atom, all 16 frames, H4/H8, FP32, dynamic radius
graphs, teacher freezing, observed-only origin, model weights and RF losses.
Default remains off. Model contracts record the option only when enabled,
preserving the default parent contract and state keys. Observations remain
no-grad, and current-coordinate gradients must survive frozen/trainable paths.

## Gates before any adoption

1. CPU toy tests cover multi-chunk metadata, padding, parameter/input gradients,
   stochastic recomputation RNG, state keys and contracts. These are not
   CPU execution of the protein model and do not qualify CUDA memory.
2. On exclusively handed-off GPU7, five targeted CUDA tests: four real
   TorchMD graph/gradient parity cases and one dynamic-rollout/static boundary
   test. No skips accepted. Coordinate/parameter equality remains exact.
3. Same registered TRAIN atlas_3da8_A 16×1578 input, off/on, same real parent,
   seed and two full-model RF updates. Compare all model/Adam/RNG digests;
   measure actual peak allocation and elapsed times, not inferred estimates.
   Each execution also cold-resumes its saved first step to verify step 2.
4. Actual maximum TRAIN atlas_6sup_A 16×17002, trainable query, H4 and H8,
   with recomputation: two updates plus cold resume per case. Do not crop,
   shorten or change precision to pass. Failed artifacts remain failures.

If all four real-input cases pass: 12 discarded qualification updates, zero
formal training and zero protein forecasts. No efficacy claim is possible.
The 1578-atom parity test does not independently prove 17002-atom equivalence
if the ordinary baseline cannot execute; disclose that limit.

## Resource handoff

The original maximum-input queue must finish all four registered cases after
long8_72 finishes training AND 192 forecasts AND stereochemistry. This queue
follows that queue's terminal evidence and exact process identity; it must
not claim a transient phase gap. GPU0–3 remain outside the authorized lanes.
Both source and immutable inputs are rehashed before each child. Reserve
9GiB on data4; do not silently remove artifacts to force a launch.

## Initial test correction

First CPU run: 10 passed, 2 failed. The negative fixture expected the encoder
padding error, but without an explicit origin the earlier centroid validator
correctly rejects frame zero. Added separate checks for both existing errors;
no runtime validation changed or tolerance relaxed. Corrected run: 16 passed,
7 CUDA tests explicitly deselected (12 new CPU tests + 4 prior GQ tests).
Actual CUDA and real protein updates remain not run at this checkpoint.

Follow-up: 43 CPU tests passed in 3.37s; the `not cuda` name selector also
deselected four CPU XML-parser tests. The six queue tests were then run without
that selector (6 passed in 2.39s), giving 47 distinct CPU tests passed, not 49.
Representative actual TRAIN record inspected with zero model calls:
atlas_3da8_A_R1_dt_100ps_w000000, shape (16,1578,3), full record digest
2c20cc751daff627efde034cae246d4485585de6f4379e7787cc3434f9b6dce9.
