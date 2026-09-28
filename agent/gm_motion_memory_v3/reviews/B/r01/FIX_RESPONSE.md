# Response to B-R01-01

Reviewed source: `43bbadf0491c8b3815be03ae1f3ad0114484227a`. Independent review source: `REVIEW.md`, preserved verbatim. Reviewer session `/root/review_b_r01` was spawned with `fork_turns=none`.

Removed `key.bias` and final `label_key` bias because a shared additive key offset cancels under softmax. Updated CUDA test and pilot gates: every participating parameter must have a finite gradient; nonzero norms are checked over meaningful light encoder/key/physical-label/slot/value/readout paths instead of every tensor. The CUDA test checks actual light-layer and query-slot optimizer updates and now covers FP32 and BF16.

No numerical verification is claimed. Python syntax compilation passed under enter-container / torch-ito; B remains unassigned in resources.json. Required next verification is the focused CUDA backward test plus one real warm-start trainer step, followed by the bounded pilot.

Also addressed nonblocking traceability/profile notes: launcher requires a fixed `--code-commit`, rejects dirty source, records config/code identity, verifies the data card hash, records GPU cost even on exceptions, and enforces the aggregate 8 GPUh pilot cap. `--profile-only` uses the median/largest bank inputs across Th9/33/129 and dt400, records teacher observed/target, graph+packing, backbone, light encoding, analysis, memory/readout, DiT, decoder and profiler backward range. These are implemented profile paths, not measured results; warm repeats and mixed-dt packed profiling still remain.

Formal comparison and numerical PASS remain withheld. A fresh focused static re-review of the repair is requested; post-pilot review remains mandatory.
