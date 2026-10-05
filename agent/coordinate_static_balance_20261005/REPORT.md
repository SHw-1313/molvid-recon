# Train-only coordinate static-balance probe — completed

2026-10-05 17:26, sourceed4e0ba. Independent branch from5885eab.
`runs/coordinate_static_balance_20261005/probe_v1` contains identity/protocol,
all96 gradient-comparison rows, execution evidence and summary.

Actual GPU6 FP32 work:16 training systems,32 dynamic +96 static forward/
backwards,**80.803s**,peak28,706,312,704bytes. **0 optimizer updates**; no
optimizer instantiated; entire model state bitwise unchanged after execution.
Static temporal/history/frozen-teacher boundaries passed.11 tests passed in
3.90s, including a real CUDA synthetic-model backward/context boundary test.
These tests are not counted as trained-system effectiveness experiments.

|Static corruption σ,Å|DiT mean cosine|Negative pairs/32|Median static/dynamic gradient norm|Readout median norm ratio|
|---|---:|---:|---:|---:|
|.05|−.41734|32|4.15063|7.03841|
|.25|−.36340|31|3.47800|7.20013|
|.50|−.24204|26|3.07820|10.28756|

Static-active DiT subspace cosines agree closely (−.41782,−.36384,−.24234),
so the sign is not explained by including dynamic-only temporal parameters
in the norm. Readout mean cosines are near zero (.01092,.00501,.01933), with
16/32 negative at each σ; very large norm ratios still make unit weighting
unsafe. This establishes **local gradient conflict/imbalance at this frozen
checkpoint**, not proof that all static pretraining is harmful or that a
particular balancing method will improve trajectories.

Important limitations:one dynamic-trained seed71 checkpoint;16 of cached32
training systems;static examples are independent other-system MD snapshots;
different conditioning/source distributions and untrained static route.
No external static structures, no complex data, no training, no generated
validation, no convergence or multi-seed claim. Row count32 is not32 independent
proteins:16 systems×2 flow times. No development metric used to select σ/weight.

Decision:reject naive equal-weight static+dynamic loss. Before a joint run,
register dynamic-only continuation and an explicitly budgeted mixed arm with
the same dynamic exposure. Consider a separate static velocity readout while
retaining the shared spatial trunk, so conflicting tasks need not share the
same output mapping; this is a hypothesis, **not implemented or trained**.
Do not introduce gradient surgery without a matched control. Parent coordinate
final48 evaluation must still finish; this gradient probe is not evidence
that its forecasts already beat A1/PVB/BioKinema.
