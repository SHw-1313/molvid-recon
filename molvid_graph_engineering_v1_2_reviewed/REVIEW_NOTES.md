# Review corrections from v1 to v1.2

The first bundle was re-read in full. This reviewed version fixes the following ambiguities:

1. Makes the full-graph engineering repair mandatory and the halo partition an optional second phase.
2. Replaces partial YAML overlays with complete runnable configurations.
3. Clarifies that partitioning still builds the full neighbor list and targets message-passing activation memory.
4. Corrects the partition acceptance language: induced subgraphs need complete core receptive fields, not identical per-call edge lists.
5. Adds a spatial-subgraph microbatch budget.
6. Separates canonical CPU topology cache from bounded per-device materialization.
7. Clarifies that only bond codes may be sorted; full geometric edges may not be sorted/coalesced.
8. Adds explicit BF16 tolerances and a post-repair throughput gate.
9. Adds sampler `set_epoch`, no-replacement exact acceptance epochs, warmup ordering, and resume-state checks.
10. Adds one-time trusted-store integrity checks rather than simply disabling validation.

11. Adds a controlled `topology` versus `distance_only` graph-input bond-source ablation.
12. Defines distance-only bonds from one deterministic canonical reference per topology (`0.5 Å < d <= 2.2 Å`), reuses them across all clips/replicas of that topology, and keeps topology-based bond supervision unchanged.
13. Adds a complete distance-only config and bond-overlap/connectivity diagnostics.
