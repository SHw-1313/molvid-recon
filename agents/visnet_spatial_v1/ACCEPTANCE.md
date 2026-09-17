# ACCEPTANCE — ViSNet spatial backbone v1

## A. Implementation gates

All are required:

- existing TorchMD configurations continue to work without adding `spatial_backbone`;
- `visnet_radius` and `visnet_bonded` output `[T,N,C]` scalar and `[T,N,3,C]` vector features;
- native ViSNet makes no call to the external graph builder;
- no spatial edge crosses sample or frame boundaries;
- scalar output is translation/rotation invariant within test tolerance;
- vector output is translation invariant and rotation equivariant within test tolerance;
- coordinate and model-parameter gradients are finite and nonzero;
- changing binary edge type changes bonded ViSNet output;
- static `T=1` and dynamic `T=16` both work;
- checkpoints save and resume with resolved backbone and graph-mode metadata;
- no NaN, Inf, OOM, CPU graph fallback, or dependency installation.

## B. Three-clip micro-overfit gate

For each backend:

- 500 optimizer steps complete;
- final total loss is less than 70% of initial total loss;
- all principal spatial, temporal, and decoder modules receive finite gradients;
- checkpoint resumes for at least one further update.

Failure blocks the 441/117 run for that backend.

## C. Final tiny-overfit protocol gate

The manifest must contain exactly:

```text
3 systems
9 trajectories
441 train clips
117 late-holdout clips
16 frames per clip
dt_100ps
0 train/holdout sample overlap
```

Every epoch:

- each train sample appears exactly once;
- no sample is missing or duplicated;
- all 117 holdout clips are evaluated;
- metrics are aggregated per system in addition to clip averages.

## D. Tiny-overfit quality gate

A ViSNet backend is a viable next-stage candidate when:

- train total loss decreases by at least 60% from its untrained initialization;
- late-holdout future dRMSD improves by at least 10% from its untrained initialization;
- no dynamic/geometric metric exhibits a catastrophic regression;
- training and resume are stable.

This gate does not require ViSNet to defeat TorchMD.

## E. Reporting gate

The final report must separate:

1. implementation correctness;
2. micro-overfit result;
3. final tiny-overfit result;
4. quality/throughput/memory tradeoff;
5. development-backend recommendation;
6. limitations: no protein isolation, one seed, natural rather than parameter-matched models.

Do not call this experiment a final scientific benchmark.
