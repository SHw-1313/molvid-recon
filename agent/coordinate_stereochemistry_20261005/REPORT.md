# Coordinate stereochemistry — implementation and execution evidence

Independent branch `exp/coordinate-stereochemistry-20261005`, parentdfa04f2.
Added `molvid/losses/stereochemistry.py`: topology-checked observed Cα centers,
detached reference sign/volume, system-equal valid-frame margin loss, no target
coordinates or dynamics-amplitude penalty. Existing trained runtime files are
unchanged. New helper does not alter inference until explicitly used by training.

5 targeted tests passed in2.70s via `enter-container`/`torch-ito`, including a
real CUDA loss/backward fixture. Covered mirrored structure correction gradient,
proper rigid motion, zero penalty for positive large volume, detached reference,
missing bonds, masked centers, packed sample-equal reduction and NaN padding.
This is not protein training or evidence of efficacy. A complete geometry
collapse may have zero determinant gradient; the soft term is no hard guarantee.

`tools.probe_coordinate_stereochemistry` uses fixed trained coordinate71 weights,
8 training proteins, H4/Q2,4-step differentiable true-source trajectories and
16-step no-grad reference. Original-runtime hashes, checkpoint SHA and final
metadata are verified before GPU work. It measures auxiliary/RF gradients,
does zero optimizer updates and requires unchanged full model state afterward.
Real data execution is next, not yet a completed experiment at this commit.

First real `probe_v1` terminated with CUDA OOM before any complete system row:
20GiB process cap,peak21,006,126,592 allocated bytes. GPU itself had59GiB free;
this was the deliberately bounded allocation, not another user's workload.
Execution failure/telemetry retained,0 updates,no efficacy result. Four-step
actual-source gradients retained spatial-encoder activations that block-only
DiT checkpointing cannot eliminate. Added full velocity-field non-reentrant
recomputation in a separate training helper; no time-step detach/truncation.
Original100 trained runtime file hashes still unchanged. Initial6 targeted
CPU/CUDA tests passed4.70s, including exact forward and numerical full parameter/
source-gradient parity against original Euler4, frozen teacher boundary and
nonzero history gradients. Registration otherwise unchanged for `probe_v2`.

## Real Euler4 diagnostic completed,18:47

Expanded7 CPU/CUDA tests passed5.14s, including both source-requires-grad and
ordinary no-source-grad cases. `probe_v2` real CUDA completed8 training proteins,
16 RF backwards +8 true-source auxiliary backwards,**55.212s**,peak8,924,907,008B,
**0 optimizer updates; full model state bitwise unchanged**. Source265aef9.

Of16 pairings,12 have nonzero auxiliary gradients (2/8 systems already satisfy
the margin). DiT mean cosine over nonzero pairs+.322821,4/12 negative;
readout+.228629,3/12 negative. Median auxiliary/RF gradient norm ratios including
zeros are.008201/.017242, max.150178/.159430 for DiT/readout respectively.
Not equivalent to static mixing's mostly opposing gradients, but this local
measurement does not prove a safe loss weight or forecast improvement.

Equal-system averages, fixed Q2:

|Path|Margin loss|CA flip fraction|Near-planar fraction|
|---|---:|---:|---:|
|Original Cartesian source|6.824804|.408028|.034770|
|CFM interpolation, two times|.985044|.207331|.046556|
|CFM one-step endpoint estimate|.000241|.002805|.002961|
|Actual-source Euler4|.001159|.004989|.003418|
|Actual-source Euler16|.003062|.010804|.008722|
|Training MD target|0|0|0|

Source corruption/interpolation deliberately leaves the stereochemical manifold;
penalizing noisy interpolation coordinates directly would fight the RF path.
Endpoint estimates also understate actual sampled violations in this small
diagnostic. **Euler16 has more errors than Euler4**, so do not infer that adding
more Euler steps fixes geometry or that a4-step auxiliary matches16-step error.
Follow-up `probe_v3` registers actual16-step gradients on the identical inputs.
Neither probe is a trained candidate or a validation/kinetic result.
