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
