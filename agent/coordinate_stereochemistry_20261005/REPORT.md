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
