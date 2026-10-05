# Coordinate static/dynamic balance — train-only diagnostic

Independent branch from coordinate5885eab. No active model source is edited.
Use fixed seed71 final2048 SHA
`ac89685606c058ff9afba8e0364a06fea693eb278c3d77ae80229510f6c46085`.
This is NOT another static training result; no optimizer is instantiated.

Select16 distinct systems by the fixed hash order in the existing train32
cache, one record per system, never open validation or sealed targets.
Dynamic history alternates H4/H8, all remaining query frames, physical100ps.
For each system use two flow times .25/.75, observed-last + iid1Å coordinate
source with fixed RNG. Static examples come from a different selected training
system at cyclic offset7, frame7; corrupt by iid .05/.25/.5Å per axis.
Use identical static Gaussian directions across corruption levels/flow times.
Static target is clean XYZ only on the interpolation/loss side; context is
always corrupted XYZ/centroids, no clean structure condition.

32 dynamic and96 static real CUDA forward/backwards, FP32/eval mode, no updates.
Report per-module shared-DiT/readout gradient norms and cosines, full-module and
static-active subspace separately; histories/time must have no static gradient.
Verify checkpoint/runtime SHA before execution and entire model state bitwise
unchanged afterwards. Two-hour GPU6 budget, no cross-GPU interference.

Questions: Is the static path substantially stronger than dynamic CFM? Is it
locally conflicting across systems? Does corruption scale alter that conflict?
Do not infer forecast improvement from cosine or use development results to
choose a static weight. No gradient projection/surgery is implemented. Any
subsequent mixed-training arm needs a separately frozen loss/compute budget,
a dynamic-only continuation control, two training seeds, and joint forecast
geometry/motion evaluation. This probe uses internal MD snapshots, not external
static-data expansion, and only one trained seed. It cannot establish a global
optimal mixture or replace a proper joint training experiment.
