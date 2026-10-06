# Query geometry × depth × long optimization

Independent branch from e16f48e. Original long4/8 controls are completed, not
retrained; reuse requires exact 16-update replay of model, Adam, RNG and the
entire scientific contract (only runtime file hashes may differ).

Four arms: depth 4/8 × seeds 71/72, trainable copied query geometry with frame
activation recomputation. Frozen observation teacher is unchanged. Same parent
coordinate2048 checkpoints, TRAIN192 schedule [2048:10240], H4/H8, iid 1Å source,
FP32, RF-only loss and 8192-step stretched LR policy as frozen-query controls.
No extra geometry branches, time architecture, static loss or chirality loss.
Query geometry uses the existing history learning rate, not a tuned new rate.

Per arm before formal continuation: 16 default-control updates, 1 largest-query
profile, 17 candidate updates and 1 cold-resume replay. Thus 35 actual qualifying
updates per arm, 140 total. Only the 17-step candidate prefix is retained;
8175 further updates per arm reach fixed8192. GPU 0–3 are independently assigned;
do not stop or use other lanes. Fail any numerical/identity/storage gate closed.

Six actual CUDA tests cover identity expansion and long-policy depth4/8 with
frozen/trainable query geometry, exact model/optimizer/RNG resume, query gradient
ownership and frozen teacher. These small numerical tests are not protein
quality measurements. Real full-model prefixes and profiles are separate gates.

Final-only 48 development systems × two anchors × two draws = 192 forecasts per
arm, plus stereochemistry audit. Report all four arms, including failed gates.
Paired system-level comparisons against matching frozen-query controls and
cross-depth comparisons must include bond/clash/angle/torsion, displacement,
MSD, RMSF profile and amplitude. No promotion for geometry-only improvements,
and lack of significance is not noninferiority. This reused development panel
cannot establish independent competition, thermodynamics or complex validity.
Long versus short changes BOTH update count and LR horizon, not only compute.

Capacity evidence motivating this factorial test: completed frozen-query long
depth8 improves geometry in both seeds but worsens RMSF in seed71; seed72 motion
differences are inconclusive. Therefore blind depth scaling is not accepted.

Preparation is not submission. Actual states, commands, source hashes, GPU
telemetry, losses, checkpoint digests and qualification proofs belong in runs/.
Original PVB/BioKinema sources and weights remain untouched. No sealed targets.
