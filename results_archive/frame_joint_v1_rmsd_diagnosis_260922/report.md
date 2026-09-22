# Frame Joint v1 RMSD diagnosis (2026-09-22)

This report compares persistence, clean-latent decoder oracle, and generated sampling on the fixed 48-system pilot valid-quick windows (8 systems, 24 trajectories, two windows each; H4/H8; 100 ps spacing; Euler 16; seed 0). All model inference and tensor metrics ran on CUDA; summaries and plots were produced on CPU.

## Protocol and provenance

- Pilot checkpoint: `runs/frame_joint_v1_48_pilot_260920/frame_joint_step_00013222.pt`; SHA `e615b36a600d56b69e004305695fa804509ecb5591d2dcb10643a63d698759fb`.
- 192 capacity checkpoint: `runs/frame_joint_v1_192_newB_260921/frame_joint_step_00111573.pt`; SHA `40eac969d2f33870a4b6fe333b9230283cf237c46f571b8d0598b095cdfd69c5`; training stage is the joint stage; this is the latest stable checkpoint selected for this report.
- Frozen codec: `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`; SHA `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`. Reference valid-quick JSON SHA: `ba79a2272f530642007db30a6a0b2ba9787dc993520dcbf9766e2e3c07072561`.
- Training status: 192 was active while the evaluation started; final read-only check found no 192 training PID and `train_metrics.jsonl` last recorded step 115625. This evaluation did not send a signal or restart it, and the disappearance is not diagnosed here.
- Latent statistics were checkpoint-matched: pilot `runs/frame_joint_v1_48/frame_statistics.pt` (file SHA `2a0093ba1e85188b69b1bb0e3f450d9e278b3ab5ed71ee48de8431d5f8c9abfa`, statistics hash `161b2de504b6bb0fdb3e1429746f2ec359b2f019cb15522a9b126da50ceff576`); 192 `runs/frame_joint_v1_192/frame_statistics.pt` (file SHA `3ce928ad96ee79096872b719f3dc07a64c0d5c0322835ada7318ef21805d1c85`, statistics hash `5f8e85833a89acd4a09b8d715b00824ad644b9654fb97075e394dbd7dd149fe7`).
- Pilot valid index SHA: `81769f7311dcd432e00f81e83930e4d8f35523c5d3a9ff81764634f77cbd6a51`; 192 valid index SHA: `ef1971a28b4c715371aa20f109c3cb1be9dff1506cc66fe05d84c9c010ff264f`.
- Coordinates are Å. Aligned RMSD uses per-frame Kabsch on `align_mask`, then RMSD/dRMSD/bond metrics use the evaluator loss-mask intersection. The outputs record loss-mask and align-mask atom counts per window; representative windows have 1199 loss atoms, 612 align atoms, 612 intersection atoms.
- Aggregation is window mean, then replica mean, then equal system mean. Persistence dynamic/RMSF correlations are unavailable when its predicted motion has zero variance and are kept null.

## Aggregate future metrics

| checkpoint | H | path | aligned RMSD | dRMSD | bond RMSE | contact F1 | RMSF AE | dynamic corr | velocity RMSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pilot_48 | H4 | Clean oracle | 0.0645 | 0.0564 | 0.0367 | 0.9831 | 0.0007 | 0.9997 | 0.0002 |
| pilot_48 | H4 | Generated | 2.3136 | 1.6901 | 0.5435 | 0.7656 | 0.2663 | 0.0009 | 0.0118 |
| pilot_48 | H4 | Persistence | 2.1546 | 1.5074 | 0.0417 | 0.8707 | 1.0779 | NA | 0.0088 |
| pilot_48 | H8 | Clean oracle | 0.0644 | 0.0562 | 0.0367 | 0.9830 | 0.0007 | 0.9997 | 0.0002 |
| pilot_48 | H8 | Generated | 2.2289 | 1.6378 | 0.5976 | 0.7543 | 0.2031 | 0.0002 | 0.0121 |
| pilot_48 | H8 | Persistence | 1.9878 | 1.3902 | 0.0416 | 0.8748 | 0.9842 | NA | 0.0087 |
| capacity_192_step111573 | H4 | Clean oracle | 0.0254 | 0.0235 | 0.0145 | 0.9935 | 0.0003 | 0.9999 | 0.0001 |
| capacity_192_step111573 | H4 | Generated | 2.3897 | 1.7781 | 0.3535 | 0.7620 | 0.1914 | 0.0030 | 0.0125 |
| capacity_192_step111573 | H4 | Persistence | 2.1546 | 1.5074 | 0.0417 | 0.8707 | 1.0779 | NA | 0.0088 |
| capacity_192_step111573 | H8 | Clean oracle | 0.0254 | 0.0235 | 0.0145 | 0.9935 | 0.0003 | 0.9999 | 0.0001 |
| capacity_192_step111573 | H8 | Generated | 2.2802 | 1.6891 | 0.3643 | 0.7610 | 0.1758 | -0.0052 | 0.0123 |
| capacity_192_step111573 | H8 | Persistence | 1.9878 | 1.3902 | 0.0416 | 0.8748 | 0.9842 | NA | 0.0087 |

## First and last future frame

| checkpoint | H | path | first distance / RMSD / bond | last distance / RMSD / bond |
|---|---:|---|---|---|
| pilot_48 | H4 | Persistence | +100 ps / 1.5192 / 0.0417 | +1200 ps / 2.4905 / 0.0418 |
| pilot_48 | H4 | Clean oracle | +100 ps / 0.0648 / 0.0369 | +1200 ps / 0.0646 / 0.0369 |
| pilot_48 | H4 | Generated | +100 ps / 1.7493 / 0.5100 | +1200 ps / 2.6811 / 0.5905 |
| pilot_48 | H8 | Persistence | +100 ps / 1.5088 / 0.0417 | +800 ps / 2.2417 / 0.0416 |
| pilot_48 | H8 | Clean oracle | +100 ps / 0.0644 / 0.0366 | +800 ps / 0.0646 / 0.0369 |
| pilot_48 | H8 | Generated | +100 ps / 1.8031 / 0.5653 | +800 ps / 2.4837 / 0.6258 |
| capacity_192_step111573 | H4 | Persistence | +100 ps / 1.5192 / 0.0417 | +1200 ps / 2.4905 / 0.0418 |
| capacity_192_step111573 | H4 | Clean oracle | +100 ps / 0.0253 / 0.0144 | +1200 ps / 0.0255 / 0.0146 |
| capacity_192_step111573 | H4 | Generated | +100 ps / 1.7970 / 0.3259 | +1200 ps / 2.7274 / 0.3865 |
| capacity_192_step111573 | H8 | Persistence | +100 ps / 1.5088 / 0.0417 | +800 ps / 2.2417 / 0.0416 |
| capacity_192_step111573 | H8 | Clean oracle | +100 ps / 0.0252 / 0.0144 | +800 ps / 0.0255 / 0.0146 |
| capacity_192_step111573 | H8 | Generated | +100 ps / 1.8233 / 0.3465 | +800 ps / 2.5655 / 0.3826 |

## System heterogeneity

The complete per-system table is in `system_summary.csv`. On the pilot, the largest generated-minus-persistence aligned-RMSD gaps are `atlas_3l4h_A` (+0.294 Å H4, +0.350 Å H8), `atlas_3f0o_B` (+0.229 Å H4, +0.296 Å H8), and H8 `atlas_5ef9_A` (+0.327 Å). The pilot H4 `atlas_2ejn_B` mean is slightly below persistence (−0.013 Å), but its generated bond RMSE remains ~0.531 Å. On the latest 192 checkpoint, every system is worse than persistence; the largest gaps are `atlas_3l4h_A` (+0.326 Å H4), `atlas_2ejn_B` (+0.502 Å H8), and `atlas_5ii7_A` (+0.249/+0.342 Å H4/H8).

## Endpoint diagnostic (future-informed; not generation performance)

The endpoint inputs intentionally contain the clean future through the training interpolation. The same source/noise seed is reused across s values within each window. The `(1-s)` endpoint factor means raw endpoint error must not be compared across s without accounting for this factor; velocity MSE and normalized endpoint latent MSE are shown together.

| H | s | endpoint RMSD | endpoint bond | flow h MSE | flow v MSE | normalized endpoint h MSE | normalized endpoint v MSE | clean oracle RMSD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H4 | 0.10 | 2.1100 | 0.5920 | 0.0758 | 0.0350 | 0.0614 | 0.0284 | 0.0645 |
| H4 | 0.50 | 1.0659 | 0.3254 | 0.1214 | 0.0337 | 0.0303 | 0.0084 | 0.0645 |
| H4 | 0.95 | 0.1564 | 0.0970 | 0.7218 | 0.3106 | 0.0018 | 0.0008 | 0.0645 |
| H8 | 0.10 | 1.9021 | 0.6006 | 0.0701 | 0.0329 | 0.0568 | 0.0267 | 0.0644 |
| H8 | 0.50 | 1.0269 | 0.3405 | 0.1160 | 0.0346 | 0.0290 | 0.0087 | 0.0644 |
| H8 | 0.95 | 0.1535 | 0.0986 | 0.7288 | 0.3341 | 0.0018 | 0.0008 | 0.0644 |

## Conclusions

- The generated path is worse than simply copying the last observed coordinates on the aggregate aligned-RMSD criterion for both H4 and H8. It is already worse at the first +100 ps frame (pilot H4: 1.749 vs 1.519 Å; H8: 1.803 vs 1.509 Å) and drifts further by the last frame. Its bond error is also much larger from the first frame (pilot H4: 0.510 vs 0.042 Å; H8: 0.565 vs 0.042 Å), so the structural and bond failures appear together rather than being only a rigid-body/dynamics issue.
- The clean-latent decoder oracle is near the reconstruction floor (pilot ~0.064 Å aligned RMSD and ~0.0367 Å bond RMSE; 192 ~0.025 Å and ~0.0146 Å), with contact F1 above 0.98 and dynamic correlation ~0.9997–0.9999. This rules out the frozen decoder/codec as the dominant explanation for the generated 2.2–2.4 Å error on this protocol.
- Endpoint diagnostics show a large reduction in coordinate/bond error as s approaches 0.95, but the normalized endpoint latent and velocity MSE expose the expected `(1-s)` scaling and do not establish that the model only handles low-noise denoising. The full sampler still fails badly, so source-to-target flow accuracy and/or trajectory integration remains the leading localization, not a decoder-only floor.
- The 192 checkpoint improves bond RMSE substantially (pilot 0.544/0.598 Å to 0.354/0.364 Å for H4/H8) and improves the clean oracle, but does not improve generated aligned RMSD (H4 2.390, H8 2.280). Thus scaling changes bond fidelity without solving the generated coordinate drift. This is an observational comparison, not a causal scaling experiment.
- Persistence has low bond error by construction but misses physical motion; its RMSF/dynamic correlations are null where predicted variance is zero. Generated dynamic correlations near zero are not, by themselves, proof of no learned dynamics; they accompany the large geometric and bond errors here.
- Most supported next experiment: diagnose/train the source-to-target latent flow and endpoint integration (including calibrated latent/velocity targets and sampler stability) while keeping the measured decoder oracle as a fixed acceptance gate. Do not treat this report as authorization to change training in this evaluation-only turn.

## Files

- Raw pilot: `runs/frame_joint_v1_rmsd_diagnosis_260922/pilot_48_full/diagnosis.json` and `coordinates.npz`.
- Raw 192: `runs/frame_joint_v1_rmsd_diagnosis_260922/192_step111573_full/diagnosis.json` and `coordinates.npz`.
- Endpoint diagnostic: `runs/frame_joint_v1_rmsd_diagnosis_260922/pilot_endpoint/diagnosis.json` and `coordinates.npz`.
- Exact commands: `commands.txt`.
- CSVs and plots are in this archive directory.
