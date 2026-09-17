# R4 source checkpoint reassessment v2

Status: `A_SOURCE_REASSESSMENT_V2_COMPLETE`

This is a descriptive `shared_adapter_legacy_diagnostic`; the Gaussian/conditional checkpoint difference is not an independent source ablation.
The frozen R4 codec/statistics, DiT/RF semantics, train/validation manifests, and test-sealed boundary were preserved.

## Inputs and selection

- Code HEAD: `54bd76bd4b2e9bd2f9b938e952fc3694982b5dc0`; required baseline: `5c2754fcce44ed77dad77db09db709408fee7634`.
- Legacy JSONL row check by arm: `True`; per-arm counts `{'conditional': {'main_batch': 52, 'subset_clip': 128}, 'gaussian': {'main_batch': 52, 'subset_clip': 128}}` vs expected `{'main_batch': 52, 'subset_clip': 128}`. Main batch rows were never split into fake systems.
- Validation selection: `8` systems, fixed R1/window30; train selection: `8` sorted systems, one preselected clip each.
- Execution device: host `neibu`, physical GPU0 UUID `GPU-65b7cbe8-a24e-f4c7-975e-7a4077cb3fe5`, evaluation PID `301250`; container-visible device `cuda:0`.
- Test payload opened: `false`.

## Existing JSONL reaggregation

Legacy main rows are reported as batch-row means with `system_equal_not_available`. Legacy subset rows are averaged draw → clip → system, separately for H and Euler steps.

- `conditional`: {'main_batch': 52, 'subset_clip': 128} rows; checkpoint SHA256 `8a3f8803655406468b8d50ca5a316f4a7858be31366f74bc2d897ca96769b12c`.
- `gaussian`: {'main_batch': 52, 'subset_clip': 128} rows; checkpoint SHA256 `545a64f34ad6a2463a0a36e14a53c09f6c1e0fe6105ea6be9e30a34e84582c95`.

## New single-segment results

CUDA contract check: `PASS`.

Generated rows: `768`; baseline rows: `64`; diversity rows: `192`.

| split | arm | H | steps | systems | future aligned RMSD | bond RMSE | contact F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | conditional | 4 | 8 | 8 | 1.6184267929022202 | 0.6637371921793266 | 0.7507156588354946 |
| train | conditional | 4 | 16 | 8 | 1.6673299208431067 | 0.6998382675675985 | 0.739722383838201 |
| train | conditional | 4 | 32 | 8 | 1.7003621302555831 | 0.7244246596881151 | 0.7324818372472841 |
| train | conditional | 8 | 8 | 8 | 1.618518598582153 | 0.6499163566982706 | 0.7541853859492745 |
| train | conditional | 8 | 16 | 8 | 1.6699700580506092 | 0.6851644201778782 | 0.7430471665773716 |
| train | conditional | 8 | 32 | 8 | 1.7039736447984484 | 0.7085767684761366 | 0.7357958696365503 |
| train | gaussian | 4 | 8 | 8 | 1.7611931379669388 | 0.8435507916966565 | 0.7247572232483205 |
| train | gaussian | 4 | 16 | 8 | 1.8189906902820305 | 0.9001546045233428 | 0.7114430004434533 |
| train | gaussian | 4 | 32 | 8 | 1.8556621381047744 | 0.93343150604927 | 0.7034009706443953 |
| train | gaussian | 8 | 8 | 8 | 1.7179194571286522 | 0.791592225372078 | 0.7363322137738452 |
| train | gaussian | 8 | 16 | 8 | 1.7746843308546583 | 0.8435720918255456 | 0.7230653082715963 |
| train | gaussian | 8 | 32 | 8 | 1.8107050435613923 | 0.8737557884281482 | 0.7151606036356568 |
| valid | conditional | 4 | 8 | 8 | 1.7749226237539066 | 0.6753697719162483 | 0.7498474165064476 |
| valid | conditional | 4 | 16 | 8 | 1.822787114743237 | 0.7133854639290885 | 0.7387981988895347 |
| valid | conditional | 4 | 32 | 8 | 1.854845274755455 | 0.7387395598514297 | 0.7317462623240563 |
| valid | conditional | 8 | 8 | 8 | 1.7479300851269217 | 0.6566099350498047 | 0.7534332627442235 |
| valid | conditional | 8 | 16 | 8 | 1.8000632385635171 | 0.6928775209453656 | 0.7421940845506221 |
| valid | conditional | 8 | 32 | 8 | 1.833989834181498 | 0.7173542767017156 | 0.7349233125098232 |
| valid | gaussian | 4 | 8 | 8 | 1.877088226933509 | 0.8614777236797427 | 0.7227694306008584 |
| valid | gaussian | 4 | 16 | 8 | 1.9308335305306181 | 0.9198469669904441 | 0.7095872447166899 |
| valid | gaussian | 4 | 32 | 8 | 1.9650329627339649 | 0.954130017735242 | 0.7016702124713515 |
| valid | gaussian | 8 | 8 | 8 | 1.791871845659038 | 0.8034051018334096 | 0.7349007924593328 |
| valid | gaussian | 8 | 16 | 8 | 1.8458470952889292 | 0.8568146598048225 | 0.7218156383770343 |
| valid | gaussian | 8 | 32 | 8 | 1.8801302266850999 | 0.8879172714640702 | 0.7140341585515207 |

Baseline system-equal summaries (one oracle and one repeat-last row per clip/H):
- `train` H4 `oracle_reconstruction`: RMSD `0.02156595966266508`, bond `0.012099310234623477`, contact F1 `0.9944345216614164`.
- `train` H4 `repeat_last_observed_frame`: RMSD `1.4173951166406944`, bond `0.041601038582423644`, contact F1 `0.8783811949363908`.
- `train` H8 `oracle_reconstruction`: RMSD `0.02156729125859493`, bond `0.01208127843346309`, contact F1 `0.9943865586904812`.
- `train` H8 `repeat_last_observed_frame`: RMSD `1.406207194507346`, bond `0.041581877318882896`, contact F1 `0.8820345140135413`.
- `valid` H4 `oracle_reconstruction`: RMSD `0.021226668296045893`, bond `0.012230301836278943`, contact F1 `0.9944083330777924`.
- `valid` H4 `repeat_last_observed_frame`: RMSD `1.5568792950328632`, bond `0.04170750398780357`, contact F1 `0.8773421710928264`.
- `valid` H8 `oracle_reconstruction`: RMSD `0.02122247870472824`, bond `0.012219552963515783`, contact F1 `0.9944063425768851`.
- `valid` H8 `repeat_last_observed_frame`: RMSD `1.5998190434912611`, bond `0.04184821674033819`, contact F1 `0.8778306207341687`.

Oracle reconstruction and repeat-last-observed-frame baselines were computed once per clip/H. Generated coordinate arrays are reused by all metrics through `predictions.npz`; per-frame bond/RMSD curves are in `per_frame_metrics.jsonl`.

Interpretation checks required by this run:
- Training-system versus validation-system differences must be read from the separate split rows, not inferred from old batch rows.
- H4 first predicted frames are available in the per-frame curve; compare them with later future frames to distinguish first-block failure from rollout growth.
- RMSF ratio, displacement summaries, diversity, and velocity-lag1 ACF are reported separately; near-constant Pearson/ACF signals are null with reasons.
- 8/16/32 steps remain separate groups and share the same epsilon seed for each sample/H/draw across both arms.

## Required scientific answers

1. **Training systems are also degraded: `True`.** At H8/16 steps, train aligned RMSD is `1.670` (conditional) and `1.775` (Gaussian), versus `1.406` for repeat-last. Bond RMSE is `0.685`/`0.844` versus `0.042`. This is not only a validation generalization gap.

2. **The first predicted geometry is already poor, then worsens.** Valid H4/16-step frame and four-frame means:

| arm | first future frame RMSD | first 4 RMSD | last 4 RMSD | first 4 bond RMSE | last 4 bond RMSE | first 4 contact F1 | last 4 contact F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditional | 1.533 | 1.663 | 1.974 | 0.702 | 0.723 | 0.747 | 0.731 |
| gaussian | 1.646 | 1.766 | 2.078 | 0.892 | 0.940 | 0.717 | 0.704 |

3. **Motion amplitude converges toward a common scale across systems: `True`.** Aggregate RMSF alone is misleading; the per-system prediction spread is much narrower than MD and per-system/atom correlations remain limited.

| split | arm | RMSF pred/MD | system RMSF SD pred/MD | system RMSF corr | atom RMSF corr | displacement pred/MD | within-block pred/MD | boundary pred/MD | velocity lag1 ACF pred/MD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | conditional | 0.793/0.730 | 0.025/0.128 | -0.252 | 0.428 | 0.816/0.891 | 0.727/0.891 | 1.348/0.892 | -0.354/-0.372 |
| train | gaussian | 0.811/0.730 | 0.010/0.128 | 0.208 | 0.427 | 0.853/0.891 | 0.766/0.891 | 1.374/0.892 | -0.367/-0.372 |
| valid | conditional | 0.802/0.830 | 0.021/0.172 | 0.267 | 0.499 | 0.827/1.059 | 0.739/1.061 | 1.355/1.045 | -0.355/-0.429 |
| valid | gaussian | 0.818/0.830 | 0.011/0.172 | 0.289 | 0.495 | 0.858/1.059 | 0.770/1.061 | 1.390/1.045 | -0.365/-0.429 |

4. **8/16/32 Euler steps change the judgment: `False`.** On valid H8, increasing steps monotonically worsens geometry for both checkpoints:

| arm | steps | aligned RMSD | bond RMSE | contact F1 | RMSF ratio |
|---|---:|---:|---:|---:|---:|
| conditional | 8 | 1.748 | 0.657 | 0.753 | 0.912 |
| conditional | 16 | 1.800 | 0.693 | 0.742 | 0.999 |
| conditional | 32 | 1.834 | 0.717 | 0.735 | 1.056 |
| gaussian | 8 | 1.792 | 0.803 | 0.735 | 0.924 |
| gaussian | 16 | 1.846 | 0.857 | 0.722 | 1.020 |
| gaussian | 32 | 1.880 | 0.888 | 0.714 | 1.080 |

5. **Generated prefixes amplify error: `True`.** Segment 1 is identical within each arm because both modes start from the same real prefix and epsilon. Relative to true-prefix conditioning:

| arm | segment | Δ aligned RMSD | Δ bond RMSE | Δ contact F1 | Δ RMSF ratio | Δ displacement |
|---|---:|---:|---:|---:|---:|---:|
| conditional | 2 | 0.480 | 0.263 | -0.060 | -0.049 | -0.090 |
| conditional | 3 | 0.577 | 0.511 | -0.112 | -0.052 | -0.096 |
| gaussian | 2 | 0.564 | 0.347 | -0.083 | -0.035 | -0.057 |
| gaussian | 3 | 0.877 | 0.675 | -0.152 | -0.022 | -0.047 |

The generated-prefix error growth is structural rather than a motion explosion: RMSF ratio and mean displacement stay similar or decrease while bond/contact geometry degrades sharply. Future-only draw diversity remains nonzero and is reported separately in `interpretation.json`. Pointwise velocity correlation is not used alone to declare dynamical failure; constant-signal correlations/ACFs remain null with reasons. `frequency_retention` retains its limited total-power-ratio meaning, and absent torsions are not reported as zero.
## Short rollout

Smoke status: `PASS`, systems `2`.
Full rollout status: `PASS`, systems `8`, segment rows `192`.
Rollout records preserve a single inverse-preprocessing gauge, local model time reset with the real dt, and global physical time. True-prefix and generated-prefix rows are separate.
Compare segment 2/3 rows in `rollout/segment_rows.jsonl` to test whether generated prefixes amplify geometry or motion errors.

## Reproduction

```bash
ssh -tt neibu 'enter-container'
conda activate torch-ito
export CUDA_VISIBLE_DEVICES=0
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage preflight
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage plan
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage reaggregate
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage cuda_check --device cuda:0
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage evaluate --device cuda:0
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage rollout_smoke --device cuda:0
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage rollout --device cuda:0
python scripts/run_dit_source_reassessment.py --config config/dit_source_reassessment_v2_neibu.yaml --stage summarize
```

Plot: `/output/dit_source_reassessment_v2/20260914_r4_source_reassessment_v2_neibu/geometry_curves.png`.
