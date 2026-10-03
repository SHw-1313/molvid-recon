# Static decoder auxiliary pilot: final report

## Outcome

The bounded experiment is complete on GPU1 without stopping, restarting, or
modifying A/B/C. All three arms used the same released-A1 parent, 500 matched
dynamic updates and the same dynamic sample/noise/LR stream. The two auxiliary
arms each added 125 decoder-only static updates at cadence 4.

The main result is two-part:

1. External PDB geometry supervision **clearly improves held-out static decoder
   reconstruction** beyond both the released-A1 continuation and the matched
   internal-MD single-frame auxiliary arm. Relative to internal MD, external
   PDB reduces held-out output coordinate MSE by 7.75--12.40% across clean and
   0.05/0.10/0.20 A per-axis perturbations, with all paired system-bootstrap
   MSE intervals below zero.
2. That improvement **does not transfer into a consistent dynamic geometry
   gain**. External PDB gives a small fixed-history bond improvement and a small
   fixed-history MSD improvement, but ordinary bond/clash/severity changes are
   inconclusive or worse in point estimate. Internal MD single-frame auxiliary
   training gives the clearest dynamic benefit: about 8% fewer clashes and 10%
   lower clash severity, at the cost of small bond and motion errors.

Therefore this run supports an external-structure diversity effect on the
decoder's static reconstruction task, but it does not establish that external
PDB data improves dynamic generation geometry under this decoder-only route.

## Data and experimental scope

Dynamic training and evaluation retain the released A1 ATLAS split: 192 train
systems and the frozen eight-system validation protocol. Sealed test
trajectories were not opened.

The internal static pool contains 186 structures: one midpoint frame from the
first indexed clip for 62 eligible train systems and each of R1/R2/R3, limited
to 303--1,000 atoms.

The external arm reuses an already processed PDB train store with 70,533
records at:

`/data4/users/sihao/data/pvb_cross_dataset_20260810/pdb_phase12_half/train`

That store was derived from the user-authorized raw source `/data5/PVB/pdb`.
The deterministic eligible subset contains 3,000 fully supervised structures,
303--998 atoms (mean 711.17). Records must have compatible atom/block types,
valid topology and shapes, finite coordinates, and a full loss mask. Exact
PDB-chain origins overlapping released A train, valid, or sealed-test manifest
IDs are excluded.

Per the user's explicit authorization for this bounded version, no sequence-
or structure-level near-homology filter was applied. This experiment is thus
not a homology-controlled novelty comparison; exact-identity exclusion must not
be described as homology filtering. Full provenance is in `artifacts/data_card.json`
and `artifacts/external_pdb_3000_eligibility.json`.

## Model path and gradient boundary

The static task is explicitly `T=1`:

`x_tilde -> frozen teacher -> single-frame observed context -> shared decoder -> x_hat`

Clean coordinates are target/loss only. Reference coordinates, distances,
coordinate-derived features and centers come from `x_tilde`; physical lag is
not applicable and no motion label, RMSF, velocity, MSD or ACF loss exists.
Static updates change only the decoder. The target teacher remains frozen in
evaluation mode and history encoder, motion encoder and DiT are inactive;
inactive parameters and pre-existing optimizer state are checked fail-closed.
Static steps do not advance the dynamic scheduler, cursor, step or RNG.

The dynamic path is the complete released A1 objective. In compact form its
active coefficients are clean coordinate 1.0, clean bond 0.1, near coordinate
0.1, near bond 0.01, generated bond 0.0517072491, and actual-source sampled
clash 0.00106389149 at cadence 8. The flow objective remains active but has no
decoder gradient. Exact activation rules and zero-weight terms are recorded in
`artifacts/loss_manifest.json`.

The static loss is:

`L_static = gamma * (coordinate_MSE + 0.1 * bond_distance_MSE + 0.00106389149 * clash_raw)`

with `gamma = 0.00017325410836320802`. Train-only calibration used nine A1
dynamic batches, the training 20% clean mixture, and the actual-source sampled
clash term. It achieved the frozen 0.075 median ratio of weighted static
decoder-gradient norm to complete dynamic decoder-gradient norm. The two
auxiliary arms use exactly the same gamma, perturbation distribution and loss.

## Paired training

Here `A0` denotes the no-static-task continuation, `A2` the internal-MD static
arm, and `A3` the external-PDB static arm.

| Arm | Dynamic updates | Static updates | Static samples | Static supervised atoms | Checkpoint SHA256 prefix | GPU h |
|---|---:|---:|---:|---:|---|---:|
| A0 baseline | 500 | 0 | 0 | 0 | `f4d0ccd956ae` | 0.5134 |
| A2 internal MD | 500 | 125 | 737 | 531,450 | `90e80ba76d96` | 0.6331 |
| A3 external PDB | 500 | 125 | 742 | 523,904 | `9004d3e02aea` | 0.5645 |

Both auxiliary exposures meet the shared target of 739 +/- 3 samples and
527,677 +/- 3,773 total/supervised atoms. Each training log has exactly 500
finite dynamic rows; each auxiliary log has exactly 125 finite static rows.

The first baseline launch reached 138 rows and OOMed because two pre-existing
GPU1 jobs occupied about 9.1 and 6.9 GB. They were not signalled. The failure is
preserved under `formal_500/baseline_oom_gpu1_coexist_20260929/`. Restarting the
same arm with only `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reproduced
early deterministic values and completed.

## Dynamic evaluation

Evaluation used the full eight-system ordinary and fixed-history validation
manifests, Euler16, FP32, identical conditions and generated seed 0 across all
arms. Persistence and clean-latent oracle paths were also retained. Each
ordinary arm has 384 records/1,152 rows; each fixed-history arm has 192
records/576 rows. There is one generated draw per condition, so between-seed
variance is not estimable. Paired intervals below resample the eight systems,
not generation seeds.

### Main aggregate metrics

| Arm | Ordinary bond RMSE A | clashes / 1000 | severity A2/atom | residue RMSF MAE A | system RMSF Pearson | bias A | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.300292 | 179.655 | 0.058711 | 0.331507 | 0.8621 | -0.23830 | 0.33386 |
| A2 internal | 0.301010 | 164.600 | 0.052776 | 0.334338 | 0.8455 | -0.24317 | 0.32845 |
| A3 external | 0.299923 | 182.003 | 0.060128 | 0.333781 | 0.8396 | -0.24151 | 0.32849 |

| Arm | Fixed bond RMSE A | clashes / 1000 | severity A2/atom | MSD-curve MAE A2 |
|---|---:|---:|---:|---:|
| A0 | 0.289216 | 147.568 | 0.043755 | 2.53809 |
| A2 internal | 0.290228 | 136.546 | 0.039429 | 2.58291 |
| A3 external | 0.288379 | 147.282 | 0.044155 | 2.49850 |

A2 versus A0 has definite paired reductions in ordinary clashes (-8.38%,
absolute CI95 -20.16 to -11.33) and severity (-10.11%, -0.00864 to -0.00377),
and similar fixed-history reductions (-7.47% and -9.89%). Its ordinary bond
error rises 0.24% (absolute CI95 0.000046 to 0.001410), residue RMSF MAE rises
0.85% (0.000061 to 0.007045), and fixed-history MSD MAE rises 1.77% (0.0112 to
0.0824).

A3 versus A0 has no supported ordinary clash/severity gain: point estimates
are +1.31% and +2.41%, with intervals crossing zero. Ordinary bond changes
-0.12%, also crossing zero. Fixed-history bond improves 0.29% (absolute CI95
-0.001609 to -0.000032) and fixed MSD MAE improves 1.56% (-0.0801 to -0.00325),
but fixed clash and severity intervals cross zero.

Both auxiliary arms pass the frozen motion guardrail because degradations stay
below its thresholds. All A0/A2/A3 remain Pareto candidates due to their
trade-offs; `keep_release_trigger=false`, so this screen does not identify a
new release winner.

### Multi-lag view

| Arm | Ordinary bond RMSE at 100/200/300/400 ps | clashes / 1000 at 100/200/300/400 ps | residue RMSF MAE at 100/200/300/400 ps |
|---|---|---|---|
| A0 | 0.2798 / 0.2954 / 0.3119 / 0.3141 | 133.1 / 170.1 / 206.6 / 208.8 | 0.2694 / 0.3311 / 0.3491 / 0.3764 |
| A2 | 0.2801 / 0.2965 / 0.3123 / 0.3152 | 120.6 / 155.8 / 188.9 / 193.2 | 0.2715 / 0.3332 / 0.3525 / 0.3801 |
| A3 | 0.2792 / 0.2950 / 0.3117 / 0.3138 | 133.7 / 171.5 / 209.2 / 213.6 | 0.2712 / 0.3328 / 0.3515 / 0.3796 |

Fixed-history MSD MAE at 100/200/300/400 ps is 2.379/2.412/2.641/2.720 for
A0, 2.441/2.427/2.695/2.769 for A2, and 2.365/2.376/2.602/2.651 for A3. Thus
the internal clash benefit is present across lags, while the external arm does
not reproduce it.

All 75 figures and 48 per-residue profiles are under
`runs/static_decoder_probe_v1/comparison_seed0/figures/` and
`per_residue_profiles/`; the exact paired table is `per_system.csv`.

## Decoder diagnostics

### Held-out static reconstruction/perturbation

The fail-closed diagnostic uses anchor window 0, one record for every eight
validation systems x three replicas, the middle valid frame, and sigmas 0,
0.05, 0.10 and 0.20 A per Cartesian axis. Record-local noise is exactly shared
across arms. Each arm has 96 rows and `test_opened=false`.

| sigma/axis A | Input MSE A2 | A0 output MSE | A2 internal | A3 external | external vs internal |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 1.88991 | 0.046828 | 0.041020 | -12.40% |
| 0.05 | 0.002486 | 1.93372 | 0.049797 | 0.043858 | -11.93% |
| 0.10 | 0.009983 | 2.05747 | 0.059454 | 0.053098 | -10.69% |
| 0.20 | 0.040011 | 2.30429 | 0.092326 | 0.085167 | -7.75% |

The external-minus-internal absolute MSE CI95 is respectively [-0.007154,
-0.004511], [-0.007291, -0.004607], [-0.007767, -0.004941], and [-0.008792,
-0.005547]. External PDB also lowers bond-distance MSE by 1.4--8.2% and raw
clash by 15.9--17.7% relative to internal MD across these sigmas.

This is a strong reconstruction-task result, but neither auxiliary checkpoint
is a net denoiser relative to passing through the raw perturbed coordinates:
output MSE remains above input perturbation MSE at every nonzero sigma. The
correct claim is improved decoder reconstruction, not complete removal of the
input perturbation.

### Clean-latent oracle

| Arm | Ordinary aligned RMSD / dRMSD / bond RMSE A | Fixed aligned RMSD / dRMSD / bond RMSE A |
|---|---|---|
| A0 | 0.03635 / 0.03392 / 0.02053 | 0.03621 / 0.03384 / 0.02033 |
| A2 | 0.03588 / 0.03284 / 0.02054 | 0.03583 / 0.03290 / 0.02039 |
| A3 | 0.03649 / 0.03416 / 0.02056 | 0.03642 / 0.03418 / 0.02040 |

A2 modestly improves oracle aligned RMSD/dRMSD; A3 is slightly worse than A0.
This reinforces that the external held-out static gain is specific to the new
T=1 reconstruction path and is not a general clean-latent dynamic decode gain.

## Verification and independent review

The implementation candidate is commit `07b6ea76a1460eb37f85154cb3770983ad64c490`.
The initial independent review of `73ba705` returned `FIX_REQUIRED` for formal
orchestration, A1 gamma calibration, external eligibility, fail-closed optimizer
boundaries and traceability. Every issue was fixed and the independent fixed
review passed. The uniform reduction from 1,000 to 500 dynamic updates was made
before comparison results and received a separate exact-identity PASS.

Focused evidence includes 29 contract/metadata tests, four GPU1 CUDA static
tests, all three eight-step CUDA preflights, strict internal step-4 to step-8
resume, FP32/BF16 and a 4,975-atom profile. The seed-configurable evaluator at
`e64bd52` received an independent PASS and all six CUDA preflights passed.

The held-out evaluator followed its own independent chain: `8792b17` received
`FIX_REQUIRED` for fail-open protocol identity; fixed commit `080511e` passed.
Its first GPU preflight then failed before metrics because approved legacy
validation records omit `frame_mask`. The failed protocol is retained. Commit
`abd8f7b` normalizes records through the public validator; the targeted
independent review passed and all three formal static evaluations completed.

Review records are preserved in `reviews/`. No reviewer edited model or
evaluator code.

## Cost and limitations

Known measured time is about 4.21 GPU-hours: 1.90 training including the
coexistence OOM, 2.17 dynamic evaluation including six preflights, and 0.146
held-out static evaluation including its failed preflight. Adding 0.27 hours
of earlier smoke/pilot/preflight time and a conservative 0.25-hour unlogged
allowance gives 4.73 GPU-hours, leaving about 3.27 hours under the authorized
eight-GPU-hour cap.

This is a short 500-update screen with one generated draw per condition. It is
not convergence evidence, a multi-seed uncertainty estimate, a strict
compute-matched comparison, a homology-controlled external-data experiment, or
production parity. The auxiliary arms perform 25% more optimizer updates than
baseline by design, so decoder-task, data-diversity and extra-compute effects
cannot be fully separated by this three-arm pilot alone.

## Recommendation

Do not scale the current external-PDB decoder-only recipe as a claimed dynamic
geometry improvement: its static signal is strong, but dynamic transfer is not.
If continuing, the most informative next experiment is a small, separately
reviewed objective that exposes external static information to the shared
latent/DiT path, with a homology-controlled evaluation split and multiple
generation seeds. The current internal-MD auxiliary arm is the better candidate
for a focused clash-reduction follow-up, provided bond and motion trade-offs are
explicitly constrained.

Exact identities, commands and machine-readable summaries are in
`artifacts/train_manifest.json`, `loss_manifest.json`, `commands.sh`,
`static_diagnostic_comparison.json`, `oracle_diagnostic.json`, and
`cost_ledger.json`.
