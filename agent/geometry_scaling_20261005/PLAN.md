# Geometry and motion improvement campaign

Started 2026-10-05 at user request. Branch exp/geometry-scaling-20261005,
parent a8da6de. This is a continuing research campaign, not another 512-step
architecture verdict. Earlier task-specific pilot stops do not prohibit the
newly authorized experiments. No default promotion without paired evidence.

## Success and reporting

Seek jointly better generated geometry and motion, then competitiveness with
original released PVB/BioKinema under compatible input/time/atom protocols.
Pre-register primary metrics and material-effect/non-inferiority margins using
training/reference variability before ranking new models. Preserve failed runs.
Separate engineering passes, exploratory development, independent confirmation,
and published-model comparison. Never label static ensembles as kinetics.

## Work streams and order

1. Audit completed C4/C8 outputs, exposure, final checkpoint identity and all
   paired rows; old report.md files are not authoritative live status.
2. Train-only latent diagnosis on current A1: h/v channel spectra, movement and
   source-noise scales, generated-error directions and equal-norm decoder
   sensitivity. Eight deterministic train systems, one clip each, no validation
   selection; target-informed probes explicitly separated from generation.
3. Expand evaluation: old valid8 becomes development-only. Inventory native
   trajectories and encoder/generator/static exposure, sequence-family clusters,
   ligand similarity and metadata. Freeze fresh development and confirmation
   manifests before reading their trajectory targets. Existing sealed payloads
   remain closed until the independent confirmation protocol is frozen.
4. Capacity/exposure lane: preserve current objective and compare 4/8 blocks at
   matched exposure with learning curves, then independent training seeds. Do
   not call a few hundred steps saturation. Larger tiers depend on measured
   profile, data coverage and curves, not merely parameter-count ambition.
5. Geometry lane: explicitly current-coordinate-aware interactions / compact
   geometry representation; train through actual generation, with separate
   clean/oracle, identity and motion controls. Change one scientific factor per
   first comparison; add end-to-end static pretraining after the spatial task
   achieves net recovery. Do not assume CA-only deterministic backmapping works.
6. Combine supported interventions and evaluate native multi-time grids,
   common endpoints, long rollouts and independent references. Compare released
   PVB/BioKinema with minimal adapters, no opportunistic baseline retraining.

## First registered latent diagnostic

Cache: previous completed CA pilot train files (64 clips,32 systems). Select
first lexical eight systems and first clip per system, fixed before results.
Load actual frozen A1/teacher/statistics. Generate Euler16 seed83 from observed
H4 only. Compute channel covariance spectra on clean latent and true increments
separately; h is mean-centered, v uses rotation-invariant channel Gram matrices
with xyz pooled (no Cartesian mixing or privileged lab-frame mean).
Compute per-atom-frame L2 norms (not mislabeled per-coefficient RMS), generated
error energy outside top8/32/64 train increment channels. Compare h-only/v-only
random and actual-error perturbations of true latent, total per-atom-frame norm
0.1 and1.0; these are explicitly target-informed sensitivity probes, not
deployable forecasts. Record coordinate-change RMS, clean bond/clash and actual
generation geometry/motion. Max2 GPUh; output below1GiB. No optimizer updates.

## Resources and authority

Use existing torch-ito via enter-container; CUDA only for model commands.
Local GPUs5–7 initially idle; do not interrupt other jobs. Local disk22GiB free:
reserve8GiB, no copies of raw data/parent weights, initial new artifacts <=4GiB.
Remote neibu has about4.3TiB free but all GPUs show activity; do not assume low
memory means unoccupied. Stage compatible code/data there when useful; do not
upgrade active environments or alter other jobs. Retain resumable latest/final
checkpoints with explicit storage limits; delete nothing in other worktrees.

GPU identity/telemetry, all optimizer steps, initial/final hashes, source/data
identity and reload verification accompany new training. Stage updates are
informational, not requests for user confirmation. Continue automatically when
the next safe, scientifically useful action is available.
