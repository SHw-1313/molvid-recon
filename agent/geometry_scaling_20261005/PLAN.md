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

## Audit finding and next registered action

An initial reading of retained_identity suggested C4 step13000 versus C8
step11000, but exact evaluator checkpoint SHA disproved this: BOTH evaluations
already use step11000. C4 retained_identity is stale. Exposure and training/aux
RNG were checked and identical. Resolve artifacts by evaluated SHA, never by
retained metadata alone. The attempted redundant C4 evaluation was interrupted
and its evidence retained; verify existing coverage instead of repeating it.
CPU aggregation is never labeled GPU training. Old valid8 remains development.

## Observed-coordinate geometry control, registered before execution

Before a new learned architecture, test an explicit geometry repair baseline
on the same eight selected TRAIN systems, cached A1 seed11.32 coordinate updates,
step0.03A, maximum per-atom shift0.5A. Objective: observed bond-length MSE×10,
reviewed clash severity×1, original-coordinate retention×0.1 and original
increment retention×0.1. No true future coordinate/bond target. Scalar second
moments shared over xyz preserve rotational covariance. Existing A1 latent and
neural weights frozen; this is coordinate optimization, NOT neural training.
Primary diagnostic: whether current-coordinate geometry can be repaired without
losing RMSF/MSD; also inspect angles/torsions and inter-residue bonds. Fixed
weights before results, no validation tuning, max2GPUh. This cannot establish
kinetic/thermodynamic correctness, and any later benchmark must report raw and
postprocessed output separately with equal postprocessing opportunities.

## Capacity continuation, registered before training

C8 was budget-stopped at11000, original schedule ends13575. Complete the
remaining2575 updates under unchanged original objective/data/schedule after
an exact next-update comparison between original source and this branch from
the same11000 checkpoint. Compare model/optimizer/step/rank RNG states, not
timestamp metadata. If parity fails, diagnose before longer training; do not
weaken tolerances silently. New outputs only, original worktree/checkpoints
read-only. Save1000-step intervals and final; max18GPUh, >=8GiB disk reserve.
This is genuine full C8 training, not the geometry-projection diagnostic or
a new independent training seed. After completion compare C4/C8 at13575 and
on expanded development; continue exposure/scale experiments based on curves.

## Expanded development protocol, frozen before reading trajectory targets

Sequence-only audit:1382 nonsealed sequences,202 known exposed/development
systems excluded,1082 eligible clean clusters under MMseqs30% identity/50%
bidirectional coverage; choose48 developer systems and reserve96 confirmation
systems, one representative per cluster, hash order balanced across length bins.
This is known-parent sequence isolation, NOT a completed whole-pretraining or
structural-family independence claim. Old sealed8 PDBs/trajectories not read.
Materialize developer48 only: R1, anchors450/2450, fixed observed H4/100ps,
native future12 frames at100/400/1000ps, no interpolation. Hash selected raw
frame/timestamp arrays and saved records; source XTC path/size/mtime retained,
not mislabeled whole-file SHA. Confirmation targets remain closed.
First common benchmark uses100ps, both anchors, seeds17/29, A1 and C4/C8;
PVB native100ps with same last-observed structure is a legal H1 baseline, not
an equal-history comparison to H4. All raw/postprocessed results separated.

PVB native adapter pilot exposed NumPy2 incompatibility (`np.compat.long`) in
the untouched official topology builder. Install NumPy1.26.4 into ignored
`.runtime/pvb_numpy126` and activate via process-local PYTHONPATH for PVB only;
do not monkeypatch official source or downgrade the shared training environment.
Record NumPy version/source, original source hashes and checkpoint hash.
The failed first pilot stays preserved. Pilot A1 completed (two generated
draws plus observed-only projection), then expanded A1 was launched.

## Gcoord: current-coordinate geometry recovery (independent, no default change)

Motivation: decoder covalent messages and DiT geometry directions currently
come from the observed structure, whereas newly generated nonbonded clashes
depend on current coordinates. Test that missing spatial feedback directly.
Four radius-graph layers, width96, radius4.5A, cap128 plus unconditional
covalent edges. Rebuild graph each layer; gradients through distances and
directions, not discrete edge membership. Atom/block embeddings, scalar
messages and scalar-gated vector corrections. Preserve per-system centroid;
SO3 covariance, packed-system isolation and padded-frame tests required.
Zero-initialize coordinate gates, so starting predictions are exactly A1.
Last layer has no unused hidden-state update parameters.

Compare Gcoord-static versus Gcoord-mixed, seeds71/72,2048 optimizer updates
each after a16-update execution pilot (pilot never parent). Identical static
exposure/noise schedules: all192 parent training systems, random native
snapshots, noise sigma0/.05/.15/.3/.6A with atom+residue components and zero
translation. Coordinate MSE+0.5 bond MSE. Mixed adds an actual frozen-A1
generated consecutive-frame pair every4 updates, from the preexisting32-system
training cache:5 observed-bond MSE+clash severity+0.1 coordinate retention+
0.1 increment-change penalty. Never regress actual draws against their single
future target. AdamW3e-4,wd0.01,clip1,FP32.6h maximum per arm,8GiB reserve.
This trains only a new geometry module, not a DiT; C8 continuation is the
separate full-model training experiment. Reusing MD snapshots is not new
external-static pretraining. Evaluate raw A1 and repaired predictions on the
same frozen48 development clips, with no checkpoint selection on development.
Do not merge if geometry gain trades away motion or if only train improves.
No thermodynamic/kinetic correctness claim follows from local recovery.

Expanded followup queue uses GPU5 only after its A1 job ends: C4/C8 at11000,
C4/C8 at13575 (wait for real C8 completion). All four final Gcoord models run
separately on GPU7, on identical cached A1 draws. Geometry screening is exploratory development:
target >=20% bond and >=30% clash reduction; motion noninferiority margin5%
for MSD/residueRMSF errors, and at least one clear motion improvement before
claiming joint improvement. Cluster/system-paired uncertainty and seed spread
must be reported. Passing this screen advances to long/native-time tests,
not automatic competitive or thermodynamic claims.

## Source-bank amplitude diagnostic (training systems only)

After the Gcoord GPU evaluation, run8 fixed training systems ×2 seeds(83/84)
×4 inference source settings: original(h1,v1),h-only(.2,1),v-only(1,.2),both(.2,.2).
No model training and no claim that a different sampling distribution is a
trained improvement. This isolates the vector/scalar source burden suggested
by measured source-to-motion norms before committing to a source-retraining
experiment. Explicit(1,1) must match the unchanged default exactly on a real
CUDA forecast; count its extra parity draw separately(64+1 total). All
geometry/motion/CA-handedness metrics retained,2h cap. The global default
remains unit noise; diagnostic arguments never change checkpoint identity.

Source-retraining infrastructure, not a launched training claim: optional
`model.frame_source={h_scale,v_scale}` is a positive finite scalar/vector
standard-deviation contract in normalized latent units. One configuration
feeds RF interpolation, actual-source auxiliary rollouts, static backbone RF,
generation and calibration; checkpoint model contracts preserve nonunit values
and strict resume must reject a mismatch. Old unit-source constructor contracts,
state tensors, arithmetic and RNG must remain identical. Changing the source
requires model-only warm start with fresh optimizer, never exact-resume labeling.
Do not scale the velocity targets or loss weights independently in this first
intervention. Validate full/streamed auxiliary gradients and old real-A1 cached
draw parity before any new-source training. The inference-only diagnostic is
restricted to the old unit-source parent; do not multiply two hidden temperatures.

GPU7 queue now launches fresh subprocesses AFTER prerequisites. This avoids a
waiting Python process importing old model code then hashing newer disk source.
The old idle source diagnostic controller was stopped before any GPU/model call;
no training or evaluation was interrupted. Source CUDA gates precede the diagnostic.

Training-data expansion reservation completed separately:192 existing +320
additional=512, no new trajectory targets opened at reservation. Protect
expanded48,reserved96,old sealed/valid and known-clean official baseline test
clusters. This is a future controlled data-size arm, not a completed training.

## Bounded source-retraining experiment (after completed source diagnostic)

The inference-only v0.2 intervention was strongly NEGATIVE (bond1.001A versus
0.263A; clashes1365.9 versus93.35). Scalar-only reduction also suppresses motion.
Do not call these trained candidates or conclude that a smaller source is better.
The result shows a strong learned-field/source dependence; it does not identify
the quality of a model trained consistently with a changed source. Test that
remaining hypothesis with a bounded matched pair, rather than change defaults.

Freeze unit(h1,v1) versus v02(h1,v0.2), two seeds71/72,2048 full-model updates
each, same A1 parent and fresh optimizer/scheduler/cursor/RNG. All192 parent
training systems are exposed by2048;512/1024 cover fewer systems and are not
substituted for a full data-coverage claim. Exact data/H/auxiliary schedule within
seed; teacher/normalization/losses/auxiliary8-step rollout held fixed. BF16,
20000 atom-frames/update, lr history2.5e-5/DiT5e-5/decoder1.25e-5,
5percent warmup and cosine to0.1. First16 updates are an execution smoke under
the SAME2048 schedule, then exact resume WITHIN that arm. This is never exact
continuation from A1. Final2048 only for development comparisons; no best-valid
selection. Maximum6h/arm,10GiB free before arm, no automatic retry of failures.
Seed71 runs v02 then unit on GPU7;seed72 runs unit then v02 on GPU6 only after
its existing C8 continuation finishes.8192 planned full-model updates are NOT
the already completed8192 small Gcoord updates. All final arms must pass both
motion and chemistry; the default remains A1 even if training loss decreases.
