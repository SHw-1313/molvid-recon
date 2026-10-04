# Independent review follow-up — execution protocol

Authorized 2026-10-05: implement and run the independent review recommendations
without inter-stage confirmation. No sealed test, no incumbent job termination,
no default checkpoint promotion, no U/K generator loss.

Parent: S `e2e47df`, directly descended from ABCD `946917b`. U/K public tools
are imported with one shared cache/reporting source; old worktrees remain
unchanged. Mamba stays isolated in the original candidate branch. This tree
retains S's module-budget calibration and gradient scope as the sole shared
static continuation implementation; the older D2 runner is not duplicated.

N0 first: E1 explicit residue roles and auditable atom selection, E2 unique
residue anchors, E3 topology content identity and mutation-safe cache, E4
per-sample compact history, E5 mask-aware clock. Preserve schema/weights for
old checkpoints. Targeted CUDA tests <=0.25 GPUh; same-H real A1 old/new
inference and probe feature comparison. Do not widen old numerical bounds.

After N0 passes, independent experiment processes may run concurrently on
currently available devices (not necessarily neibu, whose occupancy changed).

- N1: frozen A1 decoder/latent/solver and native-time diagnostic. Four existing
  validation systems, two non-overlapping anchors; exact native timestamps,
  fixed H4 observed frames at 100 ps. Primary solver study dt100 Q12, seeds
  0/1, Euler8/16 (32 draws). Additional fixed-context grids 80/200/400/1000 ps
  and common 400 ps endpoint are a bounded pilot. Separate budgets/protocols
  before execution; no claim of thermodynamic equivalence from few draws.
  First batch <=48 draws and <=2 GPUh, no training. Subsequent diagnostic
  batch only if the first batch leaves the relevant question unresolved.
- N2: stable completed S0/Sdec/Strunk raw-row reaggregation, exact paired
  protocol/identity checks, static MSE versus noisy input by sigma and system,
  dynamic system bootstrap. No new training unless static net restoration
  works but dynamic regression remains; otherwise stop static scale-up.
- N3: same immutable 168/24/8 cache, last/delta/full-history masked controls,
  matched 128x2 MLPs, seeds101/102/103, <=2000 steps/head; mean-only and
  diagonal Gaussian controls; fixed ridge comparator. K delta control uses
  the same fit-only response regions and matched budgets; no K1 repeat.
  <=1 GPUh. Report system-level metrics and paired differences per seed.
  History-time statistics cannot add independent information in a cache
  containing only fixed H4/dt100; mark that limitation instead of inventing
  a feature. No generator updates or automatic Mamba promotion.
- N4: nonsealed reference qualification, physical metadata unknowns, exposure,
  reference replicas/blocks, finite-data correlation/ESS and native grids;
  do not claim all-atom trajectory stationarity from one scalar. No new MD.

Each new runner records code tree/diff, source commit, input SHAs and protocol
before execution. Generated outputs stay under runs/review_20261005 and are
not mixed into original review snapshots. Final decision explains which
conditional experiments were not justified, versus failed or unimplemented.
