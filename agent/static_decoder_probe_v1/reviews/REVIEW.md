# Independent review of candidate `73ba705`

Verdict: **FAIL / FIX_REQUIRED for formal symmetric training.**

The static decoder kernel itself is largely sound, but the fixed commit is not
ready to launch the released-A1 comparison.

## Findings

1. **BLOCKER — no released-A1 formal orchestration exists.** The only runner
   accepts `smoke|pilot` (`tools/run_static_decoder_pilot.py:50`) and requires
   the auxiliary recipe's parent hash to equal the supplied checkpoint hash
   (`tools/run_static_decoder_pilot.py:527`). The released auxiliary recipe is
   bound to engineering P2 `aa3b8992...`, while released A1 is `52d922f8...`;
   the pilot correctly identifies itself as P2-only (`result.json:59`). There
   is no three-arm runner implementing identical A1 initialization, dynamic
   batch/order/noise, LR schedule, evaluation cadence, 4:1 mixing,
   exposure/cost accounting, or resume state. The pilot checkpoint saves
   decoder weights, the whole optimizer, and static RNG only
   (`tools/run_static_decoder_pilot.py:655`); the production checkpoint schema
   has no static RNG/count/cadence state (`molvid/training/joint.py:1039`).

2. **BLOCKER — gamma is not calibrated against the released A1
   decoder-gradient denominator.** Model construction omits `sampled_config`
   (`tools/run_static_decoder_pilot.py:201`), and dynamic calibration supplies
   no actual sampled sources (`tools/run_static_decoder_pilot.py:220`). It
   therefore excludes A1's active actual-source clash term despite reporting
   the denominator as the sum of active decoder-gradient losses
   (`molvid/training/static_decoder.py:451`). Calibration also forces zero
   clean probability (`tools/run_static_decoder_pilot.py:278`), whereas
   training uses 0.20, and it records neither raw/weighted per-term
   contributions nor zero-gradient coverage. The preserved interrupted
   calibration produced `1.810694e-4`; the final cheapest-system calibration
   produced `7.215590e-5`, a material 2.51x difference.

3. **BLOCKER — no external static corpus is presently proven eligible.** A
   real 70,533-record PDB-derived store exists, but its
   `/data4/users/sihao/data/pvb_cross_dataset_20260810/pdb_phase12_half/train/selection_report.json`
   records provenance/counts only. No homology/cluster exclusion or
   system-level comparison against A validation and sealed-test identities was
   found. Thus the corpus physically exists, but the external arm remains
   disallowed until a clean-target leakage audit is recorded.

4. **HIGH — the decoder-only optimizer boundary is not fail-closed.** The
   auxiliary trainer checks only that decoder parameters are a subset of
   optimizer parameters (`molvid/training/static_decoder.py:493`); the runner
   passes the full dynamic optimizer (`tools/run_static_decoder_pilot.py:611`).
   Fresh-start evidence shows inactive weights and new optimizer state remained
   unchanged, but it does not exercise a realistic optimizer after dynamic
   steps, where history/DiT already have moments and step counters. Formal
   readiness needs either decoder-exclusive parameter groups or a post-dynamic
   proof that inactive parameters and all inactive optimizer state remain
   bitwise unchanged across static steps, including declared scheduler
   behavior.

5. **MEDIUM — traceability is incomplete.** The interrupted audit directory
   contains only gamma and 17 metric rows, with no result/error/termination
   record, identities, or independently recorded 58 GB measurement. The
   pilot's named `peak_allocated_bytes` is a maximum of post-step live snapshots
   (`tools/run_static_decoder_pilot.py:626`), not CUDA's peak-allocation
   counter. Task handoff files are stale (`TASKS.md:9`, `HANDOFF.md:8`, and
   `DECISIONS.md:15`). The newer review request was untracked and not part of
   fixed commit `73ba705...`.

## Positive assessment

The core implementation passes static scientific review: strict T=1/time-zero
semantics, perturbed-only observed context, clean targets confined to losses,
target-mutation invariance, frozen/eval/no-grad teacher, explicit Angstrom
units and packed masks, separate RNG, and shared A clash
exclusions/normalization/autograd behavior are correctly implemented. CUDA
smoke/pilot evidence is finite and hash-consistent. The reviewer verified A1
checkpoint `52d922f8...`, A1 config `15d55300...`, auxiliary recipe
`625d11cf...`, P2 parent `aa3b8992...`, and pilot checkpoint `09ab0064...`.
These results support continuing engineering, but not starting formal training
or claiming production/scientific parity.

## Independence record

The review was performed by a fresh read-only agent with no inherited
implementation conversation. It reviewed fixed commit
`73ba705ab4e506e3274d54b62c9a351e84127125`, the task specification, required
repository documents, released identities, and the smoke/pilot evidence. It
made no file or model-code edits.
