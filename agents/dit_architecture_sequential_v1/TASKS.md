# R4 DiT architecture sequential v1 tasks

## In progress

- [ ] Establish the clean baseline: port the minimum committed parameter-isolation repair, prove
  independent training and evaluation loading, and verify or train a clean C48.
- [ ] Freeze run ID, fixed train/validation clip selection, generation seeds, and the total/round
  budgets after an end-to-end profile on an audited idle CUDA GPU.
- [ ] Round 1: implement, verify, train, evaluate, and decide pre-AdaLN vector norms.

## Blocked by sequence

- [ ] Round 2: geometry supervision. Do not implement before the Round 1 decision.
- [ ] Round 3: block-local versus cross-block temporal refiner. Do not implement before the Round
  2 decision.
- [ ] Round 4: corrupted training history. Do not implement before the Round 3 decision.
- [ ] Re-evaluate the original clean baseline and final retained combination; write the final
  per-round report and final status.

## Completed

- [x] Verified target worktree `/data4/users/sihao/workspace/molvid-dit-architecture-sequential-v1`,
  branch `exp/dit-architecture-sequential-v1`, and audited base
  `027659fe872a6c869b14ecce5b799356bd695694`.
- [x] Read the root and phase-local instructions in the required order.
- [x] Located B's committed isolation repair at
  `8eb0a0d45f26c3b761c72c740f4c7da192afc1fb` and arm-local source-decode correction at
  `f839de97840c7896a56566fec06d6cbe6f93eaea`.
- [x] Imported the minimum arm-local adapter / successful-update repair and the audited AdamW
  scalar-step restore behavior; added strict independent sequential checkpoint loading.
- [x] Implemented the Round 1 post/pre-AdaLN norm switch in both reference and factorized-v2 paths.
- [x] Added the sequential runner/config, generation evaluator, targeted CUDA test collection,
  future-only token accounting, per-epoch sampler/spec caching, atomic common-checkpoint resume, and the one-time matched extension path.
- [x] Ran metadata preflight and froze the 8-train/8-valid quick set, 72-clip final valid set, and
  historical 3-system/9-trajectory smoke set; frozen asset hashes passed and test stayed sealed.
- [x] Passed Python compilation, test collection, and 5/5 metadata/I/O seed/token/resume/cache tests inside
  `enter-container` / `torch-ito`; no CUDA claim is attached to these checks.

## Current constraint (historical preflight snapshot)

- `SEQUENTIAL_V1_BLOCKED`: repeated audits through 2026-09-14 21:37 +08:00 found all eight local
  GPUs occupied. No CUDA check, profile, smoke, or training may start until a card is confirmed idle.
- B's C48 remained in the documented neibu training queue and no completed, locally auditable C48
  checkpoint was present at the blocked stop.
- Resume from local implementation commit `333f934bfe72428ea6002b649d5b6eb5f4fdc08c`; first rerun
  preflight and the four gated CUDA tests. Rounds 2--4 remain sequence-blocked by Round 1.

The above block records the 2026-09-14 no-idle-GPU preflight only. It was superseded by the recovered
neibu-output synchronization and local CUDA continuation recorded below.

## 2026-09-17 recovered completion evidence

- [x] Recovered readable neibu outputs after the disk fault; synchronized baseline and R1--R4 final
  checkpoints and verified their SHA256 values locally.
- [x] Re-ran the exact R4 quick clean/rollout evaluation in local `enter-container`/`torch-ito`
  on idle physical GPU5; both arms passed and rollout rows matched the neibu SHA256.
- [x] Recorded all four sequential decisions and the retained final composition in the run-level
  `report.md`; test payload remains sealed and no common extension was used.
