# R4 DiT architecture sequential v1

Final status: `SEQUENTIAL_V1_COMPLETE`

The four changes were implemented and compared strictly in order. Each arm changed only the listed
variable and started from the selected parent of the preceding round. The fixed data selection was
48 train systems and 8 validation systems; the real smoke used 3 systems/9 trajectories. Test data
remained sealed throughout.

| stage | sole variable | exposure per arm | decision | selected checkpoint SHA256 |
|---|---|---:|---|---|
| baseline | audited conditional C48, clean post-AdaLN norm | 4,500 updates; 267,988,032 tokens | PASS | `808eb51e5c597629ebbd7270697af91affbc82a812cadd165eae984cba900240` |
| round 1 | FFN scalar norm: post-AdaLN vs pre-AdaLN | 5,250 updates; 194,960,744 tokens | REJECT; control | `ff66c98d6f051fbc1cd3cb8ba29bedaebd453804555a0c4d8d7e2865cdd96375` |
| round 2 | RF only vs calibrated future-bond loss (`lambda=0.2303875588749564`) | 5,000 updates; 142,319,624 tokens | KEEP geometry loss | `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34` |
| round 3 | coordinate-supervised local vs cross-block decoder refiner | 5,000 updates; 142,319,624 tokens | KEEP_PARENT; no refiner | `7fec9b339201765572ddadf0c64892fe9b7db7f59fdc8148b6f5c4036f514e34` |
| round 4 | clean history vs 50/25/25% coordinate corruption at 0/0.02/0.05 Å | 5,000 updates; 142,319,624 tokens | REJECT; clean history | `994e69e653159db0a3af3101ea8f0cae690a754dcbd8d09ec897c2649d26f2e6` |

Recorded baseline and round-training compute was 26.358955 GPU-hours. The first R4 invocation OOMed
while packing an 80k-token validation batch after 500 non-comparison updates; its artifacts remain
under `round4/attempt_001_validation_oom/`. A conservative 1 GPU-hour recovery debit was registered,
the validation cap was fixed at 56,672 tokens, and no common extension was used.

## Decisions

- Round 1 did not support the amplitude hypothesis: pre-AdaLN amplitude relative changes were
  `-0.002128` (H4) and `-0.002144` (H8), with 1/8 systems favorable in each history.
- Round 2 supported direct geometry supervision in this short free-generation test: bond error
  improved by `0.833901` (H4) and `0.838367` (H8), with 8/8 systems favorable and all clean
  contact, amplitude, and motion-collapse guards passing.
- Round 3 did not support the cross-block refiner: cross-block versus local boundary improvement was
  `-0.056095` (H4) and `-0.049682` (H8); local versus the no-refiner parent was `-0.000998`
  and `-0.001031`.
- Round 4 did not support error-corrupted history: paired rollout `E_roll_bond` was
  `0.1455386965` for control and `0.1497036192` for corruption, relative change `-0.0286173`,
  with 2/8 systems favorable. Clean guards passed and true-prefix rollout was not confounded.

The final retained composition is clean post-AdaLN norm, the Round 2 future-bond auxiliary, no
decoder refiner, and clean observed history. The final control model hash is
`4b46a643a475769ff2d5138f5425fb592333b9519aafd676ff0cda9107ac41cb`; its adapter hash is
`64ca538234e10f5247c773db60c2dc72a7c87ea054598a3c4a4513139c5f7999`. Frozen codec/statistics
hashes are unchanged.

## Cumulative clean quick change from C48

System-equal aggregates over the fixed eight validation systems:

| history | metric | C48 | retained final | relative change |
|---|---|---:|---:|---:|
| H4 | aligned RMSD | 3.721148 | 1.673524 | -55.027% |
| H4 | bond RMSE | 3.541727 | 0.477159 | -86.528% |
| H4 | contact F1 | 0.355832 | 0.794198 | +123.194% |
| H4 | amplitude error (Å) | 1.891038 | 0.296022 | -84.346% |
| H4 | dRMSD | 2.989001 | 1.281278 | -57.134% |
| H8 | aligned RMSD | 3.750142 | 1.646716 | -56.089% |
| H8 | bond RMSE | 3.530758 | 0.465640 | -86.812% |
| H8 | contact F1 | 0.356404 | 0.797820 | +123.853% |
| H8 | amplitude error (Å) | 1.571281 | 0.292176 | -81.405% |
| H8 | dRMSD | 3.006469 | 1.236565 | -58.870% |

These gains are cumulative continuation from C48, not single-round attribution or a scaling-law claim.

## Neibu recovery and local verification

The neibu disk fault made its container unusable, but the host filesystem stayed readable over SSH.
Baseline and R1--R3 final checkpoints plus both R4 final checkpoints were synchronized locally and
hash-checked. Local GPUs 5--7 were idle; the exact R4 quick evaluation was rerun in local
`enter-container`/`torch-ito` on physical GPU5 (`CUDA_VISIBLE_DEVICES=5`, logical `cuda:0`).
Both clean arm summaries and the three-segment rollout passed. Regenerated rollout rows SHA256
`48c7e9cba5775f82931bd407a5235de53517652cfc988a527cf1038210c8e1c8` matches neibu, and the local
decision rerun remains `REJECT`/control. No remote process was restarted or modified.

Targeted CUDA suites passed: metadata/I/O 5, R1 4, R2 3, R3 4, R4 5; directly affected R2+R3 and
R2+R4 regressions passed 7 each; real smoke passed with finite outputs and exact clamps. Remaining
risks are one seed, the 8-system/32-frame diagnostic, no long-rollout BPTT, and no scaling claim.
Detailed evidence is in the `baseline/`, `round1/`, `round2/`, `round3/`, and `round4/` folders.

## Motion metric recheck erratum (2026-09-17)

The historical Round 2 final and Round 3 quick summaries were re-read after finding that the
motion guard had looked under `future["rmsf"]["prediction"]` while `system_rows` use the flattened
`future["rmsf.prediction"]` field.  The original `decision.json` files and parent/child lineage are
unchanged.  The sentence above saying that Round 2's motion-collapse guard passed is superseded:
the old null/zero count was an unevaluated guard, not a pass.

| round / scope | original comparison | H4 corrected median | H8 corrected median | expected / actual systems | corrected guard |
|---|---|---:|---:|---:|---|
| R2 final | candidate / control | 0.297096802 | 0.328881603 | 8 / 8 | FAIL (`0.5` threshold) |
| R3 quick | local / parent | 0.997843888 | 0.997508324 | 8 / 8 | PASS (`0.5` threshold) |
| R3 quick | cross_block / parent | 1.016518180 | 1.018089405 | 8 / 8 | PASS (`0.5` threshold) |

Both R2 histories have complete pairs and no missing, NaN, or near-zero-denominator systems.  The
R2 candidate prediction RMSF is `0.729069899` / `0.686719220` Å (H4/H8) versus control
`2.450532685` / `2.078122252` Å, while target RMSF is `0.995433070` / `0.918731843` Å.  Thus the
bond/contact improvement was accompanied by substantially lower motion than the control; the
candidate/target ratio of means is `0.732414786` / `0.747464263`, and the candidate/control guard
ratios are only `0.297096802` / `0.328881603`.  The latter is the original guard metric; the
prediction/target diagnostics do not replace it.  Round 3 changes motion by approximately 0.2%
for local and 1.6--1.8% for cross-block relative to its parent, with complete passing guards.

The corrected Round 2 branch is therefore `TRADEOFF` / control rather than the historical `KEEP` /
candidate.  Round 3 remains the historical `KEEP_PARENT` / parent because its primary boundary
metric already failed and its corrected motion guards pass.  No downstream parent re-selection or
new evaluation was performed: the safe corrected recommendation is to retain the clean Round 1
control lineage (post-AdaLN, without the Round 2 auxiliary) pending any separately authorized rerun;
the existing R3/R4 decisions remain conditional historical evidence from the former R2 parent.

The complete tables, per-system pairs, arm prediction/target diagnostics, source hashes, and
standalone corrected decisions are under `motion_recheck_v1/`.
