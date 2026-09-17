# R2/R4 state-detail latent DiT T1 pilot v1

## Authorization and code state

This is the separately isolated pilot branch `exp/dit-state-detail-t1-pilot-v1`, created from
the pushed repair commit
`28a499f4c03deb4647a6468a5c477bd8eda8d62f` on
`feat/dit-state-detail-probe-v1`. The repair branch is reviewable independently. This pilot is
not a new architecture phase and does not authorize test access or an R2/R4 winner declaration.

All Python, tests, profiling, training, evaluation, and plotting run through `enter-container`
with `conda activate torch-ito`. No packages are installed or upgraded. Outputs belong under
`outputs/dit_state_detail_pilot_v1/`; checkpoints and large/binary artifacts are not committed.

## Frozen data manifest

Manifest:
`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`

Read-only preflight on 2026-09-08 verified:

- 64 total systems: 48 train, 8 validation, 8 sealed test;
- 8,928 train, 1,488 validation, and 1,488 sealed-test clips;
- `system_disjointness.train_test`, `.train_valid`, and `.valid_test` are all `true`;
- `test_sampling.opened` is `false` and `replacement` is `false`;
- manifest content SHA256:
  `f8a764eb38e90c7485bf9799115d570bb868f34584ebbafab8c799fec03df4d2`;
- materialization SHA256:
  `5b622ae6d0ac2a6b498f3a9388bd2053dbb29cc8aca83eecf5723987c94d49c7`.

Only train and validation clip stores may be opened. The test manifest metadata may be checked
for sealing, but no test clip store, index, data, or evaluation may be constructed or read.

## Approved frozen codecs

The only allowed codec inputs are the completed full T1 runs below. Both result files reported
`status: passed`, schema `pvb.codec.state_detail.t1_result.v1`, and the common frame-encoder source
hash `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a`. The best checkpoint
SHA256 values were independently verified against the files.

| candidate | result | best checkpoint | SHA256 |
|---|---|---|---|
| `ratio2_state_detail` | `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/result.json` | `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio2_state_detail/codec_best.pt` | `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9` |
| `ratio4_state_detail` | `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/result.json` | `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt` | `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df` |

`ratio1_state_detail` and `ratio4_matched_pooling` remain historical controls and are not pilot
candidates. Randomly initialized codecs, T0 checkpoints, incomplete checkpoints, and guessed
paths are forbidden.

## Matched pilot protocol

- Freeze the codec and frame encoder in eval mode; fail if any frozen parameter is trainable or
  its state hash changes.
- Use the identical DiT width/depth/heads, optimizer, learning-rate schedule, ordered sample
  schedule, and validation schedule for R2 and R4. Only the codec ratio and its latent statistics
  differ.
- Fit ratio-specific latent statistics from train only and persist their values, provenance, and
  hashes. Validation is never used to fit statistics.
- Train a deterministic balanced H=4/H=8 schedule. H=0 and H=2 are not training modes.
- Run a 200-step real-T1 profile for each candidate with frozen codec encoding, DiT train steps,
  validation RF loss, and checkpoint resume. Record seconds/step and peak memory.
- The fixed validation subset contains all eight validation systems, all three replicas, and
  windows `w000000`, `w000030`, and `w000061` (72 clips total) for both candidates.
- After both profiles pass, write one common `pilot_budget.json` with
  `S = min(5000, largest common budget fitting within 80% of the remaining overnight window)`.
  Require `S >= 1000`; otherwise stop after profiles.
- With two idle GPUs, use GPU A for R2 and GPU B for R4, seed `20260907` for both. An additional
  seed requires four idle GPUs and must be run for both candidates.
- Select checkpoints by validation RF loss only. Evaluate a fixed validation subset containing all
  eight validation systems with identical H=4/H=8 windows. Report validation RF loss, future
  aligned RMSD/dRMSD, codec-oracle generation gap, runtime, and peak memory.
- RMSF/ACF/contact outputs are secondary only. Do not use test data, select a winner, or begin a
  later architecture phase.

## Stop condition

At completion or a genuine blocker, record exact commands, hashes, profiles, losses, selected
checkpoints, and scope in `TASKS.md` and `HANDOFF.md`. Final status is
`WAITING_FOR_OPERATOR_REVIEW`.
