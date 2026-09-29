# Static decoder probe v1 independent review request

## Fixed subject

- Candidate commit: `73ba705ab4e506e3274d54b62c9a351e84127125`.
- Worktree: `/data4/users/sihao/workspace/molvid-recon-static-decoder-probe-v1`.
- Specification: `/data4/users/sihao/workspace/SESSION_D_STATIC_DECODER_PILOT_PROMPT_260929.md`.
- Historical repository base: `cc74a385577e57c64c9623111e0db86e54e48770`.
- A stable source lineage ends at reviewed source commit
  `af60c05`; its commits were replayed without editing A's worktree and end at
  local commit `cc1809d` before the final D clash/memory fixes.
- The reviewer is read-only.  It must not edit model code or run training.

## Released identities

- A recipe release:
  `/data4/users/sihao/workspace/molvid-gm-memory-v3-coordination/recipe_release.json`,
  SHA256 `38702552...` (verify the full value from the file).
- Selected evaluated A1 parent:
  `/data4/users/sihao/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/formal/A1/frame_joint_step_00010606.pt`,
  SHA256 `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`.
- A1 source config:
  `/data4/users/sihao/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/formal_A1.json`,
  SHA256 `15d553...` (verify the full value from the file).
- A released physical auxiliary recipe reused by the engineering pilot:
  `/data4/users/sihao/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json`,
  SHA256 `625d11cff4a579e96487029f58d88e8e3d6129c3123e80847cc6edb9c8a0e6f3`.
- Engineering-only P2 parent:
  `/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/formal/GM/frame_joint_step_00021208.pt`,
  SHA256 `aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540`.
- Frame teacher SHA256:
  `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`.
- Dynamic train data hash:
  `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1`.
- Normalization source data hash:
  `54bbdc198373ae9f477bf5803643aab77f960fe4498da8db1fcf9c5bfdd770c0`.

The P2 checkpoint was used only for engineering smoke/pilot because A's
physical auxiliary calibration is bound to that exact parent.  The pilot
checkpoint is marked `formal_parent_allowed=false` and must not seed a formal
arm.  Formal arms, if approved after fixes, must all start from the released
A1 checkpoint above with symmetric optimizer/scheduler/RNG reset.

## Evidence to inspect

- Replacement bounded smoke:
  `runs/static_decoder_probe_v1/smoke_memory_bound/`.
- Completed 128-update engineering pilot:
  `runs/static_decoder_probe_v1/pilot/`.
- Preserved interrupted high-reserved-memory attempt:
  `runs/static_decoder_probe_v1/pilot_interrupted_high_reserved_memory_20260929/`.
- Targeted CPU checks: 21 passed, 2 CUDA-only skipped.
- Targeted CUDA static checks: 3 passed.
- Completed pilot checkpoint SHA256:
  `09ab006447fc79a45fe11fd9a55db0d2cf445ca9d38971fa78297b7b5d8594b5`.
- Completed pilot: 128/128 successful static updates, 303.74 seconds; static
  loop allocated-memory peak 164,191,232 bytes and reserved-memory peak
  9,628,024,832 bytes.  Largest 4,975-atom sample peak allocation was
  802,583,040 bytes.
- The interrupted run was stopped by the implementer after a 58 GB cached
  workspace was observed.  The replacement selects the three cheapest complete
  train systems for full-dynamic gradient calibration and explicitly releases
  phase caches.  Its live allocated bytes remain flat across the pilot.
- Cumulative D CUDA time so far is far below the eight GPU-hour cap.
- Sealed test trajectory opened: no.

## External-data gate

An existing PDB static store is visible under
`/data4/users/sihao/data/pvb_cross_dataset_20260810/pdb_phase12_half`, and a
legacy PDBBind block store is also visible.  This is not yet declared eligible:
the audit has not found existing metadata proving near-homology exclusion
against the frozen A validation/test systems.  Direct split membership alone
is insufficient.  The reviewer should report whether any existing allowed
metadata closes this gate; it must not assume or fabricate an external arm.

## Requested decision

Review target isolation and perturbed-input consistency; T=1/time/mask units;
teacher freezing; decoder-only gradients, optimizer state, weight decay and
scheduler boundaries; A clash-rule reuse; train-only gamma calibration and its
dynamic denominator; RNG separation; checkpoint/data identity enforcement;
memory evidence; and readiness to implement/run symmetric short continuations
from A1.  Return severity-ranked findings with exact file/line references and
a PASS/FAIL verdict.  Missing formal comparison orchestration or missing
external-data eligibility must be called out explicitly rather than inferred
away.
