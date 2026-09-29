# Static decoder probe v1 handoff

- Branch/worktree: `feat/static-decoder-probe-v1` at
  `/data4/users/sihao/workspace/molvid-recon-static-decoder-probe-v1`.
- Base: `cc74a385577e57c64c9623111e0db86e54e48770`.
- Replayed common release and later A's independently reviewed stable source;
  local A lineage ends at `cc1809d`, sourced from reviewed `af60c05`.
- A1 is now the immutable evaluated parent: checkpoint SHA256
  `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`,
  config SHA256
  `15d55300b0bbab366f8ddd984b5c8f94b6ee94b2f6ad1d0264506e3018db1cce`,
  auxiliary SHA256
  `625d11cff4a579e96487029f58d88e8e3d6129c3123e80847cc6edb9c8a0e6f3`.
- Resource state at task start: no GPU was claimed by this lane.  A's shared
  coordination assigns GPU UUID `GPU-b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0`
  to A and forbids preemption.
- P2 engineering evidence: bounded FP32/BF16/largest-sample smoke and 128/128
  static updates completed.  The pilot checkpoint is explicitly forbidden as
  a formal parent.
- Independent review of `73ba705` is preserved at `reviews/REVIEW.md` with
  `FIX_REQUIRED`.  Its code findings have been implemented: A1 orchestration,
  full-objective gamma diagnostics, independent/resumable static scheduling and
  RNG, full-optimizer fail-closed checks, and external eligibility enforcement.
- Focused checks after the fixes: metadata/config 8 passed; Frame Joint
  contracts 19 passed; CUDA static 4 passed on GPU1.  A1 CUDA smoke and fresh
  independent re-review remain before comparison training.
- Existing PDB/PDBBind stores are ineligible because no allowed metadata proves
  near-homology exclusion against A's held-out identities.  External training
  is not authorized and will not be synthesized.
- Sealed test opened: no.
