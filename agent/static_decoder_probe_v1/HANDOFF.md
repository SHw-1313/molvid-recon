# Static decoder probe v1 handoff

- Branch/worktree: `feat/static-decoder-probe-v1` at
  `/data4/users/sihao/workspace/molvid-recon-static-decoder-probe-v1`.
- Base: `cc74a385577e57c64c9623111e0db86e54e48770`.
- Replayed common release: source commits `eec76455`, `5a745960`; local commits
  `a681da6`, `228592c`.
- A state at task start: formal A1 training running; recipe/capacity releases
  absent.  B has completed bounded pilot/review and waits for A.  C is not
  started because its release condition is unmet.
- Resource state at task start: no GPU was claimed by this lane.  A's shared
  coordination assigns GPU UUID `GPU-b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0`
  to A and forbids preemption.
- Code/checks/pilot/review/formal experiments: in progress; update only with
  completed factual evidence.
- Sealed test opened: no.

