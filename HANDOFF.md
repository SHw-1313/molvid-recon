# Refactor handoff

Current stage: P0, old-path numerical reference pending. Isolated local branch `refactor/flat-layout` at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`; no production edits or legacy deletion. The old source worktree is read-only. See `docs/migration.md` for symbol ownership, active entrypoints and verified artifact hashes.

Blocker: `enter-container` invokes `sudo`, but `sudo -n` requires a password and the current user cannot access `/var/run/docker.sock`. CUDA hardware and approved codec/statistics/manifest/DiT files are present. No Python/test/numerical command has run in this session. Next: restore authorized container access, record fixed tiny old-path tensors/gradients/update/resume, then begin P1 and run its targeted checks before another phase.
