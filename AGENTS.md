# Current repository instructions

The active implementation is the root `molvid/` package. Do not recreate
`src/`, `molvid/models/`, or root-level legacy packages. `old/` contains
historical evidence and obsolete entrypoints; it is not an import source.

Read `README.md`, `TASKS.md`, `HANDOFF.md` and `docs/migration.md` before a
further cutover. Preserve unverified scientific behavior and sealed test data.
Do not claim production parity for missing real checkpoint/data comparisons.

Run any Python, test, plotting or model command via `enter-container` and
`conda activate torch-ito`. Do not start long training, upgrade dependencies,
push, or alter external reference worktrees without explicit authorization.

Historical phase-specific instructions are retained at `old/AGENTS.md` and
`old/agents/`; they do not govern this branch.
