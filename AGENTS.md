# Current repository instructions

The active implementation is the root `molvid/` package. Do not recreate
`src/`, `molvid/models/`, root-level legacy packages, or an archive directory
for superseded source. Historical source was intentionally purged from Git.

Read `README.md`, `TASKS.md`, `HANDOFF.md` and `docs/migration.md` before a
further cutover. Preserve unverified scientific behavior and sealed test data.
Do not claim production parity for missing real checkpoint/data comparisons.

Run any Python, test, plotting or model command via `enter-container` and
`conda activate torch-ito`. Do not start long training, upgrade dependencies,
push, or alter external reference worktrees without explicit authorization.

Historical phase-specific instructions are not part of this branch and do not
govern it. Use the curated result archive only as evidence of old experiments.

## Maintain readable model code

- Keep the shallow root `molvid/` package. Put geometry representation,
  history encoding, flow prediction, coordinate decoding, losses and
  training orchestration in their named modules. Do not recreate legacy
  source archives or a parallel model framework.
- Make tensor shapes, physical units, packed-system masks and gradient
  boundaries explicit at public interfaces. Physical time and flow time
  are different quantities; use different names.
- Keep observed conditions separate from training targets. Generation
  accepts observed context and query times, never hidden future coordinates.
- Parse configuration once. Keep loss weights in one resolved source;
  behavior must not depend on hidden defaults or scattered constants.
- Model forward methods compute model outputs; CLI files only construct
  and call components. Prefer composition and direct calls over registries,
  dynamic dispatch, monkeypatches or deep pass-through wrappers.
- Reuse compatible checkpoint weights with an explicit load report.
  Distinguish warm start, exact resume and a changed-objective continuation.
- Repair or implement the requested change before running targeted checks.
  Test meaningful numerical risks on CUDA; do not add redundant whole-suite
  gates or silently fall back to CPU model execution.
- Keep the actual architecture and one training-step dataflow readable in
  docs and the model inspector. Put transient experiment settings/status
  under the task directory, not into this instruction file.

## Task authority and review handoff

- Follow the current task's explicit authorization. An older task's pause after
  tiny/pilot is not a permanent repository requirement when the user has
  authorized the new task through training and evaluation.
- For tasks requesting independent review, use a fresh session without the
  implementation conversation, review a fixed code commit and factual evidence,
  and preserve its written findings. One session owns code fixes; the reviewer
  does not edit model code.
- Keep review findings, fix responses, code/config identities and training
  artifacts traceable. Do not treat a review of different source as approval
  for a new run.
- Repair first, then run checks targeted at the changed numerical or scientific
  risks. Whole-suite regression is not an automatic prerequisite for CUDA smoke
  or every experiment.
- Source-sampled training must expose autograd, target boundaries, physical-time
  units and RNG streams explicitly. Do not call an inference-only no-grad
  sampler and label the result end-to-end training.
