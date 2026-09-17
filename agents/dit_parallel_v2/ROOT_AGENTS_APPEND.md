# 根 AGENTS.md 的追加模板

只追加下面从二级标题开始的内容到自己工作树的根 AGENTS.md。不得替换原文件。最后再加一行 `Local session: A` 或 `Local session: B`；不要两个都填。

## Active transition — DiT pilot diagnostics / factorized backend v2, 2026-09-09

The operator has authorized two independent implementation lanes. This transition supersedes previous active-phase task order, target worktree, output-root and stop instructions only for these lanes. All historical artifacts, frozen codec semantics, data-access boundaries and environment safety remain in force.

- Session A stays on `exp/dit-state-detail-t1-pilot-v1` in the existing pilot worktree. It must not create or switch branches. Read `LUNA_SESSION_A_DIT_DIAGNOSTICS_PROMPT_260909.md`, `agents/dit_parallel_v2/CONTRACT.md`, and every instruction file in `agents/dit_pilot_diagnostics_v1/`. Implement and run bounded checkpoint diagnostics; no training or backend edits.
- Session B uses the separate `perf/dit-factorized-backend-v2` branch/worktree from audited commit `a22f60c4ffd1f502ab300352a00eafe841b1f8d2`. Read `LUNA_SESSION_B_DIT_BACKEND_PROMPT_260909.md`, the shared CONTRACT, and every instruction file in `agents/dit_factorized_backend_v2/`. Implement numerical-equivalent acceleration and bounded CUDA tests/profiles; no scientific retraining or evaluator changes.
- New lane-local AGENTS/PLAN/ACCEPTANCE/TASKS/HANDOFF govern current work. Old `agents/AGENTS.md` codec-v1 task descriptions, causal-tokenization rules, CPU-first prescriptions and legacy phase stops are historical, not instructions to restart those experiments. Keep 16-frame block-local state/detail tokenization, R2/R4, H4/H8 and current model contracts.
- All Python and scientific numeric work uses the configured `enter-container` / `torch-ito` environment. New numerical acceptance tests and real-data model paths run on CUDA. Pure metadata, I/O and reporting may use CPU. A skipped CUDA test is not evidence.
- A and B own disjoint source files as specified in CONTRACT. Add only your own local root transition; do not edit another worktree. Read-only artifacts may be referenced by absolute path. Never open test clip payloads, overwrite old outputs, terminate unrelated processes or bypass permission failures.
- Implement first, then targeted CUDA checks, real-clip smoke, relevant regression and bounded evaluation/profile. Do not run the full suite as a prerequisite to implementation. Do not silently change physics, latent statistics, optimization objectives or sampling priors during a performance refactor.
- Local commits of owned code and small evidence are authorized. No automatic push, merge, PR or long training. Stop at `A_DIAGNOSTICS_READY_FOR_REVIEW` / `B_BACKEND_READY_FOR_REVIEW`, or report incomplete evidence honestly.
- `agents/dit_parallel_v2/NEXT_STAGE.md` is a future experiment design, not permission to start it. Both Gaussian-source and conditional-source arms retain the same observation history; no H0 training is authorized in this task.
