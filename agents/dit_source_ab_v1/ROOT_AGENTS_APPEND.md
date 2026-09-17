## Active transition — DiT source A/B v1, 2026-09-10

The operator has authorized the next stage on the existing `perf/dit-factorized-backend-v2` worktree. Read `LUNA_DIT_SOURCE_AB_V1_PROMPT_260910.md` and the instruction files under `agents/dit_source_ab_v1/` before implementation. This transition supersedes previous active-phase task order, lane ownership restrictions, no-training stops and old pilot step limits only for the work explicitly defined here.

- Stay on B; do not create a branch/worktree or merge A wholesale. Import only the specified diagnostics files from the fixed A commit. A and all historical experiment artifacts remain read-only.
- Integration fixes, targeted CUDA checks, source-center validation, bounded R4 Gaussian/conditional-source training, evaluation and local commits are authorized. Once prerequisite correctness checks pass, continue without stopping at the previous A/B review states.
- Preserve frozen codec/geometry/decoder/statistics, 16-frame tokenization, R4, H4/H8 and train/validation-only data access. The new source experiment is defined by `EXPERIMENT.md`; no additional architecture or loss changes are authorized.
- Use enter-container/torch-ito, actual CUDA for numerical work, and audited idle GPUs. Metadata and I/O may use CPU; numerical CPU fallback is prohibited.
- Implement before testing; targeted CUDA checks precede real smoke and directly affected regression. Do not start with full-repository pytest.
- New artifacts belong under `outputs/dit_source_ab_v1/`. Do not overwrite old outputs or open test payloads. Do not push, create PRs, merge/rebase or start extra coding agents.
- Follow the new budget and final-status rules. Finish at `SOURCE_AB_V1_COMPLETE_FOR_REVIEW`, `SOURCE_AB_V1_PARTIAL_BUDGET` or `SOURCE_AB_V1_BLOCKED` with evidence.
