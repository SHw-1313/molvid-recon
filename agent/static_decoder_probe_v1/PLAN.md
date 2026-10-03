# Static decoder probe v1 plan

## Scope

Test whether a single-frame geometry denoising/reconstruction update on the
shared trajectory decoder improves dynamic generation without erasing
system-level, atom-level, or multi-lag motion variation.

The static path is `perturbed coordinates -> frozen frame teacher -> shared
decoder -> restored coordinates`.  It does not call the history encoder,
motion encoder, or DiT, and clean coordinates are loss targets only.

## Execution order

1. Record repository, A/B/C coordination, artifact, data, and resource state.
2. Implement the standalone static record/batch/loss/update interfaces and
   targeted tests without weakening any dynamic history/future checks.
3. Run three-system CUDA FP32/BF16 checks and a bounded 100--200-update
   engineering pilot when a coordinated GPU is available.
4. Fix a candidate commit and review request.  Use a fresh read-only reviewer
   with no implementation context, preserve the report, fix every issue, and
   minimally recheck changed numerical behavior.
5. After A publishes an immutable evaluated recipe release, lock its code,
   parent checkpoint, loss/data/statistics/config identities and profile the
   matched three-arm screen.  Run 1000 dynamic updates per arm if the total is
   within eight GPU-hours, otherwise set 500 for all arms before seeing results.
6. Evaluate dynamic generation with A's frozen geometry/motion/multi-lag
   protocol and write the report and manifests.

## Hard boundaries

- Do not stop, restart, edit, or replace A/B/C worktrees, processes, configs,
  checkpoints, or resource locks.
- Do not open sealed test data, download an external structure corpus, upgrade
  dependencies, push, or start long training outside the stated budget.
- Static updates change decoder parameters only.  Teacher, history, motion,
  and DiT parameters receive neither gradients nor optimizer/weight-decay
  updates.
- The formal arms start from one released A parent with symmetric reset and
  identical dynamic RNG/order.  Static noise has a separate RNG stream.
- External-static training is not fabricated if an authorized compatible
  corpus is unavailable.

