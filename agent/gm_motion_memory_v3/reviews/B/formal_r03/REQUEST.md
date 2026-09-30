# Independent B formal OOM-fix review request

Review fixed commit `5092dcbe0e70f8864d72e0d13d3301db7d46929c` without editing model code.
The prior reviewed implementation is commit
`5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6`; intervening commit
`0af49b1f893a851af951f86394fe94a43f506179` changed documentation only.

The first M-short formal invocation stopped after 194 successful updates when
batch index 194 exhausted GPU3 memory. Preserve that failed attempt and its
paired-budget charge. The candidate changes only the streamed K=2 backward:
base and each Euler-8 source branch use detached leaf views of the already
computed learned-memory outputs, their output gradients are summed, and the
sum is propagated through the original observed-prefix graph once. The review
must determine whether this is gradient-equivalent, keeps one memory
preparation per prefix, and avoids detaching learned memory from training.

Review these files and their history:

- `molvid/training/joint.py`
- `tests/test_learned_memory.py`
- `tools/profile_learned_memory_formal_batch.py`
- `runs/learned_memory_v1/formal_r02/oom_fix_verification_5092dcbe.json`
- `runs/learned_memory_v1/formal_r02/oom_fix_profile_5092dcbe_gpu2/profile.json`
- `runs/learned_memory_v1/formal_r02/oom_fix_threshold_profile_5092dcbe_gpu2/profile.json`
- `runs/learned_memory_v1/formal_r02/configs/formal_M-short.json`
- `runs/learned_memory_v1/formal_r02/configs/formal_M-long.json`
- `runs/learned_memory_v1/formal_r02/configs/formal/M-short/cost_ledger.json`
- `runs/learned_memory_v1/formal_r02/configs/formal/paired_formal_cost_ledger.json`

Verify commit/config/profile hashes, numerical semantics, metric accounting,
CUDA headroom (including the largest nonstreamed scheduled auxiliary batch),
formal budget continuity, and whether a fresh M-short launch followed by
M-long is approved. Write the independent result to
`agent/gm_motion_memory_v3/reviews/B/formal_r03/REVIEW.md` with severity-ranked
findings and an explicit PASS/FAIL launch decision. Do not modify model code.
