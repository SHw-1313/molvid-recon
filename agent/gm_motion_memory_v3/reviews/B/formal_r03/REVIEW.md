# B learned-memory formal OOM-fix independent review r03

## Decision

**PASS / APPROVED FOR FORMAL LAUNCH** on fixed commit
`5092dcbe0e70f8864d72e0d13d3301db7d46929c`, using the unchanged frozen
M-short and M-long configurations listed below. Launch a fresh M-short first;
launch M-long only after the paired ledger records a complete 9,741-update
M-short invocation.

Two operational preconditions are mandatory:

1. Preserve the failed 194-update artifact directory losslessly by renaming
   the existing `configs/formal/M-short` directory to a failure-specific
   sibling before the fresh launch. Leave
   `configs/formal/paired_formal_cost_ledger.json` at its current path and do
   not delete, replace, or edit it. The runner deliberately uses
   `exist_ok=False` for a fresh arm, so launching against the currently
   occupied M-short path would fail closed before opening a new invocation.
2. Use a reserved 80 GiB GPU with one visible CUDA device and no memory-heavy
   co-tenant. The largest scheduled nonstreamed auxiliary batch peaks at
   70.099 GiB allocated, leaving 9.152 GiB against the 79.25 GiB device. The
   previous OOM included another process holding 12.63 GiB; that occupancy is
   incompatible with the verified nonstreamed envelope.

These are launch conditions, not requests to change model code or a license to
change either frozen configuration.

## Reviewed identity and evidence

- Reviewed HEAD and commit: `5092dcbe0e70f8864d72e0d13d3301db7d46929c`;
  parent: documentation-only approved HEAD
  `0af49b1f893a851af951f86394fe94a43f506179`; previously reviewed
  implementation: `5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6`.
- The fixed commit changes only `molvid/training/joint.py`,
  `tests/test_learned_memory.py`, and
  `tools/profile_learned_memory_formal_batch.py`. The checked-out blobs match
  that commit, and `git diff --exit-code HEAD -- molvid tools tests` passes.
  Uncommitted coordination/review documents and `runtime_paths.json` are not
  imported by the formal runner.
- M-short config SHA-256:
  `7d103a90dbd80dfd013c5953c9aaae8982eb5a2c32b778c2057850ea939f608d`.
  M-long config SHA-256:
  `5a18ff4210e600c40d1f69121785bd37ae5407aaca50f2f03ca440d895282321`.
  Removing only `arm` and `training.output_root` makes both resolve to pair
  contract `99a73d99c2b2a3be9bdc661b6595c86aba1029701d113f82e7d6ab63e7ee06b6`,
  exactly matching the durable paired ledger.
- OOM-fix verification SHA-256:
  `eef5a51e4bf9fbeeaac070cff308dfe684710643dc53dded1c5d4ca5582cef48`.
  Main profile SHA-256:
  `ec1a8f9bbf36ae42d70903853703a8657560e2444c13b8d5623771476e577f43`.
  Threshold profile SHA-256:
  `05bed3987ebd3746d87539dc434bd98429dc0d258d4ce9e7edd2ec6e94061ee0`.
  Both profile files record the fixed commit, the exact M-short config hash,
  completed/non-formal status, the same execution hash
  `b684a7852bef7aad6beff0539e0b6aa73bc4f8fbddd884f30d0ff2bb41d9beee`,
  and paired schedule hash
  `604ef984a13f2e065ee7049518dc2f48e349bc13206b8ef081b5fcf94eb9627c`.
- I independently reran, through `enter-container` / `torch-ito` on CUDA UUID
  `GPU-4172773f-10fa-ce97-72e6-1c582bdcb615`, the cited targeted suite:
  `tests/test_learned_memory.py`, `tests/test_gm_common_schedule.py`, and
  `tests/test_learned_memory_formal_safety.py`: **15 passed in 13.99 s**.
  No formal training or sealed-test access occurred.

## Numerical and gradient semantics

Let the single observed-prefix preparation produce memory outputs
`m = (h, v, slot_weights)` from learned-memory parameters `theta`. The fixed
path makes detached, gradient-requiring leaf views `m_leaf`, runs the base
loss and each of the two Euler-8 source branches against those leaves, and
collects

`g = dL_base/dm + dL_source0/dm + dL_source1/dm`.

It then calls one `autograd.backward(m, g)`. By the chain rule, the resulting
history gradient is `g * dm/dtheta`, the same objective derivative as the old
retained-graph sequence. DiT and decoder parameters still receive their base
and per-source gradients directly from each branch backward before clipping
and the single optimizer step.

The two source passes also preserve the K=2 cross-draw objective. Each pass
holds the other source at the exact detached coordinate value from the initial
no-grad draw evaluation, so the two passes compute the two partial derivatives
of the same loss at the same `(source0, source1)` point. Raw sampled metrics are
taken once from that point, while the weighted training terms use the same
four frozen physical weights. Base and sampled metrics remain disjoint and
their detached sum is the reported total.

All trainable condition-output paths used by FrameDiT are proxied: `h` and `v`
feed gathered memory values, and `slot_weights` feeds query-time features and
distance. The remaining `GroupMemory` members are mapping or observed-time
metadata. The proxy rejects missing autograd, the original graph remains live
until the final backward, and both fixed profiles report finite gradients for
every learned-memory parameter. Thus learned memory is not detached from
training.

The objective and chain-rule semantics are exact. Byte-for-byte gradients are
not claimed because summing BF16 boundary gradients before one upstream
backward changes floating-point accumulation order relative to three upstream
backwards. The CUDA test covers both FP32 and BF16 autocast and accepts only
the declared small numerical bounds (`3e-4` for BF16, tighter for FP32); it
passed independently.

## One preparation and CUDA envelope

`prepare_condition` is called once before proxy construction. The base pass,
the detached value passes, and both differentiable Euler branches reuse the
same proxy; none recomputes the encoder. The formal runner independently hooks
the history encoder and rejects any update whose preparation count is not one.
Every profiled row records one preparation and teacher calls `[2, 12]`.

I independently rebuilt epoch 0 from the frozen dataset and sampler. It has
9,741 batches and 1,215 scheduled auxiliary batches and reproduces both hashes
above. The 4,975-atom maximum and 4,125-atom maximum below the streaming
threshold each occur at several schedule positions; the profiled indices 900
and 3274 are valid representatives of those tied maxima. With Q=12 and a
50,000 atom-frame threshold, 4,125 atoms means 49,500 and therefore exercises
the unchanged nonstreamed path.

- Former OOM batch 194: old peak 65.679 GiB; fixed streamed peak 41.438 GiB,
  a 36.9% reduction. The update completed with one preparation and finite
  history gradients.
- Largest streamed atom batch, index 900: 4,975 atoms, fixed peak 48.801 GiB.
- Largest nonstreamed scheduled auxiliary batch, index 3274: 4,125 atoms,
  fixed peak 70.099 GiB on a 79.25 GiB A100, with 9.152 GiB allocated
  headroom and 8.122 GiB device-free after the update.

The pre-fix 128-update paired pilots already showed nearly identical short and
long peaks (53.617 versus 53.621 GiB) despite M-long reaching 456,918 valid raw
atom-frames. The fixed path retains only one original condition graph and
removes the simultaneously retained base/Euler graphs; it applies to either
arm. Together with the worst scheduled nonstreamed profile, this is sufficient
for an exclusive 80 GiB launch. It does not support launch with the 12.63 GiB
co-tenant present at the failed attempt or on a smaller GPU.

## Failed-attempt and budget continuity

The failed M-short directory contains 194 metric rows ending at step 194. Its
arm ledger records `failed_or_interrupted`, OOM at next batch index 194, peak
`70522501632` bytes, and no resumable checkpoint. The paired ledger records the
same invocation/config/commit, 194 successful updates, and charges
`782.675279378891` seconds = `0.21740979982746972` GPU-hours. Its SHA-256 is
`eb1c58f4eddfad67c8cee7b764ca854c7d8937ec7c696105681aa6a8ea28628e`,
and the arm ledger points to that exact digest.

The pair remains `active`. A fresh M-short invocation using the same pair
contract appends a new record and computes its cap from the prior charged
seconds, so the failed cost is preserved. Completion means exactly 9,741
successful updates in the new invocation; the discarded 194-update model is
not treated as a continuation. M-long is rejected until a complete M-short
record exists. The existing pre/post-update cap guards, checkpoint-safe stop,
strict resume state, and 32 GPU-hour pair cap remain unchanged from the r02
review.

## Severity-ranked findings

### P0

None.

### P1

None in the fixed model/training implementation or supplied evidence.

### P2

The failed `configs/formal/M-short` output path is still occupied and contains
the only per-update record of the failed attempt. A fresh invocation at the
frozen output path will fail closed until that directory is preserved under a
different sibling name. Do this before launch; keep the paired ledger in its
current parent and byte-identical. This is an operational preparation issue,
not a reason to alter code or configuration.

### P3

The worst nonstreamed evidence has 9.152 GiB allocated headroom, so the launch
must retain the reviewed one-visible-device, exclusive 80 GiB execution
condition. Peak evidence is tied to the fixed shapes, frozen warm start and
profiled states; it is not a guarantee for arbitrary co-tenancy or a different
device class.

## Launch authorization

After the two stated operational preconditions, launch fresh M-short from the
immutable A1 parent with config SHA
`7d103a90dbd80dfd013c5953c9aaae8982eb5a2c32b778c2057850ea939f608d`
at commit `5092dcbe0e70f8864d72e0d13d3301db7d46929c`. Preserve the paired ledger
through completion. Then, and only then, launch M-long with config SHA
`5a18ff4210e600c40d1f69121785bd37ae5407aaca50f2f03ca440d895282321`.
Any code/config change, ledger replacement, resumed use of the failed
194-update state, nonexclusive GPU placement, or M-long launch before completed
M-short falls outside this PASS.
