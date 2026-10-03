REVIEW_LANE: B
REVIEWED_COMMIT: d104135e89804d841943d08309d8a5660642e3fa
VERDICT: BLOCKED_ENV

## Scope and reviewed evidence

Fresh independent read-only static re-review of B-R01-01 and adjacent bounded pilot/profile changes from `43bbadf0491c8b3815be03ae1f3ad0114484227a`. HEAD matched the candidate SHA. `git diff --exit-code HEAD -- molvid tools tests` confirmed the inspected implementation/test source matches that candidate. No implementation conversation, writer report, implementation_status, SELF_REVIEW, or personal memory was read. Only this report was written.

Read the root/task AGENTS, supplied review prompt/request, EXPERIMENTS, MEMORY_DESIGN, DATA_TRAIN_EVAL, WORKFLOW, experiment_spec, pilot config, prior independent REVIEW and FIX_RESPONSE. Inspected the exact four-file implementation/test diff, full pilot launcher, memory attention/readout, CUDA fixture/test, and trainer gradient/optimizer sequence.

Factual evidence:

- `B/data_checks.json` reports data IO/indexing checks only, 198 paired-span checks, nine history-dt/query-lag combinations, no sealed-test access, and no model execution. These results were not rerun.
- Independently hashed both `B/data_card.json` and the configured `runs/learned_memory_v1/raw_pilot_r01/data_card.json`: both are `235a671c01f48b367656733f0174d45aaad9a28ed774ebc311b01e85afe42a3c`, matching the config.
- `B/artifact_identities.json` records parent `aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540`, teacher `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`, statistics-file `117dbcb674f9932c40216c4e2c3ee69153dacc3d3efebcca6e67c876a3b6bee9`, and semantic statistics `7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`, consistent with the supplied specification/config. I did not independently deserialize or rehash those external checkpoints.
- Coordination `resources.json` still has `assigned_gpu_uuids.B=[]`. Preserved `B/pilot_preflight.log` shows the earlier launcher stopping at this resource gate. Its old line numbers and invocation without the newly required `--code-commit` mean it is historical environment evidence, not an execution of this candidate.
- The working-tree `B/loss_manifest.json` is modified relative to HEAD. I inspected the committed version explicitly; the candidate launcher itself resolves loss from the SHA-checked parent and writes a fresh runtime manifest. Do not attribute the uncommitted manifest to this reviewed commit.

No Python, test, model, or CUDA command was run. There was no CPU model fallback. No numerical PASS, measured profile, real warm-start result, or post-pilot approval is claimed.

## Blocking status

No remaining source blocker was found within this focused repair scope. B-R01-01 is repaired statically. Numerical acceptance remains **BLOCKED_ENV** because B has no assigned GPU and no repaired-candidate CUDA evidence is supplied.

### B-R01-01 — repaired in source; numerical verification pending

**Locations:** `molvid/codec/learned_memory.py:290–292`; `tests/test_learned_memory.py:70–92`; `tools/pilot_learned_memory.py:167–178`.

The shared additive bias in `key` and the final `label_key` layer is now disabled. This removes the two unidentifiable offsets that cancel in source softmax. The retained first label-layer bias precedes SiLU and is not the same redundant linear offset.

The test and pilot now require finite gradients for participating memory parameters and nonzero aggregate norms for the light encoder, keys, physical-label path, query slots, values, and output paths. They no longer require each individual tensor to have nonzero gradient. The CUDA test is parameterized over FP32/BF16 and checks that an actual optimizer step changes both query slots and a light-message weight. The fixture includes mixed lengths and a three-point all-detail-masked system. This directly addresses the reported false gradient gate; no further source fix is requested for B-R01-01.

The trainer leaves gradients available after its optimizer step, so the pilot's new inspection reads the current step's gradients. `molvid/training/joint.py:807–808` only adds a profiler range around the existing backward call; it does not detach or change the objective.

**Minimum check:** On an explicitly assigned CUDA slot, run the focused FP32/BF16 backward test and one real parent-warm-start trainer update. Preserve finite loss/gradient evidence, nonzero path norms, actual light/slot updates, load report, optimizer ownership, and frozen-teacher identity. Then complete the bounded pilot. A whole-suite regression is not required.

## Adjacent changes and nonblocking notes

- `tools/pilot_learned_memory.py:28–35,61–70` requires the requested fixed commit, rejects tracked source changes, records code/config identity, and verifies the actual data card as well as banks/checkpoints. This repairs the previous traceability gap for ordinary tracked-source execution. Preserve these generated identities with subsequent evidence.
- `tools/pilot_learned_memory.py:98–119,165–166` retains two-observed/Q12 teacher budget checks, same-origin split/full latent checks, and future mutation checks. The repair does not change observation/future boundaries, packed mapping, physical-time analysis, generation interfaces, teacher freezing, loss resolution, or formal short/long scheduling. These unchanged paths retain their prior static assessment; they have not acquired numerical evidence through this review.
- `tools/pilot_learned_memory.py:125–141,153–163,185–203` now executes median/largest profile-bank updates, covers Th9/33/129 at dt100 and Th33 at dt400, and records observed/target teacher, graph/packing, backbone, light encoder, analysis, memory/readout, DiT, decoder and backward ranges. Profile mode performs optimizer updates and declares that fact. Timings are inclusive/overlapping and should not be summed as disjoint costs. Warm repeats and mixed-dt packed profiling remain explicitly incomplete; these are required evidence before post-pilot approval, not a reason to prohibit the first bounded CUDA run.
- `tools/pilot_learned_memory.py:47–60,143–147` adds aggregate prior-ledger accounting and a next-step time estimate. This is a soft stop, not a strict wall-clock cap: setup is checked only before entering updates, the estimate uses the previous step, a larger next shape may cost more, and atexit does not survive forced termination. No current overrun is evidenced. Keep headroom and externally reconcile any interrupted run before another launch; do not describe this as a guaranteed hard 8-GPUh enforcement. Minimum verification is a small-budget controlled stop plus an ordinary exception, confirming persistent cost accounting before a retry.
- No formal M-short/M-long fairness or A-recipe equivalence is established by this pilot. Its base-only auxiliary-disabled objective is explicit. The profile and training outputs are not formal parents.

## Minimum remaining evidence and permitted continuation

Data auditing and preparation may continue. Once an exclusive B GPU assignment exists, this repaired source may proceed to targeted CUDA tests, real teacher parity/future-mutation checks, the bounded pilot, and size profiles under the recorded code/config identities. Preserve teacher counts/hash, gradients/updates, SO(3), permutation, padding/support-mask and physical-time results, generation/prepared-condition reuse, peak VRAM, timing and GPU cost. Complete warm and mixed-dt packed profiles before claiming the profiling requirement satisfied.

Formal M-short/M-long training remains dependent on a completed immutable A recipe, paired exposure/RNG configuration, completed CUDA/pilot/profile evidence, and fresh independent review tied to the final code/config identity. This report closes the original source defect only; it is not post-pilot PASS or approval of scientific performance.
