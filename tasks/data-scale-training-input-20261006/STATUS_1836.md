# Actual implementation/submission evidence, 2026-10-06 18:36 CST

Numerical source671c462, independent exp/data-scale-training-input-20261006,
parent967114b. Runtime/tools/tests now frozen while the input audit is live.
No current training branch or GPU4–7 task was edited/stopped.

## Implemented

The existing full coordinate trainer can optionally consume the frozen full512
ConcatDataset and8192-entry matched calendar. It does not resample the calendar,
crop atoms, lower precision, alter model arithmetic or change the control recipe
outside coordinate_data_scale. Full immutable source/data SHA checks, complete
inventory and actual per-step view/lag/H/clipcount/aux equality are required.

Default recipes retain original data/schedule/budget logic. Actual default GPU
update equality has NOT yet been tested for this new source. CPU coverage is not
substituted for that gate. Expanded checkpoints retain the long-budget marker's
schedule_offset2048 explicitly as REFERENCE192 calendar provenance; candidate
selected offset0 and candidate global batch IDs are separately recorded. The
existing final8192 evaluator remains compatible without a model change.

Before model construction, expanded training requires the completed uncropped
maximum-system CUDA qualification, exact4/8depth resume, SHA-bound checkpoints,
unchanged qualified model/runtime files and actual maximum atom counts. Pending,
failed, CPU-only or smaller-system evidence cannot authorize expanded training.

## Tests and actual work

Final targeted CPU run:25passed, no skips,5.16s, inside enter-container/torch-ito.
Includes12 new input-boundary cases and13 existing long-budget/contract cases.
Earlier11/12-test runs overlap this set; do not add them as independent tests.
Synthetic gate fixtures explicitly are not actual CUDA evidence.
JUnit: `runs/data_scale_training_input_20261006/cpu_gate_final.xml`.
Both trainer and real-input audit CLI imports/help completed successfully.

Real full512 trainer-input audit actually submitted: hostPID3603708,PTY11651;
`runs/data_scale_training_input_20261006/full512_v1/status.json=waiting`.
It binds full-data qualifier containerPID778178/start/command, then will verify
both seeds' immutable bytes/calendars through the new input context and decode
real largest/first/last/per-view/H batches through ordinary ClipBatch collation.
It produces explicitly INPUT-ONLY recipes without completed CUDA gate pins;
these cannot start expanded training. No neural calls/updates are submitted.

## External progress independently verified

The complete80-shard training-data build has finished:19 verified reused shards
and61 new closed shards,zero model calls/updates. All320 additions now exist;
existing192 is unchanged. This is materialization, not training qualification.
Full512 concat qualifier is really running(host3602705), at verify_closed_bytes;
the matched-calendar stage follows automatically. No claim that these have passed.

Maximum17002-atom GPU qualification remains queued behind GPU0's complete original
train/eval chain. Native1ns long8 evaluations continue onGPU4/5; no final paired
quality result newly reported. Static update-budget71 completed2048 training and
continues evaluation; vector axial72 remains training. data4~14GB,data2~54GB free.

Scientific objective remains unachieved. Broader training coverage is useful
only if actual geometry AND motion improve under fixed comparisons. Same update
counts and temporal calendar do not equalize atom exposure/compute or make the
reused development48 panel independent. Equilibrium/complex/generalization gates
remain open; none is claimed solved by this engineering input qualification.
