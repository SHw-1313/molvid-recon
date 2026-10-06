# Largest TRAIN input: real numerical feasibility before data-scale training

ParentGQ f8f4ce2. Model and actual coordinate_update are unchanged. No architecture
promotion, corpus cutover, protected-target opening or new formal model training.

Frozen input: largest of the already registered512 training topologies,
atlas_6sup_A,17002 heavy atoms,16 ordinary100ps frames from the real preflight
store. No cropping, atom filtering, reduced future length or precision change.
H4/Q12 and H8/Q8 each test frozen-query and trainable-query encoders. Initialize
seed71's frozen final2048 coordinate checkpoint, then optionally clone query
geometry exactly as the GQ experiment did. Unchanged FP32/checkpointed training.

Each case: two actual disposable updates in one process, then a fresh process
loads prefix1 and repeats update2. Compare canonical SHA256 over every state
tensor byte/dtype/shape, optimizer field, CPU/CUDA/flow/Python/NumPy RNG field,
contract and cursor. This is cryptographic full-state equivalence evidence,
not a direct torch.equal comparison. Only the prefix is saved as a checkpoint;
final state digests avoid writing another two large checkpoint copies per case.
All gradients/model/Adam state are checked finite; required trunk/history/
readout/query gradient groups nonzero; frozen teacher state unchanged.

Expected if all pass:12 actual discarded optimizer updates,0 formal updates,
0 protein forecasts. A failure is retained and never called zero updates merely
because no completion summary exists. All four registered cases are attempted
independently when safe, allowing failure localization; no automatic retry or
easier data fallback. Passing is NOT512 loader/packed/distributed qualification,
chemistry efficacy, thermodynamics or generalization evidence.

Queue owns only GPU7 after long8_72's complete training+192 forecasts+chemistry+
final audit; it identifies the live long-budget controller at submission and
checks its process identity while waiting. A transient idle phase is not a
handoff. All numerical source/input hashes are pinned and rechecked. Existing
other GPU lanes remain untouched.9GiB worker/10GiB controller disk floor.

Further qualifications if successful: full additional corpus hash/clock/fixed
source-index validation and truthful loader union with existing192, then an
explicit192/512 exposure/compute comparison. If failure occurs, measure the
actual failing stage before changing memory algorithms or model architecture.
