# B implementation checkpoint

Private module candidate; NOT integrated and NOT CUDA-validated.

- `molvid/codec/learned_memory.py`: raw observed interface, residue/ligand-atom mapping, shared two-layer covalent scalar64/vector32 encoder, physical local Haar/quadrature analysis, recent and level fallback, eight scalar128/vector64 learned slots, local atom gather. M-index changes analysis clock only.
- `molvid/codec/memory_teacher.py`: private frozen two-anchor and Q-target encoding functions with shared observed-only reference origin; pending integration into the existing batch preparation. No full raw history enters these functions. This is not a claim that the old `prepare_frame_joint_batch` path has been replaced yet.
- `molvid/data/raw_history.py`: native 10ps index contracts, equal long-past eligibility for short/long, independent history dt and query lag, observed-reference alignment.
- `tools/audit_learned_memory_raw.py`: bounded raw pilot plus 192 train topology size census. No test trajectories opened.
- `tests/test_learned_memory.py`: targeted CUDA checks authored, not run while B has no allocated GPU.

Remaining integration after immutable A common release: replace old whole-clip preparation in the B backend, condition type and DiT group-slot readout, replace old R4 history/M, optimizer/load report, observed-only generation and actual-source prepared graph reuse. No parallel trainer will be introduced.

Current resource blocker: A resources.json says all eight GPUs are occupied by other users; assigned B UUID list is empty. No GPU workload started. Formal additionally requires completed evaluated A recipe and fresh review of integrated, piloted source.
