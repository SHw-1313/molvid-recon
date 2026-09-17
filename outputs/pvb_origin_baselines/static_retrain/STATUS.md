# PVB original baseline C — live status

Status: running.

## Resolved inputs

- Reference code: `/data4/users/sihao/workspace/PVB_origin`, commit `c08e5e3cd49d45c6d748387e78224843bd356f50`; the reference checkout is clean and is not edited.
- Requested checkpoint spelling was `pdbind_pretrain`; the actual directory on `/data1` is `pdbbind_pretrain` (PDBBind has two `b` characters). The selected exact checkpoint is `/data1/repo/PVB/ckpt/pdbbind_pretrain/version_1/checkpoint/epoch191_step113472.ckpt`, the top-ranked entry in its `topk_map.txt` (validation metric `0.26337328421718936`).
- Seed/policy: `20260810`, native split policy, ATLAS `dt_100ps` and MISATO `dt_80ps`, no PDB data.
- Prepared clip stores: `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/atlas/dt_100ps` and `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips/misato/dt_80ps`.

## Compatibility finding

The prepared clip stores use `storage_format=npz-v1`; their `data.bin` starts with ZIP magic `PK`, and their index has clip metadata columns. The unmodified original `PVB_origin/data/mmap_dataset.py` expects gzip-compressed JSON records with the legacy `atype/btype/x0/b0/x1/b1/edge_mask/mask/bond_index` schema. They are therefore not directly consumable by the original `train.py`.

The explicitly named host location `/data5/PVB` contains only the original-format PDB EPT stores under `pdb/ept_release/pvb_phase12_full`; those are out of scope and will not be used. The container has no `/data5` mount. The minimal in-scope repair is a read-only clip read plus a new legacy pair-store conversion written under this output directory; no file in `PVB_origin` is changed.

## Runtime

Commands run through `enter-container` in the `torch-ito` environment. The container reports Python 3.11.15, PyTorch 2.5.1+cu121, and 8 visible NVIDIA A100-SXM4-80GB GPUs.

## Current blocker / next action

No model run has started yet. The remaining compatibility blocker is the absence of non-PDB legacy pair stores. Next: convert frame 0→frame 1 from each already split clip store into output-local gzip-JSON legacy stores, validate counts and schemas, then invoke the unmodified original `train.py` through an output-local seed/config wrapper and evaluate the matching validation stores.
