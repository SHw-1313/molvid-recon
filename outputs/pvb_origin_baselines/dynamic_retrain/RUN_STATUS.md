# Baseline D current status

Status: inspection in progress; no baseline-D training or evaluation metric has been claimed.

The clean reference repo is `/data4/users/sihao/workspace/PVB_origin` at commit `c08e5e3`. The requested source checkpoint directory is `/data1/repo/PVB/ckpt/misato_from_author_pretrain`. The current pilot stores are under `/data4/users/sihao/data/pvb_cross_dataset_20260810/clips` and are `npz-v1` trajectory clips, whereas original PVB `UniDataset` expects gzip-compressed JSON pair records in `data.bin` plus `index.txt`.

Next checks: decode one clip from each source, verify model/checkpoint loading in the original environment, locate or establish the explicitly authorized original-format mapping, then run only if the mapping is faithful and no PDB data enters the dataset.
