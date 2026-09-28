# Actual commands run inside enter-container, conda activate torch-ito.
cd /workspace/molvid-recon-learned-memory-v1
python -m py_compile molvid/codec/learned_memory.py
python -m tools.audit_learned_memory_raw --manifest /workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/manifest.json --raw-root /data1/repo/BioKinema/data_atlas --output runs/learned_memory_v1/raw_pilot_r01 > runs/learned_memory_v1/raw_pilot_r01.log 2>&1
