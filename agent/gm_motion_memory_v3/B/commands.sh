# Actual commands run inside enter-container, conda activate torch-ito.
cd /workspace/molvid-recon-learned-memory-v1
python -m py_compile molvid/codec/learned_memory.py
python -m tools.audit_learned_memory_raw --manifest /workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/manifest.json --raw-root /data1/repo/BioKinema/data_atlas --output runs/learned_memory_v1/raw_pilot_r01 > runs/learned_memory_v1/raw_pilot_r01.log 2>&1
python -m py_compile molvid/codec/learned_memory.py molvid/codec/memory_teacher.py molvid/data/raw_history.py molvid/model.py molvid/dit/frame.py molvid/training/batches.py molvid/training/joint.py molvid/generation.py tools/pilot_learned_memory.py tests/test_learned_memory.py
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json > runs/learned_memory_v1/pilot_preflight.log 2>&1
# Above preflight: BLOCKED, no assigned GPU UUID; no CUDA model command was run.
PYTHONPATH=. python agent/gm_motion_memory_v3/B/check_data.py
# Prepared next commands (NOT RUN), only after resources.json assigns a free B UUID:
# CUDA_VISIBLE_DEVICES=<assigned UUID> python -m pytest -q tests/test_learned_memory.py
# CUDA_VISIBLE_DEVICES=<assigned UUID> python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit <fixed candidate>
# CUDA_VISIBLE_DEVICES=<assigned UUID> python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit <fixed candidate> --profile-only
