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
# Actual repaired-candidate preflight (resource-blocked, no CUDA model execution):
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit d104135e89804d841943d08309d8a5660642e3fa > agent/gm_motion_memory_v3/B/preflight_d104135.log 2>&1
# GPU0 continuation, explicitly authorized by the user; all inside torch-ito container.
export CUDA_VISIBLE_DEVICES=GPU-b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0
export OMP_NUM_THREADS=4
flock -n /workspace/molvid-gm-memory-v3-coordination/B_gpu.lock python -m pytest -q tests/test_learned_memory.py > runs/learned_memory_v1/cuda_smoke_r01.log 2>&1
# After exact-static repair:
flock -n /workspace/molvid-gm-memory-v3-coordination/B_gpu.lock python -m pytest -q tests/test_learned_memory.py > runs/learned_memory_v1/cuda_smoke_r02.log 2>&1
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit 62997a1bd029c7a9fd8f2461d8255710ca0c3ae1 --gpu-authorization /workspace/molvid-gm-memory-v3-coordination/B_gpu_authorization.json > runs/learned_memory_v1/cuda_pilot_r01.log 2>&1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit 0a556d7abc339f50f30472528a6e512b42e8858a --gpu-authorization /workspace/molvid-gm-memory-v3-coordination/B_gpu_authorization.json > runs/learned_memory_v1/cuda_pilot_r02.log 2>&1
# After chemistry vocabulary repair:
flock -n /workspace/molvid-gm-memory-v3-coordination/B_gpu.lock python -m pytest -q tests/test_learned_memory.py > runs/learned_memory_v1/cuda_smoke_r03.log 2>&1
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit 52617d23f79f743fd585e9fe9e143ca3e2768199 --gpu-authorization /workspace/molvid-gm-memory-v3-coordination/B_gpu_authorization.json > runs/learned_memory_v1/cuda_pilot_r03.log 2>&1
python -m tools.check_learned_memory_execution --config agent/gm_motion_memory_v3/B/pilot_config.json --checkpoint runs/learned_memory_v1/cuda_pilot_r03/pilot_model.pt --output runs/learned_memory_v1/execution_r01 --gpu-authorization /workspace/molvid-gm-memory-v3-coordination/B_gpu_authorization.json > runs/learned_memory_v1/execution_r01.log 2>&1
python -m tools.pilot_learned_memory --config agent/gm_motion_memory_v3/B/pilot_config.json --code-commit 3a484c88252ab8a6b953dd3616d8b130e183bd2d --gpu-authorization /workspace/molvid-gm-memory-v3-coordination/B_gpu_authorization.json --profile-only > runs/learned_memory_v1/cuda_pilot_r03_profile.log 2>&1
# After profile completion: rebuild factual task ledgers from artifacts and CPU annotations.
python agent/gm_motion_memory_v3/B/summarize_cuda.py
