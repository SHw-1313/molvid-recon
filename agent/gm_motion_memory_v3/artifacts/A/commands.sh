# Commands actually executed; verification replay, not a formal-training launcher.
# Host:
# enter-container
# Container:
conda activate torch-ito
cd /workspace/molvid-recon-gm-motion-v3
sha256sum /workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/formal/GM/frame_joint_step_00021208.pt
python -c 'import torch,json; p=torch.load("/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/formal/GM/frame_joint_step_00021208.pt",map_location="cpu",weights_only=False); print(json.dumps(p["contracts"],indent=2,default=str))'
python -m compileall -q molvid/losses/sampled_physical.py molvid/training/execution.py molvid/training/joint.py molvid/cli/train_frame_joint.py molvid/evaluation/families.py tools/recompute_frame_gm_metrics_v3.py
python -m pytest -q tests/test_gm_common_schedule.py --disable-warnings
sha256sum /workspace/molvid_recon/runs/frame_joint_v1_multitime/frame_statistics.pt /workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/data/fixed_history_train/fixed_history_train_manifest.json /workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/data/fixed_history_train/clip_store/train/index.txt
python -m tools.inspect_gm_execution --config configs/frame_gm_p2_formal_GM_260923.yaml --output agent/gm_motion_memory_v3/artifacts/A/metadata
