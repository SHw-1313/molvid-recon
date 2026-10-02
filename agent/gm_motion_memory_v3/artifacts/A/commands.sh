# Commands actually executed. Long formal runs were submitted only after the
# immutable review gate passed; the cost ledger preserves each resolved command.
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

# Final reviewed CUDA smoke and largest-sample profile (the resolved config,
# warm start and output paths are preserved inside their JSON artifacts).
python -u -m tools.gm_a_smoke \
  --tables runs/gm_motion_memory_v3/A/r04/physical_tables.json \
  --output runs/gm_motion_memory_v3/A/r17/smoke/smoke.json
python -u -m tools.gm_a_profile \
  --recipe runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json \
  --output runs/gm_motion_memory_v3/A/r19/profile/profile.json

# Symmetric pilot and formal sequences. All arms reload the same P2-GM parent;
# no pilot or prior arm is used as a parent.
python -u -m tools.run_gm_a_sequence \
  --phase pilot \
  --frozen runs/gm_motion_memory_v3/A/r16/configs/frozen_configs.json \
  --recipe runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json \
  --review-gate agent/gm_motion_memory_v3/reviews/A/formal_r02/review_gate.json \
  --arms A0 A1 A2 A3
python -u -m tools.run_gm_a_sequence \
  --phase formal \
  --frozen runs/gm_motion_memory_v3/A/r16/configs/frozen_configs.json \
  --recipe runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json \
  --review-gate agent/gm_motion_memory_v3/reviews/A/formal_r02/review_gate.json \
  --arms A0 A1 A2 A3

# Unified validation analysis, paired system bootstrap and all-system figures.
python -u -m tools.report_gm_a \
  --arm A0=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A0 \
  --arm A1=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A1 \
  --arm A2=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A2 \
  --arm A3=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/evaluation/A3 \
  --output /workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r21/final_analysis
