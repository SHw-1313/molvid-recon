#!/usr/bin/env bash
# Canonical record of the commands used for D. All Python/model/test commands ran
# inside `enter-container`, conda environment `torch-ito`, with GPU1 assigned.
# Output directories below contain immutable evidence; do not rerun in place.

set -euo pipefail

conda activate torch-ito
cd /workspace/molvid-recon-static-decoder-probe-v1
export PYTHONPATH=/workspace/molvid-recon-static-decoder-probe-v1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

parent=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/formal/A1/frame_joint_step_00010606.pt
parent_sha=52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9
recipe=/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json

# Formal paired training. The first baseline attempt OOMed while two unrelated
# jobs coexisted on GPU1; no process was signalled. These are the successful
# launches, all from the same released-A1 parent.
python -u -m molvid.cli.train_frame_joint \
  --config /workspace/molvid-recon-static-decoder-probe-v1/agent/static_decoder_probe_v1/configs/formal_baseline_500.json \
  --warm-start "$parent" --warm-start-sha256 "$parent_sha" --checkpoint-every 500

python -u -m molvid.cli.train_frame_joint \
  --config /workspace/molvid-recon-static-decoder-probe-v1/agent/static_decoder_probe_v1/configs/formal_internal_md_500.json \
  --warm-start "$parent" --warm-start-sha256 "$parent_sha" --checkpoint-every 500

python -u -m molvid.cli.train_frame_joint \
  --config /workspace/molvid-recon-static-decoder-probe-v1/agent/static_decoder_probe_v1/configs/formal_external_pdb_500.json \
  --warm-start "$parent" --warm-start-sha256 "$parent_sha" --checkpoint-every 500

# Full one-common-seed dynamic evaluation. The same six commands were first run
# with `--limit 1` into eval_preflight_e64bd52/<arm>/<family>.
ordinary_store=/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views/clip_store/valid
ordinary_manifest=/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views/paired_view_manifest.json
fixed_store=/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views/clip_store/valid
fixed_manifest=/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views/fixed_history_manifest.json

python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/baseline/frame_joint_step_00000500.pt --checkpoint-sha256 f4d0ccd956aeb6b2a6593b3256da55f942d8a2ccb5ca937f8c594b1928a77874 --recipe "$recipe" --store "$ordinary_store" --manifest "$ordinary_manifest" --output runs/static_decoder_probe_v1/eval_seed0/baseline/ordinary --seeds 0
python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/baseline/frame_joint_step_00000500.pt --checkpoint-sha256 f4d0ccd956aeb6b2a6593b3256da55f942d8a2ccb5ca937f8c594b1928a77874 --recipe "$recipe" --store "$fixed_store" --manifest "$fixed_manifest" --output runs/static_decoder_probe_v1/eval_seed0/baseline/fixed --seeds 0
python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/internal_md/frame_joint_step_00000500.pt --checkpoint-sha256 90e80ba76d968f155cc5b04bd1683b6fc31463ac2959fec4ca2b52889bf61d99 --recipe "$recipe" --store "$ordinary_store" --manifest "$ordinary_manifest" --output runs/static_decoder_probe_v1/eval_seed0/internal_md/ordinary --seeds 0
python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/internal_md/frame_joint_step_00000500.pt --checkpoint-sha256 90e80ba76d968f155cc5b04bd1683b6fc31463ac2959fec4ca2b52889bf61d99 --recipe "$recipe" --store "$fixed_store" --manifest "$fixed_manifest" --output runs/static_decoder_probe_v1/eval_seed0/internal_md/fixed --seeds 0
python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/external_pdb/frame_joint_step_00000500.pt --checkpoint-sha256 9004d3e02aeabf4c14b993c3faccdf479f6ab8d70f3265f4f5881e12ed1adf94 --recipe "$recipe" --store "$ordinary_store" --manifest "$ordinary_manifest" --output runs/static_decoder_probe_v1/eval_seed0/external_pdb/ordinary --seeds 0
python -u -m tools.evaluate_gm_a --checkpoint runs/static_decoder_probe_v1/formal_500/external_pdb/frame_joint_step_00000500.pt --checkpoint-sha256 9004d3e02aeabf4c14b993c3faccdf479f6ab8d70f3265f4f5881e12ed1adf94 --recipe "$recipe" --store "$fixed_store" --manifest "$fixed_manifest" --output runs/static_decoder_probe_v1/eval_seed0/external_pdb/fixed --seeds 0

ln -s fixed runs/static_decoder_probe_v1/eval_seed0/baseline/fixed_history
ln -s fixed runs/static_decoder_probe_v1/eval_seed0/internal_md/fixed_history
ln -s fixed runs/static_decoder_probe_v1/eval_seed0/external_pdb/fixed_history
python -u -m tools.report_gm_a \
  --arm A0=/workspace/molvid-recon-static-decoder-probe-v1/runs/static_decoder_probe_v1/eval_seed0/baseline \
  --arm A2=/workspace/molvid-recon-static-decoder-probe-v1/runs/static_decoder_probe_v1/eval_seed0/internal_md \
  --arm A3=/workspace/molvid-recon-static-decoder-probe-v1/runs/static_decoder_probe_v1/eval_seed0/external_pdb \
  --output runs/static_decoder_probe_v1/comparison_seed0

# Held-out MD-frame static diagnostic, identical source/noise across arms.
static_store=/workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/clip_store/valid
static_manifest=/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views/paired_view_manifest.json
model_config=configs/frame_gm_p2_formal_GM_260923.yaml

python -u -m tools.evaluate_static_decoder --checkpoint runs/static_decoder_probe_v1/formal_500/baseline/frame_joint_step_00000500.pt --checkpoint-sha256 f4d0ccd956aeb6b2a6593b3256da55f942d8a2ccb5ca937f8c594b1928a77874 --recipe "$recipe" --static-config agent/static_decoder_probe_v1/configs/formal_internal_md_500.json --store "$static_store" --manifest "$static_manifest" --model-config "$model_config" --output runs/static_decoder_probe_v1/static_eval_formal/baseline
python -u -m tools.evaluate_static_decoder --checkpoint runs/static_decoder_probe_v1/formal_500/internal_md/frame_joint_step_00000500.pt --checkpoint-sha256 90e80ba76d968f155cc5b04bd1683b6fc31463ac2959fec4ca2b52889bf61d99 --recipe "$recipe" --static-config agent/static_decoder_probe_v1/configs/formal_internal_md_500.json --store "$static_store" --manifest "$static_manifest" --model-config "$model_config" --output runs/static_decoder_probe_v1/static_eval_formal/internal_md
python -u -m tools.evaluate_static_decoder --checkpoint runs/static_decoder_probe_v1/formal_500/external_pdb/frame_joint_step_00000500.pt --checkpoint-sha256 9004d3e02aeabf4c14b993c3faccdf479f6ab8d70f3265f4f5881e12ed1adf94 --recipe "$recipe" --static-config agent/static_decoder_probe_v1/configs/formal_internal_md_500.json --store "$static_store" --manifest "$static_manifest" --model-config "$model_config" --output runs/static_decoder_probe_v1/static_eval_formal/external_pdb

# Final focused regression checks.
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_static_decoder_evaluator.py tests/test_gm_a_evaluator.py
