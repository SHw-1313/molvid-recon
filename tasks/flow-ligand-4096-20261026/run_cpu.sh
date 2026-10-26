#!/usr/bin/env bash
set -euo pipefail
[[ ${CONDA_DEFAULT_ENV:-} == torch-ito ]] || { echo 'Use existing torch-ito container'; exit 2; }
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python -u "$task_dir/finish_evaluation.py" "$@"
