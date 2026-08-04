#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_python="${MPD_PYTHON:-/home/z017/research/diffusion_model/.envs/mpd-splines/bin/python}"
dataset="${DIFFUSION_DATASET:-${project_dir}/datasets/diffusion_se3_multihomotopy_v002_300}"
output="${DIFFUSION_OUTPUT:-${project_dir}/results/diffusion_se3_three_stage_v002}"
sequence_length="${DIFFUSION_SEQUENCE_LENGTH:-128}"

if [[ ! -x "${training_python}" ]]; then
  echo "Missing PyTorch Python: ${training_python}" >&2
  exit 2
fi

for argument in "$@"; do
  if [[ "${argument}" == "--sequence-length" || "${argument}" == --sequence-length=* ]]; then
    echo "Set DIFFUSION_SEQUENCE_LENGTH instead of passing --sequence-length to keep the COAL audit consistent." >&2
    exit 2
  fi
done

mkdir -p "${output}"
"${project_dir}/run_diffusion_representation_audit.sh" \
  --dataset "${dataset}" \
  --sequence-length "${sequence_length}" \
  --output "${output}/representation_audit.json"

"${training_python}" "${project_dir}/train_se3_diffusion.py" \
  --dataset "${dataset}" \
  --output-dir "${output}" \
  --sequence-length "${sequence_length}" \
  "$@"

"${project_dir}/run_se3_diffusion_evaluation.sh" \
  --dataset "${dataset}" \
  --experiment "${output}"
