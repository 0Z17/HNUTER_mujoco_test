#!/usr/bin/env bash
# Zero-config Rerun replay of the B-spline control-point denoising process.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
curobo_python="${CUROBO_PYTHON:-/home/z017/research/curobo_env/bin/python}"
open_viewer=1
python_args=()

for arg in "$@"; do
  if [[ "${arg}" == "--no-open" ]]; then
    open_viewer=0
  else
    python_args+=("${arg}")
  fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
output="${BSPLINE_REPLAY_OUTPUT:-${project_dir}/results/bspline_steps_replay_${stamp}.rrd}"

if [[ ! -x "${curobo_python}" ]]; then
  echo "cuRobo Python not found: ${curobo_python}" >&2
  echo "Set CUROBO_PYTHON to the cuRobo environment's python." >&2
  exit 2
fi

"${curobo_python}" "${project_dir}/rerun_bspline_steps.py" \
  --output "${output}" "${python_args[@]}"

if (( open_viewer )); then
  rerun_bin="${project_dir}/.venv/bin/rerun"
  if [[ -x "${rerun_bin}" ]]; then
    exec "${rerun_bin}" --new "${output}"
  else
    echo "rerun viewer not found: ${rerun_bin}" >&2
    echo "Open manually: rerun --port auto ${output}" >&2
  fi
fi
