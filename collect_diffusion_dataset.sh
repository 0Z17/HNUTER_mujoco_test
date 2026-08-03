#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_python="${project_dir}/.venv/bin/python"

if [[ ! -x "${runtime_python}" ]]; then
  echo "Missing project Python: ${runtime_python}" >&2
  exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-hnuter}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "${runtime_python}" "${project_dir}/collect_diffusion_dataset.py" "$@"
