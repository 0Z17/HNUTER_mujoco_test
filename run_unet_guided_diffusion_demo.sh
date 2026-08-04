#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_python="${project_dir}/.venv/bin/python"
coal_env="${HNUTER_COAL_ENV:-/tmp/hnuter-coal-verify}"

if [[ ! -x "${runtime_python}" ]]; then
  echo "Missing project Python: ${runtime_python}" >&2
  exit 2
fi

if ! "${runtime_python}" -c 'import coal' >/dev/null 2>&1; then
  coal_site="${coal_env}/lib/python3.12/site-packages"
  if [[ ! -d "${coal_site}/coal" ]]; then
    echo "COAL is unavailable. Set HNUTER_COAL_ENV to its Python 3.12 environment." >&2
    exit 2
  fi
  export PYTHONPATH="${coal_site}${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_LIBRARY_PATH="${coal_env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-hnuter}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "${runtime_python}" "${project_dir}/run_unet_guided_diffusion_demo.py" "$@"
