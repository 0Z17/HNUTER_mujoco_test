#!/usr/bin/env bash
set -euo pipefail

converter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_sim_dir="${ISAAC_SIM_PATH:-/home/z017/isaacsim}"
isaac_python="${isaac_sim_dir}/python.sh"

if [[ ! -x "${isaac_python}" ]]; then
    echo "Isaac Sim python.sh not found: ${isaac_python}" >&2
    echo "Set ISAAC_SIM_PATH to your Isaac Sim installation directory." >&2
    exit 1
fi

exec "${isaac_python}" "${converter_dir}/convert_urdf_to_usd.py" "$@"
