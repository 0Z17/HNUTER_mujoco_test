#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pipeline="${project_dir}/run_overfit_cube_single_pipeline.sh"
rerun_bin="${project_dir}/.venv/bin/rerun"

seed=""
output_dir=""
open_rerun=1
render_gif=0
outer_attempts=3
pair_attempts=30
extra_args=()

usage() {
  cat <<'EOF'
Run one random collision-free SE(3) planning and MuJoCo tracking trial.

Usage:
  ./run_random_se3_trial.sh [wrapper options] [-- pipeline options]

Wrapper options:
  --seed N                 Reproduce a specific random trial.
  --output-dir DIR         Write results to DIR instead of a timestamped run.
  --outer-attempts N       Try up to N fresh random seeds (default: 3).
  --pair-attempts N        Start/goal pair attempts per seed (default: 30).
  --no-open-rerun          Save the RRD without starting the Rerun Viewer.
  --render-gif             Also render the slower MuJoCo replay GIF.
  -h, --help               Show this help.

Any arguments after -- are forwarded to run_overfit_cube_single_pipeline.py.

Examples:
  ./run_random_se3_trial.sh
  ./run_random_se3_trial.sh --seed 20270802
  ./run_random_se3_trial.sh --no-open-rerun -- --max-tilt-deg 30
EOF
}

random_seed() {
  printf '%u\n' "$(( ((RANDOM << 16) ^ (RANDOM << 1) ^ $$) & 0x7fffffff ))"
}

while (($#)); do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
      seed="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    --outer-attempts)
      [[ $# -ge 2 ]] || { echo "--outer-attempts requires a value" >&2; exit 2; }
      outer_attempts="$2"
      shift 2
      ;;
    --pair-attempts)
      [[ $# -ge 2 ]] || { echo "--pair-attempts requires a value" >&2; exit 2; }
      pair_attempts="$2"
      shift 2
      ;;
    --no-open-rerun)
      open_rerun=0
      shift
      ;;
    --render-gif)
      render_gif=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown wrapper option: $1" >&2
      echo "Use -- before options intended for the planning pipeline." >&2
      exit 2
      ;;
  esac
done

[[ -x "${pipeline}" ]] || {
  echo "Missing pipeline entry point: ${pipeline}" >&2
  exit 2
}
[[ "${outer_attempts}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--outer-attempts must be a positive integer" >&2
  exit 2
}
[[ "${pair_attempts}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--pair-attempts must be a positive integer" >&2
  exit 2
}
if [[ -n "${seed}" && ! "${seed}" =~ ^[0-9]+$ ]]; then
  echo "--seed must be a non-negative integer" >&2
  exit 2
fi

if [[ -z "${output_dir}" ]]; then
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="${project_dir}/results/random_se3_runs/run_${run_stamp}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${project_dir}/${output_dir}"
fi

mkdir -p "${output_dir}"
success=0
successful_seed=""
for ((attempt = 1; attempt <= outer_attempts; ++attempt)); do
  if [[ -n "${seed}" ]]; then
    trial_seed="${seed}"
  else
    trial_seed="$(random_seed)"
  fi
  echo "RANDOM_SE3_TRIAL attempt=${attempt}/${outer_attempts} seed=${trial_seed}"
  command=(
    "${pipeline}"
    --seed "${trial_seed}"
    --output-dir "${output_dir}"
    --maximum-pair-attempts "${pair_attempts}"
  )
  if ((render_gif == 0)); then
    command+=(--no-gif)
  fi
  command+=("${extra_args[@]}")
  if "${command[@]}"; then
    success=1
    successful_seed="${trial_seed}"
    break
  fi
  if [[ -n "${seed}" ]]; then
    break
  fi
  echo "Trial seed ${trial_seed} was rejected; sampling a fresh seed." >&2
done

if ((success == 0)); then
  echo "No complete collision-free trial succeeded after ${outer_attempts} attempt(s)." >&2
  exit 1
fi

rrd_path="${output_dir}/mujoco_mppi_tracking.rrd"
summary_path="${output_dir}/single_pipeline_summary.json"
echo "RANDOM_SE3_SUCCESS seed=${successful_seed}"
echo "SUMMARY=${summary_path}"
echo "RERUN_RECORDING=${rrd_path}"

if ((open_rerun == 1)); then
  [[ -x "${rerun_bin}" ]] || {
    echo "Rerun executable not found: ${rerun_bin}" >&2
    exit 2
  }
  exec "${rerun_bin}" --new "${rrd_path}"
fi
