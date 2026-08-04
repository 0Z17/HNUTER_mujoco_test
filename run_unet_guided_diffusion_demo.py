#!/usr/bin/env python3
"""Random U-Net-guidance planning followed by COAL/TOPP-RA/MPPI/Rerun."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
from typing import Any

import numpy as np

from coal_collision import CoalCollisionChecker
from evaluate_se3_diffusion import dense_path, path_metrics
from ompl_se3_planner import OMPLSE3Planner
from run_overfit_cube_single_pipeline import (
    load_environment,
    quaternion_distance,
    sample_valid_pose,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENVIRONMENT = PROJECT_DIR / "environment_multihomotopy_v002.json"
DEFAULT_URDF = (
    PROJECT_DIR
    / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf"
)
DEFAULT_MODEL = PROJECT_DIR / "hnuter206_4_5kg.xml"
DEFAULT_EXPERIMENT = PROJECT_DIR / "results/diffusion_se3_three_stage_v002"
DEFAULT_TORCH_PYTHON = Path(
    "/home/z017/research/diffusion_model/.envs/mpd-splines/bin/python"
)
SAMPLER = PROJECT_DIR / "sample_unet_guided_paths.py"
PIPELINE = PROJECT_DIR / "run_overfit_cube_single_pipeline.sh"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, ensure_ascii=False, default=json_default
        ) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--maximum-pair-attempts", type=int, default=8)
    parser.add_argument("--maximum-execution-candidates", type=int, default=3)
    parser.add_argument("--acceptance-clearance", type=float, default=0.08)
    parser.add_argument("--endpoint-clearance", type=float, default=0.16)
    parser.add_argument("--max-tilt-deg", type=float, default=42.0)
    parser.add_argument("--minimum-pair-distance", type=float, default=3.0)
    parser.add_argument("--minimum-y-separation", type=float, default=2.35)
    parser.add_argument("--minimum-attitude-separation-deg", type=float, default=15.0)
    parser.add_argument("--mppi-samples", type=int, default=512)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prepared", type=Path,
        default=DEFAULT_EXPERIMENT / "prepared_dataset.npz",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=DEFAULT_EXPERIMENT / "models/unet/best.pt",
    )
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH_PYTHON)
    parser.add_argument(
        "pipeline_args", nargs=argparse.REMAINDER,
        help="extra execution-pipeline arguments after --",
    )
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be non-negative")
    if (
        args.candidate_count <= 0
        or args.maximum_pair_attempts <= 0
        or args.maximum_execution_candidates <= 0
        or args.acceptance_clearance < 0.0
        or args.endpoint_clearance < args.acceptance_clearance
        or args.mppi_samples < 2
    ):
        parser.error("invalid candidate, clearance, attempt, or MPPI setting")
    return args


def candidate_metrics(
    paths: np.ndarray,
    checker: CoalCollisionChecker,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    required_clearance: float,
) -> list[dict[str, Any]]:
    records = []
    for index, path in enumerate(paths):
        workspace_valid = bool(
            np.all(np.isfinite(path))
            and np.all(path[:, :3] >= bounds_min)
            and np.all(path[:, :3] <= bounds_max)
        )
        error = None
        if workspace_valid:
            try:
                dense = dense_path(
                    path, translation_step=0.04, rotation_step_deg=3.0
                )
                clearance = checker.clearance(dense[:, :3], dense[:, 3:7])
                minimum_clearance = float(np.min(clearance))
            except (RuntimeError, ValueError) as caught:
                workspace_valid = False
                minimum_clearance = -math.inf
                error = str(caught)
        else:
            minimum_clearance = -math.inf
        metrics = path_metrics(path)
        records.append({
            "candidate_index": index,
            "workspace_valid": workspace_valid,
            "minimum_physical_clearance_m": minimum_clearance,
            "accepted_8cm": bool(
                workspace_valid and minimum_clearance >= required_clearance
            ),
            "error": error,
            **metrics,
        })
    return records


def run_streaming(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def main() -> None:
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (PROJECT_DIR / "results" / f"unet_guidance_demo_{stamp}_{seed}")
    )
    if root.exists():
        raise FileExistsError(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    environment_path = args.environment.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    environment_data, environment = load_environment(environment_path)
    checker = CoalCollisionChecker.from_urdf(
        urdf_path, environment, safety_margin=0.0
    )
    sampling_bounds = environment_data["sampling_space"]["position_bounds"]
    bounds_min = np.asarray(sampling_bounds["min"], dtype=np.float64)
    bounds_max = np.asarray(sampling_bounds["max"], dtype=np.float64)
    task = environment_data.get("task_sampling")
    if not task:
        raise RuntimeError("environment has no task_sampling start/goal regions")
    south_min = np.maximum(bounds_min, np.asarray(task["south_region"]["min"]))
    south_max = np.minimum(bounds_max, np.asarray(task["south_region"]["max"]))
    north_min = np.maximum(bounds_min, np.asarray(task["north_region"]["min"]))
    north_max = np.minimum(bounds_max, np.asarray(task["north_region"]["max"]))
    planner = OMPLSE3Planner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        obstacles=(),
        vehicle_radius=0.0,
        safety_margin=0.0,
        seed=seed,
        collision_checker=checker,
    )
    rng = np.random.default_rng(seed)
    pair_attempts: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] | None = None
    generated_paths: np.ndarray | None = None
    selected_start = selected_goal = None
    candidate_file = root / "diffusion_candidates.npz"
    for pair_attempt in range(1, args.maximum_pair_attempts + 1):
        start, start_tries, start_clearance = sample_valid_pose(
            rng, planner, south_min, south_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=3000,
        )
        goal, goal_tries, goal_clearance = sample_valid_pose(
            rng, planner, north_min, north_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=3000,
        )
        separation = float(np.linalg.norm(goal.position - start.position))
        y_separation = float(abs(goal.position[1] - start.position[1]))
        attitude_deg = math.degrees(
            quaternion_distance(start.quaternion, goal.quaternion)
        )
        record: dict[str, Any] = {
            "pair_attempt": pair_attempt,
            "start_pose": np.concatenate((start.position, start.quaternion)),
            "goal_pose": np.concatenate((goal.position, goal.quaternion)),
            "start_sampling_attempts": start_tries,
            "goal_sampling_attempts": goal_tries,
            "start_clearance_m": start_clearance,
            "goal_clearance_m": goal_clearance,
            "position_separation_m": separation,
            "y_separation_m": y_separation,
            "attitude_separation_deg": attitude_deg,
        }
        pair_attempts.append(record)
        if (
            separation < args.minimum_pair_distance
            or y_separation < args.minimum_y_separation
            or attitude_deg < args.minimum_attitude_separation_deg
        ):
            record["result"] = "rejected_separation"
            print(f"pair {pair_attempt}: rejected by separation", flush=True)
            continue
        sampler_seed = int(
            np.random.SeedSequence([seed, pair_attempt, 9000])
            .generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF
        )
        command = [
            str(args.torch_python.expanduser().resolve()),
            str(SAMPLER),
            "--prepared", str(args.prepared.expanduser().resolve()),
            "--checkpoint", str(args.checkpoint.expanduser().resolve()),
            "--output", str(candidate_file),
            "--start-pose", *map(str, np.concatenate((start.position, start.quaternion))),
            "--goal-pose", *map(str, np.concatenate((goal.position, goal.quaternion))),
            "--candidate-count", str(args.candidate_count),
            "--sampling-batch-size", str(min(args.candidate_count, 32)),
            "--seed", str(sampler_seed),
            "--device", "auto",
        ]
        print(
            f"pair {pair_attempt}: sampling {args.candidate_count} guided paths",
            flush=True,
        )
        sampler_environment = os.environ.copy()
        # The outer runtime may inject Python-3.12 COAL packages. The model
        # sampler is Python 3.10 and must resolve NumPy/Torch exclusively from
        # its own environment.
        sampler_environment.pop("PYTHONPATH", None)
        sampler_environment.pop("LD_LIBRARY_PATH", None)
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            env=sampler_environment,
        )
        record["sampler_seed"] = sampler_seed
        record["sampler_stdout"] = completed.stdout.strip()
        if completed.returncode != 0:
            record["result"] = "sampling_failed"
            record["sampler_error"] = completed.stderr.strip()
            write_json(root / "pair_attempts.json", pair_attempts)
            raise RuntimeError(
                "PyTorch sampler failed before candidate validation:\n"
                + completed.stderr.strip()
            )
        with np.load(candidate_file) as payload:
            generated_paths = payload["poses_wxyz"].astype(np.float64)
            sampling_time_s = float(payload["sampling_time_s"])
        records = candidate_metrics(
            generated_paths, checker, bounds_min, bounds_max,
            args.acceptance_clearance,
        )
        accepted = [item for item in records if item["accepted_8cm"]]
        record["candidate_count"] = len(records)
        record["accepted_candidate_count"] = len(accepted)
        record["sampling_time_s"] = sampling_time_s
        record["result"] = "accepted" if accepted else "no_safe_candidate"
        print(
            f"pair {pair_attempt}: {len(accepted)}/{len(records)} candidates "
            f"meet {100 * args.acceptance_clearance:.0f} cm COAL clearance",
            flush=True,
        )
        if accepted:
            accepted_records = records
            selected_start, selected_goal = start, goal
            break
    write_json(root / "pair_attempts.json", pair_attempts)
    if accepted_records is None or generated_paths is None:
        raise RuntimeError(
            "no sampled endpoint pair produced a COAL-certified diffusion path; "
            f"see {root / 'pair_attempts.json'}"
        )
    ranked = sorted(
        (item for item in accepted_records if item["accepted_8cm"]),
        key=lambda item: (
            -item["minimum_physical_clearance_m"],
            item["translation_length_m"],
            item["position_acceleration_rms"],
        ),
    )
    write_json(root / "candidate_metrics.json", {
        "required_clearance_m": args.acceptance_clearance,
        "candidate_count": len(accepted_records),
        "accepted_count": len(ranked),
        "records": accepted_records,
    })
    successful_summary: Path | None = None
    selected_record: dict[str, Any] | None = None
    for execution_rank, record in enumerate(
        ranked[: args.maximum_execution_candidates]
    ):
        candidate_index = int(record["candidate_index"])
        execution_dir = root / f"execution_{execution_rank:02d}"
        selected_path = root / f"selected_path_{execution_rank:02d}.npz"
        with np.load(candidate_file) as payload:
            sampling_time_s = float(payload["sampling_time_s"])
        np.savez_compressed(
            selected_path,
            poses_wxyz=generated_paths[candidate_index],
            candidate_index=np.asarray(candidate_index),
            sampling_time_s=np.asarray(sampling_time_s),
            source=np.asarray("U-Net diffusion + inference guidance, COAL selected"),
        )
        command = [
            str(PIPELINE),
            "--external-path", str(selected_path),
            "--diffusion-candidates", str(candidate_file),
            "--diffusion-candidate-metrics", str(
                root / "candidate_metrics.json"
            ),
            "--output-dir", str(execution_dir),
            "--environment", str(environment_path),
            "--urdf", str(urdf_path),
            "--model", str(args.model.expanduser().resolve()),
            "--seed", str(seed),
            "--coal-safety-margin", str(args.acceptance_clearance),
            "--path-clearance", "0.0",
            "--endpoint-clearance", "0.0",
            "--mppi-samples", str(args.mppi_samples),
            *(["--no-gif"] if args.no_gif else []),
            *args.pipeline_args,
        ]
        print(
            f"executing candidate {candidate_index} "
            f"(clearance={record['minimum_physical_clearance_m']:.3f} m)",
            flush=True,
        )
        return_code = run_streaming(
            command, root / f"execution_{execution_rank:02d}.log"
        )
        summary_path = execution_dir / "single_pipeline_summary.json"
        if return_code == 0 and summary_path.exists():
            successful_summary = summary_path
            selected_record = record
            break
        print(
            f"candidate {candidate_index} execution failed; trying next safe candidate",
            flush=True,
        )
    if successful_summary is None or selected_record is None:
        raise RuntimeError(
            "safe candidates were generated but none passed B-spline/TOPP-RA/"
            f"MPPI execution; see logs under {root}"
        )
    execution = json.loads(successful_summary.read_text(encoding="utf-8"))
    overall = {
        "pipeline_success": True,
        "seed": seed,
        "output_directory": str(root),
        "start_pose": np.concatenate((
            selected_start.position, selected_start.quaternion
        )),
        "goal_pose": np.concatenate((
            selected_goal.position, selected_goal.quaternion
        )),
        "selected_candidate": selected_record,
        "execution_summary": str(successful_summary),
        "rerun_recording": execution["outputs"]["rerun_recording"],
        "tracking_gif": execution["outputs"]["mujoco_tracking_gif"],
        "tracking_plot": execution["outputs"]["tracking_plot"],
        "view_command": (
            f"rerun --port auto {execution['outputs']['rerun_recording']}"
        ),
    }
    write_json(root / "unet_guidance_demo_summary.json", overall)
    print("UNET_GUIDANCE_DEMO=" + json.dumps(overall, default=json_default))
    print("VIEW=" + overall["view_command"])


if __name__ == "__main__":
    main()
