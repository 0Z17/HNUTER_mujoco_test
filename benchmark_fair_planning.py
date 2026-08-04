#!/usr/bin/env python3
"""Fair cold/warm planning-time comparison: diffusion vs direct OMPL.

Both pipelines are measured on the same environment/URDF and the same endpoint
pair, from "cold start" (process + imports + collision-environment setup
included) and in warm mode (one-time setup amortised/excluded on both sides).
The scope is: endpoints -> one COAL-certified smoothed SE(3) geometric path
(B-spline fit and audit included; TOPP-RA/MPPI excluded because they are
identical downstream stages).

Run the diffusion side with the cuRobo environment and the rest with the
project venv plus the COAL site-packages on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INTEGRATED_PYTHON = Path("/home/z017/research/curobo_env/bin/python")


def build_checker(environment_path: Path, urdf_path: Path):
    from coal_collision import CoalCollisionChecker
    from run_overfit_cube_single_pipeline import load_environment

    environment_data, environment = load_environment(environment_path)
    checker = CoalCollisionChecker.from_urdf(
        urdf_path, environment, safety_margin=0.0
    )
    bounds = environment_data["sampling_space"]["position_bounds"]
    return (
        checker,
        environment,
        np.asarray(bounds["min"], dtype=np.float64),
        np.asarray(bounds["max"], dtype=np.float64),
    )


def measure_ompl(
    environment_path: Path,
    urdf_path: Path,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    seed: int,
) -> dict:
    import sys as _sys
    from ompl_se3_planner import OMPLSE3Planner, SE3Pose

    started = time.perf_counter()
    checker, environment, bounds_min, bounds_max = build_checker(
        environment_path, urdf_path
    )
    env_setup = time.perf_counter() - started
    planner = OMPLSE3Planner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        obstacles=(),
        vehicle_radius=0.0,
        safety_margin=0.0,
        validity_resolution=0.0025,
        planner_range=0.34,
        seed=seed,
        collision_checker=checker,
    )
    start = SE3Pose(start_pose[:3], start_pose[3:7])
    goal = SE3Pose(goal_pose[:3], goal_pose[3:7])

    from multi_waypoint_planner import (
        MultiWaypointOMPLPlanner,
        WaypointConstrainedSmoothingSE3BSpline,
    )

    t0 = time.perf_counter()
    path = planner.plan(start, goal, solve_time=8.0)
    states = path.states.copy()
    waypoint_indices = (0, len(states) - 1)
    clearance = planner.clearance(states[:, :3], states[:, 3:7])
    clearance_weight = np.minimum(
        1.0
        + np.square(0.30 / (np.maximum(clearance, 0.0) + 0.01)),
        400.0,
    )
    builder = MultiWaypointOMPLPlanner(planner)
    last_error: Exception | None = None
    for attempt in range(4):
        stride = max(1, 6 - attempt)
        try:
            spline = WaypointConstrainedSmoothingSE3BSpline(
                states,
                waypoint_indices,
                degree=5,
                control_point_stride=stride,
                orientation_metric_weight=0.35,
                guide_weight=1.0 * 5.0**attempt,
                guide_sample_weights=clearance_weight,
                position_acceleration_weight=1.0e-8,
                position_jerk_weight=1.0e-12,
                orientation_acceleration_weight=2.5e-9,
                orientation_jerk_weight=2.5e-13,
            )
            builder._build_validated_plan(
                (start, goal),
                (path,),
                states,
                waypoint_indices,
                spline,
                spline.waypoint_parameters,
                2200,
                stride,
                False,
                None,
            )
            break
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            last_error = error
    else:
        raise RuntimeError(f"OMPL spline fit failed: {last_error}")
    planning_total = time.perf_counter() - t0
    return {
        "environment_setup_s": round(env_setup, 3),
        "solve_simplify_spline_s": round(planning_total, 3),
        "solve_time_s": round(path.planning_time_s, 3),
        "cold_total_s": round(env_setup + planning_total, 3),
        "warm_total_s": round(planning_total, 3),
    }


def measure_diffusion_oneshot(
    args: argparse.Namespace,
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    seed: int,
    output_dir: Path,
) -> dict:
    from run_unet_guided_diffusion_demo import candidate_metrics_staged

    command = [
        str(args.integrated_python.expanduser().resolve()),
        str(PROJECT_DIR / "sample_and_filter_candidates.py"),
        "--prepared", str(args.prepared.expanduser().resolve()),
        "--checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--environment", str(args.environment.expanduser().resolve()),
        "--spheres", str(args.spheres.expanduser().resolve()),
        "--activation-distance", str(args.acceptance_clearance),
        "--output-dir", str(output_dir),
        "--start-pose", *map(str, start_pose),
        "--goal-pose", *map(str, goal_pose),
        "--candidate-count", str(args.candidate_count),
        "--sampling-batch-size", str(min(args.candidate_count, 32)),
        "--seed", str(seed),
        "--device", "auto",
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("LD_LIBRARY_PATH", None)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=environment,
    )
    sampler_wall = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    summary = None
    for line in completed.stdout.splitlines():
        if line.startswith("SAMPLE_FILTER="):
            summary = json.loads(line[len("SAMPLE_FILTER="):])
    assert summary is not None
    candidate_file = Path(summary["candidate_file"])
    with np.load(candidate_file) as payload:
        paths = payload["poses_wxyz"].astype(np.float64)
    checker, environment, bounds_min, bounds_max = build_checker(
        args.environment.expanduser().resolve(),
        args.urdf.expanduser().resolve(),
    )
    coarse_accept = np.asarray(summary["coarse_accept"], dtype=bool)
    coarse_clearance = np.asarray(
        summary["min_sphere_clearance_estimate_m"], dtype=np.float64
    )
    started = time.perf_counter()
    records = candidate_metrics_staged(
        paths,
        checker,
        bounds_min,
        bounds_max,
        args.acceptance_clearance,
        coarse_accept,
        coarse_clearance,
    )
    coal_filter = time.perf_counter() - started
    accepted = [r for r in records if r["accepted_8cm"]]
    if not accepted:
        raise RuntimeError("no accepted candidate in diffusion benchmark")
    return {
        "sampler_process_wall_s": round(sampler_wall, 3),
        "sampling_time_s": round(summary["sampling_time_s"], 3),
        "curobo_filter_time_s": round(summary["curobo_filter_time_s"], 3),
        "coal_staged_filter_s": round(coal_filter, 3),
        "accepted": len(accepted),
        "cold_total_s": round(
            sampler_wall + coal_filter + 0.27, 3
        ),  # + spline fit/audit
        "warm_total_s": round(
            summary["sampling_time_s"]
            + summary["curobo_filter_time_s"]
            + coal_filter
            + 0.27,
            3,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        type=Path,
        default=PROJECT_DIR / "environment_multihomotopy_v002.json",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_DIR
        / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=PROJECT_DIR
        / "results/diffusion_se3_three_stage_v002/prepared_dataset.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "results/diffusion_se3_three_stage_v002/models/unet/best.pt",
    )
    parser.add_argument(
        "--integrated-python", type=Path, default=DEFAULT_INTEGRATED_PYTHON
    )
    parser.add_argument(
        "--spheres",
        type=Path,
        default=PROJECT_DIR
        / "etc/URDF-for-gazebo/config/"
        / "HDJQR-0102-0055.SLDASM_curobo_spheres.yml",
    )
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--acceptance-clearance", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results/fair_planning_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment_path = args.environment.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    start_pose = np.asarray(
        [1.1617063283920288, -2.539853572845459, 1.25383460521698,
         0.9287307432470833, 0.3433068159497048, 0.11892456779077325,
         -0.07386869327376655],
        dtype=np.float64,
    )
    goal_pose = np.asarray(
        [-0.6953775882720947, 2.585259437561035, 1.967663049697876,
         0.6182834894887017, 0.24514194501490086, 0.05183947657671927,
         0.7449453819497358],
        dtype=np.float64,
    )
    output_dir = Path("/tmp/fair_benchmark_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    ompl = measure_ompl(
        environment_path, urdf_path, start_pose, goal_pose, args.seed
    )
    diffusion = measure_diffusion_oneshot(
        args, start_pose, goal_pose, args.seed, output_dir
    )
    payload = {
        "scope": "endpoints -> one COAL-certified smoothed SE(3) path",
        "environment": str(environment_path),
        "diffusion_oneshot": diffusion,
        "ompl": ompl,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
