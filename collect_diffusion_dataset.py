#!/usr/bin/env python3
"""Collect diverse, collision-free SE(3) planning/tracking demonstrations.

Each accepted trajectory is produced by the same full subprocess used for the
interactive demo: OMPL RRTConnect + COAL, constrained B-spline smoothing,
TOPP-RA retiming, and MPPI tracking in MuJoCo.  Visualization artifacts are
disabled.  Both lossless variable-length logs and fixed-length tensors are
written so collection can be audited before diffusion training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE = PROJECT_DIR / "run_overfit_cube_single_pipeline.sh"
DEFAULT_ENVIRONMENT = PROJECT_DIR / "environment_multihomotopy_v002.json"
DEFAULT_URDF = (
    PROJECT_DIR
    / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf"
)
DEFAULT_MODEL = PROJECT_DIR / "hnuter206_4_5kg.xml"
SCHEMA_VERSION = 1
STATE_LAYOUT = (
    "x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz",
    "omega_x_body", "omega_y_body", "omega_z_body",
)
ACTION_LAYOUT = (
    "ax_world", "ay_world", "az_world", "alpha_x_body",
    "alpha_y_body", "alpha_z_body",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_seed(dataset_seed: int, *indices: int) -> int:
    sequence = np.random.SeedSequence([dataset_seed, *indices])
    return int(sequence.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def _make_quaternions_continuous(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(norms, 1.0e-12)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _interpolate_rows(
    source_time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
    quaternion_slice: slice | None = None,
) -> np.ndarray:
    source_time = np.asarray(source_time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    if source_time.ndim != 1 or values.ndim != 2:
        raise ValueError("source_time must be 1-D and values must be 2-D")
    if len(source_time) != len(values) or len(source_time) < 2:
        raise ValueError("source_time and values must have matching length >= 2")
    source_values = values.copy()
    if quaternion_slice is not None:
        source_values[:, quaternion_slice] = _make_quaternions_continuous(
            source_values[:, quaternion_slice]
        )
    result = np.column_stack(
        [np.interp(target_time, source_time, source_values[:, column])
         for column in range(source_values.shape[1])]
    )
    if quaternion_slice is not None:
        result[:, quaternion_slice] = _make_quaternions_continuous(
            result[:, quaternion_slice]
        )
    return result


def _resample_progress(
    values: np.ndarray,
    count: int,
    quaternion_slice: slice | None = None,
) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, count)
    return _interpolate_rows(source, values, target, quaternion_slice)


def _quaternion_rms_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_q = _make_quaternions_continuous(first)
    second_q = _make_quaternions_continuous(second)
    dot = np.clip(np.abs(np.sum(first_q * second_q, axis=1)), 0.0, 1.0)
    return float(np.degrees(np.sqrt(np.mean(np.square(2.0 * np.arccos(dot))))))


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - fraction) * first + fraction * second
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    result = (
        math.sin((1.0 - fraction) * angle) * first
        + math.sin(fraction * angle) * second
    ) / math.sin(angle)
    return result / np.linalg.norm(result)


def _sample_via_pose(
    start_pose: list[float],
    goal_pose: list[float],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    seed: int,
    anchor_path: np.ndarray | None = None,
    perturb: bool = True,
) -> list[float]:
    """Perturb a known-free route laterally without introducing upward bias."""

    rng = np.random.default_rng(seed ^ 0x76696170)
    start = np.asarray(start_pose, dtype=np.float64)
    goal = np.asarray(goal_pose, dtype=np.float64)
    fraction = float(rng.uniform(0.34, 0.66))
    if anchor_path is None:
        position = (1.0 - fraction) * start[:3] + fraction * goal[:3]
        quaternion = _slerp(start[3:7], goal[3:7], fraction)
    else:
        anchor = np.asarray(anchor_path, dtype=np.float64)
        if anchor.ndim != 2 or anchor.shape[1] != 7 or len(anchor) < 2:
            raise ValueError("anchor_path must have shape (N, 7), N >= 2")
        anchor_index = int(round(fraction * (len(anchor) - 1)))
        position = anchor[anchor_index, :3].copy()
        quaternion = anchor[anchor_index, 3:7].copy()
    planar_direction = goal[:2] - start[:2]
    planar_norm = float(np.linalg.norm(planar_direction))
    if planar_norm < 1.0e-9:
        lateral = np.asarray([1.0, 0.0])
    else:
        lateral = np.asarray(
            [-planar_direction[1], planar_direction[0]]
        ) / planar_norm
    if perturb:
        position[:2] += (
            float(rng.choice((-1.0, 1.0)))
            * float(rng.uniform(0.18, 0.58))
            * lateral
        )
        position[2] += float(rng.uniform(-0.18, 0.28))
    position = np.clip(position, bounds_min + 0.12, bounds_max - 0.12)
    return np.concatenate((position, quaternion)).tolist()


def _quaternion_from_rpy(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy_rad)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    quaternion = np.asarray((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))
    return quaternion / np.linalg.norm(quaternion)


def _sample_route_template(
    template: dict[str, Any], seed: int
) -> list[list[float]]:
    """Sample hard SE(3) waypoints inside an environment route template."""

    rng = np.random.default_rng(seed ^ 0x746F706F)
    poses: list[list[float]] = []
    for waypoint in template["waypoints"]:
        position_bounds = waypoint["position_bounds"]
        position = rng.uniform(
            np.asarray(position_bounds["min"], dtype=np.float64),
            np.asarray(position_bounds["max"], dtype=np.float64),
        )
        rpy_bounds = waypoint["rpy_deg_bounds"]
        rpy_deg = rng.uniform(
            np.asarray(rpy_bounds["min"], dtype=np.float64),
            np.asarray(rpy_bounds["max"], dtype=np.float64),
        )
        quaternion = _quaternion_from_rpy(np.radians(rpy_deg))
        poses.append(np.concatenate((position, quaternion)).tolist())
    return poses


def _crossing_point(
    positions: np.ndarray, axis: int, value: float
) -> np.ndarray | None:
    signed = positions[:, axis] - value
    for index in range(len(positions) - 1):
        first = float(signed[index])
        second = float(signed[index + 1])
        if first == 0.0:
            return positions[index].copy()
        if first * second <= 0.0 and first != second:
            fraction = -first / (second - first)
            return (
                (1.0 - fraction) * positions[index]
                + fraction * positions[index + 1]
            )
    return None


def _matches_topology_class(
    point: np.ndarray, class_spec: dict[str, Any]
) -> bool:
    axis_indices = {"x": 0, "y": 1, "z": 2}
    for axis_name, axis in axis_indices.items():
        range_key = f"{axis_name}_range"
        minimum_key = f"{axis_name}_min"
        maximum_key = f"{axis_name}_max"
        if range_key in class_spec:
            lower, upper = class_spec[range_key]
            if not float(lower) <= point[axis] <= float(upper):
                return False
        if minimum_key in class_spec and point[axis] < float(class_spec[minimum_key]):
            return False
        if maximum_key in class_spec and point[axis] > float(class_spec[maximum_key]):
            return False
    return True


def _topology_signature(
    states: np.ndarray, cuts: list[dict[str, Any]]
) -> dict[str, str]:
    """Classify which named corridor a path uses at each separator cut."""

    state_array = np.asarray(states, dtype=np.float64)
    positions = state_array[:, :3]
    axis_indices = {"x": 0, "y": 1, "z": 2}
    signature: dict[str, str] = {}
    for cut in cuts:
        crossing = _crossing_point(
            positions, axis_indices[cut["axis"]], float(cut["value"])
        )
        if crossing is None:
            signature[cut["id"]] = "not_crossed"
            continue
        label = "unclassified"
        for class_spec in cut["classes"]:
            if _matches_topology_class(crossing, class_spec):
                label = str(class_spec["label"])
                break
        signature[cut["id"]] = label
    return signature


def _signature_key(signature: dict[str, str]) -> str:
    return "|".join(f"{key}:{signature[key]}" for key in sorted(signature))


def _descriptor(npz_path: Path, count: int = 192) -> dict[str, np.ndarray]:
    with np.load(npz_path) as sample:
        planned = _resample_progress(
            sample["smoothed_path_states"], count, slice(3, 7)
        )
        actual_time = np.asarray(sample["control_time"], dtype=np.float64)
        actual_target = np.linspace(actual_time[0], actual_time[-1], count)
        actual = _interpolate_rows(
            actual_time, sample["actual_state"], actual_target, slice(6, 10)
        )
    return {"planned": planned, "actual": actual}


def _diversity(
    candidate: dict[str, np.ndarray],
    accepted: Iterable[dict[str, np.ndarray]],
    minimum_position_m: float,
    minimum_orientation_deg: float,
) -> dict[str, float | bool | None]:
    comparisons: list[dict[str, float]] = []
    for existing in accepted:
        planned_position = float(np.sqrt(np.mean(np.sum(np.square(
            candidate["planned"][:, :3] - existing["planned"][:, :3]
        ), axis=1))))
        planned_orientation = _quaternion_rms_degrees(
            candidate["planned"][:, 3:7], existing["planned"][:, 3:7]
        )
        actual_position = float(np.sqrt(np.mean(np.sum(np.square(
            candidate["actual"][:, :3] - existing["actual"][:, :3]
        ), axis=1))))
        actual_orientation = _quaternion_rms_degrees(
            candidate["actual"][:, 6:10], existing["actual"][:, 6:10]
        )
        score = max(
            planned_position / max(minimum_position_m, 1.0e-12),
            planned_orientation / max(minimum_orientation_deg, 1.0e-12),
        )
        comparisons.append({
            "planned_position_rms_m": planned_position,
            "planned_orientation_rms_deg": planned_orientation,
            "actual_position_rms_m": actual_position,
            "actual_orientation_rms_deg": actual_orientation,
            "threshold_score": score,
        })
    if not comparisons:
        return {
            "accepted_by_threshold": True,
            "nearest_threshold_score": None,
            "planned_position_rms_m": None,
            "planned_orientation_rms_deg": None,
            "actual_position_rms_m": None,
            "actual_orientation_rms_deg": None,
        }
    nearest = min(comparisons, key=lambda item: item["threshold_score"])
    return {
        "accepted_by_threshold": bool(nearest["threshold_score"] >= 1.0),
        "nearest_threshold_score": nearest["threshold_score"],
        "planned_position_rms_m": nearest["planned_position_rms_m"],
        "planned_orientation_rms_deg": nearest["planned_orientation_rms_deg"],
        "actual_position_rms_m": nearest["actual_position_rms_m"],
        "actual_orientation_rms_deg": nearest["actual_orientation_rms_deg"],
    }


def _normalize_sample(
    source_path: Path,
    destination_path: Path,
    pair_index: int,
    trajectory_index: int,
    planner_seed: int,
    steps: int,
) -> dict[str, np.ndarray]:
    with np.load(source_path) as source:
        start_pose = source["start_pose"].copy()
        goal_pose = source["goal_pose"].copy()

        reference_time = source["toppra_time"]
        target_reference_time = np.linspace(
            reference_time[0], reference_time[-1], steps
        )
        reference_state = _interpolate_rows(
            reference_time,
            source["toppra_reference_state"],
            target_reference_time,
            slice(6, 10),
        )
        reference_action_source = np.concatenate(
            (source["toppra_linear_acceleration_world"],
             source["toppra_angular_acceleration_body"]),
            axis=1,
        )
        reference_action = _interpolate_rows(
            reference_time,
            reference_action_source,
            target_reference_time,
        )
        reference_clearance = np.interp(
            target_reference_time,
            reference_time,
            source["toppra_physical_clearance"],
        )

        actual_time = source["control_time"]
        target_actual_time = np.linspace(actual_time[0], actual_time[-1], steps)
        actual_state = _interpolate_rows(
            actual_time, source["actual_state"], target_actual_time, slice(6, 10)
        )
        tracking_reference_state = _interpolate_rows(
            actual_time,
            source["reference_state"],
            target_actual_time,
            slice(6, 10),
        )
        actual_action = _interpolate_rows(
            actual_time, source["mppi_action"], target_actual_time
        )
        actual_clearance = np.interp(
            target_actual_time,
            actual_time,
            source["actual_physical_clearance"],
        )

    payload = {
        "condition_start_goal": np.concatenate((start_pose, goal_pose)),
        "normalized_progress": np.linspace(0.0, 1.0, steps),
        "reference_time": target_reference_time,
        "reference_state": reference_state,
        "reference_action": reference_action,
        "reference_physical_clearance": reference_clearance,
        "actual_time": target_actual_time,
        "actual_state": actual_state,
        "tracking_reference_state": tracking_reference_state,
        "actual_action": actual_action,
        "actual_physical_clearance": actual_clearance,
        "pair_index": np.asarray(pair_index, dtype=np.int64),
        "trajectory_index": np.asarray(trajectory_index, dtype=np.int64),
        "planner_seed": np.asarray(planner_seed, dtype=np.int64),
    }
    np.savez_compressed(destination_path, **payload)
    return payload


def _pose_from_summary(summary: dict[str, Any], name: str) -> list[float]:
    pose = summary[name]
    return [*pose["position_m"], *pose["quaternion_wxyz"]]


def _quality_check(
    summary: dict[str, Any],
    max_position_error: float,
    max_attitude_error_deg: float,
) -> tuple[bool, str]:
    if not summary.get("pipeline_success", False):
        return False, "pipeline_success is false"
    audit = summary.get("coal_collision_audit", {})
    if not audit.get("physical_collision_free_planned_path", False):
        return False, "planned path failed physical collision audit"
    if not audit.get("physical_collision_free_actual_trajectory", False):
        return False, "actual trajectory failed physical collision audit"
    metrics = summary.get("mppi_mujoco", {})
    if float(metrics.get("final_goal_position_error_m", math.inf)) > max_position_error:
        return False, "final position error exceeds collection limit"
    if float(metrics.get("final_goal_attitude_error_deg", math.inf)) > max_attitude_error_deg:
        return False, "final attitude error exceeds collection limit"
    return True, "accepted"


def _run_pipeline(
    candidate_dir: Path,
    seed: int,
    args: argparse.Namespace,
    start_pose: list[float] | None,
    goal_pose: list[float] | None,
    via_poses: list[list[float]],
    segment_position_bounds: list[dict[str, list[float]]],
    planner_range: float,
) -> tuple[bool, dict[str, Any] | None, str]:
    candidate_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(PIPELINE),
        "--seed", str(seed),
        "--output-dir", str(candidate_dir),
        "--environment", str(args.environment),
        "--urdf", str(args.urdf),
        "--model", str(args.model),
        "--maximum-pair-attempts", str(args.maximum_pair_attempts),
        "--planner-range", f"{planner_range:.9g}",
        "--coal-safety-margin", str(
            args.coal_safety_margin_m
            + args.minimum_planned_buffer_clearance_m
        ),
        "--endpoint-clearance", str(max(
            0.0,
            args.minimum_endpoint_physical_clearance_m
            - args.coal_safety_margin_m
            - args.minimum_planned_buffer_clearance_m,
        )),
        "--path-clearance", "0.0",
        "--mppi-samples", str(args.mppi_samples),
        "--no-gif", "--no-rerun", "--no-plot",
        *args.pipeline_args,
    ]
    task_sampling = getattr(args, "task_sampling", None)
    if task_sampling:
        direction = str(task_sampling.get("direction", "random")).replace("_", "-")
        command.extend(["--sampling-direction", direction])
        for name in ("south_region", "north_region"):
            region = task_sampling[name]
            option = name.replace("_", "-")
            command.extend([f"--{option}-min", *map(str, region["min"])])
            command.extend([f"--{option}-max", *map(str, region["max"])])
    if start_pose is not None and goal_pose is not None:
        command.extend(["--start-pose", *map(str, start_pose)])
        command.extend(["--goal-pose", *map(str, goal_pose)])
    for via_pose in via_poses:
        command.extend(["--via-pose", *map(str, via_pose)])
    for bounds in segment_position_bounds:
        command.extend([
            "--segment-position-bounds",
            *map(str, (*bounds["min"], *bounds["max"])),
        ])
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    log = completed.stdout
    (candidate_dir / "pipeline.log").write_text(log, encoding="utf-8")
    summary_path = candidate_dir / "single_pipeline_summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        tail = "\n".join(log.rstrip().splitlines()[-12:])
        return False, None, f"returncode={completed.returncode}; {tail}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["collection_subprocess_elapsed_s"] = elapsed
    return True, summary, "success"


def _trajectory_record(
    pair_index: int,
    trajectory_index: int,
    seed: int,
    destination: Path,
    root: Path,
    summary: dict[str, Any],
    diversity: dict[str, Any],
    diversity_relaxed: bool,
    attempt: int,
    planner_range: float,
    via_poses: list[list[float]],
    route_template_id: str | None,
    expected_topology: dict[str, str] | None,
    planned_topology: dict[str, str] | None,
    actual_topology: dict[str, str] | None,
) -> dict[str, Any]:
    metrics = summary["mppi_mujoco"]
    audit = summary["coal_collision_audit"]
    return {
        "pair_index": pair_index,
        "trajectory_index": trajectory_index,
        "planner_seed": seed,
        "attempt": attempt,
        "relative_directory": str(destination.relative_to(root)),
        "training_sample": str(
            (destination / "training_sample.npz").relative_to(root)
        ),
        "diffusion_sample": str(
            (destination / "diffusion_sample.npz").relative_to(root)
        ),
        "planner_range_m": planner_range,
        "via_poses": via_poses,
        "route_template_id": route_template_id,
        "expected_topology": expected_topology,
        "planned_topology": planned_topology,
        "actual_topology": actual_topology,
        "diversity": diversity,
        "diversity_relaxed": diversity_relaxed,
        "planning_time_s": summary["planning"]["planning_time_s"],
        "path_length_m": summary["planning"]["translation_length_m"],
        "path_rotation_deg": summary["planning"]["rotation_length_deg"],
        "reference_duration_s": summary["toppra"]["duration"],
        "position_rmse_m": metrics["position_rmse_m"],
        "attitude_rmse_deg": metrics["attitude_rmse_deg"],
        "final_position_error_m": metrics["final_goal_position_error_m"],
        "final_attitude_error_deg": metrics["final_goal_attitude_error_deg"],
        "minimum_planned_physical_clearance_m": audit[
            "minimum_planned_physical_clearance_m"
        ],
        "minimum_actual_physical_clearance_m": audit[
            "minimum_actual_physical_clearance_m"
        ],
    }


def _load_existing_state(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return [], []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        list(manifest.get("trajectories", [])),
        list(manifest.get("recent_failures", [])),
    )


def _update_manifest(
    root: Path,
    args: argparse.Namespace,
    dataset_seed: int,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    status: str,
) -> None:
    topology_counts: dict[str, int] = {}
    for record in records:
        if record.get("actual_topology"):
            key = _signature_key(record["actual_topology"])
            topology_counts[key] = topology_counts.get(key, 0) + 1
    _write_json(root / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset_seed": dataset_seed,
        "requested_pair_count": args.pair_count,
        "requested_trajectories_per_pair": args.trajectories_per_pair,
        "accepted_trajectory_count": len(records),
        "state_layout": STATE_LAYOUT,
        "action_layout": ACTION_LAYOUT,
        "topology_signature_counts": topology_counts,
        "trajectories": sorted(
            records, key=lambda row: (row["pair_index"], row["trajectory_index"])
        ),
        "failure_count": len(failures),
        "recent_failures": failures[-50:],
        "updated_unix_s": time.time(),
    })


def _write_index(root: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "pair_index", "trajectory_index", "planner_seed", "relative_directory",
        "route_template_id", "expected_topology", "planned_topology",
        "actual_topology",
        "path_length_m", "path_rotation_deg", "reference_duration_s",
        "position_rmse_m", "attitude_rmse_deg", "final_position_error_m",
        "final_attitude_error_deg", "minimum_planned_physical_clearance_m",
        "minimum_actual_physical_clearance_m", "diversity_relaxed",
    ]
    with (root / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(
            records, key=lambda row: (row["pair_index"], row["trajectory_index"])
        ))


def _group_splits(pair_count: int, dataset_seed: int) -> dict[str, list[int]]:
    indices = np.arange(pair_count, dtype=np.int64)
    np.random.default_rng(dataset_seed ^ 0x5EED5EED).shuffle(indices)
    if pair_count >= 5:
        test_count = max(1, round(0.125 * pair_count))
        validation_count = max(1, round(0.125 * pair_count))
    elif pair_count >= 3:
        test_count = 1
        validation_count = 1
    elif pair_count == 2:
        test_count = 0
        validation_count = 1
    else:
        test_count = validation_count = 0
    train_count = pair_count - validation_count - test_count
    return {
        "train_pair_indices": indices[:train_count].tolist(),
        "validation_pair_indices": indices[
            train_count:train_count + validation_count
        ].tolist(),
        "test_pair_indices": indices[train_count + validation_count:].tolist(),
    }


def _finalize_dataset(
    root: Path,
    records: list[dict[str, Any]],
    pair_count: int,
    dataset_seed: int,
) -> None:
    records = sorted(records, key=lambda row: (
        row["pair_index"], row["trajectory_index"]
    ))
    samples: list[dict[str, np.ndarray]] = []
    for record in records:
        with np.load(root / record["diffusion_sample"]) as sample:
            samples.append({key: sample[key].copy() for key in sample.files})
    stacked_keys = (
        "condition_start_goal", "normalized_progress", "reference_time", "reference_state",
        "reference_action", "reference_physical_clearance", "actual_time",
        "actual_state", "tracking_reference_state", "actual_action",
        "actual_physical_clearance", "pair_index", "trajectory_index",
        "planner_seed",
    )
    stacked = {key: np.stack([sample[key] for sample in samples])
               for key in stacked_keys}
    np.savez_compressed(root / "dataset_arrays.npz", **stacked)

    statistics: dict[str, Any] = {"sample_count": len(samples)}
    for key in (
        "reference_state", "reference_action", "actual_state",
        "tracking_reference_state", "actual_action",
    ):
        values = stacked[key]
        statistics[key] = {
            "mean": np.mean(values, axis=(0, 1)),
            "std": np.maximum(np.std(values, axis=(0, 1)), 1.0e-8),
            "minimum": np.min(values, axis=(0, 1)),
            "maximum": np.max(values, axis=(0, 1)),
        }
    _write_json(root / "normalization_stats.json", statistics)

    splits = _group_splits(pair_count, dataset_seed)
    split_payload: dict[str, Any] = {
        **splits,
        "note": "Splits are grouped by start/goal pair to prevent leakage.",
    }
    for split_name in ("train", "validation", "test"):
        pair_ids = set(splits[f"{split_name}_pair_indices"])
        split_payload[f"{split_name}_sample_indices"] = [
            index for index, record in enumerate(records)
            if record["pair_index"] in pair_ids
        ]
    _write_json(root / "splits.json", split_payload)


def _write_readme(root: Path) -> None:
    (root / "README.md").write_text(
        """# SE(3) diffusion demonstration dataset

Every accepted sample ran the complete OMPL RRTConnect/COAL -> constrained
B-spline -> TOPP-RA -> MPPI/MuJoCo pipeline and passed a zero-margin physical
collision audit. No GIF, PNG, or Rerun recording is collected.

When the environment provides route templates, every trajectory is assigned a
named corridor class. Paired hard SE(3) portal poses and per-segment position
domains keep RRTConnect on the intended side of each separator. The global
B-spline uses the collision-free OMPL states as a soft guide, preserves every
portal pose as a hard equality constraint, and is rejected if it leaves a
segment domain. Separator-cut signatures independently verify that both the
planned and actual paths used the requested topology. Environments without
templates fall back to perturbing collision-free anchor paths. The complete
B-spline and MuJoCo trace are always validated by COAL. Collection uses the
configured 8 cm safety buffer plus a default 2 cm tracking reserve.

`dataset_arrays.npz` is the fixed-length training view. `condition_start_goal`
has 14 values (`start xyz+qwxyz`, `goal xyz+qwxyz`). State order is
`xyz, linear velocity xyz, quaternion wxyz, body angular velocity xyz`.
Action order is `world linear acceleration xyz, body angular acceleration xyz`.
The `pairs/` tree retains lossless variable-length source logs, planning paths,
TOPP-RA samples, controller diagnostics, collision clearances, and MuJoCo state.

Use `splits.json`; its split is grouped by start/goal pair. Fit preprocessing
only on the training split. Quaternion values are sign-continuous, but should
normally be handled on SO(3), not standardized as unconstrained Euclidean data.
""",
        encoding="utf-8",
    )


def _copy_metadata(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for name, source in (
        ("environment", args.environment),
        ("urdf", args.urdf),
        ("mujoco_model", args.model),
    ):
        source = source.resolve()
        destination = metadata / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        result[name] = {
            "source": str(source),
            "copy": str(destination.relative_to(root)),
            "sha256": _sha256(source),
        }
    return result


def parse_args() -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Collect training-ready SE(3) planning/MPPI trajectories"
    )
    parser.add_argument("--pair-count", type=int, default=8)
    parser.add_argument("--trajectories-per-pair", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_DIR / "datasets" / f"diffusion_se3_{stamp}",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--maximum-pair-attempts", type=int, default=40)
    parser.add_argument("--maximum-trajectory-attempts", type=int, default=12)
    parser.add_argument("--minimum-position-diversity-m", type=float, default=0.025)
    parser.add_argument(
        "--minimum-orientation-diversity-deg", type=float, default=2.0
    )
    parser.add_argument("--normalized-steps", type=int, default=256)
    parser.add_argument("--mppi-samples", type=int, default=384)
    parser.add_argument("--coal-safety-margin-m", type=float, default=0.08)
    parser.add_argument(
        "--minimum-endpoint-physical-clearance-m", type=float, default=0.16
    )
    parser.add_argument(
        "--minimum-planned-buffer-clearance-m", type=float, default=0.02,
        help=(
            "extra clearance beyond the 8 cm COAL planning buffer; the "
            "default reserves tracking-error room"
        ),
    )
    parser.add_argument("--max-final-position-error-m", type=float, default=0.12)
    parser.add_argument("--max-final-attitude-error-deg", type=float, default=8.0)
    parser.add_argument("--strict-diversity", action="store_true")
    parser.add_argument(
        "--disable-topology-templates",
        action="store_true",
        help="ignore environment route templates and use geometric fallback",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "pipeline_args", nargs=argparse.REMAINDER,
        help="extra single-pipeline arguments after --",
    )
    args = parser.parse_args()
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if args.pair_count <= 0 or args.trajectories_per_pair <= 0:
        parser.error("pair and trajectory counts must be positive")
    if args.maximum_pair_attempts <= 0 or args.maximum_trajectory_attempts <= 0:
        parser.error("attempt counts must be positive")
    if args.normalized_steps < 16 or args.mppi_samples <= 0:
        parser.error("normalized steps must be >=16 and MPPI samples positive")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if args.minimum_position_diversity_m <= 0.0:
        parser.error("position diversity threshold must be positive")
    if args.minimum_orientation_diversity_deg <= 0.0:
        parser.error("orientation diversity threshold must be positive")
    if args.minimum_planned_buffer_clearance_m < 0.0:
        parser.error("planned buffer clearance must be non-negative")
    if args.coal_safety_margin_m < 0.0:
        parser.error("COAL safety margin must be non-negative")
    if args.minimum_endpoint_physical_clearance_m < 0.0:
        parser.error("endpoint physical clearance must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(
            f"output directory already exists: {root}; use --resume to continue"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / ".work").mkdir(exist_ok=True)
    if args.resume:
        records, failures = _load_existing_state(root)
    else:
        records, failures = [], []
    dataset_seed = args.seed
    sources = _copy_metadata(root, args)
    environment_payload = json.loads(
        args.environment.read_text(encoding="utf-8")
    )
    args.task_sampling = environment_payload.get("task_sampling")
    route_templates = (
        [] if args.disable_topology_templates
        else list(environment_payload.get("route_templates", []))
    )
    topology_cuts = list(environment_payload.get("topology_cuts", []))
    topology_mode = bool(route_templates and topology_cuts)
    position_bounds = environment_payload["sampling_space"]["position_bounds"]
    bounds_min = np.asarray(position_bounds["min"], dtype=np.float64)
    bounds_max = np.asarray(position_bounds["max"], dtype=np.float64)
    _write_readme(root)
    _write_json(root / "dataset_config.json", {
        "schema_version": SCHEMA_VERSION,
        "dataset_seed": dataset_seed,
        "pair_count": args.pair_count,
        "trajectories_per_pair": args.trajectories_per_pair,
        "normalized_steps": args.normalized_steps,
        "mppi_samples": args.mppi_samples,
        "coal_safety_margin_m": args.coal_safety_margin_m,
        "minimum_endpoint_physical_clearance_m": (
            args.minimum_endpoint_physical_clearance_m
        ),
        "minimum_planned_buffer_clearance_m": (
            args.minimum_planned_buffer_clearance_m
        ),
        "effective_collection_safety_margin_m": (
            args.coal_safety_margin_m
            + args.minimum_planned_buffer_clearance_m
        ),
        "maximum_pair_attempts": args.maximum_pair_attempts,
        "maximum_trajectory_attempts": args.maximum_trajectory_attempts,
        "minimum_position_diversity_m": args.minimum_position_diversity_m,
        "minimum_orientation_diversity_deg": (
            args.minimum_orientation_diversity_deg
        ),
        "strict_diversity": args.strict_diversity,
        "topology_template_mode": topology_mode,
        "route_template_ids": [template["id"] for template in route_templates],
        "task_sampling": args.task_sampling,
        "max_final_position_error_m": args.max_final_position_error_m,
        "max_final_attitude_error_deg": args.max_final_attitude_error_deg,
        "pipeline_args": args.pipeline_args,
        "sources": sources,
    })
    _update_manifest(root, args, dataset_seed, records, failures, "collecting")

    for pair_index in range(args.pair_count):
        pair_dir = root / "pairs" / f"pair_{pair_index:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_json = pair_dir / "pair.json"
        if pair_json.exists():
            pair_payload = json.loads(pair_json.read_text(encoding="utf-8"))
            start_pose = pair_payload["start_pose"]
            goal_pose = pair_payload["goal_pose"]
        else:
            start_pose = goal_pose = None

        accepted_descriptors: list[dict[str, np.ndarray]] = []
        accepted_topology_keys: set[str] = set()
        for existing in sorted(
            (row for row in records if row["pair_index"] == pair_index),
            key=lambda row: row["trajectory_index"],
        ):
            accepted_descriptors.append(_descriptor(root / existing["training_sample"]))
            if existing.get("actual_topology"):
                accepted_topology_keys.add(
                    _signature_key(existing["actual_topology"])
                )

        for trajectory_index in range(args.trajectories_per_pair):
            if any(
                row["pair_index"] == pair_index
                and row["trajectory_index"] == trajectory_index
                for row in records
            ):
                print(
                    f"PAIR {pair_index + 1}/{args.pair_count} TRAJECTORY "
                    f"{trajectory_index + 1}/{args.trajectories_per_pair}: resume-skip",
                    flush=True,
                )
                continue

            best: tuple[Any, ...] | None = None
            accepted_candidate: tuple[Any, ...] | None = None
            route_template = (
                route_templates[trajectory_index % len(route_templates)]
                if topology_mode else None
            )
            route_template_id = (
                str(route_template["id"]) if route_template else None
            )
            expected_topology = (
                dict(route_template["expected_topology"])
                if route_template else None
            )
            for attempt in range(1, args.maximum_trajectory_attempts + 1):
                seed = _derived_seed(
                    dataset_seed, pair_index, trajectory_index, attempt
                )
                if any(
                    failure.get("pair_index") == pair_index
                    and failure.get("trajectory_index") == trajectory_index
                    and failure.get("attempt") == attempt
                    and failure.get("seed") == seed
                    for failure in failures
                ):
                    print(
                        f"PAIR {pair_index + 1}/{args.pair_count} TRAJECTORY "
                        f"{trajectory_index + 1}/{args.trajectories_per_pair} "
                        f"attempt={attempt}: failed-attempt resume-skip",
                        flush=True,
                    )
                    continue
                planner_rng = np.random.default_rng(seed ^ 0x72616E67)
                planner_range = float(planner_rng.uniform(0.24, 0.44))
                via_poses: list[list[float]] = []
                segment_position_bounds: list[dict[str, list[float]]] = []
                if route_template is not None:
                    via_poses = _sample_route_template(route_template, seed)
                    segment_position_bounds = list(
                        route_template.get("segment_position_bounds", [])
                    )
                elif trajectory_index > 0 and start_pose is not None and goal_pose is not None:
                    anchor_descriptor = accepted_descriptors[
                        (trajectory_index + attempt - 1)
                        % len(accepted_descriptors)
                    ]
                    via_poses = [
                        _sample_via_pose(
                            start_pose,
                            goal_pose,
                            bounds_min,
                            bounds_max,
                            seed,
                            anchor_path=anchor_descriptor["planned"],
                            perturb=(attempt < args.maximum_trajectory_attempts),
                        )
                    ]
                candidate_dir = (
                    root / ".work" / f"pair_{pair_index:03d}"
                    / f"trajectory_{trajectory_index:03d}"
                    / f"attempt_{attempt:02d}_{seed}"
                )
                if candidate_dir.exists():
                    shutil.rmtree(candidate_dir)
                print(
                    f"PAIR {pair_index + 1}/{args.pair_count} TRAJECTORY "
                    f"{trajectory_index + 1}/{args.trajectories_per_pair} "
                    f"attempt={attempt}/{args.maximum_trajectory_attempts} seed={seed}",
                    flush=True,
                )
                success, summary, reason = _run_pipeline(
                    candidate_dir,
                    seed,
                    args,
                    start_pose,
                    goal_pose,
                    via_poses,
                    segment_position_bounds,
                    planner_range,
                )
                if not success or summary is None:
                    failures.append({
                        "pair_index": pair_index,
                        "trajectory_index": trajectory_index,
                        "attempt": attempt,
                        "seed": seed,
                        "reason": reason[-3000:],
                    })
                    shutil.rmtree(candidate_dir, ignore_errors=True)
                    _update_manifest(
                        root, args, dataset_seed, records, failures, "collecting"
                    )
                    continue
                quality_ok, quality_reason = _quality_check(
                    summary,
                    args.max_final_position_error_m,
                    args.max_final_attitude_error_deg,
                )
                if not quality_ok:
                    failures.append({
                        "pair_index": pair_index,
                        "trajectory_index": trajectory_index,
                        "attempt": attempt,
                        "seed": seed,
                        "reason": quality_reason,
                    })
                    shutil.rmtree(candidate_dir, ignore_errors=True)
                    continue

                descriptor = _descriptor(candidate_dir / "training_sample.npz")
                diversity = _diversity(
                    descriptor,
                    accepted_descriptors,
                    args.minimum_position_diversity_m,
                    args.minimum_orientation_diversity_deg,
                )
                planned_topology = (
                    _topology_signature(descriptor["planned"], topology_cuts)
                    if topology_mode else None
                )
                actual_topology = (
                    _topology_signature(descriptor["actual"], topology_cuts)
                    if topology_mode else None
                )
                if topology_mode:
                    planned_matches = planned_topology == expected_topology
                    actual_matches = actual_topology == expected_topology
                    topology_key = _signature_key(actual_topology)
                    new_topology_for_pair = (
                        topology_key not in accepted_topology_keys
                    )
                    diversity.update({
                        "route_template_id": route_template_id,
                        "expected_topology": expected_topology,
                        "planned_topology": planned_topology,
                        "actual_topology": actual_topology,
                        "planned_topology_matches": planned_matches,
                        "actual_topology_matches": actual_matches,
                        "new_topology_for_pair": new_topology_for_pair,
                    })
                    if not (planned_matches and actual_matches):
                        failures.append({
                            "pair_index": pair_index,
                            "trajectory_index": trajectory_index,
                            "attempt": attempt,
                            "seed": seed,
                            "reason": (
                                "topology mismatch: expected="
                                f"{expected_topology}, planned={planned_topology}, "
                                f"actual={actual_topology}"
                            ),
                        })
                        shutil.rmtree(candidate_dir, ignore_errors=True)
                        continue
                    if (
                        not new_topology_for_pair
                        and not bool(diversity["accepted_by_threshold"])
                    ):
                        failures.append({
                            "pair_index": pair_index,
                            "trajectory_index": trajectory_index,
                            "attempt": attempt,
                            "seed": seed,
                            "reason": (
                                "repeated topology did not meet within-class "
                                "geometric diversity thresholds"
                            ),
                        })
                        shutil.rmtree(candidate_dir, ignore_errors=True)
                        continue
                    accepted_candidate = (
                        candidate_dir, summary, descriptor, diversity,
                        attempt, seed, planner_range, via_poses,
                        route_template_id, expected_topology,
                        planned_topology, actual_topology,
                    )
                    break
                if bool(diversity["accepted_by_threshold"]):
                    accepted_candidate = (
                        candidate_dir, summary, descriptor, diversity,
                        attempt, seed, planner_range, via_poses,
                        route_template_id, expected_topology,
                        planned_topology, actual_topology,
                    )
                    break
                score = float(diversity["nearest_threshold_score"])
                if best is None or score > best[0]:
                    if best is not None:
                        shutil.rmtree(best[1], ignore_errors=True)
                    best = (
                        score, candidate_dir, summary, descriptor, diversity,
                        attempt, seed, planner_range, via_poses,
                        route_template_id, expected_topology,
                        planned_topology, actual_topology,
                    )
                else:
                    shutil.rmtree(candidate_dir, ignore_errors=True)

            diversity_relaxed = False
            if (
                not topology_mode
                and accepted_candidate is None
                and best is not None
                and not args.strict_diversity
            ):
                (
                    _, candidate_dir, summary, descriptor, diversity,
                    attempt, seed, planner_range, via_poses,
                    route_template_id, expected_topology,
                    planned_topology, actual_topology,
                ) = best
                accepted_candidate = (
                    candidate_dir, summary, descriptor, diversity,
                    attempt, seed, planner_range, via_poses,
                    route_template_id, expected_topology,
                    planned_topology, actual_topology,
                )
                diversity_relaxed = True
            if accepted_candidate is None:
                _update_manifest(root, args, dataset_seed, records, failures, "failed")
                raise RuntimeError(
                    f"could not collect pair {pair_index} trajectory {trajectory_index}"
                )

            (
                candidate_dir, summary, descriptor, diversity,
                attempt, seed, planner_range, via_poses,
                route_template_id, expected_topology,
                planned_topology, actual_topology,
            ) = accepted_candidate
            if best is not None and best[1] != candidate_dir:
                shutil.rmtree(best[1], ignore_errors=True)
            destination = pair_dir / f"trajectory_{trajectory_index:03d}"
            if destination.exists():
                raise FileExistsError(f"accepted destination exists: {destination}")
            os.replace(candidate_dir, destination)
            if start_pose is None or goal_pose is None:
                start_pose = _pose_from_summary(summary, "random_start")
                goal_pose = _pose_from_summary(summary, "random_goal")
                _write_json(pair_json, {
                    "pair_index": pair_index,
                    "start_pose": start_pose,
                    "goal_pose": goal_pose,
                })
            _normalize_sample(
                destination / "training_sample.npz",
                destination / "diffusion_sample.npz",
                pair_index,
                trajectory_index,
                seed,
                args.normalized_steps,
            )
            record = _trajectory_record(
                pair_index, trajectory_index, seed, destination, root, summary,
                diversity, diversity_relaxed, attempt, planner_range,
                via_poses, route_template_id, expected_topology,
                planned_topology, actual_topology,
            )
            records.append(record)
            accepted_descriptors.append(descriptor)
            if actual_topology:
                accepted_topology_keys.add(_signature_key(actual_topology))
            _update_manifest(root, args, dataset_seed, records, failures, "collecting")
            print(
                f"ACCEPTED pair={pair_index:03d} trajectory={trajectory_index:03d} "
                f"path={record['path_length_m']:.3f}m "
                f"pos_rmse={record['position_rmse_m']:.3f}m "
                f"topology={actual_topology} "
                f"diversity_relaxed={diversity_relaxed}",
                flush=True,
            )

    _write_index(root, records)
    _finalize_dataset(root, records, args.pair_count, dataset_seed)
    shutil.rmtree(root / ".work", ignore_errors=True)
    _update_manifest(root, args, dataset_seed, records, failures, "complete")
    print(f"DATASET_COMPLETE={root}", flush=True)
    print(f"TRAJECTORIES={len(records)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("collection interrupted; rerun with --resume", file=sys.stderr)
        raise SystemExit(130)
