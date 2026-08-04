"""Run one complete random SE(3) planning and tracking trial.

Pipeline:

1. Load the cube environment JSON and the existing rigid base-link URDF
   collision model.
2. Sample a separated, collision-free random start/goal pose pair.
3. Plan with OMPL RRTConnect (bidirectional RRT) using COAL for every SE(3)
   validity query; simplify the OMPL path and construct a collision-validated
   interpolating SE(3) B-spline.
4. Retiming the same spline with kinematic TOPP-RA.
5. Track the timed reference with the existing 6-DoF MPPI + geometric
   low-level controller on the full MuJoCo HNUTER model.
6. Save machine-readable diagnostics, a Rerun recording and an offscreen
   MuJoCo replay GIF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from coal_collision import CoalCollisionChecker, StaticCollisionObject
from hnuter_multi_waypoint_demo import MultiWaypointProblem
from hnuter_ompl_mppi_demo import run_demo, save_results
from mppi.quaternion import (
    normalize_quaternion,
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_to_euler,
)
from multi_waypoint_planner import (
    InterpolatingSE3BSpline,
    MultiWaypointOMPLPlanner,
    WaypointConstrainedSmoothingSE3BSpline,
)
from ompl_se3_planner import OMPLSE3Planner, PlannedSE3Path, SE3Pose
from rerun_bridge import Box3D
from toppra_retiming import ToppraTimedReference


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ENVIRONMENT = PROJECT_DIR / "environment_overfit_cube_v001.json"
DEFAULT_URDF = (
    PROJECT_DIR
    / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf"
)
DEFAULT_MJCF = PROJECT_DIR / "hnuter206_4_5kg.xml"
DEFAULT_OUTPUT = PROJECT_DIR / "results/overfit_cube_single_run"


COLLECTION_RGBA = {
    "FLOOR": (0.20, 0.24, 0.30, 1.0),
    "PILLARS": (0.10, 0.34, 0.56, 1.0),
    "GATES": (0.03, 0.20, 0.78, 1.0),
    "CENTRAL_OBSTACLES": (0.10, 0.34, 0.56, 1.0),
    "BEAMS": (0.92, 0.28, 0.04, 1.0),
    "PARTIAL_ENCLOSURES": (0.04, 0.54, 0.18, 1.0),
}

RERUN_COLLECTION_RGBA = {
    "FLOOR": (70, 78, 92, 255),
    "PILLARS": (25, 105, 175, 190),
    "GATES": (20, 65, 220, 195),
    "CENTRAL_OBSTACLES": (28, 115, 180, 190),
    "BEAMS": (235, 78, 18, 195),
    "PARTIAL_ENCLOSURES": (18, 165, 72, 195),
}


def _float_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def load_environment(
    path: Path,
) -> tuple[dict[str, Any], tuple[StaticCollisionObject, ...]]:
    """Load every exported collision box into the backend-neutral COAL form."""

    data = json.loads(path.read_text(encoding="utf-8"))
    environment_id = data.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id.strip():
        raise ValueError(f"missing environment_id in {path}")
    obstacles = []
    for item in data.get("obstacles", []):
        if not item.get("collision", False):
            continue
        if item.get("type") != "box":
            raise ValueError(
                f"environment object {item.get('id')!r} is not a box"
            )
        pose = item["pose"]
        obstacles.append(
            StaticCollisionObject.box(
                item["id"],
                _float_vector(item["size_xyz"], 3, "size_xyz"),
                _float_vector(pose["position"], 3, "position"),
                _float_vector(
                    pose["quaternion_wxyz"], 4, "quaternion_wxyz"
                ),
            )
        )
    if not obstacles:
        raise ValueError("environment contains no collision boxes")
    return data, tuple(obstacles)


def rerun_environment_boxes(
    environment_data: dict[str, Any],
) -> tuple[Box3D, ...]:
    """Convert the exported box map to Rerun-native oriented boxes."""

    boxes = []
    for item in environment_data.get("obstacles", []):
        if not item.get("collision", False):
            continue
        pose = item["pose"]
        collection = item["collection"]
        boxes.append(
            Box3D(
                center=_float_vector(
                    pose["position"], 3, "position"
                ),
                half_size=_float_vector(
                    item["half_extents"], 3, "half_extents"
                ),
                quaternion_wxyz=_float_vector(
                    pose["quaternion_wxyz"],
                    4,
                    "quaternion_wxyz",
                ),
                label=f"{collection}: {item['id']}",
                color=RERUN_COLLECTION_RGBA.get(
                    collection, (105, 120, 145, 190)
                ),
            )
        )
    return tuple(boxes)


def random_pose(
    rng: np.random.Generator,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_tilt_deg: float,
) -> SE3Pose:
    """Sample position, roll, pitch and unrestricted yaw."""

    position = rng.uniform(bounds_min, bounds_max)
    tilt = math.radians(max_tilt_deg)
    rpy = np.asarray(
        (
            rng.uniform(-tilt, tilt),
            rng.uniform(-tilt, tilt),
            rng.uniform(-math.pi, math.pi),
        )
    )
    return SE3Pose(position, quaternion_from_euler(rpy))


def quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(abs(np.dot(first, second)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def sample_valid_pose(
    rng: np.random.Generator,
    planner: OMPLSE3Planner,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    *,
    max_tilt_deg: float,
    minimum_clearance: float,
    maximum_attempts: int,
) -> tuple[SE3Pose, int, float]:
    for attempt in range(1, maximum_attempts + 1):
        pose = random_pose(rng, bounds_min, bounds_max, max_tilt_deg)
        if not planner.is_pose_valid(pose.position, pose.quaternion):
            continue
        clearance = float(
            planner.clearance(pose.position, pose.quaternion)
        )
        if clearance >= minimum_clearance:
            return pose, attempt, clearance
    raise RuntimeError(
        f"failed to sample a pose with {minimum_clearance:.3f} m "
        f"clearance after {maximum_attempts} attempts"
    )


def plan_pose_pair(
    planner: OMPLSE3Planner,
    start: SE3Pose,
    goal: SE3Pose,
    args: argparse.Namespace,
    via_poses: tuple[SE3Pose, ...] = (),
) -> Any:
    segment_position_bounds = None
    if args.segment_position_bounds:
        segment_position_bounds = tuple(
            (
                np.asarray(values[:3], dtype=np.float64),
                np.asarray(values[3:6], dtype=np.float64),
            )
            for values in args.segment_position_bounds
        )
    return MultiWaypointOMPLPlanner(planner).plan(
        (start, *via_poses, goal),
        segment_position_bounds=segment_position_bounds,
        solve_time_per_segment=args.solve_time,
        interpolation_resolution=args.path_resolution,
        minimum_states_per_segment=args.minimum_states,
        knot_stride=args.spline_knot_stride,
        spline_samples=args.spline_samples,
        orientation_metric_weight=args.orientation_metric_weight,
        spline_method=args.spline_method,
        smoothing_degree=args.smoothing_degree,
        smoothing_guide_weight=args.smoothing_guide_weight,
        smoothing_position_acceleration_weight=(
            args.smoothing_position_acceleration_weight
        ),
        smoothing_position_jerk_weight=(
            args.smoothing_position_jerk_weight
        ),
        smoothing_orientation_acceleration_weight=(
            args.smoothing_orientation_acceleration_weight
        ),
        smoothing_orientation_jerk_weight=(
            args.smoothing_orientation_jerk_weight
        ),
        smoothing_clearance_weight_scale=(
            args.smoothing_clearance_weight_scale
        ),
        smoothing_max_attempts=args.smoothing_max_attempts,
        shortest_orientation_guide=not args.no_orientation_shortcut,
    )


def plan_external_diffusion_path(
    planner: OMPLSE3Planner,
    path_file: Path,
    args: argparse.Namespace,
) -> tuple[SE3Pose, SE3Pose, Any, dict[str, Any]]:
    """Fit and validate the normal execution spline from a diffusion guide."""

    with np.load(path_file) as payload:
        if "poses_wxyz" in payload:
            states = payload["poses_wxyz"].astype(np.float64)
        elif "states" in payload:
            states = payload["states"].astype(np.float64)
        else:
            raise ValueError(
                "external path NPZ must contain poses_wxyz or states"
            )
        sampling_time_s = float(
            payload["sampling_time_s"]
            if "sampling_time_s" in payload else 0.0
        )
        source = str(
            payload["source"] if "source" in payload
            else "external diffusion guide"
        )
    if states.ndim == 3 and len(states) == 1:
        states = states[0]
    if (
        states.ndim != 2
        or states.shape[1] != 7
        or len(states) < 8
        or not np.all(np.isfinite(states))
    ):
        raise ValueError("external path must have finite shape (N, 7), N >= 8")
    states[:, 3:7] = normalize_quaternion(states[:, 3:7])
    for index in range(1, len(states)):
        if np.dot(states[index - 1, 3:7], states[index, 3:7]) < 0.0:
            states[index, 3:7] *= -1.0
    pose_change = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
    pose_change += args.orientation_metric_weight * np.linalg.norm(
        quaternion_error_vector(states[1:, 3:7], states[:-1, 3:7]),
        axis=1,
    )
    keep = np.concatenate(([True], pose_change > 1.0e-8))
    states = states[keep]
    if len(states) < 8:
        raise RuntimeError("external diffusion path has too few distinct poses")

    clearance = planner.clearance(states[:, :3], states[:, 3:7])
    if np.any(clearance <= 0.0):
        raise RuntimeError(
            "external diffusion guide failed the configured COAL margin "
            f"(minimum adjusted clearance {float(np.min(clearance)):.4f} m)"
        )
    start = SE3Pose(states[0, :3], states[0, 3:7])
    goal = SE3Pose(states[-1, :3], states[-1, 3:7])
    waypoints = (start, goal)
    waypoint_indices = (0, len(states) - 1)
    translation_length = float(
        np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum()
    )
    rotation_length = float(
        np.linalg.norm(
            quaternion_error_vector(states[1:, 3:7], states[:-1, 3:7]),
            axis=1,
        ).sum()
    )
    segment = PlannedSE3Path(
        states=states.copy(),
        planning_time_s=sampling_time_s,
        raw_state_count=len(states),
        path_length_m=translation_length,
        rotation_length_rad=rotation_length,
        planner_name="U-Net diffusion + inference guidance",
    )
    clearance_weight = 1.0 + np.square(
        args.smoothing_clearance_weight_scale
        / (np.maximum(clearance, 0.0) + 0.01)
    )
    clearance_weight = np.minimum(clearance_weight, 400.0)
    builder = MultiWaypointOMPLPlanner(planner)
    last_error: Exception | None = None
    for attempt in range(args.smoothing_max_attempts):
        stride = max(1, args.spline_knot_stride - attempt)
        try:
            spline = WaypointConstrainedSmoothingSE3BSpline(
                states,
                waypoint_indices,
                degree=args.smoothing_degree,
                control_point_stride=stride,
                orientation_metric_weight=args.orientation_metric_weight,
                guide_weight=args.smoothing_guide_weight * 5.0**attempt,
                guide_sample_weights=clearance_weight,
                position_acceleration_weight=(
                    args.smoothing_position_acceleration_weight
                ),
                position_jerk_weight=args.smoothing_position_jerk_weight,
                orientation_acceleration_weight=(
                    args.smoothing_orientation_acceleration_weight
                ),
                orientation_jerk_weight=(
                    args.smoothing_orientation_jerk_weight
                ),
            )
            plan = builder._build_validated_plan(
                waypoints,
                (segment,),
                states,
                waypoint_indices,
                spline,
                spline.waypoint_parameters,
                args.spline_samples,
                stride,
                False,
                None,
            )
            diagnostics = {
                "mode": "external_unet_guidance_path",
                "source": source,
                "external_path": str(path_file.resolve()),
                "diffusion_state_count": len(states),
                "diffusion_sampling_time_s": sampling_time_s,
                "diffusion_minimum_adjusted_clearance_m": float(
                    np.min(clearance)
                ),
                "spline_fit_attempt": attempt + 1,
            }
            return start, goal, plan, diagnostics
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            last_error = error
    raise RuntimeError(
        "could not fit a collision-free execution B-spline to the external "
        f"diffusion guide: {last_error}"
    )


def _via_poses_from_args(args: argparse.Namespace) -> tuple[SE3Pose, ...]:
    return tuple(
        SE3Pose(
            np.asarray(values, dtype=np.float64)[:3],
            np.asarray(values, dtype=np.float64)[3:7],
        )
        for values in (args.via_pose or ())
    )


def _validate_via_poses(
    planner: OMPLSE3Planner,
    via_poses: tuple[SE3Pose, ...],
    minimum_clearance: float,
) -> None:
    for index, pose in enumerate(via_poses):
        if not planner.is_pose_valid(pose.position, pose.quaternion):
            raise RuntimeError(f"via pose {index} is invalid or colliding")
        clearance = float(planner.clearance(pose.position, pose.quaternion))
        if clearance < minimum_clearance:
            raise RuntimeError(
                f"via pose {index} clearance {clearance:.6f} m is below "
                f"the path requirement {minimum_clearance:.6f} m"
            )


def sample_and_plan(
    rng: np.random.Generator,
    planner: OMPLSE3Planner,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    args: argparse.Namespace,
) -> tuple[SE3Pose, SE3Pose, Any, dict[str, Any]]:
    """Reject uninteresting/disconnected pairs while retaining randomness."""

    via_poses = _via_poses_from_args(args)
    _validate_via_poses(planner, via_poses, args.path_clearance)
    diagnostics: dict[str, Any] = {
        "mode": "random_pair_with_via" if via_poses else "random_pair",
        "via_pose_count": len(via_poses),
        "via_poses": [
            np.concatenate((pose.position, pose.quaternion)).tolist()
            for pose in via_poses
        ],
        "pair_attempts": [],
    }
    for pair_attempt in range(1, args.maximum_pair_attempts + 1):
        if args.south_region_min is None:
            south_min = bounds_min.copy()
            south_max = bounds_max.copy()
            south_max[1] = min(-0.55, bounds_max[1])
            south_max[2] = min(args.endpoint_z_max, bounds_max[2])
            north_min = bounds_min.copy()
            north_max = bounds_max.copy()
            north_min[1] = max(0.55, bounds_min[1])
            north_max[2] = min(args.endpoint_z_max, bounds_max[2])
        else:
            south_min = np.maximum(
                bounds_min, np.asarray(args.south_region_min, dtype=np.float64)
            )
            south_max = np.minimum(
                bounds_max, np.asarray(args.south_region_max, dtype=np.float64)
            )
            north_min = np.maximum(
                bounds_min, np.asarray(args.north_region_min, dtype=np.float64)
            )
            north_max = np.minimum(
                bounds_max, np.asarray(args.north_region_max, dtype=np.float64)
            )
        choose_south_to_north = (
            args.sampling_direction == "south-to-north"
            or (
                args.sampling_direction == "random"
                and rng.random() < 0.5
            )
        )
        if choose_south_to_north:
            start_min, start_max = south_min, south_max
            goal_min, goal_max = north_min, north_max
            sampling_direction = "south_to_north"
        else:
            start_min, start_max = north_min, north_max
            goal_min, goal_max = south_min, south_max
            sampling_direction = "north_to_south"
        start, start_attempts, start_clearance = sample_valid_pose(
            rng,
            planner,
            start_min,
            start_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=args.maximum_pose_attempts,
        )
        goal, goal_attempts, goal_clearance = sample_valid_pose(
            rng,
            planner,
            goal_min,
            goal_max,
            max_tilt_deg=args.max_tilt_deg,
            minimum_clearance=args.endpoint_clearance,
            maximum_attempts=args.maximum_pose_attempts,
        )
        position_separation = float(
            np.linalg.norm(goal.position - start.position)
        )
        y_separation = float(abs(goal.position[1] - start.position[1]))
        attitude_separation = quaternion_distance(
            start.quaternion, goal.quaternion
        )
        attempt_record = {
            "pair_attempt": pair_attempt,
            "sampling_direction": sampling_direction,
            "start_pose_attempts": start_attempts,
            "goal_pose_attempts": goal_attempts,
            "start_clearance_m": start_clearance,
            "goal_clearance_m": goal_clearance,
            "position_separation_m": position_separation,
            "y_separation_m": y_separation,
            "attitude_separation_deg": math.degrees(
                attitude_separation
            ),
        }
        diagnostics["pair_attempts"].append(attempt_record)
        if (
            position_separation < args.minimum_pair_distance
            or y_separation < args.minimum_y_separation
            or attitude_separation < math.radians(
                args.minimum_attitude_separation_deg
            )
        ):
            attempt_record["result"] = "rejected_separation"
            print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
            continue
        try:
            plan = plan_pose_pair(planner, start, goal, args, via_poses)
        except (RuntimeError, ValueError) as error:
            attempt_record["result"] = "planning_failed"
            attempt_record["error"] = str(error)
            print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
            continue
        attempt_record["spline_minimum_clearance_m"] = (
            plan.minimum_clearance_m
        )
        detour_ratio = (
            plan.spline_path.path_length_m / position_separation
        )
        path_rotation_deg = math.degrees(
            plan.spline_path.rotation_length_rad
        )
        attitude_separation_deg = math.degrees(attitude_separation)
        rotation_stretch = path_rotation_deg / max(
            attitude_separation_deg, 1.0e-6
        )
        attempt_record["detour_ratio"] = detour_ratio
        attempt_record["path_rotation_deg"] = path_rotation_deg
        attempt_record["rotation_stretch"] = rotation_stretch
        attempt_record["orientation_shortcut_applied"] = (
            plan.orientation_shortcut_applied
        )
        if plan.minimum_clearance_m < args.path_clearance:
            attempt_record["result"] = "rejected_path_clearance"
            print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
            continue
        if detour_ratio + 1.0e-9 < args.minimum_detour_ratio:
            attempt_record["result"] = "rejected_trivial_direct_path"
            print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
            continue
        exceeds_absolute_rotation_limit = bool(
            args.maximum_path_rotation_deg > 0.0
            and path_rotation_deg > args.maximum_path_rotation_deg
        )
        exceeds_rotation_stretch_limit = bool(
            args.maximum_rotation_stretch > 0.0
            and rotation_stretch > args.maximum_rotation_stretch
        )
        attempt_record["orientation_quality_warning"] = bool(
            path_rotation_deg > 180.0 or rotation_stretch > 2.2
        )
        if (
            exceeds_absolute_rotation_limit
            or exceeds_rotation_stretch_limit
        ):
            attempt_record["result"] = "rejected_orientation_winding"
            print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
            continue
        attempt_record["result"] = "accepted"
        print("PAIR_ATTEMPT=" + json.dumps(attempt_record))
        diagnostics["accepted_pair_attempt"] = pair_attempt
        return start, goal, plan, diagnostics
    raise RuntimeError(
        "no random start/goal pair produced an accepted collision-free "
        f"B-spline after {args.maximum_pair_attempts} pair attempts"
    )


def plan_fixed_pose_pair(
    planner: OMPLSE3Planner,
    args: argparse.Namespace,
) -> tuple[SE3Pose, SE3Pose, Any, dict[str, Any]]:
    """Plan a caller-supplied pair for same-task trajectory diversity."""

    assert args.start_pose is not None and args.goal_pose is not None
    start_values = np.asarray(args.start_pose, dtype=np.float64)
    goal_values = np.asarray(args.goal_pose, dtype=np.float64)
    start = SE3Pose(start_values[:3], start_values[3:7])
    goal = SE3Pose(goal_values[:3], goal_values[3:7])
    for name, pose in (("start", start), ("goal", goal)):
        if not planner.is_pose_valid(pose.position, pose.quaternion):
            raise RuntimeError(f"fixed {name} pose is invalid or colliding")
        clearance = float(planner.clearance(pose.position, pose.quaternion))
        if clearance < args.endpoint_clearance:
            raise RuntimeError(
                f"fixed {name} pose clearance {clearance:.6f} m is below "
                f"the required {args.endpoint_clearance:.6f} m"
            )
    via_poses = _via_poses_from_args(args)
    _validate_via_poses(planner, via_poses, args.path_clearance)
    plan = plan_pose_pair(planner, start, goal, args, via_poses)
    path_rotation_deg = math.degrees(
        plan.spline_path.rotation_length_rad
    )
    attitude_separation_deg = math.degrees(
        quaternion_distance(start.quaternion, goal.quaternion)
    )
    rotation_stretch = path_rotation_deg / max(
        attitude_separation_deg, 1.0e-6
    )
    if plan.minimum_clearance_m < args.path_clearance:
        raise RuntimeError(
            "fixed-pair B-spline did not meet planning clearance: "
            f"{plan.minimum_clearance_m:.6f} m"
        )
    if (
        args.maximum_path_rotation_deg > 0.0
        and path_rotation_deg > args.maximum_path_rotation_deg
    ) or (
        args.maximum_rotation_stretch > 0.0
        and rotation_stretch > args.maximum_rotation_stretch
    ):
        raise RuntimeError("fixed-pair path exceeded optional rotation limit")
    diagnostics = {
        "mode": "fixed_pair_with_via" if via_poses else "fixed_pair",
        "via_pose_count": len(via_poses),
        "via_poses": [
            np.concatenate((pose.position, pose.quaternion)).tolist()
            for pose in via_poses
        ],
        "path_rotation_deg": path_rotation_deg,
        "attitude_separation_deg": attitude_separation_deg,
        "rotation_stretch": rotation_stretch,
        "orientation_quality_warning": bool(
            path_rotation_deg > 180.0 or rotation_stretch > 2.2
        ),
        "orientation_shortcut_applied": (
            plan.orientation_shortcut_applied
        ),
    }
    return start, goal, plan, diagnostics


def _numbers(values: Any) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def translation_path_metrics(
    states: np.ndarray,
    clearance: np.ndarray,
    *,
    comparison_step_m: float = 0.025,
) -> dict[str, Any]:
    """Compute comparable shape metrics after uniform arc-length sampling."""

    state_array = np.asarray(states, dtype=np.float64)
    positions = state_array[:, :3]
    deltas = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    keep = np.concatenate(([True], segment_lengths > 1.0e-10))
    positions = positions[keep]
    if len(positions) < 2:
        raise ValueError("path must contain two distinct positions")
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    )
    length = float(cumulative[-1])
    sample_count = max(3, int(math.ceil(length / comparison_step_m)) + 1)
    sample_distance = np.linspace(0.0, length, sample_count)
    uniform = np.column_stack(
        [np.interp(sample_distance, cumulative, positions[:, axis]) for axis in range(3)]
    )
    directions = np.diff(uniform, axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    turning_deg = np.degrees(
        np.arccos(
            np.clip(
                np.sum(directions[:-1] * directions[1:], axis=1),
                -1.0,
                1.0,
            )
        )
    )
    rotation_steps = quaternion_error_vector(
        state_array[1:, 3:7], state_array[:-1, 3:7]
    )
    return {
        "state_count": len(state_array),
        "translation_length_m": length,
        "rotation_length_deg": float(
            np.degrees(np.sum(np.linalg.norm(rotation_steps, axis=1)))
        ),
        "comparison_step_m": comparison_step_m,
        "maximum_local_turn_deg": float(np.max(turning_deg)),
        "p99_local_turn_deg": float(np.percentile(turning_deg, 99.0)),
        "minimum_planning_buffer_clearance_m": float(np.min(clearance)),
    }


def build_interpolating_baseline(
    raw_states: np.ndarray,
    *,
    knot_stride: int,
    sample_count: int,
    orientation_metric_weight: float,
) -> tuple[InterpolatingSE3BSpline, np.ndarray]:
    """Rebuild the previous cubic interpolation on the identical OMPL guide."""

    selected_indices = sorted(
        set(range(0, len(raw_states), knot_stride))
        | {len(raw_states) - 1}
    )
    spline = InterpolatingSE3BSpline(
        raw_states[selected_indices],
        degree=3,
        orientation_metric_weight=orientation_metric_weight,
    )
    states = spline.evaluate(np.linspace(0.0, 1.0, sample_count))
    return spline, states


def save_comparison_path(
    output_path: Path,
    states: np.ndarray,
    clearance: np.ndarray,
) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "index",
                "x",
                "y",
                "z",
                "qw",
                "qx",
                "qy",
                "qz",
                "planning_buffer_clearance_m",
            )
        )
        for index, (state, distance) in enumerate(zip(states, clearance)):
            writer.writerow((index, *state, distance))
    return output_path


def build_combined_mjcf(
    source_mjcf: Path,
    environment_data: dict[str, Any],
    spline_states: np.ndarray,
    output_path: Path,
) -> Path:
    """Embed the box map and path markers in a standalone MuJoCo model."""

    tree = ET.parse(source_mjcf)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str((PROJECT_DIR / "meshes").resolve()))

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("source MuJoCo model lacks asset/worldbody")
    for material in list(asset.findall("material")):
        if material.get("name", "").startswith("overfit_"):
            asset.remove(material)
    for collection, rgba in COLLECTION_RGBA.items():
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"overfit_{collection.lower()}",
                "rgba": _numbers(rgba),
                "reflectance": "0.08",
                "shininess": "0.18",
            },
        )

    for element in list(worldbody):
        if element.tag == "geom" and element.get("name") == "ground":
            worldbody.remove(element)
        elif (
            element.tag in ("geom", "site")
            and element.get("name", "").startswith("env_")
        ):
            worldbody.remove(element)

    for item in environment_data["obstacles"]:
        if not item.get("collision", False):
            continue
        pose = item["pose"]
        attributes = {
            "name": item["id"],
            "type": "box",
            "pos": _numbers(pose["position"]),
            "quat": _numbers(pose["quaternion_wxyz"]),
            "size": _numbers(item["half_extents"]),
            "material": f"overfit_{item['collection'].lower()}",
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
        }
        ET.SubElement(worldbody, "geom", attributes)

    marker_stride = max(1, len(spline_states) // 90)
    for index, state in enumerate(spline_states[::marker_stride]):
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": f"env_reference_path_{index:03d}",
                "type": "sphere",
                "pos": _numbers(state[:3]),
                "size": "0.022",
                "rgba": "0.0 0.9 1.0 0.72",
                "group": "4",
            },
        )
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "env_random_start",
            "type": "sphere",
            "pos": _numbers(spline_states[0, :3]),
            "size": "0.075",
            "rgba": "0.1 0.45 1.0 0.9",
            "group": "4",
        },
    )
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "env_random_goal",
            "type": "sphere",
            "pos": _numbers(spline_states[-1, :3]),
            "size": "0.075",
            "rgba": "0.1 1.0 0.25 0.9",
            "group": "4",
        },
    )
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(output_path))
    return output_path


def create_demo_arguments(
    args: argparse.Namespace,
    combined_mjcf: Path,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(combined_mjcf),
        controller="mppi",
        samples=args.mppi_samples,
        horizon=args.mppi_horizon,
        control_dt=args.control_dt,
        terminal_multiplier=2.5,
        mppi_obstacle_margin=0.0,
        obstacle_penalty=0.0,
        temperature=args.mppi_temperature,
        action_continuity_weight=1.0,
        control_smoothing=0.12,
        iterations=1,
        seed=args.seed + 101,
        visualized_samples=20,
        attitude_lookahead_steps=2,
        attitude_feedback_source="reference",
        position_feedback_source="reference",
        duration=None,
        goal_hold=args.goal_hold,
        headless=not args.viewer,
        realtime=args.viewer,
        rerun=not args.no_rerun,
        rerun_path=(
            output_dir / "mujoco_mppi_tracking.rrd"
            if not args.no_rerun
            else None
        ),
        rerun_viewer=False,
        rerun_viewer_port=9876,
        rerun_samples=8,
        rerun_trace_stride=2,
        output_dir=output_dir,
    )


def save_geometric_and_timed_paths(
    output_dir: Path,
    planner: OMPLSE3Planner,
    multi_plan: Any,
    reference: ToppraTimedReference,
) -> tuple[Path, Path]:
    raw_path = output_dir / "ompl_simplified_dense_path.csv"
    raw_clearance = planner.clearance(
        multi_plan.raw_states[:, :3], multi_plan.raw_states[:, 3:7]
    )
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("index", "x", "y", "z", "qw", "qx", "qy", "qz", "clearance_m")
        )
        for index, (state, clearance) in enumerate(
            zip(multi_plan.raw_states, raw_clearance)
        ):
            writer.writerow((index, *state, clearance))

    trajectory_path = output_dir / "toppra_reference_trajectory.csv"
    times = np.linspace(
        0.0,
        reference.finish_time,
        max(1001, int(math.ceil(reference.finish_time / 0.01)) + 1),
    )
    samples = reference.sample_full(times)
    clearance = planner.clearance(
        samples.position, samples.quaternion_wxyz
    )
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_s",
                "path_parameter",
                "path_speed",
                "path_acceleration",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "ax",
                "ay",
                "az",
                "qw",
                "qx",
                "qy",
                "qz",
                "omega_x",
                "omega_y",
                "omega_z",
                "alpha_x",
                "alpha_y",
                "alpha_z",
                "coal_clearance_m",
            )
        )
        for index in range(len(times)):
            writer.writerow(
                (
                    times[index],
                    samples.path_position[index],
                    samples.path_speed[index],
                    samples.path_acceleration[index],
                    *samples.position[index],
                    *samples.linear_velocity_world[index],
                    *samples.linear_acceleration_world[index],
                    *samples.quaternion_wxyz[index],
                    *samples.angular_velocity_body[index],
                    *samples.angular_acceleration_body[index],
                    clearance[index],
                )
            )
    return raw_path, trajectory_path


def save_collision_audit(
    output_dir: Path,
    run: Any,
    buffered_checker: CoalCollisionChecker,
    environment: tuple[StaticCollisionObject, ...],
) -> tuple[Path, dict[str, Any]]:
    """Separate physical collision from violation of the planning buffer."""

    physical_checker = CoalCollisionChecker(
        buffered_checker.vehicle,
        environment,
        safety_margin=0.0,
    )
    positions = np.asarray(run.log.positions, dtype=np.float64)
    quaternions = np.asarray(run.log.quaternions, dtype=np.float64)
    physical_clearance = physical_checker.clearance(positions, quaternions)
    buffered_clearance = buffered_checker.clearance(positions, quaternions)
    path_states = run.problem.path.states
    planned_physical_clearance = physical_checker.clearance(
        path_states[:, :3], path_states[:, 3:7]
    )
    planned_buffered_clearance = buffered_checker.clearance(
        path_states[:, :3], path_states[:, 3:7]
    )

    audit_path = output_dir / "coal_actual_trajectory_clearance.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_s",
                "physical_signed_clearance_m",
                "planning_buffer_signed_clearance_m",
                "physical_collision_free",
                "planning_buffer_respected",
            )
        )
        for time_s, physical, buffered in zip(
            run.log.times, physical_clearance, buffered_clearance
        ):
            writer.writerow(
                (
                    time_s,
                    physical,
                    buffered,
                    int(physical > 0.0),
                    int(buffered > 0.0),
                )
            )

    audit = {
        "sample_count": len(physical_clearance),
        "sample_period_s": (
            float(np.median(np.diff(run.log.times)))
            if len(run.log.times) > 1
            else None
        ),
        "physical_collision_free_actual_trajectory": bool(
            np.all(physical_clearance > 0.0)
        ),
        "minimum_actual_physical_clearance_m": float(
            np.min(physical_clearance)
        ),
        "planning_buffer_respected_actual_trajectory": bool(
            np.all(buffered_clearance > 0.0)
        ),
        "minimum_actual_planning_buffer_clearance_m": float(
            np.min(buffered_clearance)
        ),
        "physical_collision_free_planned_path": bool(
            np.all(planned_physical_clearance > 0.0)
        ),
        "minimum_planned_physical_clearance_m": float(
            np.min(planned_physical_clearance)
        ),
        "planning_buffer_respected_planned_path": bool(
            np.all(planned_buffered_clearance > 0.0)
        ),
        "minimum_planned_buffer_clearance_m": float(
            np.min(planned_buffered_clearance)
        ),
    }
    return audit_path, audit


def save_training_sample_npz(
    output_dir: Path,
    run: Any,
    reference: ToppraTimedReference,
    multi_plan: Any,
    buffered_checker: CoalCollisionChecker,
    environment: tuple[StaticCollisionObject, ...],
) -> Path:
    """Save lossless numeric arrays required for downstream model training."""

    physical_checker = CoalCollisionChecker(
        buffered_checker.vehicle,
        environment,
        safety_margin=0.0,
    )
    actual_state = np.asarray(run.log.states, dtype=np.float64)
    reference_state = np.asarray(
        run.log.reference_states, dtype=np.float64
    )
    actions = np.concatenate(
        (
            np.asarray(run.log.linear_actions, dtype=np.float64),
            np.asarray(run.log.angular_actions, dtype=np.float64),
        ),
        axis=1,
    )
    actual_physical_clearance = physical_checker.clearance(
        actual_state[:, :3], actual_state[:, 6:10]
    )
    actual_buffer_clearance = buffered_checker.clearance(
        actual_state[:, :3], actual_state[:, 6:10]
    )
    reference_times = np.linspace(
        0.0,
        reference.finish_time,
        max(1001, int(math.ceil(reference.finish_time / 0.01)) + 1),
    )
    timed = reference.sample_full(reference_times)
    timed_physical_clearance = physical_checker.clearance(
        timed.position, timed.quaternion_wxyz
    )
    timed_buffer_clearance = buffered_checker.clearance(
        timed.position, timed.quaternion_wxyz
    )
    output_path = output_dir / "training_sample.npz"
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([1], dtype=np.int64),
        start_pose=np.concatenate(
            (run.problem.start.position, run.problem.start.quaternion)
        ),
        goal_pose=np.concatenate(
            (run.problem.goal.position, run.problem.goal.quaternion)
        ),
        waypoint_states=np.stack(
            [
                np.concatenate((pose.position, pose.quaternion))
                for pose in multi_plan.waypoints
            ]
        ),
        # ``raw_states`` is the simplified, resolution-densified OMPL guide
        # retained by MultiWaypointPlan.  Keep the old key for compatibility
        # and expose an unambiguous name for dataset consumers.
        raw_ompl_states=multi_plan.raw_states,
        simplified_ompl_states=multi_plan.raw_states,
        simplified_ompl_parameters=multi_plan.spline.parameters,
        smoothed_path_states=multi_plan.spline_path.states,
        smoothed_path_parameters=np.linspace(
            0.0, 1.0, len(multi_plan.spline_path.states)
        ),
        smoothed_path_buffer_clearance=buffered_checker.clearance(
            multi_plan.spline_path.states[:, :3],
            multi_plan.spline_path.states[:, 3:7],
        ),
        bspline_degree=np.asarray(
            [multi_plan.spline.degree], dtype=np.int64
        ),
        bspline_knots=multi_plan.spline.knots,
        bspline_position_control_points=(
            multi_plan.spline.position_control_points
        ),
        bspline_quaternion_control_points=(
            multi_plan.spline.quaternion_control_points
        ),
        waypoint_parameters=multi_plan.waypoint_parameters,
        waypoint_path_indices=np.asarray(
            multi_plan.waypoint_path_indices, dtype=np.int64
        ),
        toppra_time=reference_times,
        toppra_path_position=timed.path_position,
        toppra_path_speed=timed.path_speed,
        toppra_path_acceleration=timed.path_acceleration,
        toppra_reference_state=timed.reference,
        toppra_linear_acceleration_world=(
            timed.linear_acceleration_world
        ),
        toppra_angular_acceleration_body=(
            timed.angular_acceleration_body
        ),
        toppra_physical_clearance=timed_physical_clearance,
        toppra_buffer_clearance=timed_buffer_clearance,
        control_time=np.asarray(run.log.times, dtype=np.float64),
        actual_state=actual_state,
        reference_state=reference_state,
        mppi_action=actions,
        effective_sample_size=np.asarray(
            run.log.effective_sample_sizes, dtype=np.float64
        ),
        mppi_update_time_ms=np.asarray(
            run.log.update_times_ms, dtype=np.float64
        ),
        actual_physical_clearance=actual_physical_clearance,
        actual_buffer_clearance=actual_buffer_clearance,
        mujoco_qpos=np.asarray(run.log.mujoco_qpos, dtype=np.float64),
        mujoco_qvel=np.asarray(run.log.mujoco_qvel, dtype=np.float64),
        mujoco_ctrl=np.asarray(run.log.mujoco_ctrl, dtype=np.float64),
        mujoco_actuator_force=np.asarray(
            run.log.mujoco_actuator_force, dtype=np.float64
        ),
    )
    return output_path


def _interpolated_quaternion(
    times: np.ndarray,
    quaternions: np.ndarray,
    query: float,
) -> np.ndarray:
    upper = int(np.searchsorted(times, query, side="right"))
    upper = min(max(upper, 1), len(times) - 1)
    lower = upper - 1
    span = float(times[upper] - times[lower])
    blend = 0.0 if span <= 0.0 else (query - times[lower]) / span
    first = quaternions[lower]
    second = quaternions[upper].copy()
    if np.dot(first, second) < 0.0:
        second *= -1.0
    return normalize_quaternion((1.0 - blend) * first + blend * second)


def render_tracking_gif(
    model_path: Path,
    log: Any,
    output_path: Path,
    *,
    fps: int,
    width: int = 560,
    height: int = 420,
) -> Path:
    """Replay the MuJoCo-simulated base trajectory in the full box map."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    free_joints = np.flatnonzero(
        model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if len(free_joints) != 1:
        raise RuntimeError("combined model must contain one free joint")
    qpos_address = int(model.jnt_qposadr[int(free_joints[0])])
    times = np.asarray(log.times, dtype=np.float64)
    positions = np.asarray(log.positions, dtype=np.float64)
    quaternions = np.asarray(log.quaternions, dtype=np.float64)
    references = np.asarray(log.reference_positions, dtype=np.float64)
    frame_times = np.arange(0.0, times[-1] + 0.5 / fps, 1.0 / fps)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = (0.0, 0.0, 1.55)
    camera.distance = 8.4
    camera.azimuth = 137.0
    camera.elevation = -23.0
    scene_option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(scene_option)
    scene_option.geomgroup[:] = 1
    scene_option.sitegroup[:] = 1
    frames: list[Image.Image] = []
    try:
        for frame_time in frame_times:
            position = np.asarray(
                [
                    np.interp(frame_time, times, positions[:, axis])
                    for axis in range(3)
                ]
            )
            quaternion = _interpolated_quaternion(
                times, quaternions, frame_time
            )
            reference_position = np.asarray(
                [
                    np.interp(frame_time, times, references[:, axis])
                    for axis in range(3)
                ]
            )
            data.qpos[qpos_address : qpos_address + 3] = position
            data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera=camera, scene_option=scene_option
            )
            image = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(image)
            error = float(np.linalg.norm(position - reference_position))
            draw.rectangle((8, 8, 282, 51), fill=(0, 0, 0))
            draw.text(
                (15, 14),
                "MuJoCo + MPPI tracking",
                fill=(235, 245, 255),
            )
            draw.text(
                (15, 31),
                f"t={frame_time:5.2f}s   position error={error:.3f}m",
                fill=(120, 235, 255),
            )
            frames.append(image)
    finally:
        renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )
    return output_path


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one random OMPL/COAL/TOPP-RA/MPPI/MuJoCo trial"
    )
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20270802)
    parser.add_argument(
        "--external-path",
        type=Path,
        help=(
            "NPZ diffusion guide containing poses_wxyz/states; bypasses OMPL "
            "generation but still runs B-spline, COAL, TOPP-RA and MPPI"
        ),
    )
    parser.add_argument(
        "--start-pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--goal-pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    )
    parser.add_argument(
        "--via-pose",
        type=float,
        nargs=7,
        action="append",
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="optional fixed intermediate pose; may be repeated",
    )
    parser.add_argument(
        "--segment-position-bounds",
        type=float,
        nargs=6,
        action="append",
        metavar=("MIN_X", "MIN_Y", "MIN_Z", "MAX_X", "MAX_Y", "MAX_Z"),
        help=(
            "optional position bounds for one consecutive waypoint segment; "
            "repeat exactly via-pose-count + 1 times"
        ),
    )
    parser.add_argument(
        "--sampling-direction",
        choices=("random", "south-to-north", "north-to-south"),
        default="random",
    )
    parser.add_argument("--south-region-min", type=float, nargs=3)
    parser.add_argument("--south-region-max", type=float, nargs=3)
    parser.add_argument("--north-region-min", type=float, nargs=3)
    parser.add_argument("--north-region-max", type=float, nargs=3)
    parser.add_argument("--coal-safety-margin", type=float, default=0.080)
    parser.add_argument("--endpoint-clearance", type=float, default=0.080)
    parser.add_argument("--path-clearance", type=float, default=0.0)
    parser.add_argument("--max-tilt-deg", type=float, default=42.0)
    parser.add_argument("--endpoint-z-max", type=float, default=2.35)
    parser.add_argument("--minimum-pair-distance", type=float, default=3.0)
    parser.add_argument("--minimum-y-separation", type=float, default=2.35)
    parser.add_argument("--minimum-detour-ratio", type=float, default=1.0)
    parser.add_argument(
        "--minimum-attitude-separation-deg", type=float, default=15.0
    )
    parser.add_argument(
        "--maximum-path-rotation-deg",
        type=float,
        default=0.0,
        help="optional hard rotation-length limit; 0 disables it",
    )
    parser.add_argument(
        "--maximum-rotation-stretch",
        type=float,
        default=0.0,
        help="optional rotation/direct-geodesic ratio limit; 0 disables it",
    )
    parser.add_argument("--maximum-pose-attempts", type=int, default=3000)
    parser.add_argument("--maximum-pair-attempts", type=int, default=12)
    parser.add_argument("--solve-time", type=float, default=8.0)
    parser.add_argument("--planner-range", type=float, default=0.34)
    parser.add_argument("--validity-resolution", type=float, default=0.0025)
    parser.add_argument("--path-resolution", type=float, default=0.035)
    parser.add_argument("--minimum-states", type=int, default=140)
    parser.add_argument("--spline-knot-stride", type=int, default=6)
    parser.add_argument("--spline-samples", type=int, default=2200)
    parser.add_argument(
        "--spline-method",
        choices=("constrained-smoothing", "interpolating"),
        default="constrained-smoothing",
    )
    parser.add_argument(
        "--baseline-interpolating-stride", type=int, default=3
    )
    parser.add_argument("--smoothing-degree", type=int, default=5)
    parser.add_argument(
        "--smoothing-guide-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--smoothing-position-acceleration-weight",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--smoothing-position-jerk-weight",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument(
        "--smoothing-orientation-acceleration-weight",
        type=float,
        default=2.5e-9,
    )
    parser.add_argument(
        "--smoothing-orientation-jerk-weight",
        type=float,
        default=2.5e-13,
    )
    parser.add_argument(
        "--smoothing-clearance-weight-scale",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--smoothing-max-attempts", type=int, default=4
    )
    parser.add_argument("--no-orientation-shortcut", action="store_true")
    parser.add_argument("--orientation-metric-weight", type=float, default=0.35)
    parser.add_argument("--max-linear-speed", type=float, default=0.42)
    parser.add_argument("--max-angular-speed", type=float, default=0.65)
    parser.add_argument(
        "--max-linear-acceleration",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 0.8),
    )
    parser.add_argument(
        "--max-angular-acceleration",
        type=float,
        nargs=3,
        default=(1.8, 1.8, 1.5),
    )
    parser.add_argument("--velocity-scale", type=float, default=0.72)
    parser.add_argument("--acceleration-scale", type=float, default=0.65)
    parser.add_argument("--duration-scale", type=float, default=1.25)
    parser.add_argument("--start-delay", type=float, default=0.7)
    parser.add_argument("--goal-hold", type=float, default=3.0)
    parser.add_argument("--toppra-gridpoints", type=int, default=401)
    parser.add_argument("--toppra-validation-points", type=int, default=4001)
    parser.add_argument("--mppi-samples", type=int, default=512)
    parser.add_argument("--mppi-horizon", type=int, default=48)
    parser.add_argument("--mppi-temperature", type=float, default=260.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if (args.start_pose is None) != (args.goal_pose is None):
        parser.error("--start-pose and --goal-pose must be provided together")
    if args.external_path is not None and args.start_pose is not None:
        parser.error("--external-path cannot be combined with fixed poses")
    region_values = (
        args.south_region_min,
        args.south_region_max,
        args.north_region_min,
        args.north_region_max,
    )
    if any(value is not None for value in region_values):
        if not all(value is not None for value in region_values):
            parser.error("all four south/north region bounds are required")
        if not (
            np.all(np.asarray(args.south_region_min) < np.asarray(args.south_region_max))
            and np.all(np.asarray(args.north_region_min) < np.asarray(args.north_region_max))
        ):
            parser.error("each task-region minimum must be below its maximum")
    positive = (
        args.coal_safety_margin >= 0.0
        and args.endpoint_clearance >= 0.0
        and args.path_clearance >= 0.0
        and 0.0 < args.max_tilt_deg < 90.0
        and args.endpoint_z_max > 0.0
        and args.minimum_pair_distance > 0.0
        and args.minimum_y_separation >= 0.0
        and args.minimum_detour_ratio >= 1.0
        and args.minimum_attitude_separation_deg >= 0.0
        and args.maximum_path_rotation_deg >= 0.0
        and args.maximum_rotation_stretch >= 0.0
        and (
            args.maximum_rotation_stretch == 0.0
            or args.maximum_rotation_stretch >= 1.0
        )
        and args.solve_time > 0.0
        and args.planner_range > 0.0
        and 0.0 < args.validity_resolution <= 1.0
        and args.path_resolution > 0.0
        and args.minimum_states >= 4
        and args.spline_knot_stride >= 1
        and args.spline_samples >= 20
        and args.baseline_interpolating_stride >= 1
        and args.smoothing_degree >= 3
        and args.smoothing_guide_weight > 0.0
        and args.smoothing_position_acceleration_weight >= 0.0
        and args.smoothing_position_jerk_weight >= 0.0
        and args.smoothing_orientation_acceleration_weight >= 0.0
        and args.smoothing_orientation_jerk_weight >= 0.0
        and args.smoothing_clearance_weight_scale >= 0.0
        and args.smoothing_max_attempts >= 1
        and args.max_linear_speed > 0.0
        and args.max_angular_speed > 0.0
        and args.duration_scale >= 1.0
        and args.start_delay >= 0.0
        and args.goal_hold >= 0.0
        and args.mppi_samples >= 2
        and args.mppi_horizon >= 1
        and args.control_dt > 0.0
        and args.gif_fps >= 1
    )
    if not positive:
        parser.error("invalid planning, retiming or tracking parameter")
    expected_segment_count = len(args.via_pose or ()) + 1
    if args.segment_position_bounds and (
        len(args.segment_position_bounds) != expected_segment_count
    ):
        parser.error(
            "--segment-position-bounds must be repeated exactly once per "
            "consecutive waypoint segment"
        )
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment_data, environment = load_environment(
        args.environment.expanduser().resolve()
    )
    collision_checker = CoalCollisionChecker.from_urdf(
        args.urdf.expanduser().resolve(),
        environment,
        safety_margin=args.coal_safety_margin,
    )
    sampling = environment_data["sampling_space"]["position_bounds"]
    bounds_min = _float_vector(sampling["min"], 3, "sampling min")
    bounds_max = _float_vector(sampling["max"], 3, "sampling max")
    planner = OMPLSE3Planner(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        obstacles=(),
        vehicle_radius=0.0,
        safety_margin=0.0,
        validity_resolution=args.validity_resolution,
        planner_range=args.planner_range,
        seed=args.seed,
        collision_checker=collision_checker,
    )
    rng = np.random.default_rng(args.seed)
    if args.external_path is not None:
        start, goal, multi_plan, sampling_diagnostics = (
            plan_external_diffusion_path(
                planner, args.external_path.expanduser().resolve(), args
            )
        )
    elif args.start_pose is None:
        start, goal, multi_plan, sampling_diagnostics = sample_and_plan(
            rng, planner, bounds_min, bounds_max, args
        )
    else:
        start, goal, multi_plan, sampling_diagnostics = (
            plan_fixed_pose_pair(planner, args)
        )
    reference = ToppraTimedReference(
        multi_plan,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        max_linear_acceleration=args.max_linear_acceleration,
        max_angular_acceleration=args.max_angular_acceleration,
        start_delay=args.start_delay,
        duration_scale=args.duration_scale,
        gridpoint_count=args.toppra_gridpoints,
        validation_point_count=args.toppra_validation_points,
        max_refinement_iterations=3,
        velocity_scale=args.velocity_scale,
        acceleration_scale=args.acceleration_scale,
    )
    problem = MultiWaypointProblem(
        planner=planner,
        path=multi_plan.spline_path,
        reference=reference,
        start=start,
        goal=goal,
        intermediate_waypoints=multi_plan.intermediate_waypoints,
        intermediate_waypoint_states=np.asarray(
            [
                np.concatenate((pose.position, pose.quaternion))
                for pose in multi_plan.intermediate_waypoints
            ],
            dtype=np.float64,
        ).reshape((-1, 7)),
        raw_path_states=multi_plan.raw_states,
        multi_plan=multi_plan,
    )
    combined_mjcf = build_combined_mjcf(
        args.model.expanduser().resolve(),
        environment_data,
        multi_plan.spline_path.states,
        output_dir / "hnuter206_overfit_cube_single_run.xml",
    )
    raw_path, timed_path = save_geometric_and_timed_paths(
        output_dir, planner, multi_plan, reference
    )
    raw_path_clearance = planner.clearance(
        multi_plan.raw_states[:, :3], multi_plan.raw_states[:, 3:7]
    )
    spline_path_clearance = planner.clearance(
        multi_plan.spline_path.states[:, :3],
        multi_plan.spline_path.states[:, 3:7],
    )
    interpolating_spline, interpolating_states = (
        build_interpolating_baseline(
            multi_plan.raw_states,
            knot_stride=args.baseline_interpolating_stride,
            sample_count=args.spline_samples,
            orientation_metric_weight=args.orientation_metric_weight,
        )
    )
    interpolating_clearance = planner.clearance(
        interpolating_states[:, :3], interpolating_states[:, 3:7]
    )
    interpolating_path = save_comparison_path(
        output_dir / "interpolating_b_spline_baseline.csv",
        interpolating_states,
        interpolating_clearance,
    )
    path_comparison = {
        "raw_simplified_ompl": translation_path_metrics(
            multi_plan.raw_states, raw_path_clearance
        ),
        "cubic_interpolating_b_spline_baseline": {
            **translation_path_metrics(
                interpolating_states, interpolating_clearance
            ),
            "degree": interpolating_spline.degree,
            "method": interpolating_spline.method_name,
            "control_point_count": (
                interpolating_spline.control_point_count
            ),
            "collision_free_with_planning_buffer": bool(
                np.all(interpolating_clearance > 0.0)
            ),
        },
        "smoothed_b_spline": {
            **translation_path_metrics(
                multi_plan.spline_path.states, spline_path_clearance
            ),
            "degree": multi_plan.spline.degree,
            "method": multi_plan.spline_method,
            "control_point_count": multi_plan.control_point_count,
            "maximum_curvature_per_m": (
                multi_plan.maximum_curvature_per_m
            ),
        },
        "soft_guide_fit": {
            "position_rms_deviation_m": multi_plan.guide_position_rms_m,
            "attitude_rms_deviation_deg": math.degrees(
                multi_plan.guide_attitude_rms_rad
            ),
        },
    }
    path_comparison_path = output_dir / "path_comparison.json"
    path_comparison_path.write_text(
        json.dumps(path_comparison, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )

    demo_args = create_demo_arguments(args, combined_mjcf, output_dir)
    demo_args.rerun_environment_boxes = rerun_environment_boxes(
        environment_data
    )
    demo_args.rerun_interpolating_baseline_path = (
        interpolating_states[:, :3]
    )
    run = run_demo(demo_args, problem)
    path_file, log_file, plot_file, metrics_file = save_results(
        run, output_dir, save_plot=not args.no_plot
    )
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    collision_audit_path, collision_audit = save_collision_audit(
        output_dir, run, collision_checker, environment
    )
    training_sample_path = save_training_sample_npz(
        output_dir,
        run,
        reference,
        multi_plan,
        collision_checker,
        environment,
    )
    # ``save_results`` historically labels safety-margin-adjusted clearance as
    # physical clearance. Preserve that value under an explicit buffer name,
    # then expose the zero-margin COAL result for collision acceptance.
    metrics["minimum_actual_planning_buffer_clearance_m"] = metrics.pop(
        "minimum_actual_obstacle_clearance_m"
    )
    metrics["minimum_planned_buffer_clearance_m"] = metrics.pop(
        "minimum_planned_obstacle_clearance_m"
    )
    metrics["planning_buffer_respected_actual_trajectory"] = metrics.pop(
        "collision_free_actual_trajectory"
    )
    metrics["minimum_actual_obstacle_clearance_m"] = collision_audit[
        "minimum_actual_physical_clearance_m"
    ]
    metrics["minimum_planned_obstacle_clearance_m"] = collision_audit[
        "minimum_planned_physical_clearance_m"
    ]
    metrics["collision_free_actual_trajectory"] = collision_audit[
        "physical_collision_free_actual_trajectory"
    ]
    metrics_file.write_text(
        json.dumps(metrics, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    gif_path: Path | None = None
    if not args.no_gif:
        gif_path = render_tracking_gif(
            combined_mjcf,
            run.log,
            output_dir / "mujoco_mppi_tracking.gif",
            fps=args.gif_fps,
        )

    start_rpy = np.degrees(quaternion_to_euler(start.quaternion)).tolist()
    goal_rpy = np.degrees(quaternion_to_euler(goal.quaternion)).tolist()
    validation = reference.validation
    pipeline_success = bool(
        multi_plan.minimum_clearance_m > 0.0
        and validation.valid
        and metrics["collision_free_actual_trajectory"]
        and metrics["final_goal_position_error_m"] <= 0.35
        and metrics["final_goal_attitude_error_deg"] <= 15.0
    )
    summary = {
        "pipeline_success": pipeline_success,
        "random_seed": args.seed,
        "environment": str(args.environment.resolve()),
        "environment_collision_box_count": len(environment),
        "coal": {
            "vehicle_geometry_count": collision_checker.vehicle_geometry_count,
            "environment_object_count": collision_checker.environment_object_count,
            "safety_margin_m": args.coal_safety_margin,
        },
        "random_start": {
            "position_m": start.position.tolist(),
            "quaternion_wxyz": start.quaternion.tolist(),
            "rpy_deg": start_rpy,
        },
        "random_goal": {
            "position_m": goal.position.tolist(),
            "quaternion_wxyz": goal.quaternion.tolist(),
            "rpy_deg": goal_rpy,
        },
        "sampling_diagnostics": sampling_diagnostics,
        "planning": {
            "planner": multi_plan.spline_path.planner_name,
            "ompl_raw_rrt_states": multi_plan.spline_path.raw_state_count,
            "segment_count": len(multi_plan.segment_paths),
            "intermediate_waypoint_count": len(
                multi_plan.intermediate_waypoints
            ),
            "simplified_dense_states": len(multi_plan.raw_states),
            "b_spline_states": len(multi_plan.spline_path.states),
            "b_spline_degree": multi_plan.spline.degree,
            "b_spline_method": multi_plan.spline_method,
            "b_spline_control_points": multi_plan.control_point_count,
            "knot_stride_used": multi_plan.knot_stride_used,
            "planning_time_s": multi_plan.spline_path.planning_time_s,
            "translation_length_m": multi_plan.spline_path.path_length_m,
            "rotation_length_deg": math.degrees(
                multi_plan.spline_path.rotation_length_rad
            ),
            "minimum_coal_clearance_m": multi_plan.minimum_clearance_m,
            "orientation_shortcut_applied": (
                multi_plan.orientation_shortcut_applied
            ),
        },
        "path_comparison": path_comparison,
        "toppra": {
            **asdict(validation),
            "start_delay_s": reference.start_delay,
            "finish_time_s": reference.finish_time,
        },
        "coal_collision_audit": collision_audit,
        "mppi_mujoco": metrics,
        "outputs": {
            "combined_mjcf": str(combined_mjcf),
            "simplified_ompl_path_csv": str(raw_path),
            "b_spline_path_csv": str(path_file),
            "interpolating_baseline_csv": str(interpolating_path),
            "toppra_trajectory_csv": str(timed_path),
            "simulation_log_csv": str(log_file),
            "tracking_plot": str(plot_file) if not args.no_plot else None,
            "metrics_json": str(metrics_file),
            "coal_clearance_audit_csv": str(collision_audit_path),
            "path_comparison_json": str(path_comparison_path),
            "training_sample_npz": str(training_sample_path),
            "rerun_recording": (
                str(run.rerun_recording_path)
                if run.rerun_recording_path is not None
                else None
            ),
            "mujoco_tracking_gif": str(gif_path) if gif_path else None,
        },
    }
    summary_path = output_dir / "single_pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    print("SINGLE_PIPELINE_SUMMARY=" + json.dumps(summary, default=_json_value))
    if not pipeline_success:
        raise RuntimeError(
            f"single pipeline completed but acceptance checks failed; see {summary_path}"
        )


if __name__ == "__main__":
    main()
