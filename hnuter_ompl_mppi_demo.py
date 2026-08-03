"""OMPL Bi-RRT global planning + 6-DoF MPPI tracking demo.

Example:

    source .venv/bin/activate
    python hnuter_ompl_mppi_demo.py

Custom initial/goal poses use metres and ZYX roll-pitch-yaw degrees:

    python hnuter_ompl_mppi_demo.py \
        --start-pos -2.4 -1.6 0.9 --start-rpy-deg 0 0 -20 \
        --goal-pos 2.4 1.6 1.7 --goal-rpy-deg 20 -15 100

For a quick non-graphical check:

    python hnuter_ompl_mppi_demo.py --headless --samples 256
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from hnuter_control import HnuterController
from hnuter_mppi_demo import PROJECT_DIR
from hnuter_mppi_pose_demo import (
    PoseDemoLog,
    PoseDemoMetrics,
    PoseTrajectoryVisualizer,
    compute_pose_metrics,
)
from mppi import (
    FullyActuatedUAVDynamics,
    MPPIConfig,
    MPPIController,
    MPPIResult,
    PoseTrackingCost,
    ResidualMPPIController,
)
from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_to_euler,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    PlannedSE3Path,
    SE3PathReference,
    SE3Pose,
    SphereObstacle,
)
from rerun_bridge import (
    Box3D,
    Pose3D,
    RerunRecorderConfig,
    RerunSimulationRecorder,
    Sphere3D,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlanningProblem:
    planner: OMPLSE3Planner
    path: PlannedSE3Path
    reference: SE3PathReference
    start: SE3Pose
    goal: SE3Pose


@dataclass(frozen=True)
class DemoRun:
    log: PoseDemoLog
    last_result: MPPIResult
    problem: PlanningProblem
    simulation_duration: float
    rerun_recording_path: Path | None = None
    controller_mode: str = "mppi"


class OMPLMPPIVisualizer(PoseTrajectoryVisualizer):
    """Draw planning data and translucent robot ghosts at mission poses."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: PlannedSE3Path,
        obstacles: tuple[SphereObstacle, ...],
        collision_padding: float,
        visualized_samples: int,
        waypoint_states: FloatArray | None = None,
        raw_path_states: FloatArray | None = None,
    ) -> None:
        super().__init__(visualized_samples)
        self.path = path
        self.obstacles = obstacles
        self.collision_padding = collision_padding
        self.waypoint_states = (
            np.empty((0, 7), dtype=np.float64)
            if waypoint_states is None
            else np.asarray(waypoint_states, dtype=np.float64)
        )
        self.raw_path_states = (
            None
            if raw_path_states is None
            else np.asarray(raw_path_states, dtype=np.float64)
        )
        if (
            self.waypoint_states.ndim != 2
            or self.waypoint_states.shape[1] != 7
        ):
            raise ValueError("waypoint_states must have shape (N, 7)")
        if self.raw_path_states is not None and (
            self.raw_path_states.ndim != 2
            or self.raw_path_states.shape[1] < 3
        ):
            raise ValueError("raw_path_states must have shape (N, >=3)")
        self._model = model
        self._canonical_ghost = self._build_canonical_robot_ghost()
        self._start_ghost = self._place_robot_ghost(
            path.states[0, :3], path.states[0, 3:7]
        )
        self._goal_ghost = self._place_robot_ghost(
            path.states[-1, :3], path.states[-1, 3:7]
        )
        self._waypoint_ghosts = tuple(
            self._place_robot_ghost(
                waypoint[:3], waypoint[3:7]
            )
            for waypoint in self.waypoint_states
        )

    def update(
        self,
        viewer_handle: Any,
        result: MPPIResult,
        reference: FloatArray,
        flight_history: FloatArray,
    ) -> None:
        super().update(viewer_handle, result, reference, flight_history)
        with viewer_handle.lock():
            scene = viewer_handle.user_scn
            self._add_polyline(
                scene,
                self.path.states[:, :3],
                width=4.0,
                color=np.array(
                    [0.05, 0.95, 0.95, 0.95], dtype=np.float32
                ),
            )
            if self.raw_path_states is not None:
                self._add_polyline(
                    scene,
                    self.raw_path_states[:, :3],
                    width=2.0,
                    color=np.array(
                        [0.65, 0.35, 1.0, 0.42],
                        dtype=np.float32,
                    ),
                )
            for obstacle in self.obstacles:
                self._add_sphere(
                    scene,
                    obstacle.center,
                    obstacle.radius,
                    np.array(
                        [0.90, 0.08, 0.05, 0.38],
                        dtype=np.float32,
                    ),
                )
            self._add_frame(
                scene,
                self.path.states[0, :3],
                self.path.states[0, 3:7],
                scale=0.30,
                alpha=1.0,
            )
            self._add_frame(
                scene,
                self.path.states[-1, :3],
                self.path.states[-1, 3:7],
                scale=0.30,
                alpha=1.0,
            )
            for waypoint in self.waypoint_states:
                self._add_frame(
                    scene,
                    waypoint[:3],
                    waypoint[3:7],
                    scale=0.24,
                    alpha=0.9,
                )
            for waypoint_ghost in self._waypoint_ghosts:
                self._add_robot_ghost(
                    scene,
                    waypoint_ghost,
                    np.array(
                        [1.00, 0.42, 0.05, 0.30],
                        dtype=np.float32,
                    ),
                )
            self._add_robot_ghost(
                scene,
                self._start_ghost,
                np.array([0.10, 0.38, 1.00, 0.35], dtype=np.float32),
            )
            self._add_robot_ghost(
                scene,
                self._goal_ghost,
                np.array([0.10, 0.95, 0.35, 0.35], dtype=np.float32),
            )

    def _build_canonical_robot_ghost(
        self,
    ) -> tuple[
        tuple[int, int, FloatArray, FloatArray, FloatArray], ...
    ]:
        """Cache visible robot geoms in one connected neutral configuration."""

        data = mujoco.MjData(self._model)
        data.qpos[:] = self._model.qpos0
        drone_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        if drone_id < 0:
            raise ValueError(
                "MuJoCo model must contain a body named 'drone'"
            )
        free_joint_id = self._find_root_free_joint(drone_id)
        qpos_address = int(self._model.jnt_qposadr[free_joint_id])
        data.qpos[qpos_address : qpos_address + 3] = 0.0
        data.qpos[qpos_address + 3 : qpos_address + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        mujoco.mj_forward(self._model, data)

        subtree_geom_ids = [
            geom_id
            for geom_id in range(self._model.ngeom)
            if self._is_body_in_subtree(
                int(self._model.geom_bodyid[geom_id]), drone_id
            )
        ]
        visual_only_bodies = {
            int(self._model.geom_bodyid[geom_id])
            for geom_id in subtree_geom_ids
            if self._geom_is_visible(geom_id)
            and self._model.geom_contype[geom_id] == 0
            and self._model.geom_conaffinity[geom_id] == 0
        }
        ghost_geometries = []
        for geom_id in subtree_geom_ids:
            body_id = int(self._model.geom_bodyid[geom_id])
            if not self._geom_is_visible(geom_id):
                continue
            if (
                body_id in visual_only_bodies
                and (
                    self._model.geom_contype[geom_id] != 0
                    or self._model.geom_conaffinity[geom_id] != 0
                )
            ):
                continue
            ghost_geometries.append(
                (
                    geom_id,
                    int(self._model.geom_type[geom_id]),
                    self._model.geom_size[geom_id].copy(),
                    data.geom_xpos[geom_id].copy(),
                    data.geom_xmat[geom_id].copy(),
                )
            )
        return tuple(ghost_geometries)

    def _place_robot_ghost(
        self, position: FloatArray, quaternion: FloatArray
    ) -> tuple[
        tuple[int, int, FloatArray, FloatArray, FloatArray], ...
    ]:
        """Rigidly place the connected canonical robot at an SE(3) pose."""

        position_array = np.asarray(position, dtype=np.float64)
        quaternion_array = np.asarray(quaternion, dtype=np.float64)
        quaternion_array = quaternion_array / np.linalg.norm(
            quaternion_array
        )
        rotation = HnuterController.quaternion_to_rotation_matrix(
            quaternion_array
        )
        return tuple(
            (
                geom_id,
                geom_type,
                size.copy(),
                position_array + rotation @ local_position,
                (
                    rotation @ local_matrix.reshape(3, 3)
                ).reshape(9),
            )
            for (
                geom_id,
                geom_type,
                size,
                local_position,
                local_matrix,
            ) in self._canonical_ghost
        )

    def _geom_is_visible(self, geom_id: int) -> bool:
        """Exclude hidden or explicitly named collision-only geometry."""

        name = (
            mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            or ""
        ).lower()
        collision_name = (
            "collision" in name
            or "collider" in name
            or name.startswith("col_")
            or name.endswith("_col")
        )
        material_id = int(self._model.geom_matid[geom_id])
        material_alpha = (
            float(self._model.mat_rgba[material_id, 3])
            if material_id >= 0
            else 1.0
        )
        effective_alpha = (
            float(self._model.geom_rgba[geom_id, 3])
            * material_alpha
        )
        return not collision_name and effective_alpha > 1.0e-4

    def _find_root_free_joint(self, drone_id: int) -> int:
        joint_start = int(self._model.body_jntadr[drone_id])
        joint_count = int(self._model.body_jntnum[drone_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            if (
                self._model.jnt_type[joint_id]
                == mujoco.mjtJoint.mjJNT_FREE
            ):
                return joint_id
        raise ValueError("drone body must have a root free joint")

    def _is_body_in_subtree(self, body_id: int, root_id: int) -> bool:
        while body_id > 0:
            if body_id == root_id:
                return True
            body_id = int(self._model.body_parentid[body_id])
        return False

    def _add_robot_ghost(
        self,
        scene: mujoco.MjvScene,
        ghost: tuple[
            tuple[int, int, FloatArray, FloatArray, FloatArray], ...
        ],
        color: NDArray[np.float32],
    ) -> None:
        for geom_id, geom_type, size, position, matrix in ghost:
            if scene.ngeom >= scene.maxgeom:
                return
            geometry = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geometry,
                geom_type,
                size,
                position,
                matrix,
                color,
            )
            source_data_id = int(self._model.geom_dataid[geom_id])
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # mjvGeom uses an encoded render-mesh id: 2 * mesh_id is
                # the visual triangle mesh and 2 * mesh_id + 1 is its
                # collision convex hull. model.geom_dataid stores only the
                # unencoded mesh id.
                geometry.dataid = 2 * source_data_id
            else:
                geometry.dataid = source_data_id
            # Keep ghosts as pure decorations. Associating them with the
            # source physics geom makes Viewer collision/convex-hull flags
            # treat a ghost as another collision object.
            geometry.objtype = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
            geometry.objid = -1
            geometry.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            geometry.transparent = 1
            geometry.emission = 0.12
            scene.ngeom += 1


def create_planning_problem(args: argparse.Namespace) -> PlanningProblem:
    """Construct start/goal poses, plan with RRTConnect and parameterize."""

    obstacles = _build_obstacles(args)
    start = SE3Pose(
        np.asarray(args.start_pos, dtype=np.float64),
        quaternion_from_euler(np.radians(args.start_rpy_deg)),
    )
    goal = SE3Pose(
        np.asarray(args.goal_pos, dtype=np.float64),
        quaternion_from_euler(np.radians(args.goal_rpy_deg)),
    )
    planner = OMPLSE3Planner(
        bounds_min=args.bounds_min,
        bounds_max=args.bounds_max,
        obstacles=obstacles,
        vehicle_radius=args.vehicle_radius,
        safety_margin=args.safety_margin,
        validity_resolution=args.validity_resolution,
        planner_range=args.planner_range,
        seed=args.seed,
    )
    path = planner.plan(
        start,
        goal,
        solve_time=args.solve_time,
        interpolation_resolution=args.path_resolution,
        minimum_waypoints=args.minimum_waypoints,
        simplify=not args.no_path_simplification,
    )
    reference = SE3PathReference(
        path,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
        start_delay=args.start_delay,
        duration_scale=args.duration_scale,
    )
    return PlanningProblem(planner, path, reference, start, goal)


def create_rerun_recorder(
    args: argparse.Namespace, problem: PlanningProblem
) -> RerunSimulationRecorder | None:
    """Create/configure the optional recorder and log its static scene."""

    if not (args.rerun or args.rerun_path or args.rerun_viewer):
        return None
    recording_path = (
        args.rerun_path
        if args.rerun_path is not None
        else args.output_dir
        / (
            "ompl_mppi_recording.rrd"
            if getattr(args, "controller", "mppi") == "mppi"
            else (
                "ompl_residual_mppi_recording.rrd"
                if getattr(args, "controller", "mppi")
                == "residual-mppi"
                else "ompl_geometric_recording.rrd"
            )
        )
    ).resolve()
    recorder = RerunSimulationRecorder(
        RerunRecorderConfig(
            application_id="hnuter_ompl_mppi",
            recording_path=recording_path,
            spawn_viewer=args.rerun_viewer,
            viewer_port=args.rerun_viewer_port,
            trace_update_stride=args.rerun_trace_stride,
            robot_mjcf_path=Path(args.model),
        )
    )
    reference_times = np.linspace(
        0.0,
        problem.reference.finish_time,
        max(
            200,
            int(
                np.ceil(
                    problem.reference.finish_time / args.control_dt
                )
            )
            + 1,
        ),
    )
    timed_reference = problem.reference.sample(reference_times)
    obstacles = tuple(
        Sphere3D(
            obstacle.center,
            obstacle.radius,
            label=f"obstacle {index}",
        )
        for index, obstacle in enumerate(problem.planner.obstacles)
    )
    intermediate_waypoints = tuple(
        getattr(problem, "intermediate_waypoints", ())
    )
    recorder.log_static_scene(
        planned_path=problem.path.states[:, :3],
        raw_ompl_path=(
            np.asarray(problem.raw_path_states)[:, :3]
            if getattr(problem, "raw_path_states", None) is not None
            else None
        ),
        interpolating_baseline_path=getattr(
            args, "rerun_interpolating_baseline_path", None
        ),
        timed_reference_path=timed_reference[:, :3],
        start_pose=Pose3D(
            problem.start.position, problem.start.quaternion
        ),
        goal_pose=Pose3D(
            problem.goal.position, problem.goal.quaternion
        ),
        waypoint_poses=tuple(
            Pose3D(waypoint.position, waypoint.quaternion)
            for waypoint in intermediate_waypoints
        ),
        obstacles=obstacles,
        environment_boxes=tuple(
            getattr(args, "rerun_environment_boxes", ())
        ),
        planned_path_label=problem.path.planner_name,
        metadata={
            "controller": getattr(args, "controller", "mppi"),
            "planner": problem.path.planner_name,
            "planning_time_ms": (
                problem.path.planning_time_s * 1.0e3
            ),
            "translation_path_length_m": (
                problem.path.path_length_m
            ),
            "rotation_path_length_deg": float(
                np.degrees(problem.path.rotation_length_rad)
            ),
            "reference_duration_s": problem.reference.duration,
            "mppi_samples": args.samples,
            "mppi_horizon": args.horizon,
            "control_dt_s": args.control_dt,
            "start_position_m": problem.start.position.tolist(),
            "start_quaternion_wxyz": (
                problem.start.quaternion.tolist()
            ),
            "goal_position_m": problem.goal.position.tolist(),
            "goal_quaternion_wxyz": problem.goal.quaternion.tolist(),
            "intermediate_waypoint_count": len(
                intermediate_waypoints
            ),
            "intermediate_waypoints": [
                {
                    "position_m": waypoint.position.tolist(),
                    "quaternion_wxyz": waypoint.quaternion.tolist(),
                }
                for waypoint in intermediate_waypoints
            ],
        },
    )
    print(f"Rerun recording: {recording_path}")
    if args.rerun_viewer:
        print(
            f"Rerun live viewer: port {args.rerun_viewer_port}; "
            "the same stream is also saved to RRD"
        )
    return recorder


def mujoco_scalar_channels(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, float]:
    """Collect joint/actuator channels without exposing them to the bridge."""

    channels: dict[str, float] = {}
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            or f"joint_{joint_id}"
        )
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        channels[f"mujoco/joints/{name}/position"] = float(
            data.qpos[qpos_address]
        )
        channels[f"mujoco/joints/{name}/velocity"] = float(
            data.qvel[dof_address]
        )
    for actuator_id in range(model.nu):
        name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            or f"actuator_{actuator_id}"
        )
        channels[f"mujoco/actuators/{name}/command"] = float(
            data.ctrl[actuator_id]
        )
        channels[f"mujoco/actuators/{name}/force"] = float(
            data.actuator_force[actuator_id]
        )
    return channels


def mujoco_joint_positions(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, float]:
    """Return named hinge/slide coordinates for robot visualization."""

    positions: dict[str, float] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        if name is None:
            continue
        positions[name] = float(
            data.qpos[int(model.jnt_qposadr[joint_id])]
        )
    return positions


def geometric_reference_result(
    reference: FloatArray,
    linear_acceleration: FloatArray,
    angular_acceleration: FloatArray,
    *,
    attitude_action_index: int = 1,
) -> MPPIResult:
    """Package direct trajectory feedforward in the existing result interface.

    This is the no-MPPI ablation baseline: the geometric controller receives
    the timed trajectory's pose, velocity, and analytic acceleration directly.
    The result wrapper keeps logging and visualization identical between the
    two controller modes.
    """

    reference_array = np.asarray(reference, dtype=np.float64)
    linear_array = np.asarray(linear_acceleration, dtype=np.float64)
    angular_array = np.asarray(angular_acceleration, dtype=np.float64)
    if (
        reference_array.ndim != 2
        or reference_array.shape[1] != 13
        or len(reference_array) < 2
    ):
        raise ValueError("reference must have shape (horizon + 1, 13)")
    expected_acceleration_shape = (len(reference_array), 3)
    if (
        linear_array.shape != expected_acceleration_shape
        or angular_array.shape != expected_acceleration_shape
    ):
        raise ValueError(
            "reference accelerations must have shape "
            f"{expected_acceleration_shape}"
        )
    attitude_index = int(
        np.clip(attitude_action_index, 1, len(reference_array) - 1)
    )
    nominal_controls = reference_feedforward_controls(
        linear_array,
        angular_array,
        attitude_action_index=attitude_index,
    )
    action = nominal_controls[0].copy()
    return MPPIResult(
        action=action,
        nominal_controls=nominal_controls,
        nominal_states=reference_array.copy(),
        sampled_controls=nominal_controls[None, :, :].copy(),
        sampled_states=reference_array[None, :, :].copy(),
        costs=np.zeros(1, dtype=np.float64),
        weights=np.ones(1, dtype=np.float64),
        best_index=0,
        effective_sample_size=float("nan"),
    )


def reference_feedforward_controls(
    linear_acceleration: FloatArray,
    angular_acceleration: FloatArray,
    *,
    attitude_action_index: int = 1,
) -> FloatArray:
    """Build horizon controls aligned with pose-feedback lookahead."""

    linear_array = np.asarray(linear_acceleration, dtype=np.float64)
    angular_array = np.asarray(angular_acceleration, dtype=np.float64)
    if (
        linear_array.ndim != 2
        or linear_array.shape[1] != 3
        or angular_array.shape != linear_array.shape
        or len(linear_array) < 2
    ):
        raise ValueError(
            "linear/angular acceleration must have shape "
            "(horizon + 1, 3)"
        )
    horizon = len(linear_array) - 1
    lookahead = int(np.clip(attitude_action_index, 1, horizon))
    angular_indices = np.minimum(
        np.arange(1, horizon + 1) + lookahead - 1,
        horizon,
    )
    return np.concatenate(
        (linear_array[1:], angular_array[angular_indices]), axis=1
    )


def reference_with_acceleration(
    reference_generator: Any,
    times: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sample a timed reference and its feedforward accelerations."""

    sample_full = getattr(reference_generator, "sample_full", None)
    if callable(sample_full):
        full = sample_full(times)
        return (
            np.asarray(full.reference, dtype=np.float64),
            np.asarray(
                full.linear_acceleration_world, dtype=np.float64
            ),
            np.asarray(
                full.angular_acceleration_body, dtype=np.float64
            ),
        )

    reference = np.asarray(
        reference_generator.sample(times), dtype=np.float64
    )
    if len(times) < 2:
        raise ValueError("at least two reference times are required")
    edge_order = 2 if len(times) >= 3 else 1
    linear_acceleration = np.gradient(
        reference[:, 3:6],
        np.asarray(times, dtype=np.float64),
        axis=0,
        edge_order=edge_order,
    )
    angular_acceleration = np.gradient(
        reference[:, 10:13],
        np.asarray(times, dtype=np.float64),
        axis=0,
        edge_order=edge_order,
    )
    return reference, linear_acceleration, angular_acceleration


def run_demo(
    args: argparse.Namespace, problem: PlanningProblem
) -> DemoRun:
    """Track the planned path on the full HNUTER MuJoCo model."""

    controller_mode = str(getattr(args, "controller", "mppi"))
    if controller_mode not in (
        "mppi",
        "residual-mppi",
        "geometric",
    ):
        raise ValueError(
            "controller must be 'mppi', 'residual-mppi', or 'geometric'"
        )
    low_level = HnuterController(Path(args.model).resolve())
    low_level.set_freejoint_pose(
        problem.start.position, problem.start.quaternion
    )
    control_steps = max(1, int(round(args.control_dt / low_level.dt)))
    control_dt = control_steps * low_level.dt

    mppi: MPPIController | ResidualMPPIController | None = None
    if controller_mode in ("mppi", "residual-mppi"):
        obstacle_tuples = tuple(
            (
                float(obstacle.center[0]),
                float(obstacle.center[1]),
                float(obstacle.center[2]),
                float(obstacle.radius),
            )
            for obstacle in problem.planner.obstacles
        )
        dynamics = FullyActuatedUAVDynamics(dt=control_dt)
        cost = PoseTrackingCost(
            terminal_multiplier=args.terminal_multiplier,
            spherical_obstacles=obstacle_tuples,
            collision_radius=(
                problem.planner.collision_padding
                + float(getattr(args, "mppi_obstacle_margin", 0.0))
            ),
            obstacle_penalty=args.obstacle_penalty,
        )
        config = MPPIConfig(
            horizon=args.horizon,
            num_samples=args.samples,
            temperature=args.temperature,
            noise_sigma=(2.3, 2.3, 2.0, 2.6, 2.6, 2.1),
            control_min=(-4.0, -4.0, -3.5, -6.0, -6.0, -5.0),
            control_max=(4.0, 4.0, 3.5, 6.0, 6.0, 5.0),
            noise_correlation=0.60,
            likelihood_ratio_weight=0.08,
            action_continuity_weight=args.action_continuity_weight,
            control_smoothing=args.control_smoothing,
            num_iterations=args.iterations,
            seed=args.seed,
        )
        if controller_mode == "residual-mppi":
            mppi = ResidualMPPIController(dynamics, cost, config)
        else:
            mppi = MPPIController(dynamics, cost, config)
    visualizer = OMPLMPPIVisualizer(
        low_level.model,
        problem.path,
        problem.planner.obstacles,
        problem.planner.collision_padding,
        args.visualized_samples,
        waypoint_states=getattr(
            problem, "intermediate_waypoint_states", None
        ),
        raw_path_states=getattr(problem, "raw_path_states", None),
    )
    log = PoseDemoLog.empty()
    rerun_recorder = create_rerun_recorder(args, problem)

    initial_state = low_level.get_mppi_pose_state()
    desired = {
        "pos": initial_state[:3],
        "vel": initial_state[3:6],
        "acc": np.zeros(3),
        "quaternion": initial_state[6:10],
        "angular_velocity": np.zeros(3),
        "angular_acceleration": np.zeros(3),
    }
    simulation_duration = (
        float(args.duration)
        if args.duration is not None
        else problem.reference.finish_time + args.goal_hold
    )
    viewer_handle = None
    if not args.headless:
        viewer_handle = mujoco.viewer.launch_passive(
            low_level.model, low_level.data
        )
        # Pose ghosts must never inherit optional collision-debug overlays.
        for visual_flag in (
            mujoco.mjtVisFlag.mjVIS_CONVEXHULL,
            mujoco.mjtVisFlag.mjVIS_BODYBVH,
            mujoco.mjtVisFlag.mjVIS_MESHBVH,
        ):
            viewer_handle.opt.flags[visual_flag] = 0
        midpoint = 0.5 * (
            problem.start.position + problem.goal.position
        )
        path_span = float(
            np.max(np.ptp(problem.path.states[:, :3], axis=0))
        )
        viewer_handle.cam.distance = max(6.0, 1.45 * path_span)
        viewer_handle.cam.azimuth = 135.0
        viewer_handle.cam.elevation = -22.0
        viewer_handle.cam.lookat[:] = midpoint

    step = 0
    last_result: MPPIResult | None = None
    flight_history: list[FloatArray] = []
    last_status_time = -1.0
    wall_start = time.perf_counter()
    print(
        f"{problem.path.planner_name}: "
        f"{problem.path.raw_state_count} raw -> "
        f"{len(problem.path.states)} dense states, "
        f"{problem.path.path_length_m:.2f} m, "
        f"{np.degrees(problem.path.rotation_length_rad):.1f} deg, "
        f"{problem.path.planning_time_s * 1.0e3:.1f} ms"
    )
    controller_description = (
        f"MPPI={args.samples} samples x {args.horizon} steps"
        if controller_mode == "mppi"
        else (
            "TOPP-RA feedforward + residual MPPI="
            f"{args.samples} samples x {args.horizon} steps"
        )
        if controller_mode == "residual-mppi"
        else "direct geometric trajectory feedforward"
    )
    print(
        f"Reference duration={problem.reference.duration:.2f}s "
        f"(finish at t={problem.reference.finish_time:.2f}s); "
        f"{controller_description}, dt={control_dt:.3f}s"
    )
    if problem.planner.has_collision_constraints:
        minimum_clearance = float(
            np.min(
                problem.planner.clearance(
                    problem.path.states[:, :3],
                    problem.path.states[:, 3:7],
                )
            )
        )
        print(
            f"Planned minimum inflated-obstacle clearance="
            f"{minimum_clearance:.3f} m"
        )
    if not args.headless:
        print(
            "Viewer: cyan=global Bi-RRT path, green=reference horizon, "
            "yellow=MPPI nominal, magenta=actual, orange=inflated "
            "obstacle; translucent blue/green robots=start/goal poses"
        )

    try:
        while low_level.data.time < simulation_duration:
            if viewer_handle is not None and not viewer_handle.is_running():
                break

            if step % control_steps == 0:
                simulation_time = float(low_level.data.time)
                state = low_level.get_mppi_pose_state()
                horizon_times = simulation_time + control_dt * np.arange(
                    args.horizon + 1
                )
                update_start = time.perf_counter()
                if controller_mode == "mppi":
                    reference = problem.reference.sample(horizon_times)
                    assert mppi is not None
                    last_result = mppi.command(state, reference)
                else:
                    (
                        reference,
                        linear_acceleration,
                        angular_acceleration,
                    ) = reference_with_acceleration(
                        problem.reference, horizon_times
                    )
                    if controller_mode == "residual-mppi":
                        assert isinstance(
                            mppi, ResidualMPPIController
                        )
                        feedforward_controls = (
                            reference_feedforward_controls(
                                linear_acceleration,
                                angular_acceleration,
                                attitude_action_index=(
                                    args.attitude_lookahead_steps
                                ),
                            )
                        )
                        last_result = mppi.command(
                            state,
                            reference,
                            feedforward_controls,
                        )
                    else:
                        last_result = geometric_reference_result(
                            reference,
                            linear_acceleration,
                            angular_acceleration,
                            attitude_action_index=(
                                args.attitude_lookahead_steps
                            ),
                        )
                update_time_ms = (
                    time.perf_counter() - update_start
                ) * 1.0e3

                position_target = (
                    reference[1]
                    if (
                        controller_mode == "geometric"
                        or args.position_feedback_source == "reference"
                    )
                    else last_result.nominal_states[1]
                )
                attitude_index = min(
                    args.attitude_lookahead_steps, args.horizon
                )
                attitude_target = (
                    reference[attitude_index]
                    if (
                        controller_mode == "geometric"
                        or args.attitude_feedback_source == "reference"
                    )
                    else last_result.nominal_states[attitude_index]
                )
                desired = {
                    "pos": position_target[:3],
                    "vel": position_target[3:6],
                    "acc": last_result.action[:3],
                    "quaternion": attitude_target[6:10],
                    "angular_velocity": attitude_target[10:13],
                    "angular_acceleration": last_result.action[3:],
                }
                flight_history.append(state[:3].copy())
                log.append(
                    simulation_time,
                    state,
                    reference[0],
                    last_result,
                    update_time_ms,
                    mujoco_qpos=low_level.data.qpos,
                    mujoco_qvel=low_level.data.qvel,
                    mujoco_ctrl=low_level.data.ctrl,
                    mujoco_actuator_force=(
                        low_level.data.actuator_force
                    ),
                )
                position_error = float(
                    np.linalg.norm(state[:3] - reference[0, :3])
                )
                attitude_error_deg = float(
                    np.degrees(
                        np.linalg.norm(
                            quaternion_error_vector(
                                state[6:10], reference[0, 6:10]
                            )
                        )
                    )
                )

                if rerun_recorder is not None:
                    sampled_positions = None
                    if (
                        controller_mode != "geometric"
                        and args.rerun_samples > 0
                    ):
                        selected = np.argsort(last_result.weights)[
                            -min(
                                args.rerun_samples,
                                len(last_result.weights),
                            ) :
                        ]
                        sampled_positions = (
                            last_result.sampled_states[
                                selected, :, :3
                            ]
                        )
                    scalar_channels = {
                        "tracking/position_error_m": position_error,
                        "tracking/attitude_error_deg": (
                            attitude_error_deg
                        ),
                        "controller/update_time_ms": update_time_ms,
                    }
                    if controller_mode != "geometric":
                        scalar_channels[
                            "mppi/effective_sample_size"
                        ] = last_result.effective_sample_size
                    if problem.planner.has_collision_constraints:
                        scalar_channels[
                            "obstacles/signed_clearance_m"
                        ] = float(
                            problem.planner.clearance(
                                state[:3], state[6:10]
                            )
                        )
                    scalar_channels.update(
                        mujoco_scalar_channels(
                            low_level.model, low_level.data
                        )
                    )
                    rerun_recorder.log_frame(
                        simulation_time,
                        actual_pose=Pose3D(
                            state[:3], state[6:10]
                        ),
                        reference_pose=Pose3D(
                            reference[0, :3], reference[0, 6:10]
                        ),
                        linear_velocity=state[3:6],
                        angular_velocity=state[10:13],
                        control=last_result.action,
                        nominal_positions=(
                            last_result.nominal_states[:, :3]
                        ),
                        sampled_positions=sampled_positions,
                        joint_positions=mujoco_joint_positions(
                            low_level.model, low_level.data
                        ),
                        scalar_channels=scalar_channels,
                    )

                if viewer_handle is not None:
                    visualizer.update(
                        viewer_handle,
                        last_result,
                        reference,
                        np.asarray(flight_history),
                    )
                if simulation_time - last_status_time >= 1.0:
                    controller_status = (
                        "ESS="
                        f"{last_result.effective_sample_size:6.1f}/"
                        f"{args.samples}"
                        if controller_mode != "geometric"
                        else "controller=geometric"
                    )
                    print(
                        f"t={simulation_time:5.2f}s  "
                        f"|e_p|={position_error:5.3f}m  "
                        f"|e_R|={attitude_error_deg:5.2f}deg  "
                        f"{controller_status}"
                    )
                    last_status_time = simulation_time

            low_level.set_desired_state(desired)
            low_level.update_control()
            mujoco.mj_step(low_level.model, low_level.data)
            step += 1
            if viewer_handle is not None and step % 10 == 0:
                viewer_handle.sync()
            if args.realtime and viewer_handle is not None and step % 5 == 0:
                remaining = (
                    wall_start
                    + float(low_level.data.time)
                    - time.perf_counter()
                )
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer_handle is not None:
            viewer_handle.close()
        if rerun_recorder is not None:
            rerun_recorder.close()

    if last_result is None:
        raise RuntimeError("simulation ended before the first MPPI update")
    return DemoRun(
        log=log,
        last_result=last_result,
        problem=problem,
        simulation_duration=simulation_duration,
        rerun_recording_path=(
            rerun_recorder.recording_path
            if rerun_recorder is not None
            else None
        ),
        controller_mode=controller_mode,
    )


def save_planned_path(
    problem: PlanningProblem, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_file = output_dir / "ompl_birrt_path.csv"
    euler_deg = np.degrees(
        quaternion_to_euler(problem.path.states[:, 3:7])
    )
    with path_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "waypoint",
                "x_m",
                "y_m",
                "z_m",
                "qw",
                "qx",
                "qy",
                "qz",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "inflated_obstacle_clearance_m",
            ]
        )
        clearance = problem.planner.clearance(
            problem.path.states[:, :3],
            problem.path.states[:, 3:7],
        )
        for index, state in enumerate(problem.path.states):
            writer.writerow(
                [index, *state, *euler_deg[index], clearance[index]]
            )
    return path_file


def save_results(
    run: DemoRun, output_dir: Path, *, save_plot: bool = True
) -> tuple[Path, Path, Path, Path]:
    """Save path, closed-loop log, plot and combined planner/tracker metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path_file = save_planned_path(run.problem, output_dir)
    result_prefix = (
        "ompl_mppi"
        if run.controller_mode == "mppi"
        else "ompl_residual_mppi"
        if run.controller_mode == "residual-mppi"
        else "ompl_geometric"
    )
    log_file = output_dir / f"{result_prefix}_log.csv"
    plot_file = output_dir / f"{result_prefix}_results.png"
    metrics_file = output_dir / f"{result_prefix}_metrics.json"

    log = run.log
    metrics = compute_pose_metrics(log)
    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    reference_positions = np.asarray(log.reference_positions)
    quaternions = np.asarray(log.quaternions)
    reference_quaternions = np.asarray(log.reference_quaternions)
    euler_deg = np.degrees(quaternion_to_euler(quaternions))
    reference_euler_deg = np.degrees(
        quaternion_to_euler(reference_quaternions)
    )
    attitude_error_deg = np.degrees(
        np.linalg.norm(
            quaternion_error_vector(
                quaternions, reference_quaternions
            ),
            axis=1,
        )
    )
    actual_clearance = run.problem.planner.clearance(
        positions, quaternions
    )
    reference_clearance = run.problem.planner.clearance(
        reference_positions, reference_quaternions
    )

    with log_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_s",
                "x_m",
                "y_m",
                "z_m",
                "x_ref_m",
                "y_ref_m",
                "z_ref_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "roll_ref_deg",
                "pitch_ref_deg",
                "yaw_ref_deg",
                "attitude_error_deg",
                "actual_clearance_m",
                "reference_clearance_m",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "alpha_x_radps2",
                "alpha_y_radps2",
                "alpha_z_radps2",
                "effective_sample_size",
                "mppi_update_time_ms",
            ]
        )
        for index in range(len(times)):
            writer.writerow(
                [
                    times[index],
                    *positions[index],
                    *reference_positions[index],
                    *euler_deg[index],
                    *reference_euler_deg[index],
                    attitude_error_deg[index],
                    actual_clearance[index],
                    reference_clearance[index],
                    *log.linear_actions[index],
                    *log.angular_actions[index],
                    log.effective_sample_sizes[index],
                    log.update_times_ms[index],
                ]
            )

    final_position_error = float(
        np.linalg.norm(positions[-1] - run.problem.goal.position)
    )
    final_attitude_error_deg = float(
        np.degrees(
            np.linalg.norm(
                quaternion_error_vector(
                    quaternions[-1], run.problem.goal.quaternion
                )
            )
        )
    )
    metrics_dict = metrics.as_dict()
    if run.controller_mode == "geometric":
        metrics_dict["mean_effective_sample_size"] = None
    metrics_dict.update(
        {
            "controller": run.controller_mode,
            "planner": run.problem.path.planner_name,
            "planning_time_ms": (
                run.problem.path.planning_time_s * 1.0e3
            ),
            "raw_planner_states": run.problem.path.raw_state_count,
            "dense_path_states": len(run.problem.path.states),
            "translation_path_length_m": (
                run.problem.path.path_length_m
            ),
            "rotation_path_length_deg": float(
                np.degrees(run.problem.path.rotation_length_rad)
            ),
            "reference_duration_s": run.problem.reference.duration,
            "final_goal_position_error_m": final_position_error,
            "final_goal_attitude_error_deg": final_attitude_error_deg,
            "minimum_actual_obstacle_clearance_m": (
                float(np.min(actual_clearance))
                if run.problem.planner.has_collision_constraints
                else None
            ),
            "minimum_planned_obstacle_clearance_m": (
                float(
                    np.min(
                        run.problem.planner.clearance(
                            run.problem.path.states[:, :3],
                            run.problem.path.states[:, 3:7],
                        )
                    )
                )
                if run.problem.planner.has_collision_constraints
                else None
            ),
            "collision_free_actual_trajectory": (
                bool(np.all(actual_clearance > 0.0))
                if run.problem.planner.has_collision_constraints
                else True
            ),
        }
    )
    intermediate_waypoints = tuple(
        getattr(run.problem, "intermediate_waypoints", ())
    )
    waypoint_arrival_times = getattr(
        run.problem.reference, "waypoint_arrival_times", None
    )
    if intermediate_waypoints and waypoint_arrival_times is not None:
        waypoint_poses = (
            run.problem.start,
            *intermediate_waypoints,
            run.problem.goal,
        )
        waypoint_indices = [
            int(np.argmin(np.abs(times - arrival_time)))
            for arrival_time in waypoint_arrival_times
        ]
        metrics_dict.update(
            {
                "intermediate_waypoint_count": len(
                    intermediate_waypoints
                ),
                "waypoint_arrival_times_s": [
                    float(value) for value in waypoint_arrival_times
                ],
                "waypoint_position_errors_m": [
                    float(
                        np.linalg.norm(
                            positions[index] - waypoint.position
                        )
                    )
                    for index, waypoint in zip(
                        waypoint_indices, waypoint_poses
                    )
                ],
                "waypoint_attitude_errors_deg": [
                    float(
                        np.degrees(
                            np.linalg.norm(
                                quaternion_error_vector(
                                    quaternions[index],
                                    waypoint.quaternion,
                                )
                            )
                        )
                    )
                    for index, waypoint in zip(
                        waypoint_indices, waypoint_poses
                    )
                ],
            }
        )
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics_dict, file, indent=2)

    if save_plot:
        _save_plot(
            run,
            metrics,
            times,
            positions,
            reference_positions,
            euler_deg,
            reference_euler_deg,
            attitude_error_deg,
            actual_clearance,
            plot_file,
        )
    return path_file, log_file, plot_file, metrics_file


def _save_plot(
    run: DemoRun,
    metrics: PoseDemoMetrics,
    times: FloatArray,
    positions: FloatArray,
    reference_positions: FloatArray,
    euler_deg: FloatArray,
    reference_euler_deg: FloatArray,
    attitude_error_deg: FloatArray,
    actual_clearance: FloatArray,
    plot_file: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 3)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_position = figure.add_subplot(grid[0, 1])
    axis_attitude = figure.add_subplot(grid[0, 2])
    axis_euler = figure.add_subplot(grid[1, 1])
    axis_clearance = figure.add_subplot(grid[1, 2])

    planned = run.problem.path.states[:, :3]
    raw_path_states = getattr(run.problem, "raw_path_states", None)
    if raw_path_states is not None:
        raw_path = np.asarray(raw_path_states)[:, :3]
        axis_3d.plot(
            raw_path[:, 0],
            raw_path[:, 1],
            raw_path[:, 2],
            color="#8e5ad7",
            alpha=0.42,
            linewidth=1.2,
            label="raw OMPL segments",
        )
    axis_3d.plot(
        planned[:, 0],
        planned[:, 1],
        planned[:, 2],
        color="#00a6a6",
        linewidth=3.0,
        label=(
            "global cubic B-spline"
            if raw_path_states is not None
            else "OMPL Bi-RRT"
        ),
    )
    axis_3d.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        reference_positions[:, 2],
        "--",
        color="#2a9d3f",
        linewidth=2.0,
        label="timed reference",
    )
    axis_3d.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#d81b60",
        linewidth=2.0,
        label="MuJoCo UAV",
    )
    _plot_obstacles(axis_3d, run.problem)
    axis_3d.scatter(
        *run.problem.start.position,
        color="#1d4ed8",
        s=50,
        label="start",
    )
    axis_3d.scatter(
        *run.problem.goal.position,
        color="#111827",
        marker="*",
        s=100,
        label="goal",
    )
    intermediate_waypoints = tuple(
        getattr(run.problem, "intermediate_waypoints", ())
    )
    if intermediate_waypoints:
        waypoint_positions = np.asarray(
            [
                waypoint.position
                for waypoint in intermediate_waypoints
            ]
        )
        axis_3d.scatter(
            waypoint_positions[:, 0],
            waypoint_positions[:, 1],
            waypoint_positions[:, 2],
            color="#ff6d00",
            marker="D",
            s=48,
            label="intermediate waypoints",
        )
    axis_3d.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title="Global planning and closed-loop tracking",
    )
    axis_3d.legend(loc="upper left")

    for index, label in enumerate(("x", "y", "z")):
        axis_position.plot(times, positions[:, index], label=label)
        axis_position.plot(
            times,
            reference_positions[:, index],
            "--",
            alpha=0.65,
        )
    axis_position.set(
        title=f"Position (RMSE={metrics.position_rmse_m:.3f} m)",
        xlabel="time [s]",
        ylabel="position [m]",
    )
    axis_position.legend(ncol=3)

    axis_attitude.plot(times, attitude_error_deg, color="#6a1b9a")
    axis_attitude.set(
        title=(
            "SO(3) attitude error "
            f"(RMSE={metrics.attitude_rmse_deg:.2f} deg)"
        ),
        xlabel="time [s]",
        ylabel="rotation error [deg]",
    )
    for index, label in enumerate(("roll", "pitch", "yaw")):
        axis_euler.plot(times, euler_deg[:, index], label=label)
        axis_euler.plot(
            times,
            reference_euler_deg[:, index],
            "--",
            alpha=0.65,
        )
    axis_euler.set(
        title="Attitude: solid=actual, dashed=reference",
        xlabel="time [s]",
        ylabel="angle [deg]",
    )
    axis_euler.legend(ncol=3)

    if run.problem.planner.has_collision_constraints:
        axis_clearance.plot(
            times, actual_clearance, color="#ef6c00"
        )
        axis_clearance.axhline(
            0.0,
            color="#c62828",
            linestyle="--",
            label="planning safety boundary",
        )
        axis_clearance.set(
            title="Signed planning-buffer clearance",
            xlabel="time [s]",
            ylabel="clearance [m]",
        )
        axis_clearance.legend()
    else:
        axis_clearance.text(
            0.5,
            0.5,
            "No planning obstacles",
            ha="center",
            va="center",
            transform=axis_clearance.transAxes,
        )
        axis_clearance.set_axis_off()
    for axis in (
        axis_position,
        axis_attitude,
        axis_euler,
        axis_clearance,
    ):
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(plot_file, dpi=180)
    plt.close(figure)


def _plot_obstacles(axis: Any, problem: PlanningProblem) -> None:
    if not problem.planner.obstacles:
        return
    azimuth = np.linspace(0.0, 2.0 * np.pi, 28)
    elevation = np.linspace(0.0, np.pi, 16)
    unit_x = np.outer(np.cos(azimuth), np.sin(elevation))
    unit_y = np.outer(np.sin(azimuth), np.sin(elevation))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(elevation))
    for obstacle in problem.planner.obstacles:
        radius = obstacle.radius
        axis.plot_surface(
            obstacle.center[0] + radius * unit_x,
            obstacle.center[1] + radius * unit_y,
            obstacle.center[2] + radius * unit_z,
            color="#ef6c00",
            alpha=0.16,
            linewidth=0.0,
        )


def _build_obstacles(
    args: argparse.Namespace,
) -> tuple[SphereObstacle, ...]:
    if args.no_obstacles:
        return ()
    definitions = args.obstacle
    if definitions is None:
        definitions = ((0.0, 0.0, 1.20, 0.50),)
    return tuple(
        SphereObstacle(np.asarray(values[:3]), float(values[3]))
        for values in definitions
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan an SE(3) pose path with OMPL bidirectional RRTConnect, "
            "then track it with 6-DoF MPPI in MuJoCo"
        )
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "hnuter206_4_5kg.xml"),
    )
    parser.add_argument(
        "--start-pos",
        type=float,
        nargs=3,
        default=(-2.4, -1.6, 0.9),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--start-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -20.0),
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs=3,
        default=(2.4, 1.6, 1.7),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--goal-rpy-deg",
        type=float,
        nargs=3,
        default=(20.0, -15.0, 100.0),
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--bounds-min",
        type=float,
        nargs=3,
        default=(-3.5, -3.0, 0.25),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--bounds-max",
        type=float,
        nargs=3,
        default=(3.5, 3.0, 3.2),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--obstacle",
        type=float,
        nargs=4,
        action="append",
        metavar=("X", "Y", "Z", "RADIUS"),
        help=(
            "spherical obstacle; repeat for multiple obstacles. If omitted, "
            "one central demo obstacle is used"
        ),
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="disable both the default and explicitly supplied obstacles",
    )
    parser.add_argument("--vehicle-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.08)
    parser.add_argument("--solve-time", type=float, default=2.0)
    parser.add_argument("--planner-range", type=float, default=0.45)
    parser.add_argument(
        "--validity-resolution", type=float, default=0.005
    )
    parser.add_argument("--path-resolution", type=float, default=0.06)
    parser.add_argument("--minimum-waypoints", type=int, default=100)
    parser.add_argument(
        "--no-path-simplification", action="store_true"
    )
    parser.add_argument("--max-linear-speed", type=float, default=0.65)
    parser.add_argument("--max-angular-speed", type=float, default=0.8)
    parser.add_argument("--duration-scale", type=float, default=1.15)
    parser.add_argument("--start-delay", type=float, default=0.5)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="simulation duration; default is path finish time + goal hold",
    )
    parser.add_argument("--goal-hold", type=float, default=3.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=250.0)
    parser.add_argument("--terminal-multiplier", type=float, default=2.0)
    parser.add_argument("--obstacle-penalty", type=float, default=3.0e4)
    parser.add_argument(
        "--attitude-lookahead-steps", type=int, default=1
    )
    parser.add_argument(
        "--attitude-feedback-source",
        choices=("reference", "nominal"),
        default="reference",
    )
    parser.add_argument(
        "--position-feedback-source",
        choices=("reference", "nominal"),
        default="reference",
    )
    parser.add_argument(
        "--action-continuity-weight", type=float, default=1.0
    )
    parser.add_argument("--control-smoothing", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualized-samples", type=int, default=28)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "record the run to output-dir/ompl_mppi_recording.rrd "
            "for Rerun playback"
        ),
    )
    parser.add_argument(
        "--rerun-path",
        type=Path,
        default=None,
        help="custom .rrd output path; also enables Rerun recording",
    )
    parser.add_argument(
        "--rerun-viewer",
        action="store_true",
        help=(
            "spawn a live Rerun viewer while also saving the .rrd file"
        ),
    )
    parser.add_argument(
        "--rerun-viewer-port", type=int, default=9876
    )
    parser.add_argument(
        "--rerun-samples",
        type=int,
        default=8,
        help="number of highest-weight MPPI samples stored per frame",
    )
    parser.add_argument(
        "--rerun-trace-stride",
        type=int,
        default=2,
        help="update actual 3D trace every N control frames",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="run OMPL and save its path without starting MuJoCo",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
    )
    parser.set_defaults(realtime=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
    )
    args = parser.parse_args()
    numeric_positive = (
        args.vehicle_radius >= 0.0
        and args.safety_margin >= 0.0
        and args.solve_time > 0.0
        and args.planner_range > 0.0
        and 0.0 < args.validity_resolution <= 1.0
        and args.path_resolution > 0.0
        and args.minimum_waypoints >= 2
        and args.max_linear_speed > 0.0
        and args.max_angular_speed > 0.0
        and args.duration_scale > 0.0
        and args.start_delay >= 0.0
        and args.goal_hold >= 0.0
        and args.control_dt > 0.0
        and args.horizon >= 1
        and args.samples >= 2
        and args.temperature > 0.0
        and args.terminal_multiplier >= 0.0
        and args.obstacle_penalty >= 0.0
        and args.attitude_lookahead_steps >= 1
        and args.action_continuity_weight >= 0.0
        and 0.0 <= args.control_smoothing < 1.0
        and args.visualized_samples >= 1
        and args.rerun_samples >= 0
        and args.rerun_trace_stride >= 1
        and 1 <= args.rerun_viewer_port <= 65535
        and (args.duration is None or args.duration > 0.0)
    )
    if not numeric_positive:
        parser.error("invalid planner, trajectory, or MPPI parameter")
    if args.no_obstacles and args.obstacle:
        parser.error("--no-obstacles cannot be combined with --obstacle")
    if (
        args.rerun_path is not None
        and args.rerun_path.suffix.lower() != ".rrd"
    ):
        parser.error("--rerun-path must use the .rrd extension")
    return args


def main() -> None:
    args = parse_args()
    problem = create_planning_problem(args)
    if args.plan_only:
        path_file = save_planned_path(
            problem, args.output_dir.resolve()
        )
        rerun_recorder = create_rerun_recorder(args, problem)
        rerun_path = None
        if rerun_recorder is not None:
            rerun_path = rerun_recorder.recording_path
            rerun_recorder.close()
        print(
            f"Planned {problem.path.path_length_m:.2f} m in "
            f"{problem.path.planning_time_s * 1.0e3:.1f} ms with "
            f"{problem.path.planner_name}"
        )
        print(f"Path: {path_file}")
        if rerun_path is not None:
            print(f"Rerun: {rerun_path}")
            print(f"Replay: rerun {rerun_path}")
        return

    run = run_demo(args, problem)
    path_file, log_file, plot_file, metrics_file = save_results(
        run, args.output_dir.resolve()
    )
    metrics = compute_pose_metrics(run.log)
    final_position_error = np.linalg.norm(
        np.asarray(run.log.positions)[-1] - problem.goal.position
    )
    final_attitude_error = np.degrees(
        np.linalg.norm(
            quaternion_error_vector(
                np.asarray(run.log.quaternions)[-1],
                problem.goal.quaternion,
            )
        )
    )
    print(
        f"Done: tracking RMSE={metrics.position_rmse_m:.3f} m / "
        f"{metrics.attitude_rmse_deg:.2f} deg; final goal error="
        f"{final_position_error:.3f} m / {final_attitude_error:.2f} deg"
    )
    print(f"Path: {path_file}")
    print(f"Log: {log_file}")
    print(f"Plot: {plot_file}")
    print(f"Metrics: {metrics_file}")
    if run.rerun_recording_path is not None:
        print(f"Rerun: {run.rerun_recording_path}")
        print(f"Replay: rerun {run.rerun_recording_path}")


if __name__ == "__main__":
    main()
