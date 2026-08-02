"""Multi-waypoint OMPL + global B-spline + 6-DoF MPPI demo.

The default mission contains four intermediate waypoints with strongly varying
roll, pitch, and yaw.  Consecutive poses are planned with OMPL RRTConnect, all
segments are stitched by one interpolating cubic SE(3) B-spline, and a
kinematic TOPP-RA timing law feeds the existing pose MPPI controller.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hnuter_mppi_demo import PROJECT_DIR
from hnuter_mppi_pose_demo import compute_pose_metrics
from hnuter_ompl_mppi_demo import (
    DemoRun,
    create_rerun_recorder,
    run_demo,
    save_results,
)
from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_to_euler,
)
from multi_waypoint_planner import (
    BSplineTimeParameterizedReference,
    MultiWaypointOMPLPlanner,
    MultiWaypointPlan,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    PlannedSE3Path,
    SE3Pose,
    SphereObstacle,
)
from toppra_retiming import ToppraTimedReference


FloatArray = np.ndarray


DEFAULT_INTERMEDIATE_WAYPOINTS = (
    (-1.5, 0.6, 1.55, 35.0, -15.0, 20.0),
    (-0.4, 1.8, 1.10, -25.0, 30.0, 90.0),
    (0.8, 0.5, 2.00, 45.0, 15.0, 160.0),
    (1.5, -1.5, 1.30, -35.0, -25.0, -110.0),
)


@dataclass(frozen=True)
class MultiWaypointProblem:
    planner: OMPLSE3Planner
    path: PlannedSE3Path
    reference: (
        ToppraTimedReference | BSplineTimeParameterizedReference
    )
    start: SE3Pose
    goal: SE3Pose
    intermediate_waypoints: tuple[SE3Pose, ...]
    intermediate_waypoint_states: FloatArray
    raw_path_states: FloatArray
    multi_plan: MultiWaypointPlan


def create_multi_waypoint_problem(
    args: argparse.Namespace,
) -> MultiWaypointProblem:
    intermediate_definitions = (
        tuple(args.waypoint)
        if args.waypoint is not None
        else DEFAULT_INTERMEDIATE_WAYPOINTS
    )
    if not 3 <= len(intermediate_definitions) <= 5:
        raise ValueError(
            "the mission must contain 3 to 5 intermediate waypoints"
        )
    start = _pose(args.start_pos, args.start_rpy_deg)
    goal = _pose(args.goal_pos, args.goal_rpy_deg)
    intermediate = tuple(
        _pose(definition[:3], definition[3:])
        for definition in intermediate_definitions
    )
    waypoints = (start, *intermediate, goal)
    obstacles = _build_obstacles(args)
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
    multi_plan = MultiWaypointOMPLPlanner(planner).plan(
        waypoints,
        solve_time_per_segment=args.solve_time_per_segment,
        interpolation_resolution=args.path_resolution,
        minimum_states_per_segment=args.segment_states,
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
    )
    if args.retimer == "toppra":
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
            max_refinement_iterations=(
                args.toppra_refinement_iterations
            ),
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
        )
    else:
        reference = BSplineTimeParameterizedReference(
            multi_plan,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            start_delay=args.start_delay,
            duration_scale=args.duration_scale,
            timing_samples=args.timing_samples,
        )
    intermediate_states = np.asarray(
        [
            np.concatenate(
                (waypoint.position, waypoint.quaternion)
            )
            for waypoint in intermediate
        ]
    )
    return MultiWaypointProblem(
        planner=planner,
        path=multi_plan.spline_path,
        reference=reference,
        start=start,
        goal=goal,
        intermediate_waypoints=intermediate,
        intermediate_waypoint_states=intermediate_states,
        raw_path_states=multi_plan.raw_states,
        multi_plan=multi_plan,
    )


def save_multi_waypoint_plan(
    problem: MultiWaypointProblem, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    waypoint_file = output_dir / "multi_waypoints.csv"
    spline_file = output_dir / "multi_waypoint_bspline_path.csv"
    all_waypoints = (
        problem.start,
        *problem.intermediate_waypoints,
        problem.goal,
    )
    waypoint_euler = np.degrees(
        quaternion_to_euler(
            np.asarray(
                [waypoint.quaternion for waypoint in all_waypoints]
            )
        )
    )
    with waypoint_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "waypoint",
                "kind",
                "x_m",
                "y_m",
                "z_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "arrival_time_s",
            ]
        )
        for index, waypoint in enumerate(all_waypoints):
            kind = (
                "start"
                if index == 0
                else "goal"
                if index == len(all_waypoints) - 1
                else "intermediate"
            )
            writer.writerow(
                [
                    index,
                    kind,
                    *waypoint.position,
                    *waypoint_euler[index],
                    problem.reference.waypoint_arrival_times[index],
                ]
            )

    spline_states = problem.path.states
    spline_euler = np.degrees(
        quaternion_to_euler(spline_states[:, 3:7])
    )
    parameters = np.linspace(0.0, 1.0, len(spline_states))
    clearance = problem.planner.clearance(
        spline_states[:, :3], spline_states[:, 3:7]
    )
    (
        _,
        position_path_derivative,
        position_path_second_derivative,
        _,
        _,
    ) = problem.multi_plan.spline.evaluate_with_second_derivatives(
        parameters
    )
    derivative_norm = np.linalg.norm(
        position_path_derivative, axis=1
    )
    curvature = np.divide(
        np.linalg.norm(
            np.cross(
                position_path_derivative,
                position_path_second_derivative,
            ),
            axis=1,
        ),
        derivative_norm**3,
        out=np.zeros_like(derivative_norm),
        where=derivative_norm > 1.0e-9,
    )
    with spline_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample",
                "spline_parameter",
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
                "path_curvature_1_per_m",
            ]
        )
        for index, state in enumerate(spline_states):
            writer.writerow(
                [
                    index,
                    parameters[index],
                    *state,
                    *spline_euler[index],
                    clearance[index],
                    curvature[index],
                ]
            )
    return waypoint_file, spline_file


def waypoint_tracking_errors(
    run: DemoRun, problem: MultiWaypointProblem
) -> tuple[FloatArray, FloatArray]:
    times = np.asarray(run.log.times)
    positions = np.asarray(run.log.positions)
    quaternions = np.asarray(run.log.quaternions)
    waypoint_poses = (
        problem.start,
        *problem.intermediate_waypoints,
        problem.goal,
    )
    indices = [
        int(np.argmin(np.abs(times - arrival_time)))
        for arrival_time in problem.reference.waypoint_arrival_times
    ]
    position_error = np.asarray(
        [
            np.linalg.norm(
                positions[index] - waypoint.position
            )
            for index, waypoint in zip(indices, waypoint_poses)
        ]
    )
    attitude_error = np.asarray(
        [
            np.degrees(
                np.linalg.norm(
                    quaternion_error_vector(
                        quaternions[index], waypoint.quaternion
                    )
                )
            )
            for index, waypoint in zip(indices, waypoint_poses)
        ]
    )
    return position_error, attitude_error


def _pose(position: object, rpy_deg: object) -> SE3Pose:
    return SE3Pose(
        np.asarray(position, dtype=np.float64),
        quaternion_from_euler(
            np.radians(np.asarray(rpy_deg, dtype=np.float64))
        ),
    )


def _build_obstacles(
    args: argparse.Namespace,
) -> tuple[SphereObstacle, ...]:
    if args.no_obstacles:
        return ()
    definitions = args.obstacle
    if definitions is None:
        definitions = (
            (0.0, 0.0, 1.25, 0.38),
            (-1.0, 1.0, 1.25, 0.28),
            (1.0, -0.55, 1.50, 0.30),
        )
    return tuple(
        SphereObstacle(
            np.asarray(definition[:3], dtype=np.float64),
            float(definition[3]),
        )
        for definition in definitions
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan through 3-5 intermediate SE(3) waypoints with OMPL, "
            "globally B-spline stitch them, and track with 6-DoF MPPI"
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
        default=(-2.6, -1.8, 1.0),
    )
    parser.add_argument(
        "--start-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -30.0),
    )
    parser.add_argument(
        "--goal-pos",
        type=float,
        nargs=3,
        default=(2.6, 1.6, 1.8),
    )
    parser.add_argument(
        "--goal-rpy-deg",
        type=float,
        nargs=3,
        default=(20.0, -20.0, 140.0),
    )
    parser.add_argument(
        "--waypoint",
        type=float,
        nargs=6,
        action="append",
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help=(
            "intermediate pose in metres/degrees; repeat 3-5 times. "
            "Omitting it uses four diverse default poses"
        ),
    )
    parser.add_argument(
        "--bounds-min",
        type=float,
        nargs=3,
        default=(-3.2, -2.5, 0.25),
    )
    parser.add_argument(
        "--bounds-max",
        type=float,
        nargs=3,
        default=(3.2, 2.5, 2.8),
    )
    parser.add_argument(
        "--obstacle",
        type=float,
        nargs=4,
        action="append",
        metavar=("X", "Y", "Z", "RADIUS"),
    )
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--vehicle-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.14)
    parser.add_argument(
        "--solve-time-per-segment", type=float, default=1.5
    )
    parser.add_argument("--planner-range", type=float, default=0.5)
    parser.add_argument(
        "--validity-resolution", type=float, default=0.004
    )
    parser.add_argument("--path-resolution", type=float, default=0.07)
    parser.add_argument("--segment-states", type=int, default=45)
    parser.add_argument("--spline-knot-stride", type=int, default=4)
    parser.add_argument("--spline-samples", type=int, default=1200)
    parser.add_argument(
        "--spline-method",
        choices=("constrained-smoothing", "interpolating"),
        default="constrained-smoothing",
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
    parser.add_argument("--timing-samples", type=int, default=4000)
    parser.add_argument(
        "--retimer",
        choices=("toppra", "minimum-jerk"),
        default="toppra",
    )
    parser.add_argument("--toppra-gridpoints", type=int, default=401)
    parser.add_argument(
        "--toppra-validation-points", type=int, default=4001
    )
    parser.add_argument(
        "--toppra-refinement-iterations", type=int, default=3
    )
    parser.add_argument(
        "--orientation-metric-weight", type=float, default=0.35
    )
    parser.add_argument("--max-linear-speed", type=float, default=1.05)
    parser.add_argument("--max-angular-speed", type=float, default=1.5)
    parser.add_argument(
        "--max-linear-acceleration",
        type=float,
        nargs=3,
        default=(4.0, 4.0, 3.5),
        metavar=("AX", "AY", "AZ"),
    )
    parser.add_argument(
        "--max-angular-acceleration",
        type=float,
        nargs=3,
        default=(6.0, 6.0, 5.0),
        metavar=("ALPHA_X", "ALPHA_Y", "ALPHA_Z"),
    )
    parser.add_argument("--velocity-scale", type=float, default=0.85)
    parser.add_argument(
        "--acceleration-scale", type=float, default=0.80
    )
    parser.add_argument("--duration-scale", type=float, default=1.08)
    parser.add_argument("--start-delay", type=float, default=0.35)
    parser.add_argument("--goal-hold", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=45)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=280.0)
    parser.add_argument("--terminal-multiplier", type=float, default=2.0)
    parser.add_argument("--obstacle-penalty", type=float, default=4.0e4)
    parser.add_argument(
        "--mppi-obstacle-margin",
        type=float,
        default=0.10,
        help=(
            "extra MPPI-only avoidance buffer beyond the OMPL inflated "
            "obstacle boundary"
        ),
    )
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
    parser.add_argument("--control-smoothing", type=float, default=0.12)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--controller",
        choices=("mppi", "residual-mppi", "geometric"),
        default="mppi",
        help=(
            "outer-loop acceleration source: absolute MPPI, residual MPPI "
            "around TOPP-RA feedforward, or direct feedforward"
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "run geometric, MPPI, and residual MPPI controllers against "
            "the exact same planned trajectory and save a comparison"
        ),
    )
    parser.add_argument(
        "--ablation-final-position-tolerance",
        type=float,
        default=0.25,
        metavar="METRES",
    )
    parser.add_argument(
        "--ablation-final-attitude-tolerance",
        type=float,
        default=10.0,
        metavar="DEGREES",
    )
    parser.add_argument("--visualized-samples", type=int, default=24)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-path", type=Path, default=None)
    parser.add_argument("--rerun-viewer", action="store_true")
    parser.add_argument(
        "--rerun-viewer-port", type=int, default=9876
    )
    parser.add_argument("--rerun-samples", type=int, default=8)
    parser.add_argument("--rerun-trace-stride", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--no-realtime", dest="realtime", action="store_false"
    )
    parser.set_defaults(realtime=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results" / "multi_waypoint",
    )
    args = parser.parse_args()
    if args.waypoint is not None and not 3 <= len(args.waypoint) <= 5:
        parser.error("--waypoint must be repeated 3 to 5 times")
    positive = (
        args.vehicle_radius >= 0.0
        and args.safety_margin >= 0.0
        and args.solve_time_per_segment > 0.0
        and args.planner_range > 0.0
        and 0.0 < args.validity_resolution <= 1.0
        and args.path_resolution > 0.0
        and args.segment_states >= 4
        and args.spline_knot_stride >= 1
        and args.spline_samples >= 20
        and args.smoothing_degree >= 3
        and args.smoothing_guide_weight > 0.0
        and args.smoothing_position_acceleration_weight >= 0.0
        and args.smoothing_position_jerk_weight >= 0.0
        and args.smoothing_orientation_acceleration_weight >= 0.0
        and args.smoothing_orientation_jerk_weight >= 0.0
        and args.smoothing_clearance_weight_scale >= 0.0
        and args.smoothing_max_attempts >= 1
        and args.timing_samples >= 100
        and args.toppra_gridpoints >= 3
        and args.toppra_validation_points >= 3
        and args.toppra_refinement_iterations >= 0
        and args.orientation_metric_weight >= 0.0
        and args.max_linear_speed > 0.0
        and args.max_angular_speed > 0.0
        and np.all(np.asarray(args.max_linear_acceleration) > 0.0)
        and np.all(np.asarray(args.max_angular_acceleration) > 0.0)
        and 0.0 < args.velocity_scale <= 1.0
        and 0.0 < args.acceleration_scale <= 1.0
        and args.duration_scale >= 1.0
        and args.start_delay >= 0.0
        and args.goal_hold >= 0.0
        and args.control_dt > 0.0
        and args.horizon >= 1
        and args.samples >= 2
        and args.temperature > 0.0
        and args.terminal_multiplier >= 0.0
        and args.obstacle_penalty >= 0.0
        and args.mppi_obstacle_margin >= 0.0
        and args.attitude_lookahead_steps >= 1
        and args.action_continuity_weight >= 0.0
        and 0.0 <= args.control_smoothing < 1.0
        and args.ablation_final_position_tolerance > 0.0
        and args.ablation_final_attitude_tolerance > 0.0
        and args.visualized_samples >= 1
        and args.rerun_samples >= 0
        and args.rerun_trace_stride >= 1
        and 1 <= args.rerun_viewer_port <= 65535
        and (args.duration is None or args.duration > 0.0)
    )
    if not positive:
        parser.error("invalid planning, B-spline, timing, or MPPI parameter")
    if args.no_obstacles and args.obstacle:
        parser.error("--no-obstacles cannot be combined with --obstacle")
    if args.ablation and args.plan_only:
        parser.error("--ablation cannot be combined with --plan-only")
    if args.ablation and (
        args.rerun or args.rerun_path is not None or args.rerun_viewer
    ):
        parser.error(
            "--ablation cannot be combined with Rerun recording; run each "
            "--controller separately when recordings are needed"
        )
    if (
        args.rerun_path is not None
        and args.rerun_path.suffix.lower() != ".rrd"
    ):
        parser.error("--rerun-path must use the .rrd extension")
    return args


def main() -> None:
    args = parse_args()
    problem = create_multi_waypoint_problem(args)
    waypoint_file, spline_file = save_multi_waypoint_plan(
        problem, args.output_dir.resolve()
    )
    print(
        f"Multi-waypoint mission: "
        f"{len(problem.intermediate_waypoints)} intermediate poses, "
        f"{len(problem.multi_plan.segment_paths)} RRTConnect segments"
    )
    print(
        f"Global degree-{problem.multi_plan.spline.degree} "
        f"{problem.multi_plan.spline_method} B-spline: "
        f"{problem.path.path_length_m:.2f} m, "
        f"{np.degrees(problem.path.rotation_length_rad):.1f} deg, "
        f"clearance={problem.multi_plan.minimum_clearance_m:.4f} m, "
        f"control points={problem.multi_plan.control_point_count}, "
        f"stride={problem.multi_plan.knot_stride_used}, "
        f"max curvature="
        f"{problem.multi_plan.maximum_curvature_per_m:.1f} 1/m"
    )
    if np.isfinite(problem.multi_plan.guide_position_rms_m):
        print(
            "Soft-guide RMS: "
            f"{problem.multi_plan.guide_position_rms_m:.3f} m, "
            f"{np.degrees(problem.multi_plan.guide_attitude_rms_rad):.2f} "
            "deg; mission waypoint poses remain hard constraints"
        )
    print(
        f"Retimer={args.retimer}; speed limits: "
        f"{args.max_linear_speed:.2f} m/s, "
        f"{args.max_angular_speed:.2f} rad/s; "
        f"reference duration={problem.reference.duration:.2f} s"
    )
    if isinstance(problem.reference, ToppraTimedReference):
        validation = problem.reference.validation
        print(
            "TOPP-RA dense validation: "
            f"{validation.gridpoint_count} solve / "
            f"{validation.validation_point_count} check points, "
            f"|v|max={validation.max_linear_speed:.3f} m/s, "
            f"|omega|max={validation.max_angular_speed:.3f} rad/s, "
            "max |a_W|="
            f"{validation.max_abs_linear_acceleration_world.round(3)}, "
            "max |alpha_B|="
            f"{validation.max_abs_angular_acceleration_body.round(3)}"
        )
    print(
        "Waypoint arrivals [s]: "
        + ", ".join(
            f"{value:.2f}"
            for value in problem.reference.waypoint_arrival_times
        )
    )
    if args.plan_only:
        recorder = create_rerun_recorder(args, problem)
        if recorder is not None:
            rerun_path = recorder.recording_path
            recorder.close()
            print(f"Rerun: {rerun_path}")
        print(f"Waypoints: {waypoint_file}")
        print(f"B-spline: {spline_file}")
        return

    if args.ablation:
        from compare_multi_waypoint_controllers import (
            run_controller_ablation,
        )

        csv_file, json_file, figure_file = run_controller_ablation(
            args, problem, args.output_dir.resolve() / "ablation"
        )
        print(f"Ablation CSV: {csv_file}")
        print(f"Ablation JSON: {json_file}")
        print(f"Ablation plot: {figure_file}")
        return

    run = run_demo(args, problem)
    _, log_file, plot_file, metrics_file = save_results(
        run, args.output_dir.resolve()
    )
    metrics = compute_pose_metrics(run.log)
    waypoint_position_error, waypoint_attitude_error = (
        waypoint_tracking_errors(run, problem)
    )
    print(
        f"Done ({args.controller}): "
        f"RMSE={metrics.position_rmse_m:.3f} m / "
        f"{metrics.attitude_rmse_deg:.2f} deg; "
        f"max intermediate waypoint error="
        f"{float(np.max(waypoint_position_error[1:-1])):.3f} m / "
        f"{float(np.max(waypoint_attitude_error[1:-1])):.2f} deg"
    )
    print(f"Waypoints: {waypoint_file}")
    print(f"B-spline: {spline_file}")
    print(f"Log: {log_file}")
    print(f"Plot: {plot_file}")
    print(f"Metrics: {metrics_file}")
    if run.rerun_recording_path is not None:
        print(f"Rerun: {run.rerun_recording_path}")


if __name__ == "__main__":
    main()
