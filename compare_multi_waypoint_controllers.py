"""Controlled geometric, absolute-MPPI, and residual-MPPI ablation.

All cases reuse the exact same planned and retimed multi-waypoint trajectory.
The only changed component is the outer-loop acceleration command:

* ``geometric`` uses the TOPP-RA trajectory accelerations directly;
* ``mppi`` replaces them with the receding-horizon MPPI action.
* ``residual-mppi`` optimizes a correction around TOPP-RA feedforward.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from hnuter_mppi_pose_demo import compute_pose_metrics
from hnuter_ompl_mppi_demo import DemoRun, run_demo, save_results
from mppi.quaternion import quaternion_error_vector

if TYPE_CHECKING:
    import argparse

    from hnuter_multi_waypoint_demo import MultiWaypointProblem


@dataclass(frozen=True)
class ControllerAblationMetrics:
    controller: str
    position_rmse_m: float
    position_max_error_m: float
    attitude_rmse_deg: float
    attitude_max_error_deg: float
    final_goal_position_error_m: float
    final_goal_attitude_error_deg: float
    max_intermediate_waypoint_position_error_m: float | None
    max_intermediate_waypoint_attitude_error_deg: float | None
    linear_command_jerk_rms_mps3: float
    angular_command_jerk_rms_radps3: float
    minimum_actual_obstacle_clearance_m: float | None
    collision_free_actual_trajectory: bool
    mean_update_time_ms: float
    p95_update_time_ms: float
    mission_success: bool

    def as_dict(self) -> dict[str, str | float | bool | None]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def compute_ablation_metrics(
    run: DemoRun,
    *,
    final_position_tolerance_m: float,
    final_attitude_tolerance_deg: float,
) -> ControllerAblationMetrics:
    """Compute controller-level metrics from one closed-loop run."""

    tracking = compute_pose_metrics(run.log)
    times = np.asarray(run.log.times)
    positions = np.asarray(run.log.positions)
    quaternions = np.asarray(run.log.quaternions)
    final_position_error = float(
        np.linalg.norm(positions[-1] - run.problem.goal.position)
    )
    final_attitude_error = float(
        np.degrees(
            np.linalg.norm(
                quaternion_error_vector(
                    quaternions[-1], run.problem.goal.quaternion
                )
            )
        )
    )

    intermediate_waypoints = tuple(
        getattr(run.problem, "intermediate_waypoints", ())
    )
    arrival_times = np.asarray(
        getattr(
            run.problem.reference,
            "waypoint_arrival_times",
            (),
        ),
        dtype=np.float64,
    )
    completed_waypoints: list[tuple[float, Any]] = []
    if intermediate_waypoints and len(arrival_times) == (
        len(intermediate_waypoints) + 2
    ):
        completed_waypoints = [
            (float(arrival_time), waypoint)
            for arrival_time, waypoint in zip(
                arrival_times[1:-1], intermediate_waypoints
            )
            if arrival_time <= times[-1]
        ]
    if completed_waypoints:
        indices = [
            int(np.argmin(np.abs(times - arrival_time)))
            for arrival_time, _ in completed_waypoints
        ]
        completed_poses = [
            waypoint for _, waypoint in completed_waypoints
        ]
        waypoint_position_errors = [
            float(
                np.linalg.norm(
                    positions[index] - waypoint.position
                )
            )
            for index, waypoint in zip(indices, completed_poses)
        ]
        waypoint_attitude_errors = [
            float(
                np.degrees(
                    np.linalg.norm(
                        quaternion_error_vector(
                            quaternions[index], waypoint.quaternion
                        )
                    )
                )
            )
            for index, waypoint in zip(indices, completed_poses)
        ]
    else:
        waypoint_position_errors = []
        waypoint_attitude_errors = []

    if run.problem.planner.has_collision_constraints:
        actual_clearance = run.problem.planner.clearance(
            positions, quaternions
        )
        minimum_clearance = float(np.min(actual_clearance))
        collision_free = bool(np.all(actual_clearance > 0.0))
    else:
        minimum_clearance = None
        collision_free = True
    mission_success = bool(
        collision_free
        and final_position_error <= final_position_tolerance_m
        and final_attitude_error <= final_attitude_tolerance_deg
    )
    return ControllerAblationMetrics(
        controller=run.controller_mode,
        position_rmse_m=tracking.position_rmse_m,
        position_max_error_m=tracking.position_max_error_m,
        attitude_rmse_deg=tracking.attitude_rmse_deg,
        attitude_max_error_deg=tracking.attitude_max_error_deg,
        final_goal_position_error_m=final_position_error,
        final_goal_attitude_error_deg=final_attitude_error,
        max_intermediate_waypoint_position_error_m=(
            float(np.max(waypoint_position_errors))
            if waypoint_position_errors
            else None
        ),
        max_intermediate_waypoint_attitude_error_deg=(
            float(np.max(waypoint_attitude_errors))
            if waypoint_attitude_errors
            else None
        ),
        linear_command_jerk_rms_mps3=(
            tracking.linear_command_jerk_rms_mps3
        ),
        angular_command_jerk_rms_radps3=(
            tracking.angular_command_jerk_rms_radps3
        ),
        minimum_actual_obstacle_clearance_m=minimum_clearance,
        collision_free_actual_trajectory=collision_free,
        mean_update_time_ms=tracking.mean_update_time_ms,
        p95_update_time_ms=tracking.p95_update_time_ms,
        mission_success=mission_success,
    )


def run_controller_ablation(
    args: argparse.Namespace,
    problem: MultiWaypointProblem,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Run all controllers against one immutable planning problem."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[DemoRun, ControllerAblationMetrics]] = []
    controller_order = ("geometric", "mppi", "residual-mppi")
    for controller in controller_order:
        print(f"\n=== Ablation case: {controller} ===")
        case_args = copy.copy(args)
        case_args.controller = controller
        case_args.ablation = False
        case_args.output_dir = output_dir / controller
        # A single shared RRD path would mix the three cases. Individual runs can
        # still be recorded later with --controller and --rerun.
        case_args.rerun = False
        case_args.rerun_path = None
        case_args.rerun_viewer = False
        run = run_demo(case_args, problem)
        save_results(run, case_args.output_dir)
        metrics = compute_ablation_metrics(
            run,
            final_position_tolerance_m=(
                args.ablation_final_position_tolerance
            ),
            final_attitude_tolerance_deg=(
                args.ablation_final_attitude_tolerance
            ),
        )
        cases.append((run, metrics))

    csv_path = output_dir / "controller_ablation_summary.csv"
    json_path = output_dir / "controller_ablation_summary.json"
    figure_path = output_dir / "controller_ablation_comparison.png"
    _save_csv(cases, csv_path)
    _save_json(cases, args, json_path)
    _save_figure(cases, figure_path)
    _print_summary(cases)
    return csv_path, json_path, figure_path


def _save_csv(
    cases: list[tuple[DemoRun, ControllerAblationMetrics]],
    path: Path,
) -> None:
    rows = [metrics.as_dict() for _, metrics in cases]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lower_is_better_improvement(
    geometric: ControllerAblationMetrics,
    mppi: ControllerAblationMetrics,
) -> dict[str, float | None]:
    fields = (
        "position_rmse_m",
        "position_max_error_m",
        "attitude_rmse_deg",
        "attitude_max_error_deg",
        "final_goal_position_error_m",
        "final_goal_attitude_error_deg",
        "max_intermediate_waypoint_position_error_m",
        "max_intermediate_waypoint_attitude_error_deg",
        "linear_command_jerk_rms_mps3",
        "angular_command_jerk_rms_radps3",
    )
    improvement: dict[str, float | None] = {}
    for field in fields:
        baseline_value = getattr(geometric, field)
        candidate_value = getattr(mppi, field)
        if baseline_value is None or candidate_value is None:
            improvement[field] = None
            continue
        baseline = float(baseline_value)
        candidate = float(candidate_value)
        improvement[field] = (
            100.0 * (baseline - candidate) / baseline
            if np.isfinite(baseline)
            and np.isfinite(candidate)
            and abs(baseline) > 1.0e-12
            else None
        )
    return improvement


def _save_json(
    cases: list[tuple[DemoRun, ControllerAblationMetrics]],
    args: argparse.Namespace,
    path: Path,
) -> None:
    metrics_by_controller = {
        metrics.controller: metrics for _, metrics in cases
    }
    geometric = metrics_by_controller["geometric"]
    candidates = {
        controller: metrics_by_controller[controller]
        for controller in ("mppi", "residual-mppi")
    }
    payload = {
        "experiment": (
            "same OMPL/B-spline/TOPP-RA trajectory and same low-level "
            "geometric controller; compare direct feedforward, absolute "
            "MPPI, and feedforward-centered residual MPPI"
        ),
        "success_thresholds": {
            "final_position_error_m": (
                args.ablation_final_position_tolerance
            ),
            "final_attitude_error_deg": (
                args.ablation_final_attitude_tolerance
            ),
            "collision_free_required": True,
        },
        "cases": {
            controller: metrics.as_dict()
            for controller, metrics in metrics_by_controller.items()
        },
        "improvement_percent_vs_geometric_positive_is_better": {
            controller: _lower_is_better_improvement(
                geometric, metrics
            )
            for controller, metrics in candidates.items()
        },
        "update_time_overhead_ms_vs_geometric": {
            controller: (
                metrics.mean_update_time_ms
                - geometric.mean_update_time_ms
            )
            for controller, metrics in candidates.items()
        },
        "minimum_clearance_change_m_vs_geometric": {
            controller: (
                None
                if geometric.minimum_actual_obstacle_clearance_m
                is None
                or metrics.minimum_actual_obstacle_clearance_m is None
                else (
                    metrics.minimum_actual_obstacle_clearance_m
                    - geometric.minimum_actual_obstacle_clearance_m
                )
            )
            for controller, metrics in candidates.items()
        },
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def _save_figure(
    cases: list[tuple[DemoRun, ControllerAblationMetrics]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "geometric": "#4c78a8",
        "mppi": "#e45756",
        "residual-mppi": "#54a24b",
    }
    figure = plt.figure(figsize=(20, 9))
    grid = figure.add_gridspec(2, 4)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_position = figure.add_subplot(grid[0, 1])
    axis_attitude = figure.add_subplot(grid[0, 2])
    axis_clearance = figure.add_subplot(grid[0, 3])
    axis_position_bar = figure.add_subplot(grid[1, 1])
    axis_attitude_bar = figure.add_subplot(grid[1, 2])
    axis_update_bar = figure.add_subplot(grid[1, 3])

    problem = cases[0][0].problem
    planned = problem.path.states[:, :3]
    axis_3d.plot(
        planned[:, 0],
        planned[:, 1],
        planned[:, 2],
        "--",
        color="#2a9d8f",
        linewidth=2.5,
        label="shared reference",
    )
    for run, metrics in cases:
        times = np.asarray(run.log.times)
        positions = np.asarray(run.log.positions)
        references = np.asarray(run.log.reference_positions)
        quaternions = np.asarray(run.log.quaternions)
        reference_quaternions = np.asarray(
            run.log.reference_quaternions
        )
        position_error = np.linalg.norm(
            positions - references, axis=1
        )
        attitude_error = np.degrees(
            np.linalg.norm(
                quaternion_error_vector(
                    quaternions, reference_quaternions
                ),
                axis=1,
            )
        )
        color = colors[metrics.controller]
        axis_3d.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=color,
            linewidth=1.8,
            label=metrics.controller,
        )
        axis_position.plot(
            times,
            position_error,
            color=color,
            label=metrics.controller,
        )
        axis_attitude.plot(
            times,
            attitude_error,
            color=color,
            label=metrics.controller,
        )
        if problem.planner.has_collision_constraints:
            clearance = problem.planner.clearance(
                positions, quaternions
            )
            axis_clearance.plot(
                times,
                clearance,
                color=color,
                label=metrics.controller,
            )

    names = [metrics.controller for _, metrics in cases]
    bar_colors = [colors[name] for name in names]
    axis_position_bar.bar(
        names,
        [metrics.position_rmse_m for _, metrics in cases],
        color=bar_colors,
    )
    axis_attitude_bar.bar(
        names,
        [metrics.attitude_rmse_deg for _, metrics in cases],
        color=bar_colors,
    )
    axis_update_bar.bar(
        names,
        [metrics.mean_update_time_ms for _, metrics in cases],
        color=bar_colors,
    )
    if problem.planner.obstacles:
        azimuth = np.linspace(0.0, 2.0 * np.pi, 24)
        elevation = np.linspace(0.0, np.pi, 14)
        unit_x = np.outer(np.cos(azimuth), np.sin(elevation))
        unit_y = np.outer(np.sin(azimuth), np.sin(elevation))
        unit_z = np.outer(np.ones_like(azimuth), np.cos(elevation))
        for obstacle in problem.planner.obstacles:
            radius = (
                obstacle.radius + problem.planner.collision_padding
            )
            axis_3d.plot_surface(
                obstacle.center[0] + radius * unit_x,
                obstacle.center[1] + radius * unit_y,
                obstacle.center[2] + radius * unit_z,
                color="#f59e0b",
                alpha=0.15,
                linewidth=0.0,
            )
    axis_3d.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title="Same path, different outer loop",
    )
    axis_position.set(
        title="Position tracking error",
        xlabel="time [s]",
        ylabel="error [m]",
    )
    axis_attitude.set(
        title="SO(3) attitude tracking error",
        xlabel="time [s]",
        ylabel="error [deg]",
    )
    if problem.planner.has_collision_constraints:
        axis_clearance.axhline(
            0.0,
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label="inflated boundary",
        )
        axis_clearance.set(
            title="Inflated-obstacle clearance",
            xlabel="time [s]",
            ylabel="signed clearance [m]",
        )
        axis_clearance.legend()
    else:
        axis_clearance.text(
            0.5,
            0.5,
            "No obstacles",
            ha="center",
            va="center",
            transform=axis_clearance.transAxes,
        )
        axis_clearance.set_axis_off()
    axis_position_bar.set(
        title="Position RMSE",
        ylabel="RMSE [m]",
    )
    axis_attitude_bar.set(
        title="Attitude RMSE",
        ylabel="RMSE [deg]",
    )
    axis_update_bar.set(
        title="Outer-loop computation",
        ylabel="mean update time [ms]",
    )
    for axis in (
        axis_position_bar,
        axis_attitude_bar,
        axis_update_bar,
    ):
        axis.tick_params(axis="x", rotation=12)
    for axis in (
        axis_position,
        axis_attitude,
        axis_clearance,
        axis_position_bar,
        axis_attitude_bar,
        axis_update_bar,
    ):
        axis.grid(True, alpha=0.25)
    axis_3d.legend()
    axis_position.legend()
    axis_attitude.legend()
    figure.suptitle(
        "Multi-waypoint controller ablation: geometric, MPPI, residual MPPI"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _print_summary(
    cases: list[tuple[DemoRun, ControllerAblationMetrics]],
) -> None:
    metrics_by_controller = {
        metrics.controller: metrics for _, metrics in cases
    }
    geometric = metrics_by_controller["geometric"]
    print(
        "\ncontroller  pos RMSE[m]  att RMSE[deg]  "
        "WP max[m]  final[m]  clearance[m]  update[ms]  success"
    )
    for controller in ("geometric", "mppi", "residual-mppi"):
        metrics = metrics_by_controller[controller]
        clearance = metrics.minimum_actual_obstacle_clearance_m
        clearance_text = (
            f"{clearance:12.4f}"
            if clearance is not None
            else f"{'n/a':>12s}"
        )
        waypoint_text = (
            f"{metrics.max_intermediate_waypoint_position_error_m:9.4f}"
            if (
                metrics.max_intermediate_waypoint_position_error_m
                is not None
            )
            else f"{'n/a':>9s}"
        )
        print(
            f"{controller:14s} "
            f"{metrics.position_rmse_m:11.4f}  "
            f"{metrics.attitude_rmse_deg:13.3f}  "
            f"{waypoint_text}  "
            f"{metrics.final_goal_position_error_m:8.4f}  "
            f"{clearance_text}  "
            f"{metrics.mean_update_time_ms:10.3f}  "
            f"{str(metrics.mission_success):>7s}"
        )
    for controller in ("mppi", "residual-mppi"):
        improvement = _lower_is_better_improvement(
            geometric, metrics_by_controller[controller]
        )
        waypoint_improvement = improvement[
            "max_intermediate_waypoint_position_error_m"
        ]
        waypoint_improvement_text = (
            f"{waypoint_improvement:.1f}%"
            if waypoint_improvement is not None
            else "n/a"
        )
        print(
            f"{controller} improvement: "
            f"position RMSE={improvement['position_rmse_m']:.1f}%, "
            f"attitude RMSE={improvement['attitude_rmse_deg']:.1f}%, "
            "max waypoint position error="
            f"{waypoint_improvement_text}"
        )
