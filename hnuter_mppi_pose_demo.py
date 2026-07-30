"""6-DoF MPPI demo with simultaneous position and attitude tracking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from hnuter_control import HnuterController
from hnuter_mppi_demo import (
    FigureEightReference,
    MujocoTrajectoryVisualizer,
    PROJECT_DIR,
)
from mppi import (
    FullyActuatedUAVDynamics,
    MPPIConfig,
    MPPIController,
    MPPIResult,
    PoseTrackingCost,
)
from mppi.quaternion import (
    body_rates_from_euler_rates,
    quaternion_error_vector,
    quaternion_from_euler,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PoseReference:
    """Figure-eight translation plus independent smooth attitude motion."""

    cruise_altitude: float = 1.0
    hold_duration: float = 1.0
    ramp_duration: float = 2.0
    position_period: float = 12.0
    attitude_period: float = 8.0
    roll_amplitude: float = math.radians(25.0)
    pitch_amplitude: float = math.radians(20.0)
    yaw_amplitude: float = math.radians(45.0)

    def sample(self, times: FloatArray) -> FloatArray:
        times = np.asarray(times, dtype=np.float64)
        position_reference = FigureEightReference(
            cruise_altitude=self.cruise_altitude,
            hold_duration=self.hold_duration,
            ramp_duration=self.ramp_duration,
            period=self.position_period,
        ).sample(times)

        path_time = np.maximum(times - self.hold_duration, 0.0)
        ramp_phase = np.clip(
            path_time / self.ramp_duration, 0.0, 1.0
        )
        ramp, ramp_rate = FigureEightReference._minimum_jerk(
            ramp_phase, self.ramp_duration
        )
        active = times >= self.hold_duration
        ramp *= active
        ramp_rate *= active

        omega = 2.0 * math.pi / self.attitude_period
        roll_shape = self.roll_amplitude * np.sin(omega * path_time)
        pitch_shape = self.pitch_amplitude * np.sin(
            1.5 * omega * path_time
        )
        yaw_shape = self.yaw_amplitude * np.sin(
            0.5 * omega * path_time
        )
        roll_shape_rate = (
            self.roll_amplitude * omega * np.cos(omega * path_time)
        )
        pitch_shape_rate = (
            self.pitch_amplitude
            * 1.5
            * omega
            * np.cos(1.5 * omega * path_time)
        )
        yaw_shape_rate = (
            self.yaw_amplitude
            * 0.5
            * omega
            * np.cos(0.5 * omega * path_time)
        )

        euler = np.stack(
            (
                ramp * roll_shape,
                ramp * pitch_shape,
                ramp * yaw_shape,
            ),
            axis=1,
        )
        euler_rates = np.stack(
            (
                ramp_rate * roll_shape + ramp * roll_shape_rate,
                ramp_rate * pitch_shape + ramp * pitch_shape_rate,
                ramp_rate * yaw_shape + ramp * yaw_shape_rate,
            ),
            axis=1,
        )

        reference = np.zeros((len(times), 13), dtype=np.float64)
        reference[:, :6] = position_reference
        reference[:, 6:10] = quaternion_from_euler(euler)
        reference[:, 10:13] = body_rates_from_euler_rates(
            euler, euler_rates
        )
        return reference


class PoseTrajectoryVisualizer(MujocoTrajectoryVisualizer):
    """Position samples plus reference attitude frames in the MuJoCo scene."""

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
            frame_indices = np.linspace(
                0, len(reference) - 1, 5, dtype=int
            )
            for index in frame_indices:
                self._add_frame(
                    scene,
                    reference[index, :3],
                    reference[index, 6:10],
                    scale=0.16,
                    alpha=0.80,
                )
            self._add_frame(
                scene,
                result.nominal_states[-1, :3],
                result.nominal_states[-1, 6:10],
                scale=0.20,
                alpha=1.0,
            )

    def _add_frame(
        self,
        scene: mujoco.MjvScene,
        position: FloatArray,
        quaternion: FloatArray,
        scale: float,
        alpha: float,
    ) -> None:
        rotation = quaternion_to_rotation_matrix(quaternion)
        colors = (
            np.array([1.0, 0.15, 0.15, alpha], dtype=np.float32),
            np.array([0.15, 1.0, 0.15, alpha], dtype=np.float32),
            np.array([0.15, 0.35, 1.0, alpha], dtype=np.float32),
        )
        for axis in range(3):
            endpoint = position + scale * rotation[:, axis]
            self._add_polyline(
                scene,
                np.stack((position, endpoint)),
                width=3.0,
                color=colors[axis],
            )


@dataclass
class PoseDemoLog:
    times: list[float]
    positions: list[FloatArray]
    reference_positions: list[FloatArray]
    quaternions: list[FloatArray]
    reference_quaternions: list[FloatArray]
    linear_actions: list[FloatArray]
    angular_actions: list[FloatArray]
    effective_sample_sizes: list[float]
    update_times_ms: list[float]

    @classmethod
    def empty(cls) -> "PoseDemoLog":
        return cls([], [], [], [], [], [], [], [], [])

    def append(
        self,
        simulation_time: float,
        state: FloatArray,
        reference: FloatArray,
        result: MPPIResult,
        update_time_ms: float,
    ) -> None:
        self.times.append(simulation_time)
        self.positions.append(state[:3].copy())
        self.reference_positions.append(reference[:3].copy())
        self.quaternions.append(state[6:10].copy())
        self.reference_quaternions.append(reference[6:10].copy())
        self.linear_actions.append(result.action[:3].copy())
        self.angular_actions.append(result.action[3:].copy())
        self.effective_sample_sizes.append(result.effective_sample_size)
        self.update_times_ms.append(update_time_ms)


@dataclass(frozen=True)
class PoseDemoMetrics:
    position_rmse_m: float
    position_max_error_m: float
    attitude_rmse_deg: float
    attitude_max_error_deg: float
    roll_rmse_deg: float
    pitch_rmse_deg: float
    yaw_rmse_deg: float
    linear_command_jerk_rms_mps3: float
    angular_command_jerk_rms_radps3: float
    mean_effective_sample_size: float
    mean_update_time_ms: float
    p95_update_time_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def wrap_angle(angle: FloatArray) -> FloatArray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_pose_metrics(log: PoseDemoLog) -> PoseDemoMetrics:
    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    reference_positions = np.asarray(log.reference_positions)
    quaternions = np.asarray(log.quaternions)
    reference_quaternions = np.asarray(log.reference_quaternions)
    linear_actions = np.asarray(log.linear_actions)
    angular_actions = np.asarray(log.angular_actions)
    update_times = np.asarray(log.update_times_ms)
    if len(times) < 3:
        raise ValueError("at least three samples are needed for metrics")

    dt = float(np.median(np.diff(times)))
    position_error = np.linalg.norm(
        positions - reference_positions, axis=1
    )
    attitude_error_vector = quaternion_error_vector(
        quaternions, reference_quaternions
    )
    attitude_error = np.linalg.norm(attitude_error_vector, axis=1)
    actual_euler = quaternion_to_euler(quaternions)
    reference_euler = quaternion_to_euler(reference_quaternions)
    euler_error = wrap_angle(actual_euler - reference_euler)

    return PoseDemoMetrics(
        position_rmse_m=float(
            np.sqrt(np.mean(np.square(position_error)))
        ),
        position_max_error_m=float(np.max(position_error)),
        attitude_rmse_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(attitude_error))))
        ),
        attitude_max_error_deg=float(np.degrees(np.max(attitude_error))),
        roll_rmse_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(euler_error[:, 0]))))
        ),
        pitch_rmse_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(euler_error[:, 1]))))
        ),
        yaw_rmse_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(euler_error[:, 2]))))
        ),
        linear_command_jerk_rms_mps3=float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(np.diff(linear_actions, axis=0) / dt),
                        axis=1,
                    )
                )
            )
        ),
        angular_command_jerk_rms_radps3=float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(np.diff(angular_actions, axis=0) / dt),
                        axis=1,
                    )
                )
            )
        ),
        mean_effective_sample_size=float(
            np.mean(log.effective_sample_sizes)
        ),
        mean_update_time_ms=float(np.mean(update_times)),
        p95_update_time_ms=float(np.percentile(update_times, 95.0)),
    )


def run_pose_demo(
    args: argparse.Namespace,
) -> tuple[PoseDemoLog, MPPIResult]:
    low_level = HnuterController(Path(args.model).resolve())
    low_level.set_freejoint_pose(
        np.array([0.0, 0.0, args.initial_altitude])
    )
    reference_generator = PoseReference(
        cruise_altitude=args.initial_altitude
    )

    control_steps = max(1, int(round(args.control_dt / low_level.dt)))
    control_dt = control_steps * low_level.dt
    dynamics = FullyActuatedUAVDynamics(dt=control_dt)
    cost = PoseTrackingCost(
        terminal_multiplier=args.terminal_multiplier
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
    mppi = MPPIController(dynamics, cost, config)
    visualizer = PoseTrajectoryVisualizer(args.visualized_samples)
    log = PoseDemoLog.empty()

    initial_state = low_level.get_mppi_pose_state()
    desired = {
        "pos": initial_state[:3],
        "vel": initial_state[3:6],
        "acc": np.zeros(3),
        "quaternion": initial_state[6:10],
        "angular_velocity": np.zeros(3),
        "angular_acceleration": np.zeros(3),
    }
    viewer_handle = None
    if not args.headless:
        viewer_handle = mujoco.viewer.launch_passive(
            low_level.model, low_level.data
        )
        viewer_handle.cam.distance = 5.0
        viewer_handle.cam.azimuth = 135.0
        viewer_handle.cam.elevation = -22.0
        viewer_handle.cam.lookat[:] = np.array([0.0, 0.0, 1.0])

    step = 0
    last_result: MPPIResult | None = None
    flight_history: list[FloatArray] = []
    last_status_time = -1.0
    wall_start = time.perf_counter()
    print(
        f"6-DoF MPPI: {args.samples} samples × {args.horizon} steps, "
        f"dt={control_dt:.3f}s"
    )
    if not args.headless:
        print(
            "Viewer: blue=samples, yellow=nominal, green=reference, "
            "magenta=actual; RGB triads show reference attitude"
        )

    try:
        while low_level.data.time < args.duration:
            if viewer_handle is not None and not viewer_handle.is_running():
                break

            if step % control_steps == 0:
                simulation_time = float(low_level.data.time)
                state = low_level.get_mppi_pose_state()
                horizon_times = simulation_time + control_dt * np.arange(
                    args.horizon + 1
                )
                reference = reference_generator.sample(horizon_times)
                update_start = time.perf_counter()
                last_result = mppi.command(state, reference)
                update_time_ms = (
                    time.perf_counter() - update_start
                ) * 1.0e3
                attitude_index = min(
                    args.attitude_lookahead_steps, args.horizon
                )
                attitude_target = (
                    reference[attitude_index]
                    if args.attitude_feedback_source == "reference"
                    else last_result.nominal_states[attitude_index]
                )
                position_target = (
                    reference[1]
                    if args.position_feedback_source == "reference"
                    else last_result.nominal_states[1]
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
                )

                if viewer_handle is not None:
                    visualizer.update(
                        viewer_handle,
                        last_result,
                        reference,
                        np.asarray(flight_history),
                    )

                if simulation_time - last_status_time >= 1.0:
                    position_error = np.linalg.norm(
                        state[:3] - reference[0, :3]
                    )
                    attitude_error_deg = np.degrees(
                        np.linalg.norm(
                            quaternion_error_vector(
                                state[6:10], reference[0, 6:10]
                            )
                        )
                    )
                    print(
                        f"t={simulation_time:5.2f}s  "
                        f"|e_p|={position_error:5.3f}m  "
                        f"|e_R|={attitude_error_deg:5.2f}deg  "
                        f"ESS={last_result.effective_sample_size:6.1f}/"
                        f"{args.samples}"
                    )
                    last_status_time = simulation_time

            low_level.set_desired_state(desired)
            low_level.update_control()
            mujoco.mj_step(low_level.model, low_level.data)
            step += 1

            if viewer_handle is not None and step % 10 == 0:
                viewer_handle.sync()
            if args.realtime and not args.headless and step % 5 == 0:
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

    if last_result is None:
        raise RuntimeError("simulation ended before the first MPPI update")
    return log, last_result


def save_pose_results(
    log: PoseDemoLog,
    last_result: MPPIResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mppi_pose_demo_log.csv"
    figure_path = output_dir / "mppi_pose_demo_results.png"
    metrics_path = output_dir / "mppi_pose_demo_metrics.json"
    metrics = compute_pose_metrics(log)

    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    reference_positions = np.asarray(log.reference_positions)
    quaternions = np.asarray(log.quaternions)
    reference_quaternions = np.asarray(log.reference_quaternions)
    actual_euler = np.degrees(quaternion_to_euler(quaternions))
    reference_euler = np.degrees(
        quaternion_to_euler(reference_quaternions)
    )
    attitude_error = np.degrees(
        np.linalg.norm(
            quaternion_error_vector(quaternions, reference_quaternions),
            axis=1,
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
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
                    *actual_euler[index],
                    *reference_euler[index],
                    attitude_error[index],
                    *log.linear_actions[index],
                    *log.angular_actions[index],
                    log.effective_sample_sizes[index],
                    log.update_times_ms[index],
                ]
            )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics.as_dict(), file, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 3)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_position = figure.add_subplot(grid[0, 1])
    axis_attitude_error = figure.add_subplot(grid[0, 2])
    angle_axes = (
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    )

    selected = np.argsort(last_result.weights)[-20:]
    for index in selected:
        sample = last_result.sampled_states[index, :, :3]
        axis_3d.plot(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            color="#3a86ff",
            alpha=0.10,
            linewidth=0.8,
        )
    axis_3d.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        reference_positions[:, 2],
        "--",
        color="#2a9d3f",
        linewidth=2.0,
        label="reference",
    )
    axis_3d.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#d81b60",
        linewidth=2.0,
        label="MuJoCo UAV",
    )
    nominal = last_result.nominal_states[:, :3]
    axis_3d.plot(
        nominal[:, 0],
        nominal[:, 1],
        nominal[:, 2],
        color="#ffb000",
        linewidth=2.5,
        label="final nominal",
    )
    axis_3d.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title="6-DoF MPPI position samples",
    )
    axis_3d.legend()

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

    axis_attitude_error.plot(times, attitude_error, color="#6a1b9a")
    axis_attitude_error.set(
        title=(
            "SO(3) attitude error "
            f"(RMSE={metrics.attitude_rmse_deg:.2f} deg)"
        ),
        xlabel="time [s]",
        ylabel="rotation error [deg]",
    )

    axis_roll_pitch, axis_yaw = angle_axes
    for index, label, color in (
        (0, "roll", "#e63946"),
        (1, "pitch", "#457b9d"),
    ):
        axis_roll_pitch.plot(
            times, actual_euler[:, index], color=color, label=label
        )
        axis_roll_pitch.plot(
            times,
            reference_euler[:, index],
            "--",
            color=color,
            alpha=0.75,
        )
    axis_roll_pitch.set(
        title="Roll / pitch: solid=actual, dashed=reference",
        xlabel="time [s]",
        ylabel="angle [deg]",
    )
    axis_roll_pitch.legend()

    axis_yaw.plot(times, actual_euler[:, 2], label="actual yaw")
    axis_yaw.plot(
        times,
        reference_euler[:, 2],
        "--",
        label="reference yaw",
    )
    axis_yaw.set(
        title=f"Yaw (RMSE={metrics.yaw_rmse_deg:.2f} deg)",
        xlabel="time [s]",
        ylabel="angle [deg]",
    )
    axis_yaw.legend()

    for axis in (
        axis_position,
        axis_attitude_error,
        axis_roll_pitch,
        axis_yaw,
    ):
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return csv_path, figure_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="6-DoF position and attitude MPPI demo"
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "hnuter206_4_5kg.xml"),
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--initial-altitude", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=250.0)
    parser.add_argument("--terminal-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--attitude-lookahead-steps",
        type=int,
        default=1,
        help="reference/nominal pose index sent to the attitude inner loop",
    )
    parser.add_argument(
        "--attitude-feedback-source",
        choices=("reference", "nominal"),
        default="reference",
        help="attitude state used by the ancillary geometric feedback",
    )
    parser.add_argument(
        "--position-feedback-source",
        choices=("reference", "nominal"),
        default="reference",
        help="position state used by the ancillary geometric feedback",
    )
    parser.add_argument(
        "--action-continuity-weight", type=float, default=1.0
    )
    parser.add_argument("--control-smoothing", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualized-samples", type=int, default=48)
    parser.add_argument("--headless", action="store_true")
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
    if (
        args.duration <= 0.0
        or args.control_dt <= 0.0
        or args.initial_altitude <= 0.2
        or args.horizon < 1
        or args.samples < 2
        or args.temperature <= 0.0
        or args.terminal_multiplier < 0.0
        or args.attitude_lookahead_steps < 1
        or args.action_continuity_weight < 0.0
        or not 0.0 <= args.control_smoothing < 1.0
    ):
        parser.error("invalid non-positive MPPI/demo parameter")
    return args


def main() -> None:
    args = parse_args()
    log, last_result = run_pose_demo(args)
    csv_path, figure_path, metrics_path = save_pose_results(
        log, last_result, args.output_dir.resolve()
    )
    metrics = compute_pose_metrics(log)
    print(
        f"Done: position RMSE={metrics.position_rmse_m:.3f}m, "
        f"attitude RMSE={metrics.attitude_rmse_deg:.2f}deg, "
        f"update={metrics.mean_update_time_ms:.2f}ms"
    )
    print(f"Log: {csv_path}")
    print(f"Plot: {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
