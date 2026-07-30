"""Visual MPPI path-tracking demo for the HNUTER MuJoCo aircraft.

Run with a live MuJoCo window:

    python hnuter_mppi_demo.py

Run a short non-interactive verification:

    python hnuter_mppi_demo.py --headless --duration 8
"""

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
from mppi import (
    MPPIConfig,
    MPPIController,
    MPPIResult,
    PointMassDynamics,
    QuadraticTrackingCost,
)


FloatArray = NDArray[np.float64]
PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FigureEightReference:
    """Hover briefly, then enter a smooth three-dimensional figure-eight."""

    cruise_altitude: float = 1.0
    hold_duration: float = 1.0
    ramp_duration: float = 2.0
    amplitude: float = 1.25
    vertical_amplitude: float = 0.16
    period: float = 12.0

    def sample(self, times: FloatArray) -> FloatArray:
        times = np.asarray(times, dtype=np.float64)
        trajectory = np.zeros((len(times), 6), dtype=np.float64)
        trajectory[:, 2] = self.cruise_altitude

        path_start = self.hold_duration
        path_time = np.maximum(times - path_start, 0.0)
        ramp_phase = np.clip(path_time / self.ramp_duration, 0.0, 1.0)
        ramp, ramp_rate = self._minimum_jerk(
            ramp_phase, self.ramp_duration
        )
        active = times >= path_start
        ramp *= active
        ramp_rate *= active

        omega = 2.0 * math.pi / self.period
        sin_1 = np.sin(omega * path_time)
        cos_1 = np.cos(omega * path_time)
        sin_2 = np.sin(2.0 * omega * path_time)
        cos_2 = np.cos(2.0 * omega * path_time)

        shape_x = self.amplitude * sin_1
        shape_y = 0.62 * self.amplitude * sin_2
        shape_z = self.vertical_amplitude * sin_1
        shape_vx = self.amplitude * omega * cos_1
        shape_vy = 1.24 * self.amplitude * omega * cos_2
        shape_vz = self.vertical_amplitude * omega * cos_1

        trajectory[:, 0] = ramp * shape_x
        trajectory[:, 1] = ramp * shape_y
        trajectory[:, 2] += ramp * shape_z
        trajectory[:, 3] = ramp_rate * shape_x + ramp * shape_vx
        trajectory[:, 4] = ramp_rate * shape_y + ramp * shape_vy
        trajectory[:, 5] += ramp_rate * shape_z + ramp * shape_vz
        return trajectory

    @staticmethod
    def _minimum_jerk(
        normalized_time: FloatArray, duration: float
    ) -> tuple[FloatArray, FloatArray]:
        phase = normalized_time
        value = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        rate = (
            30.0 * phase**2
            - 60.0 * phase**3
            + 30.0 * phase**4
        ) / duration
        rate[(phase <= 0.0) | (phase >= 1.0)] = 0.0
        return value, rate


class MujocoTrajectoryVisualizer:
    """Draw MPPI rollouts in a passive viewer's user scene."""

    def __init__(self, visualized_samples: int = 56) -> None:
        self.visualized_samples = visualized_samples
        self._identity = np.eye(3, dtype=np.float64).reshape(-1)
        self._zero = np.zeros(3, dtype=np.float64)

    def update(
        self,
        viewer_handle: Any,
        result: MPPIResult,
        reference: FloatArray,
        flight_history: FloatArray,
    ) -> None:
        with viewer_handle.lock():
            scene = viewer_handle.user_scn
            scene.ngeom = 0

            sample_count = min(
                self.visualized_samples, len(result.weights)
            )
            top_indices = np.argsort(result.weights)[-sample_count:]
            top_weights = result.weights[top_indices]
            max_weight = max(float(np.max(top_weights)), 1.0e-12)

            for index in top_indices:
                relative_weight = float(result.weights[index] / max_weight)
                color = np.array(
                    [
                        0.10,
                        0.35 + 0.50 * relative_weight,
                        1.00,
                        0.10 + 0.35 * relative_weight,
                    ],
                    dtype=np.float32,
                )
                self._add_polyline(
                    scene,
                    result.sampled_states[index, :, :3],
                    width=1.0,
                    color=color,
                )

            self._add_polyline(
                scene,
                reference[:, :3],
                width=3.0,
                color=np.array([0.15, 1.0, 0.20, 0.90], dtype=np.float32),
            )
            self._add_polyline(
                scene,
                result.nominal_states[:, :3],
                width=4.0,
                color=np.array([1.0, 0.72, 0.05, 1.0], dtype=np.float32),
            )
            if len(flight_history) > 1:
                self._add_polyline(
                    scene,
                    flight_history[-500:],
                    width=3.0,
                    color=np.array(
                        [1.0, 0.15, 0.65, 0.95], dtype=np.float32
                    ),
                )
            self._add_sphere(
                scene,
                reference[0, :3],
                radius=0.045,
                color=np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
            )

    def _add_polyline(
        self,
        scene: mujoco.MjvScene,
        points: FloatArray,
        width: float,
        color: NDArray[np.float32],
    ) -> None:
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) < 1.0e-7:
                continue
            if scene.ngeom >= scene.maxgeom:
                return
            geometry = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geometry,
                mujoco.mjtGeom.mjGEOM_LINE,
                self._zero,
                self._zero,
                self._identity,
                color,
            )
            mujoco.mjv_connector(
                geometry,
                mujoco.mjtGeom.mjGEOM_LINE,
                width,
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )
            scene.ngeom += 1

    def _add_sphere(
        self,
        scene: mujoco.MjvScene,
        position: FloatArray,
        radius: float,
        color: NDArray[np.float32],
    ) -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius, dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            self._identity,
            color,
        )
        scene.ngeom += 1


@dataclass
class DemoLog:
    times: list[float]
    positions: list[FloatArray]
    velocities: list[FloatArray]
    references: list[FloatArray]
    actions: list[FloatArray]
    costs: list[float]
    effective_sample_sizes: list[float]
    update_times_ms: list[float]

    @classmethod
    def empty(cls) -> "DemoLog":
        return cls([], [], [], [], [], [], [], [])

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
        self.velocities.append(state[3:].copy())
        self.references.append(reference[:3].copy())
        self.actions.append(result.action.copy())
        self.costs.append(float(np.min(result.costs)))
        self.effective_sample_sizes.append(result.effective_sample_size)
        self.update_times_ms.append(update_time_ms)


@dataclass(frozen=True)
class DemoMetrics:
    """Tracking, smoothness, and compute metrics from a closed-loop run."""

    position_rmse_m: float
    position_max_error_m: float
    command_jerk_rms_mps3: float
    command_total_variation_per_s_mps3: float
    actual_jerk_rms_mps3: float
    mean_update_time_ms: float
    p95_update_time_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(getattr(self, field))
            for field in self.__dataclass_fields__
        }


def compute_metrics(log: DemoLog) -> DemoMetrics:
    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    velocities = np.asarray(log.velocities)
    references = np.asarray(log.references)
    actions = np.asarray(log.actions)
    update_times = np.asarray(log.update_times_ms)
    if len(times) < 4:
        raise ValueError("at least four log samples are needed for metrics")

    dt = float(np.median(np.diff(times)))
    position_error = np.linalg.norm(positions - references, axis=1)
    command_jerk = np.diff(actions, axis=0) / dt
    actual_acceleration = np.diff(velocities, axis=0) / dt
    actual_jerk = np.diff(actual_acceleration, axis=0) / dt
    duration = max(float(times[-1] - times[0]), dt)
    return DemoMetrics(
        position_rmse_m=float(
            np.sqrt(np.mean(np.square(position_error)))
        ),
        position_max_error_m=float(np.max(position_error)),
        command_jerk_rms_mps3=float(
            np.sqrt(np.mean(np.sum(np.square(command_jerk), axis=1)))
        ),
        command_total_variation_per_s_mps3=float(
            np.sum(np.linalg.norm(np.diff(actions, axis=0), axis=1))
            / duration
        ),
        actual_jerk_rms_mps3=float(
            np.sqrt(np.mean(np.sum(np.square(actual_jerk), axis=1)))
        ),
        mean_update_time_ms=float(np.mean(update_times)),
        p95_update_time_ms=float(np.percentile(update_times, 95.0)),
    )


def run_demo(args: argparse.Namespace) -> tuple[DemoLog, MPPIResult]:
    xml_path = Path(args.model).resolve()
    low_level = HnuterController(xml_path)
    reference_generator = FigureEightReference(
        cruise_altitude=args.initial_altitude
    )
    # The source XML starts close to the ground.  Starting this path-tracking
    # visualization airborne avoids contact transients obscuring MPPI behavior.
    low_level.set_freejoint_pose(
        np.array([0.0, 0.0, args.initial_altitude])
    )

    control_steps = max(1, int(round(args.control_dt / low_level.dt)))
    control_dt = control_steps * low_level.dt
    dynamics = PointMassDynamics(dt=control_dt)
    config = MPPIConfig(
        horizon=args.horizon,
        num_samples=args.samples,
        temperature=args.temperature,
        noise_sigma=(2.3, 2.3, 2.0),
        control_min=(-4.0, -4.0, -3.5),
        control_max=(4.0, 4.0, 3.5),
        noise_correlation=0.60,
        likelihood_ratio_weight=0.08,
        action_continuity_weight=args.action_continuity_weight,
        control_smoothing=args.control_smoothing,
        num_iterations=args.iterations,
        seed=args.seed,
    )
    path_cost = QuadraticTrackingCost(
        control_rate_weight=args.control_rate_weight
    )
    mppi = MPPIController(dynamics, path_cost, config)
    visualizer = MujocoTrajectoryVisualizer(args.visualized_samples)
    log = DemoLog.empty()

    desired = {
        "pos": low_level.get_state()["position"].copy(),
        "vel": np.zeros(3),
        "acc": np.zeros(3),
        "euler": np.zeros(3),
        "euler_rate": np.zeros(3),
    }
    last_result: MPPIResult | None = None
    flight_history: list[FloatArray] = []
    viewer_handle = None

    if not args.headless:
        viewer_handle = mujoco.viewer.launch_passive(
            low_level.model, low_level.data
        )
        viewer_handle.cam.distance = 5.0
        viewer_handle.cam.azimuth = 135.0
        viewer_handle.cam.elevation = -22.0
        viewer_handle.cam.lookat[:] = np.array([0.0, 0.0, 1.0])

    wall_start = time.perf_counter()
    step = 0
    last_status_time = -1.0
    print(
        f"MPPI demo: {args.samples} samples × {args.horizon} steps, "
        f"outer-loop dt={control_dt:.3f}s"
    )
    if not args.headless:
        print(
            "Viewer: blue=sampled rollouts, yellow=MPPI nominal, "
            "green=reference, magenta=actual"
        )

    try:
        while low_level.data.time < args.duration:
            if viewer_handle is not None and not viewer_handle.is_running():
                break

            if step % control_steps == 0:
                simulation_time = float(low_level.data.time)
                state = low_level.get_mppi_state()
                horizon_times = simulation_time + control_dt * np.arange(
                    args.horizon + 1
                )
                reference = reference_generator.sample(horizon_times)
                update_start = time.perf_counter()
                last_result = mppi.command(state, reference)
                update_time_ms = (
                    time.perf_counter() - update_start
                ) * 1.0e3

                # Track the first predicted state and apply MPPI acceleration as
                # feed-forward to the full nonlinear MuJoCo vehicle.
                desired = {
                    "pos": last_result.nominal_states[1, :3],
                    "vel": last_result.nominal_states[1, 3:],
                    "acc": last_result.action,
                    "euler": np.zeros(3),
                    "euler_rate": np.zeros(3),
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
                    tracking_error = np.linalg.norm(
                        state[:3] - reference[0, :3]
                    )
                    print(
                        f"t={simulation_time:5.2f}s  "
                        f"|e_p|={tracking_error:5.3f}m  "
                        f"ESS={last_result.effective_sample_size:6.1f}/"
                        f"{args.samples}  "
                        f"a={np.array2string(last_result.action, precision=2)}"
                    )
                    last_status_time = simulation_time

            low_level.set_desired_state(desired)
            low_level.update_control()
            mujoco.mj_step(low_level.model, low_level.data)
            step += 1

            if viewer_handle is not None and step % 10 == 0:
                viewer_handle.sync()
            if args.realtime and not args.headless and step % 5 == 0:
                target_wall_time = wall_start + float(low_level.data.time)
                remaining = target_wall_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer_handle is not None:
            viewer_handle.close()

    if last_result is None:
        raise RuntimeError("simulation ended before the first MPPI update")
    return log, last_result


def save_results(
    log: DemoLog, last_result: MPPIResult, output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mppi_demo_log.csv"
    figure_path = output_dir / "mppi_demo_results.png"
    metrics_path = output_dir / "mppi_demo_metrics.json"
    metrics = compute_metrics(log)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_s",
                "x_m",
                "y_m",
                "z_m",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "x_ref_m",
                "y_ref_m",
                "z_ref_m",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "minimum_sample_cost",
                "effective_sample_size",
                "mppi_update_time_ms",
            ]
        )
        for values in zip(
            log.times,
            log.positions,
            log.velocities,
            log.references,
            log.actions,
            log.costs,
            log.effective_sample_sizes,
            log.update_times_ms,
        ):
            (
                simulation_time,
                position,
                velocity,
                reference,
                action,
                cost,
                ess,
                update_time_ms,
            ) = values
            writer.writerow(
                [
                    simulation_time,
                    *position,
                    *velocity,
                    *reference,
                    *action,
                    cost,
                    ess,
                    update_time_ms,
                ]
            )
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics.as_dict(), file, indent=2)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(log.times)
    positions = np.asarray(log.positions)
    references = np.asarray(log.references)
    actions = np.asarray(log.actions)
    errors = np.linalg.norm(positions - references, axis=1)

    figure = plt.figure(figsize=(13, 9))
    grid = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_error = figure.add_subplot(grid[0, 1])
    axis_control = figure.add_subplot(grid[1, 1])

    selected = np.argsort(last_result.weights)[-20:]
    for index in selected:
        sample = last_result.sampled_states[index, :, :3]
        axis_3d.plot(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            color="#3a86ff",
            alpha=0.12,
            linewidth=0.8,
        )
    axis_3d.plot(
        references[:, 0],
        references[:, 1],
        references[:, 2],
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
        label="final MPPI nominal",
    )
    axis_3d.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title="MPPI trajectory samples and closed-loop flight",
    )
    axis_3d.legend(loc="upper left")

    axis_error.plot(times, errors, color="#d81b60")
    axis_error.set(
        xlabel="time [s]",
        ylabel="position error [m]",
        title=f"Position error (RMSE={metrics.position_rmse_m:.3f} m)",
    )
    axis_error.grid(True, alpha=0.3)

    for index, label in enumerate(("ax", "ay", "az")):
        axis_control.plot(times, actions[:, index], label=label)
    axis_control.set(
        xlabel="time [s]",
        ylabel="commanded acceleration [m/s²]",
        title=(
            "MPPI command "
            f"(jerk RMS={metrics.command_jerk_rms_mps3:.1f} m/s³)"
        ),
    )
    axis_control.grid(True, alpha=0.3)
    axis_control.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return csv_path, figure_path, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visual MPPI controller demo for the HNUTER MuJoCo UAV"
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_DIR / "hnuter206_4_5kg.xml"),
        help="MuJoCo XML model path",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--initial-altitude", type=float, default=1.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=80.0)
    parser.add_argument("--control-rate-weight", type=float, default=0.30)
    parser.add_argument(
        "--action-continuity-weight", type=float, default=2.0
    )
    parser.add_argument("--control-smoothing", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visualized-samples", type=int, default=56)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="do not pace the visual simulation to wall-clock time",
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
    ):
        parser.error(
            "duration/control-dt must be positive and initial-altitude "
            "must exceed 0.2m"
        )
    if args.visualized_samples < 1:
        parser.error("visualized-samples must be positive")
    if (
        args.control_rate_weight < 0.0
        or args.action_continuity_weight < 0.0
        or not 0.0 <= args.control_smoothing < 1.0
    ):
        parser.error(
            "smoothness weights must be non-negative and "
            "control-smoothing must be in [0, 1)"
        )
    return args


def main() -> None:
    args = parse_args()
    log, last_result = run_demo(args)
    csv_path, figure_path, metrics_path = save_results(
        log, last_result, args.output_dir.resolve()
    )
    metrics = compute_metrics(log)
    print(
        f"Done: position RMSE={metrics.position_rmse_m:.3f}m, "
        f"command jerk RMS={metrics.command_jerk_rms_mps3:.2f}m/s³, "
        f"actual jerk RMS={metrics.actual_jerk_rms_mps3:.2f}m/s³"
    )
    print(f"Log: {csv_path}")
    print(f"Plot: {figure_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
