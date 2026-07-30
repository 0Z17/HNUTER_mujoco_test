"""Reproduce the MPPI tracking/smoothness ablation used in the README.

The comparison keeps the reference path and random seed fixed, then changes
one design choice at a time:

* original: original 0.04 control-rate cost;
* higher_rate_cost: a larger within-horizon control-rate cost;
* longer_horizon: 3 s rather than 2 s, with the sample count unchanged;
* improved: rate cost + cross-update continuity + weak causal smoothing.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hnuter_mppi_demo import (
    PROJECT_DIR,
    DemoLog,
    DemoMetrics,
    compute_metrics,
    run_demo,
)


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    horizon: int
    samples: int
    control_rate_weight: float
    action_continuity_weight: float
    control_smoothing: float


CASES = (
    ComparisonCase("original", 40, 512, 0.04, 0.0, 0.0),
    ComparisonCase("higher_rate_cost", 40, 512, 0.50, 0.0, 0.0),
    ComparisonCase("longer_horizon", 60, 512, 0.04, 0.0, 0.0),
    ComparisonCase("improved", 40, 512, 0.30, 2.0, 0.20),
)


def run_comparison(
    duration: float, seed: int
) -> list[tuple[ComparisonCase, DemoLog, DemoMetrics]]:
    results = []
    for case in CASES:
        arguments = SimpleNamespace(
            model=str(PROJECT_DIR / "hnuter206_4_5kg.xml"),
            duration=duration,
            initial_altitude=1.0,
            control_dt=0.05,
            horizon=case.horizon,
            samples=case.samples,
            temperature=80.0,
            control_rate_weight=case.control_rate_weight,
            action_continuity_weight=case.action_continuity_weight,
            control_smoothing=case.control_smoothing,
            iterations=1,
            seed=seed,
            visualized_samples=1,
            headless=True,
            realtime=False,
        )
        print(f"Running {case.name} ...")
        with redirect_stdout(StringIO()):
            log, _ = run_demo(arguments)
        results.append((case, log, compute_metrics(log)))
    return results


def save_summary(
    results: list[tuple[ComparisonCase, DemoLog, DemoMetrics]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mppi_smoothing_comparison.csv"
    figure_path = output_dir / "mppi_smoothing_comparison.png"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "horizon",
                "samples",
                "control_rate_weight",
                "action_continuity_weight",
                "control_smoothing",
                *results[0][2].as_dict().keys(),
            ]
        )
        for case, _, metrics in results:
            writer.writerow(
                [
                    case.name,
                    case.horizon,
                    case.samples,
                    case.control_rate_weight,
                    case.action_continuity_weight,
                    case.control_smoothing,
                    *metrics.as_dict().values(),
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ("#d81b60", "#2a9d8f", "#7b2cbf", "#ff9f1c")
    names = [case.name for case, _, _ in results]

    for color, (case, log, _) in zip(colors, results):
        times = np.asarray(log.times)
        positions = np.asarray(log.positions)
        references = np.asarray(log.references)
        actions = np.asarray(log.actions)
        dt = float(np.median(np.diff(times)))
        position_error = np.linalg.norm(
            positions - references, axis=1
        )
        jerk = np.linalg.norm(np.diff(actions, axis=0) / dt, axis=1)
        # A short display-only moving average keeps the comparison readable.
        kernel = np.ones(7) / 7.0
        displayed_jerk = np.convolve(jerk, kernel, mode="same")
        axes[0, 0].plot(
            times, position_error, color=color, label=case.name
        )
        axes[0, 1].plot(
            times[1:], displayed_jerk, color=color, label=case.name
        )

    rmse = [metrics.position_rmse_m for _, _, metrics in results]
    command_jerk = [
        metrics.command_jerk_rms_mps3 for _, _, metrics in results
    ]
    axes[1, 0].bar(names, rmse, color=colors)
    axes[1, 1].bar(names, command_jerk, color=colors)

    axes[0, 0].set(
        title="Closed-loop position error",
        xlabel="time [s]",
        ylabel="position error [m]",
    )
    axes[0, 1].set(
        title="Command jerk (7-sample display average)",
        xlabel="time [s]",
        ylabel="jerk [m/s³]",
    )
    axes[1, 0].set(
        title="Tracking RMSE",
        ylabel="RMSE [m]",
    )
    axes[1, 1].set(
        title="Command jerk RMS",
        ylabel="jerk RMS [m/s³]",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", rotation=12)
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return csv_path, figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HNUTER MPPI smoothing configurations"
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
    )
    args = parser.parse_args()
    if args.duration <= 0.2:
        parser.error("duration must exceed 0.2 seconds")
    return args


def main() -> None:
    args = parse_args()
    results = run_comparison(args.duration, args.seed)
    csv_path, figure_path = save_summary(
        results, args.output_dir.resolve()
    )
    print(
        "\ncase                 RMSE[m]  cmd jerk  actual jerk  update[ms]"
    )
    for case, _, metrics in results:
        print(
            f"{case.name:20s} "
            f"{metrics.position_rmse_m:7.4f}  "
            f"{metrics.command_jerk_rms_mps3:8.2f}  "
            f"{metrics.actual_jerk_rms_mps3:11.2f}  "
            f"{metrics.mean_update_time_ms:10.2f}"
        )
    print(f"CSV: {csv_path}")
    print(f"Plot: {figure_path}")


if __name__ == "__main__":
    main()
