#!/usr/bin/env python3
"""Prepare a B-spline control-point training set from the expert dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bspline_control import (
    BsplineConfig,
    evaluate_control_points,
    fit_control_points,
)
from se3_diffusion import (
    PreparedData,
    _obstacle_tokens,
    pose7_to_pose9,
    pose9_to_pose7_numpy,
    resample_se3_path,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "datasets/diffusion_se3_multihomotopy_v002_300"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results/bspline_control_points_v001",
    )
    parser.add_argument("--n-control-points", type=int, default=24)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--n-points", type=int, default=128)
    parser.add_argument("--reverse-train", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    splits = json.loads(
        (dataset_root / "splits.json").read_text(encoding="utf-8")
    )
    split_sets = {
        0: set(splits["train_pair_indices"]),
        1: set(splits["validation_pair_indices"]),
        2: set(splits["test_pair_indices"]),
    }
    config = BsplineConfig(
        n_control_points=args.n_control_points,
        degree=args.degree,
        n_points=args.n_points,
    )
    control_points: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    pair_indices: list[int] = []
    trajectory_indices: list[int] = []
    split_codes: list[int] = []
    reversed_flags: list[bool] = []
    reconstructed: list[np.ndarray] = []
    fit_errors: list[float] = []
    for record in manifest["trajectories"]:
        pair = int(record["pair_index"])
        split = next(
            code for code, values in split_sets.items() if pair in values
        )
        with np.load(dataset_root / record["training_sample"]) as sample:
            smoothed = sample["smoothed_path_states"].astype(np.float64)
        dense7 = resample_se3_path(smoothed, args.n_points)
        dense9 = pose7_to_pose9(dense7).astype(np.float64)
        control = fit_control_points(
            dense9,
            n_control_points=config.n_control_points,
            degree=config.degree,
            n_points=config.n_points,
            fixed_endpoints=True,
        )
        reconstructed9 = evaluate_control_points(
            control,
            n_points=config.n_points,
            degree=config.degree,
        )
        reconstructed7 = pose9_to_pose7_numpy(reconstructed9)
        error = float(
            np.mean(
                np.linalg.norm(reconstructed9 - dense9, axis=-1)
            )
        )
        fit_errors.append(error)
        control_points.append(control.astype(np.float32))
        conditions.append(
            np.concatenate((control[0], control[-1])).astype(np.float32)
        )
        reconstructed.append(reconstructed7.astype(np.float32))
        pair_indices.append(pair)
        trajectory_indices.append(int(record["trajectory_index"]))
        split_codes.append(split)
        reversed_flags.append(False)
        if args.reverse_train and split == 0:
            reverse = control[::-1].copy()
            control_points.append(reverse.astype(np.float32))
            conditions.append(
                np.concatenate((reverse[0], reverse[-1])).astype(np.float32)
            )
            reconstructed.append(reconstructed7[::-1].astype(np.float32))
            pair_indices.append(pair)
            trajectory_indices.append(int(record["trajectory_index"]))
            split_codes.append(split)
            reversed_flags.append(True)
    path_array = np.stack(control_points)
    split_array = np.asarray(split_codes, dtype=np.int8)
    training = path_array[split_array == 0]
    mean = training.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(training.std(axis=(0, 1)), 1e-3).astype(np.float32)
    dataset_config = json.loads(
        (dataset_root / "dataset_config.json").read_text(encoding="utf-8")
    )
    environment_path = dataset_root / dataset_config["sources"]["environment"]["copy"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    obstacles, obstacle_mask = _obstacle_tokens(environment)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedData(
        paths=path_array,
        conditions=np.stack(conditions).astype(np.float32),
        pair_indices=np.asarray(pair_indices, dtype=np.int64),
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int64),
        split_codes=split_array,
        reversed_flags=np.asarray(reversed_flags, dtype=bool),
        path_mean=mean,
        path_std=std,
        obstacles=obstacles,
        obstacle_mask=obstacle_mask,
        source_dataset=str(dataset_root),
        environment_path=str(environment_path),
    )
    prepared_path = output_dir / "prepared_bspline.npz"
    prepared.save(prepared_path)
    np.savez_compressed(
        output_dir / "reconstructed_paths.npz",
        poses_wxyz=np.stack(reconstructed),
        fit_error_m=np.asarray(fit_errors),
    )
    print(
        f"prepared {len(prepared.paths)} control-point samples -> {prepared_path}"
    )
    print(
        f"control points {config.n_control_points}, degree {config.degree}, "
        f"{config.n_points} eval points"
    )
    print(
        f"fit reconstruction mean error: "
        f"{np.mean(fit_errors):.4f} (9D units), p95 "
        f"{np.percentile(fit_errors, 95):.4f}"
    )


if __name__ == "__main__":
    main()
