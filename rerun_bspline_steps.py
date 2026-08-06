#!/usr/bin/env python3
"""Rerun replay of B-spline control-point diffusion denoising.

The model denoises 24 SE(3) control points; every DDIM iteration is recorded
and visualised on the ``diffusion_step`` timeline:

  * ``world/diffusion_steps/path``          - the smooth spline evaluated on 128
    points (solid line, blue -> red over iterations);
  * ``world/diffusion_steps/control_points``- the control polygon (thin line)
    and control points (small spheres);
  * ``world/diffusion_steps/x0_pred``       - the model's x0 control-point
    estimate (translucent);
  * ``world/final_path``                    - converged spline path;
  * plots: path length, path step-delta, control-point step-delta, and
    ESDF+sphere minimum clearance per iteration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

os.environ.setdefault(
    "WARP_CACHE_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "warp")
)

from bspline_control import basis_matrix, evaluate_torch
from se3_diffusion import (
    DiffusionSchedule,
    EsdfDistanceField,
    GuidanceConfig,
    PreparedData,
    ddim_sample,
    pose7_to_pose9,
    pose9_to_pose7_numpy,
)
from train_se3_diffusion import load_model
from curobo_collision import load_curobo_spheres
from rerun_diffusion_steps import _load_accepted_pair

import rerun as rr


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREPARED = (
    PROJECT_DIR / "results/bspline_control_points_v001/prepared_bspline.npz"
)
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "results/bspline_control_points_v001/models/unet/best.pt"
)
DEFAULT_SPHERES = (
    PROJECT_DIR
    / "etc"
    / "URDF-for-gazebo"
    / "config"
    / "HDJQR-0102-0055.SLDASM_curobo_spheres.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--spheres", type=Path, default=DEFAULT_SPHERES)
    parser.add_argument("--start-pose", type=float, nargs=7)
    parser.add_argument("--goal-pose", type=float, nargs=7)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--sample-clip-x0", type=float, default=4.0)
    parser.add_argument("--esdf", type=Path)
    parser.add_argument("--evaluation-points", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_endpoints(args: argparse.Namespace):
    explicit = (
        args.start_pose is not None
        and args.goal_pose is not None
        and args.seed is not None
    )
    demo_dir = args.demo_dir
    if demo_dir is None and not explicit:
        candidates = sorted(
            PROJECT_DIR.glob("results/unet_guidance_demo_*/pair_attempts.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                _load_accepted_pair(candidate)
                demo_dir = candidate.parent
                break
            except RuntimeError:
                continue
    if demo_dir is not None:
        record = _load_accepted_pair(demo_dir / "pair_attempts.json")
        print(
            f"using demo run {demo_dir.name}: seed={record['sampler_seed']}",
            flush=True,
        )
        return (
            np.asarray(record["start_pose"], dtype=np.float64),
            np.asarray(record["goal_pose"], dtype=np.float64),
            int(record["sampler_seed"]),
        )
    if explicit:
        return (
            np.asarray(args.start_pose, dtype=np.float64),
            np.asarray(args.goal_pose, dtype=np.float64),
            int(args.seed),
        )
    raise SystemExit(
        "provide --start-pose/--goal-pose/--seed, --demo-dir, or leave empty "
        "to reuse the newest demo run"
    )


def _quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    start, goal, seed = _resolve_endpoints(args)
    cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and cuda)
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    prepared = PreparedData.load(args.prepared.resolve())
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    sequence_length = int(checkpoint["architecture"]["sequence_length"])
    degree = int(checkpoint.get("bspline_degree", 5))
    schedule = DiffusionSchedule.cosine(checkpoint["diffusion_steps"], device)
    environment = json.loads(
        Path(prepared.environment_path).read_text(encoding="utf-8")
    )
    esdf_path = args.esdf
    if esdf_path is None:
        environment_id = environment.get("environment_id", "environment")
        esdf_path = Path(f"/tmp/esdf_{environment_id}_0.025.npz")
    esdf = (
        EsdfDistanceField.load_cache(esdf_path).to(device)
        if esdf_path.exists()
        else EsdfDistanceField.from_environment(
            environment, resolution=0.025
        ).to(device)
    )
    sphere_set = load_curobo_spheres(args.spheres.expanduser().resolve())
    sphere_centers = torch.from_numpy(sphere_set.centers).to(device)
    sphere_radii = torch.from_numpy(sphere_set.radii).to(device)
    obstacles = torch.from_numpy(prepared.obstacles).float().to(device)
    obstacle_mask = torch.from_numpy(prepared.obstacle_mask).bool().to(device)
    bounds = environment["sampling_space"]["position_bounds"]
    bounds_min = torch.as_tensor(
        bounds["min"], dtype=torch.float32, device=device
    )
    bounds_max = torch.as_tensor(
        bounds["max"], dtype=torch.float32, device=device
    )
    mean = torch.from_numpy(prepared.path_mean).float().to(device)[None, None]
    std = torch.from_numpy(prepared.path_std).float().to(device)[None, None]
    condition_mean = np.tile(prepared.path_mean, 2)
    condition_std = np.tile(prepared.path_std, 2)
    guidance = GuidanceConfig(
        enabled=True,
        start_fraction=args.guidance_fraction,
        scale=args.guidance_scale,
        steps_per_diffusion_step=args.guidance_steps,
        max_perturbation=args.guidance_max_perturbation,
        clearance_m=args.guidance_clearance,
    )

    def evaluate_fn(control_points: torch.Tensor) -> torch.Tensor:
        return evaluate_torch(
            control_points,
            n_points=args.evaluation_points,
            degree=degree,
        )

    pose9 = pose7_to_pose9(np.stack((start, goal))).astype(np.float32)
    raw_condition = np.concatenate((pose9[0], pose9[1]))
    conditions = torch.from_numpy(
        ((raw_condition - condition_mean) / condition_std)[None]
    ).float().to(device)

    steps: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    def on_step(
        sample_index: int,
        timestep_value: int,
        path: torch.Tensor,
        predicted_x0: torch.Tensor,
    ) -> None:
        steps.append(
            (
                sample_index,
                timestep_value,
                path.detach().cpu().numpy().copy(),
                predicted_x0.detach().cpu().numpy().copy(),
            )
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    if device.type == "cuda":
        warmup_generator = torch.Generator(device=device).manual_seed(seed + 1)
        _ = ddim_sample(
            model, conditions, schedule, sequence_length,
            min(5, args.ddim_steps), obstacles, obstacle_mask, mean, std,
            bounds_min, bounds_max, guidance, warmup_generator,
            clip_x0=args.sample_clip_x0, esdf=esdf,
            robot_sphere_centers=sphere_centers,
            robot_sphere_radii=sphere_radii, evaluate_fn=evaluate_fn,
        )
        torch.cuda.synchronize(device)
    started = time.monotonic()
    final = ddim_sample(
        model, conditions, schedule, sequence_length, args.ddim_steps,
        obstacles, obstacle_mask, mean, std, bounds_min, bounds_max,
        guidance, generator, clip_x0=args.sample_clip_x0, esdf=esdf,
        robot_sphere_centers=sphere_centers,
        robot_sphere_radii=sphere_radii, evaluate_fn=evaluate_fn,
        on_step=on_step,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_time = time.monotonic() - started
    basis = basis_matrix(sequence_length, degree, args.evaluation_points)

    def to_poses7(normalized9: np.ndarray) -> np.ndarray:
        if normalized9.ndim == 3:
            normalized9 = normalized9[0]
        control = (
            normalized9 * prepared.path_std + prepared.path_mean
        )
        control[0] = pose9[0].astype(np.float32)
        control[-1] = pose9[1].astype(np.float32)
        dense9 = basis @ control
        poses = pose9_to_pose7_numpy(dense9).astype(np.float32)
        poses[0] = start.astype(np.float32)
        poses[-1] = goal.astype(np.float32)
        return poses

    def to_control_positions(normalized9: np.ndarray) -> np.ndarray:
        if normalized9.ndim == 3:
            normalized9 = normalized9[0]
        control = normalized9 * prepared.path_std + prepared.path_mean
        control[0] = pose9[0].astype(np.float32)
        control[-1] = pose9[1].astype(np.float32)
        return control[:, :3]

    def sphere_clearance(positions: np.ndarray, quaternions: np.ndarray) -> float:
        with torch.no_grad():
            pos = torch.from_numpy(positions.astype(np.float32)).to(device)
            matrices = np.stack(
                [_quaternion_to_matrix(q) for q in quaternions]
            )
            world = pos[:, None, :] + torch.from_numpy(
                np.einsum("nij,sj->nsi", matrices, sphere_set.centers)
            ).to(device)
            distance = esdf.sample(world) - sphere_radii
            return float(distance.amin())

    final_cp = final.detach().cpu().numpy()[0]
    final_poses = to_poses7(final_cp)
    records = []
    previous_path: np.ndarray | None = None
    previous_cp: np.ndarray | None = None
    for sample_index, timestep_value, path_np, x0_np in steps:
        poses = to_poses7(path_np)
        positions = poses[:, :3]
        length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        path_delta = (
            float(np.linalg.norm(positions - previous_path, axis=1).mean())
            if previous_path is not None
            else 0.0
        )
        control_positions = to_control_positions(path_np)
        cp_delta = (
            float(
                np.linalg.norm(
                    control_positions - previous_cp, axis=1
                ).mean()
            )
            if previous_cp is not None
            else 0.0
        )
        clearance = sphere_clearance(positions, poses[:, 3:7])
        records.append({
            "step": sample_index,
            "timestep": timestep_value,
            "translation_length_m": round(length, 4),
            "path_step_delta_m": round(path_delta, 5),
            "control_point_step_delta_m": round(cp_delta, 5),
            "min_sphere_clearance_m": round(clearance, 4),
        })
        previous_path = positions
        previous_cp = control_positions

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else PROJECT_DIR / "results" / f"bspline_steps_replay_{stamp}.rrd"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def gradient_color(fraction: float) -> tuple[int, int, int, int]:
        blue = np.asarray([30, 90, 230], dtype=np.float32)
        red = np.asarray([225, 60, 40], dtype=np.float32)
        color = blue + (red - blue) * fraction
        return (int(color[0]), int(color[1]), int(color[2]), 220)

    with rr.RecordingStream(
        "bspline_step_replay",
        recording_id=f"bspline_steps_seed{seed}",
    ) as recording:
        recording.save(output)
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        boxes = [
            obstacle
            for obstacle in environment.get("obstacles", [])
            if obstacle.get("collision", False) and obstacle.get("type") == "box"
        ]
        if boxes:
            recording.log(
                "world/environment/collision_boxes",
                rr.Boxes3D(
                    centers=[box["pose"]["position"] for box in boxes],
                    half_sizes=[
                        0.5 * np.asarray(box["size_xyz"]) for box in boxes
                    ],
                    quaternions=[
                        rr.Quaternion(
                            xyzw=np.roll(box["pose"]["quaternion_wxyz"], -1)
                        )
                        for box in boxes
                    ],
                    labels=[box["id"] for box in boxes],
                    colors=[(120, 150, 200, 140) for _ in boxes],
                    radii=0.008,
                    fill_mode="solid",
                ),
                static=True,
            )
        recording.log(
            "world/endpoints",
            rr.Points3D(
                [start[:3], goal[:3]],
                colors=[(40, 100, 255, 255), (30, 220, 85, 255)],
                radii=0.06,
                labels=["start", "goal"],
            ),
            static=True,
        )
        recording.log(
            "world/final_path",
            rr.LineStrips3D(
                [final_poses[:, :3]],
                colors=[(0, 190, 200, 255)],
                radii=[0.022],
                labels=["final B-spline path"],
            ),
            static=True,
        )
        final_control = to_control_positions(final_cp)
        recording.log(
            "world/final_control_points",
            rr.Points3D(
                final_control,
                colors=[(200, 120, 20, 255) for _ in final_control],
                radii=0.035,
                labels=["final control points"],
            ),
            static=True,
        )
        for sample_index, timestep_value, path_np, x0_np in steps:
            poses = to_poses7(path_np)
            x0_poses = to_poses7(x0_np)
            control_positions = to_control_positions(path_np)
            fraction = sample_index / max(len(steps) - 1, 1)
            color = gradient_color(fraction)
            recording.set_time("diffusion_step", sequence=sample_index)
            recording.log(
                "world/diffusion_steps/path",
                rr.LineStrips3D(
                    [poses[:, :3]],
                    colors=[color],
                    radii=[0.010],
                    labels=[f"step {sample_index} (t={timestep_value})"],
                ),
            )
            recording.log(
                "world/diffusion_steps/control_points",
                rr.LineStrips3D(
                    [control_positions],
                    colors=[(220, 180, 60, 220)],
                    radii=[0.004],
                    labels=["control polygon"],
                ),
            )
            recording.log(
                "world/diffusion_steps/control_points/points",
                rr.Points3D(
                    control_positions,
                    colors=[
                        (220, 180, 60, 255) for _ in control_positions
                    ],
                    radii=0.03,
                    labels=["control points"],
                ),
            )
            recording.log(
                "world/diffusion_steps/x0_pred",
                rr.LineStrips3D(
                    [x0_poses[:, :3]],
                    colors=[(color[0], color[1], color[2], 90)],
                    radii=[0.004],
                    labels=[f"x0 pred step {sample_index}"],
                ),
            )
        for sample_index, timestep_value, path_np, x0_np in steps:
            record = records[sample_index]
            recording.set_time("diffusion_step", sequence=sample_index)
            recording.log(
                "plots/path_length",
                rr.Scalars([record["translation_length_m"]]),
            )
            recording.log(
                "plots/path_step_delta",
                rr.Scalars([record["path_step_delta_m"]]),
            )
            recording.log(
                "plots/control_point_step_delta",
                rr.Scalars([record["control_point_step_delta_m"]]),
            )
            recording.log(
                "plots/min_clearance",
                rr.Scalars([record["min_sphere_clearance_m"]]),
            )

    summary = {
        "output": str(output),
        "seed": seed,
        "ddim_steps": len(steps),
        "control_points": sequence_length,
        "bspline_degree": degree,
        "sampling_time_s": round(sampling_time, 4),
        "final_translation_length_m": float(
            np.linalg.norm(
                np.diff(final_poses[:, :3], axis=0), axis=1
            ).sum()
        ),
        "steps": records,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"recorded {len(steps)} diffusion steps in {sampling_time:.2f}s")
    print(f"RERUN_REPLAY={output}")
    print(f"VIEW=rerun --port auto {output}")


if __name__ == "__main__":
    main()
