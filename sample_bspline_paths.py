#!/usr/bin/env python3
"""Sample B-spline control-point trajectories with optional ESDF guidance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from bspline_control import evaluate_torch
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-pose", type=float, nargs=7)
    parser.add_argument("--goal-pose", type=float, nargs=7)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--sampling-batch-size", type=int, default=32)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-guidance", action="store_true")
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--sample-clip-x0", type=float, default=4.0)
    parser.add_argument("--esdf", type=Path)
    parser.add_argument("--spheres", type=Path, default=DEFAULT_SPHERES)
    parser.add_argument("--evaluation-points", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        else None
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
        enabled=not args.no_guidance,
        start_fraction=args.guidance_fraction,
        scale=args.guidance_scale,
        steps_per_diffusion_step=args.guidance_steps,
        max_perturbation=args.guidance_max_perturbation,
        clearance_m=args.guidance_clearance,
    )
    basis_device = device

    def evaluate_fn(control_points: torch.Tensor) -> torch.Tensor:
        return evaluate_torch(
            control_points,
            n_points=args.evaluation_points,
            degree=int(checkpoint.get("bspline_degree", 5)),
        )

    if args.start_pose is None or args.goal_pose is None:
        raise SystemExit("--start-pose and --goal-pose are required")
    start = np.asarray(args.start_pose, dtype=np.float64)
    goal = np.asarray(args.goal_pose, dtype=np.float64)
    pose9 = pose7_to_pose9(np.stack((start, goal))).astype(np.float32)
    raw_condition = np.concatenate((pose9[0], pose9[1]))
    conditions = torch.from_numpy(
        (np.repeat(raw_condition[None], args.candidate_count, 0)
         - condition_mean) / condition_std
    ).float().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        warmup_generator = torch.Generator(device=device).manual_seed(
            args.seed + 1
        )
        _ = ddim_sample(
            model, conditions[:1], schedule, sequence_length,
            min(5, args.ddim_steps), obstacles, obstacle_mask, mean, std,
            bounds_min, bounds_max, guidance, warmup_generator,
            clip_x0=args.sample_clip_x0, esdf=esdf,
            robot_sphere_centers=sphere_centers,
            robot_sphere_radii=sphere_radii, evaluate_fn=evaluate_fn,
        )
        torch.cuda.synchronize(device)
    started = time.monotonic()
    normalized_batches = []
    for begin in range(0, args.candidate_count, args.sampling_batch_size):
        end = min(begin + args.sampling_batch_size, args.candidate_count)
        normalized_batches.append(
            ddim_sample(
                model, conditions[begin:end], schedule, sequence_length,
                args.ddim_steps, obstacles, obstacle_mask, mean, std,
                bounds_min, bounds_max, guidance, generator,
                clip_x0=args.sample_clip_x0, esdf=esdf,
                robot_sphere_centers=sphere_centers,
                robot_sphere_radii=sphere_radii, evaluate_fn=evaluate_fn,
            ).cpu().numpy()
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    normalized = np.concatenate(normalized_batches)
    control_points = normalized * prepared.path_std + prepared.path_mean
    control_points[:, 0] = pose9[0].astype(np.float32)
    control_points[:, -1] = pose9[1].astype(np.float32)
    dense9 = np.einsum(
        "bcd,nc->bnd",
        control_points,
        __import__("bspline_control").basis_matrix(
            sequence_length,
            int(checkpoint.get("bspline_degree", 5)),
            args.evaluation_points,
        ),
    )
    paths7 = pose9_to_pose7_numpy(dense9).astype(np.float32)
    paths7[:, 0] = start.astype(np.float32)
    paths7[:, -1] = goal.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        poses_wxyz=paths7,
        control_points=control_points.astype(np.float32),
        start_pose=start,
        goal_pose=goal,
        seed=np.asarray(args.seed, dtype=np.int64),
        sampling_time_s=np.asarray(elapsed),
        checkpoint=np.asarray(str(args.checkpoint.resolve())),
        representation=np.asarray("bspline_control_points"),
        guidance_json=np.asarray(json.dumps(guidance.__dict__)),
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "candidate_count": len(paths7),
        "device": str(device),
        "sampling_time_s": elapsed,
        "mean_sampling_time_ms": 1000.0 * elapsed / len(paths7),
    }))


if __name__ == "__main__":
    main()
