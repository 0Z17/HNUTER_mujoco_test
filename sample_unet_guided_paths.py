#!/usr/bin/env python3
"""Generate U-Net + guidance SE(3) candidates for one explicit pose pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

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


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = PROJECT_DIR / "results/diffusion_se3_three_stage_v002"
DEFAULT_SPHERES = (
    PROJECT_DIR
    / "etc"
    / "URDF-for-gazebo"
    / "config"
    / "HDJQR-0102-0055.SLDASM_curobo_spheres.yml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=DEFAULT_EXPERIMENT / "prepared_dataset.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_EXPERIMENT / "models/unet/best.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-pose", type=float, nargs=7, required=True)
    parser.add_argument("--goal-pose", type=float, nargs=7, required=True)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--sampling-batch-size", type=int, default=32)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--sample-clip-x0", type=float, default=4.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--esdf",
        type=Path,
        help=(
            "cached ESDF NPZ; when omitted a grid is built from the "
            "environment and cached under /tmp"
        ),
    )
    parser.add_argument("--esdf-resolution", type=float, default=0.025)
    parser.add_argument("--robot-spheres", type=Path, default=DEFAULT_SPHERES)
    args = parser.parse_args()
    if args.candidate_count <= 0 or args.sampling_batch_size <= 0:
        parser.error("candidate count and batch size must be positive")
    if args.ddim_steps < 2 or args.seed < 0:
        parser.error("DDIM steps must be >=2 and seed non-negative")
    return args


def _load_or_build_esdf(
    environment: dict,
    explicit_path: Path | None,
    resolution: float,
    device: torch.device,
) -> EsdfDistanceField:
    """Load a cached ESDF or build and cache one for the environment."""

    import tempfile

    if explicit_path is not None:
        cache_path = explicit_path.expanduser().resolve()
    else:
        environment_id = environment.get("environment_id", "environment")
        cache_path = Path(tempfile.gettempdir()) / (
            f"esdf_{environment_id}_{resolution:.3f}.npz"
        )
    if cache_path.exists():
        field = EsdfDistanceField.load_cache(cache_path)
    else:
        started = time.monotonic()
        field = EsdfDistanceField.from_environment(
            environment, resolution=resolution
        )
        field.save_cache(cache_path)
        print(
            f"built ESDF grid {field.grid.shape} at {resolution} m in "
            f"{time.monotonic() - started:.1f}s -> {cache_path}",
            flush=True,
        )
    return field.to(device)


def main() -> None:
    args = parse_args()
    cuda = torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and cuda)
        else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    prepared = PreparedData.load(args.prepared.resolve())
    model, checkpoint = load_model(args.checkpoint.resolve(), device)
    environment = json.loads(
        Path(prepared.environment_path).read_text(encoding="utf-8")
    )
    esdf = _load_or_build_esdf(
        environment,
        args.esdf,
        args.esdf_resolution,
        device,
    )
    import yaml

    sphere_payload = yaml.safe_load(
        args.robot_spheres.expanduser().resolve().read_text(encoding="utf-8")
    )
    sphere_items = sphere_payload["collision_spheres"]["base_link"]
    robot_sphere_centers = torch.from_numpy(
        np.asarray(
            [item["center"] for item in sphere_items], dtype=np.float32
        )
    ).to(device)
    robot_sphere_radii = torch.from_numpy(
        np.asarray(
            [item["radius"] for item in sphere_items], dtype=np.float32
        )
    ).to(device)
    sequence_length = int(checkpoint["architecture"]["sequence_length"])
    start = np.asarray(args.start_pose, dtype=np.float64)
    goal = np.asarray(args.goal_pose, dtype=np.float64)
    pose9 = pose7_to_pose9(np.stack((start, goal))).astype(np.float32)
    raw_condition = np.concatenate((pose9[0], pose9[1]))
    repeated = np.repeat(raw_condition[None], args.candidate_count, axis=0)
    condition_mean = np.tile(prepared.path_mean, 2)
    condition_std = np.tile(prepared.path_std, 2)
    conditions = torch.from_numpy(
        (repeated - condition_mean) / condition_std
    ).float().to(device)
    schedule = DiffusionSchedule.cosine(checkpoint["diffusion_steps"], device)
    obstacles = torch.from_numpy(prepared.obstacles).float().to(device)
    obstacle_mask = torch.from_numpy(prepared.obstacle_mask).bool().to(device)
    position_bounds = environment["sampling_space"]["position_bounds"]
    bounds_min = torch.as_tensor(
        position_bounds["min"], dtype=torch.float32, device=device
    )
    bounds_max = torch.as_tensor(
        position_bounds["max"], dtype=torch.float32, device=device
    )
    mean = torch.from_numpy(prepared.path_mean).float().to(device)[None, None]
    std = torch.from_numpy(prepared.path_std).float().to(device)[None, None]
    guidance = GuidanceConfig(
        enabled=True,
        start_fraction=args.guidance_fraction,
        scale=args.guidance_scale,
        steps_per_diffusion_step=args.guidance_steps,
        max_perturbation=args.guidance_max_perturbation,
        clearance_m=args.guidance_clearance,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        warmup_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        _ = ddim_sample(
            model, conditions[:1], schedule, sequence_length,
            min(5, args.ddim_steps), obstacles, obstacle_mask, mean, std,
            bounds_min, bounds_max, guidance, warmup_generator,
            clip_x0=args.sample_clip_x0,
            esdf=esdf,
            robot_sphere_centers=robot_sphere_centers,
            robot_sphere_radii=robot_sphere_radii,
        )
        torch.cuda.synchronize(device)
    started = time.monotonic()
    normalized_batches = []
    for begin in range(0, len(conditions), args.sampling_batch_size):
        end = min(begin + args.sampling_batch_size, len(conditions))
        normalized_batches.append(ddim_sample(
            model, conditions[begin:end], schedule, sequence_length,
            args.ddim_steps, obstacles, obstacle_mask, mean, std,
            bounds_min, bounds_max, guidance, generator,
            clip_x0=args.sample_clip_x0,
            esdf=esdf,
            robot_sphere_centers=robot_sphere_centers,
            robot_sphere_radii=robot_sphere_radii,
        ).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    normalized = np.concatenate(normalized_batches)
    paths9 = normalized * prepared.path_std + prepared.path_mean
    paths7 = pose9_to_pose7_numpy(paths9).astype(np.float32)
    paths7[:, 0] = start
    paths7[:, -1] = goal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        poses_wxyz=paths7,
        start_pose=start,
        goal_pose=goal,
        seed=np.asarray(args.seed, dtype=np.int64),
        sampling_time_s=np.asarray(elapsed),
        mean_sampling_time_s=np.asarray(elapsed / len(paths7)),
        checkpoint=np.asarray(str(args.checkpoint.resolve())),
        prepared_dataset=np.asarray(str(args.prepared.resolve())),
        guidance_json=np.asarray(json.dumps(guidance.__dict__)),
        source=np.asarray("U-Net diffusion + inference guidance"),
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
