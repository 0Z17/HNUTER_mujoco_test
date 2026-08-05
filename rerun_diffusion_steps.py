#!/usr/bin/env python3
"""Rerun replay of the diffusion denoising iterations for one SE(3) path.

Runs the same guided DDIM sampling as the production sampler and records the
denoised trajectory at every iteration step into a Rerun recording:

  * ``world/diffusion_steps/path``       - scrubbable on the ``diffusion_step``
    timeline (solid line, blue -> red over iterations);
  * ``world/diffusion_steps/x0_pred``    - the model's x0 estimate at each step
    (thin translucent line);
  * ``world/diffusion_steps_all/step_NN`` - every iteration as a static,
    colour-graded polyline so the whole convergence fan is visible at once;
  * ``world/final_path``                 - the converged path as a thick line;
  * plots: path length, step-to-step displacement, and ESDF+sphere minimum
    clearance per iteration.

Run with the cuRobo environment (torch + rerun-sdk):
    python rerun_diffusion_steps.py --start-pose ... --goal-pose ... \
        --environment environment_multihomotopy_v002.json --output /tmp/steps.rrd
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

import rerun as rr


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
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--spheres", type=Path, default=DEFAULT_SPHERES)
    parser.add_argument("--start-pose", type=float, nargs=7, required=True)
    parser.add_argument("--goal-pose", type=float, nargs=7, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--sample-clip-x0", type=float, default=4.0)
    parser.add_argument("--esdf", type=Path)
    parser.add_argument("--esdf-resolution", type=float, default=0.025)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
        esdf_path = Path(
            f"/tmp/esdf_{environment_id}_{args.esdf_resolution:.3f}.npz"
        )
    if esdf_path.exists():
        esdf = EsdfDistanceField.load_cache(esdf_path)
    else:
        esdf = EsdfDistanceField.from_environment(
            environment, resolution=args.esdf_resolution
        )
        esdf.save_cache(esdf_path)
    esdf = esdf.to(device)
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

    start = np.asarray(args.start_pose, dtype=np.float64)
    goal = np.asarray(args.goal_pose, dtype=np.float64)
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

    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        warmup_generator = torch.Generator(device=device).manual_seed(
            args.seed + 1
        )
        _ = ddim_sample(
            model, conditions, schedule, sequence_length,
            min(5, args.ddim_steps), obstacles, obstacle_mask, mean, std,
            bounds_min, bounds_max, guidance, warmup_generator,
            clip_x0=args.sample_clip_x0, esdf=esdf,
            robot_sphere_centers=sphere_centers,
            robot_sphere_radii=sphere_radii,
        )
        torch.cuda.synchronize(device)
    started = time.monotonic()
    final = ddim_sample(
        model, conditions, schedule, sequence_length, args.ddim_steps,
        obstacles, obstacle_mask, mean, std, bounds_min, bounds_max,
        guidance, generator, clip_x0=args.sample_clip_x0, esdf=esdf,
        robot_sphere_centers=sphere_centers,
        robot_sphere_radii=sphere_radii, on_step=on_step,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_time = time.monotonic() - started

    def to_poses7(normalized9: np.ndarray) -> np.ndarray:
        paths9 = normalized9 * prepared.path_std + prepared.path_mean
        poses = pose9_to_pose7_numpy(paths9).astype(np.float32)
        poses[0, 0] = start.astype(np.float32)
        poses[0, -1] = goal.astype(np.float32)
        return poses[0]

    def sphere_clearance(positions: np.ndarray, quaternions: np.ndarray) -> float:
        with torch.no_grad():
            pos = torch.from_numpy(positions.astype(np.float32)).to(device)
            quat = torch.from_numpy(quaternions.astype(np.float32)).to(device)
            matrices = np.stack(
                [_quaternion_to_matrix(q) for q in quaternions]
            )
            world = pos[:, None, :] + torch.from_numpy(
                np.einsum("nij,sj->nsi", matrices, sphere_set.centers)
            ).to(device)
            distance = esdf.sample(world) - sphere_radii
            return float(distance.amin())

    final_poses = to_poses7(final.detach().cpu().numpy())
    records = []
    previous: np.ndarray | None = None
    for sample_index, timestep_value, path_np, x0_np in steps:
        poses = to_poses7(path_np)
        positions = poses[:, :3]
        length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        delta = (
            float(np.linalg.norm(positions - previous, axis=1).mean())
            if previous is not None
            else 0.0
        )
        clearance = sphere_clearance(positions, poses[:, 3:7])
        records.append({
            "step": sample_index,
            "timestep": timestep_value,
            "translation_length_m": round(length, 4),
            "step_delta_m": round(delta, 5),
            "min_sphere_clearance_m": round(clearance, 4),
        })
        previous = positions

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else PROJECT_DIR / "results" / f"diffusion_steps_replay_{stamp}.rrd"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def gradient_color(fraction: float) -> tuple[int, int, int, int]:
        blue = np.asarray([30, 90, 230], dtype=np.float32)
        red = np.asarray([225, 60, 40], dtype=np.float32)
        color = blue + (red - blue) * fraction
        return (int(color[0]), int(color[1]), int(color[2]), 220)

    with rr.RecordingStream(
        "diffusion_step_replay",
        recording_id=f"diffusion_steps_seed{args.seed}",
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
                            xyzw=np.roll(
                                box["pose"]["quaternion_wxyz"], -1
                            )
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
                labels=["final denoised path"],
            ),
            static=True,
        )
        step_count = len(steps)
        for sample_index, timestep_value, path_np, x0_np in steps:
            poses = to_poses7(path_np)
            x0_poses = to_poses7(x0_np)
            fraction = sample_index / max(step_count - 1, 1)
            color = gradient_color(fraction)
            recording.log(
                f"world/diffusion_steps_all/step_{sample_index:03d}",
                rr.LineStrips3D(
                    [poses[:, :3]],
                    colors=[color],
                    radii=[0.006],
                    labels=[f"step {sample_index} (t={timestep_value})"],
                    show_labels=False,
                ),
                static=True,
            )
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
                "plots/step_delta",
                rr.Scalars([record["step_delta_m"]]),
            )
            recording.log(
                "plots/min_clearance",
                rr.Scalars([record["min_sphere_clearance_m"]]),
            )

    summary = {
        "output": str(output),
        "seed": args.seed,
        "ddim_steps": len(steps),
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
