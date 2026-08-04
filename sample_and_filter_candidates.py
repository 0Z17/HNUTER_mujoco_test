#!/usr/bin/env python3
"""One-process guided sampling plus cuRobo coarse filtering for SE(3) paths.

Runs in the cuRobo environment (torch >= 2.5 + nvidia-curobo) so that the
diffusion model, the ESDF guidance field, and the cuRobo collision scene share
a single torch/CUDA import.  This removes the per-call cuRobo subprocess
overhead and, in ``--serve`` mode, keeps the model and scene resident across
many start/goal requests.

Outputs per request under ``--output-dir``:
  * diffusion_candidates.npz        (B, N, 7) poses_wxyz
  * curobo_coarse_filter.json       coarse_accept flags + clearance estimates
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Keep Warp's compiled-kernel cache on disk so cold starts do not recompile.
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

from curobo_collision import (
    CuroboBatchChecker,
    load_curobo_spheres,
)


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
    parser.add_argument("--prepared", type=Path, default=DEFAULT_EXPERIMENT / "prepared_dataset.npz")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_EXPERIMENT / "models/unet/best.pt")
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--spheres", type=Path, default=DEFAULT_SPHERES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-pose", type=float, nargs=7)
    parser.add_argument("--goal-pose", type=float, nargs=7)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--sampling-batch-size", type=int, default=32)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--sample-clip-x0", type=float, default=4.0)
    parser.add_argument("--activation-distance", type=float, default=0.08)
    parser.add_argument("--esdf", type=Path)
    parser.add_argument("--esdf-resolution", type=float, default=0.025)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="keep the model/scene resident and serve JSON requests on stdin",
    )
    return parser.parse_args()


class CandidateGenerator:
    """Resident sampler + cuRobo coarse filter for one environment/model."""

    def __init__(self, args: argparse.Namespace) -> None:
        cuda = torch.cuda.is_available()
        if args.device == "cuda" and not cuda:
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(
            "cuda" if args.device == "cuda" or (args.device == "auto" and cuda)
            else "cpu"
        )
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self.args = args
        self.prepared = PreparedData.load(args.prepared.resolve())
        self.model, self.checkpoint = load_model(args.checkpoint.resolve(), self.device)
        self.sequence_length = int(self.checkpoint["architecture"]["sequence_length"])
        self.schedule = DiffusionSchedule.cosine(
            self.checkpoint["diffusion_steps"], self.device
        )
        self.environment = json.loads(
            Path(self.prepared.environment_path).read_text(encoding="utf-8")
        )
        self.esdf = self._load_or_build_esdf()
        sphere_set = load_curobo_spheres(args.spheres.expanduser().resolve())
        self.sphere_centers = torch.from_numpy(sphere_set.centers).to(self.device)
        self.sphere_radii = torch.from_numpy(sphere_set.radii).to(self.device)
        self.obstacles = torch.from_numpy(self.prepared.obstacles).float().to(self.device)
        self.obstacle_mask = torch.from_numpy(
            self.prepared.obstacle_mask
        ).bool().to(self.device)
        bounds = self.environment["sampling_space"]["position_bounds"]
        self.bounds_min = torch.as_tensor(
            bounds["min"], dtype=torch.float32, device=self.device
        )
        self.bounds_max = torch.as_tensor(
            bounds["max"], dtype=torch.float32, device=self.device
        )
        self.mean = torch.from_numpy(self.prepared.path_mean).float().to(self.device)[None, None]
        self.std = torch.from_numpy(self.prepared.path_std).float().to(self.device)[None, None]
        self.condition_mean = np.tile(self.prepared.path_mean, 2)
        self.condition_std = np.tile(self.prepared.path_std, 2)
        self.guidance = GuidanceConfig(
            enabled=True,
            start_fraction=args.guidance_fraction,
            scale=args.guidance_scale,
            steps_per_diffusion_step=args.guidance_steps,
            max_perturbation=args.guidance_max_perturbation,
            clearance_m=args.guidance_clearance,
        )
        self.checker = CuroboBatchChecker(
            self.environment,
            sphere_set,
            device=str(self.device) if self.device.type == "cuda" else "cpu",
            activation_distance=args.activation_distance,
        )
        self._warmup()

    def _load_or_build_esdf(self) -> EsdfDistanceField:
        explicit = self.args.esdf
        if explicit is not None:
            cache_path = explicit.expanduser().resolve()
        else:
            environment_id = self.environment.get("environment_id", "environment")
            cache_path = Path(
                os.path.join(
                    "/tmp", f"esdf_{environment_id}_{self.args.esdf_resolution:.3f}.npz"
                )
            )
        if cache_path.exists():
            field = EsdfDistanceField.load_cache(cache_path)
        else:
            started = time.monotonic()
            field = EsdfDistanceField.from_environment(
                self.environment, resolution=self.args.esdf_resolution
            )
            field.save_cache(cache_path)
            print(
                f"built ESDF grid {field.grid.shape} in "
                f"{time.monotonic() - started:.1f}s -> {cache_path}",
                flush=True,
            )
        return field.to(self.device)

    def _warmup(self) -> None:
        if self.device.type != "cuda":
            return
        generator = torch.Generator(device=self.device).manual_seed(0)
        conditions = torch.zeros(1, 18, device=self.device)
        ddim_sample(
            self.model, conditions, self.schedule, self.sequence_length,
            min(5, self.args.ddim_steps), self.obstacles, self.obstacle_mask,
            self.mean, self.std, self.bounds_min, self.bounds_max,
            self.guidance, generator, clip_x0=self.args.sample_clip_x0,
            esdf=self.esdf, robot_sphere_centers=self.sphere_centers,
            robot_sphere_radii=self.sphere_radii,
        )
        torch.cuda.synchronize(self.device)

    def sample_and_filter(
        self,
        start_pose: np.ndarray,
        goal_pose: np.ndarray,
        seed: int,
        candidate_count: int,
        output_dir: Path,
    ) -> dict:
        """Sample guided candidates and run the cuRobo coarse filter."""

        args = self.args
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pose9 = pose7_to_pose9(np.stack((start_pose, goal_pose))).astype(np.float32)
        raw_condition = np.concatenate((pose9[0], pose9[1]))
        conditions = torch.from_numpy(
            (np.repeat(raw_condition[None], candidate_count, 0)
             - self.condition_mean) / self.condition_std
        ).float().to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        started = time.monotonic()
        normalized_batches = []
        for begin in range(0, candidate_count, args.sampling_batch_size):
            end = min(begin + args.sampling_batch_size, candidate_count)
            normalized_batches.append(
                ddim_sample(
                    self.model, conditions[begin:end], self.schedule,
                    self.sequence_length, args.ddim_steps, self.obstacles,
                    self.obstacle_mask, self.mean, self.std, self.bounds_min,
                    self.bounds_max, self.guidance, generator,
                    clip_x0=args.sample_clip_x0, esdf=self.esdf,
                    robot_sphere_centers=self.sphere_centers,
                    robot_sphere_radii=self.sphere_radii,
                ).cpu().numpy()
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sampling_time = time.monotonic() - started
        normalized = np.concatenate(normalized_batches)
        paths9 = normalized * self.prepared.path_std + self.prepared.path_mean
        paths7 = pose9_to_pose7_numpy(paths9).astype(np.float32)
        paths7[:, 0] = start_pose.astype(np.float32)
        paths7[:, -1] = goal_pose.astype(np.float32)
        candidate_file = output_dir / "diffusion_candidates.npz"
        np.savez_compressed(
            candidate_file,
            poses_wxyz=paths7,
            start_pose=start_pose,
            goal_pose=goal_pose,
            seed=np.asarray(seed, dtype=np.int64),
            sampling_time_s=np.asarray(sampling_time),
            checkpoint=np.asarray(str(args.checkpoint.resolve())),
            guidance_json=np.asarray(json.dumps(self.guidance.__dict__)),
            source=np.asarray("U-Net diffusion + inference guidance (ESDF)"),
        )
        filter_started = time.monotonic()
        max_cost = self.checker.max_collision_cost_per_path(paths7)
        clearance = self.checker.path_clearance_estimate(paths7)
        filter_time = time.monotonic() - filter_started
        coarse_accept = (max_cost <= 0.0).tolist()
        filter_payload = {
            "environment": str(args.environment.expanduser().resolve()),
            "spheres": str(args.spheres.expanduser().resolve()),
            "sphere_count": int(self.sphere_centers.shape[0]),
            "activation_distance_m": args.activation_distance,
            "candidate_count": int(paths7.shape[0]),
            "max_collision_cost": max_cost.tolist(),
            "coarse_accept": coarse_accept,
            "min_sphere_clearance_estimate_m": clearance.tolist(),
        }
        filter_path = output_dir / "curobo_coarse_filter.json"
        filter_path.write_text(
            json.dumps(filter_payload, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "output_dir": str(output_dir),
            "candidate_file": str(candidate_file),
            "coarse_filter_file": str(filter_path),
            "candidate_count": int(paths7.shape[0]),
            "sampling_time_s": sampling_time,
            "curobo_filter_time_s": filter_time,
            "coarse_accept": coarse_accept,
            "min_sphere_clearance_estimate_m": clearance.tolist(),
            "coarse_accept_count": int(sum(coarse_accept)),
            "device": str(self.device),
        }


def _request_to_args(args: argparse.Namespace, request: dict) -> dict:
    """Merge per-request fields with the serve-time defaults."""

    merged = vars(args).copy()
    merged.update(request)
    return merged


def main() -> None:
    args = parse_args()
    generator = CandidateGenerator(args)
    if not args.serve:
        if args.output_dir is None or args.start_pose is None or args.goal_pose is None or args.seed is None:
            raise SystemExit(
                "--output-dir/--start-pose/--goal-pose/--seed are required "
                "outside --serve mode"
            )
        result = generator.sample_and_filter(
            np.asarray(args.start_pose, dtype=np.float64),
            np.asarray(args.goal_pose, dtype=np.float64),
            args.seed,
            args.candidate_count,
            args.output_dir,
        )
        print("SAMPLE_FILTER=" + json.dumps(result))
        return

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        try:
            request = json.loads(line)
            merged = _request_to_args(args, request)
            result = generator.sample_and_filter(
                np.asarray(merged["start_pose"], dtype=np.float64),
                np.asarray(merged["goal_pose"], dtype=np.float64),
                int(merged["seed"]),
                int(merged.get("candidate_count", args.candidate_count)),
                Path(merged["output_dir"]),
            )
            print(json.dumps({"ok": True, **result}), flush=True)
        except Exception as error:  # noqa: BLE001 - keep serving next request
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
