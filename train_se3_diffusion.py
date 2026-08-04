#!/usr/bin/env python3
"""Prepare data, train three SE(3) diffusion models, and sample four variants."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from se3_diffusion import (
    DiffusionSchedule,
    GuidanceConfig,
    PathDataset,
    PreparedData,
    build_model,
    ddim_sample,
    extract,
    hard_clamp_endpoints,
    pose9_to_pose7_numpy,
    prepare_dataset,
    robot_obstacle_signed_separations,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "datasets/diffusion_se3_multihomotopy_v002_300"
DEFAULT_OUTPUT = PROJECT_DIR / "results/diffusion_se3_three_stage_v002"
MODEL_TYPES = ("unet", "dit", "dit_cross")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    records = list(manifest.get("trajectories", []))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"dataset is not complete: {manifest.get('status')!r}")
    requested = (
        int(manifest["requested_pair_count"])
        * int(manifest["requested_trajectories_per_pair"])
    )
    if len(records) != requested:
        raise RuntimeError(f"dataset has {len(records)} accepted paths, expected {requested}")
    counts: dict[int, int] = {}
    topology_mismatches = 0
    for record in records:
        pair = int(record["pair_index"])
        counts[pair] = counts.get(pair, 0) + 1
        expected = record.get("expected_topology")
        if expected and (
            record.get("planned_topology") != expected
            or record.get("actual_topology") != expected
        ):
            topology_mismatches += 1
    per_pair_expected = int(manifest["requested_trajectories_per_pair"])
    if any(value != per_pair_expected for value in counts.values()):
        raise RuntimeError("at least one start/goal pair has an incomplete trajectory group")
    if topology_mismatches:
        raise RuntimeError(f"dataset contains {topology_mismatches} topology mismatches")
    pair_payloads = [
        json.loads(
            (dataset_root / "pairs" / f"pair_{pair:03d}" / "pair.json").read_text(
                encoding="utf-8"
            )
        )
        for pair in sorted(counts)
    ]
    starts = np.asarray([payload["start_pose"][:3] for payload in pair_payloads])
    goals = np.asarray([payload["goal_pose"][:3] for payload in pair_payloads])
    condition_positions = np.concatenate((starts, goals), axis=1)
    pairwise = np.linalg.norm(
        condition_positions[:, None] - condition_positions[None], axis=-1
    )
    pairwise += np.eye(len(pairwise)) * 1e9
    def point_statistics(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean_xyz": values.mean(axis=0).tolist(),
            "std_xyz": values.std(axis=0).tolist(),
            "minimum_xyz": values.min(axis=0).tolist(),
            "maximum_xyz": values.max(axis=0).tolist(),
        }
    def statistics(name: str) -> dict[str, float]:
        values = np.asarray([float(record[name]) for record in records])
        return {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(np.max(values)),
        }
    return {
        "accepted_trajectory_count": len(records),
        "unique_start_goal_pair_count": len(counts),
        "trajectories_per_pair": sorted(set(counts.values())),
        "topology_signature_counts": manifest.get("topology_signature_counts", {}),
        "start_position_distribution": point_statistics(starts),
        "goal_position_distribution": point_statistics(goals),
        "minimum_pairwise_start_goal_position_condition_distance_m": float(
            np.min(pairwise)
        ),
        "topology_mismatch_count": topology_mismatches,
        "diversity_relaxed_count": sum(bool(record.get("diversity_relaxed")) for record in records),
        "path_length_m": statistics("path_length_m"),
        "path_rotation_deg": statistics("path_rotation_deg"),
        "position_rmse_m": statistics("position_rmse_m"),
        "attitude_rmse_deg": statistics("attitude_rmse_deg"),
        "minimum_planned_physical_clearance_m": statistics(
            "minimum_planned_physical_clearance_m"
        ),
        "minimum_actual_physical_clearance_m": statistics(
            "minimum_actual_physical_clearance_m"
        ),
    }


@torch.no_grad()
def audit_guidance_proxy(prepared: PreparedData, dataset_root: Path) -> dict[str, Any]:
    """Catch collision surrogates that disagree with safe demonstrations."""

    original = np.flatnonzero(~prepared.reversed_flags)
    paths = torch.from_numpy(prepared.paths[original]).float()
    obstacles = torch.from_numpy(prepared.obstacles).float()
    obstacle_mask = torch.from_numpy(prepared.obstacle_mask).bool()
    proxy_values: list[float] = []
    for batch in paths.split(24):
        proxy_values.extend(
            robot_obstacle_signed_separations(
                batch, obstacles, obstacle_mask
            ).amin(dim=(1, 2, 3)).cpu().numpy().tolist()
        )
    proxy = np.asarray(proxy_values, dtype=np.float64)
    manifest = json.loads(
        (dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    exact = np.asarray([
        float(record["minimum_planned_physical_clearance_m"])
        for record in manifest["trajectories"]
    ])
    if len(proxy) != len(exact):
        raise RuntimeError("guidance proxy audit sample count mismatch")
    physical_free_rate = float(np.mean(proxy > 0.0))
    if physical_free_rate < 0.90:
        raise RuntimeError(
            "guidance collision proxy rejects too many COAL-safe demonstrations: "
            f"{100 * physical_free_rate:.1f}% proxy-free"
        )
    return {
        "sample_count": len(proxy),
        "proxy_physical_free_rate": physical_free_rate,
        "proxy_clearance_006_rate": float(np.mean(proxy > 0.06)),
        "minimum_proxy_clearance_m": float(np.min(proxy)),
        "p05_proxy_clearance_m": float(np.quantile(proxy, 0.05)),
        "median_proxy_clearance_m": float(np.median(proxy)),
        "mean_proxy_clearance_m": float(np.mean(proxy)),
        "mean_exact_planned_clearance_m": float(np.mean(exact)),
        "proxy_exact_pearson_correlation": float(np.corrcoef(proxy, exact)[0, 1]),
        "note": (
            "The SAT proxy is conservative and is used only for gradient direction; "
            "full-URDF COAL remains the acceptance certificate."
        ),
    }


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.values = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if self.values[name].is_floating_point():
                self.values[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.values[name].copy_(value.detach())

    def cpu_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.values.items()}


def fixed_endpoint_diffusion_loss(
    model: nn.Module,
    clean: Tensor,
    conditions: Tensor,
    schedule: DiffusionSchedule,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    batch = len(clean)
    device = clean.device
    timesteps = torch.randint(
        0, len(schedule.betas), (batch,), device=device, generator=generator
    )
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=device, generator=generator)
    noise[:, 0] = 0.0
    noise[:, -1] = 0.0
    alpha_bar = extract(schedule.alpha_bars, timesteps, clean.shape)
    noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
    noisy = hard_clamp_endpoints(noisy, conditions)
    predicted_velocity = model(
        noisy, timesteps, conditions,
        obstacles[None].expand(batch, -1, -1),
        obstacle_mask[None].expand(batch, -1),
    )
    target_velocity = (
        torch.sqrt(alpha_bar) * noise
        - torch.sqrt(1.0 - alpha_bar) * clean
    )
    return (
        predicted_velocity[:, 1:-1] - target_velocity[:, 1:-1]
    ).square().mean()


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for clean, conditions in loader:
        clean = clean.to(device)
        conditions = conditions.to(device)
        losses.append(float(fixed_endpoint_diffusion_loss(
            model, clean, conditions, schedule, obstacles, obstacle_mask, generator
        )))
    model.train()
    return float(np.mean(losses))


def count_parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)


def train_one_model(
    model_type: str,
    prepared: PreparedData,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    seed_everything(args.seed + MODEL_TYPES.index(model_type) * 1009)
    model = build_model(
        model_type,
        sequence_length=args.sequence_length,
        unet_channels=args.unet_channels,
        dit_dimension=args.dit_dimension,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
    ).to(device)
    train_dataset = PathDataset(prepared, 0)
    validation_dataset = PathDataset(prepared, 1)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=False, generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    iterator = iter(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup_steps = min(args.warmup_steps, max(1, args.training_steps // 10))
    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(args.training_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.minimum_learning_rate_ratio + (
            1.0 - args.minimum_learning_rate_ratio
        ) * cosine
    learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, learning_rate_multiplier
    )
    schedule = DiffusionSchedule.cosine(args.diffusion_steps, device)
    obstacles = torch.from_numpy(prepared.obstacles).float().to(device)
    obstacle_mask = torch.from_numpy(prepared.obstacle_mask).bool().to(device)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp else nullcontext()
    )
    model_output = output_dir / "models" / model_type
    model_output.mkdir(parents=True, exist_ok=True)
    best_path = model_output / "best.pt"
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_step = 0
    started = time.monotonic()
    rolling_loss = 0.0
    rolling_count = 0
    validation_intervals_without_improvement = 0
    model.train()
    for step in range(1, args.training_steps + 1):
        try:
            clean, conditions = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            clean, conditions = next(iterator)
        clean = clean.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            loss = fixed_endpoint_diffusion_loss(
                model, clean, conditions, schedule, obstacles, obstacle_mask
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        learning_rate_scheduler.step()
        ema.update(model)
        rolling_loss += float(loss.detach())
        rolling_count += 1
        if step % args.validation_interval == 0 or step == args.training_steps:
            live_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.load_state_dict(ema.values)
            validation = validation_loss(
                model, validation_loader, schedule, obstacles, obstacle_mask,
                device, args.seed + 777,
            )
            model.load_state_dict(live_state)
            entry = {
                "step": step,
                "training_loss": rolling_loss / max(rolling_count, 1),
                "validation_loss": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
            }
            history.append(entry)
            print(
                f"[{model_type}] step={step:05d}/{args.training_steps} "
                f"train={entry['training_loss']:.5f} val={validation:.5f} "
                f"elapsed={entry['elapsed_s']:.1f}s",
                flush=True,
            )
            rolling_loss = 0.0
            rolling_count = 0
            if validation < best_validation:
                best_validation = validation
                best_step = step
                validation_intervals_without_improvement = 0
                torch.save({
                    "model_type": model_type,
                    "model_state": ema.cpu_state_dict(),
                    "architecture": {
                        "sequence_length": args.sequence_length,
                        "unet_channels": args.unet_channels,
                        "dit_dimension": args.dit_dimension,
                        "dit_depth": args.dit_depth,
                        "dit_heads": args.dit_heads,
                    },
                    "diffusion_steps": args.diffusion_steps,
                    "prediction_type": "velocity",
                    "training_step": step,
                    "validation_loss": validation,
                    "path_mean": prepared.path_mean,
                    "path_std": prepared.path_std,
                    "source_dataset": prepared.source_dataset,
                }, best_path)
            else:
                validation_intervals_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and validation_intervals_without_improvement
                >= args.early_stopping_patience
            ):
                print(
                    f"[{model_type}] early stop after "
                    f"{validation_intervals_without_improvement} validation "
                    "intervals without improvement",
                    flush=True,
                )
                break
    write_json(model_output / "training_history.json", {
        "model_type": model_type,
        "parameter_count": count_parameters(model),
        "train_sample_count_with_reverse_augmentation": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "best_validation_loss": best_validation,
        "best_training_step": best_step,
        "wall_time_s": time.monotonic() - started,
        "device": str(device),
        "amp": use_amp,
        "history": history,
    })
    return best_path


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("prediction_type") != "velocity":
        raise RuntimeError(
            f"{checkpoint_path} is not a velocity-prediction checkpoint; retrain it"
        )
    architecture = checkpoint["architecture"]
    model = build_model(checkpoint["model_type"], **architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def split_conditions(prepared: PreparedData, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    split_code = {"validation": 1, "test": 2}[split_name]
    indices = np.flatnonzero(
        (prepared.split_codes == split_code) & (~prepared.reversed_flags)
    )
    pairs = np.unique(prepared.pair_indices[indices])
    conditions = []
    for pair in pairs:
        selected = indices[prepared.pair_indices[indices] == pair]
        conditions.append(prepared.conditions[selected[0]])
    return pairs, np.stack(conditions)


def sample_variant(
    variant: str,
    model_type: str,
    checkpoint_path: Path,
    prepared: PreparedData,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    guided: bool,
) -> Path:
    model, checkpoint = load_model(checkpoint_path, device)
    pairs, raw_conditions = split_conditions(prepared, args.sampling_split)
    repeated_conditions = np.repeat(raw_conditions, args.samples_per_test_pair, axis=0)
    repeated_pairs = np.repeat(pairs, args.samples_per_test_pair)
    condition_mean = np.tile(prepared.path_mean, 2)
    condition_std = np.tile(prepared.path_std, 2)
    normalized_conditions = torch.from_numpy(
        (repeated_conditions - condition_mean) / condition_std
    ).float().to(device)
    schedule = DiffusionSchedule.cosine(checkpoint["diffusion_steps"], device)
    obstacles = torch.from_numpy(prepared.obstacles).float().to(device)
    obstacle_mask = torch.from_numpy(prepared.obstacle_mask).bool().to(device)
    environment = json.loads(Path(prepared.environment_path).read_text(encoding="utf-8"))
    sampling_bounds = environment["sampling_space"]["position_bounds"]
    bounds_min = torch.as_tensor(sampling_bounds["min"], dtype=torch.float32, device=device)
    bounds_max = torch.as_tensor(sampling_bounds["max"], dtype=torch.float32, device=device)
    mean = torch.from_numpy(prepared.path_mean).float().to(device)[None, None]
    std = torch.from_numpy(prepared.path_std).float().to(device)[None, None]
    guidance = GuidanceConfig(
        enabled=guided,
        start_fraction=args.guidance_fraction,
        scale=args.guidance_scale,
        steps_per_diffusion_step=args.guidance_steps,
        max_perturbation=args.guidance_max_perturbation,
        clearance_m=args.guidance_clearance,
    )
    # Reset to the same latent-noise stream for every variant. In particular,
    # this makes guided-vs-unguided U-Net a controlled inference-only ablation.
    generator = torch.Generator(device=device).manual_seed(args.seed + 9000)
    # Exclude one-time CUDA/kernel initialization from the latency comparison.
    warmup_generator = torch.Generator(device=device).manual_seed(args.seed + 8999)
    _ = ddim_sample(
        model, normalized_conditions[:1], schedule, args.sequence_length,
        min(5, args.ddim_steps), obstacles, obstacle_mask, mean, std,
        bounds_min, bounds_max, guidance, warmup_generator,
        clip_x0=args.sample_clip_x0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    batches = []
    for begin in range(0, len(normalized_conditions), args.sampling_batch_size):
        end = begin + args.sampling_batch_size
        batches.append(ddim_sample(
            model, normalized_conditions[begin:end], schedule,
            args.sequence_length, args.ddim_steps,
            obstacles, obstacle_mask, mean, std, bounds_min, bounds_max,
            guidance, generator,
            clip_x0=args.sample_clip_x0,
        ).cpu().numpy())
    elapsed = time.monotonic() - started
    normalized_paths = np.concatenate(batches)
    paths9 = normalized_paths * prepared.path_std + prepared.path_mean
    paths7 = pose9_to_pose7_numpy(paths9).astype(np.float32)
    paths7[:, 0] = pose9_to_pose7_numpy(repeated_conditions[:, :9])
    paths7[:, -1] = pose9_to_pose7_numpy(repeated_conditions[:, 9:])
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{variant}.npz"
    np.savez_compressed(
        prediction_path,
        poses_wxyz=paths7,
        pair_indices=repeated_pairs,
        sample_indices=np.tile(np.arange(args.samples_per_test_pair), len(pairs)),
        start_goal_pose9=repeated_conditions,
        variant=np.asarray(variant),
        model_type=np.asarray(model_type),
        guided=np.asarray(guided),
        total_sampling_time_s=np.asarray(elapsed),
        mean_sampling_time_s=np.asarray(elapsed / len(paths7)),
        guidance_json=np.asarray(json.dumps(guidance.__dict__)),
        sampling_split=np.asarray(args.sampling_split),
    )
    print(
        f"[{variant}] sampled {len(paths7)} paths in {elapsed:.2f}s "
        f"({1000 * elapsed / len(paths7):.1f}ms/path)", flush=True,
    )
    return prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="replace existing best.pt checkpoints instead of resuming from them",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=list(MODEL_TYPES))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--no-reverse-augmentation", action="store_true")
    parser.add_argument("--training-steps", type=int, default=4000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=25)
    parser.add_argument("--unet-channels", type=int, default=64)
    parser.add_argument("--dit-dimension", type=int, default=128)
    parser.add_argument("--dit-depth", type=int, default=4)
    parser.add_argument("--dit-heads", type=int, default=4)
    parser.add_argument("--samples-per-test-pair", type=int, default=32)
    parser.add_argument(
        "--sampling-split", choices=("validation", "test"), default="test",
        help="endpoint-pair split used for generated prediction files",
    )
    parser.add_argument("--sampling-batch-size", type=int, default=96)
    parser.add_argument(
        "--sample-clip-x0", type=float, default=4.0,
        help="clean-sample clipping in training-standardized coordinates",
    )
    parser.add_argument("--guidance-fraction", type=float, default=0.40)
    parser.add_argument("--guidance-scale", type=float, default=0.020)
    parser.add_argument("--guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-max-perturbation", type=float, default=0.12)
    parser.add_argument("--guidance-clearance", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    if args.sequence_length < 16 or args.sequence_length % 4:
        parser.error("sequence length must be >=16 and divisible by 4")
    if args.training_steps <= 0 or args.batch_size <= 0:
        parser.error("training steps and batch size must be positive")
    if args.validation_interval <= 0 or args.early_stopping_patience < 0:
        parser.error("validation interval must be positive and patience non-negative")
    if args.warmup_steps < 0 or not 0.0 <= args.minimum_learning_rate_ratio <= 1.0:
        parser.error("warmup must be non-negative and minimum LR ratio in [0, 1]")
    if args.train_only and args.sample_only:
        parser.error("--train-only and --sample-only are mutually exclusive")
    if not 0.0 < args.guidance_fraction <= 1.0:
        parser.error("guidance fraction must be in (0, 1]")
    if args.sample_clip_x0 <= 0.0:
        parser.error("sample clip must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(args.dataset.resolve())
    write_json(args.output_dir / "dataset_quality.json", dataset_audit)
    cache_path = args.output_dir / "prepared_dataset.npz"
    if cache_path.exists():
        prepared = PreparedData.load(cache_path)
        if prepared.source_dataset != str(args.dataset.resolve()):
            raise RuntimeError("prepared cache belongs to a different source dataset")
        if prepared.paths.shape[1] != args.sequence_length:
            raise RuntimeError(
                "prepared cache sequence length differs from --sequence-length; "
                "use another output directory"
            )
    else:
        prepared = prepare_dataset(
            args.dataset, cache_path, args.sequence_length,
            reverse_train=not args.no_reverse_augmentation,
        )
    guidance_proxy_audit = audit_guidance_proxy(
        prepared, args.dataset.resolve()
    )
    write_json(
        args.output_dir / "guidance_proxy_audit.json",
        guidance_proxy_audit,
    )
    split_counts = {
        name: int(np.count_nonzero(prepared.split_codes == code))
        for name, code in (("train", 0), ("validation", 1), ("test", 2))
    }
    run_config = {
        "arguments": vars(args) | {"dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "split_sample_counts": split_counts,
        "unique_pair_counts": {
            name: int(len(np.unique(prepared.pair_indices[prepared.split_codes == code])))
            for name, code in (("train", 0), ("validation", 1), ("test", 2))
        },
        "representation": (
            f"{args.sequence_length} x (xyz + continuous SO(3) 6D), "
            "endpoints hard-clamped"
        ),
        "diffusion_prediction_type": "velocity",
        "reverse_augmentation_train_only": not args.no_reverse_augmentation,
    }
    # Sampling-only reruns are used for validation tuning and final frozen
    # guidance. Do not overwrite the configuration that produced checkpoints.
    config_name = "sampling_config.json" if args.sample_only else "experiment_config.json"
    write_json(args.output_dir / config_name, run_config)
    if args.prepare_only:
        print(f"Prepared dataset: {cache_path}")
        return
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and cuda_available) else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    print(
        f"device={device} source_samples={len(prepared.paths)} splits={split_counts}",
        flush=True,
    )
    checkpoints: dict[str, Path] = {}
    for model_type in args.models:
        checkpoint = args.output_dir / "models" / model_type / "best.pt"
        if args.sample_only:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        elif checkpoint.exists() and not args.force_retrain:
            metadata = torch.load(checkpoint, map_location="cpu")
            if metadata.get("source_dataset") != prepared.source_dataset:
                raise RuntimeError(f"existing {checkpoint} belongs to another dataset")
            print(f"[{model_type}] reusing existing checkpoint {checkpoint}", flush=True)
        else:
            checkpoint = train_one_model(
                model_type, prepared, args.output_dir, args, device
            )
        checkpoints[model_type] = checkpoint
    if args.train_only:
        return
    variant_specs = (
        ("unet_no_guidance", "unet", False),
        ("unet_guidance", "unet", True),
        ("dit_no_guidance", "dit", False),
        ("dit_cross_environment", "dit_cross", False),
    )
    for variant, model_type, guided in variant_specs:
        if model_type not in checkpoints:
            continue
        sample_variant(
            variant, model_type, checkpoints[model_type], prepared,
            args.output_dir, args, device, guided,
        )


if __name__ == "__main__":
    main()
