#!/usr/bin/env python3
"""Train the B-spline control-point SE(3) diffusion model (MPD-style)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from se3_diffusion import PreparedData
from train_se3_diffusion import train_one_model, write_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREPARED = (
    PROJECT_DIR / "results/bspline_control_points_v001/prepared_bspline.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results/bspline_control_points_v001",
    )
    parser.add_argument("--model-type", choices=("unet", "dit", "dit_cross"), default="unet")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--n-control-points", type=int, default=24)
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--evaluation-points", type=int, default=128)
    parser.add_argument("--unet-channels", type=int, default=64)
    parser.add_argument("--dit-dimension", type=int, default=128)
    parser.add_argument("--dit-depth", type=int, default=4)
    parser.add_argument("--dit-heads", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and cuda)
        else "cpu"
    )
    prepared = PreparedData.load(args.prepared.resolve())
    if prepared.paths.shape[1] != args.sequence_length:
        raise RuntimeError(
            f"prepared paths have {prepared.paths.shape[1]} nodes but "
            f"--sequence-length is {args.sequence_length}"
        )
    train_args = SimpleNamespace(
        sequence_length=args.sequence_length,
        unet_channels=args.unet_channels,
        dit_dimension=args.dit_dimension,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        diffusion_steps=args.diffusion_steps,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        ema_decay=args.ema_decay,
        validation_interval=args.validation_interval,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        disable_amp=args.disable_amp,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_one_model(
        args.model_type, prepared, output_dir, train_args, device
    )
    write_json(output_dir / "experiment_config.json", {
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "representation": {
            "type": "bspline_control_points",
            "n_control_points": args.n_control_points,
            "bspline_degree": args.bspline_degree,
            "evaluation_points": args.evaluation_points,
            "hard_conditioned": ["first_control_point", "last_control_point"],
            "reference": "MPD (Carvalho et al.): degree-5 B-spline, 22 control points, 100 diffusion steps, 128 dense points",
        },
        "checkpoint": str(best_path),
    })
    print(f"BSPLINE_TRAINED={best_path}")


if __name__ == "__main__":
    main()
