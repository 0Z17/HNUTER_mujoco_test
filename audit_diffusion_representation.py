#!/usr/bin/env python3
"""Certify the fixed-length path representation before diffusion training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from coal_collision import CoalCollisionChecker
from collect_diffusion_dataset import _topology_signature
from evaluate_se3_diffusion import dense_path, resample_se3_path, write_json
from run_overfit_cube_single_pipeline import load_environment


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=PROJECT_DIR / "datasets/diffusion_se3_multihomotopy_v002_300",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--required-clearance-m", type=float, default=0.08)
    parser.add_argument("--translation-check-step", type=float, default=0.04)
    parser.add_argument("--rotation-check-step-deg", type=float, default=3.0)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("dataset must be complete before representation audit")
    config = json.loads((args.dataset / "dataset_config.json").read_text(encoding="utf-8"))
    environment_path = args.dataset / config["sources"]["environment"]["copy"]
    environment_data, environment = load_environment(environment_path)
    checker = CoalCollisionChecker.from_urdf(
        Path(config["sources"]["urdf"]["source"]), environment,
        safety_margin=0.0,
    )
    cuts = environment_data.get("topology_cuts", [])
    records = []
    started = time.monotonic()
    for index, record in enumerate(manifest["trajectories"]):
        with np.load(args.dataset / record["training_sample"]) as sample:
            compressed = resample_se3_path(
                sample["smoothed_path_states"], args.sequence_length
            )
        dense = dense_path(
            compressed, args.translation_check_step,
            args.rotation_check_step_deg,
        )
        clearance = checker.clearance(dense[:, :3], dense[:, 3:7])
        signature = _topology_signature(dense, cuts)
        minimum = float(np.min(clearance))
        records.append({
            "pair_index": int(record["pair_index"]),
            "trajectory_index": int(record["trajectory_index"]),
            "minimum_physical_clearance_m": minimum,
            "physical_collision_free": minimum > 0.0,
            "required_clearance_satisfied": minimum > args.required_clearance_m,
            "expected_topology": record.get("expected_topology"),
            "compressed_topology": signature,
            "topology_matches": signature == record.get("expected_topology"),
        })
        if (index + 1) % 25 == 0:
            print(
                f"representation audit {index + 1}/{len(manifest['trajectories'])} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    clearance_values = np.asarray([
        record["minimum_physical_clearance_m"] for record in records
    ])
    summary = {
        "dataset": str(args.dataset.resolve()),
        "sequence_length": args.sequence_length,
        "translation_check_step_m": args.translation_check_step,
        "rotation_check_step_deg": args.rotation_check_step_deg,
        "required_clearance_m": args.required_clearance_m,
        "sample_count": len(records),
        "physical_collision_free_count": sum(
            record["physical_collision_free"] for record in records
        ),
        "required_clearance_satisfied_count": sum(
            record["required_clearance_satisfied"] for record in records
        ),
        "topology_match_count": sum(record["topology_matches"] for record in records),
        "minimum_clearance_m": float(np.min(clearance_values)),
        "mean_clearance_m": float(np.mean(clearance_values)),
        "p05_clearance_m": float(np.quantile(clearance_values, 0.05)),
        "wall_time_s": time.monotonic() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if summary["physical_collision_free_count"] != len(records):
        raise RuntimeError("compressed representation introduced a physical collision")
    if summary["required_clearance_satisfied_count"] != len(records):
        raise RuntimeError(
            "compressed representation violated the required clearance; "
            "increase --sequence-length"
        )
    if summary["topology_match_count"] != len(records):
        raise RuntimeError("compressed representation changed a route topology")


if __name__ == "__main__":
    main()

