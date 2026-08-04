#!/usr/bin/env python3
"""Exact COAL and diversity evaluation for generated SE(3) path variants."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from coal_collision import CoalCollisionChecker
from collect_diffusion_dataset import _signature_key, _topology_signature
from run_overfit_cube_single_pipeline import load_environment


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR / "datasets/diffusion_se3_multihomotopy_v002_300"
DEFAULT_EXPERIMENT = PROJECT_DIR / "results/diffusion_se3_three_stage_v002"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def quaternion_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - fraction) * first + fraction * second
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * angle) * first
        + math.sin(fraction * angle) * second
    ) / math.sin(angle)


def dense_path(path: np.ndarray, translation_step: float, rotation_step_deg: float,
               maximum_points: int = 30000) -> np.ndarray:
    output = [np.asarray(path[0], dtype=np.float64)]
    rotation_step = math.radians(rotation_step_deg)
    for first, second in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(second[:3] - first[:3]))
        angle = float(quaternion_angles(first[None, 3:7], second[None, 3:7])[0])
        count = max(1, int(math.ceil(max(
            distance / translation_step, angle / rotation_step
        ))))
        for index in range(1, count + 1):
            fraction = index / count
            output.append(np.concatenate((
                (1.0 - fraction) * first[:3] + fraction * second[:3],
                slerp(first[3:7], second[3:7], fraction),
            )))
            if len(output) > maximum_points:
                raise ValueError("path requires too many dense collision-check samples")
    return np.stack(output)


def resample_se3_path(path: np.ndarray, count: int = 64,
                      rotation_weight_m_per_rad: float = 0.22) -> np.ndarray:
    quaternions = continuous_quaternions(path[:, 3:7])
    increments = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    increments += rotation_weight_m_per_rad * quaternion_angles(
        quaternions[:-1], quaternions[1:]
    )
    source = np.concatenate(([0.0], np.cumsum(increments)))
    if source[-1] <= 1e-12:
        return np.repeat(path[:1], count, axis=0)
    targets = np.linspace(0.0, source[-1], count)
    result = np.empty((count, 7), dtype=np.float64)
    result[:, :3] = np.column_stack([
        np.interp(targets, source, path[:, axis]) for axis in range(3)
    ])
    for index, target in enumerate(targets):
        right = min(max(int(np.searchsorted(source, target, side="right")), 1), len(path) - 1)
        left = right - 1
        fraction = (target - source[left]) / max(source[right] - source[left], 1e-12)
        result[index, 3:7] = slerp(quaternions[left], quaternions[right], float(fraction))
    result[0] = path[0]
    result[-1] = path[-1]
    return result


def path_metrics(path: np.ndarray) -> dict[str, float]:
    position = path[:, :3]
    quaternion = continuous_quaternions(path[:, 3:7])
    differences = np.diff(position, axis=0)
    acceleration = np.diff(position, n=2, axis=0)
    jerk = np.diff(position, n=3, axis=0)
    return {
        "translation_length_m": float(np.linalg.norm(differences, axis=1).sum()),
        "rotation_length_deg": float(np.degrees(quaternion_angles(quaternion[:-1], quaternion[1:])).sum()),
        "position_acceleration_rms": float(np.sqrt(np.mean(np.sum(acceleration ** 2, axis=1)))),
        "position_jerk_rms": float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))),
    }


def load_experts(dataset: Path, manifest: dict[str, Any]) -> dict[int, list[np.ndarray]]:
    experts: dict[int, list[np.ndarray]] = {}
    for record in manifest["trajectories"]:
        pair = int(record["pair_index"])
        with np.load(dataset / record["training_sample"]) as sample:
            expert = sample["smoothed_path_states"].copy()
        experts.setdefault(pair, []).append(expert)
    return experts


def expert_distance(path: np.ndarray, experts: list[np.ndarray]) -> tuple[float, float]:
    comparisons = []
    for raw_expert in experts:
        expert = resample_se3_path(raw_expert, len(path))
        position = float(np.sqrt(np.mean(np.sum((path[:, :3] - expert[:, :3]) ** 2, axis=1))))
        rotation = float(np.degrees(np.sqrt(np.mean(
            quaternion_angles(path[:, 3:7], expert[:, 3:7]) ** 2
        ))))
        comparisons.append((position, rotation))
    return min(comparisons, key=lambda value: value[0] + 0.01 * value[1])


def pairwise_diversity(paths: np.ndarray, pair_indices: np.ndarray) -> tuple[float, float]:
    position_values: list[float] = []
    rotation_values: list[float] = []
    for pair in np.unique(pair_indices):
        group = paths[pair_indices == pair]
        for first_index in range(len(group)):
            for second_index in range(first_index + 1, len(group)):
                first, second = group[first_index], group[second_index]
                position_values.append(float(np.sqrt(np.mean(np.sum(
                    (first[:, :3] - second[:, :3]) ** 2, axis=1
                )))))
                rotation_values.append(float(np.degrees(np.sqrt(np.mean(
                    quaternion_angles(first[:, 3:7], second[:, 3:7]) ** 2
                )))))
    if not position_values:
        return 0.0, 0.0
    return float(np.mean(position_values)), float(np.mean(rotation_values))


def pair_cluster_confidence_interval(
    records: list[dict[str, Any]], field: str, seed: int = 20260803,
) -> list[float]:
    """Bootstrap whole start/goal tasks so paths from one task stay correlated."""

    pairs = sorted({int(row["pair_index"]) for row in records})
    pair_means = np.asarray([
        np.mean([float(row[field]) for row in records if row["pair_index"] == pair])
        for pair in pairs
    ])
    if len(pair_means) <= 1:
        value = float(pair_means[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    samples = pair_means[rng.integers(0, len(pair_means), size=(2000, len(pair_means)))].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def paired_task_delta(
    records: list[dict[str, Any]], candidate: str, baseline: str, field: str,
    seed: int = 20260804,
) -> dict[str, Any]:
    candidate_rows = [row for row in records if row["variant"] == candidate]
    baseline_rows = [row for row in records if row["variant"] == baseline]
    pairs = sorted(
        {int(row["pair_index"]) for row in candidate_rows}
        & {int(row["pair_index"]) for row in baseline_rows}
    )
    deltas = np.asarray([
        np.mean([float(row[field]) for row in candidate_rows if row["pair_index"] == pair])
        - np.mean([float(row[field]) for row in baseline_rows if row["pair_index"] == pair])
        for pair in pairs
    ])
    rng = np.random.default_rng(seed)
    bootstrap = deltas[
        rng.integers(0, len(deltas), size=(2000, len(deltas)))
    ].mean(axis=1)
    return {
        "candidate_minus_baseline": float(np.mean(deltas)),
        "pair_cluster_bootstrap_95ci": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def aggregate(records: list[dict[str, Any]], paths: np.ndarray,
              pair_indices: np.ndarray, sampling_time: float) -> dict[str, Any]:
    numeric_fields = (
        "minimum_physical_clearance_m", "minimum_008_clearance_m",
        "minimum_010_clearance_m", "translation_length_m", "rotation_length_deg",
        "position_acceleration_rms", "position_jerk_rms",
        "best_expert_position_rms_m", "best_expert_rotation_rms_deg",
        "endpoint_position_error_m", "endpoint_rotation_error_deg",
    )
    result: dict[str, Any] = {
        "sample_count": len(records),
        "physical_collision_free_rate": float(np.mean([row["physical_collision_free"] for row in records])),
        "workspace_valid_rate": float(np.mean([row["workspace_valid"] for row in records])),
        "clearance_008_success_rate": float(np.mean([row["clearance_008_success"] for row in records])),
        "clearance_010_success_rate": float(np.mean([row["clearance_010_success"] for row in records])),
        "known_topology_rate": float(np.mean([row["known_topology"] for row in records])),
        "collision_free_known_topology_rate": float(np.mean([
            row["physical_collision_free"] and row["known_topology"] for row in records
        ])),
        "mean_sampling_time_ms": 1000.0 * sampling_time,
    }
    for field in (
        "physical_collision_free", "clearance_008_success", "known_topology",
        "best_expert_position_rms_m", "translation_length_m",
    ):
        result[f"pair_cluster_bootstrap_95ci_{field}"] = (
            pair_cluster_confidence_interval(records, field)
        )
    for field in numeric_fields:
        values = np.asarray([row[field] for row in records], dtype=np.float64)
        result[f"mean_{field}"] = float(np.mean(values))
        result[f"median_{field}"] = float(np.median(values))
    diversity_position, diversity_rotation = pairwise_diversity(paths, pair_indices)
    result["pairwise_position_diversity_rms_m"] = diversity_position
    result["pairwise_rotation_diversity_rms_deg"] = diversity_rotation
    collision_free_mask = np.asarray([
        row["physical_collision_free"] for row in records
    ], dtype=bool)
    if np.count_nonzero(collision_free_mask) >= 2:
        free_position, free_rotation = pairwise_diversity(
            paths[collision_free_mask], pair_indices[collision_free_mask]
        )
    else:
        free_position = free_rotation = 0.0
    result["collision_free_pairwise_position_diversity_rms_m"] = free_position
    result["collision_free_pairwise_rotation_diversity_rms_deg"] = free_rotation
    per_pair_coverage = []
    per_pair_any_physical = []
    per_pair_any_008 = []
    for pair in np.unique(pair_indices):
        all_group = [row for row in records if row["pair_index"] == int(pair)]
        group = [row for row in all_group if row["physical_collision_free"]]
        per_pair_any_physical.append(any(row["physical_collision_free"] for row in all_group))
        per_pair_any_008.append(any(row["clearance_008_success"] for row in all_group))
        per_pair_coverage.append(len({row["topology_key"] for row in group if row["known_topology"]}))
    result["pair_any_physical_success_rate"] = float(np.mean(per_pair_any_physical))
    result["pair_any_008_success_rate"] = float(np.mean(per_pair_any_008))
    result["mean_collision_free_topology_classes_per_pair"] = float(np.mean(per_pair_coverage))
    result["minimum_collision_free_topology_classes_per_pair"] = int(min(per_pair_coverage))
    topology_counts = Counter(
        row["topology_key"] for row in records if row["known_topology"]
    )
    collision_free_counts = Counter(
        row["topology_key"] for row in records
        if row["known_topology"] and row["physical_collision_free"]
    )
    result["known_topology_counts"] = dict(sorted(topology_counts.items()))
    result["collision_free_topology_counts"] = dict(sorted(collision_free_counts.items()))
    if collision_free_counts:
        probabilities = np.asarray(list(collision_free_counts.values()), dtype=np.float64)
        probabilities /= probabilities.sum()
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
        result["collision_free_topology_normalized_entropy"] = (
            entropy / math.log(6.0)
        )
    else:
        result["collision_free_topology_normalized_entropy"] = 0.0
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SE(3) diffusion variant comparison", "",
        "All collision metrics below were recomputed with the full URDF geometry in COAL; the guidance surrogate was not used for scoring.", "",
        "| Variant | In bounds | Physical free | 8 cm safe | Task solved | Known topology | Free+known | Modes/pair | Best expert RMS (m) | Diversity (m) | Length (m) | ms/path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, values in summary["variants"].items():
        lines.append(
            f"| {variant} | {100 * values['workspace_valid_rate']:.1f}% | "
            f"{100 * values['physical_collision_free_rate']:.1f}% | "
            f"{100 * values['clearance_008_success_rate']:.1f}% | "
            f"{100 * values['pair_any_physical_success_rate']:.1f}% | "
            f"{100 * values['known_topology_rate']:.1f}% | "
            f"{100 * values['collision_free_known_topology_rate']:.1f}% | "
            f"{values['mean_collision_free_topology_classes_per_pair']:.2f} | "
            f"{values['mean_best_expert_position_rms_m']:.3f} | "
            f"{values['collision_free_pairwise_position_diversity_rms_m']:.3f} | "
            f"{values['mean_translation_length_m']:.3f} | "
            f"{values['mean_sampling_time_ms']:.1f} |"
        )
    lines.extend((
        "", "Interpretation:", "",
        "- `Physical free` uses zero safety margin; `8 cm safe` uses the planning safety margin.",
        "- `Known topology` means both separator cuts map to one of the six route-template signatures.",
        "- `Modes/pair` counts distinct collision-free topology signatures recovered for each held-out start/goal pair.",
        "- `Task solved` is best-of-32: at least one physical collision-free path for a held-out start/goal pair.",
        "- Table diversity is computed only among physical collision-free paths; lower expert RMS, path length, acceleration, and jerk are better.",
        "- `comparison.json` also reports 95% confidence intervals from a 2000-draw start/goal-pair cluster bootstrap.",
    ))
    if summary.get("training"):
        lines.extend((
            "", "## Training efficiency", "",
            "| Checkpoint | Parameters | Best step | Best validation noise MSE | Training minutes |",
            "|---|---:|---:|---:|---:|",
        ))
        for model_type, values in summary["training"].items():
            lines.append(
                f"| {model_type} | {values['parameter_count']:,} | "
                f"{values['best_training_step']} | "
                f"{values['best_validation_loss']:.5f} | "
                f"{values['wall_time_s'] / 60.0:.2f} |"
            )
    if summary.get("paired_task_deltas"):
        lines.extend((
            "", "## Paired held-out-task deltas", "",
            "Candidate minus baseline; positive is better for free/topology, negative is better for expert RMS.", "",
            "| Comparison | Physical-free delta | Known-topology delta | Expert RMS delta (m) |",
            "|---|---:|---:|---:|",
        ))
        for comparison, values in summary["paired_task_deltas"].items():
            lines.append(
                f"| {comparison} | "
                f"{100 * values['physical_collision_free']['candidate_minus_baseline']:+.1f} pp | "
                f"{100 * values['known_topology']['candidate_minus_baseline']:+.1f} pp | "
                f"{values['best_expert_position_rms_m']['candidate_minus_baseline']:+.3f} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--translation-check-step", type=float, default=0.04)
    parser.add_argument("--rotation-check-step-deg", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((args.dataset / "dataset_config.json").read_text(encoding="utf-8"))
    environment_path = args.dataset / config["sources"]["environment"]["copy"]
    environment_data, environment = load_environment(environment_path)
    sampling_bounds = environment_data["sampling_space"]["position_bounds"]
    bounds_min = np.asarray(sampling_bounds["min"], dtype=np.float64)
    bounds_max = np.asarray(sampling_bounds["max"], dtype=np.float64)
    urdf_path = Path(config["sources"]["urdf"]["source"])
    physical_checker = CoalCollisionChecker.from_urdf(
        urdf_path, environment, safety_margin=0.0
    )
    experts = load_experts(args.dataset, manifest)
    known_signatures = {
        _signature_key(record["expected_topology"])
        for record in manifest["trajectories"] if record.get("expected_topology")
    }
    cuts = environment_data.get("topology_cuts", [])
    prediction_paths = sorted((args.experiment / "predictions").glob("*.npz"))
    if not prediction_paths:
        raise FileNotFoundError("no prediction NPZ files found")
    output_dir = args.experiment / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_summary: dict[str, Any] = {}
    for model_type in ("unet", "dit", "dit_cross"):
        history_path = args.experiment / "models" / model_type / "training_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            best_step = history.get("best_training_step")
            if best_step is None:
                best_step = min(
                    history["history"], key=lambda row: row["validation_loss"]
                )["step"]
            training_summary[model_type] = {
                "parameter_count": history["parameter_count"],
                "best_validation_loss": history["best_validation_loss"],
                "best_training_step": best_step,
                "wall_time_s": history["wall_time_s"],
                "device": history["device"],
                "amp": history["amp"],
            }
    all_variant_summary: dict[str, Any] = {}
    csv_records: list[dict[str, Any]] = []
    sampling_splits: set[str] = set()
    for prediction_path in prediction_paths:
        with np.load(prediction_path) as prediction:
            paths = prediction["poses_wxyz"].astype(np.float64)
            pair_indices = prediction["pair_indices"].astype(np.int64)
            sample_indices = prediction["sample_indices"].astype(np.int64)
            variant = str(prediction["variant"])
            sampling_time = float(prediction["mean_sampling_time_s"])
            sampling_splits.add(
                str(prediction["sampling_split"])
                if "sampling_split" in prediction else "legacy-unspecified"
            )
        records = []
        started = time.monotonic()
        for index, (path, pair, sample_index) in enumerate(zip(paths, pair_indices, sample_indices)):
            workspace_valid = bool(
                np.all(np.isfinite(path))
                and np.all(path[:, :3] >= bounds_min)
                and np.all(path[:, :3] <= bounds_max)
            )
            if workspace_valid:
                try:
                    dense = dense_path(
                        path, args.translation_check_step,
                        args.rotation_check_step_deg,
                    )
                except ValueError:
                    workspace_valid = False
            if workspace_valid:
                physical = physical_checker.clearance(dense[:, :3], dense[:, 3:7])
                # CoalCollisionChecker defines margin-adjusted clearance as the
                # same exact signed distance minus the requested scalar margin.
                clearance_008 = physical - 0.08
                clearance_010 = physical - 0.10
            else:
                dense = path
                physical = np.asarray([-10.0])
                clearance_008 = np.asarray([-10.08])
                clearance_010 = np.asarray([-10.10])
            metrics = path_metrics(path)
            expert_position, expert_rotation = expert_distance(path, experts[int(pair)])
            signature = _topology_signature(dense, cuts)
            topology_key = _signature_key(signature)
            endpoint_position = max(
                np.linalg.norm(path[0, :3] - experts[int(pair)][0][0, :3]),
                np.linalg.norm(path[-1, :3] - experts[int(pair)][0][-1, :3]),
            )
            endpoint_rotation = max(
                quaternion_angles(path[None, 0, 3:7], experts[int(pair)][0][None, 0, 3:7])[0],
                quaternion_angles(path[None, -1, 3:7], experts[int(pair)][0][None, -1, 3:7])[0],
            )
            record = {
                "variant": variant,
                "pair_index": int(pair),
                "sample_index": int(sample_index),
                "workspace_valid": workspace_valid,
                "physical_collision_free": bool(workspace_valid and np.min(physical) > 0.0),
                "clearance_008_success": bool(workspace_valid and np.min(clearance_008) > 0.0),
                "clearance_010_success": bool(workspace_valid and np.min(clearance_010) > 0.0),
                "minimum_physical_clearance_m": float(np.min(physical)),
                "minimum_008_clearance_m": float(np.min(clearance_008)),
                "minimum_010_clearance_m": float(np.min(clearance_010)),
                "topology_key": topology_key,
                "known_topology": topology_key in known_signatures,
                "best_expert_position_rms_m": expert_position,
                "best_expert_rotation_rms_deg": expert_rotation,
                "endpoint_position_error_m": float(endpoint_position),
                "endpoint_rotation_error_deg": float(np.degrees(endpoint_rotation)),
                **metrics,
            }
            records.append(record)
            if (index + 1) % 16 == 0:
                print(
                    f"[{variant}] exact COAL {index + 1}/{len(paths)} "
                    f"elapsed={time.monotonic() - started:.1f}s", flush=True
                )
        all_variant_summary[variant] = aggregate(
            records, paths, pair_indices, sampling_time
        )
        csv_records.extend(records)
    paired_deltas: dict[str, Any] = {}
    comparisons = (
        ("guidance_vs_unet", "unet_guidance", "unet_no_guidance"),
        ("dit_vs_unet", "dit_no_guidance", "unet_no_guidance"),
        ("cross_environment_vs_dit", "dit_cross_environment", "dit_no_guidance"),
    )
    available_variants = {row["variant"] for row in csv_records}
    for name, candidate, baseline in comparisons:
        if candidate in available_variants and baseline in available_variants:
            paired_deltas[name] = {
                field: paired_task_delta(csv_records, candidate, baseline, field)
                for field in (
                    "physical_collision_free", "known_topology",
                    "best_expert_position_rms_m",
                )
            }
    summary = {
        "source_dataset": str(args.dataset.resolve()),
        "prediction_directory": str((args.experiment / "predictions").resolve()),
        "sampling_split": (
            next(iter(sampling_splits)) if len(sampling_splits) == 1
            else sorted(sampling_splits)
        ),
        "collision_check_translation_step_m": args.translation_check_step,
        "collision_check_rotation_step_deg": args.rotation_check_step_deg,
        "known_topology_signatures": sorted(known_signatures),
        "training": training_summary,
        "paired_task_deltas": paired_deltas,
        "variants": all_variant_summary,
    }
    write_json(output_dir / "comparison.json", summary)
    fields = list(csv_records[0].keys())
    with (output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_records)
    report = markdown_report(summary)
    (output_dir / "COMPARISON.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
