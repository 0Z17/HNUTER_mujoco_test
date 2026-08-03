#!/usr/bin/env python3
"""Interactive Rerun visualization of the collected diffusion dataset.

Logs the environment, all collected trajectories grouped by start/goal pair
(one color per pair), start/goal markers, body-frame orientation triads, and
an optional animated drone pose per trajectory.  Data can be streamed live to
the Rerun viewer or saved as an ``.rrd`` file.

Usage:
    .venv/bin/python rerun_visualize_dataset.py --spawn
    .venv/bin/python rerun_visualize_dataset.py --save out.rrd --no-spawn

The viewer must be able to open a window; in a headless environment save an
``.rrd`` and open it later with ``.venv/bin/rerun out.rrd``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to Rerun's [x, y, z, w] order."""
    return np.asarray(q, dtype=float)[[1, 2, 3, 0]]


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton [w,x,y,z] quaternion -> body-to-world rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_environment_metadata(dataset_dir: Path) -> dict:
    """Load the exact environment copy recorded with this dataset."""
    config_path = dataset_dir / "dataset_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        relative = (
            config.get("sources", {})
            .get("environment", {})
            .get("copy")
        )
        if relative:
            candidate = dataset_dir / relative
            if candidate.exists():
                return json.loads(candidate.read_text())

    candidates = sorted((dataset_dir / "metadata").glob("environment*.json"))
    if len(candidates) == 1:
        return json.loads(candidates[0].read_text())
    if not candidates:
        raise FileNotFoundError(
            f"no copied environment JSON found under {dataset_dir / 'metadata'}"
        )
    raise RuntimeError(
        "multiple environment JSON files found and dataset_config.json does not "
        "identify one: " + ", ".join(str(path) for path in candidates)
    )


def load_dataset(dataset_dir: Path):
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    splits = json.loads((dataset_dir / "splits.json").read_text())
    split_by_pair = {}
    for split in ("train", "validation", "test"):
        for pair in splits[f"{split}_pair_indices"]:
            split_by_pair[pair] = split
    obstacles = load_environment_metadata(dataset_dir)["obstacles"]

    pairs: dict[int, dict] = {}
    for entry in manifest["trajectories"]:
        pair_idx = entry["pair_index"]
        traj_dir = dataset_dir / entry["relative_directory"]
        if pair_idx not in pairs:
            pair_json = json.loads((traj_dir.parent / "pair.json").read_text())
            pairs[pair_idx] = {
                "start_pose": np.asarray(pair_json["start_pose"]),
                "goal_pose": np.asarray(pair_json["goal_pose"]),
                "trajectories": [],
            }
        sample = np.load(traj_dir / "training_sample.npz")
        pairs[pair_idx]["trajectories"].append(
            {
                "index": entry["trajectory_index"],
                "reference": sample["toppra_reference_state"],
                "reference_time": sample["toppra_time"],
                "actual": sample["actual_state"],
                "actual_time": sample["control_time"],
                "metrics": {
                    "path_length_m": entry.get("path_length_m"),
                    "position_rmse_m": entry.get("position_rmse_m"),
                    "attitude_rmse_deg": entry.get("attitude_rmse_deg"),
                    "final_position_error_m": entry.get("final_position_error_m"),
                    "final_attitude_error_deg": entry.get("final_attitude_error_deg"),
                    "min_planned_clearance_m": entry.get(
                        "minimum_planned_physical_clearance_m"
                    ),
                    "min_actual_clearance_m": entry.get(
                        "minimum_actual_physical_clearance_m"
                    ),
                },
            }
        )
    return pairs, obstacles, split_by_pair


PAIR_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
]
TRIAD_COLORS = [(230, 60, 60, 255), (60, 170, 60, 255), (60, 90, 230, 255)]


def log_static(rec, entity_path: str, arch) -> None:
    rec.log(entity_path, arch, static=True)


def log_environment(rec, obstacles) -> None:
    import rerun as rr

    floor_boxes, env_boxes = [], []
    for ob in obstacles:
        half = np.asarray(ob["half_extents"], dtype=float)
        center = np.asarray(ob["pose"]["position"], dtype=float)
        quat = rr.Quaternion(xyzw=quat_wxyz_to_xyzw(ob["pose"]["quaternion_wxyz"]))
        box = (center, half, quat)
        if ob.get("role") == "floor":
            floor_boxes.append(box)
        else:
            env_boxes.append(box)

    def log_boxes(path, boxes, color):
        if not boxes:
            return
        centers = np.stack([b[0] for b in boxes])
        half_sizes = np.stack([b[1] for b in boxes])
        quats = [b[2] for b in boxes]
        log_static(
            rec,
            path,
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quats,
                colors=[color] * len(boxes),
            ),
        )

    log_boxes("world/obstacles/floor", floor_boxes, (154, 165, 177, 110))
    log_boxes("world/obstacles/boxes", env_boxes, (196, 80, 70, 85))


def log_trajectories(rec, pairs, split_by_pair) -> None:
    import rerun as rr

    for pair_idx in sorted(pairs):
        color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
        color_a = (*color, 160)
        pair_data = pairs[pair_idx]
        split = split_by_pair.get(pair_idx, "?")

        start = pair_data["start_pose"]
        goal = pair_data["goal_pose"]
        log_static(
            rec,
            f"world/start_goal/pair_{pair_idx:03d}/start",
            rr.Points3D(
                positions=[start[:3]],
                colors=[(*color, 255)],
                radii=0.06,
                labels=[f"Pair {pair_idx} start ({split})"],
            ),
        )
        log_static(
            rec,
            f"world/start_goal/pair_{pair_idx:03d}/goal",
            rr.Points3D(
                positions=[goal[:3]],
                colors=[(*color, 255)],
                radii=0.06,
                labels=[f"Pair {pair_idx} goal ({split})"],
            ),
        )

        for traj in pair_data["trajectories"]:
            ref = traj["reference"]
            act = traj["actual"]
            base = f"world/trajectories/pair_{pair_idx:03d}/trajectory_{traj['index']:03d}"
            log_static(
                rec,
                f"{base}/reference",
                rr.LineStrips3D(
                    strips=[ref[:, :3]],
                    colors=[(*color, 255)],
                    radii=0.025,
                ),
            )
            log_static(
                rec,
                f"{base}/actual",
                rr.LineStrips3D(
                    strips=[act[:, :3]],
                    colors=[color_a],
                    radii=0.015,
                ),
            )


def log_orientation_triads(rec, pairs) -> None:
    import rerun as rr

    for pair_idx in sorted(pairs):
        pair_data = pairs[pair_idx]
        for pose_name, pose in (("start", pair_data["start_pose"]), ("goal", pair_data["goal_pose"])):
            R = quat_to_rotmat(pose[3:7])
            pos = pose[:3]
            length = 0.35
            strips = [np.stack([pos, pos + R[:, axis] * length]) for axis in range(3)]
            log_static(
                rec,
                f"world/orientation/pair_{pair_idx:03d}/{pose_name}_triad",
                rr.LineStrips3D(strips=strips, colors=TRIAD_COLORS, radii=0.02),
            )


def log_metrics(rec, pairs) -> None:
    import rerun as rr

    for pair_idx in sorted(pairs):
        for traj in pairs[pair_idx]["trajectories"]:
            m = traj["metrics"]
            text = (
                f"pair {pair_idx} traj {traj['index']} | "
                f"path {m['path_length_m']:.2f} m | "
                f"pos RMSE {m['position_rmse_m']*100:.1f} cm | "
                f"att RMSE {m['attitude_rmse_deg']:.2f} deg | "
                f"final pos {m['final_position_error_m']*100:.1f} cm | "
                f"final att {m['final_attitude_error_deg']:.2f} deg | "
                f"clear plan/actual {m['min_planned_clearance_m']*100:.1f}/"
                f"{m['min_actual_clearance_m']*100:.1f} cm"
            )
            log_static(
                rec,
                f"world/metrics/pair_{pair_idx:03d}/trajectory_{traj['index']:03d}",
                rr.TextLog(text),
            )


def log_animation(rec, pairs, downsample: int) -> None:
    import rerun as rr

    for pair_idx in sorted(pairs):
        color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
        for traj in pairs[pair_idx]["trajectories"]:
            base = (
                f"world/animation/pair_{pair_idx:03d}/trajectory_{traj['index']:03d}"
            )
            for kind, states, times, stride in (
                ("reference", traj["reference"], traj["reference_time"], downsample),
                ("actual", traj["actual"], traj["actual_time"], 1),
            ):
                entity = f"{base}/{kind}"
                log_static(
                    rec,
                    f"{entity}/local_axes",
                    rr.LineStrips3D(
                        strips=[
                            np.stack([np.zeros(3), np.eye(3)[axis] * 0.30])
                            for axis in range(3)
                        ],
                        colors=TRIAD_COLORS,
                        radii=0.02,
                    ),
                )
                log_static(
                    rec,
                    f"{entity}/drone_box",
                    rr.Boxes3D(
                        centers=[[0.0, 0.0, 0.0]],
                        half_sizes=[[0.11, 0.07, 0.05]],
                        colors=[(*color, 230)],
                    ),
                )
                for i in range(0, len(states), stride):
                    rr.set_time("step", sequence=i)
                    rr.set_time("time_s", duration=float(times[i]))
                    rec.log(
                        entity,
                        rr.Transform3D(
                            translation=states[i, :3],
                            rotation=rr.Quaternion(
                                xyzw=quat_wxyz_to_xyzw(states[i, 6:10])
                            ),
                        ),
                    )
                rr.reset_time()


def make_blueprint():
    from rerun.blueprint import Blueprint, Spatial3DView, TextLogView, Vertical

    return Blueprint(
        Vertical(
            Spatial3DView(origin="world", name="3D trajectories"),
            TextLogView(
                origin="world/metrics",
                name="Trajectory metrics",
                contents="$origin/**",
            ),
            row_shares=[0.78, 0.22],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/diffusion_se3_bootstrap_20260802_v2"),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=".rrd recording path (default: <dataset>/visualizations/diffusion_dataset.rrd)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not write an .rrd file"
    )
    parser.add_argument(
        "--spawn", action="store_true", help="open the Rerun viewer (stream live)"
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="skip animated drone poses (smaller recording)",
    )
    parser.add_argument(
        "--downsample", type=int, default=4, help="reference animation stride"
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    save_path = args.save or (dataset_dir / "visualizations" / "diffusion_dataset.rrd")

    try:
        import rerun as rr
    except ImportError:
        sys.exit(
            "rerun-sdk is not installed. Use the project venv:\n"
            "    .venv/bin/python rerun_visualize_dataset.py --spawn"
        )

    pairs, obstacles, split_by_pair = load_dataset(dataset_dir)
    total = sum(len(p["trajectories"]) for p in pairs.values())
    print(f"Loaded {len(pairs)} pairs, {total} trajectories, "
          f"{len(obstacles)} obstacles.")

    rr.init(
        "diffusion_se3_dataset_viewer",
        default_blueprint=make_blueprint(),
    )
    if args.spawn:
        viewer_candidates = []
        if shutil.which("rerun"):
            viewer_candidates.append(shutil.which("rerun"))
        # `.venv/bin/python` is a symlink to the system interpreter, so use
        # both the un-resolved and resolved interpreter directories.
        for exe_dir in (Path(sys.executable).parent, Path(sys.executable).resolve().parent):
            viewer_candidate = exe_dir / "rerun"
            if viewer_candidate.exists():
                viewer_candidates.append(str(viewer_candidate))
        spawned = False
        last_error = None
        for viewer in viewer_candidates:
            try:
                # connect=False: launch the viewer without blocking on the
                # handshake so a broken/headless display cannot hang the
                # script; data still streams once the viewer connects.
                rr.spawn(executable_path=viewer, connect=False)
                spawned = True
                break
            except Exception as exc:  # noqa: BLE001 - headless environments
                last_error = exc
        if not spawned:
            print(f"[warn] could not open viewer ({last_error}); "
                  "continuing and saving the recording instead.")

    log_environment(rr, obstacles)
    log_trajectories(rr, pairs, split_by_pair)
    log_orientation_triads(rr, pairs)
    log_metrics(rr, pairs)
    if not args.no_animation:
        log_animation(rr, pairs, args.downsample)
        print("Logged animated drone poses.")

    if not args.no_save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_path)
        print(f"Saved recording: {save_path}")
    if not args.spawn:
        print(f"Open it interactively with:  .venv/bin/rerun {save_path}")


if __name__ == "__main__":
    main()
