#!/usr/bin/env python3
"""Visualize the collected SE(3) diffusion bootstrap dataset.

Each start/goal pair is drawn with a single color; the four diverse
trajectories of a pair share that color.  The planner's TOPP-RA reference
trajectory is drawn as a solid line and the executed MPPI/MuJoCo trajectory
as a dashed line.  Environment collision boxes are drawn as translucent
boxes, and start/goal poses are marked with their body-frame axis triads.

Outputs (next to the dataset):
    visualizations/overview_3d.png           perspective + top-down overview
    visualizations/pairs_grid_3d.png         2x4 grid, one panel per pair
    visualizations/start_goal_orientation.png
                                             start/goal triads per pair
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC"]
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

STATE_LAYOUT = [
    "x", "y", "z", "vx", "vy", "vz",
    "qw", "qx", "qy", "qz",
    "omega_x_body", "omega_y_body", "omega_z_body",
]
SPLIT_BY_PAIR = {}


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton quaternion (w, x, y, z) -> body-to-world rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def box_corners(pose_position, quat_wxyz, half_extents):
    """World-frame corners of an oriented box."""
    center = np.asarray(pose_position, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    R = quat_to_rotmat(np.asarray(quat_wxyz, dtype=float))
    local = np.array(
        [
            [sx, sy, sz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=float,
    ) * half
    return center + local @ R.T


def draw_box(ax, obstacle, color, alpha=0.14, edge_alpha=0.35):
    """Draw one axis-aligned or rotated box as a translucent polyhedron."""
    corners = box_corners(
        obstacle["pose"]["position"],
        obstacle["pose"]["quaternion_wxyz"],
        obstacle["half_extents"],
    )
    faces = [
        [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5],
    ]
    poly = Poly3DCollection(
        [corners[f] for f in faces],
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=0.4,
    )
    poly.set_edgecolors([(*np.asarray(matplotlib.colors.to_rgb(color)), edge_alpha)])
    ax.add_collection3d(poly)


def add_triad(ax, pos, quat_wxyz, length, colors=("r", "g", "b"), lw=2.0):
    """Draw a body-frame XYZ triad (world-frame axes) at a pose."""
    R = quat_to_rotmat(np.asarray(quat_wxyz, dtype=float))
    pos = np.asarray(pos, dtype=float)
    for axis in range(3):
        tip = pos + R[:, axis] * length
        ax.plot(
            [pos[0], tip[0]],
            [pos[1], tip[1]],
            [pos[2], tip[2]],
            color=colors[axis],
            linewidth=lw,
            solid_capstyle="round",
            zorder=20,
        )


def setup_axes(ax, obstacles, title=None, elev=30, azim=-60):
    """Common 3D styling: obstacles, equal aspect, labels."""
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    for obstacle in obstacles:
        role = obstacle.get("role", "")
        if role == "floor":
            draw_box(ax, obstacle, "#9aa5b1", alpha=0.18, edge_alpha=0.15)
        else:
            draw_box(ax, obstacle, "#c0504d")
    ax.view_init(elev=elev, azim=azim)


def equal_aspect(ax, obstacles, paths):
    """Set equal scaling by enlarging the box to the largest axis span."""
    xs, ys, zs = [], [], []
    for path in paths:
        xs.extend(path[:, 0]), ys.extend(path[:, 1]), zs.extend(path[:, 2])
    for obstacle in obstacles:
        corners = box_corners(
            obstacle["pose"]["position"],
            obstacle["pose"]["quaternion_wxyz"],
            obstacle["half_extents"],
        )
        xs.extend(corners[:, 0]), ys.extend(corners[:, 1]), zs.extend(corners[:, 2])
    mins = np.array([min(xs), min(ys), min(zs)])
    maxs = np.array([max(xs), max(ys), max(zs)])
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins).max() + 0.35
    lim = np.array([center - half, center + half]).T
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_zlim(*lim[2])
    ax.set_box_aspect((1, 1, 1))


def load_environment_metadata(dataset_dir: Path) -> dict:
    """Load the environment copy recorded by the collector, with v1 fallback."""
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
    for split in ("train", "validation", "test"):
        for pair in splits[f"{split}_pair_indices"]:
            SPLIT_BY_PAIR[pair] = split

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
                "trajectory_index": entry["trajectory_index"],
                "reference": sample["toppra_reference_state"],
                "actual": sample["actual_state"],
            }
        )
    return pairs, obstacles


def plot_overview(pairs, obstacles, out_path: Path):
    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(17, 8))
    for sub, elev, azim, name in (
        (121, 30, -60, "透视视图"),
        (122, 90, -90, "俯视图 (XY 平面)"),
    ):
        ax = fig.add_subplot(sub, projection="3d")
        setup_axes(ax, obstacles, title=f"数据集轨迹总览 —— {name}", elev=elev, azim=azim)
        paths_for_scale = []
        for pair_idx in sorted(pairs):
            color = colors[pair_idx % len(colors)]
            for traj in pairs[pair_idx]["trajectories"]:
                paths_for_scale.append(traj["reference"])
                ax.plot(
                    traj["reference"][:, 0],
                    traj["reference"][:, 1],
                    traj["reference"][:, 2],
                    color=color,
                    linewidth=1.3,
                    alpha=0.9,
                )
                ax.plot(
                    traj["actual"][:, 0],
                    traj["actual"][:, 1],
                    traj["actual"][:, 2],
                    color=color,
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.7,
                )
            start = pairs[pair_idx]["start_pose"]
            goal = pairs[pair_idx]["goal_pose"]
            ax.scatter(
                *start[:3], color=color, s=70, marker="o", edgecolors="k",
                linewidths=0.7, zorder=25,
            )
            ax.scatter(
                *goal[:3], color=color, s=120, marker="*", edgecolors="k",
                linewidths=0.7, zorder=25,
            )
        equal_aspect(ax, obstacles, paths_for_scale)

    handles = [
        Patch(
            facecolor=colors[i],
            label=f"Pair {i} ({'train' if SPLIT_BY_PAIR.get(i) == 'train' else SPLIT_BY_PAIR.get(i, '?')})",
        )
        for i in sorted(pairs)
    ]
    handles += [
        plt.Line2D([], [], color="0.3", label="参考轨迹 (TOPP-RA)"),
        plt.Line2D([], [], color="0.3", linestyle="--", label="实际轨迹 (MPPI/MuJoCo)"),
        plt.Line2D([], [], color="0.3", marker="o", linestyle="None",
                   markersize=8, label="起点"),
        plt.Line2D([], [], color="0.3", marker="*", linestyle="None",
                   markersize=11, label="终点"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pairs_grid(pairs, obstacles, out_path: Path):
    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(17, 16))
    for pair_idx in sorted(pairs):
        ax = fig.add_subplot(2, 4, pair_idx + 1, projection="3d")
        split = SPLIT_BY_PAIR.get(pair_idx, "?")
        setup_axes(
            ax, obstacles,
            title=f"Pair {pair_idx} ({split}) —— 4 条轨迹",
            elev=28, azim=-58,
        )
        color = colors[pair_idx % len(colors)]
        paths_for_scale = []
        for traj in pairs[pair_idx]["trajectories"]:
            paths_for_scale.append(traj["reference"])
            ax.plot(
                traj["reference"][:, 0], traj["reference"][:, 1],
                traj["reference"][:, 2], color=color, linewidth=1.5, alpha=0.95,
                label=f"参考 #{traj['trajectory_index']}" if traj["trajectory_index"] == 0 else None,
            )
            ax.plot(
                traj["actual"][:, 0], traj["actual"][:, 1],
                traj["actual"][:, 2], color=color, linewidth=0.9,
                linestyle="--", alpha=0.75,
            )
        start = pairs[pair_idx]["start_pose"]
        goal = pairs[pair_idx]["goal_pose"]
        ax.scatter(*start[:3], color=color, s=60, marker="o", edgecolors="k",
                   linewidths=0.7, zorder=25)
        ax.scatter(*goal[:3], color=color, s=110, marker="*", edgecolors="k",
                   linewidths=0.7, zorder=25)
        equal_aspect(ax, obstacles, paths_for_scale)
    fig.suptitle("按起终点分组：每组 4 条多样轨迹，同一组使用同一颜色", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_orientations(pairs, obstacles, out_path: Path):
    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(17, 16))
    for pair_idx in sorted(pairs):
        ax = fig.add_subplot(2, 4, pair_idx + 1, projection="3d")
        split = SPLIT_BY_PAIR.get(pair_idx, "?")
        setup_axes(
            ax, obstacles,
            title=f"Pair {pair_idx} ({split}) 起终点姿态",
            elev=28, azim=-58,
        )
        start = pairs[pair_idx]["start_pose"]
        goal = pairs[pair_idx]["goal_pose"]
        color = colors[pair_idx % len(colors)]
        add_triad(ax, start[:3], start[3:7], 0.35, lw=2.6)
        add_triad(ax, goal[:3], goal[3:7], 0.35, lw=2.6)
        ax.scatter(*start[:3], color=color, s=60, marker="o", edgecolors="k",
                   linewidths=0.7, zorder=25)
        ax.scatter(*goal[:3], color=color, s=110, marker="*", edgecolors="k",
                   linewidths=0.7, zorder=25)
        paths_for_scale = [pairs[pair_idx]["trajectories"][0]["reference"]]
        equal_aspect(ax, obstacles, paths_for_scale)
    fig.suptitle("起点与终点姿态（红=X 绿=Y 蓝=Z 机身轴，长度 0.35 m）", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/diffusion_se3_bootstrap_20260802_v2"),
        help="Dataset root (contains manifest.json and pairs/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save PNGs (default: <dataset>/visualizations).",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    out_dir = args.output_dir or (dataset_dir / "visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplcache"))

    print(f"Loading dataset from {dataset_dir} ...")
    pairs, obstacles = load_dataset(dataset_dir)
    total = sum(len(p["trajectories"]) for p in pairs.values())
    print(f"Loaded {len(pairs)} pairs, {total} trajectories, "
          f"{len(obstacles)} obstacles.")

    plot_overview(pairs, obstacles, out_dir / "overview_3d.png")
    print("Saved overview_3d.png")
    plot_pairs_grid(pairs, obstacles, out_dir / "pairs_grid_3d.png")
    print("Saved pairs_grid_3d.png")
    plot_orientations(pairs, obstacles, out_dir / "start_goal_orientation.png")
    print("Saved start_goal_orientation.png")


if __name__ == "__main__":
    main()
