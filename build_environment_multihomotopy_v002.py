"""Build a versioned SE(3) scene with intentional alternative corridors.

This script is executed by Blender against the existing robot-containing blend.
It reuses the validated export/preview machinery from v001, but replaces only
the workspace bounds, obstacle layout, and dataset-routing metadata.  The v001
blend/JSON/XML files are not overwritten.
"""

from pathlib import Path
import json
import math
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_environment_overfit_cube_v001 as base


base.ENVIRONMENT_ID = "multihomotopy_cube_v002"
base.SCHEMA_VERSION = "2.0"
base.ENV_BOUNDS_MIN = (-3.0, -3.5, 0.0)
base.ENV_BOUNDS_MAX = (3.0, 3.5, 4.0)
base.SCRIPT_PATH = Path(__file__).resolve()
base.OUTPUT_DIR = HERE
base.BLEND_PATH = HERE / "environment_multihomotopy_v002.blend"
base.JSON_PATH = HERE / "environment_multihomotopy_v002.json"
base.XML_PATH = HERE / "environment_multihomotopy_v002.xml"
base.PREVIEW_PATH = HERE / "environment_multihomotopy_v002_preview.png"


def build_obstacle_specs(robot):
    margin = robot["safety_margin"]
    horizontal_min = robot["horizontal_min"]
    horizontal_max = robot["horizontal_max"]
    gate_gap = max(1.62, horizontal_min + 2.0 * margin + 0.08)
    gate_clear_height = 1.58
    gate_beam_height = 0.26
    gate_center_y = -1.35
    post_width = 0.28
    post_depth = 0.24
    post_x = 0.5 * (gate_gap + post_width)
    specs = []

    def add(identifier, collection, position, size, role, material, tags, group=None):
        specs.append({
            "id": identifier,
            "collection": collection,
            "position": tuple(float(v) for v in position),
            "size": tuple(float(v) for v in size),
            "rotation_z": 0.0,
            "role": role,
            "material": material,
            "tags": list(tags),
            "group": group or identifier,
        })

    add(
        "env_floor_001", "FLOOR", (0.0, 0.0, -0.04), (6.0, 7.0, 0.08),
        "floor", "floor",
        ["cube_only", "ground", "single_solid_floor", "reference_robot_overlap_allowed"],
        "floor",
    )

    # Central doorway.  Unlike v001 it sits well inside the task workspace, so
    # the southern endpoint region has meaningful depth behind the gate.
    add(
        "env_gate_left_001", "GATES",
        (-post_x, gate_center_y, 0.5 * gate_clear_height),
        (post_width, post_depth, gate_clear_height),
        "gate_post", "gate",
        ["cube_only", "floor_supported", "orientation_sensitive", "topology_separator"],
        "gate_001",
    )
    add(
        "env_gate_right_001", "GATES",
        (post_x, gate_center_y, 0.5 * gate_clear_height),
        (post_width, post_depth, gate_clear_height),
        "gate_post", "gate",
        ["cube_only", "floor_supported", "orientation_sensitive", "topology_separator"],
        "gate_001",
    )
    add(
        "env_gate_top_001", "GATES",
        (0.0, gate_center_y, gate_clear_height + 0.5 * gate_beam_height),
        (gate_gap + 2.0 * post_width, post_depth, gate_beam_height),
        "gate_beam", "gate",
        ["cube_only", "supported", "orientation_sensitive", "topology_separator"],
        "gate_001",
    )

    # A tall grounded central separator leaves stable left and right corridors;
    # its top is high enough that ordinary samples do not collapse to overflight.
    add(
        "env_center_block_001", "CENTRAL_OBSTACLES",
        (0.0, 0.52, 1.55), (1.00, 0.46, 3.10),
        "central_block", "obstacle",
        ["cube_only", "floor_supported", "left_right_bypass", "topology_separator"],
        "central_separator",
    )
    add(
        "env_pillar_left_001", "PILLARS",
        (-2.08, 1.35, 1.35), (0.28, 0.28, 2.70),
        "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "high_altitude_blocker"],
        "pillars",
    )
    add(
        "env_pillar_right_001", "PILLARS",
        (2.05, -0.20, 1.45), (0.30, 0.30, 2.90),
        "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "high_altitude_blocker"],
        "pillars",
    )
    add(
        "env_pillar_left_inner_001", "PILLARS",
        (-1.28, -0.15, 0.58), (0.24, 0.24, 1.16),
        "pillar", "obstacle",
        ["cube_only", "floor_supported", "low", "breaks_vertical_symmetry"],
        "pillars",
    )
    add(
        "env_pillar_right_inner_001", "PILLARS",
        (1.38, 1.28, 0.72), (0.24, 0.24, 1.44),
        "pillar", "obstacle",
        ["cube_only", "floor_supported", "low", "breaks_vertical_symmetry"],
        "pillars",
    )

    # A grounded inverted-L near the north side adds under/side choices without
    # blocking the endpoint region or becoming a floating deployment artifact.
    add(
        "env_beam_001", "BEAMS",
        (0.66, 1.95, 2.48), (1.32, 0.26, 0.24),
        "supported_beam", "beam",
        ["cube_only", "supported", "underpass", "inverted_L"],
        "beam_001",
    )
    add(
        "env_beam_support_001", "BEAMS",
        (1.41, 1.95, 1.24), (0.18, 0.26, 2.48),
        "beam_support", "beam",
        ["cube_only", "floor_supported", "inverted_L_support"],
        "beam_001",
    )
    add(
        "env_low_block_001", "BEAMS",
        (1.18, 1.68, 0.42), (0.52, 0.40, 0.84),
        "low_block", "beam",
        ["cube_only", "floor_supported", "breaks_vertical_symmetry"],
        "beam_001",
    )

    derived = {
        "gate_gap": gate_gap,
        "gate_gap_lower_bound": horizontal_min + 2.0 * margin,
        "gate_gap_upper_bound": horizontal_max + 2.0 * margin,
        "gate_clear_height": gate_clear_height,
        "gate_rotation_radians": 0.0,
        "normal_passage": max(base.NORMAL_PASSAGE_SCALE * horizontal_min, horizontal_min + 2.0 * margin),
        "wide_passage": max(base.WIDE_PASSAGE_SCALE * horizontal_min, horizontal_min + 2.0 * margin),
        "floor_mode": "single_solid_box",
        "reference_robot_overlap_allowed": True,
        "grounded_layout": True,
    }
    return specs, derived


base.build_obstacle_specs = build_obstacle_specs


def snapshot_non_environment_scene():
    """Exclude every versioned environment while protecting robot objects."""

    snapshot = {}
    for obj in base.bpy.data.objects:
        if obj.get("environment_id"):
            continue
        snapshot[obj.name] = {
            "parent": obj.parent.name if obj.parent else None,
            "matrix_world": tuple(
                round(float(value), 9)
                for row in obj.matrix_world for value in row
            ),
            "scale": tuple(round(float(value), 9) for value in obj.scale),
            "collections": tuple(sorted(c.name for c in obj.users_collection)),
        }
    return snapshot


base.snapshot_existing_scene = snapshot_non_environment_scene


TASK_SAMPLING = {
    "direction": "south_to_north",
    "south_region": {
        "min": [-1.65, -2.66, 0.68],
        "max": [1.65, -2.16, 2.30],
    },
    "north_region": {
        "min": [-1.65, 2.22, 0.68],
        "max": [1.65, 2.66, 2.30],
    },
    "purpose": "Every task straddles the interior doorway and central separator.",
}


def waypoint(position_min, position_max, yaw_min=-8.0, yaw_max=8.0):
    return {
        "position_bounds": {"min": position_min, "max": position_max},
        "rpy_deg_bounds": {
            "min": [-5.0, -5.0, yaw_min],
            "max": [5.0, 5.0, yaw_max],
        },
    }


def segment_bounds(gate_class, center_class):
    """Keep each RRTConnect segment on the intended side of topology cuts."""

    full_min = [-2.261, -2.723, 0.403]
    full_max = [2.261, 2.723, 3.597]
    gate_corridors = {
        "through": ([-0.30, -1.70, 0.55], [0.30, -1.00, 1.12]),
        "left": ([-2.25, -1.70, 0.55], [-1.08, -1.00, 1.90]),
        "right": ([1.08, -1.70, 0.55], [2.25, -1.00, 1.90]),
        "above": ([-2.25, -1.70, 2.05], [2.25, -1.00, 3.55]),
    }
    center_corridors = {
        "left": ([-2.25, 0.15, 0.55], [-0.65, 0.90, 3.55]),
        "right": ([0.65, 0.15, 0.55], [2.25, 0.90, 3.55]),
    }
    gate_min, gate_max = gate_corridors[gate_class]
    center_min, center_max = center_corridors[center_class]
    return [
        {"min": full_min, "max": [full_max[0], -1.351, full_max[2]]},
        {"min": gate_min, "max": gate_max},
        {
            "min": [full_min[0], -1.349, full_min[2]],
            "max": [full_max[0], 0.519, full_max[2]],
        },
        {"min": center_min, "max": center_max},
        {"min": [full_min[0], 0.521, full_min[2]], "max": full_max},
    ]


ROUTE_TEMPLATES = [
    {
        "id": "doorway_then_left",
        "expected_topology": {"gate_cut": "through", "center_cut": "left"},
        "segment_position_bounds": segment_bounds("through", "left"),
        "waypoints": [
            waypoint([-0.01, -1.66, 0.70], [0.01, -1.60, 0.95], 54.0, 56.0),
            waypoint([-0.01, -1.10, 0.70], [0.01, -1.04, 0.95], 54.0, 56.0),
            waypoint([-1.48, 0.20, 2.00], [-1.42, 0.26, 2.30], 118.0, 122.0),
            waypoint([-1.48, 0.80, 2.00], [-1.42, 0.86, 2.30], 118.0, 122.0),
        ],
    },
    {
        "id": "doorway_then_right",
        "expected_topology": {"gate_cut": "through", "center_cut": "right"},
        "segment_position_bounds": segment_bounds("through", "right"),
        "waypoints": [
            waypoint([-0.01, -1.66, 0.70], [0.01, -1.60, 0.95], -56.0, -54.0),
            waypoint([-0.01, -1.10, 0.70], [0.01, -1.04, 0.95], -56.0, -54.0),
            waypoint([1.18, 0.20, 2.00], [1.22, 0.26, 2.30], -162.0, -158.0),
            waypoint([1.18, 0.80, 2.00], [1.22, 0.86, 2.30], -162.0, -158.0),
        ],
    },
    {
        "id": "left_of_gate_then_left",
        "expected_topology": {"gate_cut": "left", "center_cut": "left"},
        "segment_position_bounds": segment_bounds("left", "left"),
        "waypoints": [
            waypoint([-2.15, -1.66, 1.20], [-2.08, -1.60, 1.68], -2.0, 2.0),
            waypoint([-2.15, -1.10, 1.20], [-2.08, -1.04, 1.68], -2.0, 2.0),
            waypoint([-1.48, 0.20, 2.00], [-1.42, 0.26, 2.30], 118.0, 122.0),
            waypoint([-1.48, 0.80, 2.00], [-1.42, 0.86, 2.30], 118.0, 122.0),
        ],
    },
    {
        "id": "right_of_gate_then_right",
        "expected_topology": {"gate_cut": "right", "center_cut": "right"},
        "segment_position_bounds": segment_bounds("right", "right"),
        "waypoints": [
            waypoint([2.05, -1.66, 1.30], [2.15, -1.60, 1.72], 88.0, 92.0),
            waypoint([2.05, -1.10, 1.30], [2.15, -1.04, 1.72], 88.0, 92.0),
            waypoint([1.18, 0.20, 2.00], [1.22, 0.26, 2.30], -162.0, -158.0),
            waypoint([1.18, 0.80, 2.00], [1.22, 0.86, 2.30], -162.0, -158.0),
        ],
    },
    {
        "id": "above_gate_then_left",
        "expected_topology": {"gate_cut": "above", "center_cut": "left"},
        "segment_position_bounds": segment_bounds("above", "left"),
        "waypoints": [
            waypoint([-0.20, -1.66, 2.75], [0.20, -1.60, 2.90], 55.0, 65.0),
            waypoint([-0.20, -1.10, 2.75], [0.20, -1.04, 2.90], 55.0, 65.0),
            waypoint([-1.48, 0.20, 2.00], [-1.42, 0.26, 2.30], 118.0, 122.0),
            waypoint([-1.48, 0.80, 2.00], [-1.42, 0.86, 2.30], 118.0, 122.0),
        ],
    },
    {
        "id": "above_gate_then_right",
        "expected_topology": {"gate_cut": "above", "center_cut": "right"},
        "segment_position_bounds": segment_bounds("above", "right"),
        "waypoints": [
            waypoint([-0.20, -1.66, 2.75], [0.20, -1.60, 2.90], 55.0, 65.0),
            waypoint([-0.20, -1.10, 2.75], [0.20, -1.04, 2.90], 55.0, 65.0),
            waypoint([1.18, 0.20, 2.00], [1.22, 0.26, 2.30], -162.0, -158.0),
            waypoint([1.18, 0.80, 2.00], [1.22, 0.86, 2.30], -162.0, -158.0),
        ],
    },
]


TOPOLOGY_CUTS = [
    {
        "id": "gate_cut",
        "axis": "y",
        "value": -1.35,
        "classes": [
            {"label": "through", "x_range": [-0.80, 0.80], "z_range": [0.45, 1.56]},
            {"label": "above", "z_min": 1.94},
            {"label": "left", "x_max": -1.08},
            {"label": "right", "x_min": 1.08},
        ],
    },
    {
        "id": "center_cut",
        "axis": "y",
        "value": 0.52,
        "classes": [
            {"label": "above", "z_min": 3.46},
            {"label": "left", "x_max": 0.0},
            {"label": "right", "x_min": 0.0},
        ],
    },
]


def append_dataset_routing_metadata():
    data = json.loads(base.JSON_PATH.read_text(encoding="utf-8"))
    data["description"] = (
        "Grounded cube-only SE3 environment with explicit multi-corridor tasks"
    )
    data["task_sampling"] = TASK_SAMPLING
    data["topology_cuts"] = TOPOLOGY_CUTS
    data["route_templates"] = ROUTE_TEMPLATES
    data["derived_design"]["multi_homotopy_strategy"] = (
        "interior doorway plus left/right central separator corridors"
    )
    data["derived_design"]["doorway_center_y_m"] = -1.35
    data["derived_design"]["workspace_x_width_m"] = 6.0
    base.JSON_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    summary = base.run()
    append_dataset_routing_metadata()
    print("MULTIHOMOTOPY_V002_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
