"""Build a deterministic, grounded cube-only SE(3) planning environment.

Run inside Blender.  The script intentionally never reparents, rescales, or moves the
existing robot.  All generated objects live under the ENVIRONMENT collection.
"""

import bpy
import json
import math
import random
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path

from mathutils import Euler, Vector


# -----------------------------------------------------------------------------
# Reproducible configuration
# -----------------------------------------------------------------------------

ENVIRONMENT_ID = "overfit_cube_v001"
RANDOM_SEED = 20270802

MARGIN_RATIO = 0.10
MIN_MARGIN = 0.05
MAX_MARGIN = 0.15

NORMAL_PASSAGE_SCALE = 1.35
WIDE_PASSAGE_SCALE = 1.70
ORIENTATION_SENSITIVE_SCALE = 1.10

SCHEMA_VERSION = "1.1"
ENV_BOUNDS_MIN = (-2.0, -3.5, 0.0)
ENV_BOUNDS_MAX = (2.0, 3.5, 4.0)
VOXEL_RESOLUTION = 0.18
RANDOM_POSITION_SAMPLES = 1500

SCRIPT_PATH = Path(
    globals().get("__file__", Path.cwd() / "build_environment_overfit_cube_v001.py")
).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
BLEND_PATH = OUTPUT_DIR / "environment_overfit_cube_v001.blend"
JSON_PATH = OUTPUT_DIR / "environment_overfit_cube_v001.json"
XML_PATH = OUTPUT_DIR / "environment_overfit_cube_v001.xml"
PREVIEW_PATH = OUTPUT_DIR / "environment_overfit_cube_v001_preview.png"

COLLECTION_NAMES = (
    "FLOOR",
    "BOUNDARIES",
    "PILLARS",
    "GATES",
    "CENTRAL_OBSTACLES",
    "BEAMS",
    "PARTIAL_ENCLOSURES",
    "DEBUG",
)

COLORS = {
    "floor": (0.10, 0.13, 0.18, 1.0),
    "obstacle": (0.10, 0.30, 0.48, 1.0),
    "gate": (0.025, 0.16, 0.58, 1.0),
    "beam": (0.72, 0.16, 0.025, 1.0),
    "partial": (0.035, 0.38, 0.13, 1.0),
    "boundary": (0.035, 0.40, 0.64, 1.0),
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def is_in_collection(obj, name):
    return any(c.name.lower() == name.lower() for c in obj.users_collection)


def object_props_as_strings(obj):
    return {str(key): str(obj[key]) for key in obj.keys() if key != "_RNA_UI"}


def select_base_link():
    candidates = []
    for obj in bpy.data.objects:
        lower = obj.name.lower()
        score = 0
        if obj.name == "base_link":
            score += 100
        if lower == "base_link":
            score += 80
        elif "base_link" in lower:
            score += 20
        if obj.type == "ARMATURE":
            score += 20
        if is_in_collection(obj, "link"):
            score += 20
        if obj.parent is None:
            score += 10
        props = object_props_as_strings(obj)
        if any("base_link" in (str(k) + str(v)).lower() for k, v in props.items()):
            score += 15
        plausible_root = obj.type == "ARMATURE" and is_in_collection(obj, "link") and obj.parent is None
        if "base_link" in lower or plausible_root:
            candidates.append((score, obj))
    if not candidates:
        raise RuntimeError("No reliable Phobos base-link candidate was found; environment build stopped.")
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    selected = candidates[0][1]
    if candidates[0][0] < 80:
        raise RuntimeError("Base-link identification confidence was too low; environment build stopped.")
    return selected, [obj.name for _, obj in candidates]


def is_collision_object(obj):
    if obj.type != "MESH":
        return False
    props = object_props_as_strings(obj)
    metadata_collision = any(
        "collision" in (str(key) + "=" + str(value)).lower()
        for key, value in props.items()
        if "phobos" in str(key).lower() or "type" in str(key).lower()
    )
    return is_in_collection(obj, "collision") or metadata_collision


def is_descendant_of(obj, ancestor):
    parent = obj.parent
    while parent is not None:
        if parent == ancestor:
            return True
        parent = parent.parent
    return False


def evaluated_world_vertices(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        return [matrix @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def union_aabb(objects, target_inverse=None):
    points = []
    for obj in objects:
        world_points = evaluated_world_vertices(obj)
        if target_inverse is None:
            points.extend(world_points)
        else:
            points.extend(target_inverse @ point for point in world_points)
    if not points:
        raise RuntimeError("The selected collision-object set contains no evaluated mesh vertices.")
    minimum = [min(point[i] for point in points) for i in range(3)]
    maximum = [max(point[i] for point in points) for i in range(3)]
    size = [maximum[i] - minimum[i] for i in range(3)]
    center = [(minimum[i] + maximum[i]) * 0.5 for i in range(3)]
    return {"min": minimum, "max": maximum, "size": size, "center": center}


def analyze_robot():
    base_link, candidates = select_base_link()
    all_collisions = [
        obj for obj in bpy.data.objects
        if is_collision_object(obj) and is_descendant_of(obj, base_link)
    ]
    base_rigid = [obj for obj in all_collisions if obj.parent == base_link]
    if not base_rigid:
        raise RuntimeError(
            "No collision objects rigidly parented to the selected base_link were found; build stopped."
        )

    base_local = union_aabb(base_rigid, base_link.matrix_world.inverted())
    all_local = union_aabb(all_collisions, base_link.matrix_world.inverted())
    all_world = union_aabb(all_collisions)
    size = base_local["size"]
    if min(size) <= 1e-6 or max(size) > 3.5:
        raise RuntimeError(f"Extracted base collision size is not reliable: {size}")

    horizontal_max = max(size[0], size[1])
    horizontal_min = min(size[0], size[1])
    safety_margin = clamp(MARGIN_RATIO * horizontal_max, MIN_MARGIN, MAX_MARGIN)
    movable = [obj.name for obj in all_collisions if obj not in base_rigid]

    return {
        "base_link": base_link,
        "base_link_candidates": candidates,
        "base_rigid_objects": base_rigid,
        "all_collision_objects": all_collisions,
        "movable_collision_names": movable,
        "base_local": base_local,
        "all_local": all_local,
        "all_world": all_world,
        "size": size,
        "horizontal_max": horizontal_max,
        "horizontal_min": horizontal_min,
        "vertical_extent": size[2],
        "safety_margin": safety_margin,
    }


def snapshot_existing_scene():
    snapshot = {}
    for obj in bpy.data.objects:
        if obj.get("environment_id") == ENVIRONMENT_ID:
            continue
        snapshot[obj.name] = {
            "parent": obj.parent.name if obj.parent else None,
            "matrix_world": tuple(round(float(v), 9) for row in obj.matrix_world for v in row),
            "scale": tuple(round(float(v), 9) for v in obj.scale),
            "collections": tuple(sorted(c.name for c in obj.users_collection)),
        }
    return snapshot


def verify_existing_scene_unchanged(snapshot):
    changed = []
    for name, before in snapshot.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            changed.append(f"{name}: missing")
            continue
        after = {
            "parent": obj.parent.name if obj.parent else None,
            "matrix_world": tuple(round(float(v), 9) for row in obj.matrix_world for v in row),
            "scale": tuple(round(float(v), 9) for v in obj.scale),
            "collections": tuple(sorted(c.name for c in obj.users_collection)),
        }
        if before != after:
            changed.append(name)
    return changed


def collection_tree(collection):
    result = [collection]
    for child in collection.children:
        result.extend(collection_tree(child))
    return result


def remove_previous_environment():
    root = bpy.data.collections.get("ENVIRONMENT")
    if root is None:
        return
    collections = collection_tree(root)
    objects = {obj for collection in collections for obj in collection.objects}
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in reversed(collections):
        bpy.data.collections.remove(collection)


def create_collection_hierarchy():
    root = bpy.data.collections.new("ENVIRONMENT")
    bpy.context.scene.collection.children.link(root)
    children = {}
    for name in COLLECTION_NAMES:
        collection = bpy.data.collections.new(name)
        root.children.link(collection)
        children[name] = collection
    return root, children


def make_material(name, color, metallic=0.0, roughness=0.72):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = roughness
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = color[3]
    return material


def add_cube_object(spec, collection, material, collision=True, render=True):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=spec["position"])
    obj = bpy.context.active_object
    obj.name = spec["id"]
    for linked in list(obj.users_collection):
        linked.objects.unlink(obj)
    collection.objects.link(obj)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, spec.get("rotation_z", 0.0))
    obj.scale = tuple(value * 0.5 for value in spec["size"])
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.select_set(False)
    if material is not None:
        obj.data.materials.append(material)
    obj["environment_id"] = ENVIRONMENT_ID
    obj["collision"] = bool(collision)
    obj["visual"] = bool(render)
    obj["static"] = True
    obj["export_to_environment"] = bool(collision)
    obj["role"] = spec.get("role", "debug")
    obj["collection"] = collection.name
    obj["size_xyz"] = list(spec["size"])
    obj["tags"] = ",".join(spec.get("tags", []))
    obj.hide_render = not render
    return obj


def build_obstacle_specs(robot):
    margin = robot["safety_margin"]
    size_x, size_y, size_z = robot["size"]
    horizontal_min = robot["horizontal_min"]
    horizontal_max = robot["horizontal_max"]

    # This gap lies strictly between the short- and long-axis conservative widths.
    gate_gap_lower = horizontal_min + 2.0 * margin
    gate_gap_upper = horizontal_max + 2.0 * margin
    gate_gap = gate_gap_lower + 0.66 * (gate_gap_upper - gate_gap_lower)
    gate_angle = math.radians(24.0)
    gate_center = (0.25, -2.45)
    post_width = clamp(0.22 * horizontal_max, 0.25, 0.30)
    gate_depth = 0.24
    gate_clear_height = max(1.45, NORMAL_PASSAGE_SCALE * size_z + 2.0 * margin)
    gate_beam_height = 0.26
    post_offset = gate_gap * 0.5 + post_width * 0.5

    def gate_xy(local_x):
        cosine, sine = math.cos(gate_angle), math.sin(gate_angle)
        return (
            gate_center[0] + cosine * local_x,
            gate_center[1] + sine * local_x,
        )

    floor_thickness = 0.08

    specs = []

    def add(identifier, collection, position, size, role, material, tags, rotation_z=0.0, group=None):
        specs.append({
            "id": identifier,
            "collection": collection,
            "position": tuple(float(v) for v in position),
            "size": tuple(float(v) for v in size),
            "rotation_z": float(rotation_z),
            "role": role,
            "material": material,
            "tags": list(tags),
            "group": group or identifier,
        })

    # The imported robot is only a collision-size reference.  A single unbroken
    # floor is therefore used even though the reference pose intersects it.
    add("env_floor_001", "FLOOR", (0.0, 0.0, -floor_thickness * 0.5),
        (4.0, 7.0, floor_thickness), "floor", "floor",
        ["cube_only", "ground", "single_solid_floor", "reference_robot_overlap_allowed"],
        group="floor")

    pillar_cross = clamp(0.24 * horizontal_max, 0.26, 0.34)
    add("env_pillar_floor_001", "PILLARS", (-1.50, -1.16, 1.05),
        (pillar_cross, pillar_cross, 2.10), "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "raised_height"], group="pillars")
    add("env_pillar_floor_002", "PILLARS", (-0.66, -0.86, 0.55),
        (0.25, 0.25, 1.10), "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "low"], group="pillars")
    add("env_pillar_tall_001", "PILLARS", (-1.34, 0.10, 1.65),
        (0.30, 0.30, 3.30), "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "high_altitude_blocker"], group="pillars")
    add("env_pillar_tall_002", "PILLARS", (-1.06, 1.92, 1.40),
        (0.26, 0.26, 2.80), "pillar", "obstacle",
        ["cube_only", "floor_supported", "staggered", "high_altitude_blocker"], group="pillars")

    left_xy = gate_xy(-post_offset)
    right_xy = gate_xy(post_offset)
    add("env_gate_left_001", "GATES", (left_xy[0], left_xy[1], gate_clear_height * 0.5),
        (post_width, gate_depth, gate_clear_height), "gate_post", "gate",
        ["cube_only", "floor_supported", "orientation_sensitive"], gate_angle, "gate_001")
    add("env_gate_right_001", "GATES", (right_xy[0], right_xy[1], gate_clear_height * 0.5),
        (post_width, gate_depth, gate_clear_height), "gate_post", "gate",
        ["cube_only", "floor_supported", "orientation_sensitive"], gate_angle, "gate_001")
    add("env_gate_top_001", "GATES",
        (gate_center[0], gate_center[1], gate_clear_height + gate_beam_height * 0.5),
        (gate_gap + 2.0 * post_width, gate_depth, gate_beam_height), "gate_beam", "gate",
        ["cube_only", "overhead", "orientation_sensitive"], gate_angle, "gate_001")

    add("env_center_block_001", "CENTRAL_OBSTACLES", (-0.05, 1.08, 1.40),
        (0.75, 0.58, 2.80), "central_block", "obstacle",
        ["cube_only", "floor_supported", "left_right_bypass", "limited_overflight", "raised_height"], group="central")
    add("env_offset_pillar_001", "CENTRAL_OBSTACLES", (1.43, 1.62, 0.90),
        (0.26, 0.26, 1.80), "offset_pillar", "obstacle",
        ["cube_only", "floor_supported", "asymmetric_route", "raised_height"], group="central")

    add("env_beam_001", "BEAMS", (0.15, 2.55, 2.45),
        (0.80, 0.26, 0.24), "supported_beam", "beam",
        ["cube_only", "overhead", "underpass", "overflight", "left_side_bypass", "inverted_L"], group="beam_001")
    add("env_beam_support_001", "BEAMS", (0.64, 2.55, 1.225),
        (0.18, 0.26, 2.45), "beam_support", "beam",
        ["cube_only", "floor_supported", "inverted_L_support"], group="beam_001")
    add("env_low_block_001", "BEAMS", (0.61, 2.28, 0.45),
        (0.55, 0.42, 0.90), "low_block", "beam",
        ["cube_only", "floor_supported", "breaks_vertical_symmetry"], group="beam_001")

    # A shallow C with a wide west opening and a second height-limited exit.
    add("env_partial_wall_back_001", "PARTIAL_ENCLOSURES", (1.43, 0.18, 0.50),
        (0.16, 1.10, 1.00), "partial_wall", "partial",
        ["cube_only", "floor_supported", "partial_enclosure"], group="partial_001")
    add("env_partial_wall_south_001", "PARTIAL_ENCLOSURES", (1.10, -0.29, 0.50),
        (0.50, 0.16, 1.00), "partial_wall", "partial",
        ["cube_only", "floor_supported", "wide_exit"], group="partial_001")
    add("env_partial_wall_north_001", "PARTIAL_ENCLOSURES", (1.17, 0.65, 0.325),
        (0.36, 0.16, 0.65), "partial_wall", "partial",
        ["cube_only", "floor_supported", "height_limited_exit"], group="partial_001")

    derived = {
        "gate_gap": gate_gap,
        "gate_gap_lower_bound": gate_gap_lower,
        "gate_gap_upper_bound": gate_gap_upper,
        "gate_clear_height": gate_clear_height,
        "gate_rotation_radians": gate_angle,
        "normal_passage": max(NORMAL_PASSAGE_SCALE * horizontal_min, horizontal_min + 2.0 * margin),
        "wide_passage": max(WIDE_PASSAGE_SCALE * horizontal_min, horizontal_min + 2.0 * margin),
        "floor_mode": "single_solid_box",
        "reference_robot_overlap_allowed": True,
        "grounded_layout": True,
    }
    return specs, derived


def add_boundary_markers(collection, material):
    thickness = 0.025
    specs = []
    index = 1
    # Bottom and top rectangular rings.
    for z in (0.02, 3.98):
        for y in (ENV_BOUNDS_MIN[1], ENV_BOUNDS_MAX[1]):
            specs.append((f"env_boundary_marker_{index:03d}", (0.0, y, z), (4.0, thickness, thickness)))
            index += 1
        for x in (ENV_BOUNDS_MIN[0], ENV_BOUNDS_MAX[0]):
            specs.append((f"env_boundary_marker_{index:03d}", (x, 0.0, z), (thickness, 7.0, thickness)))
            index += 1
    for x in (ENV_BOUNDS_MIN[0], ENV_BOUNDS_MAX[0]):
        for y in (ENV_BOUNDS_MIN[1], ENV_BOUNDS_MAX[1]):
            specs.append((f"env_boundary_marker_{index:03d}", (x, y, 2.0), (thickness, thickness, 4.0)))
            index += 1
    objects = []
    for identifier, position, size in specs:
        spec = {
            "id": identifier,
            "position": position,
            "size": size,
            "rotation_z": 0.0,
            "role": "boundary_marker",
            "tags": ["cube_only", "debug", "non_collision"],
        }
        obj = add_cube_object(spec, collection, material, collision=False, render=True)
        obj["export_to_environment"] = False
        objects.append(obj)
    return objects


def spec_quaternion(spec):
    quaternion = Euler((0.0, 0.0, spec.get("rotation_z", 0.0)), "XYZ").to_quaternion()
    quaternion.normalize()
    return [quaternion.w, quaternion.x, quaternion.y, quaternion.z]


def spec_world_aabb(spec):
    hx, hy, hz = (value * 0.5 for value in spec["size"])
    angle = spec.get("rotation_z", 0.0)
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    world_half = (cosine * hx + sine * hy, sine * hx + cosine * hy, hz)
    return (
        [spec["position"][i] - world_half[i] for i in range(3)],
        [spec["position"][i] + world_half[i] for i in range(3)],
    )


def aabb_has_positive_overlap(a, b, tolerance=1e-6):
    return all(min(a[1][i], b[1][i]) - max(a[0][i], b[0][i]) > tolerance for i in range(3))


def find_overlap_warnings(specs):
    warnings = []
    for index, first in enumerate(specs):
        for second in specs[index + 1:]:
            if first["group"] == second["group"]:
                continue
            if aabb_has_positive_overlap(spec_world_aabb(first), spec_world_aabb(second)):
                warnings.append({"object_a": first["id"], "object_b": second["id"]})
    return warnings


def current_robot_overlap_names(robot, specs):
    robot_aabb = (robot["all_world"]["min"], robot["all_world"]["max"])
    return [spec["id"] for spec in specs if aabb_has_positive_overlap(robot_aabb, spec_world_aabb(spec))]


def grounding_failures(specs):
    supported_overhead = {"env_gate_top_001", "env_beam_001"}
    failures = []
    for spec in specs:
        if spec["role"] == "floor" or spec["id"] in supported_overhead:
            continue
        bottom_z = spec["position"][2] - 0.5 * spec["size"][2]
        if abs(bottom_z) > 1e-6:
            failures.append(spec["id"])
    return failures


def sampling_bounds(robot):
    size = robot["size"]
    margin = robot["safety_margin"]
    minimum = [ENV_BOUNDS_MIN[i] + 0.5 * size[i] + margin for i in range(3)]
    maximum = [ENV_BOUNDS_MAX[i] - 0.5 * size[i] - margin for i in range(3)]
    if any(minimum[i] >= maximum[i] for i in range(3)):
        raise RuntimeError("Robot-derived sampling bounds are empty.")
    return minimum, maximum


def point_collides_conservative(point, spec, robot_half_with_margin):
    dx = point[0] - spec["position"][0]
    dy = point[1] - spec["position"][1]
    dz = point[2] - spec["position"][2]
    angle = spec.get("rotation_z", 0.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    local_z = dz
    hx, hy, hz = (value * 0.5 for value in spec["size"])
    rx, ry, rz = robot_half_with_margin
    expansion_x = abs(cosine) * rx + abs(sine) * ry
    expansion_y = abs(sine) * rx + abs(cosine) * ry
    return (
        abs(local_x) <= hx + expansion_x
        and abs(local_y) <= hy + expansion_y
        and abs(local_z) <= hz + rz
    )


def validate_free_space(robot, specs, sample_min, sample_max):
    margin = robot["safety_margin"]
    robot_half = tuple(0.5 * robot["size"][i] + margin for i in range(3))

    def is_free(point):
        return not any(point_collides_conservative(point, spec, robot_half) for spec in specs)

    resolution = VOXEL_RESOLUTION
    counts = [max(1, int(math.ceil((sample_max[i] - sample_min[i]) / resolution))) for i in range(3)]
    steps = [(sample_max[i] - sample_min[i]) / counts[i] for i in range(3)]
    free = set()
    for ix in range(counts[0]):
        x = sample_min[0] + (ix + 0.5) * steps[0]
        for iy in range(counts[1]):
            y = sample_min[1] + (iy + 0.5) * steps[1]
            for iz in range(counts[2]):
                z = sample_min[2] + (iz + 0.5) * steps[2]
                if is_free((x, y, z)):
                    free.add((ix, iy, iz))

    remaining = set(free)
    components = []
    neighbor_offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        size = 1
        while queue:
            current = queue.popleft()
            for offset in neighbor_offsets:
                neighbor = tuple(current[i] + offset[i] for i in range(3))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    size += 1
        components.append(size)

    random.seed(RANDOM_SEED)
    free_samples = 0
    for _ in range(RANDOM_POSITION_SAMPLES):
        point = tuple(random.uniform(sample_min[i], sample_max[i]) for i in range(3))
        free_samples += int(is_free(point))

    total_voxels = counts[0] * counts[1] * counts[2]
    largest_ratio = max(components) / len(free) if free and components else 0.0
    return {
        "voxel_resolution": resolution,
        "grid_shape": counts,
        "grid_free_ratio": len(free) / total_voxels if total_voxels else 0.0,
        "free_space_component_count": len(components),
        "largest_free_space_component_ratio": largest_ratio,
        "coarse_free_sample_ratio": free_samples / RANDOM_POSITION_SAMPLES,
        "random_sample_count": RANDOM_POSITION_SAMPLES,
    }


def obstacle_json(spec):
    size = list(spec["size"])
    return {
        "id": spec["id"],
        "type": "box",
        "pose": {
            "position": rounded(spec["position"]),
            "quaternion_wxyz": rounded(spec_quaternion(spec)),
        },
        "size_xyz": rounded(size),
        "half_extents": rounded([value * 0.5 for value in size]),
        "collision": True,
        "visual": True,
        "static": True,
        "role": spec["role"],
        "collection": spec["collection"],
        "tags": spec["tags"],
    }


def write_environment_json(robot, specs, derived, validation, sample_min, sample_max):
    base = robot["base_local"]
    all_local = robot["all_local"]
    counts = dict(sorted(Counter(spec["collection"] for spec in specs).items()))
    data = {
        "schema_version": SCHEMA_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "description": "Moderate difficulty grounded cube-only SE3 planning environment",
        "random_seed": RANDOM_SEED,
        "units": {"length": "meter", "angle": "radian"},
        "coordinate_frame": {
            "name": "world", "handedness": "right", "up_axis": "Z", "forward_axis": "Y"
        },
        "bounds": {"min": list(ENV_BOUNDS_MIN), "max": list(ENV_BOUNDS_MAX)},
        "generation_parameters": {
            "margin_ratio": MARGIN_RATIO,
            "minimum_margin": MIN_MARGIN,
            "maximum_margin": MAX_MARGIN,
            "normal_passage_scale": NORMAL_PASSAGE_SCALE,
            "wide_passage_scale": WIDE_PASSAGE_SCALE,
            "orientation_sensitive_scale": ORIENTATION_SENSITIVE_SCALE,
        },
        "robot_reference": {
            "source": "existing_phobos_collision_objects",
            "base_link_candidates": robot["base_link_candidates"],
            "base_link_object": robot["base_link"].name,
            "collision_object_names": [obj.name for obj in robot["base_rigid_objects"]],
            "all_robot_collision_object_names": [obj.name for obj in robot["all_collision_objects"]],
            "movable_arm_collision_object_names": robot["movable_collision_names"],
            "movable_arm_collisions_included": bool(robot["movable_collision_names"]),
            "base_collision_aabb_local": {"min": rounded(base["min"]), "max": rounded(base["max"])},
            "base_collision_center_local": rounded(base["center"]),
            "base_collision_size_xyz": rounded(base["size"]),
            "base_collision_max_horizontal_extent": round(robot["horizontal_max"], 6),
            "base_horizontal_max": round(robot["horizontal_max"], 6),
            "base_horizontal_min": round(robot["horizontal_min"], 6),
            "base_collision_vertical_extent": round(robot["vertical_extent"], 6),
            "base_vertical_extent": round(robot["vertical_extent"], 6),
            "robot_default_configuration_collision_aabb_local": {
                "min": rounded(all_local["min"]), "max": rounded(all_local["max"])
            },
            "safety_margin": round(robot["safety_margin"], 6),
            "margin_ratio": MARGIN_RATIO,
        },
        "sampling_space": {
            "position_bounds": {"min": rounded(sample_min), "max": rounded(sample_max)},
            "orientation": {"space": "SO3", "sampling": "configured_by_dataset_generator"},
            "fixed_start_region": False,
            "fixed_goal_region": False,
            "requires_full_robot_collision_check": True,
            "requires_start_goal_connectivity_check": True,
        },
        "derived_design": {
            "normal_passage_width": round(derived["normal_passage"], 6),
            "wide_passage_width": round(derived["wide_passage"], 6),
            "floor_mode": derived["floor_mode"],
            "grounded_layout": derived["grounded_layout"],
            "reference_robot_overlap_allowed": derived["reference_robot_overlap_allowed"],
            "upper_space_strategy": "raised central block and tall grounded pillars discourage trivial overflight",
        },
        "obstacles": [obstacle_json(spec) for spec in specs],
        "passages": [
            {
                "id": "passage_gate_001",
                "type": "doorway",
                "clearance_size_xyz": rounded([
                    derived["gate_gap"], 0.24, derived["gate_clear_height"]
                ]),
                "difficulty": "orientation_sensitive",
                "orientation_sensitive": True,
                "orientation_sensitive_bounds": {
                    "short_axis_conservative_width": round(derived["gate_gap_lower_bound"], 6),
                    "long_axis_conservative_width": round(derived["gate_gap_upper_bound"], 6),
                    "gate_rotation_radians": round(derived["gate_rotation_radians"], 6),
                },
                "alternative_routes": ["side", "above"],
            },
            {
                "id": "passage_beam_001",
                "type": "supported_inverted_L_beam",
                "clearance_size_xyz": [1.75, 0.26, 2.33],
                "difficulty": "medium",
                "orientation_sensitive": False,
                "alternative_routes": ["under", "above", "left_side"],
            },
            {
                "id": "passage_partial_001",
                "type": "partial_enclosure",
                "clearance_size_xyz": [1.0, 1.0, 1.3],
                "difficulty": "medium",
                "orientation_sensitive": False,
                "alternative_routes": ["wide_west_exit", "above_north_wall", "outside_bypass"],
            },
        ],
        "validation": {
            "objects_have_unique_ids": len({spec["id"] for spec in specs}) == len(specs),
            "environment_scales_applied": validation["scales_applied"],
            "robot_model_unchanged": validation["robot_model_unchanged"],
            "current_robot_pose_collision_free": not validation["robot_overlap_names"],
            "reference_robot_pose_is_collision_constraint": False,
            "reference_robot_pose_environment_overlap_allowed": True,
            "current_robot_pose_check_type": "conservative_world_AABB",
            "current_robot_pose_overlap_names": validation["robot_overlap_names"],
            "aabb_overlap_warnings": validation["overlap_warnings"],
            "structurally_grounded_or_supported": not validation["grounding_failures"],
            "grounding_failures": validation["grounding_failures"],
            "xml_well_formed": validation["xml_well_formed"],
            "mujoco_load_check": validation["mujoco_load_check"],
            "obstacle_count": len(specs),
            "collection_counts": counts,
            **validation["free_space"],
            "connectivity_notes": [
                "Occupancy uses conservatively inflated box geometry and 6-connected voxels.",
                "The check is geometric screening, not a replacement for full-robot SE(3) collision checking.",
                "The imported robot pose is a collision-size reference and is intentionally allowed to overlap the solid floor.",
                "All non-floor obstacles are directly grounded or supported by grounded cube members.",
            ],
        },
    }
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def write_environment_xml(specs):
    material_by_key = {
        "floor": "env_mat_floor",
        "obstacle": "env_mat_obstacle",
        "gate": "env_mat_gate",
        "beam": "env_mat_beam",
        "partial": "env_mat_partial",
    }
    rgba_by_key = {
        key: " ".join(f"{value:.4f}" for value in COLORS[key])
        for key in material_by_key
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<mujocoinclude>',
        '  <asset>',
    ]
    for key, name in material_by_key.items():
        lines.append(f'    <material name="{name}" rgba="{rgba_by_key[key]}"/>')
    lines.extend(['  </asset>', '  <worldbody>'])
    for spec in specs:
        position = " ".join(f"{value:.8g}" for value in spec["position"])
        quaternion = " ".join(f"{value:.8g}" for value in spec_quaternion(spec))
        half = " ".join(f"{0.5 * value:.8g}" for value in spec["size"])
        material = material_by_key[spec["material"]]
        lines.append(
            f'    <geom name="{spec["id"]}" type="box" pos="{position}" '
            f'quat="{quaternion}" size="{half}" material="{material}" '
            'contype="1" conaffinity="1" group="3"/>'
        )
    lines.extend(['  </worldbody>', '</mujocoinclude>', ''])
    XML_PATH.write_text("\n".join(lines), encoding="utf-8")


def validate_mujoco_xml():
    try:
        ET.parse(XML_PATH)
        well_formed = True
    except Exception as exc:
        return False, f"XML parse failed: {exc}"

    try:
        import mujoco
    except Exception:
        return well_formed, "not_available: Python mujoco module is not installed in Blender"

    wrapper = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8", delete=False) as handle:
            wrapper = Path(handle.name)
            handle.write(
                '<mujoco model="environment_validation">\n'
                f'  <include file="{XML_PATH.as_posix()}"/>\n'
                '</mujoco>\n'
            )
        model = mujoco.MjModel.from_xml_path(str(wrapper))
        return well_formed, f"passed: ngeom={model.ngeom}"
    except Exception as exc:
        return well_formed, f"failed: {exc}"
    finally:
        if wrapper is not None and wrapper.exists():
            wrapper.unlink()


def create_preview_assets(collection):
    scene = bpy.context.scene
    camera_data = bpy.data.cameras.new("env_preview_camera_data")
    camera = bpy.data.objects.new("env_preview_camera", camera_data)
    collection.objects.link(camera)
    camera.location = (6.35, -8.75, 6.25)
    target = Vector((0.0, 0.05, 1.45))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 48.0
    camera_data.sensor_width = 36.0
    camera["environment_id"] = ENVIRONMENT_ID
    camera["collision"] = False
    camera["export_to_environment"] = False
    scene.camera = camera

    def add_light(name, light_type, location, energy, color, size=4.0, rotation=None):
        data = bpy.data.lights.new(name + "_data", light_type)
        data.energy = energy
        data.color = color
        if light_type == "AREA":
            data.shape = "DISK"
            data.size = size
        obj = bpy.data.objects.new(name, data)
        collection.objects.link(obj)
        obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        else:
            obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
        obj["environment_id"] = ENVIRONMENT_ID
        obj["collision"] = False
        obj["export_to_environment"] = False
        return obj

    add_light("env_preview_key", "AREA", (4.0, -3.5, 7.5), 430.0, (1.0, 0.82, 0.64), 5.0)
    add_light("env_preview_fill", "AREA", (-4.5, -0.5, 4.5), 260.0, (0.46, 0.68, 1.0), 4.0)
    add_light("env_preview_rim", "AREA", (0.5, 5.5, 6.5), 360.0, (0.67, 0.82, 1.0), 3.5)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.018, 0.026, 0.045, 1.0)
        background.inputs["Strength"].default_value = 0.16

    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(PREVIEW_PATH)
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.color_mode = "RGBA"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass


def embed_build_script():
    if not SCRIPT_PATH.exists():
        return
    text = bpy.data.texts.get(SCRIPT_PATH.name) or bpy.data.texts.new(SCRIPT_PATH.name)
    text.clear()
    text.write(SCRIPT_PATH.read_text(encoding="utf-8"))


def run():
    random.seed(RANDOM_SEED)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"

    robot = analyze_robot()
    original_snapshot = snapshot_existing_scene()
    remove_previous_environment()
    _, collections = create_collection_hierarchy()

    materials = {
        key: make_material(f"env_material_{key}", color, metallic=0.04 if key != "floor" else 0.0)
        for key, color in COLORS.items()
    }
    specs, derived = build_obstacle_specs(robot)
    environment_objects = []
    for spec in specs:
        environment_objects.append(
            add_cube_object(
                spec,
                collections[spec["collection"]],
                materials[spec["material"]],
                collision=True,
                render=True,
            )
        )
    add_boundary_markers(collections["BOUNDARIES"], materials["boundary"])

    sample_min, sample_max = sampling_bounds(robot)
    overlap_warnings = find_overlap_warnings(specs)
    robot_overlap_names = current_robot_overlap_names(robot, specs)
    unsupported_objects = grounding_failures(specs)
    free_space = validate_free_space(robot, specs, sample_min, sample_max)
    scales_applied = all(
        all(abs(float(value) - 1.0) < 1e-7 for value in obj.scale)
        and not obj.modifiers
        and obj.parent is None
        for obj in environment_objects
    )
    changed_existing_objects = verify_existing_scene_unchanged(original_snapshot)

    write_environment_xml(specs)
    xml_well_formed, mujoco_load_check = validate_mujoco_xml()
    validation = {
        "scales_applied": scales_applied,
        "robot_model_unchanged": not changed_existing_objects,
        "changed_existing_objects": changed_existing_objects,
        "robot_overlap_names": robot_overlap_names,
        "grounding_failures": unsupported_objects,
        "overlap_warnings": overlap_warnings,
        "free_space": free_space,
        "xml_well_formed": xml_well_formed,
        "mujoco_load_check": mujoco_load_check,
    }
    data = write_environment_json(robot, specs, derived, validation, sample_min, sample_max)

    create_preview_assets(collections["DEBUG"])
    embed_build_script()
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), check_existing=False)

    summary = {
        "base_link_candidates": robot["base_link_candidates"],
        "base_link": robot["base_link"].name,
        "base_collision_objects": [obj.name for obj in robot["base_rigid_objects"]],
        "base_collision_aabb_min": rounded(robot["base_local"]["min"]),
        "base_collision_aabb_max": rounded(robot["base_local"]["max"]),
        "base_collision_size_xyz": rounded(robot["size"]),
        "base_collision_center_xyz": rounded(robot["base_local"]["center"]),
        "base_collision_max_horizontal_extent": round(robot["horizontal_max"], 6),
        "base_collision_vertical_extent": round(robot["vertical_extent"], 6),
        "safety_margin": round(robot["safety_margin"], 6),
        "environment_collision_cube_count": len(specs),
        "collection_counts": data["validation"]["collection_counts"],
        "narrowest_passage_width": round(derived["gate_gap"], 6),
        "orientation_sensitive": True,
        "sampling_bounds": {"min": rounded(sample_min), "max": rounded(sample_max)},
        "coarse_free_sample_ratio": free_space["coarse_free_sample_ratio"],
        "free_space_component_count": free_space["free_space_component_count"],
        "largest_free_space_component_ratio": free_space["largest_free_space_component_ratio"],
        "aabb_overlap_warnings": overlap_warnings,
        "current_robot_pose_overlap_names": robot_overlap_names,
        "reference_robot_pose_overlap_allowed": True,
        "grounding_failures": unsupported_objects,
        "robot_model_unchanged": not changed_existing_objects,
        "mujoco_load_check": mujoco_load_check,
        "outputs": {
            "blend": str(BLEND_PATH),
            "json": str(JSON_PATH),
            "xml": str(XML_PATH),
            "preview": str(PREVIEW_PATH),
            "script": str(SCRIPT_PATH),
        },
    }
    print("ENVIRONMENT_BUILD_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
