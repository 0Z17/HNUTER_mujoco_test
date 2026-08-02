#!/usr/bin/env python3
"""Convert a URDF robot to a Lula-editor-compatible USD asset with Isaac Sim.

Run this file with Isaac Sim's ``python.sh`` (or use the adjacent shell
wrapper).  The script resolves package:// mesh URIs in a temporary URDF, so
the source URDF is never modified.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = SCRIPT_DIR / "urdf" / "HDJQR-0102-0055.SLDASM.urdf"
DEFAULT_MESH_ROOT = SCRIPT_DIR / "meshes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert URDF to USD with Isaac Sim's URDF Importer API."
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help=f"Input URDF (default: {DEFAULT_URDF})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Main output USD. By default: <script_dir>/usd/<urdf_stem>/<urdf_stem>.usd",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=DEFAULT_MESH_ROOT,
        help=f"Fallback directory containing URDF meshes (default: {DEFAULT_MESH_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow updating an already generated output asset.",
    )
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument(
        "--floating-base",
        dest="floating_base",
        action="store_true",
        default=True,
        help="Do not add a fixed joint between the world and base_link (default for this UAV).",
    )
    base_group.add_argument(
        "--fixed-base",
        dest="floating_base",
        action="store_false",
        help="Add a fixed joint between the world and base_link.",
    )
    parser.add_argument(
        "--merge-fixed-joints",
        action="store_true",
        help="Merge links connected by fixed joints. Disabled by default to preserve link frames for Lula.",
    )
    parser.add_argument(
        "--keep-instanceable",
        action="store_true",
        help="Keep Isaac Sim's instanceable mesh references. This disables Lula's automatic sphere generation.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return args


def default_output_path(urdf_path: Path) -> Path:
    return SCRIPT_DIR / "usd" / urdf_path.stem / f"{urdf_path.stem}.usd"


def unique_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen:
            seen.add(normalized)
            result.append(path)
    return result


def resolve_mesh_uri(uri: str, urdf_path: Path, mesh_root: Path) -> Path:
    """Resolve common URDF mesh URI forms to an existing local file."""

    uri = uri.strip()
    candidates: list[Path] = []

    if uri.startswith("package://"):
        package_path = Path(unquote(uri[len("package://") :]))
        parts = package_path.parts
        if len(parts) < 2:
            raise ValueError(f"Invalid package URI: {uri}")

        package_name = parts[0]
        package_relative = Path(*parts[1:])
        search_bases = [urdf_path.parent, *urdf_path.parent.parents]
        for base in search_bases:
            candidates.append(base / package_name / package_relative)
            candidates.append(base / package_relative)

        # SolidWorks exports often place every STL in one sibling meshes/
        # directory even though the URDF retains a ROS package URI.
        candidates.append(mesh_root / package_relative)
        candidates.append(mesh_root / package_relative.name)
    elif uri.startswith("file://"):
        parsed = urlparse(uri)
        candidates.append(Path(unquote(parsed.path)))
    else:
        mesh_path = Path(unquote(uri))
        if mesh_path.is_absolute():
            candidates.append(mesh_path)
        else:
            candidates.append(urdf_path.parent / mesh_path)
            candidates.append(mesh_root / mesh_path)
            candidates.append(mesh_root / mesh_path.name)

    checked = unique_paths(candidates)
    for candidate in checked:
        if candidate.is_file():
            return candidate.resolve()

    checked_text = "\n    ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Cannot resolve mesh URI {uri!r}. Checked:\n    {checked_text}")


def make_resolved_urdf(source: Path, destination: Path, mesh_root: Path) -> tuple[int, int, int]:
    """Write a temporary URDF whose mesh filenames are absolute paths."""

    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected <robot> as XML root in {source}")

    mesh_elements = list(root.iter("mesh"))
    if not mesh_elements:
        raise ValueError(f"No <mesh> elements found in {source}")

    for mesh in mesh_elements:
        filename = mesh.get("filename")
        if not filename:
            raise ValueError("Found a <mesh> element without a filename attribute")
        mesh.set("filename", resolve_mesh_uri(filename, source, mesh_root).as_posix())

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return len(list(root.iter("link"))), len(list(root.iter("joint"))), len(mesh_elements)


def deinstance_robot_meshes(stage) -> list[str]:
    """Make imported mesh reference roots editable by Lula's sphere generator."""

    instance_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsInstance()]
    if not instance_paths:
        return []

    stage.SetEditTarget(stage.GetRootLayer())
    for prim_path in instance_paths:
        stage.GetPrimAtPath(prim_path).SetInstanceable(False)
    stage.GetRootLayer().Save()
    return instance_paths


def validate_usd(output_path: Path, expected_links: int, expected_joints: int) -> dict[str, object]:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(output_path.as_posix(), load=Usd.Stage.LoadAll)
    if not stage:
        raise RuntimeError(f"USD stage cannot be opened: {output_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("Generated USD has no default prim")

    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    meshes = [UsdGeom.Mesh(prim) for prim in prims if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(
            "Generated USD contains no mesh prims. Check the original URDF package:// paths."
        )

    empty_meshes = [str(mesh.GetPath()) for mesh in meshes if not (mesh.GetPointsAttr().Get() or [])]
    if empty_meshes:
        raise RuntimeError(f"Generated USD contains empty meshes: {empty_meshes}")

    instance_proxies = [str(prim.GetPath()) for prim in prims if prim.IsInstanceProxy()]
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)]
    rigid_bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    articulations = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]

    rigid_body_paths = sorted(
        (str(prim.GetPath()) for prim in rigid_bodies), key=len, reverse=True
    )
    links_with_meshes: set[str] = set()
    unassigned_meshes: list[str] = []
    for mesh in meshes:
        mesh_path = str(mesh.GetPath())
        owning_link = next(
            (link_path for link_path in rigid_body_paths if mesh_path.startswith(f"{link_path}/")),
            None,
        )
        if owning_link is None:
            unassigned_meshes.append(mesh_path)
        else:
            links_with_meshes.add(owning_link)

    # A fixed-base import adds one root joint in addition to the URDF joints.
    if len(joints) < expected_joints:
        raise RuntimeError(
            f"Generated USD has only {len(joints)} joints; URDF contains {expected_joints}."
        )
    if len(rigid_bodies) < expected_links - 1:
        raise RuntimeError(
            f"Generated USD has only {len(rigid_bodies)} rigid bodies; URDF contains {expected_links} links."
        )
    if not articulations:
        raise RuntimeError("Generated USD has no ArticulationRootAPI prim")
    if unassigned_meshes:
        raise RuntimeError(f"Meshes are not nested under robot links: {unassigned_meshes}")
    if len(links_with_meshes) < expected_links:
        raise RuntimeError(
            f"Only {len(links_with_meshes)} of {expected_links} links contain meshes."
        )

    return {
        "default_prim": str(default_prim.GetPath()),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "mesh_count": len(meshes),
        "joint_count": len(joints),
        "rigid_body_count": len(rigid_bodies),
        "links_with_meshes": len(links_with_meshes),
        "articulation_count": len(articulations),
        "instance_proxy_count": len(instance_proxies),
    }


def convert(args: argparse.Namespace) -> Path:
    urdf_path = args.urdf.expanduser().resolve()
    mesh_root = args.mesh_root.expanduser().resolve()
    output_path = (args.output or default_output_path(urdf_path)).expanduser().resolve()

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    if output_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"Output must end in .usd, .usda, or .usdc: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\nUse --overwrite to update it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SimulationApp parses sys.argv as Kit arguments. The script arguments have
    # already been consumed, so keep Kit from seeing them a second time.
    sys.argv = [sys.argv[0]]

    with tempfile.TemporaryDirectory(prefix="isaac_urdf_import_") as temp_dir:
        resolved_urdf = Path(temp_dir) / urdf_path.name
        link_count, urdf_joint_count, mesh_ref_count = make_resolved_urdf(
            urdf_path, resolved_urdf, mesh_root
        )
        print(
            f"[URDF] {link_count} links, {urdf_joint_count} joints, "
            f"{mesh_ref_count} resolved mesh references",
            flush=True,
        )

        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "hide_ui": True,
                "disable_viewport_updates": True,
            }
        )
        try:
            import omni.kit.commands
            from isaacsim.core.utils.extensions import enable_extension
            from pxr import Usd

            if not enable_extension("isaacsim.asset.importer.urdf"):
                raise RuntimeError("Could not enable isaacsim.asset.importer.urdf")
            simulation_app.update()

            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            if not status:
                raise RuntimeError("URDFCreateImportConfig failed")

            import_config.set_merge_fixed_joints(args.merge_fixed_joints)
            import_config.set_replace_cylinders_with_capsules(False)
            import_config.set_convex_decomp(False)
            import_config.set_import_inertia_tensor(True)
            import_config.set_fix_base(not args.floating_base)
            import_config.set_self_collision(False)
            import_config.set_density(0.0)
            import_config.set_distance_scale(1.0)
            import_config.set_default_drive_type(1)
            import_config.set_default_drive_strength(1000.0)
            import_config.set_default_position_drive_damping(100.0)
            import_config.set_up_vector(0.0, 0.0, 1.0)
            import_config.set_make_default_prim(True)
            import_config.set_parse_mimic(True)
            import_config.set_create_physics_scene(False)
            import_config.set_collision_from_visuals(False)

            status, imported_prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=resolved_urdf.as_posix(),
                import_config=import_config,
                dest_path=output_path.as_posix(),
                get_articulation_root=True,
            )
            if not status:
                raise RuntimeError("URDFParseAndImportFile failed")

            # Let any asset-converter tasks and USD notices finish before the
            # stage is reopened for post-processing.
            for _ in range(5):
                simulation_app.update()

            stage = Usd.Stage.Open(output_path.as_posix(), load=Usd.Stage.LoadAll)
            if not stage:
                raise RuntimeError(f"Cannot reopen generated USD: {output_path}")

            deinstanced: list[str] = []
            if not args.keep_instanceable:
                deinstanced = deinstance_robot_meshes(stage)
                stage = None

            result = validate_usd(output_path, link_count, urdf_joint_count)
            if not args.keep_instanceable and result["instance_proxy_count"]:
                raise RuntimeError(
                    "Some meshes remain instance proxies and cannot be used by Lula's automatic sphere generator"
                )

            print(f"[USD] imported articulation: {imported_prim_path}", flush=True)
            print(f"[USD] de-instanced reference roots: {len(deinstanced)}", flush=True)
            print(
                "[USD] validation: "
                + ", ".join(f"{key}={value}" for key, value in result.items()),
                flush=True,
            )
            print(f"[DONE] {output_path}", flush=True)
        finally:
            simulation_app.close()

    return output_path


def main() -> int:
    try:
        convert(parse_args())
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
