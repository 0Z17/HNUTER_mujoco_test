"""COAL collision checking for a rigid URDF body moving in SE(3).

The vehicle is represented as a collection of collision geometries with fixed
poses relative to one URDF link.  Geometry and narrow-phase query objects are
created once.  A query therefore only composes the sampled world pose of that
link with the precomputed local geometry poses before asking COAL to collide
the vehicle with the static environment.

COAL was named ``hpp-fcl`` before version 3.0.  The current ``coal`` Python
module is preferred; importing the legacy ``hppfcl`` module is supported to
make the interface usable with ROS distributions that still ship that name.
"""

from __future__ import annotations

import importlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mppi.quaternion import (
    normalize_quaternion,
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
)


FloatArray = NDArray[np.float64]
GeometryKind = Literal["box", "sphere", "cylinder", "capsule", "mesh"]


def _finite_vector(
    value: ArrayLike,
    length: int,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array.copy()


@dataclass(frozen=True)
class CollisionPose:
    """Translation and ``wxyz`` quaternion for a collision geometry."""

    position: FloatArray
    quaternion: FloatArray

    def __post_init__(self) -> None:
        position = _finite_vector(self.position, 3, "position")
        quaternion = normalize_quaternion(
            _finite_vector(self.quaternion, 4, "quaternion")
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion)

    @classmethod
    def identity(cls) -> CollisionPose:
        return cls(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_rpy(
        cls,
        position: ArrayLike = (0.0, 0.0, 0.0),
        rpy: ArrayLike = (0.0, 0.0, 0.0),
    ) -> CollisionPose:
        return cls(position, quaternion_from_euler(rpy))


@dataclass(frozen=True)
class CollisionGeometry:
    """Backend-independent description of one supported COAL geometry."""

    kind: GeometryKind
    parameters: tuple[float, ...]
    mesh_path: Path | None = None

    def __post_init__(self) -> None:
        parameters = tuple(float(value) for value in self.parameters)
        if not parameters or not all(
            math.isfinite(value) and value > 0.0 for value in parameters
        ):
            raise ValueError("geometry parameters must be finite and positive")
        expected_count = {
            "box": 3,
            "sphere": 1,
            "cylinder": 2,
            "capsule": 2,
            "mesh": 3,
        }.get(self.kind)
        if expected_count is None:
            raise ValueError(f"unsupported collision geometry: {self.kind}")
        if len(parameters) != expected_count:
            raise ValueError(
                f"{self.kind} expects {expected_count} parameters, "
                f"received {len(parameters)}"
            )
        mesh_path = self.mesh_path
        if self.kind == "mesh":
            if mesh_path is None:
                raise ValueError("mesh geometry requires mesh_path")
            mesh_path = Path(mesh_path).expanduser().resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"collision mesh not found: {mesh_path}")
        elif mesh_path is not None:
            raise ValueError("mesh_path is only valid for mesh geometry")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "mesh_path", mesh_path)

    @classmethod
    def box(cls, size: ArrayLike) -> CollisionGeometry:
        return cls("box", tuple(_finite_vector(size, 3, "box size")))

    @classmethod
    def sphere(cls, radius: float) -> CollisionGeometry:
        return cls("sphere", (radius,))

    @classmethod
    def cylinder(cls, radius: float, length: float) -> CollisionGeometry:
        return cls("cylinder", (radius, length))

    @classmethod
    def capsule(cls, radius: float, length: float) -> CollisionGeometry:
        return cls("capsule", (radius, length))

    @classmethod
    def mesh(
        cls,
        path: str | Path,
        scale: ArrayLike = (1.0, 1.0, 1.0),
    ) -> CollisionGeometry:
        return cls(
            "mesh",
            tuple(_finite_vector(scale, 3, "mesh scale")),
            Path(path),
        )


@dataclass(frozen=True)
class FixedCollisionGeometry:
    """One named geometry fixed to a rigid body's reference frame."""

    name: str
    geometry: CollisionGeometry
    local_pose: CollisionPose

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("collision geometry name must not be empty")


@dataclass(frozen=True)
class StaticCollisionObject:
    """One named environment geometry with a fixed world pose."""

    name: str
    geometry: CollisionGeometry
    pose: CollisionPose

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("environment object name must not be empty")

    @classmethod
    def sphere(
        cls,
        name: str,
        center: ArrayLike,
        radius: float,
    ) -> StaticCollisionObject:
        return cls(
            name,
            CollisionGeometry.sphere(radius),
            CollisionPose(center, (1.0, 0.0, 0.0, 0.0)),
        )

    @classmethod
    def box(
        cls,
        name: str,
        size: ArrayLike,
        position: ArrayLike = (0.0, 0.0, 0.0),
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
    ) -> StaticCollisionObject:
        return cls(
            name,
            CollisionGeometry.box(size),
            CollisionPose(position, quaternion),
        )

    @classmethod
    def cylinder(
        cls,
        name: str,
        radius: float,
        length: float,
        position: ArrayLike = (0.0, 0.0, 0.0),
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
    ) -> StaticCollisionObject:
        return cls(
            name,
            CollisionGeometry.cylinder(radius, length),
            CollisionPose(position, quaternion),
        )

    @classmethod
    def mesh(
        cls,
        name: str,
        path: str | Path,
        position: ArrayLike = (0.0, 0.0, 0.0),
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
        scale: ArrayLike = (1.0, 1.0, 1.0),
    ) -> StaticCollisionObject:
        return cls(
            name,
            CollisionGeometry.mesh(path, scale),
            CollisionPose(position, quaternion),
        )


@dataclass(frozen=True)
class RigidURDFCollisionModel:
    """Collision geometry attached directly to one URDF link."""

    link_name: str
    geometries: tuple[FixedCollisionGeometry, ...]
    urdf_path: Path

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        link_name: str = "base_link",
        *,
        package_roots: Mapping[str, str | Path] | None = None,
        require_all_collisions_on_link: bool = True,
    ) -> RigidURDFCollisionModel:
        """Load collision elements fixed directly to ``link_name``.

        ``require_all_collisions_on_link`` protects the rigid-body assumption:
        it rejects active collision elements on any other URDF link.  XML
        comments (such as the visual meshes commented out in the HNUTER URDF)
        are naturally ignored by the parser.
        """

        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")
        root = ET.parse(path).getroot()
        links = {
            element.get("name", ""): element
            for element in root.findall("link")
        }
        if link_name not in links:
            raise ValueError(f"URDF has no link named {link_name!r}")
        if require_all_collisions_on_link:
            other_links = [
                name
                for name, element in links.items()
                if name != link_name and element.findall("collision")
            ]
            if other_links:
                raise ValueError(
                    "rigid collision model requires every active collision "
                    f"element on {link_name!r}; found collisions on "
                    + ", ".join(repr(name) for name in other_links)
                )

        geometries = tuple(
            _parse_urdf_collision(
                collision,
                index,
                path,
                package_roots or {},
            )
            for index, collision in enumerate(
                links[link_name].findall("collision")
            )
        )
        if not geometries:
            raise ValueError(
                f"URDF link {link_name!r} has no active collision elements"
            )
        return cls(link_name, geometries, path)


@dataclass(frozen=True)
class CoalCollisionResult:
    """Minimum-clearance result for one vehicle pose."""

    collision_free: bool
    signed_clearance: float
    vehicle_geometry: str | None
    environment_object: str | None


@dataclass(frozen=True)
class _CoalShape:
    name: str
    geometry: Any
    rotation: FloatArray
    translation: FloatArray


@dataclass(frozen=True)
class _CoalPair:
    vehicle_index: int
    environment_index: int
    collide: Any
    distance: Any


class CoalCollisionChecker:
    """Pose-aware collision checker suitable for ``OMPLSE3Planner``.

    The checker is intentionally independent of OMPL.  Its public methods take
    NumPy-compatible positions and ``wxyz`` quaternions, making it usable for
    post-planning validation and for collision checks outside the planner too.
    """

    def __init__(
        self,
        vehicle: RigidURDFCollisionModel,
        environment: Sequence[StaticCollisionObject],
        *,
        safety_margin: float = 0.0,
    ) -> None:
        if not math.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin must be finite and non-negative")
        self.vehicle = vehicle
        self.environment = tuple(environment)
        self.safety_margin = float(safety_margin)
        self._coal = _load_coal()
        self._transform_type = getattr(
            self._coal,
            "Transform3s",
            getattr(self._coal, "Transform3f", None),
        )
        if self._transform_type is None:
            raise RuntimeError("COAL binding has no Transform3s/Transform3f")
        mesh_loader = self._coal.MeshLoader()
        self._vehicle_shapes = tuple(
            _CoalShape(
                item.name,
                _make_coal_geometry(self._coal, mesh_loader, item.geometry),
                quaternion_to_rotation_matrix(item.local_pose.quaternion),
                item.local_pose.position,
            )
            for item in vehicle.geometries
        )
        self._environment_shapes = tuple(
            _CoalShape(
                item.name,
                _make_coal_geometry(self._coal, mesh_loader, item.geometry),
                quaternion_to_rotation_matrix(item.pose.quaternion),
                item.pose.position,
            )
            for item in self.environment
        )
        self._environment_transforms = tuple(
            self._make_transform(item.rotation, item.translation)
            for item in self._environment_shapes
        )
        self._pairs = tuple(
            _CoalPair(
                vehicle_index,
                environment_index,
                self._coal.ComputeCollision(
                    vehicle_shape.geometry,
                    environment_shape.geometry,
                ),
                self._coal.ComputeDistance(
                    vehicle_shape.geometry,
                    environment_shape.geometry,
                ),
            )
            for vehicle_index, vehicle_shape in enumerate(self._vehicle_shapes)
            for environment_index, environment_shape in enumerate(
                self._environment_shapes
            )
        )
        self._collision_request = self._coal.CollisionRequest()
        self._collision_request.security_margin = self.safety_margin
        self._distance_request = self._coal.DistanceRequest()
        if hasattr(self._distance_request, "enable_signed_distance"):
            self._distance_request.enable_signed_distance = True

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        environment: Sequence[StaticCollisionObject],
        *,
        link_name: str = "base_link",
        safety_margin: float = 0.0,
        package_roots: Mapping[str, str | Path] | None = None,
    ) -> CoalCollisionChecker:
        vehicle = RigidURDFCollisionModel.from_urdf(
            urdf_path,
            link_name,
            package_roots=package_roots,
        )
        return cls(vehicle, environment, safety_margin=safety_margin)

    @property
    def vehicle_geometry_count(self) -> int:
        return len(self._vehicle_shapes)

    @property
    def environment_object_count(self) -> int:
        return len(self._environment_shapes)

    def is_collision_free(
        self,
        position: ArrayLike,
        quaternion: ArrayLike,
    ) -> bool:
        """Return whether one world-frame vehicle pose is collision-free."""

        position_array = _finite_vector(position, 3, "position")
        quaternion_array = normalize_quaternion(
            _finite_vector(quaternion, 4, "quaternion")
        )
        if not self._pairs:
            return True
        vehicle_transforms = self._vehicle_transforms(
            position_array, quaternion_array
        )
        for pair in self._pairs:
            result = self._coal.CollisionResult()
            pair.collide(
                vehicle_transforms[pair.vehicle_index],
                self._environment_transforms[pair.environment_index],
                self._collision_request,
                result,
            )
            if result.isCollision():
                return False
        return True

    def clearance(
        self,
        positions: ArrayLike,
        quaternions: ArrayLike,
    ) -> FloatArray:
        """Return safety-margin-adjusted signed clearance for SE(3) poses."""

        position_array, quaternion_array = _validate_pose_arrays(
            positions, quaternions
        )
        output = np.empty(position_array.shape[:-1], dtype=np.float64)
        flat_output = output.reshape(-1)
        for index, (position, quaternion) in enumerate(
            zip(
                position_array.reshape(-1, 3),
                quaternion_array.reshape(-1, 4),
            )
        ):
            flat_output[index] = self.check_pose(
                position, quaternion
            ).signed_clearance
        return output

    def check_pose(
        self,
        position: ArrayLike,
        quaternion: ArrayLike,
    ) -> CoalCollisionResult:
        """Return minimum clearance and the responsible geometry pair."""

        position_array = _finite_vector(position, 3, "position")
        quaternion_array = normalize_quaternion(
            _finite_vector(quaternion, 4, "quaternion")
        )
        if not self._pairs:
            return CoalCollisionResult(True, math.inf, None, None)
        vehicle_transforms = self._vehicle_transforms(
            position_array, quaternion_array
        )
        minimum = math.inf
        minimum_pair: _CoalPair | None = None
        for pair in self._pairs:
            result = self._coal.DistanceResult()
            pair.distance(
                vehicle_transforms[pair.vehicle_index],
                self._environment_transforms[pair.environment_index],
                self._distance_request,
                result,
            )
            clearance = float(result.min_distance) - self.safety_margin
            if clearance < minimum:
                minimum = clearance
                minimum_pair = pair
        assert minimum_pair is not None
        return CoalCollisionResult(
            collision_free=minimum > 0.0,
            signed_clearance=minimum,
            vehicle_geometry=self._vehicle_shapes[
                minimum_pair.vehicle_index
            ].name,
            environment_object=self._environment_shapes[
                minimum_pair.environment_index
            ].name,
        )

    def _vehicle_transforms(
        self,
        position: FloatArray,
        quaternion: FloatArray,
    ) -> tuple[Any, ...]:
        root_rotation = quaternion_to_rotation_matrix(quaternion)
        return tuple(
            self._make_transform(
                root_rotation @ shape.rotation,
                position + root_rotation @ shape.translation,
            )
            for shape in self._vehicle_shapes
        )

    def _make_transform(
        self,
        rotation: FloatArray,
        translation: FloatArray,
    ) -> Any:
        return self._transform_type(
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
        )


def _load_coal() -> Any:
    for module_name in ("coal", "hppfcl"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "COAL Python bindings were not found. Install the official package "
        "with `conda install -c conda-forge coal` (the old module name "
        "`hppfcl` is also supported)."
    )


def _make_coal_geometry(
    coal: Any,
    mesh_loader: Any,
    specification: CollisionGeometry,
) -> Any:
    parameters = specification.parameters
    if specification.kind == "box":
        return coal.Box(*parameters)
    if specification.kind == "sphere":
        return coal.Sphere(parameters[0])
    if specification.kind == "cylinder":
        # COAL constructors take full length and store halfLength internally.
        return coal.Cylinder(parameters[0], parameters[1])
    if specification.kind == "capsule":
        return coal.Capsule(parameters[0], parameters[1])
    if specification.kind == "mesh":
        assert specification.mesh_path is not None
        return mesh_loader.load(
            str(specification.mesh_path),
            np.asarray(parameters, dtype=np.float64),
        )
    raise AssertionError(f"unhandled geometry kind: {specification.kind}")


def _validate_pose_arrays(
    positions: ArrayLike,
    quaternions: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    position_array = np.asarray(positions, dtype=np.float64)
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    if position_array.shape[-1:] != (3,) or not np.all(
        np.isfinite(position_array)
    ):
        raise ValueError("positions must be finite with trailing dimension 3")
    if quaternion_array.shape[-1:] != (4,) or not np.all(
        np.isfinite(quaternion_array)
    ):
        raise ValueError(
            "quaternions must be finite with trailing dimension 4"
        )
    if position_array.shape[:-1] != quaternion_array.shape[:-1]:
        raise ValueError("positions and quaternions must have matching shapes")
    return position_array, normalize_quaternion(quaternion_array)


def _parse_urdf_collision(
    element: ET.Element,
    index: int,
    urdf_path: Path,
    package_roots: Mapping[str, str | Path],
) -> FixedCollisionGeometry:
    name = element.get("name") or f"collision_{index}"
    origin = element.find("origin")
    xyz = _parse_float_attribute(origin, "xyz", 3, (0.0, 0.0, 0.0))
    rpy = _parse_float_attribute(origin, "rpy", 3, (0.0, 0.0, 0.0))
    geometry_element = element.find("geometry")
    if geometry_element is None:
        raise ValueError(f"URDF collision {name!r} has no geometry")
    children = list(geometry_element)
    if len(children) != 1:
        raise ValueError(
            f"URDF collision {name!r} must contain exactly one geometry"
        )
    shape = children[0]
    if shape.tag == "box":
        size = _required_float_attribute(shape, "size", 3, name)
        geometry = CollisionGeometry.box(size)
    elif shape.tag == "sphere":
        radius = _required_scalar_attribute(shape, "radius", name)
        geometry = CollisionGeometry.sphere(radius)
    elif shape.tag == "cylinder":
        radius = _required_scalar_attribute(shape, "radius", name)
        length = _required_scalar_attribute(shape, "length", name)
        geometry = CollisionGeometry.cylinder(radius, length)
    elif shape.tag == "mesh":
        filename = shape.get("filename")
        if not filename:
            raise ValueError(f"URDF collision {name!r} mesh has no filename")
        scale = _parse_float_attribute(
            shape, "scale", 3, (1.0, 1.0, 1.0)
        )
        geometry = CollisionGeometry.mesh(
            _resolve_urdf_mesh(filename, urdf_path, package_roots),
            scale,
        )
    else:
        raise ValueError(
            f"URDF collision {name!r} uses unsupported {shape.tag!r} geometry"
        )
    return FixedCollisionGeometry(
        name,
        geometry,
        CollisionPose.from_rpy(xyz, rpy),
    )


def _parse_float_attribute(
    element: ET.Element | None,
    attribute: str,
    count: int,
    default: Iterable[float],
) -> tuple[float, ...]:
    if element is None or element.get(attribute) is None:
        return tuple(float(value) for value in default)
    values = tuple(float(value) for value in element.get(attribute, "").split())
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"URDF attribute {attribute!r} must contain {count} finite numbers"
        )
    return values


def _required_float_attribute(
    element: ET.Element,
    attribute: str,
    count: int,
    collision_name: str,
) -> tuple[float, ...]:
    if element.get(attribute) is None:
        raise ValueError(
            f"URDF collision {collision_name!r} has no {attribute!r} attribute"
        )
    return _parse_float_attribute(element, attribute, count, ())


def _required_scalar_attribute(
    element: ET.Element,
    attribute: str,
    collision_name: str,
) -> float:
    return _required_float_attribute(
        element, attribute, 1, collision_name
    )[0]


def _resolve_urdf_mesh(
    filename: str,
    urdf_path: Path,
    package_roots: Mapping[str, str | Path],
) -> Path:
    if filename.startswith("file://"):
        return Path(filename[7:]).expanduser().resolve()
    if filename.startswith("package://"):
        package_path = filename[len("package://") :]
        package_name, separator, relative = package_path.partition("/")
        if not separator:
            raise ValueError(f"invalid URDF package URI: {filename}")
        if package_name in package_roots:
            return (
                Path(package_roots[package_name]).expanduser() / relative
            ).resolve()
        # Most exported URDF packages keep ``urdf/`` and ``meshes/`` beside
        # each other. This resolves that layout without requiring ROS.
        sibling_candidate = (urdf_path.parent.parent / relative).resolve()
        if sibling_candidate.is_file():
            return sibling_candidate
        raise FileNotFoundError(
            f"cannot resolve {filename!r}; pass package_roots={{"
            f"{package_name!r}: '/path/to/package'}}"
        )
    return (urdf_path.parent / filename).resolve()


__all__ = [
    "CoalCollisionChecker",
    "CoalCollisionResult",
    "CollisionGeometry",
    "CollisionPose",
    "FixedCollisionGeometry",
    "RigidURDFCollisionModel",
    "StaticCollisionObject",
]
