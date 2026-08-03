"""Decoupled Rerun recording support for robotics simulations.

This module intentionally does not import MuJoCo, OMPL, or MPPI.  Callers
provide NumPy-compatible poses, joint coordinates, paths, and scalar channels.
Rerun itself is also imported lazily, so simulations that do not enable
recording keep working without the optional ``rerun-sdk`` dependency.
"""

from __future__ import annotations

import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence
from xml.etree import ElementTree

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
Color = tuple[int, int, int, int]
_PROJECT_DIR = Path(__file__).resolve().parent
_DEFAULT_ROBOT_URDF = (
    _PROJECT_DIR
    / "etc"
    / "URDF-for-gazebo"
    / "urdf"
    / "HDJQR-0102-0055.SLDASM.urdf"
)
_DEFAULT_ROBOT_MJCF = _PROJECT_DIR / "hnuter206_4_5kg.xml"


@dataclass(frozen=True)
class RerunRecorderConfig:
    """Output and visualization options for :class:`RerunSimulationRecorder`."""

    application_id: str = "hnuter_ompl_mppi"
    recording_id: str | None = None
    recording_path: Path | None = None
    spawn_viewer: bool = False
    viewer_port: int = 9876
    trace_update_stride: int = 2
    robot_urdf_path: Path = _DEFAULT_ROBOT_URDF
    robot_mjcf_path: Path = _DEFAULT_ROBOT_MJCF

    def __post_init__(self) -> None:
        if not self.application_id.strip():
            raise ValueError("application_id must not be empty")
        if self.recording_path is None and not self.spawn_viewer:
            raise ValueError(
                "at least one Rerun sink is required: recording_path or "
                "spawn_viewer"
            )
        if not 1 <= self.viewer_port <= 65535:
            raise ValueError("viewer_port must lie in [1, 65535]")
        if self.trace_update_stride < 1:
            raise ValueError("trace_update_stride must be positive")
        object.__setattr__(
            self, "robot_urdf_path", Path(self.robot_urdf_path).resolve()
        )
        object.__setattr__(
            self, "robot_mjcf_path", Path(self.robot_mjcf_path).resolve()
        )


@dataclass(frozen=True)
class Pose3D:
    """Position plus a ``[w, x, y, z]`` unit quaternion."""

    position: FloatArray
    quaternion_wxyz: FloatArray

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        quaternion = _vector(
            self.quaternion_wxyz, 4, "quaternion_wxyz"
        )
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-12:
            raise ValueError("quaternion_wxyz norm must be nonzero")
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self, "quaternion_wxyz", quaternion / norm
        )


@dataclass(frozen=True)
class Sphere3D:
    """A sphere used for static scene visualization."""

    center: FloatArray
    radius: float
    label: str = "obstacle"
    color: Color = (235, 80, 45, 90)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "center", _vector(self.center, 3, "center")
        )
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        _validate_color(self.color)


@dataclass(frozen=True)
class Box3D:
    """An oriented static box used for environment visualization."""

    center: FloatArray
    half_size: FloatArray
    quaternion_wxyz: FloatArray
    label: str = "environment box"
    color: Color = (80, 120, 180, 150)

    def __post_init__(self) -> None:
        center = _vector(self.center, 3, "center")
        half_size = _vector(self.half_size, 3, "half_size")
        quaternion = _vector(
            self.quaternion_wxyz, 4, "quaternion_wxyz"
        )
        if np.any(half_size <= 0.0):
            raise ValueError("box half_size entries must be positive")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-12:
            raise ValueError("quaternion_wxyz norm must be nonzero")
        if not self.label:
            raise ValueError("box label must not be empty")
        _validate_color(self.color)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "half_size", half_size)
        object.__setattr__(
            self, "quaternion_wxyz", quaternion / norm
        )


@dataclass(frozen=True)
class _RobotLink:
    """One visual link, positioned using the matching MJCF body."""

    name: str
    parent: str | None
    joint_name: str | None
    joint_axis: FloatArray
    translation: FloatArray
    quaternion_wxyz: FloatArray
    mesh_path: Path
    visual_translation: FloatArray
    visual_quaternion_wxyz: FloatArray
    visual_scale: FloatArray
    color: Color


@dataclass(frozen=True)
class _RobotModel:
    """URDF visuals combined with the authoritative MJCF kinematics."""

    root_link: str
    links: tuple[_RobotLink, ...]

    @property
    def joint_names(self) -> frozenset[str]:
        return frozenset(
            link.joint_name
            for link in self.links
            if link.joint_name is not None
        )


@dataclass(frozen=True)
class _UrdfVisual:
    mesh_path: Path
    translation: FloatArray
    quaternion_wxyz: FloatArray
    scale: FloatArray
    color: Color


class RerunSimulationRecorder:
    """Record a 3D robotics run to Rerun without simulator coupling.

    Use this class as a context manager or call :meth:`close` explicitly.
    Closing finalizes the RRD footer, making the file immediately seekable and
    suitable for timeline replay.
    """

    _ACTUAL_COLOR: Color = (230, 35, 105, 255)
    _REFERENCE_COLOR: Color = (40, 185, 70, 255)
    _NOMINAL_COLOR: Color = (255, 180, 0, 255)
    _SAMPLE_COLOR: Color = (45, 120, 255, 70)
    _PLANNED_COLOR: Color = (0, 180, 190, 255)
    _RAW_OMPL_COLOR: Color = (145, 85, 215, 190)
    _INTERPOLATING_COLOR: Color = (245, 145, 20, 220)

    def __init__(self, config: RerunRecorderConfig) -> None:
        self.config = config
        self._robot_model = _load_robot_model(
            config.robot_urdf_path,
            config.robot_mjcf_path,
        )
        self._rr = _import_rerun()
        self._exit_stack = ExitStack()
        self._recording = self._exit_stack.enter_context(
            self._rr.RecordingStream(
                config.application_id,
                recording_id=config.recording_id,
            )
        )
        self._closed = False
        self._frame_index = 0
        self._actual_trace: list[FloatArray] = []
        self._configure_sinks()
        self._log_coordinate_system()

    @property
    def recording_path(self) -> Path | None:
        if self.config.recording_path is None:
            return None
        return Path(self.config.recording_path).resolve()

    @property
    def frame_count(self) -> int:
        return self._frame_index

    def __enter__(self) -> "RerunSimulationRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def log_static_scene(
        self,
        *,
        planned_path: ArrayLike,
        raw_ompl_path: ArrayLike | None = None,
        interpolating_baseline_path: ArrayLike | None = None,
        timed_reference_path: ArrayLike | None = None,
        start_pose: Pose3D | None = None,
        goal_pose: Pose3D | None = None,
        waypoint_poses: Sequence[Pose3D] = (),
        obstacles: Sequence[Sphere3D] = (),
        environment_boxes: Sequence[Box3D] = (),
        planned_path_label: str = "OMPL Bi-RRT",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Log static paths, endpoints, environment, and run metadata."""

        self._ensure_open()
        planned = _positions(planned_path, "planned_path")
        self._recording.log(
            "world/paths/global_plan",
            self._rr.LineStrips3D(
                [planned],
                colors=[self._PLANNED_COLOR],
                radii=[0.018],
                labels=[planned_path_label],
            ),
            static=True,
        )
        if raw_ompl_path is not None:
            raw_path = _positions(raw_ompl_path, "raw_ompl_path")
            self._recording.log(
                "world/paths/raw_ompl",
                self._rr.LineStrips3D(
                    [raw_path],
                    colors=[self._RAW_OMPL_COLOR],
                    radii=[0.009],
                    labels=["simplified OMPL RRTConnect"],
                    show_labels=False,
                ),
                static=True,
            )
        if interpolating_baseline_path is not None:
            interpolating_path = _positions(
                interpolating_baseline_path,
                "interpolating_baseline_path",
            )
            self._recording.log(
                "world/paths/interpolating_baseline",
                self._rr.LineStrips3D(
                    [interpolating_path],
                    colors=[self._INTERPOLATING_COLOR],
                    radii=[0.011],
                    labels=["degree-3 interpolating B-spline"],
                    show_labels=False,
                ),
                static=True,
            )
        if timed_reference_path is not None:
            reference = _positions(
                timed_reference_path, "timed_reference_path"
            )
            self._recording.log(
                "world/paths/timed_reference",
                self._rr.LineStrips3D(
                    [reference],
                    colors=[self._REFERENCE_COLOR],
                    radii=[0.012],
                    labels=["time-parameterized reference"],
                ),
                static=True,
            )
        if start_pose is not None:
            self._log_endpoint(
                "world/endpoints/start",
                start_pose,
                (40, 100, 255, 120),
                "start",
            )
        if goal_pose is not None:
            self._log_endpoint(
                "world/endpoints/goal",
                goal_pose,
                (30, 220, 85, 120),
                "goal",
            )
        for index, waypoint_pose in enumerate(waypoint_poses, start=1):
            self._log_endpoint(
                f"world/endpoints/waypoint_{index}",
                waypoint_pose,
                (255, 105, 20, 130),
                f"waypoint {index}",
            )
        if obstacles:
            centers = np.asarray(
                [obstacle.center for obstacle in obstacles],
                dtype=np.float64,
            )
            half_sizes = np.asarray(
                [
                    (obstacle.radius,) * 3
                    for obstacle in obstacles
                ],
                dtype=np.float64,
            )
            self._recording.log(
                "world/obstacles",
                self._rr.Ellipsoids3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    colors=[obstacle.color for obstacle in obstacles],
                    labels=[obstacle.label for obstacle in obstacles],
                    line_radii=0.008,
                ),
                static=True,
            )
        if environment_boxes:
            self._recording.log(
                "world/environment/collision_boxes",
                self._rr.Boxes3D(
                    centers=[box.center for box in environment_boxes],
                    half_sizes=[
                        box.half_size for box in environment_boxes
                    ],
                    quaternions=[
                        self._rr.Quaternion(
                            xyzw=np.roll(box.quaternion_wxyz, -1)
                        )
                        for box in environment_boxes
                    ],
                    colors=[box.color for box in environment_boxes],
                    labels=[box.label for box in environment_boxes],
                    show_labels=False,
                    radii=0.008,
                    fill_mode="solid",
                ),
                static=True,
            )
        if metadata:
            import json

            self._recording.log(
                "metadata/run",
                self._rr.TextDocument(
                    json.dumps(
                        dict(metadata),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    media_type="application/json",
                ),
                static=True,
            )

    def log_frame(
        self,
        simulation_time_s: float,
        *,
        actual_pose: Pose3D,
        reference_pose: Pose3D | None = None,
        linear_velocity: ArrayLike | None = None,
        angular_velocity: ArrayLike | None = None,
        control: ArrayLike | None = None,
        nominal_positions: ArrayLike | None = None,
        sampled_positions: ArrayLike | None = None,
        joint_positions: Mapping[str, float] | None = None,
        scalar_channels: Mapping[str, float] | None = None,
    ) -> None:
        """Log one replayable simulation frame on ``sim_time``."""

        self._ensure_open()
        if not np.isfinite(simulation_time_s) or simulation_time_s < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        self._recording.set_time(
            "sim_time", duration=float(simulation_time_s)
        )
        self._recording.set_time(
            "control_step", sequence=self._frame_index
        )

        self._log_pose("world/uav/actual", actual_pose)
        self._log_robot_transforms(
            "world/uav/actual/model",
            joint_positions=joint_positions,
        )
        self._actual_trace.append(actual_pose.position.copy())
        self._recording.log(
            "world/uav/actual/position",
            self._rr.Points3D(
                [[0.0, 0.0, 0.0]],
                colors=[self._ACTUAL_COLOR],
                radii=[0.045],
            ),
        )
        if (
            self._frame_index % self.config.trace_update_stride == 0
            or self._frame_index == 0
        ):
            self._recording.log(
                "world/paths/actual_trace",
                self._rr.LineStrips3D(
                    [np.asarray(self._actual_trace)],
                    colors=[self._ACTUAL_COLOR],
                    radii=[0.014],
                    labels=["MuJoCo actual"],
                ),
            )

        if reference_pose is not None:
            self._log_pose("world/uav/reference", reference_pose)
            self._recording.log(
                "world/uav/reference/position",
                self._rr.Points3D(
                    [[0.0, 0.0, 0.0]],
                    colors=[self._REFERENCE_COLOR],
                    radii=[0.035],
                ),
            )
            self._log_vector_pair(
                "plots/position",
                actual_pose.position,
                reference_pose.position,
                ("x", "y", "z"),
            )
            self._log_vector_pair(
                "plots/attitude_deg",
                np.degrees(
                    _quaternion_to_euler(actual_pose.quaternion_wxyz)
                ),
                np.degrees(
                    _quaternion_to_euler(
                        reference_pose.quaternion_wxyz
                    )
                ),
                ("roll", "pitch", "yaw"),
            )

        if linear_velocity is not None:
            self._log_vector(
                "plots/velocity/linear",
                _vector(linear_velocity, 3, "linear_velocity"),
                ("x", "y", "z"),
            )
        if angular_velocity is not None:
            self._log_vector(
                "plots/velocity/angular",
                _vector(angular_velocity, 3, "angular_velocity"),
                ("x", "y", "z"),
            )
        if control is not None:
            control_array = np.asarray(control, dtype=np.float64)
            if (
                control_array.ndim != 1
                or not np.all(np.isfinite(control_array))
            ):
                raise ValueError("control must be a finite vector")
            names = (
                ("ax", "ay", "az", "alpha_x", "alpha_y", "alpha_z")
                if len(control_array) == 6
                else tuple(str(index) for index in range(len(control_array)))
            )
            self._log_vector("plots/control", control_array, names)

        if nominal_positions is not None:
            nominal = _positions(
                nominal_positions, "nominal_positions"
            )
            self._recording.log(
                "world/prediction/nominal",
                self._rr.LineStrips3D(
                    [nominal],
                    colors=[self._NOMINAL_COLOR],
                    radii=[0.012],
                    labels=["MPPI nominal"],
                ),
            )
        if sampled_positions is not None:
            samples = np.asarray(sampled_positions, dtype=np.float64)
            if (
                samples.ndim != 3
                or samples.shape[2] != 3
                or not np.all(np.isfinite(samples))
            ):
                raise ValueError(
                    "sampled_positions must have shape (N, T, 3)"
                )
            self._recording.log(
                "world/prediction/samples",
                self._rr.LineStrips3D(
                    list(samples),
                    colors=[self._SAMPLE_COLOR] * len(samples),
                    radii=[0.004] * len(samples),
                ),
            )
        if scalar_channels:
            for channel, value in scalar_channels.items():
                scalar = float(value)
                if not np.isfinite(scalar):
                    continue
                self._recording.log(
                    f"plots/{_entity_path(channel)}",
                    self._rr.Scalars([scalar]),
                )
        self._frame_index += 1

    def flush(self) -> None:
        self._ensure_open()
        self._recording.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._recording.flush()
        finally:
            self._exit_stack.close()
            self._closed = True

    def _configure_sinks(self) -> None:
        path = self.recording_path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        blueprint = _make_blueprint()
        if self.config.spawn_viewer:
            self._recording.spawn(
                port=self.config.viewer_port,
                hide_welcome_screen=True,
            )
            if path is not None:
                self._recording.set_sinks(
                    self._rr.GrpcSink(),
                    self._rr.FileSink(path),
                    default_blueprint=blueprint,
                )
        elif path is not None:
            self._recording.save(path, default_blueprint=blueprint)

    def _log_coordinate_system(self) -> None:
        self._recording.log(
            "world",
            self._rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )
        self._log_robot_geometry(
            "world/uav/actual/model",
            static_transforms=False,
        )
        self._log_robot_geometry(
            "world/uav/reference/model",
            tint=(40, 185, 70, 80),
            static_transforms=True,
        )

    def _log_endpoint(
        self, path: str, pose: Pose3D, color: Color, label: str
    ) -> None:
        self._log_pose(path, pose, static=True)
        self._log_robot_geometry(
            f"{path}/model",
            tint=color,
            static_transforms=True,
        )
        self._recording.log(
            f"{path}/label",
            self._rr.Points3D(
                [[0.0, 0.0, 0.0]],
                colors=[color],
                labels=[label],
                radii=[0.035],
            ),
            static=True,
        )

    def _log_robot_geometry(
        self,
        path: str,
        *,
        tint: Color | None = None,
        static_transforms: bool,
    ) -> None:
        """Log reusable STL assets and optionally their zero-joint transforms."""

        link_paths = self._robot_link_paths(path)
        for link in self._robot_model.links:
            link_path = link_paths[link.name]
            visual_path = f"{link_path}/visual"
            self._recording.log(
                visual_path,
                self._rr.Transform3D(
                    translation=link.visual_translation,
                    rotation=_rerun_quaternion(
                        self._rr, link.visual_quaternion_wxyz
                    ),
                    scale=link.visual_scale,
                ),
                static=True,
            )
            self._recording.log(
                visual_path,
                self._rr.Asset3D(
                    path=link.mesh_path,
                    albedo_factor=tint or link.color,
                ),
                static=True,
            )
        if static_transforms:
            self._log_robot_transforms(path, static=True)

    def _log_robot_transforms(
        self,
        path: str,
        *,
        joint_positions: Mapping[str, float] | None = None,
        static: bool = False,
    ) -> None:
        """Log the MJCF joint tree under a robot root entity."""

        positions = _validated_joint_positions(
            joint_positions, self._robot_model.joint_names
        )
        link_paths = self._robot_link_paths(path)
        for link in self._robot_model.links:
            quaternion = link.quaternion_wxyz
            if link.joint_name is not None:
                joint_quaternion = _axis_angle_quaternion(
                    link.joint_axis,
                    positions.get(link.joint_name, 0.0),
                )
                quaternion = _quaternion_multiply(
                    quaternion, joint_quaternion
                )
            self._recording.log(
                link_paths[link.name],
                self._rr.Transform3D(
                    translation=link.translation,
                    rotation=_rerun_quaternion(self._rr, quaternion),
                ),
                static=static,
            )

    def _robot_link_paths(self, root_path: str) -> dict[str, str]:
        paths: dict[str, str] = {}
        for link in self._robot_model.links:
            if link.parent is None:
                paths[link.name] = f"{root_path}/{link.name}"
            else:
                paths[link.name] = f"{paths[link.parent]}/{link.name}"
        return paths

    def _log_pose(
        self, path: str, pose: Pose3D, static: bool = False
    ) -> None:
        self._recording.log(
            path,
            self._rr.Transform3D(
                translation=pose.position,
                rotation=_rerun_quaternion(
                    self._rr, pose.quaternion_wxyz
                ),
            ),
            static=static,
        )

    def _log_vector(
        self, path: str, values: FloatArray, names: Sequence[str]
    ) -> None:
        if len(values) != len(names):
            raise ValueError("values and names must have equal length")
        for name, value in zip(names, values):
            self._recording.log(
                f"{path}/{_entity_path(name)}",
                self._rr.Scalars([float(value)]),
            )

    def _log_vector_pair(
        self,
        path: str,
        actual: FloatArray,
        reference: FloatArray,
        names: Sequence[str],
    ) -> None:
        for name, actual_value, reference_value in zip(
            names, actual, reference
        ):
            safe_name = _entity_path(name)
            self._recording.log(
                f"{path}/{safe_name}/actual",
                self._rr.Scalars([float(actual_value)]),
            )
            self._recording.log(
                f"{path}/{safe_name}/reference",
                self._rr.Scalars([float(reference_value)]),
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Rerun recorder is already closed")


def _load_robot_model(urdf_path: Path, mjcf_path: Path) -> _RobotModel:
    """Combine URDF visual meshes with the matching MJCF body transforms."""

    if not urdf_path.is_file():
        raise FileNotFoundError(f"robot URDF does not exist: {urdf_path}")
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"robot MJCF does not exist: {mjcf_path}")

    urdf_root = ElementTree.parse(urdf_path).getroot()
    urdf_visuals = {
        link_name: _parse_urdf_visual(link, urdf_path)
        for link in urdf_root.findall("link")
        if (link_name := link.get("name"))
    }
    if not urdf_visuals:
        raise ValueError(f"URDF contains no visual links: {urdf_path}")

    urdf_joints: dict[str, tuple[str, str]] = {}
    child_links: set[str] = set()
    for joint in urdf_root.findall("joint"):
        joint_name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            not joint_name
            or parent is None
            or child is None
            or not parent.get("link")
            or not child.get("link")
        ):
            raise ValueError("every URDF joint must name its parent and child")
        child_name = child.get("link")
        assert child_name is not None
        urdf_joints[child_name] = (joint_name, parent.get("link", ""))
        child_links.add(child_name)

    root_links = set(urdf_visuals) - child_links
    if len(root_links) != 1:
        raise ValueError(
            "URDF must contain exactly one visual root link; found "
            f"{sorted(root_links)}"
        )
    root_link = next(iter(root_links))

    mjcf_root = ElementTree.parse(mjcf_path).getroot()
    materials = _parse_mjcf_materials(mjcf_root)
    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF contains no worldbody: {mjcf_path}")
    drone_body = next(
        (
            body
            for body in worldbody.findall("body")
            if body.get("name") == "drone"
        ),
        None,
    )
    if drone_body is None:
        raise ValueError("MJCF worldbody must contain a body named 'drone'")

    links: list[_RobotLink] = []

    def add_link(
        body: ElementTree.Element,
        link_name: str,
        parent_link: str | None,
    ) -> None:
        visual = urdf_visuals.get(link_name)
        if visual is None:
            raise ValueError(
                f"MJCF body {link_name!r} has no matching URDF visual link"
            )

        joint_name: str | None = None
        joint_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if parent_link is not None:
            joints = body.findall("joint")
            if len(joints) != 1 or not joints[0].get("name"):
                raise ValueError(
                    f"MJCF body {link_name!r} must contain one named joint"
                )
            joint_name = joints[0].get("name")
            assert joint_name is not None
            joint_axis = _xml_vector(
                joints[0].get("axis"), 3, default=(0.0, 0.0, 1.0)
            )
            expected_joint = urdf_joints.get(link_name)
            if expected_joint != (joint_name, parent_link):
                raise ValueError(
                    f"URDF/MJCF topology mismatch at link {link_name!r}: "
                    f"URDF={expected_joint}, "
                    f"MJCF={(joint_name, parent_link)}"
                )

        geom = next(
            (
                candidate
                for candidate in body.findall("geom")
                if candidate.get("type") == "mesh"
                or candidate.get("mesh") is not None
            ),
            None,
        )
        color = _mjcf_geom_color(geom, materials, visual.color)
        translation = (
            np.zeros(3, dtype=np.float64)
            if parent_link is None
            else _xml_vector(
                body.get("pos"), 3, default=(0.0, 0.0, 0.0)
            )
        )
        orientation = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if parent_link is None
            else _mjcf_orientation(body)
        )
        links.append(
            _RobotLink(
                name=link_name,
                parent=parent_link,
                joint_name=joint_name,
                joint_axis=joint_axis,
                translation=translation,
                quaternion_wxyz=orientation,
                mesh_path=visual.mesh_path,
                visual_translation=visual.translation,
                visual_quaternion_wxyz=visual.quaternion_wxyz,
                visual_scale=visual.scale,
                color=color,
            )
        )
        for child_body in body.findall("body"):
            child_name = child_body.get("name")
            if not child_name:
                raise ValueError("every MJCF robot body must have a name")
            add_link(child_body, child_name, link_name)

    add_link(drone_body, root_link, None)
    mjcf_link_names = {link.name for link in links}
    if mjcf_link_names != set(urdf_visuals):
        raise ValueError(
            "URDF/MJCF link sets differ: "
            f"only_urdf={sorted(set(urdf_visuals) - mjcf_link_names)}, "
            f"only_mjcf={sorted(mjcf_link_names - set(urdf_visuals))}"
        )
    return _RobotModel(root_link=root_link, links=tuple(links))


def _parse_urdf_visual(
    link: ElementTree.Element, urdf_path: Path
) -> _UrdfVisual:
    link_name = link.get("name", "<unnamed>")
    visual = link.find("visual")
    if visual is None:
        raise ValueError(f"URDF link {link_name!r} has no visual")
    mesh = visual.find("geometry/mesh")
    if mesh is None or not mesh.get("filename"):
        raise ValueError(f"URDF link {link_name!r} has no visual mesh")
    mesh_path = _resolve_urdf_mesh(
        urdf_path, mesh.get("filename", "")
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"mesh for URDF link {link_name!r} does not exist: {mesh_path}"
        )

    origin = visual.find("origin")
    translation = _xml_vector(
        origin.get("xyz") if origin is not None else None,
        3,
        default=(0.0, 0.0, 0.0),
    )
    rpy = _xml_vector(
        origin.get("rpy") if origin is not None else None,
        3,
        default=(0.0, 0.0, 0.0),
    )
    color_element = visual.find("material/color")
    rgba = _xml_vector(
        color_element.get("rgba") if color_element is not None else None,
        4,
        default=(0.75, 0.75, 0.78, 1.0),
    )
    return _UrdfVisual(
        mesh_path=mesh_path,
        translation=translation,
        quaternion_wxyz=_rpy_to_quaternion(rpy),
        scale=_xml_vector(
            mesh.get("scale"), 3, default=(1.0, 1.0, 1.0)
        ),
        color=_rgba_float_to_color(rgba),
    )


def _resolve_urdf_mesh(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        package_relative = filename[len("package://") :]
        parts = Path(package_relative).parts
        if len(parts) < 2:
            raise ValueError(f"invalid URDF package mesh URI: {filename}")
        return (urdf_path.parent.parent / Path(*parts[1:])).resolve()
    return (urdf_path.parent / filename).resolve()


def _parse_mjcf_materials(
    root: ElementTree.Element,
) -> dict[str, Color]:
    materials: dict[str, Color] = {}
    asset = root.find("asset")
    if asset is None:
        return materials
    for material in asset.findall("material"):
        name = material.get("name")
        if name and material.get("rgba"):
            materials[name] = _rgba_float_to_color(
                _xml_vector(material.get("rgba"), 4)
            )
    return materials


def _mjcf_geom_color(
    geom: ElementTree.Element | None,
    materials: Mapping[str, Color],
    fallback: Color,
) -> Color:
    if geom is None:
        return fallback
    if geom.get("rgba"):
        return _rgba_float_to_color(_xml_vector(geom.get("rgba"), 4))
    material = geom.get("material")
    return materials.get(material, fallback)


def _mjcf_orientation(element: ElementTree.Element) -> FloatArray:
    if element.get("quat"):
        quaternion = _xml_vector(element.get("quat"), 4)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-12:
            raise ValueError("MJCF quaternion norm must be nonzero")
        return quaternion / norm
    if element.get("euler"):
        return _rpy_to_quaternion(_xml_vector(element.get("euler"), 3))
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _xml_vector(
    value: str | None,
    size: int,
    *,
    default: Sequence[float] | None = None,
) -> FloatArray:
    if value is None:
        if default is None:
            raise ValueError(f"missing XML {size}-vector")
        array = np.asarray(default, dtype=np.float64)
    else:
        array = np.fromstring(value, sep=" ", dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid XML {size}-vector: {value!r}")
    return array.copy()


def _rgba_float_to_color(rgba: FloatArray) -> Color:
    if np.any(rgba < 0.0) or np.any(rgba > 1.0):
        raise ValueError(f"RGBA channels must lie in [0, 1]: {rgba}")
    values = np.rint(rgba * 255.0).astype(np.uint8)
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _rpy_to_quaternion(rpy: FloatArray) -> FloatArray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _axis_angle_quaternion(axis: FloatArray, angle: float) -> FloatArray:
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        raise ValueError("joint axis norm must be nonzero")
    half_angle = angle * 0.5
    return np.concatenate(
        (
            np.array([np.cos(half_angle)], dtype=np.float64),
            axis / norm * np.sin(half_angle),
        )
    )


def _quaternion_multiply(
    left: FloatArray, right: FloatArray
) -> FloatArray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    quaternion = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return quaternion / np.linalg.norm(quaternion)


def _validated_joint_positions(
    joint_positions: Mapping[str, float] | None,
    valid_joint_names: frozenset[str],
) -> dict[str, float]:
    if joint_positions is None:
        return {}
    unknown = set(joint_positions) - valid_joint_names
    if unknown:
        raise ValueError(f"unknown robot joints: {sorted(unknown)}")
    positions: dict[str, float] = {}
    for name, value in joint_positions.items():
        position = float(value)
        if not np.isfinite(position):
            raise ValueError(f"joint position for {name!r} must be finite")
        positions[name] = position
    return positions


def _rerun_quaternion(rr: ModuleType, quaternion_wxyz: FloatArray) -> object:
    qw, qx, qy, qz = quaternion_wxyz
    return rr.Quaternion(xyzw=[qx, qy, qz, qw])


def _import_rerun() -> ModuleType:
    try:
        import rerun as rr
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Rerun recording was requested but rerun-sdk is not installed. "
            "Install it with: pip install rerun-sdk"
        ) from error
    return rr


def _make_blueprint() -> object:
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="world",
                name="MuJoCo / OMPL / MPPI 3D",
                background=(22, 28, 38),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="plots/position",
                    name="Position tracking",
                ),
                rrb.TimeSeriesView(
                    origin="plots/tracking",
                    name="Tracking errors",
                ),
                rrb.TimeSeriesView(
                    origin="plots/attitude_deg",
                    name="Attitude tracking [deg]",
                ),
                rrb.TimeSeriesView(
                    origin="plots/control",
                    name="MPPI controls",
                ),
                row_shares=[2.0, 1.0, 1.0, 1.0],
            ),
            column_shares=[2.0, 1.0],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True,
    )


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array.copy()


def _positions(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must have shape (N, 3), N >= 1")
    return array.copy()


def _quaternion_to_euler(quaternion_wxyz: FloatArray) -> FloatArray:
    w, x, y, z = quaternion_wxyz
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(
        np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    )
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _entity_path(value: object) -> str:
    path = str(value).strip().replace("\\", "/")
    components = [
        re.sub(r"[^a-zA-Z0-9_.-]+", "_", component).strip("_")
        for component in path.split("/")
    ]
    return "/".join(component or "unnamed" for component in components)


def _validate_color(color: Color) -> None:
    if len(color) != 4 or any(
        not isinstance(channel, int) or not 0 <= channel <= 255
        for channel in color
    ):
        raise ValueError("color must contain four uint8-compatible integers")
