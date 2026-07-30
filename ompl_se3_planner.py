"""OMPL SE(3) Bi-RRT planning and MPPI reference generation.

The planner deliberately has no MuJoCo dependency.  It plans a collision-free
geometric path in ``SE(3)`` with OMPL's bidirectional ``RRTConnect`` planner,
then :class:`SE3PathReference` time-parameterizes that path into the 13-state
layout consumed by the pose MPPI controller:

``[position_world, velocity_world, quaternion_wxyz, omega_body]``.
"""

from __future__ import annotations

import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mppi.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_from_rotation_vector,
    quaternion_multiply,
)


FloatArray = NDArray[np.float64]
_OMPL_SEEDED = False


def _load_ompl() -> tuple[Any, Any, Any]:
    """Import OMPL, also supporting a system user-site from inside a venv."""

    try:
        from ompl import base as ob
        from ompl import geometric as og
        from ompl import util as ou

        return ob, og, ou
    except ModuleNotFoundError as original_error:
        # The project venv may intentionally exclude system/user packages while
        # OMPL is installed system-wide.  Add only the standard user-site path;
        # a custom installation can still be supplied through PYTHONPATH.
        user_site = Path(site.getusersitepackages())
        if user_site.is_dir() and str(user_site) not in sys.path:
            sys.path.append(str(user_site))
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            return ob, og, ou
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "OMPL Python bindings were not found. Install OMPL for this "
                "Python version or add its site-packages directory to "
                "PYTHONPATH."
            ) from original_error


@dataclass(frozen=True)
class SE3Pose:
    """A world-frame position and a MuJoCo/OMPL ``wxyz`` quaternion."""

    position: FloatArray
    quaternion: FloatArray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        quaternion = np.asarray(self.quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite 3-vector")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion must be a finite 4-vector")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(
            self, "quaternion", normalize_quaternion(quaternion)
        )


@dataclass(frozen=True)
class SphereObstacle:
    """Spherical planning obstacle before vehicle-radius inflation."""

    center: FloatArray
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("obstacle center must be a finite 3-vector")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive")
        object.__setattr__(self, "center", center.copy())


@dataclass(frozen=True)
class PlannedSE3Path:
    """A detached, dense OMPL solution and planning diagnostics."""

    states: FloatArray
    planning_time_s: float
    raw_state_count: int
    path_length_m: float
    rotation_length_rad: float
    planner_name: str = "OMPL RRTConnect (bidirectional RRT)"

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 7 or len(states) < 2:
            raise ValueError("states must have shape (N, 7), N >= 2")
        if not np.all(np.isfinite(states)):
            raise ValueError("path states must be finite")
        states = states.copy()
        states[:, 3:7] = normalize_quaternion(states[:, 3:7])
        object.__setattr__(self, "states", states)


class OMPLSE3Planner:
    """Collision-aware SE(3) planner backed by OMPL RRTConnect.

    The HNUTER is conservatively represented by a bounding sphere.  Spherical
    obstacles are inflated by ``vehicle_radius + safety_margin`` so OMPL's
    point-position validity check accounts for vehicle extent.
    """

    def __init__(
        self,
        bounds_min: ArrayLike,
        bounds_max: ArrayLike,
        obstacles: Sequence[SphereObstacle] = (),
        vehicle_radius: float = 0.25,
        safety_margin: float = 0.08,
        validity_resolution: float = 0.01,
        planner_range: float = 0.45,
        seed: int = 7,
    ) -> None:
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.obstacles = tuple(obstacles)
        self.vehicle_radius = float(vehicle_radius)
        self.safety_margin = float(safety_margin)
        self.validity_resolution = float(validity_resolution)
        self.planner_range = float(planner_range)
        self.seed = int(seed)
        if (
            self.bounds_min.shape != (3,)
            or self.bounds_max.shape != (3,)
            or not np.all(np.isfinite(self.bounds_min))
            or not np.all(np.isfinite(self.bounds_max))
            or np.any(self.bounds_min >= self.bounds_max)
        ):
            raise ValueError(
                "bounds_min/bounds_max must be finite ordered 3-vectors"
            )
        if self.vehicle_radius < 0.0 or self.safety_margin < 0.0:
            raise ValueError(
                "vehicle_radius and safety_margin must be non-negative"
            )
        if not 0.0 < self.validity_resolution <= 1.0:
            raise ValueError("validity_resolution must lie in (0, 1]")
        if self.planner_range <= 0.0:
            raise ValueError("planner_range must be positive")

    @property
    def collision_padding(self) -> float:
        return self.vehicle_radius + self.safety_margin

    def clearance(self, positions: ArrayLike) -> FloatArray:
        """Signed clearance to the closest inflated obstacle."""

        position_array = np.asarray(positions, dtype=np.float64)
        if position_array.shape[-1] != 3:
            raise ValueError("positions must have trailing dimension 3")
        if not self.obstacles:
            return np.full(position_array.shape[:-1], np.inf)
        clearances = [
            np.linalg.norm(position_array - obstacle.center, axis=-1)
            - obstacle.radius
            - self.collision_padding
            for obstacle in self.obstacles
        ]
        return np.min(np.stack(clearances, axis=-1), axis=-1)

    def is_position_valid(self, position: ArrayLike) -> bool:
        position_array = np.asarray(position, dtype=np.float64)
        if position_array.shape != (3,) or not np.all(
            np.isfinite(position_array)
        ):
            return False
        if np.any(position_array < self.bounds_min) or np.any(
            position_array > self.bounds_max
        ):
            return False
        return bool(self.clearance(position_array) > 0.0)

    def plan(
        self,
        start: SE3Pose,
        goal: SE3Pose,
        solve_time: float = 2.0,
        interpolation_resolution: float = 0.08,
        minimum_waypoints: int = 80,
        simplify: bool = True,
    ) -> PlannedSE3Path:
        """Plan and return a dense path detached from OMPL-owned state memory."""

        if solve_time <= 0.0:
            raise ValueError("solve_time must be positive")
        if interpolation_resolution <= 0.0 or minimum_waypoints < 2:
            raise ValueError(
                "interpolation_resolution must be positive and "
                "minimum_waypoints must be at least two"
            )
        for name, pose in (("start", start), ("goal", goal)):
            if not self.is_position_valid(pose.position):
                raise ValueError(
                    f"{name} position {pose.position.tolist()} is outside "
                    "the workspace or collides with an inflated obstacle"
                )

        ob, og, ou = _load_ompl()
        global _OMPL_SEEDED
        if not _OMPL_SEEDED:
            try:
                ou.RNG.setSeed(self.seed)
            except RuntimeError:
                # Some OMPL builds disallow resetting the global seed after a
                # sampler was created. Planning remains valid, only
                # nondeterministic.
                pass
            _OMPL_SEEDED = True
        if hasattr(ou, "setLogLevel") and hasattr(ou, "LOG_WARN"):
            ou.setLogLevel(ou.LOG_WARN)

        space = ob.SE3StateSpace()
        bounds = ob.RealVectorBounds(3)
        for axis in range(3):
            bounds.setLow(axis, float(self.bounds_min[axis]))
            bounds.setHigh(axis, float(self.bounds_max[axis]))
        space.setBounds(bounds)
        setup = og.SimpleSetup(space)
        setup.setStateValidityChecker(
            lambda state: self.is_position_valid(
                (state.getX(), state.getY(), state.getZ())
            )
        )
        setup.getSpaceInformation().setStateValidityCheckingResolution(
            self.validity_resolution
        )

        start_state = space.allocState()
        goal_state = space.allocState()
        self._write_state(start_state, start)
        self._write_state(goal_state, goal)
        setup.setStartAndGoalStates(start_state, goal_state, 1.0e-4)

        planner = og.RRTConnect(setup.getSpaceInformation())
        planner.setRange(self.planner_range)
        setup.setPlanner(planner)
        planning_start = time.perf_counter()
        status = setup.solve(float(solve_time))
        planning_time = time.perf_counter() - planning_start
        if not bool(status) or not setup.haveExactSolutionPath():
            status_name = (
                status.asString() if hasattr(status, "asString") else str(status)
            )
            raise RuntimeError(
                f"OMPL RRTConnect did not find an exact path in "
                f"{solve_time:.2f}s (status: {status_name})"
            )

        path = setup.getSolutionPath()
        raw_state_count = int(path.getStateCount())
        if simplify:
            setup.simplifySolution()
            path = setup.getSolutionPath()
            # A few validity-preserving B-spline passes reduce sharp velocity
            # changes at RRT vertices while keeping collision checks in OMPL.
            setup.getPathSimplifier().smoothBSpline(path, 3, 0.01)

        state_count = max(
            minimum_waypoints,
            int(np.ceil(float(path.length()) / interpolation_resolution)) + 1,
        )
        path.interpolate(state_count)
        states = np.empty((path.getStateCount(), 7), dtype=np.float64)
        for index in range(path.getStateCount()):
            state = path.getState(index)
            rotation = state.rotation()
            states[index] = (
                state.getX(),
                state.getY(),
                state.getZ(),
                rotation.w,
                rotation.x,
                rotation.y,
                rotation.z,
            )

        states[:, 3:7] = normalize_quaternion(states[:, 3:7])
        self._make_quaternion_sequence_continuous(states[:, 3:7])
        if np.any(self.clearance(states[:, :3]) <= -1.0e-9):
            raise RuntimeError(
                "OMPL returned a path that failed the application's "
                "post-planning collision check"
            )
        translation_delta = np.diff(states[:, :3], axis=0)
        rotation_delta = _relative_rotation_vectors(
            states[:-1, 3:7], states[1:, 3:7]
        )
        return PlannedSE3Path(
            states=states,
            planning_time_s=planning_time,
            raw_state_count=raw_state_count,
            path_length_m=float(
                np.sum(np.linalg.norm(translation_delta, axis=1))
            ),
            rotation_length_rad=float(
                np.sum(np.linalg.norm(rotation_delta, axis=1))
            ),
        )

    @staticmethod
    def _write_state(state: Any, pose: SE3Pose) -> None:
        state.setXYZ(*(float(value) for value in pose.position))
        rotation = state.rotation()
        rotation.w, rotation.x, rotation.y, rotation.z = (
            float(value) for value in pose.quaternion
        )

    @staticmethod
    def _make_quaternion_sequence_continuous(quaternions: FloatArray) -> None:
        for index in range(1, len(quaternions)):
            if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
                quaternions[index] *= -1.0


class SE3PathReference:
    """Minimum-jerk time parameterization of an OMPL SE(3) path."""

    def __init__(
        self,
        path: PlannedSE3Path | ArrayLike,
        max_linear_speed: float = 0.65,
        max_angular_speed: float = 0.8,
        start_delay: float = 0.5,
        duration_scale: float = 1.0,
    ) -> None:
        states = path.states if isinstance(path, PlannedSE3Path) else path
        self.states = np.asarray(states, dtype=np.float64).copy()
        if (
            self.states.ndim != 2
            or self.states.shape[1] != 7
            or len(self.states) < 2
            or not np.all(np.isfinite(self.states))
        ):
            raise ValueError("path must have shape (N, 7), N >= 2")
        if (
            max_linear_speed <= 0.0
            or max_angular_speed <= 0.0
            or start_delay < 0.0
            or duration_scale <= 0.0
        ):
            raise ValueError(
                "speed limits/duration_scale must be positive and "
                "start_delay non-negative"
            )
        self.states[:, 3:7] = normalize_quaternion(self.states[:, 3:7])
        OMPLSE3Planner._make_quaternion_sequence_continuous(
            self.states[:, 3:7]
        )
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.start_delay = float(start_delay)
        self.duration_scale = float(duration_scale)

        self._position_delta = np.diff(self.states[:, :3], axis=0)
        self._rotation_vector = _relative_rotation_vectors(
            self.states[:-1, 3:7], self.states[1:, 3:7]
        )
        translation_time = (
            np.linalg.norm(self._position_delta, axis=1)
            / self.max_linear_speed
        )
        rotation_time = (
            np.linalg.norm(self._rotation_vector, axis=1)
            / self.max_angular_speed
        )
        segment_time = np.maximum(translation_time, rotation_time)
        segment_time = np.maximum(segment_time, 1.0e-9)
        self._raw_cumulative_time = np.concatenate(
            ([0.0], np.cumsum(segment_time))
        )
        self._raw_duration = float(self._raw_cumulative_time[-1])
        # The derivative of 10u^3-15u^4+6u^5 peaks at 1.875. This factor
        # makes the user-provided limits true peak limits after time warping.
        self.duration = 1.875 * self._raw_duration * self.duration_scale
        self.finish_time = self.start_delay + self.duration

    def sample(self, times: ArrayLike) -> FloatArray:
        """Sample position, velocity, quaternion and body angular velocity."""

        time_array = np.asarray(times, dtype=np.float64)
        if time_array.ndim != 1 or not np.all(np.isfinite(time_array)):
            raise ValueError("times must be a finite one-dimensional array")
        reference = np.zeros((len(time_array), 13), dtype=np.float64)

        phase = np.clip(
            (time_array - self.start_delay) / self.duration, 0.0, 1.0
        )
        progress = (
            10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        )
        progress_rate = (
            30.0 * phase**2
            - 60.0 * phase**3
            + 30.0 * phase**4
        ) / self.duration
        inactive = (time_array <= self.start_delay) | (
            time_array >= self.finish_time
        )
        progress_rate[inactive] = 0.0

        raw_path_time = progress * self._raw_duration
        indices = np.searchsorted(
            self._raw_cumulative_time, raw_path_time, side="right"
        ) - 1
        indices = np.clip(indices, 0, len(self.states) - 2)
        segment_start = self._raw_cumulative_time[indices]
        segment_duration = (
            self._raw_cumulative_time[indices + 1] - segment_start
        )
        local_progress = np.clip(
            (raw_path_time - segment_start) / segment_duration, 0.0, 1.0
        )
        local_rate = (
            self._raw_duration * progress_rate / segment_duration
        )

        reference[:, :3] = (
            self.states[indices, :3]
            + local_progress[:, None] * self._position_delta[indices]
        )
        reference[:, 3:6] = (
            local_rate[:, None] * self._position_delta[indices]
        )
        delta_quaternion = quaternion_from_rotation_vector(
            local_progress[:, None] * self._rotation_vector[indices]
        )
        reference[:, 6:10] = normalize_quaternion(
            quaternion_multiply(
                self.states[indices, 3:7], delta_quaternion
            )
        )
        reference[:, 10:13] = (
            local_rate[:, None] * self._rotation_vector[indices]
        )
        return reference


def _relative_rotation_vectors(
    start_quaternion: ArrayLike, end_quaternion: ArrayLike
) -> FloatArray:
    """Shortest body-frame rotation vectors taking ``start`` to ``end``."""

    relative = normalize_quaternion(
        quaternion_multiply(
            quaternion_conjugate(start_quaternion), end_quaternion
        )
    )
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(
        vector_norm, np.clip(relative[..., :1], 0.0, 1.0)
    )
    scale = np.full_like(vector_norm, 2.0)
    nonzero = vector_norm > 1.0e-9
    scale[nonzero] = angle[nonzero] / vector_norm[nonzero]
    return vector * scale
