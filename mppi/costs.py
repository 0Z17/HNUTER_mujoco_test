"""Trajectory costs for the MPPI controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quaternion import quaternion_error_vector


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuadraticTrackingCost:
    """Quadratic path-tracking cost with flight-envelope soft constraints."""

    position_weight: tuple[float, float, float] = (8.0, 8.0, 14.0)
    velocity_weight: tuple[float, float, float] = (1.2, 1.2, 2.0)
    terminal_multiplier: float = 8.0
    control_weight: tuple[float, float, float] = (0.025, 0.025, 0.035)
    control_rate_weight: float = 0.30
    minimum_altitude: float = 0.12
    altitude_penalty: float = 2.0e4
    workspace_radius: float | None = 4.5
    workspace_penalty: float = 2.0e3

    def __post_init__(self) -> None:
        for name in (
            "position_weight",
            "velocity_weight",
            "control_weight",
        ):
            values = getattr(self, name)
            if len(values) != 3 or any(value < 0.0 for value in values):
                raise ValueError(f"{name} must contain three non-negative values")
        if self.terminal_multiplier < 0.0:
            raise ValueError("terminal_multiplier must be non-negative")
        if self.control_rate_weight < 0.0:
            raise ValueError("control_rate_weight must be non-negative")

    def trajectory_cost(
        self,
        states: FloatArray,
        controls: FloatArray,
        reference: FloatArray,
    ) -> FloatArray:
        """Evaluate all samples without Python loops over trajectories."""

        if states.ndim != 3 or states.shape[2] != 6:
            raise ValueError("states must have shape (batch, horizon + 1, 6)")
        if controls.ndim != 3 or controls.shape[2] != 3:
            raise ValueError("controls must have shape (batch, horizon, 3)")
        if reference.shape != states.shape[1:]:
            raise ValueError(
                f"reference must have shape {states.shape[1:]}, "
                f"got {reference.shape}"
            )
        if controls.shape[:2] != (
            states.shape[0],
            states.shape[1] - 1,
        ):
            raise ValueError("states and controls have inconsistent horizons")

        state_error = states[:, 1:, :] - reference[None, 1:, :]
        position_weight = np.asarray(self.position_weight)
        velocity_weight = np.asarray(self.velocity_weight)
        control_weight = np.asarray(self.control_weight)

        stage_cost = np.sum(
            np.square(state_error[:, :, :3])
            * position_weight[None, None, :],
            axis=(1, 2),
        )
        stage_cost += np.sum(
            np.square(state_error[:, :, 3:])
            * velocity_weight[None, None, :],
            axis=(1, 2),
        )
        stage_cost += np.sum(
            np.square(controls) * control_weight[None, None, :],
            axis=(1, 2),
        )

        if controls.shape[1] > 1 and self.control_rate_weight > 0.0:
            control_delta = np.diff(controls, axis=1)
            stage_cost += self.control_rate_weight * np.sum(
                np.square(control_delta), axis=(1, 2)
            )

        terminal_error = state_error[:, -1, :]
        terminal_cost = self.terminal_multiplier * (
            np.sum(
                np.square(terminal_error[:, :3])
                * position_weight[None, :],
                axis=1,
            )
            + np.sum(
                np.square(terminal_error[:, 3:])
                * velocity_weight[None, :],
                axis=1,
            )
        )

        altitude_violation = np.maximum(
            self.minimum_altitude - states[:, 1:, 2], 0.0
        )
        envelope_cost = self.altitude_penalty * np.sum(
            np.square(altitude_violation), axis=1
        )

        if self.workspace_radius is not None:
            horizontal_radius = np.linalg.norm(states[:, 1:, :2], axis=2)
            workspace_violation = np.maximum(
                horizontal_radius - self.workspace_radius, 0.0
            )
            envelope_cost += self.workspace_penalty * np.sum(
                np.square(workspace_violation), axis=1
            )

        return stage_cost + terminal_cost + envelope_cost


@dataclass(frozen=True)
class PoseTrackingCost:
    """6-DoF position/quaternion tracking cost for a fully actuated UAV."""

    position_weight: tuple[float, float, float] = (8.0, 8.0, 14.0)
    velocity_weight: tuple[float, float, float] = (1.2, 1.2, 2.0)
    attitude_weight: tuple[float, float, float] = (14.0, 14.0, 9.0)
    angular_velocity_weight: tuple[float, float, float] = (1.5, 1.5, 1.0)
    terminal_multiplier: float = 2.0
    control_weight: tuple[float, float, float, float, float, float] = (
        0.025,
        0.025,
        0.035,
        0.020,
        0.020,
        0.015,
    )
    control_rate_weight: tuple[float, float, float, float, float, float] = (
        0.30,
        0.30,
        0.30,
        0.08,
        0.08,
        0.06,
    )
    minimum_altitude: float = 0.12
    altitude_penalty: float = 2.0e4
    workspace_radius: float | None = 4.5
    workspace_penalty: float = 2.0e3
    spherical_obstacles: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    collision_radius: float = 0.0
    obstacle_penalty: float = 2.0e4

    def __post_init__(self) -> None:
        for name, expected_length in (
            ("position_weight", 3),
            ("velocity_weight", 3),
            ("attitude_weight", 3),
            ("angular_velocity_weight", 3),
            ("control_weight", 6),
            ("control_rate_weight", 6),
        ):
            values = getattr(self, name)
            if len(values) != expected_length or any(
                value < 0.0 for value in values
            ):
                raise ValueError(
                    f"{name} must contain {expected_length} "
                    "non-negative values"
                )
        if self.terminal_multiplier < 0.0:
            raise ValueError("terminal_multiplier must be non-negative")
        if self.collision_radius < 0.0 or self.obstacle_penalty < 0.0:
            raise ValueError(
                "collision_radius and obstacle_penalty must be non-negative"
            )
        for obstacle in self.spherical_obstacles:
            if (
                len(obstacle) != 4
                or not np.all(np.isfinite(obstacle))
                or obstacle[3] <= 0.0
            ):
                raise ValueError(
                    "each spherical obstacle must be a finite "
                    "(x, y, z, positive_radius) tuple"
                )

    def trajectory_cost(
        self,
        states: FloatArray,
        controls: FloatArray,
        reference: FloatArray,
    ) -> FloatArray:
        if states.ndim != 3 or states.shape[2] != 13:
            raise ValueError(
                "states must have shape (batch, horizon + 1, 13)"
            )
        if controls.ndim != 3 or controls.shape[2] != 6:
            raise ValueError(
                "controls must have shape (batch, horizon, 6)"
            )
        if reference.shape != states.shape[1:]:
            raise ValueError(
                f"reference must have shape {states.shape[1:]}, "
                f"got {reference.shape}"
            )
        if controls.shape[:2] != (
            states.shape[0],
            states.shape[1] - 1,
        ):
            raise ValueError("states and controls have inconsistent horizons")

        predicted = states[:, 1:, :]
        desired = reference[None, 1:, :]
        position_error = predicted[:, :, :3] - desired[:, :, :3]
        velocity_error = predicted[:, :, 3:6] - desired[:, :, 3:6]
        attitude_error = quaternion_error_vector(
            predicted[:, :, 6:10],
            desired[:, :, 6:10],
        )
        angular_velocity_error = (
            predicted[:, :, 10:13] - desired[:, :, 10:13]
        )

        stage_cost = self._state_cost(
            position_error,
            velocity_error,
            attitude_error,
            angular_velocity_error,
        )
        stage_cost += np.sum(
            np.square(controls)
            * np.asarray(self.control_weight)[None, None, :],
            axis=(1, 2),
        )
        if controls.shape[1] > 1:
            stage_cost += np.sum(
                np.square(np.diff(controls, axis=1))
                * np.asarray(self.control_rate_weight)[None, None, :],
                axis=(1, 2),
            )

        terminal_cost = self.terminal_multiplier * self._state_cost(
            position_error[:, -1:, :],
            velocity_error[:, -1:, :],
            attitude_error[:, -1:, :],
            angular_velocity_error[:, -1:, :],
        )

        altitude_violation = np.maximum(
            self.minimum_altitude - predicted[:, :, 2], 0.0
        )
        envelope_cost = self.altitude_penalty * np.sum(
            np.square(altitude_violation), axis=1
        )
        if self.workspace_radius is not None:
            horizontal_radius = np.linalg.norm(
                predicted[:, :, :2], axis=2
            )
            workspace_violation = np.maximum(
                horizontal_radius - self.workspace_radius, 0.0
            )
            envelope_cost += self.workspace_penalty * np.sum(
                np.square(workspace_violation), axis=1
            )
        for obstacle in self.spherical_obstacles:
            center = np.asarray(obstacle[:3], dtype=np.float64)
            minimum_distance = obstacle[3] + self.collision_radius
            obstacle_distance = np.linalg.norm(
                predicted[:, :, :3] - center[None, None, :],
                axis=2,
            )
            obstacle_violation = np.maximum(
                minimum_distance - obstacle_distance, 0.0
            )
            envelope_cost += self.obstacle_penalty * np.sum(
                np.square(obstacle_violation), axis=1
            )
        return stage_cost + terminal_cost + envelope_cost

    def _state_cost(
        self,
        position_error: FloatArray,
        velocity_error: FloatArray,
        attitude_error: FloatArray,
        angular_velocity_error: FloatArray,
    ) -> FloatArray:
        return (
            np.sum(
                np.square(position_error)
                * np.asarray(self.position_weight)[None, None, :],
                axis=(1, 2),
            )
            + np.sum(
                np.square(velocity_error)
                * np.asarray(self.velocity_weight)[None, None, :],
                axis=(1, 2),
            )
            + np.sum(
                np.square(attitude_error)
                * np.asarray(self.attitude_weight)[None, None, :],
                axis=(1, 2),
            )
            + np.sum(
                np.square(angular_velocity_error)
                * np.asarray(self.angular_velocity_weight)[None, None, :],
                axis=(1, 2),
            )
        )
