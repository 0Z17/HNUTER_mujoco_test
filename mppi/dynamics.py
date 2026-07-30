"""Reduced-order dynamics models used for fast MPPI rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quaternion import (
    normalize_quaternion,
    quaternion_from_rotation_vector,
    quaternion_multiply,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PointMassDynamics:
    """World-frame translational model for the omnidirectional UAV.

    State is ``[x, y, z, vx, vy, vz]`` and control is commanded world-frame
    acceleration ``[ax, ay, az]``.  The high-bandwidth geometric controller in
    ``hnuter_control.py`` realizes this acceleration on the full MuJoCo model.
    """

    dt: float
    linear_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)

    state_dim: int = 6
    control_dim: int = 3

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if len(self.linear_drag) != 3 or any(
            coefficient < 0.0 for coefficient in self.linear_drag
        ):
            raise ValueError("linear_drag must contain three non-negative values")

    def rollout(
        self, initial_state: FloatArray, controls: FloatArray
    ) -> FloatArray:
        """Batch rollout using constant acceleration over each time interval."""

        initial_state = np.asarray(initial_state, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        if initial_state.shape != (self.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({self.state_dim},)"
            )
        if controls.ndim != 3 or controls.shape[2] != self.control_dim:
            raise ValueError(
                "controls must have shape (batch, horizon, control_dim)"
            )

        batch_size, horizon, _ = controls.shape
        states = np.empty(
            (batch_size, horizon + 1, self.state_dim), dtype=np.float64
        )
        states[:, 0, :] = initial_state
        drag = np.asarray(self.linear_drag, dtype=np.float64)
        dt = self.dt

        for index in range(horizon):
            position = states[:, index, :3]
            velocity = states[:, index, 3:]
            acceleration = controls[:, index, :] - drag * velocity
            states[:, index + 1, :3] = (
                position + velocity * dt + 0.5 * acceleration * dt**2
            )
            states[:, index + 1, 3:] = velocity + acceleration * dt

        return states


@dataclass(frozen=True)
class FullyActuatedUAVDynamics:
    """Reduced-order 6-DoF model for an omnidirectional UAV.

    State layout is ``[p_world(3), v_world(3), quaternion_wxyz(4),
    omega_body(3)]``.  Control is ``[linear_acceleration_world(3),
    angular_acceleration_body(3)]``.
    """

    dt: float
    linear_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_drag: tuple[float, float, float] = (0.0, 0.0, 0.0)

    state_dim: int = 13
    control_dim: int = 6

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        for name in ("linear_drag", "angular_drag"):
            values = getattr(self, name)
            if len(values) != 3 or any(value < 0.0 for value in values):
                raise ValueError(
                    f"{name} must contain three non-negative values"
                )

    def rollout(
        self, initial_state: FloatArray, controls: FloatArray
    ) -> FloatArray:
        initial_state = np.asarray(initial_state, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        if initial_state.shape != (self.state_dim,):
            raise ValueError(
                f"initial_state must have shape ({self.state_dim},)"
            )
        if controls.ndim != 3 or controls.shape[2] != self.control_dim:
            raise ValueError(
                "controls must have shape (batch, horizon, 6)"
            )

        batch_size, horizon, _ = controls.shape
        states = np.empty(
            (batch_size, horizon + 1, self.state_dim), dtype=np.float64
        )
        states[:, 0, :] = initial_state
        states[:, 0, 6:10] = normalize_quaternion(
            states[:, 0, 6:10]
        )
        linear_drag = np.asarray(self.linear_drag, dtype=np.float64)
        angular_drag = np.asarray(self.angular_drag, dtype=np.float64)
        dt = self.dt

        for index in range(horizon):
            position = states[:, index, :3]
            velocity = states[:, index, 3:6]
            quaternion = states[:, index, 6:10]
            angular_velocity = states[:, index, 10:13]

            linear_acceleration = (
                controls[:, index, :3] - linear_drag * velocity
            )
            angular_acceleration = (
                controls[:, index, 3:] - angular_drag * angular_velocity
            )
            angular_velocity_midpoint = (
                angular_velocity + 0.5 * angular_acceleration * dt
            )
            delta_quaternion = quaternion_from_rotation_vector(
                angular_velocity_midpoint * dt
            )

            states[:, index + 1, :3] = (
                position
                + velocity * dt
                + 0.5 * linear_acceleration * dt**2
            )
            states[:, index + 1, 3:6] = (
                velocity + linear_acceleration * dt
            )
            states[:, index + 1, 6:10] = normalize_quaternion(
                quaternion_multiply(quaternion, delta_quaternion)
            )
            states[:, index + 1, 10:13] = (
                angular_velocity + angular_acceleration * dt
            )

        return states
