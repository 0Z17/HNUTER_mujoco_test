"""Kinematic TOPP-RA retiming for the project's interpolating SE(3) path.

TOPP-RA sees only the scalar identity path ``y(s) = s``.  Position and
orientation remain on :class:`InterpolatingSE3BSpline`; custom constraints map
the scalar path speed and acceleration to world-frame linear kinematics and
body-frame angular kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
import toppra as ta
import toppra.algorithm as algo
from toppra.constraint import (
    DiscretizationType,
    LinearConstraint,
)
from toppra.constraint.linear_constraint import (
    canlinear_colloc_to_interpolate,
)

from mppi.quaternion import (
    quaternion_conjugate,
    quaternion_multiply,
)
from multi_waypoint_planner import (
    InterpolatingSE3BSpline,
    MultiWaypointPlan,
)


FloatArray = NDArray[np.float64]
_PATH_START: Final = 0.0
_PATH_END: Final = 1.0


@dataclass(frozen=True)
class SE3PathDerivatives:
    """SE(3) geometric quantities differentiated with respect to ``s``."""

    position: FloatArray
    quaternion_wxyz: FloatArray
    position_path_derivative: FloatArray
    position_path_second_derivative: FloatArray
    omega_path_body: FloatArray
    omega_path_body_derivative: FloatArray


class SE3PathKinematics:
    """Expose the derivatives required to retime an SE(3) B-spline."""

    def __init__(self, spline: InterpolatingSE3BSpline) -> None:
        self.spline = spline

    def evaluate(self, path_position: ArrayLike) -> SE3PathDerivatives:
        (
            pose,
            position_path_derivative,
            position_path_second_derivative,
            quaternion_path_derivative,
            quaternion_path_second_derivative,
        ) = self.spline.evaluate_with_second_derivatives(path_position)
        quaternion = pose[..., 3:7]
        omega_path_body = 2.0 * quaternion_multiply(
            quaternion_conjugate(quaternion),
            quaternion_path_derivative,
        )[..., 1:]
        omega_path_body_derivative = 2.0 * (
            quaternion_multiply(
                quaternion_conjugate(quaternion_path_derivative),
                quaternion_path_derivative,
            )
            + quaternion_multiply(
                quaternion_conjugate(quaternion),
                quaternion_path_second_derivative,
            )
        )[..., 1:]
        return SE3PathDerivatives(
            position=pose[..., :3],
            quaternion_wxyz=quaternion,
            position_path_derivative=position_path_derivative,
            position_path_second_derivative=(
                position_path_second_derivative
            ),
            omega_path_body=omega_path_body,
            omega_path_body_derivative=omega_path_body_derivative,
        )

    def position(self, path_position: ArrayLike) -> FloatArray:
        return self.evaluate(path_position).position

    def orientation(self, path_position: ArrayLike) -> FloatArray:
        return self.evaluate(path_position).quaternion_wxyz

    def position_derivative(
        self, path_position: ArrayLike
    ) -> FloatArray:
        return self.evaluate(path_position).position_path_derivative

    def position_second_derivative(
        self, path_position: ArrayLike
    ) -> FloatArray:
        return self.evaluate(path_position).position_path_second_derivative

    def body_angular_path_rate(
        self, path_position: ArrayLike
    ) -> FloatArray:
        return self.evaluate(path_position).omega_path_body

    def body_angular_path_rate_derivative(
        self, path_position: ArrayLike
    ) -> FloatArray:
        return self.evaluate(path_position).omega_path_body_derivative


class SE3SpeedConstraint(LinearConstraint):
    """Bound world linear-speed norm and body angular-speed norm."""

    def __init__(
        self,
        se3_path: SE3PathKinematics,
        max_linear_speed: float,
        max_angular_speed: float,
        *,
        safety_scale: float = 0.85,
        path_speed_ceiling: float = 1.0e3,
    ) -> None:
        super().__init__()
        if (
            max_linear_speed <= 0.0
            or max_angular_speed <= 0.0
            or not 0.0 < safety_scale <= 1.0
            or path_speed_ceiling <= 0.0
        ):
            raise ValueError("invalid SE(3) speed constraint")
        self.dof = 1
        self.se3_path = se3_path
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.safety_scale = float(safety_scale)
        self.path_speed_ceiling = float(path_speed_ceiling)
        self.set_discretization_type(DiscretizationType.Collocation)

    def compute_constraint_params(
        self, path: ta.AbstractGeometricPath, gridpoints: FloatArray
    ) -> tuple[
        None,
        None,
        None,
        None,
        None,
        None,
        FloatArray,
    ]:
        if path.dof != self.dof:
            raise ValueError(
                f"constraint dof {self.dof} does not match path dof "
                f"{path.dof}"
            )
        derivatives = self.se3_path.evaluate(gridpoints)
        linear_norm = np.linalg.norm(
            derivatives.position_path_derivative, axis=1
        )
        angular_norm = np.linalg.norm(
            derivatives.omega_path_body, axis=1
        )
        linear_limit = np.divide(
            self.max_linear_speed,
            linear_norm,
            out=np.full_like(linear_norm, np.inf),
            where=linear_norm > 1.0e-12,
        )
        angular_limit = np.divide(
            self.max_angular_speed,
            angular_norm,
            out=np.full_like(angular_norm, np.inf),
            where=angular_norm > 1.0e-12,
        )
        path_speed_limit = np.minimum(linear_limit, angular_limit)
        path_speed_limit = np.minimum(
            path_speed_limit, self.path_speed_ceiling
        )
        path_speed_limit *= self.safety_scale
        xbound = np.column_stack(
            (
                np.zeros(len(gridpoints), dtype=np.float64),
                np.square(path_speed_limit),
            )
        )
        return None, None, None, None, None, None, xbound


class SE3AccelerationConstraint(LinearConstraint):
    """Apply axis limits to ``a_world`` and ``alpha_body``."""

    def __init__(
        self,
        se3_path: SE3PathKinematics,
        linear_axis_max: ArrayLike,
        angular_axis_max: ArrayLike,
        *,
        safety_scale: float = 0.80,
        discretization: DiscretizationType = (
            DiscretizationType.Interpolation
        ),
    ) -> None:
        super().__init__()
        linear_limits = _positive_vector3(
            linear_axis_max, "linear_axis_max"
        )
        angular_limits = _positive_vector3(
            angular_axis_max, "angular_axis_max"
        )
        if not 0.0 < safety_scale <= 1.0:
            raise ValueError("safety_scale must lie in (0, 1]")
        self.dof = 1
        self.se3_path = se3_path
        self.linear_axis_max = linear_limits
        self.angular_axis_max = angular_limits
        self.safety_scale = float(safety_scale)
        effective_limits = (
            np.concatenate((linear_limits, angular_limits))
            * self.safety_scale
        )
        self.F = np.vstack((np.eye(6), -np.eye(6)))
        self.g = np.concatenate((effective_limits, effective_limits))
        self.identical = True
        self.set_discretization_type(discretization)

    def compute_constraint_params(
        self, path: ta.AbstractGeometricPath, gridpoints: FloatArray
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        None,
        None,
    ]:
        if path.dof != self.dof:
            raise ValueError(
                f"constraint dof {self.dof} does not match path dof "
                f"{path.dof}"
            )
        derivatives = self.se3_path.evaluate(gridpoints)
        a = np.concatenate(
            (
                derivatives.position_path_derivative,
                derivatives.omega_path_body,
            ),
            axis=1,
        )
        b = np.concatenate(
            (
                derivatives.position_path_second_derivative,
                derivatives.omega_path_body_derivative,
            ),
            axis=1,
        )
        c = np.zeros_like(a)
        if self.discretization_type == DiscretizationType.Collocation:
            return a, b, c, self.F, self.g, None, None
        if self.discretization_type == DiscretizationType.Interpolation:
            return canlinear_colloc_to_interpolate(
                a,
                b,
                c,
                self.F,
                self.g,
                None,
                None,
                gridpoints,
                identical=True,
            )
        raise NotImplementedError(
            "unsupported TOPP-RA acceleration discretization"
        )


@dataclass(frozen=True)
class ToppraValidationReport:
    """Independent dense validation of a scalar retiming solution."""

    valid: bool
    gridpoint_count: int
    validation_point_count: int
    duration: float
    max_linear_speed: float
    max_angular_speed: float
    max_abs_linear_acceleration_world: FloatArray
    max_abs_angular_acceleration_body: FloatArray
    minimum_path_speed: float
    start_path_speed: float
    end_path_speed: float
    monotonic: bool


@dataclass(frozen=True)
class ToppraRetimingResult:
    """Scalar TOPP-RA trajectory and its dense validation report."""

    scalar_trajectory: ta.AbstractGeometricPath
    gridpoints: FloatArray
    validation: ToppraValidationReport


class ToppraRetimer:
    """Solve rest-to-rest scalar TOPP-RA with SE(3) custom constraints."""

    def __init__(
        self,
        se3_path: SE3PathKinematics,
        *,
        max_linear_speed: float,
        max_angular_speed: float,
        max_linear_acceleration: ArrayLike,
        max_angular_acceleration: ArrayLike,
        velocity_scale: float = 0.85,
        acceleration_scale: float = 0.80,
        duration_scale: float = 1.0,
        gridpoint_count: int = 401,
        validation_point_count: int = 4001,
        max_refinement_iterations: int = 3,
        solver_wrapper: str = "seidel",
        validation_tolerance: float = 1.0e-5,
    ) -> None:
        if max_linear_speed <= 0.0 or max_angular_speed <= 0.0:
            raise ValueError("speed limits must be positive")
        if not 0.0 < velocity_scale <= 1.0:
            raise ValueError("velocity_scale must lie in (0, 1]")
        if not 0.0 < acceleration_scale <= 1.0:
            raise ValueError("acceleration_scale must lie in (0, 1]")
        if duration_scale < 1.0:
            raise ValueError("duration_scale must be at least 1")
        if gridpoint_count < 3 or validation_point_count < 3:
            raise ValueError("TOPP-RA grids must contain at least 3 points")
        if max_refinement_iterations < 0:
            raise ValueError(
                "max_refinement_iterations must be non-negative"
            )
        if validation_tolerance <= 0.0:
            raise ValueError("validation_tolerance must be positive")

        self.se3_path = se3_path
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.max_linear_acceleration = _positive_vector3(
            max_linear_acceleration, "max_linear_acceleration"
        )
        self.max_angular_acceleration = _positive_vector3(
            max_angular_acceleration, "max_angular_acceleration"
        )
        self.velocity_scale = float(velocity_scale)
        self.acceleration_scale = float(acceleration_scale)
        self.duration_scale = float(duration_scale)
        self.gridpoint_count = int(gridpoint_count)
        self.validation_point_count = int(validation_point_count)
        self.max_refinement_iterations = int(max_refinement_iterations)
        self.solver_wrapper = solver_wrapper
        self.validation_tolerance = float(validation_tolerance)

    def compute(self) -> ToppraRetimingResult:
        gridpoint_count = self.gridpoint_count
        last_report: ToppraValidationReport | None = None
        for _ in range(self.max_refinement_iterations + 1):
            gridpoints = np.unique(
                np.concatenate(
                    (
                        np.linspace(
                            _PATH_START, _PATH_END, gridpoint_count
                        ),
                        np.clip(
                            self.se3_path.spline.knots,
                            _PATH_START,
                            _PATH_END,
                        ),
                    )
                )
            )
            scalar_path = ta.SplineInterpolator(
                np.asarray([_PATH_START, _PATH_END]),
                np.asarray([[_PATH_START], [_PATH_END]]),
            )
            constraints = [
                SE3SpeedConstraint(
                    self.se3_path,
                    self.max_linear_speed,
                    self.max_angular_speed,
                    safety_scale=self.velocity_scale,
                ),
                SE3AccelerationConstraint(
                    self.se3_path,
                    self.max_linear_acceleration,
                    self.max_angular_acceleration,
                    safety_scale=self.acceleration_scale,
                ),
            ]
            instance = algo.TOPPRA(
                constraints,
                scalar_path,
                gridpoints=gridpoints,
                solver_wrapper=self.solver_wrapper,
                parametrizer="ParametrizeConstAccel",
            )
            scalar_trajectory = instance.compute_trajectory(
                sd_start=0.0, sd_end=0.0
            )
            if scalar_trajectory is None:
                return_code = getattr(
                    instance.problem_data, "return_code", "unknown"
                )
                raise RuntimeError(
                    "TOPP-RA failed to find a rest-to-rest trajectory "
                    f"({return_code})"
                )
            last_report = self._validate(
                scalar_trajectory, len(gridpoints)
            )
            if last_report.valid:
                return ToppraRetimingResult(
                    scalar_trajectory=scalar_trajectory,
                    gridpoints=gridpoints,
                    validation=last_report,
                )
            gridpoint_count = 2 * gridpoint_count - 1

        assert last_report is not None
        raise RuntimeError(
            "TOPP-RA solution failed dense validation after adaptive "
            f"refinement: vmax={last_report.max_linear_speed:.6g}, "
            f"omegamax={last_report.max_angular_speed:.6g}, "
            "amax="
            f"{last_report.max_abs_linear_acceleration_world.tolist()}, "
            "alphamax="
            f"{last_report.max_abs_angular_acceleration_body.tolist()}"
        )

    def _validate(
        self,
        scalar_trajectory: ta.AbstractGeometricPath,
        gridpoint_count: int,
    ) -> ToppraValidationReport:
        raw_duration = _trajectory_duration(scalar_trajectory)
        raw_times = np.linspace(
            0.0, raw_duration, self.validation_point_count
        )
        path_position = _scalar_trajectory_values(
            scalar_trajectory, raw_times, order=0
        )
        path_speed = (
            _scalar_trajectory_values(
                scalar_trajectory, raw_times, order=1
            )
            / self.duration_scale
        )
        path_acceleration = (
            _scalar_trajectory_values(
                scalar_trajectory, raw_times, order=2
            )
            / self.duration_scale**2
        )
        derivatives = self.se3_path.evaluate(path_position)
        linear_velocity = (
            derivatives.position_path_derivative * path_speed[:, None]
        )
        angular_velocity = (
            derivatives.omega_path_body * path_speed[:, None]
        )
        linear_acceleration = (
            derivatives.position_path_derivative
            * path_acceleration[:, None]
            + derivatives.position_path_second_derivative
            * np.square(path_speed)[:, None]
        )
        angular_acceleration = (
            derivatives.omega_path_body
            * path_acceleration[:, None]
            + derivatives.omega_path_body_derivative
            * np.square(path_speed)[:, None]
        )
        max_linear_speed = float(
            np.max(np.linalg.norm(linear_velocity, axis=1))
        )
        max_angular_speed = float(
            np.max(np.linalg.norm(angular_velocity, axis=1))
        )
        max_linear_acceleration = np.max(
            np.abs(linear_acceleration), axis=0
        )
        max_angular_acceleration = np.max(
            np.abs(angular_acceleration), axis=0
        )
        tolerance = self.validation_tolerance
        monotonic = bool(
            np.all(np.diff(path_position) >= -tolerance)
            and float(np.min(path_position)) >= _PATH_START - tolerance
            and float(np.max(path_position)) <= _PATH_END + tolerance
        )
        valid = bool(
            monotonic
            and float(np.min(path_speed)) >= -tolerance
            and abs(float(path_speed[0])) <= tolerance
            and abs(float(path_speed[-1])) <= tolerance
            and max_linear_speed
            <= self.max_linear_speed * (1.0 + tolerance) + tolerance
            and max_angular_speed
            <= self.max_angular_speed * (1.0 + tolerance) + tolerance
            and np.all(
                max_linear_acceleration
                <= self.max_linear_acceleration * (1.0 + tolerance)
                + tolerance
            )
            and np.all(
                max_angular_acceleration
                <= self.max_angular_acceleration * (1.0 + tolerance)
                + tolerance
            )
        )
        return ToppraValidationReport(
            valid=valid,
            gridpoint_count=gridpoint_count,
            validation_point_count=self.validation_point_count,
            duration=raw_duration * self.duration_scale,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_abs_linear_acceleration_world=(
                max_linear_acceleration
            ),
            max_abs_angular_acceleration_body=(
                max_angular_acceleration
            ),
            minimum_path_speed=float(np.min(path_speed)),
            start_path_speed=float(path_speed[0]),
            end_path_speed=float(path_speed[-1]),
            monotonic=monotonic,
        )


@dataclass(frozen=True)
class SE3TrajectorySamples:
    """Fully sampled retimed trajectory, including acceleration feedforward."""

    time: FloatArray
    path_position: FloatArray
    path_speed: FloatArray
    path_acceleration: FloatArray
    position: FloatArray
    linear_velocity_world: FloatArray
    linear_acceleration_world: FloatArray
    quaternion_wxyz: FloatArray
    angular_velocity_body: FloatArray
    angular_acceleration_body: FloatArray

    @property
    def reference(self) -> FloatArray:
        """Return the existing MPPI ``[p, v, q, omega_body]`` layout."""

        return np.concatenate(
            (
                self.position,
                self.linear_velocity_world,
                self.quaternion_wxyz,
                self.angular_velocity_body,
            ),
            axis=1,
        )


class ToppraTimedReference:
    """Drop-in MPPI reference backed by a kinematic TOPP-RA trajectory."""

    def __init__(
        self,
        plan_or_spline: MultiWaypointPlan | InterpolatingSE3BSpline,
        *,
        max_linear_speed: float = 1.05,
        max_angular_speed: float = 1.50,
        max_linear_acceleration: ArrayLike = (4.0, 4.0, 3.5),
        max_angular_acceleration: ArrayLike = (6.0, 6.0, 5.0),
        start_delay: float = 0.35,
        duration_scale: float = 1.0,
        gridpoint_count: int = 401,
        validation_point_count: int = 4001,
        max_refinement_iterations: int = 3,
        velocity_scale: float = 0.85,
        acceleration_scale: float = 0.80,
        solver_wrapper: str = "seidel",
    ) -> None:
        if start_delay < 0.0:
            raise ValueError("start_delay must be non-negative")
        if isinstance(plan_or_spline, MultiWaypointPlan):
            self.plan: MultiWaypointPlan | None = plan_or_spline
            self.spline = plan_or_spline.spline
            waypoint_parameters = plan_or_spline.waypoint_parameters
        elif isinstance(plan_or_spline, InterpolatingSE3BSpline):
            self.plan = None
            self.spline = plan_or_spline
            waypoint_parameters = np.asarray([_PATH_START, _PATH_END])
        else:
            raise TypeError(
                "plan_or_spline must be MultiWaypointPlan or "
                "InterpolatingSE3BSpline"
            )

        self.se3_path = SE3PathKinematics(self.spline)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.max_linear_acceleration = _positive_vector3(
            max_linear_acceleration, "max_linear_acceleration"
        )
        self.max_angular_acceleration = _positive_vector3(
            max_angular_acceleration, "max_angular_acceleration"
        )
        self.start_delay = float(start_delay)
        self.duration_scale = float(duration_scale)
        self.velocity_scale = float(velocity_scale)
        self.acceleration_scale = float(acceleration_scale)
        retimer = ToppraRetimer(
            self.se3_path,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            max_linear_acceleration=self.max_linear_acceleration,
            max_angular_acceleration=self.max_angular_acceleration,
            velocity_scale=self.velocity_scale,
            acceleration_scale=self.acceleration_scale,
            duration_scale=self.duration_scale,
            gridpoint_count=gridpoint_count,
            validation_point_count=validation_point_count,
            max_refinement_iterations=max_refinement_iterations,
            solver_wrapper=solver_wrapper,
        )
        result = retimer.compute()
        self.scalar_trajectory = result.scalar_trajectory
        self.gridpoints = result.gridpoints
        self.validation = result.validation
        self._raw_duration = _trajectory_duration(
            self.scalar_trajectory
        )
        self.duration = self._raw_duration * self.duration_scale
        self.finish_time = self.start_delay + self.duration
        self.waypoint_arrival_times = self._waypoint_arrival_times(
            waypoint_parameters
        )

    def sample(self, times: ArrayLike) -> FloatArray:
        """Sample the existing 13-state MPPI reference layout."""

        return self.sample_full(times).reference

    def sample_full(self, times: ArrayLike) -> SE3TrajectorySamples:
        """Sample pose, velocities, accelerations and scalar path states."""

        time_array = np.asarray(times, dtype=np.float64)
        if time_array.ndim != 1 or not np.all(np.isfinite(time_array)):
            raise ValueError("times must be a finite one-dimensional array")
        raw_time = np.clip(
            (time_array - self.start_delay) / self.duration_scale,
            0.0,
            self._raw_duration,
        )
        path_position = np.clip(
            _scalar_trajectory_values(
                self.scalar_trajectory, raw_time, order=0
            ),
            _PATH_START,
            _PATH_END,
        )
        path_speed = (
            _scalar_trajectory_values(
                self.scalar_trajectory, raw_time, order=1
            )
            / self.duration_scale
        )
        path_acceleration = (
            _scalar_trajectory_values(
                self.scalar_trajectory, raw_time, order=2
            )
            / self.duration_scale**2
        )
        inactive = (time_array <= self.start_delay) | (
            time_array >= self.finish_time
        )
        path_speed[inactive] = 0.0
        path_acceleration[inactive] = 0.0

        derivatives = self.se3_path.evaluate(path_position)
        path_speed_squared = np.square(path_speed)[:, None]
        linear_velocity = (
            derivatives.position_path_derivative * path_speed[:, None]
        )
        angular_velocity = (
            derivatives.omega_path_body * path_speed[:, None]
        )
        linear_acceleration = (
            derivatives.position_path_derivative
            * path_acceleration[:, None]
            + derivatives.position_path_second_derivative
            * path_speed_squared
        )
        angular_acceleration = (
            derivatives.omega_path_body
            * path_acceleration[:, None]
            + derivatives.omega_path_body_derivative
            * path_speed_squared
        )
        return SE3TrajectorySamples(
            time=time_array.copy(),
            path_position=path_position,
            path_speed=path_speed,
            path_acceleration=path_acceleration,
            position=derivatives.position,
            linear_velocity_world=linear_velocity,
            linear_acceleration_world=linear_acceleration,
            quaternion_wxyz=derivatives.quaternion_wxyz,
            angular_velocity_body=angular_velocity,
            angular_acceleration_body=angular_acceleration,
        )

    def _waypoint_arrival_times(
        self, waypoint_parameters: FloatArray
    ) -> FloatArray:
        raw_time = np.linspace(
            0.0,
            self._raw_duration,
            max(10001, 10 * len(self.gridpoints) + 1),
        )
        path_position = np.maximum.accumulate(
            _scalar_trajectory_values(
                self.scalar_trajectory, raw_time, order=0
            )
        )
        return self.start_delay + self.duration_scale * np.interp(
            waypoint_parameters, path_position, raw_time
        )


def _positive_vector3(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape != (3,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError(f"{name} must contain three positive values")
    return array.copy()


def _trajectory_duration(
    trajectory: ta.AbstractGeometricPath,
) -> float:
    interval = np.asarray(trajectory.path_interval, dtype=np.float64)
    duration = float(interval[-1] - interval[0])
    if not np.isfinite(duration) or duration <= 0.0:
        raise RuntimeError("TOPP-RA returned an invalid trajectory duration")
    return duration


def _scalar_trajectory_values(
    trajectory: ta.AbstractGeometricPath,
    times: FloatArray,
    *,
    order: int,
) -> FloatArray:
    values = np.asarray(
        trajectory(times, order=order), dtype=np.float64
    )
    if values.size != len(times):
        raise RuntimeError(
            "TOPP-RA scalar trajectory returned an unexpected shape "
            f"{values.shape}"
        )
    return values.reshape(-1)
