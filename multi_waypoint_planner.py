"""Multi-waypoint OMPL planning with a global interpolating SE(3) B-spline.

Each consecutive waypoint pair is planned by OMPL RRTConnect.  The resulting
collision-free segments are concatenated and interpolated by one clamped cubic
B-spline.  Position and sign-continuous quaternion coefficients share the same
chord-length parameterization; quaternions are normalized after evaluation.

The timed reference uses the analytic spline derivatives and allocates time
against both linear- and angular-speed limits.  A minimum-jerk time warp makes
the vehicle start and finish at rest while passing intermediate waypoints
without stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mppi.quaternion import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    PlannedSE3Path,
    SE3Pose,
)


FloatArray = NDArray[np.float64]


class InterpolatingSE3BSpline:
    """Clamped interpolating B-spline for position and quaternion samples."""

    def __init__(
        self,
        states: ArrayLike,
        parameters: ArrayLike | None = None,
        degree: int = 3,
        orientation_metric_weight: float = 0.35,
    ) -> None:
        state_array = np.asarray(states, dtype=np.float64)
        if (
            state_array.ndim != 2
            or state_array.shape[1] != 7
            or len(state_array) < 2
            or not np.all(np.isfinite(state_array))
        ):
            raise ValueError("states must have shape (N, 7), N >= 2")
        if degree < 1:
            raise ValueError("degree must be positive")
        if orientation_metric_weight < 0.0:
            raise ValueError(
                "orientation_metric_weight must be non-negative"
            )

        self.states = state_array.copy()
        self.states[:, 3:7] = normalize_quaternion(
            self.states[:, 3:7]
        )
        _make_quaternions_continuous(self.states[:, 3:7])
        self.degree = min(int(degree), len(self.states) - 1)
        if parameters is None:
            self.parameters = _se3_chord_parameters(
                self.states, orientation_metric_weight
            )
        else:
            self.parameters = np.asarray(
                parameters, dtype=np.float64
            )
            if (
                self.parameters.shape != (len(self.states),)
                or not np.all(np.isfinite(self.parameters))
                or abs(float(self.parameters[0])) > 1.0e-12
                or abs(float(self.parameters[-1] - 1.0)) > 1.0e-12
                or np.any(np.diff(self.parameters) <= 0.0)
            ):
                raise ValueError(
                    "parameters must be strictly increasing from 0 to 1"
                )
        self.knots = _averaged_clamped_knots(
            self.parameters, self.degree
        )
        interpolation_matrix = _basis_matrix(
            self.parameters,
            self.degree,
            self.knots,
            len(self.states),
        )
        condition_number = float(
            np.linalg.cond(interpolation_matrix)
        )
        if not np.isfinite(condition_number) or condition_number > 1.0e12:
            raise RuntimeError(
                "B-spline interpolation matrix is ill-conditioned "
                f"(condition={condition_number:.3e})"
            )
        self.position_control_points = np.linalg.solve(
            interpolation_matrix, self.states[:, :3]
        )
        self.quaternion_control_points = np.linalg.solve(
            interpolation_matrix, self.states[:, 3:7]
        )

    def evaluate(self, parameters: ArrayLike) -> FloatArray:
        states, _, _ = self.evaluate_with_derivatives(parameters)
        return states

    def evaluate_with_derivatives(
        self, parameters: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return states, ``dp/du``, and normalized ``dq/du``."""

        values = np.asarray(parameters, dtype=np.float64)
        scalar_input = values.ndim == 0
        values = np.atleast_1d(values)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite scalars or a 1D array")
        if np.any(values < -1.0e-12) or np.any(values > 1.0 + 1.0e-12):
            raise ValueError("B-spline parameters must lie in [0, 1]")
        values = np.clip(values, 0.0, 1.0)

        basis = _basis_matrix(
            values,
            self.degree,
            self.knots,
            len(self.states),
        )
        derivative_basis = _basis_derivative_matrix(
            values,
            self.degree,
            self.knots,
            len(self.states),
        )
        positions = basis @ self.position_control_points
        position_derivative = (
            derivative_basis @ self.position_control_points
        )
        raw_quaternion = basis @ self.quaternion_control_points
        raw_derivative = (
            derivative_basis @ self.quaternion_control_points
        )
        raw_norm = np.linalg.norm(
            raw_quaternion, axis=1, keepdims=True
        )
        if np.any(raw_norm < 1.0e-8):
            raise RuntimeError(
                "quaternion B-spline approached zero norm"
            )
        quaternions = raw_quaternion / raw_norm
        tangent_projection = np.sum(
            quaternions * raw_derivative, axis=1, keepdims=True
        )
        quaternion_derivative = (
            raw_derivative - quaternions * tangent_projection
        ) / raw_norm
        states = np.concatenate((positions, quaternions), axis=1)
        if scalar_input:
            return (
                states[0],
                position_derivative[0],
                quaternion_derivative[0],
            )
        return states, position_derivative, quaternion_derivative


@dataclass(frozen=True)
class MultiWaypointPlan:
    """Segment plans plus their globally stitched B-spline."""

    waypoints: tuple[SE3Pose, ...]
    segment_paths: tuple[PlannedSE3Path, ...]
    raw_states: FloatArray
    spline: InterpolatingSE3BSpline
    spline_path: PlannedSE3Path
    waypoint_parameters: FloatArray
    waypoint_path_indices: tuple[int, ...]
    minimum_clearance_m: float
    knot_stride_used: int

    @property
    def intermediate_waypoints(self) -> tuple[SE3Pose, ...]:
        return self.waypoints[1:-1]


class MultiWaypointOMPLPlanner:
    """Plan consecutive waypoint segments and globally B-spline stitch them."""

    def __init__(self, planner: OMPLSE3Planner) -> None:
        self.planner = planner

    def plan(
        self,
        waypoints: Sequence[SE3Pose],
        *,
        solve_time_per_segment: float = 1.5,
        interpolation_resolution: float = 0.07,
        minimum_states_per_segment: int = 50,
        knot_stride: int = 3,
        spline_samples: int = 1000,
        orientation_metric_weight: float = 0.35,
    ) -> MultiWaypointPlan:
        waypoint_tuple = tuple(waypoints)
        if len(waypoint_tuple) < 2:
            raise ValueError("at least start and goal poses are required")
        if knot_stride < 1 or spline_samples < 20:
            raise ValueError(
                "knot_stride must be positive and spline_samples >= 20"
            )

        segments = tuple(
            self.planner.plan(
                start,
                goal,
                solve_time=solve_time_per_segment,
                interpolation_resolution=interpolation_resolution,
                minimum_waypoints=minimum_states_per_segment,
                simplify=True,
            )
            for start, goal in zip(
                waypoint_tuple[:-1], waypoint_tuple[1:]
            )
        )
        raw_states, waypoint_indices = _concatenate_segments(segments)

        last_error: RuntimeError | None = None
        for stride in range(knot_stride, 0, -1):
            selected_indices = sorted(
                set(range(0, len(raw_states), stride))
                | set(waypoint_indices)
                | {len(raw_states) - 1}
            )
            interpolation_states = raw_states[selected_indices]
            try:
                spline = InterpolatingSE3BSpline(
                    interpolation_states,
                    orientation_metric_weight=orientation_metric_weight,
                )
                selected_lookup = {
                    raw_index: selected_index
                    for selected_index, raw_index in enumerate(
                        selected_indices
                    )
                }
                waypoint_parameters = np.asarray(
                    [
                        spline.parameters[selected_lookup[index]]
                        for index in waypoint_indices
                    ]
                )
                plan = self._build_validated_plan(
                    waypoint_tuple,
                    segments,
                    raw_states,
                    waypoint_indices,
                    spline,
                    waypoint_parameters,
                    spline_samples,
                    stride,
                )
                return plan
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(
            "global B-spline stitching failed even with every OMPL state "
            f"used as an interpolation node: {last_error}"
        )

    def _build_validated_plan(
        self,
        waypoints: tuple[SE3Pose, ...],
        segments: tuple[PlannedSE3Path, ...],
        raw_states: FloatArray,
        waypoint_indices: tuple[int, ...],
        spline: InterpolatingSE3BSpline,
        waypoint_parameters: FloatArray,
        spline_samples: int,
        stride: int,
    ) -> MultiWaypointPlan:
        parameters = np.linspace(0.0, 1.0, spline_samples)
        spline_states = spline.evaluate(parameters)
        clearance = self.planner.clearance(spline_states[:, :3])
        in_bounds = np.all(
            (spline_states[:, :3] >= self.planner.bounds_min)
            & (spline_states[:, :3] <= self.planner.bounds_max),
            axis=1,
        )
        if not np.all(in_bounds):
            raise RuntimeError("B-spline left the planning workspace")
        if np.any(clearance <= 0.0):
            raise RuntimeError(
                "B-spline collided with an inflated obstacle "
                f"(minimum clearance {float(np.min(clearance)):.4f} m)"
            )

        waypoint_states = spline.evaluate(waypoint_parameters)
        waypoint_positions = np.asarray(
            [waypoint.position for waypoint in waypoints]
        )
        waypoint_quaternions = np.asarray(
            [waypoint.quaternion for waypoint in waypoints]
        )
        position_error = np.linalg.norm(
            waypoint_states[:, :3] - waypoint_positions, axis=1
        )
        attitude_error = np.linalg.norm(
            _relative_rotation_vectors(
                waypoint_states[:, 3:7], waypoint_quaternions
            ),
            axis=1,
        )
        if (
            float(np.max(position_error)) > 1.0e-7
            or float(np.max(attitude_error)) > 1.0e-7
        ):
            raise RuntimeError(
                "interpolating B-spline did not pass waypoint poses"
            )

        translation_delta = np.diff(spline_states[:, :3], axis=0)
        rotation_delta = _relative_rotation_vectors(
            spline_states[:-1, 3:7], spline_states[1:, 3:7]
        )
        spline_path = PlannedSE3Path(
            states=spline_states,
            planning_time_s=float(
                sum(segment.planning_time_s for segment in segments)
            ),
            raw_state_count=int(
                sum(segment.raw_state_count for segment in segments)
            ),
            path_length_m=float(
                np.sum(np.linalg.norm(translation_delta, axis=1))
            ),
            rotation_length_rad=float(
                np.sum(np.linalg.norm(rotation_delta, axis=1))
            ),
            planner_name=(
                f"OMPL RRTConnect x{len(segments)} + global cubic "
                "SE(3) B-spline"
            ),
        )
        path_indices = tuple(
            int(np.argmin(np.abs(parameters - parameter)))
            for parameter in waypoint_parameters
        )
        return MultiWaypointPlan(
            waypoints=waypoints,
            segment_paths=segments,
            raw_states=raw_states,
            spline=spline,
            spline_path=spline_path,
            waypoint_parameters=waypoint_parameters,
            waypoint_path_indices=path_indices,
            minimum_clearance_m=float(np.min(clearance)),
            knot_stride_used=stride,
        )


class BSplineTimeParameterizedReference:
    """Speed-limited time allocation for an SE(3) B-spline."""

    def __init__(
        self,
        plan: MultiWaypointPlan,
        *,
        max_linear_speed: float = 1.0,
        max_angular_speed: float = 1.4,
        start_delay: float = 0.35,
        duration_scale: float = 1.08,
        timing_samples: int = 4000,
    ) -> None:
        if (
            max_linear_speed <= 0.0
            or max_angular_speed <= 0.0
            or start_delay < 0.0
            or duration_scale <= 0.0
            or timing_samples < 100
        ):
            raise ValueError("invalid B-spline timing parameter")
        self.plan = plan
        self.spline = plan.spline
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.start_delay = float(start_delay)
        self.duration_scale = float(duration_scale)

        self._timing_parameters = np.linspace(
            0.0, 1.0, timing_samples
        )
        states, position_du, quaternion_du = (
            self.spline.evaluate_with_derivatives(
                self._timing_parameters
            )
        )
        angular_du = _body_angular_rate_per_parameter(
            states[:, 3:7], quaternion_du
        )
        seconds_per_parameter = np.maximum(
            np.linalg.norm(position_du, axis=1)
            / self.max_linear_speed,
            np.linalg.norm(angular_du, axis=1)
            / self.max_angular_speed,
        )
        seconds_per_parameter = np.maximum(
            seconds_per_parameter, 1.0e-6
        )
        parameter_delta = np.diff(self._timing_parameters)
        segment_time = (
            0.5
            * (
                seconds_per_parameter[:-1]
                + seconds_per_parameter[1:]
            )
            * parameter_delta
        )
        self._raw_cumulative_time = np.concatenate(
            ([0.0], np.cumsum(segment_time))
        )
        self._raw_duration = float(self._raw_cumulative_time[-1])
        self.duration = (
            1.875 * self._raw_duration * self.duration_scale
        )
        self.finish_time = self.start_delay + self.duration
        self.waypoint_arrival_times = self._waypoint_arrival_times()

    def sample(self, times: ArrayLike) -> FloatArray:
        time_array = np.asarray(times, dtype=np.float64)
        if time_array.ndim != 1 or not np.all(np.isfinite(time_array)):
            raise ValueError("times must be a finite one-dimensional array")
        phase = np.clip(
            (time_array - self.start_delay) / self.duration,
            0.0,
            1.0,
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
        raw_time = progress * self._raw_duration

        indices = np.searchsorted(
            self._raw_cumulative_time, raw_time, side="right"
        ) - 1
        indices = np.clip(
            indices, 0, len(self._timing_parameters) - 2
        )
        raw_start = self._raw_cumulative_time[indices]
        raw_delta = (
            self._raw_cumulative_time[indices + 1] - raw_start
        )
        interpolation = np.clip(
            (raw_time - raw_start) / raw_delta, 0.0, 1.0
        )
        parameter_start = self._timing_parameters[indices]
        parameter_delta = (
            self._timing_parameters[indices + 1] - parameter_start
        )
        parameters = (
            parameter_start + interpolation * parameter_delta
        )
        parameter_rate = (
            parameter_delta
            / raw_delta
            * self._raw_duration
            * progress_rate
        )

        pose, position_du, quaternion_du = (
            self.spline.evaluate_with_derivatives(parameters)
        )
        reference = np.zeros((len(time_array), 13), dtype=np.float64)
        reference[:, :3] = pose[:, :3]
        reference[:, 3:6] = position_du * parameter_rate[:, None]
        reference[:, 6:10] = pose[:, 3:7]
        reference[:, 10:13] = (
            _body_angular_rate_per_parameter(
                pose[:, 3:7], quaternion_du
            )
            * parameter_rate[:, None]
        )
        return reference

    def _waypoint_arrival_times(self) -> FloatArray:
        raw_waypoint_time = np.interp(
            self.plan.waypoint_parameters,
            self._timing_parameters,
            self._raw_cumulative_time,
        )
        target_progress = raw_waypoint_time / self._raw_duration
        phase_grid = np.linspace(0.0, 1.0, 20001)
        progress_grid = (
            10.0 * phase_grid**3
            - 15.0 * phase_grid**4
            + 6.0 * phase_grid**5
        )
        phases = np.interp(
            target_progress, progress_grid, phase_grid
        )
        return self.start_delay + phases * self.duration


def _concatenate_segments(
    segments: Sequence[PlannedSE3Path],
) -> tuple[FloatArray, tuple[int, ...]]:
    if not segments:
        raise ValueError("at least one segment is required")
    chunks: list[FloatArray] = []
    waypoint_indices = [0]
    state_count = 0
    previous_quaternion: FloatArray | None = None
    for segment_index, segment in enumerate(segments):
        states = segment.states.copy()
        if (
            previous_quaternion is not None
            and np.dot(previous_quaternion, states[0, 3:7]) < 0.0
        ):
            states[:, 3:7] *= -1.0
        if segment_index > 0:
            states = states[1:]
        chunks.append(states)
        state_count += len(states)
        waypoint_indices.append(state_count - 1)
        previous_quaternion = states[-1, 3:7]
    combined = np.concatenate(chunks, axis=0)
    _make_quaternions_continuous(combined[:, 3:7])
    return combined, tuple(waypoint_indices)


def _se3_chord_parameters(
    states: FloatArray, orientation_weight: float
) -> FloatArray:
    translation = np.linalg.norm(
        np.diff(states[:, :3], axis=0), axis=1
    )
    rotation = np.linalg.norm(
        _relative_rotation_vectors(
            states[:-1, 3:7], states[1:, 3:7]
        ),
        axis=1,
    )
    distance = translation + orientation_weight * rotation
    if np.any(distance <= 1.0e-10):
        distance = np.maximum(distance, 1.0e-10)
    cumulative = np.concatenate(([0.0], np.cumsum(distance)))
    return cumulative / cumulative[-1]


def _averaged_clamped_knots(
    parameters: FloatArray, degree: int
) -> FloatArray:
    count = len(parameters)
    knots = np.zeros(count + degree + 1, dtype=np.float64)
    knots[-(degree + 1) :] = 1.0
    last_control_index = count - 1
    for index in range(1, last_control_index - degree + 1):
        knots[index + degree] = float(
            np.mean(parameters[index : index + degree])
        )
    return knots


def _basis_values(
    parameter: float,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    return _basis_matrix(
        np.asarray([parameter]),
        degree,
        knots,
        control_count,
    )[0]


def _basis_matrix(
    parameters: FloatArray,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    values = np.asarray(parameters, dtype=np.float64)
    values_2d = values[:, None]
    basis = (
        (values_2d >= knots[None, :-1])
        & (values_2d < knots[None, 1:])
    ).astype(np.float64)
    endpoint_rows = values >= 1.0 - 1.0e-14
    basis[endpoint_rows] = 0.0
    basis[endpoint_rows, control_count - 1] = 1.0
    if degree == 0:
        return basis[:, :control_count]

    for current_degree in range(1, degree + 1):
        next_count = basis.shape[1] - 1
        left_denominator = (
            knots[current_degree : current_degree + next_count]
            - knots[:next_count]
        )
        right_denominator = (
            knots[current_degree + 1 : current_degree + 1 + next_count]
            - knots[1 : 1 + next_count]
        )
        left_coefficient = np.divide(
            values_2d - knots[None, :next_count],
            left_denominator[None, :],
            out=np.zeros((len(values), next_count)),
            where=left_denominator[None, :] > 0.0,
        )
        right_coefficient = np.divide(
            (
                knots[
                    None,
                    current_degree
                    + 1 : current_degree
                    + 1
                    + next_count,
                ]
                - values_2d
            ),
            right_denominator[None, :],
            out=np.zeros((len(values), next_count)),
            where=right_denominator[None, :] > 0.0,
        )
        basis = (
            left_coefficient * basis[:, :next_count]
            + right_coefficient * basis[:, 1 : next_count + 1]
        )
        basis[endpoint_rows] = 0.0
        if current_degree == degree:
            basis[endpoint_rows, control_count - 1] = 1.0
    return basis[:, :control_count]


def _basis_derivative_matrix(
    parameters: FloatArray,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    values = np.asarray(parameters, dtype=np.float64)
    lower_values = np.where(
        values >= 1.0 - 1.0e-14, 1.0 - 1.0e-12, values
    )
    lower_basis = _basis_matrix(
        lower_values,
        degree - 1,
        knots,
        control_count + 1,
    )
    left_denominator = (
        knots[degree : degree + control_count]
        - knots[:control_count]
    )
    right_denominator = (
        knots[degree + 1 : degree + 1 + control_count]
        - knots[1 : 1 + control_count]
    )
    left = np.divide(
        degree * lower_basis[:, :control_count],
        left_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=left_denominator[None, :] > 0.0,
    )
    right = np.divide(
        degree * lower_basis[:, 1 : control_count + 1],
        right_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=right_denominator[None, :] > 0.0,
    )
    return left - right


def _basis_derivatives(
    parameter: float,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    return _basis_derivative_matrix(
        np.asarray([parameter]),
        degree,
        knots,
        control_count,
    )[0]


def _relative_rotation_vectors(
    start_quaternion: ArrayLike, end_quaternion: ArrayLike
) -> FloatArray:
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


def _body_angular_rate_per_parameter(
    quaternion: FloatArray, quaternion_derivative: FloatArray
) -> FloatArray:
    tangent_quaternion = quaternion_multiply(
        quaternion_conjugate(quaternion), quaternion_derivative
    )
    return 2.0 * tangent_quaternion[..., 1:]


def _make_quaternions_continuous(quaternions: FloatArray) -> None:
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
