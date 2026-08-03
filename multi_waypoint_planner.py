"""Multi-waypoint OMPL planning with a globally smoothed SE(3) B-spline.

Each consecutive waypoint pair is planned by OMPL RRTConnect.  The resulting
collision-free segments are concatenated and used as a soft guide for one
clamped B-spline.  Mission waypoint poses remain hard equality constraints,
while acceleration and jerk penalties suppress the sharp turns inherited from
the sampling-based path.

The spline exposes analytic first and second path derivatives so that either
the legacy minimum-jerk retimer or the kinematic TOPP-RA retimer can preserve
the same geometric path.
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
        self.control_point_count = len(self.states)
        self.method_name = "interpolating"
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
            self.control_point_count,
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

        states, position_derivative, _, quaternion_derivative, _ = (
            self.evaluate_with_second_derivatives(parameters)
        )
        return states, position_derivative, quaternion_derivative

    def evaluate_with_second_derivatives(
        self, parameters: ArrayLike
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """Return pose plus first/second derivatives with respect to path.

        Quaternion derivatives are those of the normalized ``[w, x, y, z]``
        quaternion curve, not of its unnormalized four-dimensional B-spline.
        """

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
            self.control_point_count,
        )
        derivative_basis = _basis_derivative_matrix(
            values,
            self.degree,
            self.knots,
            self.control_point_count,
        )
        second_derivative_basis = _basis_second_derivative_matrix(
            values,
            self.degree,
            self.knots,
            self.control_point_count,
        )
        positions = basis @ self.position_control_points
        position_derivative = (
            derivative_basis @ self.position_control_points
        )
        position_second_derivative = (
            second_derivative_basis @ self.position_control_points
        )
        raw_quaternion = basis @ self.quaternion_control_points
        raw_derivative = (
            derivative_basis @ self.quaternion_control_points
        )
        raw_second_derivative = (
            second_derivative_basis @ self.quaternion_control_points
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
        tangent_projection_derivative = np.sum(
            quaternion_derivative * raw_derivative
            + quaternions * raw_second_derivative,
            axis=1,
            keepdims=True,
        )
        quaternion_second_derivative = (
            (
                raw_second_derivative
                - quaternions * tangent_projection_derivative
            )
            / raw_norm
            - 2.0
            * quaternion_derivative
            * tangent_projection
            / raw_norm
        )
        states = np.concatenate((positions, quaternions), axis=1)
        if scalar_input:
            return (
                states[0],
                position_derivative[0],
                position_second_derivative[0],
                quaternion_derivative[0],
                quaternion_second_derivative[0],
            )
        return (
            states,
            position_derivative,
            position_second_derivative,
            quaternion_derivative,
            quaternion_second_derivative,
        )


class WaypointConstrainedSmoothingSE3BSpline(
    InterpolatingSE3BSpline
):
    """Smooth an OMPL guide while exactly preserving mission waypoint poses.

    The optimization is a linearly equality-constrained least-squares problem.
    OMPL samples are soft observations.  Position and raw quaternion second and
    third path derivatives are regularized; evaluated quaternions are
    normalized exactly as in :class:`InterpolatingSE3BSpline`.
    """

    def __init__(
        self,
        guide_states: ArrayLike,
        waypoint_indices: Sequence[int],
        *,
        parameters: ArrayLike | None = None,
        degree: int = 5,
        control_point_count: int | None = None,
        control_point_stride: int = 6,
        orientation_metric_weight: float = 0.35,
        guide_weight: float = 1.0,
        guide_sample_weights: ArrayLike | None = None,
        position_acceleration_weight: float = 1.0e-8,
        position_jerk_weight: float = 1.0e-12,
        orientation_acceleration_weight: float = 2.5e-9,
        orientation_jerk_weight: float = 2.5e-13,
        regularization: float = 1.0e-10,
    ) -> None:
        state_array = np.asarray(guide_states, dtype=np.float64)
        if (
            state_array.ndim != 2
            or state_array.shape[1] != 7
            or len(state_array) < 2
            or not np.all(np.isfinite(state_array))
        ):
            raise ValueError("guide_states must have shape (N, 7), N >= 2")
        waypoint_index_array = np.asarray(
            waypoint_indices, dtype=np.int64
        )
        if (
            waypoint_index_array.ndim != 1
            or len(waypoint_index_array) < 2
            or waypoint_index_array[0] != 0
            or waypoint_index_array[-1] != len(state_array) - 1
            or np.any(np.diff(waypoint_index_array) <= 0)
        ):
            raise ValueError(
                "waypoint_indices must be increasing and include endpoints"
            )
        if degree < 3 or control_point_stride < 1:
            raise ValueError(
                "smoothing degree must be >= 3 and stride positive"
            )
        weights = np.asarray(
            [
                guide_weight,
                position_acceleration_weight,
                position_jerk_weight,
                orientation_acceleration_weight,
                orientation_jerk_weight,
                regularization,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("smoothing weights must be finite and non-negative")
        if guide_weight <= 0.0 or regularization <= 0.0:
            raise ValueError(
                "guide_weight and regularization must be positive"
            )
        if orientation_metric_weight < 0.0:
            raise ValueError(
                "orientation_metric_weight must be non-negative"
            )

        self.states = state_array.copy()
        self.states[:, 3:7] = normalize_quaternion(
            self.states[:, 3:7]
        )
        _make_quaternions_continuous(self.states[:, 3:7])
        if parameters is None:
            self.parameters = _se3_chord_parameters(
                self.states, orientation_metric_weight
            )
        else:
            self.parameters = np.asarray(parameters, dtype=np.float64)
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

        requested_control_count = (
            int(control_point_count)
            if control_point_count is not None
            else int(np.ceil(len(self.states) / control_point_stride))
        )
        minimum_control_count = max(
            int(degree) + 1, len(waypoint_index_array) + int(degree) - 1
        )
        self.control_point_count = min(
            len(self.states),
            max(requested_control_count, minimum_control_count),
        )
        self.degree = min(
            int(degree), self.control_point_count - 1
        )
        self.method_name = "waypoint-constrained smoothing"
        self.waypoint_indices = tuple(
            int(index) for index in waypoint_index_array
        )
        self.waypoint_parameters = self.parameters[
            waypoint_index_array
        ].copy()
        self.knots = _open_uniform_clamped_knots(
            self.control_point_count, self.degree
        )
        if guide_sample_weights is None:
            sample_weights = np.ones(
                len(self.states), dtype=np.float64
            )
        else:
            sample_weights = np.asarray(
                guide_sample_weights, dtype=np.float64
            )
            if (
                sample_weights.shape != (len(self.states),)
                or not np.all(np.isfinite(sample_weights))
                or np.any(sample_weights <= 0.0)
            ):
                raise ValueError(
                    "guide_sample_weights must be positive with shape (N,)"
                )
            sample_weights = sample_weights / float(
                np.mean(sample_weights)
            )

        guide_basis = _basis_matrix(
            self.parameters,
            self.degree,
            self.knots,
            self.control_point_count,
        )
        hard_basis = guide_basis[waypoint_index_array]
        smooth_parameters = np.linspace(
            0.0, 1.0, max(201, 4 * self.control_point_count + 1)
        )
        second_derivative_basis = _basis_second_derivative_matrix(
            smooth_parameters,
            self.degree,
            self.knots,
            self.control_point_count,
        )
        third_derivative_basis = _basis_third_derivative_matrix(
            smooth_parameters,
            self.degree,
            self.knots,
            self.control_point_count,
        )

        self.position_control_points = (
            _solve_equality_constrained_smoothing(
                guide_basis,
                self.states[:, :3],
                hard_basis,
                self.states[waypoint_index_array, :3],
                second_derivative_basis,
                third_derivative_basis,
                guide_weight=guide_weight,
                guide_sample_weights=sample_weights,
                acceleration_weight=position_acceleration_weight,
                jerk_weight=position_jerk_weight,
                regularization=regularization,
            )
        )
        self.quaternion_control_points = (
            _solve_equality_constrained_smoothing(
                guide_basis,
                self.states[:, 3:7],
                hard_basis,
                self.states[waypoint_index_array, 3:7],
                second_derivative_basis,
                third_derivative_basis,
                guide_weight=guide_weight,
                guide_sample_weights=sample_weights,
                acceleration_weight=orientation_acceleration_weight,
                jerk_weight=orientation_jerk_weight,
                regularization=regularization,
            )
        )

        hard_states = self.evaluate(self.waypoint_parameters)
        hard_position_error = np.linalg.norm(
            hard_states[:, :3]
            - self.states[waypoint_index_array, :3],
            axis=1,
        )
        hard_attitude_error = np.linalg.norm(
            _relative_rotation_vectors(
                hard_states[:, 3:7],
                self.states[waypoint_index_array, 3:7],
            ),
            axis=1,
        )
        if (
            float(np.max(hard_position_error)) > 1.0e-8
            or float(np.max(hard_attitude_error)) > 1.0e-8
        ):
            raise RuntimeError(
                "constrained smoother did not preserve hard waypoints"
            )

        guide_fit = self.evaluate(self.parameters)
        self.guide_position_rms_m = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            guide_fit[:, :3] - self.states[:, :3]
                        ),
                        axis=1,
                    )
                )
            )
        )
        guide_attitude_error = _relative_rotation_vectors(
            guide_fit[:, 3:7], self.states[:, 3:7]
        )
        self.guide_attitude_rms_rad = float(
            np.sqrt(
                np.mean(
                    np.sum(np.square(guide_attitude_error), axis=1)
                )
            )
        )


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
    spline_method: str
    control_point_count: int
    guide_position_rms_m: float
    guide_attitude_rms_rad: float
    maximum_curvature_per_m: float
    orientation_shortcut_applied: bool

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
        segment_position_bounds: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
        solve_time_per_segment: float = 1.5,
        interpolation_resolution: float = 0.07,
        minimum_states_per_segment: int = 50,
        knot_stride: int = 3,
        spline_samples: int = 1000,
        orientation_metric_weight: float = 0.35,
        spline_method: str = "constrained-smoothing",
        smoothing_degree: int = 5,
        smoothing_guide_weight: float = 1.0,
        smoothing_position_acceleration_weight: float = 1.0e-8,
        smoothing_position_jerk_weight: float = 1.0e-12,
        smoothing_orientation_acceleration_weight: float = 2.5e-9,
        smoothing_orientation_jerk_weight: float = 2.5e-13,
        smoothing_clearance_weight_scale: float = 0.30,
        smoothing_max_attempts: int = 4,
        shortest_orientation_guide: bool = False,
    ) -> MultiWaypointPlan:
        waypoint_tuple = tuple(waypoints)
        if len(waypoint_tuple) < 2:
            raise ValueError("at least start and goal poses are required")
        if knot_stride < 1 or spline_samples < 20:
            raise ValueError(
                "knot_stride must be positive and spline_samples >= 20"
            )
        if spline_method not in (
            "constrained-smoothing",
            "interpolating",
        ):
            raise ValueError("unsupported spline_method")
        if (
            smoothing_degree < 3
            or smoothing_clearance_weight_scale < 0.0
            or smoothing_max_attempts < 1
        ):
            raise ValueError("invalid constrained smoothing configuration")

        segment_count = len(waypoint_tuple) - 1
        if segment_position_bounds is None:
            segment_planners = (self.planner,) * segment_count
            validated_segment_bounds = None
        else:
            if len(segment_position_bounds) != segment_count:
                raise ValueError(
                    "segment_position_bounds must contain one bound pair per "
                    "consecutive waypoint segment"
                )
            planners = []
            normalized_bounds = []
            for index, (lower, upper) in enumerate(segment_position_bounds):
                bounds_min = np.maximum(
                    self.planner.bounds_min,
                    np.asarray(lower, dtype=np.float64),
                )
                bounds_max = np.minimum(
                    self.planner.bounds_max,
                    np.asarray(upper, dtype=np.float64),
                )
                if (
                    bounds_min.shape != (3,)
                    or bounds_max.shape != (3,)
                    or np.any(bounds_min >= bounds_max)
                ):
                    raise ValueError(f"invalid bounds for segment {index}")
                for pose_name, pose in (
                    ("start", waypoint_tuple[index]),
                    ("goal", waypoint_tuple[index + 1]),
                ):
                    if np.any(pose.position < bounds_min) or np.any(
                        pose.position > bounds_max
                    ):
                        raise ValueError(
                            f"segment {index} {pose_name} pose lies outside "
                            "its position bounds"
                        )
                planners.append(OMPLSE3Planner(
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                    obstacles=self.planner.obstacles,
                    vehicle_radius=self.planner.vehicle_radius,
                    safety_margin=self.planner.safety_margin,
                    validity_resolution=self.planner.validity_resolution,
                    planner_range=self.planner.planner_range,
                    seed=self.planner.seed + index,
                    collision_checker=self.planner.collision_checker,
                ))
                normalized_bounds.append((bounds_min, bounds_max))
            segment_planners = tuple(planners)
            validated_segment_bounds = tuple(normalized_bounds)

        segments = tuple(
            segment_planner.plan(
                start,
                goal,
                solve_time=solve_time_per_segment,
                interpolation_resolution=interpolation_resolution,
                minimum_waypoints=minimum_states_per_segment,
                simplify=True,
            )
            for segment_planner, start, goal in zip(
                segment_planners, waypoint_tuple[:-1], waypoint_tuple[1:]
            )
        )
        raw_states, waypoint_indices = _concatenate_segments(segments)
        orientation_shortcut_applied = False
        if shortest_orientation_guide:
            shortcut_states = _shortest_orientation_guide(
                raw_states, waypoint_indices, waypoint_tuple
            )
            shortcut_clearance = self.planner.clearance(
                shortcut_states[:, :3], shortcut_states[:, 3:7]
            )
            if np.all(shortcut_clearance > 0.0):
                raw_states = shortcut_states
                orientation_shortcut_applied = True

        if spline_method == "constrained-smoothing":
            raw_parameters = _se3_chord_parameters(
                raw_states, orientation_metric_weight
            )
            raw_clearance = self.planner.clearance(
                raw_states[:, :3], raw_states[:, 3:7]
            )
            clearance_weight = 1.0 + np.square(
                smoothing_clearance_weight_scale
                / (np.maximum(raw_clearance, 0.0) + 0.01)
            )
            clearance_weight = np.minimum(clearance_weight, 400.0)
            last_error: RuntimeError | None = None
            for attempt in range(smoothing_max_attempts):
                stride = max(1, knot_stride - attempt)
                try:
                    spline = WaypointConstrainedSmoothingSE3BSpline(
                        raw_states,
                        waypoint_indices,
                        parameters=raw_parameters,
                        degree=smoothing_degree,
                        control_point_stride=stride,
                        orientation_metric_weight=(
                            orientation_metric_weight
                        ),
                        guide_weight=(
                            smoothing_guide_weight * 5.0**attempt
                        ),
                        guide_sample_weights=clearance_weight,
                        position_acceleration_weight=(
                            smoothing_position_acceleration_weight
                        ),
                        position_jerk_weight=(
                            smoothing_position_jerk_weight
                        ),
                        orientation_acceleration_weight=(
                            smoothing_orientation_acceleration_weight
                        ),
                        orientation_jerk_weight=(
                            smoothing_orientation_jerk_weight
                        ),
                    )
                    return self._build_validated_plan(
                        waypoint_tuple,
                        segments,
                        raw_states,
                        waypoint_indices,
                        spline,
                        spline.waypoint_parameters,
                        spline_samples,
                        stride,
                        orientation_shortcut_applied,
                        validated_segment_bounds,
                    )
                except RuntimeError as error:
                    last_error = error
            raise RuntimeError(
                "waypoint-constrained smoothing failed collision or "
                f"workspace validation after {smoothing_max_attempts} "
                f"attempts: {last_error}"
            )

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
                    orientation_shortcut_applied,
                    validated_segment_bounds,
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
        orientation_shortcut_applied: bool,
        segment_position_bounds: tuple[tuple[FloatArray, FloatArray], ...] | None,
    ) -> MultiWaypointPlan:
        parameters = np.linspace(0.0, 1.0, spline_samples)
        spline_states = spline.evaluate(parameters)
        clearance = self.planner.clearance(
            spline_states[:, :3], spline_states[:, 3:7]
        )
        in_bounds = np.all(
            (spline_states[:, :3] >= self.planner.bounds_min)
            & (spline_states[:, :3] <= self.planner.bounds_max),
            axis=1,
        )
        if not np.all(in_bounds):
            raise RuntimeError("B-spline left the planning workspace")
        if segment_position_bounds is not None:
            for index, (lower, upper) in enumerate(segment_position_bounds):
                parameter_lower = float(waypoint_parameters[index])
                parameter_upper = float(waypoint_parameters[index + 1])
                mask = (
                    (parameters >= parameter_lower)
                    & (parameters <= parameter_upper)
                )
                segment_positions = spline_states[mask, :3]
                if len(segment_positions) == 0:
                    midpoint = 0.5 * (parameter_lower + parameter_upper)
                    segment_positions = spline.evaluate(
                        np.asarray([midpoint])
                    )[:, :3]
                if np.any(segment_positions < lower - 1.0e-9) or np.any(
                    segment_positions > upper + 1.0e-9
                ):
                    raise RuntimeError(
                        f"B-spline left the position bounds for segment {index}"
                    )
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
                "SE(3) B-spline did not pass hard waypoint poses"
            )

        translation_delta = np.diff(spline_states[:, :3], axis=0)
        rotation_delta = _relative_rotation_vectors(
            spline_states[:-1, 3:7], spline_states[1:, 3:7]
        )
        (
            _,
            position_path_derivative,
            position_path_second_derivative,
            _,
            _,
        ) = spline.evaluate_with_second_derivatives(parameters)
        derivative_norm = np.linalg.norm(
            position_path_derivative, axis=1
        )
        curvature = np.divide(
            np.linalg.norm(
                np.cross(
                    position_path_derivative,
                    position_path_second_derivative,
                ),
                axis=1,
            ),
            derivative_norm**3,
            out=np.zeros_like(derivative_norm),
            where=derivative_norm > 1.0e-9,
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
                f"OMPL RRTConnect x{len(segments)} + global degree-"
                f"{spline.degree} {spline.method_name} SE(3) B-spline"
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
            spline_method=spline.method_name,
            control_point_count=spline.control_point_count,
            guide_position_rms_m=float(
                getattr(spline, "guide_position_rms_m", np.nan)
            ),
            guide_attitude_rms_rad=float(
                getattr(spline, "guide_attitude_rms_rad", np.nan)
            ),
            maximum_curvature_per_m=float(np.max(curvature)),
            orientation_shortcut_applied=orientation_shortcut_applied,
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


def _shortest_orientation_guide(
    raw_states: FloatArray,
    waypoint_indices: Sequence[int],
    waypoints: Sequence[SE3Pose],
) -> FloatArray:
    """Replace SO(3) wandering with segment-wise shortest-path SLERP."""

    guide = np.asarray(raw_states, dtype=np.float64).copy()
    for segment_index, (begin, end) in enumerate(
        zip(waypoint_indices[:-1], waypoint_indices[1:])
    ):
        positions = guide[begin : end + 1, :3]
        cumulative = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.linalg.norm(np.diff(positions, axis=0), axis=1)
                ),
            )
        )
        if cumulative[-1] <= 1.0e-10:
            fractions = np.linspace(0.0, 1.0, len(positions))
        else:
            fractions = cumulative / cumulative[-1]
        guide[begin : end + 1, 3:7] = _shortest_quaternion_slerp(
            waypoints[segment_index].quaternion,
            waypoints[segment_index + 1].quaternion,
            fractions,
        )
    _make_quaternions_continuous(guide[:, 3:7])
    return guide


def _shortest_quaternion_slerp(
    start: ArrayLike,
    end: ArrayLike,
    fractions: ArrayLike,
) -> FloatArray:
    start_quaternion = normalize_quaternion(start)
    end_quaternion = normalize_quaternion(end)
    values = np.asarray(fractions, dtype=np.float64)
    dot = float(np.dot(start_quaternion, end_quaternion))
    if dot < 0.0:
        end_quaternion = -end_quaternion
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        quaternions = (
            (1.0 - values[:, None]) * start_quaternion
            + values[:, None] * end_quaternion
        )
        return normalize_quaternion(quaternions)
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    start_weight = np.sin((1.0 - values) * angle) / denominator
    end_weight = np.sin(values * angle) / denominator
    return normalize_quaternion(
        start_weight[:, None] * start_quaternion
        + end_weight[:, None] * end_quaternion
    )


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


def _open_uniform_clamped_knots(
    control_count: int, degree: int
) -> FloatArray:
    if control_count <= degree:
        raise ValueError("control_count must exceed degree")
    knots = np.zeros(control_count + degree + 1, dtype=np.float64)
    knots[-(degree + 1) :] = 1.0
    interior_count = control_count - degree - 1
    if interior_count > 0:
        knots[degree + 1 : -(degree + 1)] = np.linspace(
            0.0, 1.0, interior_count + 2
        )[1:-1]
    return knots


def _solve_equality_constrained_smoothing(
    guide_basis: FloatArray,
    guide_values: FloatArray,
    hard_basis: FloatArray,
    hard_values: FloatArray,
    second_derivative_basis: FloatArray,
    third_derivative_basis: FloatArray,
    *,
    guide_weight: float,
    guide_sample_weights: FloatArray,
    acceleration_weight: float,
    jerk_weight: float,
    regularization: float,
) -> FloatArray:
    control_count = guide_basis.shape[1]
    output_dimension = guide_values.shape[1]
    guide_scale = np.sqrt(
        guide_weight * guide_sample_weights / len(guide_basis)
    )
    rows = [guide_scale[:, None] * guide_basis]
    targets = [guide_scale[:, None] * guide_values]
    if acceleration_weight > 0.0:
        scale = np.sqrt(
            acceleration_weight / len(second_derivative_basis)
        )
        rows.append(scale * second_derivative_basis)
        targets.append(
            np.zeros(
                (len(second_derivative_basis), output_dimension),
                dtype=np.float64,
            )
        )
    if jerk_weight > 0.0:
        scale = np.sqrt(jerk_weight / len(third_derivative_basis))
        rows.append(scale * third_derivative_basis)
        targets.append(
            np.zeros(
                (len(third_derivative_basis), output_dimension),
                dtype=np.float64,
            )
        )
    rows.append(np.sqrt(regularization) * np.eye(control_count))
    targets.append(
        np.zeros((control_count, output_dimension), dtype=np.float64)
    )
    design = np.vstack(rows)
    target = np.vstack(targets)
    hessian = design.T @ design
    gradient = design.T @ target
    constraint_count = len(hard_basis)
    kkt = np.block(
        [
            [
                hessian,
                hard_basis.T,
            ],
            [
                hard_basis,
                np.zeros(
                    (constraint_count, constraint_count),
                    dtype=np.float64,
                ),
            ],
        ]
    )
    right_hand_side = np.vstack((gradient, hard_values))
    try:
        solution = np.linalg.solve(kkt, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "waypoint-constrained B-spline solve was singular"
        ) from error
    control_points = solution[:control_count]
    equality_error = hard_basis @ control_points - hard_values
    if float(np.max(np.abs(equality_error))) > 1.0e-8:
        raise RuntimeError(
            "waypoint-constrained B-spline equality solve was inaccurate"
        )
    return control_points


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


def _basis_second_derivative_matrix(
    parameters: FloatArray,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    values = np.asarray(parameters, dtype=np.float64)
    if degree < 2:
        return np.zeros((len(values), control_count), dtype=np.float64)

    lower_values = np.where(
        values >= 1.0 - 1.0e-14, 1.0 - 1.0e-12, values
    )
    lower_basis = _basis_matrix(
        lower_values,
        degree - 2,
        knots,
        control_count + 2,
    )
    lower_count = control_count + 1
    lower_left_denominator = (
        knots[degree - 1 : degree - 1 + lower_count]
        - knots[:lower_count]
    )
    lower_right_denominator = (
        knots[degree : degree + lower_count]
        - knots[1 : 1 + lower_count]
    )
    lower_derivative = np.divide(
        (degree - 1) * lower_basis[:, :lower_count],
        lower_left_denominator[None, :],
        out=np.zeros((len(values), lower_count)),
        where=lower_left_denominator[None, :] > 0.0,
    )
    lower_derivative -= np.divide(
        (degree - 1) * lower_basis[:, 1 : lower_count + 1],
        lower_right_denominator[None, :],
        out=np.zeros((len(values), lower_count)),
        where=lower_right_denominator[None, :] > 0.0,
    )

    left_denominator = (
        knots[degree : degree + control_count]
        - knots[:control_count]
    )
    right_denominator = (
        knots[degree + 1 : degree + 1 + control_count]
        - knots[1 : 1 + control_count]
    )
    second_derivative = np.divide(
        degree * lower_derivative[:, :control_count],
        left_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=left_denominator[None, :] > 0.0,
    )
    second_derivative -= np.divide(
        degree * lower_derivative[:, 1 : control_count + 1],
        right_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=right_denominator[None, :] > 0.0,
    )
    return second_derivative


def _basis_third_derivative_matrix(
    parameters: FloatArray,
    degree: int,
    knots: FloatArray,
    control_count: int,
) -> FloatArray:
    values = np.asarray(parameters, dtype=np.float64)
    if degree < 3:
        return np.zeros((len(values), control_count), dtype=np.float64)
    lower_values = np.where(
        values >= 1.0 - 1.0e-14, 1.0 - 1.0e-12, values
    )
    lower_second_derivative = _basis_second_derivative_matrix(
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
    third_derivative = np.divide(
        degree * lower_second_derivative[:, :control_count],
        left_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=left_denominator[None, :] > 0.0,
    )
    third_derivative -= np.divide(
        degree * lower_second_derivative[:, 1 : control_count + 1],
        right_denominator[None, :],
        out=np.zeros((len(values), control_count)),
        where=right_denominator[None, :] > 0.0,
    )
    return third_derivative


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
