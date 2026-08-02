"""Tests for scalar TOPP-RA retiming of the interpolating SE(3) path."""

from __future__ import annotations

import unittest

import numpy as np

from mppi.quaternion import quaternion_from_euler
from multi_waypoint_planner import (
    InterpolatingSE3BSpline,
    WaypointConstrainedSmoothingSE3BSpline,
)
from toppra_retiming import (
    SE3PathKinematics,
    ToppraTimedReference,
)


class ToppraRetimingTest(unittest.TestCase):
    def setUp(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.4, 0.3, 1.2],
                [0.9, -0.1, 1.4],
                [1.4, 0.2, 1.1],
                [1.8, 0.0, 1.3],
            ]
        )
        quaternions = quaternion_from_euler(
            np.radians(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, -5.0, 20.0],
                    [-15.0, 20.0, 50.0],
                    [20.0, 5.0, 80.0],
                    [0.0, -10.0, 110.0],
                ]
            )
        )
        self.spline = InterpolatingSE3BSpline(
            np.concatenate((positions, quaternions), axis=1)
        )

    def test_analytic_second_derivatives_match_finite_difference(
        self,
    ) -> None:
        parameters = np.asarray([0.07, 0.18, 0.39, 0.61, 0.82, 0.93])
        (
            _,
            _,
            position_second_derivative,
            _,
            quaternion_second_derivative,
        ) = self.spline.evaluate_with_second_derivatives(parameters)
        epsilon = 1.0e-5
        _, position_plus, quaternion_plus = (
            self.spline.evaluate_with_derivatives(parameters + epsilon)
        )
        _, position_minus, quaternion_minus = (
            self.spline.evaluate_with_derivatives(parameters - epsilon)
        )
        np.testing.assert_allclose(
            position_second_derivative,
            (position_plus - position_minus) / (2.0 * epsilon),
            atol=2.0e-6,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            quaternion_second_derivative,
            (quaternion_plus - quaternion_minus) / (2.0 * epsilon),
            atol=2.0e-5,
            rtol=2.0e-5,
        )

        kinematics = SE3PathKinematics(self.spline)
        analytic_omega_derivative = (
            kinematics.body_angular_path_rate_derivative(parameters)
        )
        numeric_omega_derivative = (
            kinematics.body_angular_path_rate(parameters + epsilon)
            - kinematics.body_angular_path_rate(parameters - epsilon)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            analytic_omega_derivative,
            numeric_omega_derivative,
            atol=3.0e-5,
            rtol=3.0e-5,
        )

    def test_rest_to_rest_trajectory_obeys_dense_kinematic_limits(
        self,
    ) -> None:
        max_linear_speed = 0.9
        max_angular_speed = 1.2
        max_linear_acceleration = np.asarray([2.0, 2.2, 1.8])
        max_angular_acceleration = np.asarray([3.0, 3.5, 2.8])
        smoothed_spline = WaypointConstrainedSmoothingSE3BSpline(
            self.spline.states,
            (0, 2, 4),
            control_point_stride=2,
        )
        reference = ToppraTimedReference(
            smoothed_spline,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_linear_acceleration=max_linear_acceleration,
            max_angular_acceleration=max_angular_acceleration,
            start_delay=0.2,
            duration_scale=1.03,
            gridpoint_count=101,
            validation_point_count=2001,
            max_refinement_iterations=3,
            velocity_scale=0.90,
            acceleration_scale=0.85,
        )
        self.assertTrue(reference.validation.valid)

        times = np.linspace(
            0.0, reference.finish_time + 0.2, 5001
        )
        trajectory = reference.sample_full(times)
        self.assertTrue(
            np.all(np.diff(trajectory.path_position) >= -1.0e-10)
        )
        self.assertGreater(reference.duration, 0.0)
        np.testing.assert_allclose(
            trajectory.path_position[[0, -1]], [0.0, 1.0], atol=1.0e-12
        )
        np.testing.assert_allclose(
            trajectory.linear_velocity_world[[0, -1]],
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            trajectory.angular_velocity_body[[0, -1]],
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            trajectory.linear_acceleration_world[[0, -1]],
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            trajectory.angular_acceleration_body[[0, -1]],
            0.0,
            atol=1.0e-12,
        )

        self.assertLessEqual(
            float(
                np.max(
                    np.linalg.norm(
                        trajectory.linear_velocity_world, axis=1
                    )
                )
            ),
            max_linear_speed + 1.0e-3,
        )
        self.assertLessEqual(
            float(
                np.max(
                    np.linalg.norm(
                        trajectory.angular_velocity_body, axis=1
                    )
                )
            ),
            max_angular_speed + 1.0e-3,
        )
        self.assertTrue(
            np.all(
                np.max(
                    np.abs(trajectory.linear_acceleration_world), axis=0
                )
                <= max_linear_acceleration + 2.0e-3
            )
        )
        self.assertTrue(
            np.all(
                np.max(
                    np.abs(trajectory.angular_acceleration_body), axis=0
                )
                <= max_angular_acceleration + 2.0e-3
            )
        )

        geometric_pose = smoothed_spline.evaluate(
            trajectory.path_position
        )
        np.testing.assert_allclose(
            trajectory.position, geometric_pose[:, :3], atol=1.0e-12
        )
        np.testing.assert_allclose(
            trajectory.quaternion_wxyz,
            geometric_pose[:, 3:7],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            reference.sample(times), trajectory.reference, atol=1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
