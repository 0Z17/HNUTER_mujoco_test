"""Tests for multi-waypoint OMPL planning and global B-spline timing."""

from __future__ import annotations

import unittest

import numpy as np

from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
)
from multi_waypoint_planner import (
    BSplineTimeParameterizedReference,
    InterpolatingSE3BSpline,
    MultiWaypointOMPLPlanner,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    SE3Pose,
    SphereObstacle,
)


class MultiWaypointPlannerTest(unittest.TestCase):
    def test_cubic_spline_interpolates_all_pose_nodes(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.4, 0.6, 1.2],
                [1.0, 0.2, 1.5],
                [1.4, -0.3, 1.1],
                [2.0, 0.0, 1.4],
            ]
        )
        quaternions = quaternion_from_euler(
            np.radians(
                [
                    [0.0, 0.0, 0.0],
                    [20.0, -10.0, 35.0],
                    [-25.0, 30.0, 100.0],
                    [35.0, 15.0, 160.0],
                    [0.0, -20.0, -120.0],
                ]
            )
        )
        spline = InterpolatingSE3BSpline(
            np.concatenate((positions, quaternions), axis=1)
        )
        states = spline.evaluate(spline.parameters)
        np.testing.assert_allclose(
            states[:, :3], positions, atol=1.0e-10
        )
        attitude_error = np.linalg.norm(
            quaternion_error_vector(
                states[:, 3:7], quaternions
            ),
            axis=1,
        )
        self.assertLess(float(np.max(attitude_error)), 1.0e-9)
        np.testing.assert_allclose(
            np.linalg.norm(states[:, 3:7], axis=1),
            1.0,
            atol=1.0e-12,
        )

    def test_multi_segment_plan_and_timing(self) -> None:
        poses = (
            ((-1.5, -0.7, 1.0), (0.0, 0.0, -20.0)),
            ((-0.8, 0.8, 1.4), (25.0, -15.0, 30.0)),
            ((0.0, 1.2, 1.0), (-20.0, 25.0, 90.0)),
            ((0.8, -0.6, 1.6), (35.0, 10.0, 150.0)),
            ((1.5, 0.7, 1.3), (10.0, -20.0, -130.0)),
        )
        waypoints = tuple(
            SE3Pose(
                np.asarray(position),
                quaternion_from_euler(np.radians(rpy)),
            )
            for position, rpy in poses
        )
        planner = OMPLSE3Planner(
            bounds_min=(-2.0, -1.8, 0.2),
            bounds_max=(2.0, 1.8, 2.2),
            obstacles=(
                SphereObstacle(np.array([0.0, 0.0, 1.2]), 0.22),
            ),
            vehicle_radius=0.15,
            safety_margin=0.08,
            planner_range=0.35,
            seed=19,
        )
        plan = MultiWaypointOMPLPlanner(planner).plan(
            waypoints,
            solve_time_per_segment=0.5,
            minimum_states_per_segment=30,
            knot_stride=3,
            spline_samples=600,
        )
        self.assertEqual(len(plan.segment_paths), 4)
        self.assertGreater(plan.minimum_clearance_m, 0.0)

        waypoint_states = plan.spline.evaluate(
            plan.waypoint_parameters
        )
        np.testing.assert_allclose(
            waypoint_states[:, :3],
            np.asarray([pose.position for pose in waypoints]),
            atol=1.0e-7,
        )

        reference = BSplineTimeParameterizedReference(
            plan,
            max_linear_speed=0.9,
            max_angular_speed=1.3,
            duration_scale=1.08,
            timing_samples=2000,
        )
        times = np.linspace(
            0.0, reference.finish_time + 0.3, 6000
        )
        states = reference.sample(times)
        self.assertLessEqual(
            float(np.max(np.linalg.norm(states[:, 3:6], axis=1))),
            0.9 + 0.015,
        )
        self.assertLessEqual(
            float(
                np.max(np.linalg.norm(states[:, 10:13], axis=1))
            ),
            1.3 + 0.02,
        )
        np.testing.assert_allclose(
            states[[0, -1], 3:6], 0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            states[[0, -1], 10:13], 0.0, atol=1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
