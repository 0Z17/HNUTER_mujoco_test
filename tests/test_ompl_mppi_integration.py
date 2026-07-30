"""Tests for OMPL SE(3) planning and its MPPI reference bridge."""

from __future__ import annotations

import math
import unittest

import numpy as np

from mppi import PoseTrackingCost
from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
)
from ompl_se3_planner import (
    OMPLSE3Planner,
    SE3PathReference,
    SE3Pose,
    SphereObstacle,
)


class OMPLMPPIIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obstacle = SphereObstacle(np.array([0.0, 0.0, 1.0]), 0.4)
        self.planner = OMPLSE3Planner(
            bounds_min=(-2.0, -2.0, 0.2),
            bounds_max=(2.0, 2.0, 2.5),
            obstacles=(self.obstacle,),
            vehicle_radius=0.2,
            safety_margin=0.05,
            validity_resolution=0.005,
            planner_range=0.4,
            seed=11,
        )
        self.start = SE3Pose(
            np.array([-1.2, -0.7, 1.0]),
            quaternion_from_euler((0.0, 0.0, -0.2)),
        )
        self.goal = SE3Pose(
            np.array([1.2, 0.7, 1.3]),
            quaternion_from_euler(
                (math.radians(20.0), -0.1, math.radians(100.0))
            ),
        )

    def test_birrt_path_is_collision_free_and_reaches_pose(self) -> None:
        path = self.planner.plan(
            self.start,
            self.goal,
            solve_time=1.0,
            interpolation_resolution=0.06,
            minimum_waypoints=80,
        )

        np.testing.assert_allclose(
            path.states[0, :3], self.start.position, atol=1.0e-9
        )
        np.testing.assert_allclose(
            path.states[-1, :3], self.goal.position, atol=1.0e-9
        )
        self.assertLess(
            np.linalg.norm(
                quaternion_error_vector(
                    path.states[-1, 3:7], self.goal.quaternion
                )
            ),
            1.0e-8,
        )
        self.assertGreater(
            float(np.min(self.planner.clearance(path.states[:, :3]))),
            -1.0e-8,
        )
        self.assertIn("RRTConnect", path.planner_name)

    def test_time_parameterization_obeys_limits_and_holds_endpoints(
        self,
    ) -> None:
        path = self.planner.plan(
            self.start,
            self.goal,
            solve_time=1.0,
            minimum_waypoints=80,
        )
        reference = SE3PathReference(
            path,
            max_linear_speed=0.7,
            max_angular_speed=0.8,
            start_delay=0.3,
        )
        times = np.linspace(
            0.0, reference.finish_time + 0.5, 3000
        )
        states = reference.sample(times)

        np.testing.assert_allclose(
            states[0, :3], self.start.position, atol=1.0e-10
        )
        np.testing.assert_allclose(
            states[-1, :3], self.goal.position, atol=1.0e-10
        )
        np.testing.assert_allclose(states[[0, -1], 3:6], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            states[[0, -1], 10:13], 0.0, atol=1.0e-12
        )
        self.assertLessEqual(
            float(np.max(np.linalg.norm(states[:, 3:6], axis=1))),
            0.7 + 2.0e-4,
        )
        self.assertLessEqual(
            float(np.max(np.linalg.norm(states[:, 10:13], axis=1))),
            0.8 + 2.0e-4,
        )
        np.testing.assert_allclose(
            np.linalg.norm(states[:, 6:10], axis=1),
            1.0,
            atol=1.0e-12,
        )

    def test_pose_mppi_cost_penalizes_obstacle_collision(self) -> None:
        cost = PoseTrackingCost(
            minimum_altitude=-10.0,
            workspace_radius=None,
            spherical_obstacles=((0.0, 0.0, 1.0, 0.4),),
            collision_radius=0.25,
            obstacle_penalty=1.0e4,
        )
        states = np.zeros((2, 3, 13))
        states[:, :, 2] = 1.0
        states[:, :, 6] = 1.0
        states[0, :, 0] = 1.5
        states[1, :, 0] = 0.0
        reference = states[0].copy()
        controls = np.zeros((2, 2, 6))
        costs = cost.trajectory_cost(states, controls, reference)
        self.assertGreater(float(costs[1]), float(costs[0]) + 1000.0)


if __name__ == "__main__":
    unittest.main()
