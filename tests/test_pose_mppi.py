"""Tests for quaternion 6-DoF MPPI dynamics and costs."""

from __future__ import annotations

import math
import unittest

import numpy as np

from mppi import (
    FullyActuatedUAVDynamics,
    MPPIConfig,
    MPPIController,
    PoseTrackingCost,
)
from mppi.quaternion import (
    quaternion_error_vector,
    quaternion_from_euler,
)


class PoseMPPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.dynamics = FullyActuatedUAVDynamics(dt=0.05)
        self.cost = PoseTrackingCost(
            minimum_altitude=-10.0,
            workspace_radius=None,
        )

    def test_rollout_preserves_unit_quaternion(self) -> None:
        state = np.zeros(13)
        state[2] = 1.0
        state[6] = 1.0
        controls = np.zeros((8, 20, 6))
        controls[:, :, 3:] = np.array([0.4, -0.2, 0.3])
        states = self.dynamics.rollout(state, controls)
        quaternion_norms = np.linalg.norm(states[:, :, 6:10], axis=2)
        np.testing.assert_allclose(quaternion_norms, 1.0, atol=1.0e-12)

    def test_attitude_cost_is_quaternion_sign_invariant(self) -> None:
        state = np.zeros((2, 2, 13))
        state[:, :, 2] = 1.0
        state[:, :, 6] = 1.0
        reference = state[0].copy()
        state[1, :, 6:10] *= -1.0
        controls = np.zeros((2, 1, 6))
        costs = self.cost.trajectory_cost(state, controls, reference)
        self.assertAlmostEqual(float(costs[0]), float(costs[1]), places=12)

    def test_pose_controller_shapes_and_attitude_direction(self) -> None:
        config = MPPIConfig(
            horizon=24,
            num_samples=512,
            temperature=40.0,
            noise_sigma=(1.5, 1.5, 1.5, 2.0, 2.0, 2.0),
            control_min=(-3.0, -3.0, -3.0, -5.0, -5.0, -5.0),
            control_max=(3.0, 3.0, 3.0, 5.0, 5.0, 5.0),
            action_continuity_weight=0.5,
            control_smoothing=0.1,
            seed=5,
        )
        controller = MPPIController(
            self.dynamics, self.cost, config
        )
        state = np.zeros(13)
        state[2] = 1.0
        state[6] = 1.0
        reference = np.repeat(
            state[None, :], config.horizon + 1, axis=0
        )
        reference[:, 6:10] = quaternion_from_euler(
            np.array([0.0, 0.0, math.radians(30.0)])
        )
        initial_error = np.linalg.norm(
            quaternion_error_vector(state[6:10], reference[0, 6:10])
        )

        for _ in range(24):
            result = controller.command(state, reference)
            state = self.dynamics.rollout(
                state, result.action.reshape(1, 1, 6)
            )[0, 1]

        final_error = np.linalg.norm(
            quaternion_error_vector(state[6:10], reference[0, 6:10])
        )
        self.assertEqual(result.action.shape, (6,))
        self.assertEqual(result.nominal_states.shape, (25, 13))
        self.assertLess(final_error, 0.6 * initial_error)


if __name__ == "__main__":
    unittest.main()
