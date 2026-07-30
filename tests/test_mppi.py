"""Unit tests for the simulator-independent MPPI module."""

from __future__ import annotations

import unittest

import numpy as np

from mppi import (
    MPPIConfig,
    MPPIController,
    PointMassDynamics,
    QuadraticTrackingCost,
)


class MPPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.dynamics = PointMassDynamics(dt=0.05)
        self.config = MPPIConfig(
            horizon=30,
            num_samples=384,
            temperature=8.0,
            noise_sigma=(2.0, 2.0, 2.0),
            control_min=(-3.0, -3.0, -3.0),
            control_max=(3.0, 3.0, 3.0),
            seed=11,
        )
        self.cost = QuadraticTrackingCost(
            minimum_altitude=-10.0,
            workspace_radius=None,
        )

    def test_command_shapes_weights_and_limits(self) -> None:
        controller = MPPIController(
            self.dynamics, self.cost, self.config
        )
        state = np.zeros(6)
        reference = np.zeros((self.config.horizon + 1, 6))
        reference[:, 0] = 1.0
        result = controller.command(state, reference)

        self.assertEqual(result.action.shape, (3,))
        self.assertEqual(
            result.nominal_states.shape, (self.config.horizon + 1, 6)
        )
        self.assertEqual(
            result.sampled_states.shape,
            (self.config.num_samples, self.config.horizon + 1, 6),
        )
        self.assertAlmostEqual(float(np.sum(result.weights)), 1.0)
        self.assertTrue(np.all(result.action <= 3.0))
        self.assertTrue(np.all(result.action >= -3.0))
        self.assertGreaterEqual(result.effective_sample_size, 1.0)
        self.assertLessEqual(
            result.effective_sample_size, self.config.num_samples + 1.0e-9
        )

    def test_closed_loop_moves_toward_goal(self) -> None:
        controller = MPPIController(
            self.dynamics, self.cost, self.config
        )
        state = np.zeros(6)
        reference = np.zeros((self.config.horizon + 1, 6))
        reference[:, 0] = 1.5
        initial_error = abs(reference[0, 0] - state[0])

        for _ in range(35):
            result = controller.command(state, reference)
            state = self.dynamics.rollout(
                state, result.action.reshape(1, 1, 3)
            )[0, 1]

        final_error = abs(reference[0, 0] - state[0])
        self.assertLess(final_error, 0.35 * initial_error)

    def test_rejects_bad_reference_shape(self) -> None:
        controller = MPPIController(
            self.dynamics, self.cost, self.config
        )
        with self.assertRaises(ValueError):
            controller.command(np.zeros(6), np.zeros((10, 6)))

    def test_rejects_invalid_smoothing(self) -> None:
        invalid_config = MPPIConfig(control_smoothing=1.0)
        with self.assertRaises(ValueError):
            MPPIController(self.dynamics, self.cost, invalid_config)


if __name__ == "__main__":
    unittest.main()
