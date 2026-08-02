"""Unit tests for the direct geometric-controller ablation path."""

from __future__ import annotations

import unittest

import numpy as np

from hnuter_ompl_mppi_demo import (
    geometric_reference_result,
    reference_feedforward_controls,
    reference_with_acceleration,
)


class _PolynomialReference:
    def sample(self, times: np.ndarray) -> np.ndarray:
        reference = np.zeros((len(times), 13), dtype=np.float64)
        reference[:, 3] = np.square(times)
        reference[:, 10] = 3.0 * times
        reference[:, 6] = 1.0
        return reference


class ControllerAblationTest(unittest.TestCase):
    def test_finite_difference_fallback_recovers_acceleration(
        self,
    ) -> None:
        times = np.arange(5.0)
        _, linear, angular = reference_with_acceleration(
            _PolynomialReference(), times
        )

        np.testing.assert_allclose(linear[:, 0], 2.0 * times)
        np.testing.assert_allclose(angular[:, 0], 3.0)
        np.testing.assert_allclose(linear[:, 1:], 0.0)
        np.testing.assert_allclose(angular[:, 1:], 0.0)

    def test_geometric_result_uses_timed_feedforward(self) -> None:
        reference = np.zeros((5, 13), dtype=np.float64)
        reference[:, 6] = 1.0
        linear = np.arange(15.0).reshape(5, 3)
        angular = -linear

        result = geometric_reference_result(
            reference,
            linear,
            angular,
            attitude_action_index=2,
        )

        np.testing.assert_array_equal(result.action[:3], linear[1])
        np.testing.assert_array_equal(result.action[3:], angular[2])
        np.testing.assert_array_equal(
            result.nominal_controls[:, :3], linear[1:]
        )
        np.testing.assert_array_equal(
            result.nominal_states, reference
        )
        self.assertTrue(np.isnan(result.effective_sample_size))

    def test_feedforward_respects_attitude_lookahead(self) -> None:
        linear = np.arange(15.0).reshape(5, 3)
        angular = -linear

        controls = reference_feedforward_controls(
            linear,
            angular,
            attitude_action_index=2,
        )

        np.testing.assert_array_equal(controls[:, :3], linear[1:])
        np.testing.assert_array_equal(
            controls[:, 3:],
            angular[[2, 3, 4, 4]],
        )


if __name__ == "__main__":
    unittest.main()
