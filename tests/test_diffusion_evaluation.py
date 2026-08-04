import unittest

import numpy as np

from evaluate_se3_diffusion import (
    dense_path,
    pair_cluster_confidence_interval,
    resample_se3_path,
)


class DiffusionEvaluationTest(unittest.TestCase):
    def test_dense_path_limits_translation_step(self) -> None:
        path = np.asarray([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.21, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ])
        dense = dense_path(path, translation_step=0.04, rotation_step_deg=3.0)
        self.assertLessEqual(
            float(np.max(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1))),
            0.04 + 1e-12,
        )

    def test_arc_resampling_preserves_pose_endpoints(self) -> None:
        path = np.asarray([
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 1.0, 0.9238795, 0.0, 0.0, 0.3826834],
            [1.0, 0.0, 1.0, 0.7071068, 0.0, 0.0, 0.7071068],
        ])
        result = resample_se3_path(path, 64)
        np.testing.assert_allclose(result[0], path[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], path[-1], atol=1e-6)

    def test_bootstrap_clusters_by_pair(self) -> None:
        records = [
            {"pair_index": 0, "success": value} for value in (0, 0, 0, 0)
        ] + [
            {"pair_index": 1, "success": value} for value in (1, 1, 1, 1)
        ]
        lower, upper = pair_cluster_confidence_interval(records, "success")
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)


if __name__ == "__main__":
    unittest.main()

