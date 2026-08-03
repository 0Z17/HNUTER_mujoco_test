import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_diffusion_dataset import (
    _diversity,
    _group_splits,
    _interpolate_rows,
    _normalize_sample,
    _sample_route_template,
    _topology_signature,
)


class DiffusionDatasetCollectorTest(unittest.TestCase):
    def test_quaternion_interpolation_is_normalized_and_continuous(self):
        values = np.zeros((2, 13))
        values[:, 6] = (1.0, -1.0)
        result = _interpolate_rows(
            np.asarray([0.0, 1.0]),
            values,
            np.linspace(0.0, 1.0, 7),
            slice(6, 10),
        )
        np.testing.assert_allclose(
            np.linalg.norm(result[:, 6:10], axis=1), 1.0
        )
        self.assertTrue(np.all(result[:, 6] > 0.0))

    def test_diversity_uses_nearest_existing_reference(self):
        base = {
            "planned": np.tile(np.asarray([[0, 0, 0, 1, 0, 0, 0.0]]), (8, 1)),
            "actual": np.tile(np.asarray([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0.0, 0, 0, 0]]), (8, 1)),
        }
        shifted = {key: value.copy() for key, value in base.items()}
        shifted["planned"][:, 0] += 0.05
        result = _diversity(shifted, [base], 0.025, 2.0)
        self.assertTrue(result["accepted_by_threshold"])
        self.assertAlmostEqual(result["planned_position_rms_m"], 0.05)

    def test_group_split_has_no_pair_leakage(self):
        splits = _group_splits(8, 123)
        groups = [set(values) for values in splits.values()]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(set.union(*groups), set(range(8)))

    def test_normalized_sample_has_fixed_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.npz"
            destination = Path(temporary) / "normalized.npz"
            time = np.linspace(0.0, 1.0, 11)
            state = np.zeros((11, 13))
            state[:, 6] = 1.0
            np.savez_compressed(
                source,
                start_pose=np.asarray([0, 0, 0, 1, 0, 0, 0]),
                goal_pose=np.asarray([1, 0, 0, 1, 0, 0, 0]),
                toppra_time=time,
                toppra_reference_state=state,
                toppra_linear_acceleration_world=np.zeros((11, 3)),
                toppra_angular_acceleration_body=np.zeros((11, 3)),
                toppra_physical_clearance=np.ones(11),
                control_time=time,
                actual_state=state,
                reference_state=state,
                mppi_action=np.zeros((11, 6)),
                actual_physical_clearance=np.ones(11),
            )
            result = _normalize_sample(source, destination, 2, 3, 4, 32)
            self.assertEqual(result["reference_state"].shape, (32, 13))
            self.assertEqual(result["actual_action"].shape, (32, 6))
            self.assertTrue(destination.exists())

    def test_route_template_sampling_stays_inside_bounds(self):
        template = {
            "waypoints": [{
                "position_bounds": {
                    "min": [-0.1, -1.4, 0.8],
                    "max": [0.1, -1.3, 0.9],
                },
                "rpy_deg_bounds": {
                    "min": [-5.0, -5.0, -3.0],
                    "max": [5.0, 5.0, 3.0],
                },
            }]
        }
        pose = np.asarray(_sample_route_template(template, 42)[0])
        self.assertTrue(np.all(pose[:3] >= [-0.1, -1.4, 0.8]))
        self.assertTrue(np.all(pose[:3] <= [0.1, -1.3, 0.9]))
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)

    def test_topology_signature_distinguishes_corridors(self):
        cuts = [{
            "id": "gate",
            "axis": "y",
            "value": 0.0,
            "classes": [
                {"label": "through", "x_range": [-0.5, 0.5],
                 "z_range": [0.0, 1.0]},
                {"label": "above", "z_min": 1.5},
                {"label": "left", "x_max": -0.8},
                {"label": "right", "x_min": 0.8},
            ],
        }]
        through = np.asarray([[-0.1, -1.0, 0.5], [0.1, 1.0, 0.5]])
        left = np.asarray([[-1.2, -1.0, 0.5], [-1.2, 1.0, 0.5]])
        above = np.asarray([[0.0, -1.0, 2.0], [0.0, 1.0, 2.0]])
        self.assertEqual(_topology_signature(through, cuts), {"gate": "through"})
        self.assertEqual(_topology_signature(left, cuts), {"gate": "left"})
        self.assertEqual(_topology_signature(above, cuts), {"gate": "above"})


if __name__ == "__main__":
    unittest.main()
