"""Tests for the cuRobo coarse filter integration."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from curobo_collision import load_curobo_spheres


try:
    from run_unet_guided_diffusion_demo import candidate_metrics_staged
except (ImportError, OSError):  # pragma: no cover - COAL is environment-side
    candidate_metrics_staged = None


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class CuroboSphereConfigTest(unittest.TestCase):
    def test_sphere_set_loads(self) -> None:
        from pathlib import Path

        sphere_set = load_curobo_spheres(
            Path(
                "etc/URDF-for-gazebo/config/"
                "HDJQR-0102-0055.SLDASM_curobo_spheres.yml"
            )
        )
        self.assertEqual(sphere_set.count, 42)
        self.assertEqual(sphere_set.centers.shape, (42, 3))
        self.assertEqual(sphere_set.radii.shape, (42,))
        self.assertTrue(np.all(sphere_set.radii > 0.0))


@unittest.skipIf(
    candidate_metrics_staged is None, "demo module is not importable here"
)
class StagedFilterTest(unittest.TestCase):
    def test_coarse_accept_skips_coal_and_keeps_acceptance(self) -> None:
        paths = np.asarray(
            [
                [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 4,
                [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * 4,
            ],
            dtype=np.float64,
        )
        bounds_min = np.asarray([-3.0, -3.5, 0.0])
        bounds_max = np.asarray([3.0, 3.5, 4.0])

        class FakeChecker:
            def clearance(self, position, quaternion):
                # Candidate 1 (index 1) would fail COAL with 0.03 m clearance.
                return np.full(len(position), 0.03)

        records = candidate_metrics_staged(
            paths,
            FakeChecker(),
            bounds_min,
            bounds_max,
            0.08,
            coarse_accept=np.asarray([True, False]),
            coarse_clearance=np.asarray([0.15, 0.03]),
        )
        self.assertTrue(records[0]["accepted_8cm"])
        self.assertEqual(records[0]["verified_by"], "curobo_spheres")
        self.assertGreaterEqual(
            records[0]["minimum_physical_clearance_m"], 0.08
        )
        self.assertFalse(records[1]["accepted_8cm"])
        self.assertEqual(records[1]["verified_by"], "coal")


if __name__ == "__main__":
    unittest.main()
