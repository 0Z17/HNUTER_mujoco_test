"""Tests for the rigid-URDF COAL collision interface."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coal_collision import (
    CoalCollisionChecker,
    RigidURDFCollisionModel,
    StaticCollisionObject,
)
from mppi.quaternion import quaternion_from_euler
from ompl_se3_planner import OMPLSE3Planner, SE3Pose


PROJECT_DIR = Path(__file__).resolve().parents[1]
HNUTER_URDF = (
    PROJECT_DIR
    / "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf"
)


class _AttitudeAwareTestChecker:
    """Small structural implementation of the planner checker protocol."""

    def is_collision_free(self, position, quaternion) -> bool:
        del position
        # Reject attitudes whose body x axis points toward world -x.
        return bool(np.asarray(quaternion, dtype=float)[0] > 0.5)

    def clearance(self, positions, quaternions):
        del positions
        quaternion_array = np.asarray(quaternions, dtype=float)
        return quaternion_array[..., 0] - 0.5


class RigidURDFModelTest(unittest.TestCase):
    def test_hnuter_base_link_collision_primitives_are_loaded(self) -> None:
        model = RigidURDFCollisionModel.from_urdf(HNUTER_URDF)

        self.assertEqual(model.link_name, "base_link")
        self.assertEqual(len(model.geometries), 7)
        self.assertEqual(
            [item.geometry.kind for item in model.geometries],
            [
                "box",
                "cylinder",
                "sphere",
                "cylinder",
                "sphere",
                "cylinder",
                "cylinder",
            ],
        )
        np.testing.assert_allclose(
            model.geometries[-1].local_pose.position,
            (-0.768, 0.0, 0.0034),
        )

    def test_collision_on_another_link_breaks_rigid_body_assumption(
        self,
    ) -> None:
        document = """
        <robot name="moving">
          <link name="base_link">
            <collision><geometry><sphere radius="1"/></geometry></collision>
          </link>
          <link name="arm">
            <collision><geometry><box size="1 1 1"/></geometry></collision>
          </link>
        </robot>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moving.urdf"
            path.write_text(document)
            with self.assertRaisesRegex(ValueError, "collisions on 'arm'"):
                RigidURDFCollisionModel.from_urdf(path)


class OMPLCollisionInterfaceTest(unittest.TestCase):
    def test_planner_passes_orientation_to_external_checker(self) -> None:
        planner = OMPLSE3Planner(
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            collision_checker=_AttitudeAwareTestChecker(),
        )
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        half_turn = quaternion_from_euler((math.pi, 0.0, 0.0))

        self.assertTrue(planner.is_pose_valid((0.0, 0.0, 0.0), identity))
        self.assertFalse(
            planner.is_pose_valid((0.0, 0.0, 0.0), half_turn)
        )
        with self.assertRaisesRegex(ValueError, "quaternions are required"):
            planner.clearance(np.zeros((2, 3)))
        clearances = planner.clearance(
            np.zeros((2, 3)), np.stack((identity, half_turn))
        )
        self.assertGreater(clearances[0], 0.0)
        self.assertLess(clearances[1], 0.0)


@unittest.skipUnless(
    importlib.util.find_spec("coal") is not None
    or importlib.util.find_spec("hppfcl") is not None,
    "COAL Python bindings are not installed",
)
class CoalBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = (
            StaticCollisionObject.sphere(
                "tail_obstacle", (-1.04, 0.0, 0.0034), 0.10
            ),
        )
        self.checker = CoalCollisionChecker.from_urdf(
            HNUTER_URDF,
            self.environment,
            safety_margin=0.01,
        )

    def test_real_urdf_geometry_collision_depends_on_attitude(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        yaw_half_turn = quaternion_from_euler((0.0, 0.0, math.pi))

        colliding = self.checker.check_pose((0.0, 0.0, 0.0), identity)
        free = self.checker.check_pose((0.0, 0.0, 0.0), yaw_half_turn)

        self.assertEqual(self.checker.vehicle_geometry_count, 7)
        self.assertFalse(colliding.collision_free)
        self.assertLess(colliding.signed_clearance, 0.0)
        self.assertEqual(
            colliding.vehicle_geometry, "tail_rotoer_collision"
        )
        self.assertEqual(colliding.environment_object, "tail_obstacle")
        self.assertTrue(free.collision_free)
        self.assertGreater(free.signed_clearance, 0.0)

    def test_batch_clearance_preserves_pose_leading_shape(self) -> None:
        positions = np.zeros((2, 2, 3))
        quaternions = np.empty((2, 2, 4))
        quaternions[:] = (1.0, 0.0, 0.0, 0.0)
        clearances = self.checker.clearance(positions, quaternions)
        self.assertEqual(clearances.shape, (2, 2))
        self.assertTrue(np.all(clearances < 0.0))

    def test_ompl_path_is_validated_with_full_urdf_model(self) -> None:
        checker = CoalCollisionChecker.from_urdf(
            HNUTER_URDF,
            (
                StaticCollisionObject.sphere(
                    "center_obstacle", (0.0, 0.0, 0.0), 0.35
                ),
            ),
            safety_margin=0.05,
        )
        planner = OMPLSE3Planner(
            (-2.5, -2.0, -1.0),
            (2.5, 2.0, 1.0),
            vehicle_radius=0.0,
            safety_margin=0.0,
            collision_checker=checker,
            validity_resolution=0.005,
            planner_range=0.35,
            seed=23,
        )
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        path = planner.plan(
            SE3Pose(np.array([-1.7, -0.8, 0.0]), identity),
            SE3Pose(np.array([1.7, 0.8, 0.0]), identity),
            solve_time=3.0,
            interpolation_resolution=0.05,
            minimum_waypoints=100,
        )
        clearance = planner.clearance(
            path.states[:, :3], path.states[:, 3:7]
        )
        self.assertTrue(np.all(clearance > -1.0e-9))


if __name__ == "__main__":
    unittest.main()
