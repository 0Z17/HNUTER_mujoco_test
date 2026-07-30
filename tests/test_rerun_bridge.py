"""Tests for the simulator-independent Rerun bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rerun_bridge import (
    Pose3D,
    RerunRecorderConfig,
    RerunSimulationRecorder,
    Sphere3D,
    _DEFAULT_ROBOT_MJCF,
    _DEFAULT_ROBOT_URDF,
    _load_robot_model,
)


class RerunBridgeTest(unittest.TestCase):
    def test_config_requires_a_sink(self) -> None:
        with self.assertRaisesRegex(ValueError, "sink"):
            RerunRecorderConfig()

    def test_pose_normalizes_quaternion(self) -> None:
        pose = Pose3D(np.zeros(3), np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(pose.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])

    def test_urdf_visuals_use_mjcf_kinematics(self) -> None:
        model = _load_robot_model(
            _DEFAULT_ROBOT_URDF, _DEFAULT_ROBOT_MJCF
        )
        links = {link.name: link for link in model.links}

        self.assertEqual(model.root_link, "base_link")
        self.assertEqual(len(links), 10)
        self.assertEqual(
            model.joint_names,
            frozenset(
                {
                    "rj2",
                    "rj1",
                    "lj2",
                    "lj1",
                    "xyj1",
                    "xyj2",
                    "xyj3",
                    "xyj4",
                    "xyj5",
                }
            ),
        )
        self.assertEqual(links["xy3"].parent, "l1")
        self.assertEqual(links["r2"].parent, "base_link")
        self.assertEqual(links["r2"].joint_name, "rj2")
        self.assertEqual(links["r1"].parent, "r2")
        self.assertEqual(links["r1"].joint_name, "rj1")
        np.testing.assert_allclose(
            links["l2"].joint_axis, [0.0, 0.0, -1.0]
        )
        np.testing.assert_allclose(
            links["xy3"].quaternion_wxyz,
            np.array([0.706825, 0.707388, 0.0, 0.0297145])
            / np.linalg.norm([0.706825, 0.707388, 0.0, 0.0297145]),
        )
        self.assertTrue(
            all(link.mesh_path.is_file() for link in model.links)
        )

    def test_rrd_is_finalized_and_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording_path = Path(directory) / "bridge_test.rrd"
            recorder = RerunSimulationRecorder(
                RerunRecorderConfig(
                    application_id="hnuter_rerun_bridge_test",
                    recording_path=recording_path,
                    trace_update_stride=1,
                )
            )
            recorder.log_static_scene(
                planned_path=np.array(
                    [[0.0, 0.0, 1.0], [1.0, 0.5, 1.2]]
                ),
                timed_reference_path=np.array(
                    [[0.0, 0.0, 1.0], [1.0, 0.4, 1.2]]
                ),
                start_pose=Pose3D(
                    np.array([0.0, 0.0, 1.0]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                ),
                goal_pose=Pose3D(
                    np.array([1.0, 0.5, 1.2]),
                    np.array([0.9239, 0.0, 0.0, 0.3827]),
                ),
                obstacles=(
                    Sphere3D(np.array([0.5, 0.2, 1.0]), 0.2),
                ),
                metadata={"test": True},
            )
            for index in range(4):
                position = np.array([0.1 * index, 0.0, 1.0])
                recorder.log_frame(
                    0.05 * index,
                    actual_pose=Pose3D(
                        position,
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    ),
                    reference_pose=Pose3D(
                        position + np.array([0.0, 0.01, 0.0]),
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    ),
                    linear_velocity=np.array([0.2, 0.0, 0.0]),
                    angular_velocity=np.zeros(3),
                    control=np.zeros(6),
                    nominal_positions=np.array(
                        [position, position + np.array([0.1, 0.0, 0.0])]
                    ),
                    sampled_positions=np.zeros((2, 3, 3)),
                    joint_positions={
                        "rj2": 0.1 * index,
                        "lj2": -0.1 * index,
                        "xyj1": 0.8 * index,
                        "xyj2": -0.8 * index,
                    },
                    scalar_channels={
                        "tracking/position_error_m": 0.01
                    },
                )
            self.assertEqual(recorder.frame_count, 4)
            recorder.close()
            recorder.close()

            self.assertTrue(recording_path.is_file())
            self.assertGreater(recording_path.stat().st_size, 1024)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                recorder.log_frame(
                    1.0,
                    actual_pose=Pose3D(
                        np.zeros(3),
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
