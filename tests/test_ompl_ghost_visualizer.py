"""Regression tests for mission-pose robot ghost visualization."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from hnuter_ompl_mppi_demo import OMPLMPPIVisualizer
from mppi.quaternion import quaternion_from_euler
from ompl_se3_planner import PlannedSE3Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _path(states: np.ndarray) -> PlannedSE3Path:
    return PlannedSE3Path(
        states=states,
        planning_time_s=0.0,
        raw_state_count=len(states),
        path_length_m=float(
            np.sum(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1))
        ),
        rotation_length_rad=0.0,
    )


class GhostVisualizerTest(unittest.TestCase):
    def test_hidden_collision_geom_is_not_copied(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="drone">
                  <freejoint/>
                  <geom name="visual" type="box" size=".2 .1 .05"
                        contype="0" conaffinity="0"/>
                  <geom name="collision" type="sphere" size=".3"
                        rgba="1 0 0 0"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        states = np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        visualizer = OMPLMPPIVisualizer(
            model, _path(states), (), 0.0, 1
        )

        self.assertEqual(len(visualizer._canonical_ghost), 1)
        visual_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "visual"
        )
        self.assertEqual(
            visualizer._canonical_ghost[0][0], visual_geom_id
        )

    def test_r1_r2_connection_is_rigid_across_all_ghosts(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(PROJECT_DIR / "hnuter206_4_5kg.xml")
        )
        start_quaternion = quaternion_from_euler((0.1, -0.2, 0.3))
        goal_quaternion = quaternion_from_euler((-0.3, 0.25, 1.1))
        waypoint_quaternion = quaternion_from_euler((0.4, 0.2, -0.8))
        states = np.asarray(
            [
                [-1.0, -0.5, 1.0, *start_quaternion],
                [1.2, 0.8, 1.7, *goal_quaternion],
            ]
        )
        waypoint = np.asarray(
            [[0.2, 1.0, 1.4, *waypoint_quaternion]]
        )
        visualizer = OMPLMPPIVisualizer(
            model,
            _path(states),
            (),
            0.0,
            1,
            waypoint_states=waypoint,
        )
        r2_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "r2_mesh"
        )
        r1_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "r1_mesh"
        )
        canonical = {
            geometry[0]: geometry
            for geometry in visualizer._canonical_ghost
        }
        canonical_r2_rotation = canonical[r2_id][4].reshape(3, 3)
        canonical_relative_position = (
            canonical_r2_rotation.T
            @ (canonical[r1_id][3] - canonical[r2_id][3])
        )
        canonical_relative_rotation = (
            canonical_r2_rotation.T
            @ canonical[r1_id][4].reshape(3, 3)
        )
        for ghost in (
            visualizer._start_ghost,
            *visualizer._waypoint_ghosts,
            visualizer._goal_ghost,
        ):
            geometry = {item[0]: item for item in ghost}
            r2_rotation = geometry[r2_id][4].reshape(3, 3)
            np.testing.assert_allclose(
                r2_rotation.T
                @ (geometry[r1_id][3] - geometry[r2_id][3]),
                canonical_relative_position,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                r2_rotation.T
                @ geometry[r1_id][4].reshape(3, 3),
                canonical_relative_rotation,
                atol=1.0e-12,
            )

    def test_ghost_geoms_are_decorations_not_physics_objects(self) -> None:
        model = mujoco.MjModel.from_xml_path(
            str(PROJECT_DIR / "hnuter206_4_5kg.xml")
        )
        states = np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        visualizer = OMPLMPPIVisualizer(
            model, _path(states), (), 0.0, 1
        )
        scene = mujoco.MjvScene(model, maxgeom=100)
        visualizer._add_robot_ghost(
            scene,
            visualizer._start_ghost,
            np.asarray([0.2, 0.5, 1.0, 0.3], dtype=np.float32),
        )

        self.assertGreater(scene.ngeom, 0)
        for geometry, source in zip(
            scene.geoms[: scene.ngeom],
            visualizer._start_ghost,
        ):
            source_geom_id, source_geom_type = source[:2]
            self.assertEqual(
                geometry.objtype, mujoco.mjtObj.mjOBJ_UNKNOWN
            )
            self.assertEqual(geometry.objid, -1)
            self.assertEqual(
                geometry.category, mujoco.mjtCatBit.mjCAT_DECOR
            )
            if source_geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                self.assertEqual(
                    geometry.dataid,
                    2 * int(model.geom_dataid[source_geom_id]),
                )

        r2_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "r2_mesh"
        )
        r1_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "r1_mesh"
        )
        rendered_by_source = {
            source[0]: geometry
            for source, geometry in zip(
                visualizer._start_ghost,
                scene.geoms[: scene.ngeom],
            )
        }
        self.assertEqual(rendered_by_source[r2_id].dataid, 2)
        self.assertEqual(rendered_by_source[r1_id].dataid, 4)


if __name__ == "__main__":
    unittest.main()
