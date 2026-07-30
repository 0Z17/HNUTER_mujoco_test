"""Low-level geometric controller for the HNUTER MuJoCo aircraft.

This module extracts the reusable flight-control part of the original demo
scripts.  It accepts a desired position, velocity, acceleration, and attitude;
the MPPI controller can therefore remain a clean outer-loop module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class HnuterController:
    """SE(3) tracking and nonlinear actuator allocation for the HNUTER model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.dt = float(self.model.opt.timestep)
        self.gravity = 9.81
        self.mass = 4.5
        self.inertia = np.diag([0.10, 0.04, 0.08])
        self.l1 = 0.0855
        self.l2 = 0.768

        self.position_gain = np.diag([15.0, 15.0, 30.0])
        self.velocity_gain = np.diag([8.0, 8.0, 12.0])
        self.attitude_gain = np.array([6.0, 4.0, 4.0])
        self.angular_rate_gain = np.array([1.2, 0.8, 1.0])

        self.target_position = np.array([0.0, 0.0, 1.0])
        self.target_velocity = np.zeros(3)
        self.target_acceleration = np.zeros(3)
        self.target_rotation_matrix = np.eye(3)
        self.target_attitude_rate = np.zeros(3)
        self.target_angular_acceleration = np.zeros(3)

        self.T12 = 0.0
        self.T34 = 0.0
        self.T5 = 0.0
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self.theta1 = 0.0
        self.theta2 = 0.0

        self._body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, "drone"
        )
        self._actuator_ids = {
            "arm_pitch_right": self._required_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_rj2"
            ),
            "arm_pitch_left": self._required_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_lj2"
            ),
            "prop_tilt_right": self._required_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_rj1"
            ),
            "prop_tilt_left": self._required_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_lj1"
            ),
        }
        for name in (
            "motor_xy1",
            "motor_xy2",
            "motor_xy3",
            "motor_xy4",
            "motor_xy5",
        ):
            self._actuator_ids[name] = self._required_id(
                mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo model is missing required object {name!r}")
        return object_id

    def get_state(self) -> dict[str, FloatArray]:
        """Read the drone body state in the world/FLU convention."""

        quaternion = self.data.xquat[self._body_id].copy()
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        angular_velocity_world = self.data.cvel[self._body_id, :3].copy()
        return {
            "position": self.data.xpos[self._body_id].copy(),
            "quaternion": quaternion,
            "velocity": self.data.cvel[self._body_id, 3:6].copy(),
            "angular_velocity": (
                rotation_matrix.T @ angular_velocity_world
            ),
            "rotation_matrix": rotation_matrix,
            "euler": self.quaternion_to_euler(quaternion),
        }

    def get_mppi_state(self) -> FloatArray:
        """Return ``[position, velocity]`` in the MPPI model's state layout."""

        state = self.get_state()
        return np.concatenate((state["position"], state["velocity"]))

    def get_mppi_pose_state(self) -> FloatArray:
        """Return the 13-state quaternion layout used by 6-DoF MPPI."""

        state = self.get_state()
        return np.concatenate(
            (
                state["position"],
                state["velocity"],
                state["quaternion"],
                state["angular_velocity"],
            )
        )

    def set_freejoint_pose(
        self,
        position: ArrayLike,
        quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Set the root free-joint pose, primarily for demo initialization."""

        body_joint_start = int(self.model.body_jntadr[self._body_id])
        body_joint_count = int(self.model.body_jntnum[self._body_id])
        free_joint_id = -1
        for joint_id in range(
            body_joint_start, body_joint_start + body_joint_count
        ):
            if (
                self.model.jnt_type[joint_id]
                == mujoco.mjtJoint.mjJNT_FREE
            ):
                free_joint_id = joint_id
                break
        if free_joint_id < 0:
            raise ValueError("drone body does not have a root free joint")

        position_array = np.asarray(position, dtype=np.float64)
        quaternion_array = np.asarray(quaternion, dtype=np.float64)
        if position_array.shape != (3,):
            raise ValueError("position must be a 3-vector")
        if quaternion_array.shape != (4,):
            raise ValueError("quaternion must be a 4-vector")
        quaternion_norm = np.linalg.norm(quaternion_array)
        if quaternion_norm < 1.0e-9:
            raise ValueError("quaternion norm must be nonzero")

        qpos_address = int(self.model.jnt_qposadr[free_joint_id])
        dof_address = int(self.model.jnt_dofadr[free_joint_id])
        self.data.qpos[qpos_address : qpos_address + 3] = position_array
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            quaternion_array / quaternion_norm
        )
        self.data.qvel[dof_address : dof_address + 6] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_desired_state(self, desired: Mapping[str, ArrayLike]) -> None:
        """Set the desired state consumed by :meth:`update_control`."""

        self.target_position = self._vector(desired, "pos")
        self.target_velocity = self._vector(desired, "vel")
        self.target_acceleration = self._vector(desired, "acc")

        if "quaternion" in desired:
            desired_quaternion = np.asarray(
                desired["quaternion"], dtype=np.float64
            )
            if desired_quaternion.shape != (4,) or not np.all(
                np.isfinite(desired_quaternion)
            ):
                raise ValueError(
                    "desired['quaternion'] must be a finite 4-vector"
                )
            quaternion_norm = np.linalg.norm(desired_quaternion)
            if quaternion_norm < 1.0e-9:
                raise ValueError("desired quaternion norm must be nonzero")
            self.target_rotation_matrix = (
                self.quaternion_to_rotation_matrix(
                    desired_quaternion / quaternion_norm
                )
            )
        elif "euler" in desired:
            desired_euler = self._vector(desired, "euler")
            self.target_rotation_matrix = self.euler_to_rotation_matrix(
                desired_euler
            )
        else:
            desired_euler = np.array(
                [0.0, 0.0, float(desired.get("yaw", 0.0))]
            )
            self.target_rotation_matrix = self.euler_to_rotation_matrix(
                desired_euler
            )

        if "angular_velocity" in desired:
            self.target_attitude_rate = self._vector(
                desired, "angular_velocity"
            )
        elif "euler_rate" in desired:
            self.target_attitude_rate = self._vector(
                desired, "euler_rate"
            )
        else:
            self.target_attitude_rate = np.array(
                [0.0, 0.0, float(desired.get("yaw_rate", 0.0))]
            )
        if "angular_acceleration" in desired:
            self.target_angular_acceleration = self._vector(
                desired, "angular_acceleration"
            )
        else:
            self.target_angular_acceleration = np.zeros(3)

    @staticmethod
    def _vector(
        desired: Mapping[str, ArrayLike], key: str
    ) -> FloatArray:
        vector = np.asarray(desired[key], dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"desired[{key!r}] must be a finite 3-vector")
        return vector.copy()

    def update_control(self) -> None:
        """Run one low-level feedback and actuator-allocation update."""

        state = self.get_state()
        force_body, torque_body = self.compute_control_wrench(state)
        actuator_command = self.allocate_actuators(
            force_body, torque_body
        )
        self.set_actuators(*actuator_command)

    def compute_control_wrench(
        self, state: Mapping[str, FloatArray]
    ) -> tuple[FloatArray, FloatArray]:
        position_error = self.target_position - state["position"]
        velocity_error = self.target_velocity - state["velocity"]
        desired_acceleration = (
            self.target_acceleration
            + self.position_gain @ position_error
            + self.velocity_gain @ velocity_error
        )
        force_world = self.mass * (
            desired_acceleration + np.array([0.0, 0.0, self.gravity])
        )

        rotation = state["rotation_matrix"]
        attitude_error = 0.5 * self.vee(
            self.target_rotation_matrix.T @ rotation
            - rotation.T @ self.target_rotation_matrix
        )
        angular_rate_error = (
            state["angular_velocity"]
            - rotation.T
            @ self.target_rotation_matrix
            @ self.target_attitude_rate
        )
        desired_angular_acceleration_body = (
            rotation.T
            @ self.target_rotation_matrix
            @ self.target_angular_acceleration
        )
        gyroscopic_torque = np.cross(
            state["angular_velocity"],
            self.inertia @ state["angular_velocity"],
        )
        torque_body = (
            -self.attitude_gain * attitude_error
            - self.angular_rate_gain * angular_rate_error
            + self.inertia @ desired_angular_acceleration_body
            + gyroscopic_torque
        )
        return rotation.T @ force_world, torque_body

    def allocate_actuators(
        self, force_body: FloatArray, torque_body: FloatArray
    ) -> tuple[float, float, float, float, float, float, float]:
        """Map a desired body wrench to rotor thrusts and tilt angles."""

        fx, fy, fz = force_body
        tau_x, tau_y, tau_z = torque_body

        tail_thrust = tau_y / self.l2
        right_fx = fx / 2.0 - tau_z / (2.0 * self.l1)
        left_fx = fx / 2.0 + tau_z / (2.0 * self.l1)
        right_fz = fz / 2.0 + tau_x / (2.0 * self.l1)
        left_fz = fz / 2.0 - tau_x / (2.0 * self.l1)
        right_fy = -fy / 2.0
        left_fy = -fy / 2.0

        right_thrust = math.sqrt(
            right_fx**2 + right_fy**2 + right_fz**2
        )
        left_thrust = math.sqrt(
            left_fx**2 + left_fy**2 + left_fz**2
        )
        right_safe = max(right_thrust, 1.0e-8)
        left_safe = max(left_thrust, 1.0e-8)

        alpha1 = math.atan2(right_fx, right_fz)
        alpha2 = math.atan2(left_fx, left_fz)
        theta1 = math.asin(np.clip(right_fy / right_safe, -0.99, 0.99))
        theta2 = math.asin(np.clip(left_fy / left_safe, -0.99, 0.99))

        right_thrust = float(np.clip(right_thrust, 0.0, 50.0))
        left_thrust = float(np.clip(left_thrust, 0.0, 50.0))
        tail_thrust = float(np.clip(tail_thrust, -20.0, 20.0))
        tilt_limit = math.radians(200.0)
        alpha1 = float(np.clip(alpha1, -tilt_limit, tilt_limit))
        alpha2 = float(np.clip(alpha2, -tilt_limit, tilt_limit))
        theta1 = float(np.clip(theta1, -tilt_limit, tilt_limit))
        theta2 = float(np.clip(theta2, -tilt_limit, tilt_limit))

        self.T12 = right_thrust
        self.T34 = left_thrust
        self.T5 = tail_thrust
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.theta1 = theta1
        self.theta2 = theta2
        return (
            right_thrust,
            left_thrust,
            tail_thrust,
            alpha1,
            alpha2,
            theta1,
            theta2,
        )

    def set_actuators(
        self,
        right_thrust: float,
        left_thrust: float,
        tail_thrust: float,
        alpha1: float,
        alpha2: float,
        theta1: float,
        theta2: float,
    ) -> None:
        controls = self.data.ctrl
        controls[self._actuator_ids["arm_pitch_right"]] = alpha2
        controls[self._actuator_ids["arm_pitch_left"]] = alpha1
        controls[self._actuator_ids["prop_tilt_right"]] = theta2
        controls[self._actuator_ids["prop_tilt_left"]] = theta1
        controls[self._actuator_ids["motor_xy1"]] = left_thrust / 2.0
        controls[self._actuator_ids["motor_xy2"]] = left_thrust / 2.0
        controls[self._actuator_ids["motor_xy3"]] = right_thrust / 2.0
        controls[self._actuator_ids["motor_xy4"]] = right_thrust / 2.0
        controls[self._actuator_ids["motor_xy5"]] = tail_thrust * 12.0

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: ArrayLike) -> FloatArray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        )

    @staticmethod
    def quaternion_to_euler(quaternion: ArrayLike) -> FloatArray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        roll = math.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = (
            math.copysign(math.pi / 2.0, sin_pitch)
            if abs(sin_pitch) >= 1.0
            else math.asin(sin_pitch)
        )
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return np.array([roll, pitch, yaw])

    @staticmethod
    def euler_to_rotation_matrix(euler: ArrayLike) -> FloatArray:
        roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        rotation_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_roll, -sin_roll],
                [0.0, sin_roll, cos_roll],
            ]
        )
        rotation_y = np.array(
            [
                [cos_pitch, 0.0, sin_pitch],
                [0.0, 1.0, 0.0],
                [-sin_pitch, 0.0, cos_pitch],
            ]
        )
        rotation_z = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return rotation_z @ rotation_y @ rotation_x

    @staticmethod
    def vee(skew_matrix: FloatArray) -> FloatArray:
        return np.array(
            [skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]]
        )
