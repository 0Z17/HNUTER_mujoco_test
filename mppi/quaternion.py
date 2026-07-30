"""Vectorized quaternion utilities using MuJoCo's ``[w, x, y, z]`` layout."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def normalize_quaternion(quaternion: ArrayLike) -> FloatArray:
    quaternion_array = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion_array, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise ValueError("quaternion norm must be nonzero")
    return quaternion_array / norm


def quaternion_conjugate(quaternion: ArrayLike) -> FloatArray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Hamilton product supporting arbitrary matching leading dimensions."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left_array, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right_array, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_from_rotation_vector(rotation_vector: ArrayLike) -> FloatArray:
    rotation_vector_array = np.asarray(
        rotation_vector, dtype=np.float64
    )
    angle = np.linalg.norm(rotation_vector_array, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1.0e-7
    scale[small] = 0.5 - np.square(angle[small]) / 48.0
    scale[~small] = np.sin(half_angle[~small]) / angle[~small]
    return normalize_quaternion(
        np.concatenate(
            (np.cos(half_angle), rotation_vector_array * scale), axis=-1
        )
    )


def quaternion_from_euler(euler: ArrayLike) -> FloatArray:
    """Convert ZYX roll/pitch/yaw angles to ``[w, x, y, z]`` quaternions."""

    euler_array = np.asarray(euler, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(euler_array, -1, 0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return normalize_quaternion(
        np.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            axis=-1,
        )
    )


def quaternion_to_euler(quaternion: ArrayLike) -> FloatArray:
    quaternion_array = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion_array, -1, 0)
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.stack((roll, pitch, yaw), axis=-1)


def quaternion_to_rotation_matrix(quaternion: ArrayLike) -> FloatArray:
    quaternion_array = normalize_quaternion(quaternion)
    w, x, y, z = np.moveaxis(quaternion_array, -1, 0)
    matrix = np.empty(quaternion_array.shape[:-1] + (3, 3))
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - w * z)
    matrix[..., 0, 2] = 2.0 * (x * z + w * y)
    matrix[..., 1, 0] = 2.0 * (x * y + w * z)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - w * x)
    matrix[..., 2, 0] = 2.0 * (x * z - w * y)
    matrix[..., 2, 1] = 2.0 * (y * z + w * x)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def quaternion_error_vector(
    actual: ArrayLike, desired: ArrayLike
) -> FloatArray:
    """Shortest desired-to-actual rotation vector in desired body axes."""

    error_quaternion = normalize_quaternion(
        quaternion_multiply(
            quaternion_conjugate(desired),
            actual,
        )
    )
    sign = np.where(error_quaternion[..., :1] < 0.0, -1.0, 1.0)
    error_quaternion = error_quaternion * sign
    vector = error_quaternion[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(
        vector_norm,
        np.clip(error_quaternion[..., :1], 0.0, 1.0),
    )
    scale = np.ones_like(vector_norm) * 2.0
    nonsmall = vector_norm > 1.0e-8
    scale[nonsmall] = angle[nonsmall] / vector_norm[nonsmall]
    return vector * scale


def body_rates_from_euler_rates(
    euler: ArrayLike, euler_rates: ArrayLike
) -> FloatArray:
    """Convert ZYX Euler derivatives to body-frame angular velocity."""

    euler_array = np.asarray(euler, dtype=np.float64)
    rates_array = np.asarray(euler_rates, dtype=np.float64)
    roll, pitch, _ = np.moveaxis(euler_array, -1, 0)
    roll_rate, pitch_rate, yaw_rate = np.moveaxis(rates_array, -1, 0)
    return np.stack(
        (
            roll_rate - yaw_rate * np.sin(pitch),
            pitch_rate * np.cos(roll)
            + yaw_rate * np.sin(roll) * np.cos(pitch),
            -pitch_rate * np.sin(roll)
            + yaw_rate * np.cos(roll) * np.cos(pitch),
        ),
        axis=-1,
    )
