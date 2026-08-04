#!/usr/bin/env python3
"""Compact conditional diffusion models for collision-aware SE(3) paths.

The module deliberately has no dependency on the MPD source tree.  It follows
the useful MPD design choices (task-grouped splits, endpoint conditioning,
DDIM sampling, and differentiable inference-time cost guidance) while keeping
the representation and networks specific to this rigid-body SE(3) dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


POSE_DIM = 9  # xyz + first two columns of SO(3)
CONDITION_DIM = 2 * POSE_DIM
OBSTACLE_DIM = 10  # xyz + size xyz + quaternion wxyz


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = (1.0 - fraction) * first + fraction * second
        return value / np.linalg.norm(value)
    angle = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * angle) * first
        + math.sin(fraction * angle) * second
    ) / math.sin(angle)


def resample_se3_path(
    states: np.ndarray,
    count: int,
    rotation_weight_m_per_rad: float = 0.22,
) -> np.ndarray:
    """Arc-length resample xyz+wxyz states while using quaternion SLERP."""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) < 2:
        raise ValueError("states must have shape (N, 7), N >= 2")
    positions = states[:, :3]
    quaternions = _continuous_quaternions(states[:, 3:7])
    increments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    increments += rotation_weight_m_per_rad * _quaternion_angle(
        quaternions[:-1], quaternions[1:]
    )
    distance = np.concatenate(([0.0], np.cumsum(increments)))
    if distance[-1] <= 1e-12:
        indices = np.zeros(count, dtype=np.int64)
        result = np.concatenate((positions[indices], quaternions[indices]), axis=1)
        return result
    targets = np.linspace(0.0, distance[-1], count)
    result = np.empty((count, 7), dtype=np.float64)
    for output_index, target in enumerate(targets):
        right = int(np.searchsorted(distance, target, side="right"))
        right = min(max(right, 1), len(distance) - 1)
        left = right - 1
        denominator = distance[right] - distance[left]
        fraction = 0.0 if denominator <= 1e-12 else (
            float(target - distance[left]) / float(denominator)
        )
        result[output_index, :3] = (
            (1.0 - fraction) * positions[left] + fraction * positions[right]
        )
        result[output_index, 3:7] = _slerp(
            quaternions[left], quaternions[right], fraction
        )
    result[0] = np.concatenate((positions[0], quaternions[0]))
    result[-1] = np.concatenate((positions[-1], quaternions[-1]))
    return result


def quaternion_wxyz_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
        2 * (x * z + y * w), 2 * (x * y + z * w),
        1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(quaternion.shape[:-1] + (3, 3))


def pose7_to_pose9(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    rotation = quaternion_wxyz_to_matrix_numpy(pose[..., 3:7])
    rotation_6d = np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)
    return np.concatenate((pose[..., :3], rotation_6d), axis=-1)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    first = F.normalize(rotation_6d[..., :3], dim=-1, eps=1e-8)
    second_raw = rotation_6d[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_quaternion_wxyz(matrix: Tensor) -> Tensor:
    """Differentiable matrix-to-quaternion conversion, output wxyz."""

    batch_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    output = torch.empty((len(flat), 4), dtype=matrix.dtype, device=matrix.device)
    for index, item in enumerate(flat):
        trace = item.trace()
        if trace > 0:
            scale = torch.sqrt(trace + 1.0) * 2.0
            output[index] = torch.stack((
                0.25 * scale,
                (item[2, 1] - item[1, 2]) / scale,
                (item[0, 2] - item[2, 0]) / scale,
                (item[1, 0] - item[0, 1]) / scale,
            ))
        else:
            diagonal = torch.diagonal(item)
            largest = int(torch.argmax(diagonal).item())
            if largest == 0:
                scale = torch.sqrt(1.0 + item[0, 0] - item[1, 1] - item[2, 2]) * 2.0
                output[index] = torch.stack(((item[2, 1] - item[1, 2]) / scale,
                    0.25 * scale, (item[0, 1] + item[1, 0]) / scale,
                    (item[0, 2] + item[2, 0]) / scale))
            elif largest == 1:
                scale = torch.sqrt(1.0 + item[1, 1] - item[0, 0] - item[2, 2]) * 2.0
                output[index] = torch.stack(((item[0, 2] - item[2, 0]) / scale,
                    (item[0, 1] + item[1, 0]) / scale, 0.25 * scale,
                    (item[1, 2] + item[2, 1]) / scale))
            else:
                scale = torch.sqrt(1.0 + item[2, 2] - item[0, 0] - item[1, 1]) * 2.0
                output[index] = torch.stack(((item[1, 0] - item[0, 1]) / scale,
                    (item[0, 2] + item[2, 0]) / scale,
                    (item[1, 2] + item[2, 1]) / scale, 0.25 * scale))
    return F.normalize(output.reshape(batch_shape + (4,)), dim=-1)


def pose9_to_pose7_tensor(pose: Tensor) -> Tensor:
    rotation = rotation_6d_to_matrix(pose[..., 3:9])
    quaternion = matrix_to_quaternion_wxyz(rotation)
    return torch.cat((pose[..., :3], quaternion), dim=-1)


def pose9_to_pose7_numpy(pose: np.ndarray) -> np.ndarray:
    value = torch.as_tensor(pose, dtype=torch.float64)
    return pose9_to_pose7_tensor(value).cpu().numpy()


@dataclass
class PreparedData:
    paths: np.ndarray
    conditions: np.ndarray
    pair_indices: np.ndarray
    trajectory_indices: np.ndarray
    split_codes: np.ndarray
    reversed_flags: np.ndarray
    path_mean: np.ndarray
    path_std: np.ndarray
    obstacles: np.ndarray
    obstacle_mask: np.ndarray
    source_dataset: str
    environment_path: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{
            name: value for name, value in asdict(self).items()
            if isinstance(value, np.ndarray)
        }, source_dataset=np.asarray(self.source_dataset),
           environment_path=np.asarray(self.environment_path))

    @classmethod
    def load(cls, path: Path) -> "PreparedData":
        with np.load(path) as data:
            return cls(**{name: data[name].copy() for name in (
                "paths", "conditions", "pair_indices", "trajectory_indices",
                "split_codes", "reversed_flags", "path_mean", "path_std",
                "obstacles", "obstacle_mask",
            )}, source_dataset=str(data["source_dataset"]),
               environment_path=str(data["environment_path"]))


def _obstacle_tokens(environment: dict[str, Any], maximum: int = 32) -> tuple[np.ndarray, np.ndarray]:
    active = [item for item in environment.get("obstacles", [])
              if item.get("collision", False) and item.get("type") == "box"]
    if len(active) > maximum:
        raise ValueError(f"environment has {len(active)} obstacles; maximum is {maximum}")
    tokens = np.zeros((maximum, OBSTACLE_DIM), dtype=np.float32)
    mask = np.zeros(maximum, dtype=bool)
    for index, item in enumerate(active):
        pose = item["pose"]
        tokens[index] = np.asarray([
            *pose["position"], *item["size_xyz"], *pose["quaternion_wxyz"]
        ], dtype=np.float32)
        mask[index] = True
    return tokens, mask


def prepare_dataset(
    dataset_root: Path,
    output_path: Path,
    sequence_length: int = 128,
    reverse_train: bool = True,
) -> PreparedData:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    splits_path = dataset_root / "splits.json"
    if not manifest_path.exists() or not splits_path.exists():
        raise FileNotFoundError("dataset must be complete and contain manifest.json and splits.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"dataset status is {manifest.get('status')!r}, expected 'complete'")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    split_sets = {
        0: set(splits["train_pair_indices"]),
        1: set(splits["validation_pair_indices"]),
        2: set(splits["test_pair_indices"]),
    }
    paths: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    pair_indices: list[int] = []
    trajectory_indices: list[int] = []
    split_codes: list[int] = []
    reversed_flags: list[bool] = []
    for record in manifest["trajectories"]:
        pair = int(record["pair_index"])
        split = next(code for code, values in split_sets.items() if pair in values)
        with np.load(dataset_root / record["training_sample"]) as sample:
            pose7 = resample_se3_path(sample["smoothed_path_states"], sequence_length)
        pose9 = pose7_to_pose9(pose7).astype(np.float32)
        paths.append(pose9)
        conditions.append(np.concatenate((pose9[0], pose9[-1])))
        pair_indices.append(pair)
        trajectory_indices.append(int(record["trajectory_index"]))
        split_codes.append(split)
        reversed_flags.append(False)
        if reverse_train and split == 0:
            reverse = pose9[::-1].copy()
            paths.append(reverse)
            conditions.append(np.concatenate((reverse[0], reverse[-1])))
            pair_indices.append(pair)
            trajectory_indices.append(int(record["trajectory_index"]))
            split_codes.append(split)
            reversed_flags.append(True)
    path_array = np.stack(paths)
    split_array = np.asarray(split_codes, dtype=np.int8)
    training = path_array[split_array == 0]
    mean = training.mean(axis=(0, 1)).astype(np.float32)
    std = np.maximum(training.std(axis=(0, 1)), 1e-3).astype(np.float32)
    dataset_config = json.loads((dataset_root / "dataset_config.json").read_text(encoding="utf-8"))
    environment_relative = dataset_config["sources"]["environment"]["copy"]
    environment_path = dataset_root / environment_relative
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    obstacles, obstacle_mask = _obstacle_tokens(environment)
    prepared = PreparedData(
        paths=path_array,
        conditions=np.stack(conditions).astype(np.float32),
        pair_indices=np.asarray(pair_indices, dtype=np.int64),
        trajectory_indices=np.asarray(trajectory_indices, dtype=np.int64),
        split_codes=split_array,
        reversed_flags=np.asarray(reversed_flags, dtype=bool),
        path_mean=mean,
        path_std=std,
        obstacles=obstacles,
        obstacle_mask=obstacle_mask,
        source_dataset=str(dataset_root),
        environment_path=str(environment_path.resolve()),
    )
    prepared.save(output_path)
    return prepared


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, prepared: PreparedData, split_code: int) -> None:
        indices = np.flatnonzero(prepared.split_codes == split_code)
        self.paths = torch.from_numpy(
            (prepared.paths[indices] - prepared.path_mean) / prepared.path_std
        ).float()
        condition_mean = np.tile(prepared.path_mean, 2)
        condition_std = np.tile(prepared.path_std, 2)
        self.conditions = torch.from_numpy(
            (prepared.conditions[indices] - condition_mean) / condition_std
        ).float()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.paths[index], self.conditions[index]


def timestep_embedding(timesteps: Tensor, dimension: int, maximum_period: int = 10000) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(maximum_period) * torch.arange(half, device=timesteps.device) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualFiLM1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(8, in_channels)
        while in_channels % groups:
            groups -= 1
        out_groups = min(8, out_channels)
        while out_channels % out_groups:
            out_groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1.0 + scale[..., None]) + shift[..., None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, path_dim: int = POSE_DIM, context_dim: int = CONDITION_DIM,
                 base_channels: int = 64, condition_dim: int = 256) -> None:
        super().__init__()
        self.model_type = "unet"
        self.condition_dim = condition_dim
        self.time_mlp = nn.Sequential(nn.Linear(64, condition_dim), nn.SiLU(),
                                      nn.Linear(condition_dim, condition_dim))
        self.context_mlp = nn.Sequential(nn.Linear(context_dim, condition_dim), nn.SiLU(),
                                         nn.Linear(condition_dim, condition_dim))
        self.input = nn.Conv1d(path_dim, base_channels, 3, padding=1)
        self.enc1 = ResidualFiLM1D(base_channels, base_channels, condition_dim)
        self.down1 = nn.Conv1d(base_channels, 2 * base_channels, 4, stride=2, padding=1)
        self.enc2 = ResidualFiLM1D(2 * base_channels, 2 * base_channels, condition_dim)
        self.down2 = nn.Conv1d(2 * base_channels, 4 * base_channels, 4, stride=2, padding=1)
        self.mid1 = ResidualFiLM1D(4 * base_channels, 4 * base_channels, condition_dim)
        self.mid_attention_norm = nn.GroupNorm(8, 4 * base_channels)
        attention_heads = min(8, 4 * base_channels)
        while (4 * base_channels) % attention_heads:
            attention_heads -= 1
        self.mid_attention = nn.MultiheadAttention(
            4 * base_channels, attention_heads, batch_first=True
        )
        self.mid2 = ResidualFiLM1D(4 * base_channels, 4 * base_channels, condition_dim)
        self.up2 = nn.ConvTranspose1d(4 * base_channels, 2 * base_channels, 4, stride=2, padding=1)
        self.dec2 = ResidualFiLM1D(4 * base_channels, 2 * base_channels, condition_dim)
        self.up1 = nn.ConvTranspose1d(2 * base_channels, base_channels, 4, stride=2, padding=1)
        self.dec1 = ResidualFiLM1D(2 * base_channels, base_channels, condition_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(),
                                    nn.Conv1d(base_channels, path_dim, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, path: Tensor, timestep: Tensor, context: Tensor,
                obstacles: Tensor | None = None, obstacle_mask: Tensor | None = None) -> Tensor:
        del obstacles, obstacle_mask
        condition = self.time_mlp(timestep_embedding(timestep, 64)) + self.context_mlp(context)
        value = self.input(path.transpose(1, 2))
        skip1 = self.enc1(value, condition)
        skip2 = self.enc2(self.down1(skip1), condition)
        value = self.mid1(self.down2(skip2), condition)
        attention_input = self.mid_attention_norm(value).transpose(1, 2)
        value = value + self.mid_attention(
            attention_input, attention_input, attention_input,
            need_weights=False,
        )[0].transpose(1, 2)
        value = self.mid2(value, condition)
        value = self.dec2(torch.cat((self.up2(value), skip2), dim=1), condition)
        value = self.dec1(torch.cat((self.up1(value), skip1), dim=1), condition)
        return self.output(value).transpose(1, 2)


class DiTBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, condition_dim: int,
                 cross_attention: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.cross_attention = (
            nn.MultiheadAttention(dimension, heads, batch_first=True)
            if cross_attention else None
        )
        self.norm_cross = nn.LayerNorm(dimension, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dimension, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.GELU(),
                                 nn.Linear(4 * dimension, dimension))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 6 * dimension))

    def forward(self, value: Tensor, condition: Tensor,
                environment: Tensor | None, environment_mask: Tensor | None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, dim=-1)
        query = self.norm1(value) * (1.0 + scale1[:, None]) + shift1[:, None]
        attended = self.self_attention(query, query, query, need_weights=False)[0]
        value = value + gate1[:, None] * attended
        if self.cross_attention is not None:
            if environment is None:
                raise ValueError("cross-attention DiT requires environment tokens")
            query = self.norm_cross(value)
            attended = self.cross_attention(
                query, environment, environment,
                key_padding_mask=(~environment_mask if environment_mask is not None else None),
                need_weights=False,
            )[0]
            value = value + attended
        hidden = self.norm2(value) * (1.0 + scale2[:, None]) + shift2[:, None]
        return value + gate2[:, None] * self.mlp(hidden)


class ConditionalDiT(nn.Module):
    def __init__(self, sequence_length: int = 128, path_dim: int = POSE_DIM,
                 context_dim: int = CONDITION_DIM, dimension: int = 128,
                 depth: int = 4, heads: int = 4, cross_attention: bool = False) -> None:
        super().__init__()
        self.model_type = "dit_cross" if cross_attention else "dit"
        self.cross_attention = cross_attention
        self.dimension = dimension
        self.input = nn.Linear(path_dim, dimension)
        self.position_embedding = nn.Parameter(torch.randn(1, sequence_length, dimension) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(64, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.context_mlp = nn.Sequential(nn.Linear(context_dim, dimension), nn.SiLU(),
                                         nn.Linear(dimension, dimension))
        self.environment_mlp = (
            nn.Sequential(nn.Linear(OBSTACLE_DIM, dimension), nn.SiLU(),
                          nn.Linear(dimension, dimension))
            if cross_attention else None
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dimension, heads, dimension, cross_attention) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))
        self.output = nn.Linear(dimension, path_dim)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, path: Tensor, timestep: Tensor, context: Tensor,
                obstacles: Tensor | None = None, obstacle_mask: Tensor | None = None) -> Tensor:
        condition = self.time_mlp(timestep_embedding(timestep, 64)) + self.context_mlp(context)
        value = self.input(path) + self.position_embedding[:, :path.shape[1]]
        environment = self.environment_mlp(obstacles) if self.environment_mlp is not None else None
        for block in self.blocks:
            value = block(value, condition, environment, obstacle_mask)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        value = self.final_norm(value) * (1.0 + scale[:, None]) + shift[:, None]
        return self.output(value)


def build_model(model_type: str, sequence_length: int = 128,
                unet_channels: int = 64, dit_dimension: int = 128,
                dit_depth: int = 4, dit_heads: int = 4) -> nn.Module:
    if model_type == "unet":
        return ConditionalUNet1D(base_channels=unet_channels)
    if model_type in {"dit", "dit_cross"}:
        return ConditionalDiT(
            sequence_length=sequence_length, dimension=dit_dimension,
            depth=dit_depth, heads=dit_heads,
            cross_attention=model_type == "dit_cross",
        )
    raise ValueError(f"unknown model type: {model_type}")


@dataclass
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor

    @classmethod
    def cosine(cls, steps: int, device: torch.device) -> "DiffusionSchedule":
        offset = 0.008
        points = torch.linspace(0, steps, steps + 1, device=device)
        alpha_bar = torch.cos(((points / steps + offset) / (1 + offset)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        alphas = 1.0 - betas
        return cls(betas, alphas, torch.cumprod(alphas, dim=0))


def extract(values: Tensor, timesteps: Tensor, shape: Iterable[int]) -> Tensor:
    result = values.gather(0, timesteps)
    return result.reshape(len(timesteps), *((1,) * (len(tuple(shape)) - 1)))


def hard_clamp_endpoints(path: Tensor, context: Tensor) -> Tensor:
    result = path.clone()
    result[:, 0] = context[:, :POSE_DIM]
    result[:, -1] = context[:, POSE_DIM:]
    return result


@dataclass
class GuidanceConfig:
    enabled: bool = False
    start_fraction: float = 0.40
    scale: float = 0.020
    steps_per_diffusion_step: int = 2
    max_perturbation: float = 0.12
    collision_weight: float = 8.0
    smoothness_weight: float = 0.8
    length_weight: float = 0.10
    bounds_weight: float = 2.0
    # The primitive SAT proxy is conservative relative to COAL. Validation
    # calibration found that 6 cm proxy clearance best targets the exact 8 cm
    # planning margin without pushing paths off the demonstrated manifold.
    clearance_m: float = 0.06
    temperature_m: float = 0.06


# These primitives mirror the active collision elements on ``base_link`` in
# HDJQR-0102-0055.SLDASM.urdf.  Keeping the primitive types matters: the old
# five-large-sphere approximation marked most expert paths as colliding and
# therefore supplied a systematically wrong guidance gradient.
ROBOT_BOX_CENTERS = ((0.00318, -0.00008, -0.03412),)
ROBOT_BOX_HALF_SIZES = ((0.49607 / 2.0, 0.23920 / 2.0, 0.22884 / 2.0),)
ROBOT_BOX_RPY = ((0.0, 0.0, 0.0),)

ROBOT_SPHERE_CENTERS = (
    (0.00343, 0.40827, -0.00572),
    (0.00343, -0.40827, -0.00572),
)
ROBOT_SPHERE_RADII = (0.23920, 0.23920)

# center xyz, local rpy, radius, full length
ROBOT_CYLINDERS = (
    ((0.11595, 0.18060, -0.19515), (0.03489, 0.44007, -2.09440), 0.02193, 0.23920),
    ((0.11577, -0.18087, -0.19514), (-0.03031, 0.44007, 2.09205), 0.02193, 0.23920),
    ((-0.34587, -0.00008, -0.19523), (0.0, 0.44007, 0.0), 0.02193, 0.23920),
    ((-0.76800, 0.0, 0.00340), (0.0, 0.0, 0.0), 0.19934, 0.05980),
)


def _rpy_matrix_torch(rpy: Tensor) -> Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ), dim=-1).reshape(rpy.shape[:-1] + (3, 3))


def _quaternion_to_matrix_torch(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


def _support_separation(
    difference: Tensor,
    axis: Tensor,
    rotation_first: Tensor,
    half_first: Tensor,
    rotation_second: Tensor,
    half_second: Tensor,
) -> Tensor:
    """Signed separating-axis gap for two OBBs (positive means separated)."""

    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    valid = norm[..., 0] > 1e-6
    unit = axis / torch.clamp(norm, min=1e-6)
    center_projection = torch.abs((difference * unit).sum(dim=-1))
    first_projection = (
        torch.abs(torch.einsum("...ij,...i->...j", rotation_first, unit))
        * half_first
    ).sum(dim=-1)
    second_projection = (
        torch.abs(torch.einsum("...ij,...i->...j", rotation_second, unit))
        * half_second
    ).sum(dim=-1)
    separation = center_projection - first_projection - second_projection
    return torch.where(valid, separation, torch.full_like(separation, -1e6))


def _obb_box_signed_separation(
    centers: Tensor,
    rotations: Tensor,
    half_sizes: Tensor,
    obstacle_centers: Tensor,
    obstacle_rotations: Tensor,
    obstacle_half_sizes: Tensor,
) -> Tensor:
    """SAT signed separation for moving OBBs against static obstacle OBBs."""

    outputs = []
    for shape_index in range(centers.shape[2]):
        shape_outputs = []
        first_center = centers[:, :, shape_index]
        first_rotation = rotations[:, :, shape_index]
        first_half = half_sizes[shape_index]
        for obstacle_index in range(len(obstacle_centers)):
            second_center = obstacle_centers[obstacle_index]
            second_rotation = obstacle_rotations[obstacle_index]
            second_half = obstacle_half_sizes[obstacle_index]
            difference = first_center - second_center
            axes = [first_rotation[..., :, index] for index in range(3)]
            axes.extend(second_rotation[:, index].expand_as(first_center) for index in range(3))
            axes.extend(
                torch.cross(
                    first_rotation[..., :, first_axis],
                    second_rotation[:, second_axis].expand_as(first_center),
                    dim=-1,
                )
                for first_axis in range(3)
                for second_axis in range(3)
            )
            shape_outputs.append(torch.stack([
                _support_separation(
                    difference, axis, first_rotation, first_half,
                    second_rotation.expand_as(first_rotation), second_half,
                )
                for axis in axes
            ], dim=-1).amax(dim=-1))
        outputs.append(torch.stack(shape_outputs, dim=-1))
    return torch.stack(outputs, dim=2)


def _cylinder_box_signed_separation(
    centers: Tensor,
    axes: Tensor,
    radii: Tensor,
    half_lengths: Tensor,
    obstacle_centers: Tensor,
    obstacle_rotations: Tensor,
    obstacle_half_sizes: Tensor,
) -> Tensor:
    """Differentiable finite-cylinder/OBB SAT approximation.

    The support function of the cylinder is exact for every tested axis.  The
    tested obstacle normals, cylinder axis, and edge cross-products give a much
    tighter proxy than a circumscribed sphere, especially for the thin tail
    rotor disk.
    """

    outputs = []
    for shape_index in range(centers.shape[2]):
        shape_outputs = []
        center = centers[:, :, shape_index]
        cylinder_axis = axes[:, :, shape_index]
        radius = radii[shape_index]
        half_length = half_lengths[shape_index]
        for obstacle_index in range(len(obstacle_centers)):
            obstacle_center = obstacle_centers[obstacle_index]
            obstacle_rotation = obstacle_rotations[obstacle_index]
            obstacle_half = obstacle_half_sizes[obstacle_index]
            difference = center - obstacle_center
            candidates = [
                obstacle_rotation[:, index].expand_as(center) for index in range(3)
            ]
            candidates.append(cylinder_axis)
            candidates.extend(
                torch.cross(
                    cylinder_axis,
                    obstacle_rotation[:, index].expand_as(center),
                    dim=-1,
                )
                for index in range(3)
            )
            separations = []
            for candidate in candidates:
                norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
                valid = norm[..., 0] > 1e-6
                unit = candidate / torch.clamp(norm, min=1e-6)
                axis_dot = torch.clamp(
                    torch.abs((cylinder_axis * unit).sum(dim=-1)), 0.0, 1.0
                )
                cylinder_support = (
                    half_length * axis_dot
                    + radius * torch.sqrt(torch.clamp(
                        1.0 - axis_dot.square(), min=1e-8
                    ))
                )
                obstacle_support = (
                    torch.abs(torch.einsum("ij,...i->...j", obstacle_rotation, unit))
                    * obstacle_half
                ).sum(dim=-1)
                separation = (
                    torch.abs((difference * unit).sum(dim=-1))
                    - cylinder_support - obstacle_support
                )
                separations.append(torch.where(
                    valid, separation, torch.full_like(separation, -1e6)
                ))
            shape_outputs.append(torch.stack(separations, dim=-1).amax(dim=-1))
        outputs.append(torch.stack(shape_outputs, dim=-1))
    return torch.stack(outputs, dim=2)


def robot_obstacle_signed_separations(path: Tensor, obstacles: Tensor,
                                       obstacle_mask: Tensor) -> Tensor:
    """Return per-pose signed proxy clearances for the URDF primitives."""

    position = path[..., :3]
    root_rotation = rotation_6d_to_matrix(path[..., 3:9])
    active = obstacles[obstacle_mask]
    obstacle_centers = active[:, :3]
    obstacle_half_sizes = 0.5 * active[:, 3:6]
    obstacle_rotations = _quaternion_to_matrix_torch(active[:, 6:10])

    sphere_centers = torch.as_tensor(
        ROBOT_SPHERE_CENTERS, dtype=path.dtype, device=path.device
    )
    sphere_radii = torch.as_tensor(
        ROBOT_SPHERE_RADII, dtype=path.dtype, device=path.device
    )
    world_spheres = position[:, :, None, :] + torch.einsum(
        "blij,sj->blsi", root_rotation, sphere_centers
    )
    relative = world_spheres[:, :, :, None, :] - obstacle_centers[None, None, None]
    local = torch.einsum("blsoi,oij->blsoj", relative, obstacle_rotations)
    box_delta = torch.abs(local) - obstacle_half_sizes[None, None, None]
    outside = torch.linalg.vector_norm(F.relu(box_delta), dim=-1)
    inside = torch.minimum(box_delta.amax(dim=-1), torch.zeros_like(outside))
    sphere_signed = outside + inside - sphere_radii[None, None, :, None]

    box_centers_local = torch.as_tensor(
        ROBOT_BOX_CENTERS, dtype=path.dtype, device=path.device
    )
    box_half_sizes = torch.as_tensor(
        ROBOT_BOX_HALF_SIZES, dtype=path.dtype, device=path.device
    )
    box_local_rotations = _rpy_matrix_torch(torch.as_tensor(
        ROBOT_BOX_RPY, dtype=path.dtype, device=path.device
    ))
    world_box_centers = position[:, :, None, :] + torch.einsum(
        "blij,sj->blsi", root_rotation, box_centers_local
    )
    world_box_rotations = torch.einsum(
        "blij,sjk->blsik", root_rotation, box_local_rotations
    )
    box_signed = _obb_box_signed_separation(
        world_box_centers, world_box_rotations, box_half_sizes,
        obstacle_centers, obstacle_rotations, obstacle_half_sizes,
    )

    cylinder_centers_local = torch.as_tensor(
        [item[0] for item in ROBOT_CYLINDERS], dtype=path.dtype, device=path.device
    )
    cylinder_local_rotations = _rpy_matrix_torch(torch.as_tensor(
        [item[1] for item in ROBOT_CYLINDERS], dtype=path.dtype, device=path.device
    ))
    cylinder_local_axes = cylinder_local_rotations[..., :, 2]
    cylinder_radii = torch.as_tensor(
        [item[2] for item in ROBOT_CYLINDERS], dtype=path.dtype, device=path.device
    )
    cylinder_half_lengths = 0.5 * torch.as_tensor(
        [item[3] for item in ROBOT_CYLINDERS], dtype=path.dtype, device=path.device
    )
    world_cylinder_centers = position[:, :, None, :] + torch.einsum(
        "blij,sj->blsi", root_rotation, cylinder_centers_local
    )
    world_cylinder_axes = torch.einsum(
        "blij,sj->blsi", root_rotation, cylinder_local_axes
    )
    cylinder_signed = _cylinder_box_signed_separation(
        world_cylinder_centers, world_cylinder_axes,
        cylinder_radii, cylinder_half_lengths,
        obstacle_centers, obstacle_rotations, obstacle_half_sizes,
    )
    return torch.cat((sphere_signed, box_signed, cylinder_signed), dim=2)


def guidance_cost(path_normalized: Tensor, mean: Tensor, std: Tensor,
                  obstacles: Tensor, obstacle_mask: Tensor,
                  bounds_min: Tensor, bounds_max: Tensor,
                  config: GuidanceConfig) -> Tensor:
    path = path_normalized * std + mean
    position = path[..., :3]
    signed = robot_obstacle_signed_separations(path, obstacles, obstacle_mask)
    minimum_signed = signed.amin(dim=(2, 3))
    violation = F.softplus(
        (config.clearance_m - minimum_signed) / config.temperature_m
    ) * config.temperature_m
    collision = violation.square().mean(dim=1)
    acceleration = position[:, 2:] - 2 * position[:, 1:-1] + position[:, :-2]
    smoothness = acceleration.square().mean(dim=(1, 2))
    segment = position[:, 1:] - position[:, :-1]
    length = segment.square().sum(dim=-1).mean(dim=1)
    below = F.relu(bounds_min[None, None] - position)
    above = F.relu(position - bounds_max[None, None])
    bounds = (below.square() + above.square()).mean(dim=(1, 2))
    return (
        config.collision_weight * collision
        + config.smoothness_weight * smoothness
        + config.length_weight * length
        + config.bounds_weight * bounds
    )


def guided_prediction(
    predicted_x0: Tensor,
    context: Tensor,
    mean: Tensor,
    std: Tensor,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    bounds_min: Tensor,
    bounds_max: Tensor,
    config: GuidanceConfig,
) -> Tensor:
    original = predicted_x0.detach()
    current = original
    for _ in range(config.steps_per_diffusion_step):
        current = current.detach().requires_grad_(True)
        cost = guidance_cost(
            current, mean, std, obstacles, obstacle_mask,
            bounds_min, bounds_max, config,
        ).sum()
        gradient = torch.autograd.grad(cost, current)[0]
        gradient[:, 0] = 0.0
        gradient[:, -1] = 0.0
        rms = torch.sqrt(gradient.square().mean(dim=(1, 2), keepdim=True) + 1e-10)
        current = current - config.scale * gradient / rms
        delta = torch.clamp(
            current - original, -config.max_perturbation, config.max_perturbation
        )
        current = hard_clamp_endpoints(original + delta, context)
    return current.detach()


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    conditions: Tensor,
    schedule: DiffusionSchedule,
    sequence_length: int,
    sampling_steps: int,
    obstacles: Tensor,
    obstacle_mask: Tensor,
    mean: Tensor,
    std: Tensor,
    bounds_min: Tensor,
    bounds_max: Tensor,
    guidance: GuidanceConfig,
    generator: torch.Generator,
    clip_x0: float = 4.0,
) -> Tensor:
    device = conditions.device
    path = torch.randn(
        (len(conditions), sequence_length, POSE_DIM),
        device=device, generator=generator,
    )
    path = hard_clamp_endpoints(path, conditions)
    indices = torch.linspace(
        len(schedule.betas) - 1, 0, sampling_steps, device=device
    ).round().long().unique_consecutive()
    for sample_index, timestep_value in enumerate(indices):
        timestep = torch.full(
            (len(path),), int(timestep_value), device=device, dtype=torch.long
        )
        predicted_velocity = model(
            path, timestep, conditions,
            obstacles[None].expand(len(path), -1, -1),
            obstacle_mask[None].expand(len(path), -1),
        )
        alpha_bar = schedule.alpha_bars[timestep_value]
        # Velocity prediction stays well-conditioned even at the near-zero
        # signal-to-noise end of the cosine schedule.
        predicted_x0 = (
            torch.sqrt(alpha_bar) * path
            - torch.sqrt(1.0 - alpha_bar) * predicted_velocity
        )
        predicted_noise = (
            torch.sqrt(1.0 - alpha_bar) * path
            + torch.sqrt(alpha_bar) * predicted_velocity
        )
        predicted_x0 = torch.clamp(predicted_x0, -clip_x0, clip_x0)
        predicted_x0 = hard_clamp_endpoints(predicted_x0, conditions)
        predicted_noise = (
            path - torch.sqrt(alpha_bar) * predicted_x0
        ) / torch.clamp(torch.sqrt(1.0 - alpha_bar), min=1e-6)
        progress = sample_index / max(len(indices) - 1, 1)
        if guidance.enabled and progress >= (1.0 - guidance.start_fraction):
            with torch.enable_grad():
                predicted_x0 = guided_prediction(
                    predicted_x0, conditions, mean, std,
                    obstacles, obstacle_mask, bounds_min, bounds_max, guidance,
                )
            predicted_x0 = torch.clamp(predicted_x0, -clip_x0, clip_x0)
            predicted_x0 = hard_clamp_endpoints(predicted_x0, conditions)
            predicted_noise = (
                path - torch.sqrt(alpha_bar) * predicted_x0
            ) / torch.sqrt(1.0 - alpha_bar)
        if sample_index == len(indices) - 1:
            path = predicted_x0
            continue
        previous = indices[sample_index + 1]
        previous_alpha_bar = schedule.alpha_bars[previous]
        path = (
            torch.sqrt(previous_alpha_bar) * predicted_x0
            + torch.sqrt(1.0 - previous_alpha_bar) * predicted_noise
        )
        path = hard_clamp_endpoints(path, conditions)
    return hard_clamp_endpoints(path, conditions)
