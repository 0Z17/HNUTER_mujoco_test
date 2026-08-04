"""GPU-batched cuRobo collision checks for SE(3) diffusion candidates.

The robot is represented by the conservative sphere set in
``etc/URDF-for-gazebo/config/HDJQR-0102-0055.SLDASM_curobo_spheres.yml`` and
the environment by its collision cuboids.  cuRobo evaluates, in one GPU
launch, the signed distance between every robot sphere and every obstacle for
whole candidate batches, so it can act as a very fast coarse filter in front
of the exact COAL audit.

Semantics of the returned scalar (see ``wp_collision_common.py``):

* ``cost == 0``  -> every sphere is at least ``activation_distance`` away from
  every obstacle;
* ``cost > 0``   -> at least one sphere is closer than ``activation_distance``.

Because the sphere set is a conservative superset of the URDF collision
primitives, ``cost == 0`` certifies at least ``activation_distance`` clearance
under the exact COAL geometry as well.  This module is CUDA-only: it must run
with the cuRobo Python environment, never with the plain project venv.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the checker needs the cuRobo env
    torch = None


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SPHERES = (
    PROJECT_DIR
    / "etc"
    / "URDF-for-gazebo"
    / "config"
    / "HDJQR-0102-0055.SLDASM_curobo_spheres.yml"
)


@dataclass(frozen=True)
class CuroboSphereSet:
    """Robot collision spheres in the base_link frame."""

    centers: np.ndarray  # (S, 3) metres
    radii: np.ndarray  # (S,) metres

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])


def load_curobo_spheres(path: Path | str) -> CuroboSphereSet:
    """Load the cuRobo sphere config YAML into numpy arrays."""

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spheres = payload["collision_spheres"]["base_link"]
    centers = np.asarray(
        [list(item["center"]) for item in spheres], dtype=np.float32
    )
    radii = np.asarray(
        [float(item["radius"]) for item in spheres], dtype=np.float32
    )
    if centers.ndim != 2 or centers.shape[1] != 3 or len(radii) != len(centers):
        raise ValueError(f"invalid cuRobo sphere config: {path}")
    return CuroboSphereSet(centers=centers, radii=radii)


def environment_collision_cuboids(environment: dict) -> list[dict]:
    """Return the collision boxes of an environment JSON payload."""

    boxes = [
        obstacle
        for obstacle in environment.get("obstacles", [])
        if obstacle.get("collision", False) and obstacle.get("type") == "box"
    ]
    if not boxes:
        raise ValueError("environment contains no collision boxes")
    return boxes


def _quaternion_to_matrix_batched(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Vectorised (..., 4) wxyz quaternions -> (..., 3, 3) rotation matrix."""

    w, x, y, z = (
        quaternion_wxyz[..., 0],
        quaternion_wxyz[..., 1],
        quaternion_wxyz[..., 2],
        quaternion_wxyz[..., 3],
    )
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion_wxyz.shape[:-1], 3, 3)


class CuroboBatchChecker:
    """GPU-batched sphere-to-world distance queries for one environment."""

    def __init__(
        self,
        environment: dict,
        sphere_set: CuroboSphereSet,
        device: str = "cuda:0",
        activation_distance: float = 0.08,
        max_distance: float = 1.0,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "CuroboBatchChecker requires the cuRobo Python environment "
                "(torch + curobo installed)"
            )
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.geom.collision.collision_scene import (
            SceneCollision,
            SceneCollisionCfg,
        )
        from curobo._src.robot.kinematics.kinematics_state import (
            KinematicsState,
        )
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo.scene import Cuboid, Scene

        self.device_cfg = DeviceCfg(
            device=device,
            dtype=torch.float32,
            collision_distance_dtype=torch.float32,
        )
        cuboids = [
            Cuboid(
                name=box["id"],
                dims=box["size_xyz"],
                pose=[
                    *box["pose"]["position"],
                    *box["pose"]["quaternion_wxyz"],
                ],
            )
            for box in environment_collision_cuboids(environment)
        ]
        self.scene = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=self.device_cfg,
                scene_model=Scene(cuboid=cuboids),
                num_envs=1,
                max_distance=float(max_distance),
            )
        )
        self._buffer_type = CollisionBuffer
        self._state_type = KinematicsState
        self.sphere_set = sphere_set
        self.device = device
        self.activation_distance = float(activation_distance)
        self._sphere_tensor: torch.Tensor | None = None
        self._weight = torch.ones(1, device=device, dtype=torch.float32)
        self._eta = torch.tensor(
            [self.activation_distance], device=device, dtype=torch.float32
        )
        self._eta_wide = torch.tensor([1.0], device=device, dtype=torch.float32)
        self._buffer_wide: object | None = None
        self._buffer: object | None = None

    def _world_spheres(self, poses_wxyz: torch.Tensor) -> torch.Tensor:
        """Map the base-link sphere set by (B, H, 7) wxyz poses."""

        if self._sphere_tensor is None:
            centers = torch.from_numpy(self.sphere_set.centers).to(self.device)
            radii = torch.from_numpy(self.sphere_set.radii).to(self.device)
            self._sphere_tensor = torch.cat((centers, radii[:, None]), dim=-1)
        positions = poses_wxyz[..., :3][..., None, :]
        rotations = _quaternion_to_matrix_batched(poses_wxyz[..., 3:7])
        centers = self._sphere_tensor[..., :3]
        radii = self._sphere_tensor[..., 3]
        batch, horizon = poses_wxyz.shape[0], poses_wxyz.shape[1]
        world = positions + torch.einsum(
            "...ij,sj->...si", rotations, centers
        )
        return torch.cat(
            (
                world,
                radii.expand(batch, horizon, -1)[..., None],
            ),
            dim=-1,
        )

    def _allocate_buffer(self, shape: tuple[int, ...]) -> object:
        if self._buffer is None or self._buffer.distance.shape != shape:
            self._buffer = self._buffer_type.from_shape(
                torch.Size(shape), self.device_cfg
            )
            self._buffer_wide = self._buffer_type.from_shape(
                torch.Size(shape), self.device_cfg
            )
        return self._buffer

    def query(
        self,
        poses_wxyz: np.ndarray,
        *,
        wide_eta: bool = False,
        return_gradients: bool = False,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Return per-(batch, horizon, sphere) collision cost on GPU.

        ``poses_wxyz`` has shape (B, H, 7).  With ``wide_eta=True`` the query
        uses a 1 m activation distance so the returned value is
        approximately ``clearance + 0.5 m`` for positive clearance, which is
        useful for reporting; the acceptance test always uses the narrow eta.
        """

        poses = torch.from_numpy(np.asarray(poses_wxyz, dtype=np.float32)).to(
            self.device
        )
        if poses.ndim != 3 or poses.shape[2] != 7:
            raise ValueError("poses_wxyz must have shape (B, H, 7)")
        results = []
        for begin in range(0, poses.shape[0], batch_size):
            chunk = self._world_spheres(poses[begin : begin + batch_size])
            buffer = self._allocate_buffer(chunk.shape)
            buffer.zero_()
            cost = self.scene.get_sphere_distance(
                self._state_type(robot_spheres=chunk),
                buffer,
                self._weight,
                self._eta_wide if wide_eta else self._eta,
            )
            results.append(cost)
        torch.cuda.synchronize(self.device)
        return torch.cat(results, dim=0)

    def max_collision_cost_per_path(self, poses_wxyz: np.ndarray) -> np.ndarray:
        """Return the worst narrow-eta collision cost per path (B,).

        A path is accepted by the coarse filter iff every node and every
        sphere has zero cost, i.e. iff the returned maximum is zero.  Any
        positive value means at least one sphere came closer than
        ``activation_distance`` to an obstacle at some node.
        """

        cost = self.query(poses_wxyz, wide_eta=False)
        return cost.amax(dim=(1, 2)).cpu().numpy()

    def path_clearance_estimate(self, poses_wxyz: np.ndarray) -> np.ndarray:
        """Return an approximate minimum sphere clearance per path (B,)."""

        cost = self.query(poses_wxyz, wide_eta=True)
        # With a 1 m activation distance, the collision activation is
        #   cost = 0.5 * (1 - clearance)^2      for 0 < clearance < 1 m
        #   cost = 0.5 - clearance              for clearance <= 0
        #   cost = 0                            for clearance >= 1 m
        # so the per-element clearance estimate is inverted piecewise and then
        # minimised over nodes and spheres.
        element_clearance = torch.where(
            cost >= 0.5,
            0.5 - cost,
            torch.where(
                cost > 0.0,
                1.0 - torch.sqrt(torch.clamp(2.0 * cost, min=0.0)),
                torch.full_like(cost, 1.0),
            ),
        )
        return element_clearance.amin(dim=(1, 2)).cpu().numpy()

    def close(self) -> None:
        self.scene.clear_cache()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cuRobo GPU batch collision costs for SE(3) diffusion "
            "candidate paths and write coarse acceptance flags."
        )
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--spheres", type=Path, default=DEFAULT_SPHERES)
    parser.add_argument("--activation-distance", type=float, default=0.08)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    sphere_set = load_curobo_spheres(args.spheres)
    with np.load(args.candidates) as payload:
        if "poses_wxyz" in payload:
            paths = payload["poses_wxyz"].astype(np.float32)
        elif "states" in payload:
            paths = payload["states"].astype(np.float32)
        else:
            raise ValueError(
                "candidates NPZ must contain poses_wxyz or states"
            )
    if paths.ndim == 2:
        paths = paths[np.newaxis, ...]
    if paths.ndim != 3 or paths.shape[2] != 7:
        raise ValueError("candidates must have shape (B, N, 7)")
    checker = CuroboBatchChecker(
        environment,
        sphere_set,
        device=args.device,
        activation_distance=args.activation_distance,
    )
    max_cost = checker.max_collision_cost_per_path(paths)
    clearance = checker.path_clearance_estimate(paths)
    checker.close()
    write_json(
        args.output,
        {
            "environment": str(args.environment.resolve()),
            "spheres": str(args.spheres.resolve()),
            "sphere_count": sphere_set.count,
            "activation_distance_m": args.activation_distance,
            "candidate_count": int(paths.shape[0]),
            "max_collision_cost": max_cost.tolist(),
            "coarse_accept": (max_cost <= 0.0).tolist(),
            "min_sphere_clearance_estimate_m": clearance.tolist(),
        },
    )
    print(
        f"CUROBO_FILTER accepted {int((max_cost <= 0.0).sum())}/"
        f"{len(max_cost)} candidates at {args.activation_distance} m"
    )


if __name__ == "__main__":
    main()
