"""B-spline control-point representation for SE(3) diffusion.

MPD-style: the model generates a small number of B-spline control points and
the trajectory is the spline evaluated on a dense grid, so raw samples are
smooth by construction.  Positions and the 6D rotation coordinates are
treated as plain coordinates; evaluation projects the 9D representation back
to valid SE(3) poses with :func:`se3_diffusion.pose9_to_pose7_numpy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch


def uniform_clamped_knots(
    n_control_points: int, degree: int
) -> np.ndarray:
    """Clamped uniform knots for ``n_control_points`` and ``degree``."""

    interior = np.linspace(0.0, 1.0, n_control_points - degree + 1)
    return np.concatenate(
        (
            np.zeros(degree),
            interior,
            np.ones(degree),
        )
    ).astype(np.float64)


@lru_cache(maxsize=16)
def _basis_matrix_cached(
    n_control_points: int,
    degree: int,
    n_points: int,
) -> np.ndarray:
    """(n_points, n_control_points) Cox-de Boor basis on ``linspace(0,1,n)``."""

    knots = uniform_clamped_knots(n_control_points, degree)
    u = np.linspace(0.0, 1.0, n_points)
    basis = np.zeros((n_points, n_control_points), dtype=np.float64)
    for point_index in range(n_points):
        # Nonzero basis functions at u are those in [knots[i], knots[i+p+1]).
        for control_index in range(n_control_points):
            basis[point_index, control_index] = _cox_de_boor(
                control_index, degree, u[point_index], knots
            )
    # Basis sum should be exactly one on the clamped domain.
    sums = basis.sum(axis=1)
    basis /= np.maximum(sums[:, None], 1e-12)
    return basis


def basis_matrix(
    n_control_points: int,
    degree: int,
    n_points: int,
) -> np.ndarray:
    """Cached (n_points, n_control_points) Cox-de Boor basis."""

    return _basis_matrix_cached(n_control_points, degree, n_points).copy()


def _cox_de_boor(
    index: int, degree: int, u: float, knots: np.ndarray
) -> float:
    if degree == 0:
        return 1.0 if knots[index] <= u < knots[index + 1] else 0.0
    denominator_left = knots[index + degree] - knots[index]
    denominator_right = knots[index + degree + 1] - knots[index + 1]
    left = 0.0
    if denominator_left > 1e-12:
        left = (u - knots[index]) / denominator_left * _cox_de_boor(
            index, degree - 1, u, knots
        )
    right = 0.0
    if denominator_right > 1e-12:
        right = (knots[index + degree + 1] - u) / denominator_right * (
            _cox_de_boor(index + 1, degree - 1, u, knots)
        )
    return left + right


def fit_control_points(
    points: np.ndarray,
    n_control_points: int = 22,
    degree: int = 5,
    n_points: int = 128,
    fixed_endpoints: bool = True,
) -> np.ndarray:
    """Least-squares B-spline fit with hard start/goal control points.

    ``points`` has shape (N, D).  Returns control points (n_control_points, D)
    whose first/last rows equal ``points[0]``/``points[-1]`` when
    ``fixed_endpoints`` is true.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (N, D)")
    basis = basis_matrix(n_control_points, degree, n_points)
    if not fixed_endpoints:
        control, _, _, _ = np.linalg.lstsq(
            basis, points, rcond=None
        )
        return control.T
    interior = basis[:, 1:-1]
    boundary = basis[:, [0, -1]]
    free, _, _, _ = np.linalg.lstsq(
        interior,
        points - boundary @ points[[0, -1]],
        rcond=None,
    )
    control = np.empty((n_control_points, points.shape[1]), dtype=np.float64)
    control[0] = points[0]
    control[-1] = points[-1]
    control[1:-1] = free
    return control


def evaluate_control_points(
    control_points: np.ndarray,
    n_points: int = 128,
    degree: int = 5,
) -> np.ndarray:
    """Evaluate control points (n_cp, D) on the dense grid (n_points, D)."""

    control_points = np.asarray(control_points, dtype=np.float64)
    n_control_points = control_points.shape[0]
    basis = basis_matrix(n_control_points, degree, n_points)
    return basis @ control_points


def evaluate_torch(
    control_points: torch.Tensor,
    n_points: int,
    degree: int = 5,
) -> torch.Tensor:
    """Differentiable evaluation: returns (B, n_points, D)."""

    n_control_points = control_points.shape[-2]
    basis = torch.from_numpy(
        _basis_matrix_cached(n_control_points, degree, n_points)
    ).to(dtype=control_points.dtype, device=control_points.device)
    return torch.einsum("bcd,nc->bnd", control_points, basis)


@dataclass(frozen=True)
class BsplineConfig:
    n_control_points: int = 22
    degree: int = 5
    n_points: int = 128
