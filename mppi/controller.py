"""Vectorized Model Predictive Path Integral controller.

The implementation is deliberately independent of MuJoCo.  A system only needs
to provide a vectorized ``rollout`` method and a trajectory-cost object, which
makes the controller reusable in simulations and on a real flight stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class RolloutDynamics(Protocol):
    """Interface required by :class:`MPPIController`."""

    state_dim: int
    control_dim: int

    def rollout(
        self, initial_state: FloatArray, controls: FloatArray
    ) -> FloatArray:
        """Roll out ``controls`` with shape ``(batch, horizon, control_dim)``."""


class TrajectoryCost(Protocol):
    """Interface required by :class:`MPPIController`."""

    def trajectory_cost(
        self,
        states: FloatArray,
        controls: FloatArray,
        reference: FloatArray,
    ) -> FloatArray:
        """Return one scalar cost per sampled trajectory."""


@dataclass(frozen=True)
class MPPIConfig:
    """Numerical parameters and limits for MPPI.

    ``noise_sigma`` is the exploration standard deviation for each control
    dimension.  ``noise_correlation`` adds temporal correlation, producing
    dynamically smoother trajectory samples.
    """

    horizon: int = 40
    num_samples: int = 512
    temperature: float = 80.0
    noise_sigma: tuple[float, ...] = (2.2, 2.2, 1.8)
    control_min: tuple[float, ...] = (-4.0, -4.0, -3.5)
    control_max: tuple[float, ...] = (4.0, 4.0, 3.5)
    noise_correlation: float = 0.55
    likelihood_ratio_weight: float = 0.08
    action_continuity_weight: float = 2.0
    control_smoothing: float = 0.20
    num_iterations: int = 1
    seed: int | None = 7


@dataclass(frozen=True)
class MPPIResult:
    """Output of one receding-horizon MPPI update."""

    action: FloatArray
    nominal_controls: FloatArray
    nominal_states: FloatArray
    sampled_controls: FloatArray
    sampled_states: FloatArray
    costs: FloatArray
    weights: FloatArray
    best_index: int
    effective_sample_size: float


class MPPIController:
    """Sampling-based receding-horizon controller.

    The controller maintains and shifts a nominal control sequence internally.
    Call :meth:`command` once per outer-loop control period.
    """

    def __init__(
        self,
        dynamics: RolloutDynamics,
        cost: TrajectoryCost,
        config: MPPIConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.cost = cost
        self.config = config or MPPIConfig()
        self._validate_config()

        self._sigma = np.asarray(self.config.noise_sigma, dtype=np.float64)
        self._control_min = np.asarray(
            self.config.control_min, dtype=np.float64
        )
        self._control_max = np.asarray(
            self.config.control_max, dtype=np.float64
        )
        self._rng = rng or np.random.default_rng(self.config.seed)
        self._nominal_controls = np.zeros(
            (self.config.horizon, self.dynamics.control_dim), dtype=np.float64
        )
        self._previous_action = np.zeros(
            self.dynamics.control_dim, dtype=np.float64
        )

    @property
    def nominal_controls(self) -> FloatArray:
        """A copy of the warm-start sequence stored for the next update."""

        return self._nominal_controls.copy()

    def reset(
        self,
        controls: ArrayLike | None = None,
        previous_action: ArrayLike | None = None,
    ) -> None:
        """Clear/replace the warm start and its previous-action anchor."""

        if controls is None:
            self._nominal_controls.fill(0.0)
        else:
            controls_array = np.asarray(controls, dtype=np.float64)
            if controls_array.shape != self._nominal_controls.shape:
                raise ValueError(
                    "controls must have shape "
                    f"{self._nominal_controls.shape}, got "
                    f"{controls_array.shape}"
                )
            self._nominal_controls[:] = np.clip(
                controls_array, self._control_min, self._control_max
            )

        if previous_action is None:
            self._previous_action.fill(0.0)
        else:
            previous_action_array = np.asarray(
                previous_action, dtype=np.float64
            )
            if previous_action_array.shape != (
                self.dynamics.control_dim,
            ):
                raise ValueError(
                    "previous_action must have shape "
                    f"({self.dynamics.control_dim},)"
                )
            self._previous_action[:] = np.clip(
                previous_action_array, self._control_min, self._control_max
            )

    def command(
        self, state: ArrayLike, reference: ArrayLike
    ) -> MPPIResult:
        """Optimize a control sequence and return its first control.

        Args:
            state: Current state with shape ``(state_dim,)``.
            reference: Desired state trajectory with shape
                ``(horizon + 1, state_dim)``.
        """

        initial_state = np.asarray(state, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        self._validate_inputs(initial_state, reference_array)

        sampled_controls: FloatArray | None = None
        sampled_states: FloatArray | None = None
        costs: FloatArray | None = None
        weights: FloatArray | None = None

        for _ in range(self.config.num_iterations):
            noise = self._sample_noise()
            sampled_controls = np.clip(
                self._nominal_controls[None, :, :] + noise,
                self._control_min,
                self._control_max,
            )
            # Saturation changes the actual perturbation used by MPPI.
            effective_noise = (
                sampled_controls - self._nominal_controls[None, :, :]
            )
            sampled_states = self.dynamics.rollout(
                initial_state, sampled_controls
            )
            costs = self.cost.trajectory_cost(
                sampled_states, sampled_controls, reference_array
            )
            if self.config.action_continuity_weight > 0.0:
                first_action_delta = (
                    sampled_controls[:, 0, :] - self._previous_action
                )
                costs += self.config.action_continuity_weight * np.sum(
                    np.square(first_action_delta), axis=1
                )

            # Path-integral likelihood-ratio control cost.
            cross_cost = (
                self.config.likelihood_ratio_weight
                * self.config.temperature
                * np.sum(
                    self._nominal_controls[None, :, :]
                    * effective_noise
                    / (self._sigma[None, None, :] ** 2),
                    axis=(1, 2),
                )
            )
            costs = costs + cross_cost
            weights = self._importance_weights(costs)

            update = np.einsum(
                "k,khu->hu", weights, effective_noise, optimize=True
            )
            self._nominal_controls[:] = np.clip(
                self._nominal_controls + update,
                self._control_min,
                self._control_max,
            )

        assert sampled_controls is not None
        assert sampled_states is not None
        assert costs is not None
        assert weights is not None

        optimized_controls = self._smooth_control_sequence(
            self._nominal_controls
        )
        self._nominal_controls[:] = optimized_controls
        nominal_states = self.dynamics.rollout(
            initial_state, optimized_controls[None, :, :]
        )[0]
        action = optimized_controls[0].copy()
        self._previous_action[:] = action
        best_index = int(np.argmin(costs))
        effective_sample_size = float(1.0 / np.sum(np.square(weights)))

        # Standard receding-horizon warm start for the next command.
        self._nominal_controls[:-1] = optimized_controls[1:]
        self._nominal_controls[-1] = optimized_controls[-1]

        return MPPIResult(
            action=action,
            nominal_controls=optimized_controls,
            nominal_states=nominal_states,
            sampled_controls=sampled_controls,
            sampled_states=sampled_states,
            costs=costs,
            weights=weights,
            best_index=best_index,
            effective_sample_size=effective_sample_size,
        )

    def _smooth_control_sequence(
        self, controls: FloatArray
    ) -> FloatArray:
        """Causal low-pass smoothing anchored to the executed last action."""

        smoothing = self.config.control_smoothing
        if smoothing <= 0.0:
            return controls.copy()

        smoothed = np.empty_like(controls)
        anchor = self._previous_action
        for index in range(self.config.horizon):
            smoothed[index] = (
                smoothing * anchor + (1.0 - smoothing) * controls[index]
            )
            anchor = smoothed[index]
        return np.clip(smoothed, self._control_min, self._control_max)

    def _sample_noise(self) -> FloatArray:
        shape = (
            self.config.num_samples,
            self.config.horizon,
            self.dynamics.control_dim,
        )
        white_noise = self._rng.normal(size=shape)
        correlation = self.config.noise_correlation

        if correlation > 0.0:
            colored_noise = np.empty_like(white_noise)
            colored_noise[:, 0, :] = white_noise[:, 0, :]
            innovation_scale = np.sqrt(1.0 - correlation**2)
            for index in range(1, self.config.horizon):
                colored_noise[:, index, :] = (
                    correlation * colored_noise[:, index - 1, :]
                    + innovation_scale * white_noise[:, index, :]
                )
            white_noise = colored_noise

        noise = white_noise * self._sigma[None, None, :]
        # Always retain one unperturbed rollout for a stable cost baseline.
        noise[0] = 0.0
        return noise

    def _importance_weights(self, costs: FloatArray) -> FloatArray:
        if not np.all(np.isfinite(costs)):
            raise FloatingPointError("MPPI received a non-finite trajectory cost")
        shifted = costs - np.min(costs)
        exponent = np.clip(
            -shifted / self.config.temperature, -700.0, 0.0
        )
        unnormalized = np.exp(exponent)
        denominator = np.sum(unnormalized)
        if not np.isfinite(denominator) or denominator <= 0.0:
            weights = np.zeros_like(costs)
            weights[int(np.argmin(costs))] = 1.0
            return weights
        return unnormalized / denominator

    def _validate_config(self) -> None:
        config = self.config
        control_dim = self.dynamics.control_dim
        if config.horizon < 1:
            raise ValueError("horizon must be positive")
        if config.num_samples < 2:
            raise ValueError("num_samples must be at least two")
        if config.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if config.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if config.likelihood_ratio_weight < 0.0:
            raise ValueError("likelihood_ratio_weight must be non-negative")
        if config.action_continuity_weight < 0.0:
            raise ValueError("action_continuity_weight must be non-negative")
        if not 0.0 <= config.control_smoothing < 1.0:
            raise ValueError("control_smoothing must be in [0, 1)")
        if not 0.0 <= config.noise_correlation < 1.0:
            raise ValueError("noise_correlation must be in [0, 1)")

        for name, values in (
            ("noise_sigma", config.noise_sigma),
            ("control_min", config.control_min),
            ("control_max", config.control_max),
        ):
            if len(values) != control_dim:
                raise ValueError(
                    f"{name} must contain {control_dim} values"
                )

        sigma = np.asarray(config.noise_sigma)
        lower = np.asarray(config.control_min)
        upper = np.asarray(config.control_max)
        if np.any(sigma <= 0.0):
            raise ValueError("all noise_sigma values must be positive")
        if np.any(lower >= upper):
            raise ValueError("each control_min must be below control_max")

    def _validate_inputs(
        self, state: FloatArray, reference: FloatArray
    ) -> None:
        expected_state_shape = (self.dynamics.state_dim,)
        expected_reference_shape = (
            self.config.horizon + 1,
            self.dynamics.state_dim,
        )
        if state.shape != expected_state_shape:
            raise ValueError(
                f"state must have shape {expected_state_shape}, got {state.shape}"
            )
        if reference.shape != expected_reference_shape:
            raise ValueError(
                "reference must have shape "
                f"{expected_reference_shape}, got {reference.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(
            np.isfinite(reference)
        ):
            raise ValueError("state and reference must contain finite values")


class ResidualMPPIController(MPPIController):
    """MPPI correction around a time-varying feedforward control sequence.

    The optimized command is ``u = u_ff + delta_u``.  Only the residual
    sequence is shifted between receding-horizon updates, so an analytic
    trajectory feedforward remains the nominal solution while MPPI supplies
    state-dependent tracking and obstacle-avoidance corrections.
    """

    def __init__(
        self,
        dynamics: RolloutDynamics,
        cost: TrajectoryCost,
        config: MPPIConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(dynamics, cost, config, rng)
        self._nominal_residuals = np.zeros_like(
            self._nominal_controls
        )
        self._previous_residual_action = np.zeros(
            self.dynamics.control_dim, dtype=np.float64
        )

    @property
    def nominal_residuals(self) -> FloatArray:
        """Return the warm-start correction stored for the next update."""

        return self._nominal_residuals.copy()

    def reset_residuals(
        self,
        residuals: ArrayLike | None = None,
        previous_residual_action: ArrayLike | None = None,
        previous_action: ArrayLike | None = None,
    ) -> None:
        """Clear or replace residual warm starts and continuity anchors."""

        if residuals is None:
            self._nominal_residuals.fill(0.0)
        else:
            residual_array = np.asarray(residuals, dtype=np.float64)
            if residual_array.shape != self._nominal_residuals.shape:
                raise ValueError(
                    "residuals must have shape "
                    f"{self._nominal_residuals.shape}"
                )
            if not np.all(np.isfinite(residual_array)):
                raise ValueError("residuals must contain finite values")
            self._nominal_residuals[:] = residual_array

        if previous_residual_action is None:
            self._previous_residual_action.fill(0.0)
        else:
            residual_action = np.asarray(
                previous_residual_action, dtype=np.float64
            )
            if residual_action.shape != (self.dynamics.control_dim,):
                raise ValueError(
                    "previous_residual_action must have shape "
                    f"({self.dynamics.control_dim},)"
                )
            self._previous_residual_action[:] = residual_action

        if previous_action is None:
            self._previous_action.fill(0.0)
        else:
            previous_action_array = np.asarray(
                previous_action, dtype=np.float64
            )
            if previous_action_array.shape != (
                self.dynamics.control_dim,
            ):
                raise ValueError(
                    "previous_action must have shape "
                    f"({self.dynamics.control_dim},)"
                )
            self._previous_action[:] = np.clip(
                previous_action_array,
                self._control_min,
                self._control_max,
            )

    def command(
        self,
        state: ArrayLike,
        reference: ArrayLike,
        feedforward_controls: ArrayLike,
    ) -> MPPIResult:
        """Optimize residuals around ``feedforward_controls``."""

        initial_state = np.asarray(state, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        feedforward = np.asarray(
            feedforward_controls, dtype=np.float64
        )
        self._validate_inputs(initial_state, reference_array)
        expected_controls_shape = (
            self.config.horizon,
            self.dynamics.control_dim,
        )
        if feedforward.shape != expected_controls_shape:
            raise ValueError(
                "feedforward_controls must have shape "
                f"{expected_controls_shape}, got {feedforward.shape}"
            )
        if not np.all(np.isfinite(feedforward)):
            raise ValueError(
                "feedforward_controls must contain finite values"
            )
        feedforward = np.clip(
            feedforward, self._control_min, self._control_max
        )

        sampled_controls: FloatArray | None = None
        sampled_states: FloatArray | None = None
        costs: FloatArray | None = None
        weights: FloatArray | None = None

        for _ in range(self.config.num_iterations):
            nominal_controls = np.clip(
                feedforward + self._nominal_residuals,
                self._control_min,
                self._control_max,
            )
            noise = self._sample_noise()
            sampled_controls = np.clip(
                nominal_controls[None, :, :] + noise,
                self._control_min,
                self._control_max,
            )
            effective_noise = (
                sampled_controls - nominal_controls[None, :, :]
            )
            sampled_states = self.dynamics.rollout(
                initial_state, sampled_controls
            )
            costs = self.cost.trajectory_cost(
                sampled_states, sampled_controls, reference_array
            )
            if self.config.action_continuity_weight > 0.0:
                first_action_delta = (
                    sampled_controls[:, 0, :] - self._previous_action
                )
                costs += self.config.action_continuity_weight * np.sum(
                    np.square(first_action_delta), axis=1
                )

            cross_cost = (
                self.config.likelihood_ratio_weight
                * self.config.temperature
                * np.sum(
                    self._nominal_residuals[None, :, :]
                    * effective_noise
                    / (self._sigma[None, None, :] ** 2),
                    axis=(1, 2),
                )
            )
            costs = costs + cross_cost
            weights = self._importance_weights(costs)
            update = np.einsum(
                "k,khu->hu", weights, effective_noise, optimize=True
            )
            updated_controls = np.clip(
                feedforward + self._nominal_residuals + update,
                self._control_min,
                self._control_max,
            )
            self._nominal_residuals[:] = (
                updated_controls - feedforward
            )

        assert sampled_controls is not None
        assert sampled_states is not None
        assert costs is not None
        assert weights is not None

        optimized_residuals = self._smooth_residual_sequence(
            self._nominal_residuals
        )
        optimized_controls = np.clip(
            feedforward + optimized_residuals,
            self._control_min,
            self._control_max,
        )
        optimized_residuals = optimized_controls - feedforward
        nominal_states = self.dynamics.rollout(
            initial_state, optimized_controls[None, :, :]
        )[0]
        action = optimized_controls[0].copy()
        self._previous_action[:] = action
        self._previous_residual_action[:] = optimized_residuals[0]
        best_index = int(np.argmin(costs))
        effective_sample_size = float(
            1.0 / np.sum(np.square(weights))
        )

        self._nominal_residuals[:-1] = optimized_residuals[1:]
        self._nominal_residuals[-1] = optimized_residuals[-1]

        return MPPIResult(
            action=action,
            nominal_controls=optimized_controls,
            nominal_states=nominal_states,
            sampled_controls=sampled_controls,
            sampled_states=sampled_states,
            costs=costs,
            weights=weights,
            best_index=best_index,
            effective_sample_size=effective_sample_size,
        )

    def _smooth_residual_sequence(
        self, residuals: FloatArray
    ) -> FloatArray:
        smoothing = self.config.control_smoothing
        if smoothing <= 0.0:
            return residuals.copy()

        smoothed = np.empty_like(residuals)
        anchor = self._previous_residual_action
        for index in range(self.config.horizon):
            smoothed[index] = (
                smoothing * anchor
                + (1.0 - smoothing) * residuals[index]
            )
            anchor = smoothed[index]
        return smoothed
