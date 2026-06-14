"""Small optimization helpers built on JAX autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from multilayer_tmm.layers import Stack, stack_thicknesses, stack_with_thicknesses
from multilayer_tmm.resonance import (
    FeatureKind,
    ResonanceResult,
    analyze_resonance,
    smooth_resonance_metrics,
)
from multilayer_tmm.tmm import simulate_spectrum


Objective = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class OptimizationResult:
    """Result from a thickness optimization run."""

    thicknesses_nm: jnp.ndarray
    history: jnp.ndarray


@dataclass(frozen=True)
class ResonanceOptimizationResult:
    """Result from target-resonance thickness optimization."""

    thicknesses_nm: jnp.ndarray
    variable_thicknesses_nm: jnp.ndarray
    history: jnp.ndarray
    resonance: ResonanceResult


def mean_reflectivity(
    stack: Stack,
    wavelengths_nm: jnp.ndarray,
    angle_deg: float | None = None,
    angle_rad: float | None = None,
    polarization: str = "s",
) -> jnp.ndarray:
    """Return mean reflectivity over a wavelength band."""

    result = simulate_spectrum(
        stack,
        wavelengths_nm=wavelengths_nm,
        angle_deg=angle_deg,
        angle_rad=angle_rad,
        polarization=polarization,
    )
    return jnp.mean(result.R)


def optimize_thicknesses(
    objective: Objective,
    initial_thicknesses_nm: jnp.ndarray,
    steps: int = 100,
    learning_rate: float = 0.1,
    lower_bound_nm: float | None = 0.0,
) -> OptimizationResult:
    """Optimize thicknesses with Optax Adam if available, else gradient descent."""

    params = jnp.asarray(initial_thicknesses_nm)
    history = []

    try:
        import optax  # type: ignore

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        value_and_grad = jax.value_and_grad(objective)
        for _ in range(steps):
            value, gradients = value_and_grad(params)
            updates, opt_state = optimizer.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            if lower_bound_nm is not None:
                params = jnp.maximum(params, lower_bound_nm)
            history.append(value)
    except ImportError:
        value_and_grad = jax.value_and_grad(objective)
        for _ in range(steps):
            value, gradients = value_and_grad(params)
            params = params - learning_rate * gradients
            if lower_bound_nm is not None:
                params = jnp.maximum(params, lower_bound_nm)
            history.append(value)

    return OptimizationResult(
        thicknesses_nm=params,
        history=jnp.asarray(history),
    )


def resonance_target_loss(
    wavelength_nm: jnp.ndarray,
    spectrum_values: jnp.ndarray,
    target_wavelength_nm: float,
    target_q: float,
    feature: FeatureKind = "peak",
    wavelength_weight: float = 1.0,
    q_weight: float = 1.0,
    sharpness: float = 20.0,
) -> jnp.ndarray:
    """Differentiable loss for targeting resonance wavelength and Q."""

    metrics = smooth_resonance_metrics(
        wavelength_nm=wavelength_nm,
        spectrum=spectrum_values,
        feature=feature,
        sharpness=sharpness,
    )
    target_wavelength = jnp.asarray(target_wavelength_nm, dtype=metrics.resonance_wavelength_nm.dtype)
    target_quality = jnp.asarray(target_q, dtype=metrics.quality_factor.dtype)
    wavelength_error = (metrics.resonance_wavelength_nm - target_wavelength) / target_wavelength
    q_error = (metrics.quality_factor - target_quality) / target_quality
    return wavelength_weight * wavelength_error**2 + q_weight * q_error**2


def optimize_resonance_target(
    stack: Stack,
    wavelengths_nm: jnp.ndarray,
    target_wavelength_nm: float,
    target_q: float,
    spectrum: str = "R",
    feature: FeatureKind = "peak",
    variable_layer_indices: tuple[int, ...] | None = None,
    angle_deg: float | None = 0.0,
    angle_rad: float | None = None,
    polarization: str = "s",
    steps: int = 100,
    learning_rate: float = 0.1,
    lower_bound_nm: float | None = 0.0,
    wavelength_weight: float = 1.0,
    q_weight: float = 1.0,
    sharpness: float = 20.0,
) -> ResonanceOptimizationResult:
    """Optimize selected layer thicknesses toward target resonance and Q."""

    base_thicknesses = stack_thicknesses(stack)
    if variable_layer_indices is None:
        indices = tuple(range(stack.num_layers))
    else:
        indices = tuple(variable_layer_indices)
    if not indices:
        raise ValueError("At least one variable layer index is required.")
    index_array = jnp.asarray(indices, dtype=jnp.int32)
    initial_variables = base_thicknesses[index_array]
    wavelengths = jnp.asarray(wavelengths_nm)

    def full_thicknesses(variable_thicknesses: jnp.ndarray) -> jnp.ndarray:
        return base_thicknesses.at[index_array].set(variable_thicknesses)

    def objective(variable_thicknesses: jnp.ndarray) -> jnp.ndarray:
        candidate_stack = stack_with_thicknesses(stack, full_thicknesses(variable_thicknesses))
        result = simulate_spectrum(
            candidate_stack,
            wavelengths_nm=wavelengths,
            angle_deg=angle_deg,
            angle_rad=angle_rad,
            polarization=polarization,
        )
        values = _select_spectrum(result, spectrum)
        return resonance_target_loss(
            wavelength_nm=wavelengths,
            spectrum_values=values,
            target_wavelength_nm=target_wavelength_nm,
            target_q=target_q,
            feature=feature,
            wavelength_weight=wavelength_weight,
            q_weight=q_weight,
            sharpness=sharpness,
        )

    optimized = optimize_thicknesses(
        objective,
        initial_thicknesses_nm=initial_variables,
        steps=steps,
        learning_rate=learning_rate,
        lower_bound_nm=lower_bound_nm,
    )
    final_thicknesses = full_thicknesses(optimized.thicknesses_nm)
    final_stack = stack_with_thicknesses(stack, final_thicknesses)
    final_result = simulate_spectrum(
        final_stack,
        wavelengths_nm=wavelengths,
        angle_deg=angle_deg,
        angle_rad=angle_rad,
        polarization=polarization,
    )
    final_spectrum = _select_spectrum(final_result, spectrum)
    resonance = analyze_resonance(wavelengths, final_spectrum, feature=feature)
    return ResonanceOptimizationResult(
        thicknesses_nm=final_thicknesses,
        variable_thicknesses_nm=optimized.thicknesses_nm,
        history=optimized.history,
        resonance=resonance,
    )


def thickness_gradient(
    objective: Objective,
    thicknesses_nm: jnp.ndarray,
) -> jnp.ndarray:
    """Convenience wrapper for ``jax.grad(objective)(thicknesses_nm)``."""

    return jax.grad(objective)(thicknesses_nm)


def initial_thicknesses(stack: Stack) -> jnp.ndarray:
    """Return the stack's finite-layer thicknesses."""

    return stack_thicknesses(stack)


def _select_spectrum(result: object, spectrum: str) -> jnp.ndarray:
    key = spectrum.upper()
    if key == "R":
        values = result.R
    elif key == "T":
        values = result.T
    elif key == "A":
        values = result.A
    else:
        raise ValueError('spectrum must be "R", "T", or "A".')
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(
                "Target optimization expects one polarization; pass polarization='s' or 'p'."
            )
        return values[0]
    return values
