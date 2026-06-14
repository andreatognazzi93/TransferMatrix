"""Coherent transfer-matrix method implemented with JAX."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from multilayer_tmm.layers import Stack, stack_thicknesses


@dataclass(frozen=True)
class SimulationResult:
    """Spectral optical response for one or more polarizations."""

    wavelength_nm: jnp.ndarray
    R: jnp.ndarray
    T: jnp.ndarray
    A: jnp.ndarray
    r: jnp.ndarray
    t: jnp.ndarray
    polarizations: tuple[str, ...]


def stack_to_arrays(stack: Stack, wavelengths_nm: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a user-friendly ``Stack`` into JAX arrays.

    Returns:
        ``n_by_wavelength`` with shape ``(num_wavelengths, num_media)`` and
        ``thicknesses_nm`` with shape ``(num_finite_layers,)``.
    """

    wavelengths = _as_1d_wavelengths(wavelengths_nm)
    indices = [material(wavelengths) for material in stack.materials]
    n_by_wavelength = jnp.stack(indices, axis=-1)
    thicknesses_nm = stack_thicknesses(stack, dtype=wavelengths.dtype)
    return n_by_wavelength, thicknesses_nm


def simulate_spectrum(
    stack: Stack,
    wavelengths_nm: Any,
    angle_deg: Any | None = None,
    angle_rad: Any | None = None,
    polarization: str = "s",
) -> SimulationResult:
    """Simulate wavelength-dependent ``R``, ``T``, ``A``, ``r``, and ``t``.

    Args:
        stack: Incident medium, finite layers, and substrate.
        wavelengths_nm: One-dimensional wavelength array in nanometers.
        angle_deg: Incident angle in degrees. Defaults to normal incidence.
        angle_rad: Incident angle in radians. Mutually exclusive with
            ``angle_deg``.
        polarization: ``"s"``, ``"p"``, or ``"both"``.
    """

    wavelengths = _as_1d_wavelengths(wavelengths_nm)
    theta0 = _angle_to_radians(
        angle_deg=angle_deg,
        angle_rad=angle_rad,
        dtype=wavelengths.dtype,
    )
    n_by_wavelength, thicknesses_nm = stack_to_arrays(stack, wavelengths)
    return simulate_spectrum_arrays(
        n_by_wavelength=n_by_wavelength,
        thicknesses_nm=thicknesses_nm,
        wavelengths_nm=wavelengths,
        angle_rad=theta0,
        polarization=polarization,
    )


def simulate_spectrum_arrays(
    n_by_wavelength: Any,
    thicknesses_nm: Any,
    wavelengths_nm: Any,
    angle_rad: Any = 0.0,
    polarization: str = "s",
) -> SimulationResult:
    """Functional JAX-compatible simulation interface.

    ``n_by_wavelength`` must have shape ``(num_wavelengths, num_layers + 2)``.
    The first and last media are the incident and substrate media. This function
    is useful for optimization loops that already have material arrays and
    differentiable thickness arrays.
    """

    wavelengths = _as_1d_wavelengths(wavelengths_nm)
    n_array = jnp.asarray(n_by_wavelength)
    thicknesses = jnp.asarray(thicknesses_nm, dtype=wavelengths.dtype)
    theta0 = jnp.asarray(angle_rad, dtype=wavelengths.dtype)

    expected_media = thicknesses.shape[0] + 2
    if n_array.ndim != 2:
        raise ValueError("n_by_wavelength must be a two-dimensional array.")
    if n_array.shape[0] != wavelengths.shape[0]:
        raise ValueError(
            "n_by_wavelength first dimension must match wavelengths_nm length."
        )
    if n_array.shape[1] != expected_media:
        raise ValueError(
            "n_by_wavelength second dimension must equal len(thicknesses_nm) + 2."
        )

    polarizations = _normalize_polarization(polarization)
    per_polarization = [
        _simulate_one_polarization(
            n_by_wavelength=n_array,
            thicknesses_nm=thicknesses,
            wavelengths_nm=wavelengths,
            angle_rad=theta0,
            polarization=pol,
        )
        for pol in polarizations
    ]

    if len(per_polarization) == 1:
        R, T, A, r, t = per_polarization[0]
    else:
        R = jnp.stack([item[0] for item in per_polarization])
        T = jnp.stack([item[1] for item in per_polarization])
        A = jnp.stack([item[2] for item in per_polarization])
        r = jnp.stack([item[3] for item in per_polarization])
        t = jnp.stack([item[4] for item in per_polarization])

    return SimulationResult(
        wavelength_nm=wavelengths,
        R=R,
        T=T,
        A=A,
        r=r,
        t=t,
        polarizations=polarizations,
    )


@partial(jax.jit, static_argnames=("polarization",))
def _simulate_one_polarization(
    n_by_wavelength: jnp.ndarray,
    thicknesses_nm: jnp.ndarray,
    wavelengths_nm: jnp.ndarray,
    angle_rad: jnp.ndarray,
    polarization: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jax.vmap(
        lambda wavelength_nm, n_at_wavelength: _coherent_tmm_single(
            wavelength_nm=wavelength_nm,
            n_at_wavelength=n_at_wavelength,
            thicknesses_nm=thicknesses_nm,
            angle_rad=angle_rad,
            polarization=polarization,
        )
    )(wavelengths_nm, n_by_wavelength)


def _coherent_tmm_single(
    wavelength_nm: jnp.ndarray,
    n_at_wavelength: jnp.ndarray,
    thicknesses_nm: jnp.ndarray,
    angle_rad: jnp.ndarray,
    polarization: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-wavelength coherent transfer-matrix calculation."""

    n = jnp.asarray(n_at_wavelength)
    kx = n[0] * jnp.sin(angle_rad)
    cos_theta = _forward_cos_theta(n, kx)
    admittance = _optical_admittance(n, cos_theta, polarization)

    n_layers = n[1:-1]
    cos_layers = cos_theta[1:-1]
    y_layers = _safe_complex(admittance[1:-1])
    phase = 2.0 * jnp.pi * n_layers * cos_layers * thicknesses_nm / wavelength_nm

    layer_matrices = _characteristic_matrices(phase, y_layers)
    identity = jnp.eye(2, dtype=n.dtype)
    system_matrix, _ = jax.lax.scan(
        lambda accumulated, layer_matrix: (accumulated @ layer_matrix, None),
        identity,
        layer_matrices,
    )

    y0 = admittance[0]
    ys = admittance[-1]

    # Characteristic admittance relation:
    # B = M_11 + M_12 Y_s, C = M_21 + M_22 Y_s
    # r = (Y_0 B - C) / (Y_0 B + C), t = 2 Y_0 / (Y_0 B + C)
    b = system_matrix[0, 0] + system_matrix[0, 1] * ys
    c = system_matrix[1, 0] + system_matrix[1, 1] * ys
    denominator = _safe_complex(y0 * b + c)
    r = (y0 * b - c) / denominator
    t = 2.0 * y0 / denominator

    R = jnp.real(jnp.abs(r) ** 2)
    # Power-flux correction. For s polarization Y = n cos(theta);
    # for p polarization Y = cos(theta) / n.
    T = jnp.real(ys) / _safe_real(jnp.real(y0)) * jnp.real(jnp.abs(t) ** 2)
    A = 1.0 - R - T
    return R, T, A, r, t


def _characteristic_matrices(
    phase: jnp.ndarray,
    admittance: jnp.ndarray,
) -> jnp.ndarray:
    """Build characteristic matrices for finite layers.

    With the ``n + i k`` refractive-index convention, positive ``k`` is lossy
    for fields proportional to ``exp(i omega t)``. The negative imaginary sign
    below keeps absorbing films passive under that convention.

    M_j = [[cos(delta_j), -i sin(delta_j) / Y_j],
           [-i Y_j sin(delta_j), cos(delta_j)]]
    """

    cos_delta = jnp.cos(phase)
    sin_delta = jnp.sin(phase)
    top = jnp.stack((cos_delta, -1.0j * sin_delta / admittance), axis=-1)
    bottom = jnp.stack((-1.0j * admittance * sin_delta, cos_delta), axis=-1)
    return jnp.stack((top, bottom), axis=-2)


def _forward_cos_theta(n: jnp.ndarray, kx: jnp.ndarray) -> jnp.ndarray:
    """Complex Snell law with a forward-propagating square-root branch."""

    sin_theta = kx / _safe_complex(n)
    cos_theta = jnp.sqrt(1.0 + 0.0j - sin_theta**2)
    kz = n * cos_theta
    flip = (jnp.real(kz) < 0.0) | (
        (jnp.abs(jnp.real(kz)) < 1e-12) & (jnp.imag(kz) < 0.0)
    )
    return jnp.where(flip, -cos_theta, cos_theta)


def _optical_admittance(
    n: jnp.ndarray,
    cos_theta: jnp.ndarray,
    polarization: str,
) -> jnp.ndarray:
    if polarization == "s":
        return n * cos_theta
    if polarization == "p":
        return cos_theta / _safe_complex(n)
    raise ValueError(f"Unknown polarization: {polarization!r}")


def _safe_complex(value: jnp.ndarray) -> jnp.ndarray:
    tiny = jnp.asarray(1e-30, dtype=jnp.real(value).dtype)
    return jnp.where(jnp.abs(value) < tiny, tiny + 0.0j, value)


def _safe_real(value: jnp.ndarray) -> jnp.ndarray:
    tiny = jnp.asarray(1e-30, dtype=value.dtype)
    return jnp.where(jnp.abs(value) < tiny, tiny, value)


def _as_1d_wavelengths(wavelengths_nm: Any) -> jnp.ndarray:
    wavelengths = jnp.asarray(wavelengths_nm)
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_nm must be a one-dimensional array.")
    return wavelengths


def _angle_to_radians(
    angle_deg: Any | None,
    angle_rad: Any | None,
    dtype: Any,
) -> jnp.ndarray:
    if angle_deg is not None and angle_rad is not None:
        raise ValueError("Pass angle_deg or angle_rad, not both.")
    if angle_rad is not None:
        return jnp.asarray(angle_rad, dtype=dtype)
    if angle_deg is None:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.deg2rad(jnp.asarray(angle_deg, dtype=dtype))


def _normalize_polarization(polarization: str) -> tuple[str, ...]:
    normalized = polarization.lower()
    if normalized in {"s", "te"}:
        return ("s",)
    if normalized in {"p", "tm"}:
        return ("p",)
    if normalized in {"both", "sp", "s,p", "p,s"}:
        return ("s", "p")
    raise ValueError('polarization must be "s", "p", or "both".')
