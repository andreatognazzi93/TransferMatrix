"""Resonance and quality-factor analysis for spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

FeatureKind = Literal["peak", "dip"]


@dataclass(frozen=True)
class ResonanceResult:
    """Dominant resonance extracted from one spectrum."""

    resonance_wavelength_nm: float
    linewidth_nm: float
    quality_factor: float
    extremum_value: float
    half_level: float
    left_wavelength_nm: float
    right_wavelength_nm: float
    feature: str


@dataclass(frozen=True)
class SmoothResonanceMetrics:
    """Differentiable resonance estimates for optimization objectives."""

    resonance_wavelength_nm: jnp.ndarray
    linewidth_nm: jnp.ndarray
    quality_factor: jnp.ndarray


def analyze_resonance(
    wavelength_nm: jnp.ndarray,
    spectrum: jnp.ndarray,
    feature: FeatureKind = "peak",
    fraction: float = 0.5,
    baseline: float | None = None,
) -> ResonanceResult:
    """Extract dominant resonance wavelength and Q from a sampled spectrum.

    ``feature="peak"`` uses the global maximum and its full width at the
    requested fraction of prominence. ``feature="dip"`` applies the same logic
    to the inverted spectrum. The default ``fraction=0.5`` is FWHM, so
    ``Q = lambda0 / FWHM``. Pass ``baseline`` when the off-resonance value is
    known; otherwise the sampled spectrum minimum or maximum is used.
    """

    if feature not in {"peak", "dip"}:
        raise ValueError('feature must be "peak" or "dip".')
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1.")

    wavelengths = np.asarray(wavelength_nm, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if wavelengths.ndim != 1 or values.ndim != 1:
        raise ValueError("wavelength_nm and spectrum must be one-dimensional.")
    if wavelengths.shape != values.shape:
        raise ValueError("wavelength_nm and spectrum must have matching shapes.")
    if wavelengths.size < 3:
        raise ValueError("At least three spectral samples are required.")

    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    values = values[order]

    profile = values if feature == "peak" else -values
    extremum_index = int(np.argmax(profile))
    if baseline is None:
        profile_min = float(np.min(profile))
    else:
        profile_min = float(baseline if feature == "peak" else -baseline)
    profile_max = float(profile[extremum_index])
    profile_level = profile_min + fraction * (profile_max - profile_min)
    half_level = profile_level if feature == "peak" else -profile_level

    left = _crossing_wavelength(
        wavelengths[: extremum_index + 1],
        profile[: extremum_index + 1],
        profile_level,
        side="left",
    )
    right = _crossing_wavelength(
        wavelengths[extremum_index:],
        profile[extremum_index:],
        profile_level,
        side="right",
    )
    linewidth = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
    resonance_wavelength = float(wavelengths[extremum_index])
    quality_factor = resonance_wavelength / linewidth if linewidth > 0.0 else np.nan

    return ResonanceResult(
        resonance_wavelength_nm=resonance_wavelength,
        linewidth_nm=float(linewidth),
        quality_factor=float(quality_factor),
        extremum_value=float(values[extremum_index]),
        half_level=float(half_level),
        left_wavelength_nm=float(left),
        right_wavelength_nm=float(right),
        feature=feature,
    )


def smooth_resonance_metrics(
    wavelength_nm: jnp.ndarray,
    spectrum: jnp.ndarray,
    feature: FeatureKind = "peak",
    sharpness: float = 20.0,
    eps: float = 1e-12,
) -> SmoothResonanceMetrics:
    """Differentiable resonance estimates based on soft spectral moments.

    The discrete FWHM calculation in ``analyze_resonance`` is appropriate for
    reporting, but not for gradient-based optimization. This function estimates
    the dominant feature center with softmax weights and approximates linewidth
    as the Gaussian FWHM corresponding to the weighted standard deviation.
    """

    if feature not in {"peak", "dip"}:
        raise ValueError('feature must be "peak" or "dip".')

    wavelengths = jnp.asarray(wavelength_nm)
    values = jnp.asarray(spectrum)
    profile = values if feature == "peak" else -values
    profile_min = jnp.min(profile)
    profile_max = jnp.max(profile)
    normalized = (profile - profile_min) / (profile_max - profile_min + eps)
    weights = jnp.exp(sharpness * (normalized - jnp.max(normalized)))
    weights = weights / (jnp.sum(weights) + eps)

    center = jnp.sum(weights * wavelengths)
    variance = jnp.sum(weights * (wavelengths - center) ** 2)
    linewidth = 2.354820045 * jnp.sqrt(variance + eps)
    quality_factor = center / (linewidth + eps)
    return SmoothResonanceMetrics(
        resonance_wavelength_nm=center,
        linewidth_nm=linewidth,
        quality_factor=quality_factor,
    )


def _crossing_wavelength(
    wavelengths: np.ndarray,
    profile: np.ndarray,
    level: float,
    side: Literal["left", "right"],
) -> float:
    above = profile >= level
    if side == "left":
        candidates = np.nonzero((~above[:-1]) & above[1:])[0]
        if candidates.size == 0:
            return float(wavelengths[0]) if bool(above[0]) else np.nan
        index = int(candidates[-1])
    else:
        candidates = np.nonzero(above[:-1] & (~above[1:]))[0]
        if candidates.size == 0:
            return float(wavelengths[-1]) if bool(above[-1]) else np.nan
        index = int(candidates[0])

    x0 = float(wavelengths[index])
    x1 = float(wavelengths[index + 1])
    y0 = float(profile[index])
    y1 = float(profile[index + 1])
    if y1 == y0:
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
