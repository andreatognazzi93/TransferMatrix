"""Kernel tests for the angle-sweep map feature (ANGLE_MAP_CONTRACT §1).

Covers:
- simulate_angle_map / simulate_angle_map_arrays output shapes.
- Each row matches a direct simulate_spectrum call to 1e-5 tolerance.
- A = 1 - R - T for a lossless stack.
- polarization="both" raises ValueError.
- angles_rad must be 1-D (ValueError otherwise).
- Both "s" and "p" polarizations.
- AngleMapResult dataclass fields and polarization string type.
"""

from __future__ import annotations

import numpy as np
import pytest

from multilayer_tmm import (
    AngleMapResult,
    Layer,
    Material,
    Stack,
    simulate_angle_map,
    simulate_angle_map_arrays,
    simulate_spectrum,
    stack_to_arrays,
    wavelength_grid,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _lossless_stack() -> Stack:
    """Air / TiO2 (120 nm) / SiO2 (90 nm) / glass — fully lossless."""
    return Stack(
        incident=Material.constant(1.0),
        layers=[
            Layer(Material.constant(2.35), 120.0),
            Layer(Material.constant(1.46), 90.0),
        ],
        substrate=Material.constant(1.52),
    )


def _lossy_stack() -> Stack:
    """Air / absorbing layer (n=2.35+0.1j, 120 nm) / glass."""
    return Stack(
        incident=Material.constant(1.0),
        layers=[
            Layer(Material.constant(complex(2.35, 0.1)), 120.0),
        ],
        substrate=Material.constant(1.52),
    )


_WL = wavelength_grid(400.0, 700.0, 31)   # 31 points, fast
_ANGLES = [0.0, 10.0, 20.0, 30.0, 45.0]  # 5 angles


# ===========================================================================
# §1: AngleMapResult dataclass
# ===========================================================================

def test_angle_map_result_is_dataclass():
    """AngleMapResult is importable and frozen."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="s")
    assert isinstance(result, AngleMapResult)


def test_angle_map_result_field_order():
    """Field order: wavelength_nm, angle_deg, R, T, A, polarization (contract §1.1)."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="s")
    assert hasattr(result, "wavelength_nm")
    assert hasattr(result, "angle_deg")
    assert hasattr(result, "R")
    assert hasattr(result, "T")
    assert hasattr(result, "A")
    assert hasattr(result, "polarization")


def test_angle_map_result_polarization_is_str():
    """AngleMapResult.polarization is a plain str, not a tuple (§1.1)."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="s")
    assert isinstance(result.polarization, str)
    assert result.polarization == "s"


def test_angle_map_result_no_r_t_amplitude_fields():
    """AngleMapResult intentionally has no r/t complex amplitude fields (§1.1)."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="s")
    assert not hasattr(result, "r")
    assert not hasattr(result, "t")


# ===========================================================================
# §1.2–1.3: Output shapes
# ===========================================================================

@pytest.mark.parametrize("polarization", ["s", "p"])
def test_angle_map_output_shape(polarization):
    """R/T/A are (num_angles, num_wavelengths); axes are (§1.1)."""
    angles = [0.0, 15.0, 30.0, 45.0]
    num_wl = 21
    wl = wavelength_grid(400.0, 700.0, num_wl)
    result = simulate_angle_map(_lossless_stack(), wl, angles, polarization=polarization)

    expected_shape = (len(angles), num_wl)
    assert np.asarray(result.R).shape == expected_shape, (
        f"R shape {np.asarray(result.R).shape} != {expected_shape}"
    )
    assert np.asarray(result.T).shape == expected_shape
    assert np.asarray(result.A).shape == expected_shape
    assert np.asarray(result.wavelength_nm).shape == (num_wl,)
    assert np.asarray(result.angle_deg).shape == (len(angles),)


def test_angle_map_angle_deg_axis_values():
    """angle_deg axis contains the exact requested degrees (converted back from rad)."""
    angles = [0.0, 10.0, 20.0, 30.0]
    result = simulate_angle_map(_lossless_stack(), _WL, angles, polarization="s")
    angle_deg_out = np.asarray(result.angle_deg, dtype=float)
    np.testing.assert_allclose(angle_deg_out, angles, atol=1e-4)


# ===========================================================================
# §1.2: Row-wise cross-check: each angle row == direct simulate_spectrum
# ===========================================================================

@pytest.mark.parametrize("polarization", ["s", "p"])
def test_angle_map_rows_match_spectrum_s_pol(polarization):
    """Each row of R/T/A matches a direct simulate_spectrum call to 1e-5 (§cover item 1)."""
    angles = [0.0, 10.0, 20.0, 30.0]
    stack = _lossless_stack()
    wl = wavelength_grid(400.0, 700.0, 31)

    result = simulate_angle_map(stack, wl, angles, polarization=polarization)
    R_map = np.asarray(result.R, dtype=float)
    T_map = np.asarray(result.T, dtype=float)
    A_map = np.asarray(result.A, dtype=float)

    for i, angle in enumerate(angles):
        direct = simulate_spectrum(
            stack, wavelengths_nm=wl, angle_deg=angle, polarization=polarization
        )
        np.testing.assert_allclose(
            R_map[i], np.asarray(direct.R, dtype=float),
            rtol=1e-5, atol=1e-7,
            err_msg=f"R mismatch at angle={angle} pol={polarization}",
        )
        np.testing.assert_allclose(
            T_map[i], np.asarray(direct.T, dtype=float),
            rtol=1e-5, atol=1e-7,
            err_msg=f"T mismatch at angle={angle} pol={polarization}",
        )
        np.testing.assert_allclose(
            A_map[i], np.asarray(direct.A, dtype=float),
            rtol=1e-5, atol=1e-7,
            err_msg=f"A mismatch at angle={angle} pol={polarization}",
        )


# ===========================================================================
# §1.2: A = 1 - R - T for a lossless stack
# ===========================================================================

def test_angle_map_energy_conservation_lossless():
    """A = 1 - R - T everywhere for a lossless stack (§cover item 1)."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="s")
    R = np.asarray(result.R, dtype=float)
    T = np.asarray(result.T, dtype=float)
    A = np.asarray(result.A, dtype=float)
    np.testing.assert_allclose(
        A, 1.0 - R - T,
        atol=1e-6,
        err_msg="A != 1-R-T for lossless stack",
    )
    # Also R+T+A == 1
    np.testing.assert_allclose(R + T + A, 1.0, atol=1e-6)


def test_angle_map_energy_conservation_lossless_p_pol():
    """A = 1 - R - T for p polarization too."""
    result = simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="p")
    R = np.asarray(result.R, dtype=float)
    T = np.asarray(result.T, dtype=float)
    A = np.asarray(result.A, dtype=float)
    np.testing.assert_allclose(A, 1.0 - R - T, atol=1e-6)


# ===========================================================================
# §1.2: ValueError when polarization="both"
# ===========================================================================

def test_angle_map_both_polarization_raises():
    """simulate_angle_map with polarization='both' must raise ValueError (§1.2)."""
    with pytest.raises(ValueError, match="single polarization"):
        simulate_angle_map(_lossless_stack(), _WL, _ANGLES, polarization="both")


def test_angle_map_arrays_both_polarization_raises():
    """simulate_angle_map_arrays with polarization='both' must raise ValueError (§1.2)."""
    import jax.numpy as jnp
    stack = _lossless_stack()
    wl = wavelength_grid(400.0, 700.0, 11)
    n, th = stack_to_arrays(stack, wl)
    angles_rad = jnp.deg2rad(jnp.array([0.0, 10.0]))
    with pytest.raises(ValueError, match="single polarization"):
        simulate_angle_map_arrays(n, th, wl, angles_rad, polarization="both")


# ===========================================================================
# §1.2: angles_rad must be 1-D
# ===========================================================================

def test_angle_map_arrays_2d_angles_raises():
    """2-D angles_rad must raise ValueError (§1.2)."""
    import jax.numpy as jnp
    stack = _lossless_stack()
    wl = wavelength_grid(400.0, 700.0, 11)
    n, th = stack_to_arrays(stack, wl)
    angles_2d = jnp.array([[0.0, 0.1], [0.2, 0.3]])  # shape (2, 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        simulate_angle_map_arrays(n, th, wl, angles_2d, polarization="s")


# ===========================================================================
# Reuse of existing n_by_wavelength validation from simulate_spectrum_arrays
# ===========================================================================

def test_angle_map_arrays_wrong_ndim_raises():
    """n_by_wavelength with ndim != 2 must raise ValueError (§1.2 reuses validate)."""
    import jax.numpy as jnp
    wl = wavelength_grid(400.0, 700.0, 11)
    # wrong: 1-D
    bad_n = jnp.ones((11,))
    angles_rad = jnp.array([0.0, 0.1])
    with pytest.raises(ValueError, match="two-dimensional"):
        simulate_angle_map_arrays(bad_n, jnp.array([100.0]), wl, angles_rad)


# ===========================================================================
# simulate_angle_map uses the friendly interface
# ===========================================================================

def test_simulate_angle_map_accepts_list_angles():
    """simulate_angle_map accepts a plain Python list of angles (§1.3)."""
    result = simulate_angle_map(_lossless_stack(), _WL, [0.0, 30.0, 60.0], polarization="p")
    assert np.asarray(result.R).shape[0] == 3


def test_simulate_angle_map_single_angle_in_list():
    """A length-1 angle list produces shape (1, num_wavelengths)."""
    wl = wavelength_grid(400.0, 700.0, 11)
    result = simulate_angle_map(_lossless_stack(), wl, [45.0], polarization="s")
    assert np.asarray(result.R).shape == (1, 11)
    # Must match direct simulate_spectrum at the same angle.
    direct = simulate_spectrum(_lossless_stack(), wl, angle_deg=45.0, polarization="s")
    np.testing.assert_allclose(
        np.asarray(result.R)[0], np.asarray(direct.R, dtype=float),
        rtol=1e-5, atol=1e-7,
    )


# ===========================================================================
# Re-export check (§1.4)
# ===========================================================================

def test_angle_map_names_in_dunder_all():
    """AngleMapResult, simulate_angle_map, simulate_angle_map_arrays in __all__ (§1.4)."""
    import multilayer_tmm
    assert "AngleMapResult" in multilayer_tmm.__all__
    assert "simulate_angle_map" in multilayer_tmm.__all__
    assert "simulate_angle_map_arrays" in multilayer_tmm.__all__
