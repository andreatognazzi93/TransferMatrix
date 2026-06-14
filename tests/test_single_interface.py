import jax.numpy as jnp

from multilayer_tmm import Material, Stack, simulate_spectrum


def test_single_air_glass_interface_matches_fresnel_normal_incidence():
    air = Material.constant(1.0 + 0.0j)
    glass = Material.constant(1.5 + 0.0j)
    stack = Stack(incident=air, layers=[], substrate=glass)

    result = simulate_spectrum(
        stack,
        wavelengths_nm=jnp.array([550.0]),
        angle_deg=0.0,
        polarization="s",
    )

    expected_R = jnp.abs((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    assert jnp.allclose(result.R[0], expected_R, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(result.T[0], 1.0 - expected_R, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(result.A[0], 0.0, atol=1e-6)
