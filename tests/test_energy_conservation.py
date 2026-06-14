import jax.numpy as jnp

from multilayer_tmm import Layer, Material, Stack, simulate_spectrum


def test_non_absorbing_stack_conserves_energy():
    air = Material.constant(1.0 + 0.0j)
    mgf2 = Material.constant(1.38 + 0.0j)
    glass = Material.constant(1.52 + 0.0j)
    stack = Stack(
        incident=air,
        layers=[Layer(material=mgf2, thickness_nm=100.0)],
        substrate=glass,
    )

    wavelengths = jnp.linspace(400.0, 700.0, 41)
    result = simulate_spectrum(
        stack,
        wavelengths_nm=wavelengths,
        angle_deg=20.0,
        polarization="p",
    )

    assert jnp.allclose(result.R + result.T, 1.0, rtol=2e-5, atol=2e-5)
    assert jnp.allclose(result.A, 0.0, atol=2e-5)


def test_absorbing_layer_has_positive_absorption():
    air = Material.constant(1.0 + 0.0j)
    metal = Material.constant(0.2 + 3.0j)
    glass = Material.constant(1.5 + 0.0j)
    stack = Stack(
        incident=air,
        layers=[Layer(material=metal, thickness_nm=50.0)],
        substrate=glass,
    )

    result = simulate_spectrum(
        stack,
        wavelengths_nm=jnp.array([550.0]),
        angle_deg=0.0,
        polarization="s",
    )

    assert result.A[0] > 0.0
    assert result.R[0] >= 0.0
    assert result.T[0] >= 0.0
