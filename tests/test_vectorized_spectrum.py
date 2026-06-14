import jax.numpy as jnp

from multilayer_tmm import Layer, Material, Stack, simulate_spectrum


def test_vectorized_wavelength_simulation_has_expected_shape():
    air = Material.constant(1.0 + 0.0j)
    coating = Material.constant(1.9 + 0.0j)
    glass = Material.constant(1.5 + 0.0j)
    stack = Stack(
        incident=air,
        layers=[Layer(material=coating, thickness_nm=80.0)],
        substrate=glass,
    )
    wavelengths = jnp.linspace(400.0, 800.0, 25)

    result = simulate_spectrum(
        stack,
        wavelengths_nm=wavelengths,
        angle_deg=35.0,
        polarization="both",
    )

    assert result.wavelength_nm.shape == (25,)
    assert result.R.shape == (2, 25)
    assert result.T.shape == (2, 25)
    assert result.A.shape == (2, 25)
    assert result.r.shape == (2, 25)
    assert result.t.shape == (2, 25)
    assert result.polarizations == ("s", "p")
