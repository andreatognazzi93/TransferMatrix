import jax
import jax.numpy as jnp

from multilayer_tmm import Layer, Material, Stack, simulate_spectrum


def test_gradient_with_respect_to_layer_thickness_is_finite():
    air = Material.constant(1.0 + 0.0j)
    coating = Material.constant(1.38 + 0.0j)
    glass = Material.constant(1.52 + 0.0j)
    wavelengths = jnp.linspace(450.0, 650.0, 31)

    def mean_reflectivity(thickness_nm):
        stack = Stack(
            incident=air,
            layers=[Layer(material=coating, thickness_nm=thickness_nm)],
            substrate=glass,
        )
        return jnp.mean(
            simulate_spectrum(
                stack,
                wavelengths_nm=wavelengths,
                angle_deg=0.0,
                polarization="s",
            ).R
        )

    grad_value = jax.grad(mean_reflectivity)(95.0)

    assert jnp.isfinite(grad_value)
