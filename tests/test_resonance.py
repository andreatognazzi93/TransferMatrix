import jax
import jax.numpy as jnp

from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    analyze_resonance,
    optimize_resonance_target,
    read_spectrum_csv,
    simulate_spectrum,
    smooth_resonance_metrics,
)


def test_analyze_resonance_extracts_lorentzian_peak_quality_factor():
    wavelengths = jnp.linspace(500.0, 700.0, 2001)
    lambda0 = 600.0
    fwhm = 20.0
    spectrum = 1.0 / (1.0 + 4.0 * ((wavelengths - lambda0) / fwhm) ** 2)

    resonance = analyze_resonance(wavelengths, spectrum, feature="peak", baseline=0.0)

    assert abs(resonance.resonance_wavelength_nm - lambda0) < 0.2
    assert abs(resonance.linewidth_nm - fwhm) < 0.2
    assert abs(resonance.quality_factor - lambda0 / fwhm) < 0.05


def test_analyze_resonance_extracts_lorentzian_dip_quality_factor():
    wavelengths = jnp.linspace(500.0, 700.0, 2001)
    lambda0 = 610.0
    fwhm = 25.0
    spectrum = 1.0 - 0.8 / (1.0 + 4.0 * ((wavelengths - lambda0) / fwhm) ** 2)

    resonance = analyze_resonance(wavelengths, spectrum, feature="dip", baseline=1.0)

    assert abs(resonance.resonance_wavelength_nm - lambda0) < 0.2
    assert abs(resonance.linewidth_nm - fwhm) < 0.3
    assert abs(resonance.quality_factor - lambda0 / fwhm) < 0.05


def test_smooth_resonance_metrics_are_differentiable():
    wavelengths = jnp.linspace(500.0, 700.0, 101)

    def estimated_wavelength(center_nm):
        spectrum = jnp.exp(-0.5 * ((wavelengths - center_nm) / 12.0) ** 2)
        return smooth_resonance_metrics(
            wavelengths,
            spectrum,
            feature="peak",
            sharpness=20.0,
        ).resonance_wavelength_nm

    gradient = jax.grad(estimated_wavelength)(610.0)

    assert jnp.isfinite(gradient)
    assert gradient > 0.0


def test_dbr_reflectivity_peak_is_near_quarter_wave_center():
    air = Material.constant(1.0 + 0.0j, name="air")
    low = Material.constant(1.5 + 0.0j, name="n1")
    high = Material.constant(2.5 + 0.0j, name="n2")
    layers = []
    for _ in range(8):
        layers.extend(
            [
                Layer(material=low, thickness_nm=105.0),
                Layer(material=high, thickness_nm=63.0),
            ]
        )
    stack = Stack(incident=air, layers=layers, substrate=high)
    wavelengths = jnp.linspace(420.0, 1260.0, 500)

    result = simulate_spectrum(stack, wavelengths, angle_deg=0.0, polarization="s")
    resonance = analyze_resonance(wavelengths, result.R, feature="peak")

    assert abs(resonance.resonance_wavelength_nm - 630.0) < 5.0
    assert resonance.quality_factor > 1.0


def test_optimize_resonance_target_returns_finite_history():
    air = Material.constant(1.0 + 0.0j)
    film = Material.constant(1.8 + 0.0j)
    stack = Stack(
        incident=air,
        layers=[Layer(material=film, thickness_nm=120.0)],
        substrate=air,
    )
    wavelengths = jnp.linspace(450.0, 750.0, 81)

    result = optimize_resonance_target(
        stack,
        wavelengths_nm=wavelengths,
        target_wavelength_nm=600.0,
        target_q=4.0,
        spectrum="T",
        feature="peak",
        steps=3,
        learning_rate=0.05,
    )

    assert result.history.shape == (3,)
    assert result.thicknesses_nm.shape == (1,)
    assert jnp.all(jnp.isfinite(result.history))
    assert jnp.all(jnp.isfinite(result.thicknesses_nm))


def test_read_spectrum_csv_converts_wavelength_units_and_sorts(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("Wavelength (um),Linear\n1.0,0.5\n0.5,0.2\n")

    wavelengths, values = read_spectrum_csv(path, wavelength_scale=1000.0)

    assert jnp.allclose(wavelengths, jnp.array([500.0, 1000.0]))
    assert jnp.allclose(values, jnp.array([0.2, 0.5]))
