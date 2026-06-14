import numpy as np
import jax.numpy as jnp

from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    export_simulation,
    simulate_spectrum,
)
from multilayer_tmm.io import _show_method


def test_export_simulation_writes_connected_text_files_and_plots(tmp_path):
    air = Material.constant(1.0 + 0.0j, name="air")
    coating = Material.constant(1.38 + 0.0j, name="MgF2")
    glass = Material.constant(1.52 + 0.0j, name="glass")
    stack = Stack(
        incident=air,
        layers=[Layer(material=coating, thickness_nm=100.0)],
        substrate=glass,
    )
    wavelengths = jnp.linspace(400.0, 700.0, 5)
    result = simulate_spectrum(
        stack,
        wavelengths_nm=wavelengths,
        angle_deg=0.0,
        polarization="s",
    )

    paths = export_simulation(
        stack=stack,
        result=result,
        output_dir=tmp_path,
        simulation_name="ar_test",
        timestamp="20260601_143015",
    )

    assert paths.structure_txt.name == "ar_test_20260601_143015_structure.txt"
    assert paths.spectra_txt.name == "ar_test_20260601_143015_spectra.txt"
    assert paths.reflectivity_plot.name == "ar_test_20260601_143015_reflectivity.png"
    assert paths.transmission_plot.name == "ar_test_20260601_143015_transmission.png"
    assert paths.absorption_plot.name == "ar_test_20260601_143015_absorption.png"
    assert paths.combined_plot.name == "ar_test_20260601_143015_RTA.png"

    structure = paths.structure_txt.read_text()
    assert "simulation_name: ar_test" in structure
    assert "timestamp: 20260601_143015" in structure
    assert "role,layer_index,material,thickness_nm,refractive_index" in structure
    assert "finite,1,MgF2,100" in structure
    assert "substrate,,glass,semi-infinite" in structure

    spectra = np.loadtxt(paths.spectra_txt, comments="#")
    assert spectra.shape == (5, 4)
    np.testing.assert_allclose(spectra[:, 0], np.asarray(wavelengths))
    np.testing.assert_allclose(spectra[:, 1], np.asarray(result.R), rtol=1e-6)
    np.testing.assert_allclose(spectra[:, 2], np.asarray(result.T), rtol=1e-6)
    np.testing.assert_allclose(spectra[:, 3], np.asarray(result.A), rtol=1e-6)

    assert paths.reflectivity_plot.stat().st_size > 0
    assert paths.transmission_plot.stat().st_size > 0
    assert paths.absorption_plot.stat().st_size > 0
    assert paths.combined_plot.stat().st_size > 0


def test_show_true_uses_macos_backend_even_without_display_env(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)

    assert _show_method(show=True, backend="macosx", in_ipython=False) == "none"
    assert _show_method(show=True, backend="QtAgg", in_ipython=False) == "pyplot"
    assert _show_method(show=True, backend="module://matplotlib_inline.backend_inline", in_ipython=False) == "ipython"
    assert _show_method(show=True, backend="macosx", in_ipython=True) == "ipython"
    assert _show_method(show=True, backend="Agg", in_ipython=False) == "none"
    assert _show_method(show=True, backend="Agg", in_ipython=True) == "ipython"
    assert _show_method(show=False, backend="macosx", in_ipython=True) == "none"
