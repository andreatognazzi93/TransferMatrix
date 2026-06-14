from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_backend import use_inline_backend_if_available
from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    analyze_resonance,
    export_simulation,
    read_spectrum_csv,
    simulate_spectrum,
)

use_inline_backend_if_available()


benchmark_path = Path("/Users/andrea/Downloads/chart_data_and_script/data.csv")
results_dir = Path(__file__).with_name("results")
simulation_name = "dbr_fdtd_benchmark"

fdtd_wavelength_nm, fdtd_reflectivity = read_spectrum_csv(
    benchmark_path,
    wavelength_scale=1000.0,
)

air = Material.constant(1.0 + 0.0j, name="air")
n1 = Material.constant(1.5 + 0.0j, name="n1")
tio2 = Material.constant(2.5 + 0.0j, name="TiO2")

layers = []
for _ in range(8):
    layers.extend(
        [
            Layer(material=n1, thickness_nm=105.0),
            Layer(material=tio2, thickness_nm=63.0),
        ]
    )

stack = Stack(incident=air, layers=layers, substrate=tio2)
tmm_result = simulate_spectrum(
    stack,
    wavelengths_nm=fdtd_wavelength_nm,
    angle_deg=0.0,
    polarization="s",
)

export_paths = export_simulation(
    stack=stack,
    result=tmm_result,
    output_dir=results_dir,
    simulation_name=simulation_name,
    show=True,
)
file_prefix = export_paths.spectra_txt.name.removesuffix("_spectra.txt")

fdtd_resonance = analyze_resonance(
    fdtd_wavelength_nm,
    fdtd_reflectivity,
    feature="peak",
)
tmm_resonance = analyze_resonance(
    fdtd_wavelength_nm,
    tmm_result.R,
    feature="peak",
)

comparison = np.column_stack(
    (
        np.asarray(fdtd_wavelength_nm),
        np.asarray(fdtd_reflectivity),
        np.asarray(tmm_result.R),
        np.asarray(tmm_result.R) - np.asarray(fdtd_reflectivity),
    )
)
np.savetxt(
    results_dir / f"{file_prefix}_comparison.txt",
    comparison,
    header="wavelength_nm fdtd_reflectivity tmm_reflectivity tmm_minus_fdtd",
    comments="# ",
    fmt="%.12g",
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
ax.plot(fdtd_wavelength_nm, fdtd_reflectivity, label="FDTD", linewidth=2.0)
ax.plot(fdtd_wavelength_nm, tmm_result.R, label="TMM", linewidth=2.0)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectivity")
ax.set_title("DBR benchmark reflectivity")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(results_dir / f"{file_prefix}_reflectivity_comparison.png", dpi=180)
plt.close(fig)

(results_dir / f"{file_prefix}_resonance_summary.txt").write_text(
    "\n".join(
        [
            f"simulation_name: {simulation_name}",
            "structure: air / 8 x (n1 105 nm, TiO2 63 nm) / semi-infinite TiO2",
            "feature: reflectivity peak",
            f"fdtd_resonance_wavelength_nm: {fdtd_resonance.resonance_wavelength_nm:.12g}",
            f"fdtd_linewidth_nm: {fdtd_resonance.linewidth_nm:.12g}",
            f"fdtd_quality_factor: {fdtd_resonance.quality_factor:.12g}",
            f"tmm_resonance_wavelength_nm: {tmm_resonance.resonance_wavelength_nm:.12g}",
            f"tmm_linewidth_nm: {tmm_resonance.linewidth_nm:.12g}",
            f"tmm_quality_factor: {tmm_resonance.quality_factor:.12g}",
        ]
    )
    + "\n"
)
