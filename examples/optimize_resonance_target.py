from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_backend import use_inline_backend_if_available
from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    export_simulation,
    optimize_resonance_target,
    simulate_spectrum,
    stack_with_thicknesses,
)

use_inline_backend_if_available()


air = Material.constant(1.0 + 0.0j, name="air")
low = Material.constant(1.45 + 0.0j, name="low_index")
high = Material.constant(2.1 + 0.0j, name="high_index")
cavity = Material.constant(1.6 + 0.0j, name="cavity")

layers = []
for _ in range(3):
    layers.extend(
        [
            Layer(material=high, thickness_nm=72.0),
            Layer(material=low, thickness_nm=103.0),
        ]
    )
layers.append(Layer(material=cavity, thickness_nm=190.0))
for _ in range(3):
    layers.extend(
        [
            Layer(material=low, thickness_nm=103.0),
            Layer(material=high, thickness_nm=72.0),
        ]
    )

stack = Stack(incident=air, layers=layers, substrate=air)
wavelengths = jnp.linspace(520.0, 720.0, 241)
cavity_layer_index = 6

optimization = optimize_resonance_target(
    stack,
    wavelengths_nm=wavelengths,
    target_wavelength_nm=620.0,
    target_q=40.0,
    spectrum="T",
    feature="peak",
    variable_layer_indices=(cavity_layer_index,),
    steps=80,
    learning_rate=5.0,
    wavelength_weight=20.0,
    q_weight=0.25,
    sharpness=8.0,
)

optimized_stack = stack_with_thicknesses(stack, optimization.thicknesses_nm)
optimized_spectrum = simulate_spectrum(
    optimized_stack,
    wavelengths_nm=wavelengths,
    angle_deg=0.0,
    polarization="s",
)

export_paths = export_simulation(
    stack=optimized_stack,
    result=optimized_spectrum,
    output_dir=Path(__file__).with_name("results"),
    simulation_name="target_resonance_optimization",
    show=True,
)
file_prefix = export_paths.spectra_txt.name.removesuffix("_spectra.txt")

(Path(__file__).with_name("results") / f"{file_prefix}_summary.txt").write_text(
    "\n".join(
        [
            "simulation_name: target_resonance_optimization",
            "target_spectrum: transmission peak",
            "target_wavelength_nm: 620",
            "target_q: 40",
            f"optimized_cavity_thickness_nm: {float(optimization.variable_thicknesses_nm[0]):.12g}",
            f"final_resonance_wavelength_nm: {optimization.resonance.resonance_wavelength_nm:.12g}",
            f"final_linewidth_nm: {optimization.resonance.linewidth_nm:.12g}",
            f"final_quality_factor: {optimization.resonance.quality_factor:.12g}",
            f"final_loss: {float(optimization.history[-1]):.12g}",
        ]
    )
    + "\n"
)
