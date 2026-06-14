from pathlib import Path
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_backend import use_inline_backend_if_available
from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    export_simulation,
    optimize_thicknesses,
    simulate_spectrum,
    stack_with_thicknesses,
)

use_inline_backend_if_available()


air = Material.constant(1.0 + 0.0j, name="air")
mgf2 = Material.constant(1.38 + 0.0j, name="MgF2")
glass = Material.constant(1.52 + 0.0j, name="glass")

base_stack = Stack(
    incident=air,
    layers=[Layer(material=mgf2, thickness_nm=80.0)],
    substrate=glass,
)
wavelengths = jnp.linspace(400.0, 700.0, 121)


def objective(thicknesses_nm):
    stack = stack_with_thicknesses(base_stack, thicknesses_nm)
    return jnp.mean(
        simulate_spectrum(
            stack,
            wavelengths_nm=wavelengths,
            angle_deg=0.0,
            polarization="s",
        ).R
    )


initial_thicknesses = jnp.array([80.0])
initial_value, initial_gradient = jax.value_and_grad(objective)(initial_thicknesses)

result = optimize_thicknesses(
    objective,
    initial_thicknesses_nm=initial_thicknesses,
    steps=120,
    learning_rate=0.25,
    lower_bound_nm=1.0,
)

optimized_stack = stack_with_thicknesses(base_stack, result.thicknesses_nm)
optimized_spectrum = simulate_spectrum(
    optimized_stack,
    wavelengths_nm=wavelengths,
    angle_deg=0.0,
    polarization="s",
)

export_simulation(
    stack=optimized_stack,
    result=optimized_spectrum,
    output_dir=Path(__file__).with_name("results"),
    simulation_name="optimized_antireflection",
    show=True,
)
