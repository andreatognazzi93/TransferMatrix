from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_backend import use_inline_backend_if_available
from multilayer_tmm import Layer, Material, Stack, export_simulation, simulate_spectrum

use_inline_backend_if_available()


air = Material.constant(1.0 + 0.0j, name="air")
coating = Material.constant(1.38 + 0.0j, name="MgF2")
glass = Material.constant(1.52 + 0.0j, name="glass")

stack = Stack(
    incident=air,
    layers=[Layer(material=coating, thickness_nm=100.0)],
    substrate=glass,
)

wavelengths = jnp.linspace(400.0, 700.0, 11)
result = simulate_spectrum(
    stack,
    wavelengths_nm=wavelengths,
    angle_deg=0.0,
    polarization="s",
)

export_simulation(
    stack=stack,
    result=result,
    output_dir=Path(__file__).with_name("results"),
    simulation_name="basic_stack",
    show=True,
)
