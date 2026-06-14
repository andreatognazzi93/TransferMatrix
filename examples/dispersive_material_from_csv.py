from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_backend import use_inline_backend_if_available
from multilayer_tmm import Layer, Material, Stack, export_simulation, simulate_spectrum

use_inline_backend_if_available()


data_path = Path(__file__).with_name("gold_sample.csv")

air = Material.constant(1.0 + 0.0j, name="air")
gold = Material.from_csv(data_path, name="gold sample")
glass = Material.constant(1.5 + 0.0j, name="glass")

stack = Stack(
    incident=air,
    layers=[Layer(material=gold, thickness_nm=40.0)],
    substrate=glass,
)

wavelengths = jnp.linspace(420.0, 800.0, 20)
result = simulate_spectrum(
    stack,
    wavelengths_nm=wavelengths,
    angle_deg=15.0,
    polarization="s",
)

export_simulation(
    stack=stack,
    result=result,
    output_dir=Path(__file__).with_name("results"),
    simulation_name="gold_film_stack",
    show=True,
)
