"""JAX transfer-matrix simulation for coherent multilayer thin films."""

from multilayer_tmm.io import ExportPaths, export_simulation, read_spectrum_csv
from multilayer_tmm.layers import Layer, Stack, stack_thicknesses, stack_with_thicknesses
from multilayer_tmm.materials import Material
from multilayer_tmm.optimize import (
    OptimizationResult,
    ResonanceOptimizationResult,
    mean_reflectivity,
    optimize_resonance_target,
    optimize_thicknesses,
    resonance_target_loss,
)
from multilayer_tmm.resonance import (
    ResonanceResult,
    SmoothResonanceMetrics,
    analyze_resonance,
    smooth_resonance_metrics,
)
from multilayer_tmm.tmm import (
    AngleMapResult,
    SimulationResult,
    simulate_angle_map,
    simulate_angle_map_arrays,
    simulate_spectrum,
    simulate_spectrum_arrays,
    stack_to_arrays,
)
from multilayer_tmm.utils import available_devices, print_jax_devices, wavelength_grid

__all__ = [
    "AngleMapResult",
    "Layer",
    "Material",
    "OptimizationResult",
    "ExportPaths",
    "ResonanceOptimizationResult",
    "ResonanceResult",
    "SimulationResult",
    "SmoothResonanceMetrics",
    "Stack",
    "analyze_resonance",
    "available_devices",
    "export_simulation",
    "mean_reflectivity",
    "optimize_resonance_target",
    "optimize_thicknesses",
    "print_jax_devices",
    "read_spectrum_csv",
    "resonance_target_loss",
    "simulate_angle_map",
    "simulate_angle_map_arrays",
    "simulate_spectrum",
    "simulate_spectrum_arrays",
    "smooth_resonance_metrics",
    "stack_thicknesses",
    "stack_to_arrays",
    "stack_with_thicknesses",
    "wavelength_grid",
]
