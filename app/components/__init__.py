"""Presentation-only component builders for the Dash GUI.

Each module exposes one builder returning a Dash component (ARCHITECTURE §6.6).
Component ids come exclusively from :mod:`app.ids`; Italian labels from
:mod:`app.config`. These builders contain **no** business logic — they declare
widgets with stable ids that the callback layer (:mod:`app.callbacks`) wires.
"""

from __future__ import annotations

from app.components.material_input import build_material_input
from app.components.optimize_panel import build_optimize_panel
from app.components.results_panel import build_results_panel
from app.components.simulate_panel import build_simulate_panel
from app.components.stack_builder import build_stack_builder

__all__ = [
    "build_material_input",
    "build_optimize_panel",
    "build_results_panel",
    "build_simulate_panel",
    "build_stack_builder",
]
