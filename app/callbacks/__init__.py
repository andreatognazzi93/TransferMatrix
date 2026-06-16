"""Callback registration entrypoint (ARCHITECTURE §6.5).

``register_callbacks(app, cache)`` is the single public entrypoint the app
factory in ``app.main`` calls. It delegates to the submodules, each of which
registers a thin set of adapter callbacks:

- :mod:`app.callbacks.stack_callbacks`   — keep ``stack_config_store`` in sync
  with the stack-builder widgets (incident/substrate/grid/angle/pol + the
  finite-layer DataTable).
- :mod:`app.callbacks.simulate_callbacks` — workflow 1: read Stores ->
  ``state.run_simulation`` -> ``plots.spectrum_figure`` -> write Stores.
- :mod:`app.callbacks.optimize_callbacks` — workflow 2: background callback
  running ``state.run_resonance_optimization`` / ``run_thickness_optimization``,
  then ``plots.history_figure`` / ``plots.resonance_overlay_figure``.
- :mod:`app.callbacks.sketch_callbacks` — §10 mini-sketch: rebuild
  ``plots.sketch_figure`` on stack-store change for both tabs.

Adapters contain no domain math — they only read Stores, call ``app.state``,
call ``app.plots``, and write Stores.
"""

from __future__ import annotations

from app.callbacks import (
    optimize_callbacks,
    simulate_callbacks,
    sketch_callbacks,
    stack_callbacks,
)


def register_callbacks(app, cache) -> None:
    """Register all GUI callbacks on ``app``.

    Args:
        app: the ``dash.Dash`` instance.
        cache: the ``DiskcacheManager`` used for background callbacks
            (ARCHITECTURE §4). Passed through to the optimize submodule.
    """

    stack_callbacks.register(app)
    simulate_callbacks.register(app)
    optimize_callbacks.register(app, cache)
    sketch_callbacks.register(app)


__all__ = ["register_callbacks"]
