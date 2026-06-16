"""Mini-sketch callbacks (ARCHITECTURE §10).

Two thin adapters that rebuild the stack schematic on store changes:

- Simulazione tab: ``STACK_CONFIG_STORE`` (flat §2.2 dict) ->
  ``plots.sketch_figure(..., grouped=False)``.
- Ottimizzazione tab: ``OPT_STACK_CONFIG_STORE`` (grouped §9.1 dict) ->
  ``plots.sketch_figure(..., grouped=True)``.

The incidence angle is read from each store's own ``angle_deg`` field, so the
sketch live-updates on stack *or* angle changes (both flow through the store).
Callbacks contain no domain math — read store -> ``plots.sketch_figure`` ->
write ``figure``.
"""

from __future__ import annotations

from dash import Input, Output, State

from app import config, ids, plots


def register(app) -> None:
    """Register the two sketch-update callbacks."""

    @app.callback(
        Output(ids.SIMULATE_SKETCH_GRAPH, "figure"),
        Input(ids.STACK_CONFIG_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
    )
    def update_simulate_sketch(stack_config, lang):
        """Rebuild the flat-stack sketch from the Simulazione store."""

        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        if not stack_config:
            return plots.empty_figure(labels["empty_plot"], lang=lang)
        angle_deg = stack_config.get("angle_deg", config.DEFAULT_ANGLE_DEG)
        return plots.sketch_figure(
            stack_config,
            angle_deg=angle_deg,
            grouped=False,
            title=labels["sketch_title"],
            lang=lang,
        )

    @app.callback(
        Output(ids.OPTIMIZE_SKETCH_GRAPH, "figure"),
        Input(ids.OPT_STACK_CONFIG_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
    )
    def update_optimize_sketch(opt_stack_config, lang):
        """Rebuild the grouped/cavity sketch from the Ottimizzazione store."""

        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        if not opt_stack_config:
            return plots.empty_figure(labels["empty_plot"], lang=lang)
        angle_deg = opt_stack_config.get("angle_deg", config.DEFAULT_ANGLE_DEG)
        return plots.sketch_figure(
            opt_stack_config,
            angle_deg=angle_deg,
            grouped=True,
            title=labels["sketch_title"],
            lang=lang,
        )
