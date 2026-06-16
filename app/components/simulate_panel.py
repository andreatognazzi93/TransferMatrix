"""Workflow-1 controls (ARCHITECTURE §6.6): run button, channel selection, status.

The ``SIMULATE_CHANNELS_INPUT`` checklist holds the subset of R/T/A to plot
(value is a list, consumed by ``simulate_callbacks.render_spectrum``). The
graph + resonance readout live in the shared results panel, not here.

Default UI language is English; pass ``lang="it"`` for Italian (§12).
"""

from __future__ import annotations

from dash import dcc, html

from app import config, ids, plots


def build_simulate_panel(lang: str = "en"):
    """Build the simulate workflow control panel.

    Args:
        lang: active UI language code (default ``"en"``). All display text is
            resolved from :func:`app.config.labels_for(lang)`.
    """
    labels = config.labels_for(lang)

    return html.Div(
        className="simulate-panel",
        children=[
            # Mini-sketch of the flat stack (§10), grouped=False, updated from
            # STACK_CONFIG_STORE by a thin callback.
            dcc.Graph(
                id=ids.SIMULATE_SKETCH_GRAPH,
                # NOTE: pass lang=lang here once gui-viz lands plots.sketch_figure(lang=) (§12.5).
                figure=plots.sketch_figure(
                    config.default_stack_config(),
                    angle_deg=config.DEFAULT_ANGLE_DEG,
                    grouped=False,
                    title=labels["sketch_title"],
                ),
            ),
            html.Div(
                className="simulate-controls",
                children=[
                    html.Label(labels["simulate_channels"]),
                    dcc.Checklist(
                        id=ids.SIMULATE_CHANNELS_INPUT,
                        options=config.options_for(config.SPECTRUM_VALUES, "ch_", lang),
                        value=list(config.SPECTRUM_VALUES),
                        className="simulate-channels",
                    ),
                    html.Button(
                        labels["simulate_run"],
                        id=ids.SIMULATE_BUTTON,
                        n_clicks=0,
                        className="run-button simulate-button",
                    ),
                    html.Div(
                        labels["simulate_status_ready"],
                        id=ids.SIMULATE_STATUS,
                        className="status-text",
                    ),
                ],
            ),
        ],
    )
