"""Workflow-1 controls (ARCHITECTURE §6.6): run button, channel selection, status.

The ``SIMULATE_CHANNELS_INPUT`` checklist holds the subset of R/T/A to plot
(value is a list, consumed by ``simulate_callbacks.render_spectrum``). The
graph + resonance readout live in the shared results panel, not here.

Default UI language is English; pass ``lang="it"`` for Italian (§12).

Angle-sweep mode UI (ANGLE_MAP_CONTRACT §4, §8.4):
  - ``ids.SIMULATE_MODE_INPUT``      — dcc.RadioItems toggle "single" | "angle_map"
  - ``ids.SIMULATE_SINGLE_ANGLE_CONTAINER`` — placeholder div shown in single mode;
    the actual ANGLE_INPUT + POLARIZATION_INPUT live in the shared stack_builder,
    NOT here. This div is the show/hide callback's valid target for single mode.
  - ``ids.SIMULATE_ANGLE_SWEEP_CONTAINER``  — holds start/stop/step inputs; hidden
    by default (display:none). gui-core's show/hide callback (§8.4) toggles it.
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

    # Angle-sweep defaults are defined in config.DEFAULT_ANGLE_SWEEP (§3.1)
    # by gui-core. We pull them here so initial input values are consistent
    # with the store default without hard-coding literals.
    sweep_defaults = config.DEFAULT_ANGLE_SWEEP  # {"start_deg": 0.0, "stop_deg": 80.0, "step_deg": 1.0}

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
                    # ----------------------------------------------------------
                    # Simulation mode toggle (ANGLE_MAP_CONTRACT §4)
                    # config.SIM_MODE_VALUES = ("single", "angle_map")
                    # options built via config.options_for(..., "sim_mode_", lang)
                    # ----------------------------------------------------------
                    html.Label(labels["sim_mode_label"]),
                    dcc.RadioItems(
                        id=ids.SIMULATE_MODE_INPUT,
                        options=config.options_for(
                            config.SIM_MODE_VALUES, "sim_mode_", lang
                        ),
                        value="single",
                        className="simulate-mode-radio",
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "12px"},
                    ),

                    # ----------------------------------------------------------
                    # Single-angle placeholder container (ANGLE_MAP_CONTRACT §4,
                    # §8.4). The actual ANGLE_INPUT + POLARIZATION_INPUT are in
                    # the shared stack_builder (app/components/stack_builder.py)
                    # and are NOT duplicated here. This empty div gives the
                    # gui-core show/hide callback (§8.4) a valid Output target
                    # for the "single" branch so both container ids always exist
                    # on the page. It is shown in single mode (display:block,
                    # toggled by the callback) and hidden in angle_map mode.
                    # ----------------------------------------------------------
                    html.Div(
                        id=ids.SIMULATE_SINGLE_ANGLE_CONTAINER,
                        style={"display": "block"},
                    ),

                    # ----------------------------------------------------------
                    # Angle-sweep inputs container (ANGLE_MAP_CONTRACT §4, §8.4)
                    # Hidden by default (display:none) — the show/hide callback
                    # (§8.4) sets display:block when mode == "angle_map".
                    # ----------------------------------------------------------
                    html.Div(
                        id=ids.SIMULATE_ANGLE_SWEEP_CONTAINER,
                        style={"display": "none"},
                        className="angle-sweep-container",
                        children=[
                            html.Legend(labels["angle_sweep_section"]),
                            html.Div(
                                className="angle-sweep-row",
                                children=[
                                    html.Div(
                                        className="field-cell",
                                        children=[
                                            html.Label(labels["angle_start"]),
                                            dcc.Input(
                                                id=ids.SIMULATE_ANGLE_START_INPUT,
                                                type="number",
                                                value=sweep_defaults["start_deg"],
                                                min=0,
                                                max=90,
                                                step=0.5,
                                                debounce=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-cell",
                                        children=[
                                            html.Label(labels["angle_stop"]),
                                            dcc.Input(
                                                id=ids.SIMULATE_ANGLE_STOP_INPUT,
                                                type="number",
                                                value=sweep_defaults["stop_deg"],
                                                min=0,
                                                max=90,
                                                step=0.5,
                                                debounce=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="field-cell",
                                        children=[
                                            html.Label(labels["angle_step"]),
                                            dcc.Input(
                                                id=ids.SIMULATE_ANGLE_STEP_INPUT,
                                                type="number",
                                                value=sweep_defaults["step_deg"],
                                                min=0.1,
                                                step=0.5,
                                                debounce=True,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.P(
                                labels["angle_map_pol_hint"],
                                className="angle-sweep-hint",
                            ),
                        ],
                    ),

                    # ----------------------------------------------------------
                    # Channel checklist, run button, status (unchanged)
                    # ----------------------------------------------------------
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
