"""Shared output area (ARCHITECTURE §6.6): spectrum graph + resonance readout.

The resonance readout is a small key/value ``DataTable`` whose rows are produced
by ``simulate_callbacks.render_resonance_readout`` — its column ids are fixed to
``"grandezza"`` / ``"valore"`` to match those rows. The graph id is passed in so
the simulate tab can use the canonical :data:`app.ids.SIMULATE_GRAPH` id that the
render callback targets, while sub-widgets (resonance table, export) follow the
``results_id`` prefix contract.

Default UI language is English; pass ``lang="it"`` for Italian (§12).
"""

from __future__ import annotations

from dash import dash_table, dcc, html

from app import config, ids, plots


def build_results_panel(id_prefix: str, graph_id: str | None = None, lang: str = "en"):
    """Build the shared results panel for the given id prefix.

    Args:
        id_prefix: prefix for ``results_id`` sub-widgets (resonance table,
            export button).
        graph_id: explicit id for the spectrum :class:`dcc.Graph`. Defaults to
            ``results_id(prefix, "graph")`` when not supplied.
        lang: active UI language code (default ``"en"``). All display text is
            resolved from :func:`app.config.labels_for(lang)`.
    """
    labels = config.labels_for(lang)

    if graph_id is None:
        graph_id = ids.results_id(id_prefix, ids.RESULTS_GRAPH_SUFFIX)

    return html.Div(
        className="results-panel",
        children=[
            html.H4(labels["results"]),
            # NOTE: pass lang=lang once gui-viz lands plots.empty_figure(lang=) (§12.5).
            dcc.Graph(
                id=graph_id,
                figure=plots.empty_figure(labels["empty_plot"]),
            ),
            html.H4(labels["resonance"]),
            dash_table.DataTable(
                id=ids.results_id(id_prefix, ids.RESULTS_RESONANCE_TABLE_SUFFIX),
                columns=[
                    {"name": labels["res_table_metric"], "id": "grandezza"},
                    {"name": labels["res_table_value"], "id": "valore"},
                ],
                data=[],
                style_cell={
                    "textAlign": "left",
                    "padding": "1px 6px",
                    "fontSize": "0.8rem",
                    "maxWidth": "160px",
                    "whiteSpace": "normal",
                },
                style_table={"overflowX": "auto", "maxWidth": "320px"},
            ),
            # Export the spectra + parameters as one ZIP of two .txt files
            # (browsers drop the 2nd of two simultaneous downloads); wired in
            # simulate_callbacks.export_simulation for the simulate tab.
            html.Div(
                className="results-export-row",
                children=[
                    html.Button(
                        labels["export"],
                        id=ids.results_id(id_prefix, ids.RESULTS_EXPORT_BUTTON_SUFFIX),
                        n_clicks=0,
                        className="export-button",
                    ),
                    html.Span(
                        "",
                        id=ids.results_id(id_prefix, ids.RESULTS_EXPORT_STATUS_SUFFIX),
                        className="status-text",
                    ),
                ],
            ),
            dcc.Download(id=ids.results_id(id_prefix, ids.RESULTS_EXPORT_DOWNLOAD_SUFFIX)),
        ],
    )
