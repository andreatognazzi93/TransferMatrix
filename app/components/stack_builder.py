"""Shared stack editor (ARCHITECTURE §3): incident/substrate material inputs,
the dynamic finite-layer ``DataTable``, and the grid/angle/polarization controls.

A single instance of this builder is embedded by both tabs — that is the
"one stack-builder shared by Simulazione and Ottimizzazione" requirement.
The DataTable rows round-trip losslessly into ``stack_config["layers"]`` via
the column ids declared in :mod:`app.ids`.

Default UI language is English; pass ``lang="it"`` for Italian (§12).
"""

from __future__ import annotations

from dash import dash_table, dcc, html

from app import config, ids
from app.components.material_input import build_material_input


def _default_layer_rows() -> list[dict]:
    """Initial DataTable rows mirroring :func:`config.default_stack_config`."""

    return [
        {
            ids.LAYER_COL_MATERIAL_KIND: "constant",
            ids.LAYER_COL_N: config.DEFAULT_LAYER_MATERIAL["n"],
            ids.LAYER_COL_K: config.DEFAULT_LAYER_MATERIAL["k"],
            ids.LAYER_COL_CSV_NAME: None,
            ids.LAYER_COL_THICKNESS: config.DEFAULT_LAYER_THICKNESS_NM,
        }
    ]


def _layer_table(lang: str = "en"):
    """Editable finite-layer table (material kind/n/k/csv + thickness)."""
    labels = config.labels_for(lang)

    return dash_table.DataTable(
        id=ids.LAYER_TABLE,
        data=_default_layer_rows(),
        editable=True,
        row_deletable=True,
        columns=[
            {
                "name": labels["material_kind"],
                "id": ids.LAYER_COL_MATERIAL_KIND,
                "presentation": "dropdown",
            },
            {
                "name": labels["refractive_index_n"],
                "id": ids.LAYER_COL_N,
                "type": "numeric",
            },
            {
                "name": labels["extinction_k"],
                "id": ids.LAYER_COL_K,
                "type": "numeric",
            },
            {
                "name": labels["material_name"],
                "id": ids.LAYER_COL_CSV_NAME,
                "presentation": "dropdown",
            },
            {
                "name": labels["thickness_nm"],
                "id": ids.LAYER_COL_THICKNESS,
                "type": "numeric",
            },
        ],
        dropdown={
            ids.LAYER_COL_MATERIAL_KIND: {
                "options": config.options_for(config.MATERIAL_KIND_VALUES, "matkind_", lang),
                "clearable": False,
            },
            # csv_name options are populated from material_library_store by a
            # callback registered in app.main (per-row dropdown of uploaded CSVs).
            ids.LAYER_COL_CSV_NAME: {"options": []},
        },
        style_table={"overflowX": "auto", "maxWidth": "520px"},
        style_cell={
            "textAlign": "center",
            "padding": "1px 4px",
            "fontSize": "0.8rem",
            "minWidth": "55px",
            "width": "85px",
            "maxWidth": "110px",
            "whiteSpace": "normal",
        },
        # Widen the "Material type" column so its header/value isn't clipped.
        style_cell_conditional=[
            {
                "if": {"column_id": ids.LAYER_COL_MATERIAL_KIND},
                "minWidth": "120px",
                "width": "130px",
                "maxWidth": "150px",
            },
        ],
    )


def build_stack_builder(lang: str = "en"):
    """Build the shared stack editor component (ARCHITECTURE §6.6).

    Args:
        lang: active UI language code (default ``"en"``). All display text is
            resolved from :func:`app.config.labels_for(lang)`.
    """
    labels = config.labels_for(lang)

    return html.Div(
        className="stack-builder",
        children=[
            build_material_input(ids.INCIDENT_MATERIAL_PREFIX, labels["incident"], lang=lang),
            html.Div(
                className="layers-section",
                children=[
                    html.H4(labels["layers"]),
                    _layer_table(lang=lang),
                    html.Button(
                        labels["add_layer"],
                        id=ids.ADD_LAYER_BUTTON,
                        n_clicks=0,
                        className="add-layer-button",
                    ),
                ],
            ),
            build_material_input(ids.SUBSTRATE_MATERIAL_PREFIX, labels["substrate"], lang=lang),
            html.Fieldset(
                className="grid-controls",
                children=[
                    html.Legend(labels["grid_section_legend"]),
                    html.Div(
                        className="grid-row",
                        children=[
                            html.Div(
                                className="field-cell",
                                children=[
                                    html.Label(labels["grid_start"]),
                                    dcc.Input(
                                        id=ids.GRID_START_INPUT,
                                        type="number",
                                        value=config.DEFAULT_GRID["start_nm"],
                                        step=1,
                                        debounce=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="field-cell",
                                children=[
                                    html.Label(labels["grid_stop"]),
                                    dcc.Input(
                                        id=ids.GRID_STOP_INPUT,
                                        type="number",
                                        value=config.DEFAULT_GRID["stop_nm"],
                                        step=1,
                                        debounce=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="field-cell",
                                children=[
                                    html.Label(labels["grid_num"]),
                                    dcc.Input(
                                        id=ids.GRID_NUM_INPUT,
                                        type="number",
                                        value=config.DEFAULT_GRID["num"],
                                        min=2,
                                        step=1,
                                        debounce=True,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="grid-row",
                        children=[
                            html.Div(
                                className="field-cell",
                                children=[
                                    html.Label(labels["angle_deg"]),
                                    dcc.Input(
                                        id=ids.ANGLE_INPUT,
                                        type="number",
                                        value=config.DEFAULT_ANGLE_DEG,
                                        step=1,
                                        debounce=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="field-cell",
                                children=[
                                    html.Label(labels["polarization"]),
                                    dcc.Dropdown(
                                        id=ids.POLARIZATION_INPUT,
                                        options=config.options_for(
                                            config.POLARIZATION_VALUES, "pol_", lang
                                        ),
                                        value=config.DEFAULT_POLARIZATION,
                                        clearable=False,
                                        className="polarization-dropdown",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
