"""Workflow-2 controls (ARCHITECTURE §4, §6.6, §9.4, §10, §12).

Hosts two things:

1. The **grouped/cavity stack editor** (§9.4) that is the source of truth for
   the Ottimizzazione tab's structure (``opt_stack_config_store``, §9.1):
   incident/substrate ``material_input``, two period-definition ``DataTable``s
   (input/output mirror groups) with M/K repeat integer inputs, a cavity row
   (``material_input`` + thickness + enabled toggle), grid/angle/polarization
   controls, and the ``variable`` selector (default = cavity only).
2. The **optimize controls** (mode, target lambda/Q, feature, steps, lr,
   weights, sharpness) and outputs (loss-history graph, optimized-spectrum
   graph, thickness readout, status, progress placeholder), plus a mini-sketch
   ``dcc.Graph`` (§10) rebuilt with ``grouped=True``.

Presentation only — the grouped-config sync + sketch callbacks live in
:mod:`app.callbacks`.

Default UI language is English; pass ``lang="it"`` for Italian (§12).
Each optimize-control field label carries a CSS-only "?" help tooltip (§12.7a).
"""

from __future__ import annotations

from dash import dash_table, dcc, html

from app import config, ids, plots
from app.components.material_input import build_material_input


# ---------------------------------------------------------------------------
# Help-icon builder (§12, Task 6)
# ---------------------------------------------------------------------------

def _help_icon(field: str, tip_text: str) -> html.Span:
    """Render a keyboard-focusable "?" badge with a CSS hover/focus tooltip.

    Markup (§12.7a)::

        <span class="help" tabindex="0" id={help_icon_id(field)}>?
          <span class="help-text" role="tooltip">{tip_text}</span>
        </span>

    The ``id`` is a pattern-matching dict (``ids.help_icon_id(field)``) for
    uniqueness; no callback ever targets these ids.
    """
    return html.Span(
        id=ids.help_icon_id(field),
        className="help",
        tabIndex=0,
        children=[
            "?",
            html.Span(tip_text, className="help-text", role="tooltip"),
        ],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _num(label_text: str, input_id: str, value, *, step=None, mn=None,
         tip_field: str | None = None, tip_text: str = ""):
    """Small labelled numeric input row, optionally with a help icon.

    Args:
        label_text: already-localized label string.
        input_id: Dash component id for the ``dcc.Input``.
        value: initial numeric value.
        step: optional step for the numeric input.
        mn: optional minimum for the numeric input.
        tip_field: if set, append a ``_help_icon(tip_field, tip_text)`` after
            the label. ``tip_text`` must be the translated tooltip string.
        tip_text: tooltip body text (resolved from the catalog by the caller).
    """
    kwargs: dict = {"type": "number", "value": value, "debounce": True}
    if step is not None:
        kwargs["step"] = step
    if mn is not None:
        kwargs["min"] = mn

    label_children: list = [label_text]
    if tip_field:
        label_children.append(_help_icon(tip_field, tip_text))

    return html.Div(
        className="optimize-row",
        children=[
            html.Span(className="opt-label", children=label_children),
            dcc.Input(id=input_id, **kwargs),
        ],
    )


def _period_rows(layers) -> list[dict]:
    """Convert §9.1 period ``layers`` into DataTable rows."""

    rows = []
    for layer in layers:
        material = layer.get("material", {})
        rows.append(
            {
                ids.LAYER_COL_MATERIAL_KIND: material.get("kind", "constant"),
                ids.LAYER_COL_N: material.get("n", 1.0),
                ids.LAYER_COL_K: material.get("k", 0.0),
                ids.LAYER_COL_CSV_NAME: material.get("name")
                if material.get("kind") == "csv"
                else None,
                ids.LAYER_COL_THICKNESS: layer.get("thickness_nm", config.DEFAULT_LAYER_THICKNESS_NM),
            }
        )
    return rows


def _group_table(table_id: str, rows: list[dict], lang: str = "en"):
    """Editable period-definition table (same column set as §3 layer table)."""
    labels = config.labels_for(lang)

    return dash_table.DataTable(
        id=table_id,
        data=rows,
        editable=True,
        row_deletable=True,
        columns=[
            {
                "name": labels["material_kind"],
                "id": ids.LAYER_COL_MATERIAL_KIND,
                "presentation": "dropdown",
            },
            {"name": labels["refractive_index_n"], "id": ids.LAYER_COL_N, "type": "numeric"},
            {"name": labels["extinction_k"], "id": ids.LAYER_COL_K, "type": "numeric"},
            {
                "name": labels["material_name"],
                "id": ids.LAYER_COL_CSV_NAME,
                "presentation": "dropdown",
            },
            {"name": labels["thickness_nm"], "id": ids.LAYER_COL_THICKNESS, "type": "numeric"},
        ],
        dropdown={
            ids.LAYER_COL_MATERIAL_KIND: {
                "options": config.options_for(config.MATERIAL_KIND_VALUES, "matkind_", lang),
                "clearable": False,
            },
            ids.LAYER_COL_CSV_NAME: {"options": []},
        },
        style_table={"overflowX": "auto", "maxWidth": "460px"},
        style_cell={
            "textAlign": "center",
            "padding": "1px 4px",
            "fontSize": "0.8rem",
            "minWidth": "55px",
            "width": "85px",
            "maxWidth": "110px",
            "whiteSpace": "normal",
        },
    )


def _variable_layer_options(layers) -> list[dict]:
    """Checklist options for selecting period-layer indices as variables."""

    options = []
    for index, layer in enumerate(layers):
        name = (layer.get("material") or {}).get("name") or f"layer {index}"
        options.append({"label": f"{index}: {name}", "value": index})
    return options


def _grouped_stack_editor(lang: str = "en"):
    """Build the grouped/cavity stack editor (§9.4)."""
    labels = config.labels_for(lang)
    cfg = config.default_opt_stack_config()
    input_layers = cfg["input_group"]["layers"]
    output_layers = cfg["output_group"]["layers"]

    return html.Div(
        className="opt-stack-editor",
        children=[
            html.H4(labels["opt_stack_title"]),
            build_material_input(
                ids.OPT_INCIDENT_MATERIAL_PREFIX, labels["incident"], lang=lang
            ),
            # --- input mirror group ---
            html.Div(
                className="opt-group-section",
                children=[
                    html.H5(labels["opt_input_group"]),
                    _group_table(ids.OPT_INPUT_GROUP_TABLE, _period_rows(input_layers), lang=lang),
                    html.Button(
                        labels["add_layer"],
                        id=ids.OPT_INPUT_ADD_LAYER_BUTTON,
                        n_clicks=0,
                        className="add-layer-button",
                    ),
                    html.Div(
                        className="optimize-row",
                        children=[
                            html.Label(labels["opt_input_repeat"]),
                            dcc.Input(
                                id=ids.OPT_INPUT_REPEAT_INPUT,
                                type="number",
                                value=cfg["input_group"]["repeat"],
                                min=0,
                                step=1,
                                debounce=True,
                            ),
                        ],
                    ),
                ],
            ),
            # --- cavity ---
            html.Div(
                className="opt-cavity-section",
                children=[
                    html.H5(labels["opt_cavity"]),
                    build_material_input(
                        ids.OPT_CAVITY_MATERIAL_PREFIX, labels["opt_cavity"], lang=lang
                    ),
                    html.Div(
                        className="optimize-row",
                        children=[
                            html.Label(labels["opt_cavity_thickness"]),
                            dcc.Input(
                                id=ids.OPT_CAVITY_THICKNESS_INPUT,
                                type="number",
                                value=cfg["cavity"]["thickness_nm"],
                                step=1,
                                min=0,
                                debounce=True,
                            ),
                        ],
                    ),
                    dcc.Checklist(
                        id=ids.OPT_CAVITY_ENABLED_INPUT,
                        options=[{"label": labels["opt_cavity_enabled"], "value": "enabled"}],
                        value=["enabled"] if cfg["cavity"]["enabled"] else [],
                        className="opt-cavity-enabled",
                    ),
                ],
            ),
            # --- output mirror group ---
            html.Div(
                className="opt-group-section",
                children=[
                    html.H5(labels["opt_output_group"]),
                    _group_table(
                        ids.OPT_OUTPUT_GROUP_TABLE, _period_rows(output_layers), lang=lang
                    ),
                    html.Button(
                        labels["add_layer"],
                        id=ids.OPT_OUTPUT_ADD_LAYER_BUTTON,
                        n_clicks=0,
                        className="add-layer-button",
                    ),
                    html.Div(
                        className="optimize-row",
                        children=[
                            html.Label(labels["opt_output_repeat"]),
                            dcc.Input(
                                id=ids.OPT_OUTPUT_REPEAT_INPUT,
                                type="number",
                                value=cfg["output_group"]["repeat"],
                                min=0,
                                step=1,
                                debounce=True,
                            ),
                        ],
                    ),
                ],
            ),
            build_material_input(
                ids.OPT_SUBSTRATE_MATERIAL_PREFIX, labels["substrate"], lang=lang
            ),
            # --- grid / angle / polarization ---
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
                                        id=ids.OPT_GRID_START_INPUT,
                                        type="number",
                                        value=cfg["grid"]["start_nm"],
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
                                        id=ids.OPT_GRID_STOP_INPUT,
                                        type="number",
                                        value=cfg["grid"]["stop_nm"],
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
                                        id=ids.OPT_GRID_NUM_INPUT,
                                        type="number",
                                        value=cfg["grid"]["num"],
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
                                        id=ids.OPT_ANGLE_INPUT,
                                        type="number",
                                        value=cfg["angle_deg"],
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
                                        id=ids.OPT_POLARIZATION_INPUT,
                                        # "both" is rejected by the optimizers (§6.1); offer s/p only.
                                        options=config.options_for(
                                            tuple(v for v in config.POLARIZATION_VALUES if v != "both"),
                                            "pol_",
                                            lang,
                                        ),
                                        value=cfg["polarization"],
                                        clearable=False,
                                        className="polarization-dropdown",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # --- variable selector (default: cavity only) ---
            # §11: two tabs ("Per periodo" / "Per singolo strato") inside the
            # existing fieldset. The active tab value mirrors
            # opt_stack_config_store["variable"]["mode"] via stack_callbacks.py.
            html.Fieldset(
                className="opt-variable-section",
                children=[
                    html.Legend(labels["opt_variable_section"]),
                    dcc.Tabs(
                        id=ids.OPT_VARIABLE_MODE_TABS,
                        value=ids.OPT_VARIABLE_MODE_TIED,
                        children=[
                            # --- Tab 1: Per periodo (tied mode) ---
                            dcc.Tab(
                                label=labels["opt_variable_mode_tied"],
                                value=ids.OPT_VARIABLE_MODE_TIED,
                                children=[
                                    dcc.Checklist(
                                        id=ids.OPT_VARIABLE_CAVITY_INPUT,
                                        options=[
                                            {
                                                "label": labels["opt_variable_cavity"],
                                                "value": "cavity",
                                            }
                                        ],
                                        value=["cavity"] if cfg["variable"]["cavity"] else [],
                                    ),
                                    html.Div(
                                        className="optimize-row",
                                        children=[
                                            html.Label(
                                                labels["opt_variable_input_layers"]
                                            ),
                                            dcc.Checklist(
                                                id=ids.OPT_VARIABLE_INPUT_LAYERS_INPUT,
                                                options=_variable_layer_options(input_layers),
                                                value=list(cfg["variable"]["input_layers"]),
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="optimize-row",
                                        children=[
                                            html.Label(
                                                labels["opt_variable_output_layers"]
                                            ),
                                            dcc.Checklist(
                                                id=ids.OPT_VARIABLE_OUTPUT_LAYERS_INPUT,
                                                options=_variable_layer_options(output_layers),
                                                value=list(cfg["variable"]["output_layers"]),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # --- Tab 2: Per singolo strato (independent mode) ---
                            dcc.Tab(
                                label=labels["opt_variable_mode_independent"],
                                value=ids.OPT_VARIABLE_MODE_INDEPENDENT,
                                children=[
                                    html.Label(labels["opt_variable_flat_layers"]),
                                    # Options are populated dynamically by gui-core's
                                    # _flat_layer_options callback in stack_callbacks.py.
                                    # This component starts empty; do not add options here.
                                    dcc.Checklist(
                                        id=ids.OPT_VARIABLE_FLAT_LAYERS_INPUT,
                                        options=[],
                                        value=[],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _labeled_row_with_icon(label_text: str, tip_field: str, tip_text: str,
                            input_component, *, wide: bool = False) -> html.Div:
    """Row with label + help icon + an arbitrary Dash input component.

    Used for dropdown/radio rows that cannot use ``_num`` (which is
    number-input only). The label wraps both the text and the icon in an
    ``opt-label`` span.

    When ``wide`` is True the row carries the ``optimize-row--wide`` modifier
    so the CSS grid lets it span the full width (used for the mode radio).
    """
    return html.Div(
        className="optimize-row optimize-row--wide" if wide else "optimize-row",
        children=[
            html.Span(
                className="opt-label",
                children=[
                    label_text,
                    _help_icon(tip_field, tip_text),
                ],
            ),
            input_component,
        ],
    )


def _optimize_controls(lang: str = "en"):
    """Build the optimize scalar-control column (mode/targets/steps/etc.)."""
    labels = config.labels_for(lang)
    defaults = config.default_optimize_config()

    return html.Div(
        className="optimize-controls",
        children=[
            # ---- mode selector with help icon (spans full width) ----
            _labeled_row_with_icon(
                labels["optimize_mode"],
                "mode",
                labels["tip_mode"],
                dcc.RadioItems(
                    id=ids.OPTIMIZE_MODE_INPUT,
                    options=config.options_for(config.OPTIMIZE_MODE_VALUES, "optmode_", lang),
                    value="resonance",
                    className="optimize-mode",
                ),
                wide=True,
            ),
            # ---- spectral channel with help icon ----
            _labeled_row_with_icon(
                labels["optimize_spectrum"],
                "spectrum",
                labels["tip_spectrum"],
                dcc.Dropdown(
                    id=ids.OPTIMIZE_SPECTRUM_INPUT,
                    # Spectrum display keys are "ch_R", "ch_T", "ch_A" in the catalog.
                    options=config.options_for(config.SPECTRUM_VALUES, "ch_", lang),
                    value=defaults["spectrum"],
                    clearable=False,
                ),
            ),
            # ---- feature with help icon ----
            _labeled_row_with_icon(
                labels["optimize_feature"],
                "feature",
                labels["tip_feature"],
                dcc.Dropdown(
                    id=ids.OPTIMIZE_FEATURE_INPUT,
                    options=config.options_for(config.FEATURE_VALUES, "feat_", lang),
                    value=defaults["feature"],
                    clearable=False,
                ),
            ),
            # ---- resonance-mode target fields ----
            _num(
                labels["optimize_target_wavelength"],
                ids.OPTIMIZE_TARGET_WAVELENGTH_INPUT,
                defaults["target_wavelength_nm"],
                step=1,
                tip_field="target_wavelength",
                tip_text=labels["tip_target_wavelength"],
            ),
            _num(
                labels["optimize_target_q"],
                ids.OPTIMIZE_TARGET_Q_INPUT,
                defaults["target_q"],
                step=1,
                mn=0,
                tip_field="target_q",
                tip_text=labels["tip_target_q"],
            ),
            _num(
                labels["optimize_wavelength_weight"],
                ids.OPTIMIZE_WAVELENGTH_WEIGHT_INPUT,
                defaults["wavelength_weight"],
                step=0.1,
                mn=0,
                tip_field="wavelength_weight",
                tip_text=labels["tip_wavelength_weight"],
            ),
            _num(
                labels["optimize_q_weight"],
                ids.OPTIMIZE_Q_WEIGHT_INPUT,
                defaults["q_weight"],
                step=0.1,
                mn=0,
                tip_field="q_weight",
                tip_text=labels["tip_q_weight"],
            ),
            _num(
                labels["optimize_sharpness"],
                ids.OPTIMIZE_SHARPNESS_INPUT,
                defaults["sharpness"],
                step=1,
                mn=0,
                tip_field="sharpness",
                tip_text=labels["tip_sharpness"],
            ),
            # ---- shared optimization controls ----
            _num(
                labels["optimize_steps"],
                ids.OPTIMIZE_STEPS_INPUT,
                defaults["steps"],
                step=1,
                mn=1,
                tip_field="steps",
                tip_text=labels["tip_steps"],
            ),
            _num(
                labels["optimize_learning_rate"],
                ids.OPTIMIZE_LEARNING_RATE_INPUT,
                defaults["learning_rate"],
                step=0.01,
                mn=0,
                tip_field="learning_rate",
                tip_text=labels["tip_learning_rate"],
            ),
            _num(
                labels["optimize_lower_bound"],
                ids.OPTIMIZE_LOWER_BOUND_INPUT,
                defaults["lower_bound_nm"],
                step=1,
                mn=0,
                tip_field="lower_bound",
                tip_text=labels["tip_lower_bound"],
            ),
            html.Button(
                labels["optimize_run"],
                id=ids.OPTIMIZE_BUTTON,
                n_clicks=0,
                className="run-button optimize-button",
            ),
            html.Div(
                labels["optimize_status_ready"],
                id=ids.OPTIMIZE_STATUS,
                className="status-text",
            ),
            html.Div(id=ids.OPTIMIZE_PROGRESS_BAR, className="optimize-progress"),
        ],
    )


def build_optimize_panel(lang: str = "en"):
    """Build the optimize workflow control + output panel (§9.4 + §10).

    Args:
        lang: active UI language code (default ``"en"``). All display text is
            resolved from :func:`app.config.labels_for(lang)`.
    """
    labels = config.labels_for(lang)

    return html.Div(
        className="optimize-panel",
        children=[
            _grouped_stack_editor(lang=lang),
            # Mini-sketch of the grouped/cavity stack (§10), grouped=True.
            # NOTE: pass lang=lang once gui-viz lands plots.sketch_figure(lang=) (§12.5).
            dcc.Graph(
                id=ids.OPTIMIZE_SKETCH_GRAPH,
                figure=plots.sketch_figure(
                    config.default_opt_stack_config(),
                    angle_deg=config.DEFAULT_ANGLE_DEG,
                    grouped=True,
                    title=labels["sketch_title"],
                ),
            ),
            _optimize_controls(lang=lang),
            html.Div(
                className="optimize-output",
                children=[
                    html.Div(id=ids.OPTIMIZE_THICKNESS_READOUT, className="thickness-readout"),
                    # Export the optimized result as a single ZIP holding two
                    # connected .txt files (spectra + parameters); see
                    # optimize_callbacks.export_optimization.
                    html.Div(
                        className="optimize-export-row",
                        children=[
                            html.Button(
                                labels["optimize_export"],
                                id=ids.OPTIMIZE_EXPORT_BUTTON,
                                n_clicks=0,
                                className="export-button",
                            ),
                            html.Span(
                                "",
                                id=ids.OPTIMIZE_EXPORT_STATUS,
                                className="status-text",
                            ),
                        ],
                    ),
                    dcc.Download(id=ids.OPTIMIZE_EXPORT_DOWNLOAD),
                    # NOTE: pass lang=lang once gui-viz lands plots.empty_figure(lang=) (§12.5).
                    dcc.Graph(
                        id=ids.OPTIMIZE_HISTORY_GRAPH,
                        figure=plots.empty_figure(labels["empty_plot"]),
                    ),
                    dcc.Graph(
                        id=ids.OPTIMIZE_RESULT_GRAPH,
                        figure=plots.empty_figure(labels["empty_plot"]),
                    ),
                ],
            ),
        ],
    )
