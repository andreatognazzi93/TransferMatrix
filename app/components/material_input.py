"""Single material editor: constant (n, k) OR CSV upload (ARCHITECTURE §5).

Reused for the incident and substrate media. Callable materials are **not**
offered — only ``constant`` and ``csv`` (the discriminator values from
:data:`app.config.MATERIAL_KIND_VALUES`). The CSV ``dcc.Upload`` is parsed
in-memory by the upload callback and written into ``material_library_store``
keyed by the material name; the stack-sync callback then resolves csv-kind
materials by that name.

Default UI language is English; pass ``lang="it"`` for Italian (§12).
"""

from __future__ import annotations

from dash import dcc, html

from app import config, ids


def build_material_input(id_prefix: str, label: str, lang: str = "en"):
    """Build one material editor with the given id prefix and label text.

    Sub-widget ids follow ``f"{id_prefix}_{suffix}"`` (see
    :func:`app.ids.material_id`) so the stack-sync callback can read them.

    Args:
        id_prefix: unique prefix string, e.g. ``ids.INCIDENT_MATERIAL_PREFIX``.
        label: already-localized legend text; the caller resolves this from
            ``config.labels_for(lang)`` before passing it in.
        lang: active UI language (default ``"en"``); used for option-list
            display text and placeholder strings inside this widget.
    """
    labels = config.labels_for(lang)

    return html.Fieldset(
        className="material-input",
        children=[
            html.Legend(label),
            html.Div(
                className="material-input-row",
                children=[
                    html.Label(labels["material_kind"]),
                    dcc.Dropdown(
                        id=ids.material_id(id_prefix, ids.MATERIAL_KIND_SUFFIX),
                        options=config.options_for(config.MATERIAL_KIND_VALUES, "matkind_", lang),
                        value="constant",
                        clearable=False,
                        className="material-kind-dropdown",
                    ),
                ],
            ),
            html.Div(
                className="material-input-row",
                children=[
                    html.Label(labels["material_name"]),
                    dcc.Input(
                        id=ids.material_id(id_prefix, ids.MATERIAL_NAME_SUFFIX),
                        type="text",
                        value=None,
                        debounce=True,
                        placeholder=labels["material_name"],
                    ),
                ],
            ),
            # --- constant (n, k) inputs ---
            html.Div(
                className="material-constant-fields",
                children=[
                    html.Div(
                        className="material-input-row",
                        children=[
                            html.Label(labels["refractive_index_n"]),
                            dcc.Input(
                                id=ids.material_id(id_prefix, ids.MATERIAL_N_SUFFIX),
                                type="number",
                                value=1.0,
                                step=0.01,
                                debounce=True,
                            ),
                        ],
                    ),
                    html.Div(
                        className="material-input-row",
                        children=[
                            html.Label(labels["extinction_k"]),
                            dcc.Input(
                                id=ids.material_id(id_prefix, ids.MATERIAL_K_SUFFIX),
                                type="number",
                                value=0.0,
                                step=0.01,
                                debounce=True,
                            ),
                        ],
                    ),
                ],
            ),
            # --- CSV upload ---
            html.Div(
                className="material-csv-fields",
                children=[
                    dcc.Upload(
                        id=ids.material_id(id_prefix, ids.MATERIAL_UPLOAD_SUFFIX),
                        children=html.Div(labels["upload_csv_hint"]),
                        accept=".csv,text/csv",
                        multiple=False,
                        className="material-upload",
                    ),
                    html.Div(
                        id=ids.material_id(id_prefix, ids.MATERIAL_UPLOAD_STATUS_SUFFIX),
                        className="material-upload-status",
                    ),
                ],
            ),
        ],
    )
