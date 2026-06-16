"""Workflow-shared stack-builder callbacks: keep ``stack_config_store`` in sync.

Thin adapters (ARCHITECTURE §1, §3): read the stack-builder widgets and the
finite-layer ``DataTable``, assemble the §2.2 stack-config dict, and write it to
``stack_config_store``. The DataTable rows round-trip losslessly into
``stack_config["layers"]`` (one callback over ``data`` — no per-row callbacks).
"""

from __future__ import annotations

from dash import Input, Output, State, callback_context, no_update
from dash import dash_table  # noqa: F401  (DataTable lives here; layout owns widget)

from app import config, ids, state


def _material_from_constant(n, k, name) -> dict:
    """Build a §2.1 constant-material dict from the numeric inputs."""

    return {
        "kind": "constant",
        "n": float(n) if n is not None else 0.0,
        "k": float(k) if k is not None else 0.0,
        "name": name or None,
    }


def _resolve_material(kind, n, k, name, csv_name, library) -> dict:
    """Resolve a material dict from a kind selector + inputs + CSV library.

    For ``kind == "csv"`` the CSV-material dict is looked up by ``csv_name`` in
    ``material_library_store``; for ``"constant"`` it is built from (n, k).
    """

    if kind == "csv":
        library = library or {}
        material = library.get(csv_name)
        if material is None:
            # Fall back to a constant so the stack stays buildable; validation
            # surfaces the missing-material problem to the user.
            return _material_from_constant(n, k, name)
        return material
    return _material_from_constant(n, k, name)


def _layers_from_table(rows, library) -> list[dict]:
    """Convert DataTable ``data`` rows into ``stack_config["layers"]``."""

    layers: list[dict] = []
    for row in rows or []:
        material = _resolve_material(
            kind=row.get(ids.LAYER_COL_MATERIAL_KIND, "constant"),
            n=row.get(ids.LAYER_COL_N),
            k=row.get(ids.LAYER_COL_K),
            name=None,
            csv_name=row.get(ids.LAYER_COL_CSV_NAME),
            library=library,
        )
        thickness = row.get(ids.LAYER_COL_THICKNESS, config.DEFAULT_LAYER_THICKNESS_NM)
        try:
            thickness = float(thickness)
        except (TypeError, ValueError):
            thickness = config.DEFAULT_LAYER_THICKNESS_NM
        layers.append({"material": material, "thickness_nm": thickness})
    return layers


def register(app) -> None:
    """Register stack-builder synchronization callbacks."""

    @app.callback(
        Output(ids.STACK_CONFIG_STORE, "data"),
        Input(ids.material_id(ids.INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_KIND_SUFFIX), "value"),
        Input(ids.material_id(ids.INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_N_SUFFIX), "value"),
        Input(ids.material_id(ids.INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_K_SUFFIX), "value"),
        Input(ids.material_id(ids.INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_NAME_SUFFIX), "value"),
        Input(ids.material_id(ids.SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_KIND_SUFFIX), "value"),
        Input(ids.material_id(ids.SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_N_SUFFIX), "value"),
        Input(ids.material_id(ids.SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_K_SUFFIX), "value"),
        Input(ids.material_id(ids.SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_NAME_SUFFIX), "value"),
        Input(ids.LAYER_TABLE, "data"),
        Input(ids.GRID_START_INPUT, "value"),
        Input(ids.GRID_STOP_INPUT, "value"),
        Input(ids.GRID_NUM_INPUT, "value"),
        Input(ids.ANGLE_INPUT, "value"),
        Input(ids.POLARIZATION_INPUT, "value"),
        State(ids.MATERIAL_LIBRARY_STORE, "data"),
    )
    def sync_stack_config(
        incident_kind,
        incident_n,
        incident_k,
        incident_name,
        substrate_kind,
        substrate_n,
        substrate_k,
        substrate_name,
        layer_rows,
        grid_start,
        grid_stop,
        grid_num,
        angle_deg,
        polarization,
        library,
    ):
        """Assemble the §2.2 stack-config dict from the builder widgets."""

        try:
            grid = {
                "start_nm": float(grid_start) if grid_start is not None else config.DEFAULT_GRID["start_nm"],
                "stop_nm": float(grid_stop) if grid_stop is not None else config.DEFAULT_GRID["stop_nm"],
                "num": int(grid_num) if grid_num is not None else config.DEFAULT_GRID["num"],
            }
        except (TypeError, ValueError):
            grid = dict(config.DEFAULT_GRID)

        return {
            "incident": _resolve_material(
                incident_kind, incident_n, incident_k, incident_name, incident_name, library
            ),
            "layers": _layers_from_table(layer_rows, library),
            "substrate": _resolve_material(
                substrate_kind, substrate_n, substrate_k, substrate_name, substrate_name, library
            ),
            "grid": grid,
            "angle_deg": float(angle_deg) if angle_deg is not None else config.DEFAULT_ANGLE_DEG,
            "polarization": polarization or config.DEFAULT_POLARIZATION,
        }

    @app.callback(
        Output(ids.LAYER_TABLE, "data", allow_duplicate=True),
        Input(ids.ADD_LAYER_BUTTON, "n_clicks"),
        State(ids.LAYER_TABLE, "data"),
        prevent_initial_call=True,
    )
    def add_layer_row(n_clicks, rows):
        """Append a default finite-layer row to the DataTable."""

        if not n_clicks:
            return no_update
        rows = list(rows or [])
        rows.append(
            {
                ids.LAYER_COL_MATERIAL_KIND: "constant",
                ids.LAYER_COL_N: config.DEFAULT_LAYER_MATERIAL["n"],
                ids.LAYER_COL_K: config.DEFAULT_LAYER_MATERIAL["k"],
                ids.LAYER_COL_CSV_NAME: None,
                ids.LAYER_COL_THICKNESS: config.DEFAULT_LAYER_THICKNESS_NM,
            }
        )
        return rows

    def _append_default_group_row(n_clicks, rows):
        """Append a default period-layer row to a mirror-group DataTable.

        Shared by the input/output "add layer" buttons; same column set and
        defaults as :func:`add_layer_row` (the §3 finite-layer table).
        """

        if not n_clicks:
            return no_update
        rows = list(rows or [])
        rows.append(
            {
                ids.LAYER_COL_MATERIAL_KIND: "constant",
                ids.LAYER_COL_N: config.DEFAULT_LAYER_MATERIAL["n"],
                ids.LAYER_COL_K: config.DEFAULT_LAYER_MATERIAL["k"],
                ids.LAYER_COL_CSV_NAME: None,
                ids.LAYER_COL_THICKNESS: config.DEFAULT_LAYER_THICKNESS_NM,
            }
        )
        return rows

    @app.callback(
        Output(ids.OPT_INPUT_GROUP_TABLE, "data", allow_duplicate=True),
        Input(ids.OPT_INPUT_ADD_LAYER_BUTTON, "n_clicks"),
        State(ids.OPT_INPUT_GROUP_TABLE, "data"),
        prevent_initial_call=True,
    )
    def add_input_group_row(n_clicks, rows):
        """Append a default period-layer row to the input mirror group table."""

        return _append_default_group_row(n_clicks, rows)

    @app.callback(
        Output(ids.OPT_OUTPUT_GROUP_TABLE, "data", allow_duplicate=True),
        Input(ids.OPT_OUTPUT_ADD_LAYER_BUTTON, "n_clicks"),
        State(ids.OPT_OUTPUT_GROUP_TABLE, "data"),
        prevent_initial_call=True,
    )
    def add_output_group_row(n_clicks, rows):
        """Append a default period-layer row to the output mirror group table."""

        return _append_default_group_row(n_clicks, rows)

    # --- Grouped/cavity sync (§9.1): widgets -> OPT_STACK_CONFIG_STORE --------
    @app.callback(
        Output(ids.OPT_STACK_CONFIG_STORE, "data"),
        # incident material_input
        Input(ids.material_id(ids.OPT_INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_KIND_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_N_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_K_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_INCIDENT_MATERIAL_PREFIX, ids.MATERIAL_NAME_SUFFIX), "value"),
        # input mirror group
        Input(ids.OPT_INPUT_GROUP_TABLE, "data"),
        Input(ids.OPT_INPUT_REPEAT_INPUT, "value"),
        # cavity
        Input(ids.material_id(ids.OPT_CAVITY_MATERIAL_PREFIX, ids.MATERIAL_KIND_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_CAVITY_MATERIAL_PREFIX, ids.MATERIAL_N_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_CAVITY_MATERIAL_PREFIX, ids.MATERIAL_K_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_CAVITY_MATERIAL_PREFIX, ids.MATERIAL_NAME_SUFFIX), "value"),
        Input(ids.OPT_CAVITY_THICKNESS_INPUT, "value"),
        Input(ids.OPT_CAVITY_ENABLED_INPUT, "value"),
        # output mirror group
        Input(ids.OPT_OUTPUT_GROUP_TABLE, "data"),
        Input(ids.OPT_OUTPUT_REPEAT_INPUT, "value"),
        # substrate material_input
        Input(ids.material_id(ids.OPT_SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_KIND_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_N_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_K_SUFFIX), "value"),
        Input(ids.material_id(ids.OPT_SUBSTRATE_MATERIAL_PREFIX, ids.MATERIAL_NAME_SUFFIX), "value"),
        # grid / angle / polarization
        Input(ids.OPT_GRID_START_INPUT, "value"),
        Input(ids.OPT_GRID_STOP_INPUT, "value"),
        Input(ids.OPT_GRID_NUM_INPUT, "value"),
        Input(ids.OPT_ANGLE_INPUT, "value"),
        Input(ids.OPT_POLARIZATION_INPUT, "value"),
        # variable selector
        Input(ids.OPT_VARIABLE_MODE_TABS, "value"),
        Input(ids.OPT_VARIABLE_CAVITY_INPUT, "value"),
        Input(ids.OPT_VARIABLE_INPUT_LAYERS_INPUT, "value"),
        Input(ids.OPT_VARIABLE_OUTPUT_LAYERS_INPUT, "value"),
        Input(ids.OPT_VARIABLE_FLAT_LAYERS_INPUT, "value"),
        State(ids.MATERIAL_LIBRARY_STORE, "data"),
    )
    def sync_opt_stack_config(
        incident_kind,
        incident_n,
        incident_k,
        incident_name,
        input_rows,
        input_repeat,
        cavity_kind,
        cavity_n,
        cavity_k,
        cavity_name,
        cavity_thickness,
        cavity_enabled,
        output_rows,
        output_repeat,
        substrate_kind,
        substrate_n,
        substrate_k,
        substrate_name,
        grid_start,
        grid_stop,
        grid_num,
        angle_deg,
        polarization,
        variable_mode,
        variable_cavity,
        variable_input_layers,
        variable_output_layers,
        variable_flat_layers,
        library,
    ):
        """Assemble the §9.1 grouped/cavity stack-config dict from the widgets.

        Mirrors :func:`sync_stack_config` but emits the grouped schema consumed
        by ``state.run_*_optimization`` via ``expand_optimization_config``.
        """

        try:
            grid = {
                "start_nm": float(grid_start) if grid_start is not None else config.DEFAULT_OPT_GRID["start_nm"],
                "stop_nm": float(grid_stop) if grid_stop is not None else config.DEFAULT_OPT_GRID["stop_nm"],
                "num": int(grid_num) if grid_num is not None else config.DEFAULT_OPT_GRID["num"],
            }
        except (TypeError, ValueError):
            grid = dict(config.DEFAULT_OPT_GRID)

        def _repeat(value, default):
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default

        def _cavity_thickness():
            try:
                return float(cavity_thickness)
            except (TypeError, ValueError):
                return config.DEFAULT_OPT_CAVITY_THICKNESS_NM

        return {
            "incident": _resolve_material(
                incident_kind, incident_n, incident_k, incident_name, incident_name, library
            ),
            "input_group": {
                "layers": _layers_from_table(input_rows, library),
                "repeat": _repeat(input_repeat, config.DEFAULT_OPT_REPEAT_M),
            },
            "cavity": {
                "material": _resolve_material(
                    cavity_kind, cavity_n, cavity_k, cavity_name, cavity_name, library
                ),
                "thickness_nm": _cavity_thickness(),
                "enabled": bool(cavity_enabled and "enabled" in cavity_enabled),
            },
            "output_group": {
                "layers": _layers_from_table(output_rows, library),
                "repeat": _repeat(output_repeat, config.DEFAULT_OPT_REPEAT_K),
            },
            "substrate": _resolve_material(
                substrate_kind, substrate_n, substrate_k, substrate_name, substrate_name, library
            ),
            "grid": grid,
            "angle_deg": float(angle_deg) if angle_deg is not None else config.DEFAULT_ANGLE_DEG,
            "polarization": polarization or config.DEFAULT_POLARIZATION,
            "variable": {
                "mode": variable_mode or ids.OPT_VARIABLE_MODE_TIED,
                "cavity": bool(variable_cavity and "cavity" in variable_cavity),
                "input_layers": [int(i) for i in (variable_input_layers or [])],
                "output_layers": [int(j) for j in (variable_output_layers or [])],
                "flat_layers": [int(n) for n in (variable_flat_layers or [])],
            },
        }

    # --- Dynamic options for the independent-mode checklist (§11.5) ----------
    @app.callback(
        Output(ids.OPT_VARIABLE_FLAT_LAYERS_INPUT, "options"),
        Input(ids.OPT_STACK_CONFIG_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
    )
    def update_flat_layer_options(opt_stack_config, lang):
        """Populate the "Per singolo strato" checklist from the expanded stack.

        Driven by the store (not raw widgets) so the options always reflect the
        post-expansion truth (M, K, cavity on/off, period-table edits). Labels
        come from :func:`state.enumerate_expanded_layers` and follow the current
        language (read from ``LANGUAGE_STORE``; default English).
        """

        if not opt_stack_config:
            return []
        lang = lang or config.DEFAULT_LANG
        try:
            entries = state.enumerate_expanded_layers(opt_stack_config, lang=lang)
        except ValueError:
            # Malformed/in-progress config: no selectable options yet.
            return []
        return [
            {"label": e["label"], "value": e["flat_index"]} for e in entries
        ]
