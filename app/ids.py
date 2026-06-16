"""Canonical component-id and Store-key string constants.

All component ids and ``dcc.Store`` keys are declared **once** here in English
snake_case and imported everywhere (never inlined as string literals). UI text
itself is Italian and lives in :mod:`app.config`.

This module is the single source of truth for the id contract shared by
``gui-frontend`` (components/layout/callbacks) and the Store reads/writes that
``gui-core`` performs in the callback adapters.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# dcc.Store ids (App-state model, ARCHITECTURE §2.4)
# ---------------------------------------------------------------------------
#: §2.2 **flat** stack-config dict. Source of truth for the Simulazione tab.
STACK_CONFIG_STORE = "stack_config_store"
#: §9.1 **grouped/cavity** stack-config dict. Source of truth for the
#: Ottimizzazione tab's structure (input/output mirror groups, cavity, grid,
#: angle, polarization, and the `variable` selector). Added in §9; does NOT
#: replace STACK_CONFIG_STORE — the two tabs own distinct stack stores.
OPT_STACK_CONFIG_STORE = "opt_stack_config_store"
#: §2.3 workflow-2-only **scalar** inputs (target wl/Q, feature, steps, lr, ...).
#: The variable-layer selection now lives in OPT_STACK_CONFIG_STORE["variable"].
OPTIMIZE_CONFIG_STORE = "optimize_config_store"
#: last SimulationResult as a JSON dict (state.result_to_dict).
SIMULATION_RESULT_STORE = "simulation_result_store"
#: last optimization outcome as a JSON dict (history + thicknesses + resonance).
OPTIMIZATION_RESULT_STORE = "optimization_result_store"
#: incremental progress for the running background optimize job.
OPTIMIZATION_PROGRESS_STORE = "optimization_progress_store"
#: library of uploaded (CSV) materials keyed by name, referenced by layer rows.
MATERIAL_LIBRARY_STORE = "material_library_store"

# ---------------------------------------------------------------------------
# Stack builder — incident / substrate material inputs
# ---------------------------------------------------------------------------
#: id_prefix passed to build_material_input for the incident medium.
INCIDENT_MATERIAL_PREFIX = "incident_material"
#: id_prefix passed to build_material_input for the substrate medium.
SUBSTRATE_MATERIAL_PREFIX = "substrate_material"

# material_input sub-ids are derived as f"{prefix}_{suffix}"; suffixes here.
MATERIAL_KIND_SUFFIX = "kind"          # dropdown: "constant" | "csv"
MATERIAL_N_SUFFIX = "n"                # numeric input (constant n)
MATERIAL_K_SUFFIX = "k"                # numeric input (constant k)
MATERIAL_NAME_SUFFIX = "name"          # text input (optional name)
MATERIAL_UPLOAD_SUFFIX = "upload"      # dcc.Upload (CSV)
MATERIAL_UPLOAD_STATUS_SUFFIX = "upload_status"  # html feedback for the upload


def material_id(prefix: str, suffix: str) -> str:
    """Return the canonical id of a material_input sub-widget."""

    return f"{prefix}_{suffix}"


# ---------------------------------------------------------------------------
# Finite-layer editor (Dash DataTable, ARCHITECTURE §3)
# ---------------------------------------------------------------------------
LAYER_TABLE = "layer_table"
ADD_LAYER_BUTTON = "add_layer_button"

# DataTable column ids (round-trip into stack_config["layers"], see §2.2).
LAYER_COL_MATERIAL_KIND = "material_kind"   # "constant" | "csv"
LAYER_COL_N = "n"                           # constant n
LAYER_COL_K = "k"                           # constant k
LAYER_COL_CSV_NAME = "csv_name"             # name into MATERIAL_LIBRARY_STORE
LAYER_COL_THICKNESS = "thickness_nm"        # numeric, editable

# ---------------------------------------------------------------------------
# Grid / angle / polarization controls
# ---------------------------------------------------------------------------
GRID_START_INPUT = "grid_start_input"
GRID_STOP_INPUT = "grid_stop_input"
GRID_NUM_INPUT = "grid_num_input"
ANGLE_INPUT = "angle_input"
POLARIZATION_INPUT = "polarization_input"

# ---------------------------------------------------------------------------
# Simulate panel (workflow 1)
# ---------------------------------------------------------------------------
SIMULATE_BUTTON = "simulate_button"
SIMULATE_CHANNELS_INPUT = "simulate_channels_input"   # which of R/T/A to plot
SIMULATE_STATUS = "simulate_status"
SIMULATE_GRAPH = "simulate_graph"
#: mini-sketch of the flat stack (§10), updated from STACK_CONFIG_STORE.
SIMULATE_SKETCH_GRAPH = "simulate_sketch_graph"

# ---------------------------------------------------------------------------
# Optimize panel (workflow 2)
# ---------------------------------------------------------------------------
OPTIMIZE_MODE_INPUT = "optimize_mode_input"           # "resonance" | "mean_r"
OPTIMIZE_TARGET_WAVELENGTH_INPUT = "optimize_target_wavelength_input"
OPTIMIZE_TARGET_Q_INPUT = "optimize_target_q_input"
OPTIMIZE_FEATURE_INPUT = "optimize_feature_input"      # "peak" | "dip"
OPTIMIZE_SPECTRUM_INPUT = "optimize_spectrum_input"    # "R" | "T" | "A"
OPTIMIZE_STEPS_INPUT = "optimize_steps_input"
OPTIMIZE_LEARNING_RATE_INPUT = "optimize_learning_rate_input"
OPTIMIZE_LOWER_BOUND_INPUT = "optimize_lower_bound_input"
OPTIMIZE_WAVELENGTH_WEIGHT_INPUT = "optimize_wavelength_weight_input"
OPTIMIZE_Q_WEIGHT_INPUT = "optimize_q_weight_input"
OPTIMIZE_SHARPNESS_INPUT = "optimize_sharpness_input"
# --- Grouped/cavity stack editor (§9.4); writes OPT_STACK_CONFIG_STORE ------
#: id_prefix passed to build_material_input for the grouped-tab incident medium.
OPT_INCIDENT_MATERIAL_PREFIX = "opt_incident_material"
#: id_prefix passed to build_material_input for the grouped-tab substrate.
OPT_SUBSTRATE_MATERIAL_PREFIX = "opt_substrate_material"
#: id_prefix passed to build_material_input for the cavity single layer.
OPT_CAVITY_MATERIAL_PREFIX = "opt_cavity_material"

#: period-definition DataTable for the input mirror group (one period only).
OPT_INPUT_GROUP_TABLE = "opt_input_group_table"
#: period-definition DataTable for the output mirror group (one period only).
OPT_OUTPUT_GROUP_TABLE = "opt_output_group_table"
#: "add layer" buttons that append a default row to each mirror-group table
#: (parallel to ADD_LAYER_BUTTON for the finite-layer table in §3).
OPT_INPUT_ADD_LAYER_BUTTON = "opt_input_add_layer_button"
OPT_OUTPUT_ADD_LAYER_BUTTON = "opt_output_add_layer_button"
#: integer input for the input-group repeat count M (fixed during a run).
OPT_INPUT_REPEAT_INPUT = "opt_input_repeat_input"
#: integer input for the output-group repeat count K (fixed during a run).
OPT_OUTPUT_REPEAT_INPUT = "opt_output_repeat_input"
#: numeric input for the cavity thickness (nm).
OPT_CAVITY_THICKNESS_INPUT = "opt_cavity_thickness_input"
#: checkbox/toggle for cavity.enabled (off => omit cavity from expansion).
OPT_CAVITY_ENABLED_INPUT = "opt_cavity_enabled_input"
#: checkbox: cavity selected as a variable thickness (default = checked).
OPT_VARIABLE_CAVITY_INPUT = "opt_variable_cavity_input"
#: checklist: which input-group period-layer indices i are variable.
OPT_VARIABLE_INPUT_LAYERS_INPUT = "opt_variable_input_layers_input"
#: checklist: which output-group period-layer indices j are variable.
OPT_VARIABLE_OUTPUT_LAYERS_INPUT = "opt_variable_output_layers_input"
# --- §11.2 Two-mode variable selector (tied vs independent) ----------------
#: dcc.Tabs that switches between "Per periodo" (tied) and "Per singolo strato"
#: (independent) inside the "Spessori variabili" fieldset. Its `value` mirrors
#: opt_stack_config_store["variable"]["mode"] (written by stack_callbacks.py).
OPT_VARIABLE_MODE_TABS = "opt_variable_mode_tabs"
#: dcc.Tab value for the "tied" (per-periodo) mode. Matches the store key.
OPT_VARIABLE_MODE_TIED = "tied"
#: dcc.Tab value for the "independent" (per-singolo-strato) mode.
OPT_VARIABLE_MODE_INDEPENDENT = "independent"
#: dcc.Checklist listing every expanded flat layer (options populated dynamically
#: by gui-core's _flat_layer_options callback; starts empty).
OPT_VARIABLE_FLAT_LAYERS_INPUT = "opt_variable_flat_layers_input"
#: grid / angle / polarization controls for the Ottimizzazione tab.
OPT_GRID_START_INPUT = "opt_grid_start_input"
OPT_GRID_STOP_INPUT = "opt_grid_stop_input"
OPT_GRID_NUM_INPUT = "opt_grid_num_input"
OPT_ANGLE_INPUT = "opt_angle_input"
OPT_POLARIZATION_INPUT = "opt_polarization_input"

OPTIMIZE_BUTTON = "optimize_button"
OPTIMIZE_STATUS = "optimize_status"
OPTIMIZE_PROGRESS_BAR = "optimize_progress_bar"
OPTIMIZE_HISTORY_GRAPH = "optimize_history_graph"
OPTIMIZE_RESULT_GRAPH = "optimize_result_graph"
OPTIMIZE_THICKNESS_READOUT = "optimize_thickness_readout"
#: "Export" button for the optimized result. One click downloads a single ZIP
#: holding the two connected .txt files (spectra + parameters). A ZIP is used
#: because browsers drop the 2nd of two simultaneous programmatic downloads.
OPTIMIZE_EXPORT_BUTTON = "optimize_export_button"
OPTIMIZE_EXPORT_DOWNLOAD = "optimize_export_download"
#: short status line shown next to the export button (errors / confirmation).
OPTIMIZE_EXPORT_STATUS = "optimize_export_status"
#: mini-sketch of the grouped/cavity stack (§10), updated from
#: OPT_STACK_CONFIG_STORE with grouped=True.
OPTIMIZE_SKETCH_GRAPH = "optimize_sketch_graph"

# ---------------------------------------------------------------------------
# Results panel (shared output: spectrum + resonance readout + export)
# ---------------------------------------------------------------------------
#: id_prefix for the simulate-tab results panel.
SIMULATE_RESULTS_PREFIX = "simulate_results"
#: id_prefix for the optimize-tab results panel.
OPTIMIZE_RESULTS_PREFIX = "optimize_results"

RESULTS_GRAPH_SUFFIX = "graph"
RESULTS_RESONANCE_TABLE_SUFFIX = "resonance_table"
RESULTS_EXPORT_BUTTON_SUFFIX = "export_button"
#: single dcc.Download target (one ZIP of spectra + parameters .txt files).
RESULTS_EXPORT_DOWNLOAD_SUFFIX = "export_download"
#: short status line shown next to the export button.
RESULTS_EXPORT_STATUS_SUFFIX = "export_status"


def results_id(prefix: str, suffix: str) -> str:
    """Return the canonical id of a results_panel sub-widget."""

    return f"{prefix}_{suffix}"


# ---------------------------------------------------------------------------
# Top-level layout
# ---------------------------------------------------------------------------
APP_TABS = "app_tabs"
TAB_SIMULATE = "tab_simulate"
TAB_OPTIMIZE = "tab_optimize"

# ---------------------------------------------------------------------------
# i18n — language store + selector (ARCHITECTURE §12.4)
# ---------------------------------------------------------------------------
#: dcc.Store seeded by build_layout(lang) with the active language code.
#: Callbacks read it via State(LANGUAGE_STORE, "data") to localize output.
LANGUAGE_STORE = "language_store"

#: Header dcc.RadioItems / dcc.Dropdown (EN/IT) — value is the active lang code.
#: A clientside callback reloads with ?lang=<value> on change (gui-core wires it).
LANGUAGE_SELECTOR = "language_selector"

# ---------------------------------------------------------------------------
# Help-icon ids (ARCHITECTURE §12.4)
# ---------------------------------------------------------------------------
#: Pattern-matching `type` for all opt-field help icons.
HELP_ICON_TYPE = "opt-help-icon"


def help_icon_id(field: str) -> dict:
    """Canonical pattern-matching id for an optimize-field help icon.

    Returns ``{"type": "opt-help-icon", "field": field}``.
    The ``field`` value matches the ``tip_*`` suffix in the translation catalog
    (e.g. ``"target_wavelength"`` for key ``"tip_target_wavelength"``).
    These ids are purely presentational — never used as callback Input/Output.
    """
    return {"type": HELP_ICON_TYPE, "field": field}
