"""Pure domain<->dict boundary between the GUI Stores and ``multilayer_tmm``.

This module is intentionally free of Dash and Plotly imports — it depends only
on the ``multilayer_tmm`` public API and NumPy. Callbacks are thin adapters that
read Stores, call functions here, then hand JSON-safe dicts to ``app.plots``
builders and write Stores.

Key invariants (ARCHITECTURE §0, §2, §5, §6.1):
- All inputs/outputs that cross a ``dcc.Store`` are JSON-safe Python
  (``float``/``int``/``str``/``list``/``bool``/``None``). ``jnp``/``np`` arrays
  never leak out; every JAX value is converted at the boundary via NumPy.
- Materials are ``constant`` (n, k) or ``csv`` (tabulated) only. No code-eval,
  no callable materials.
- "both" polarization yields R/T/A of shape ``(2, N)`` ordered ``(s, p)``.
  Analysis and optimization reject 2-row spectra (the library does too).
"""

from __future__ import annotations

import base64
import csv
import io
import zipfile
from datetime import datetime
from typing import Callable

import numpy as np

from multilayer_tmm import (
    Layer,
    Material,
    OptimizationResult,
    ResonanceOptimizationResult,
    ResonanceResult,
    SimulationResult,
    Stack,
    analyze_resonance,
    mean_reflectivity,
    optimize_resonance_target,  # noqa: F401  (public API; retained for callers)
    optimize_thicknesses,
    resonance_target_loss,
    simulate_spectrum,
    stack_thicknesses,
    stack_with_thicknesses,
    wavelength_grid,
)

# Names re-used in type hints / docstrings; jnp is only needed for annotations.
import jax.numpy as jnp  # noqa: F401  (used as a type hint name in signatures)

# state.py stays config-free (mirrors plots.py): it owns its own catalog and its
# own default-language constant rather than importing app.config. This must match
# config.DEFAULT_LANG ("en"); the language switch is canonical English-default.
DEFAULT_LANG: str = "en"

__all__ = [
    "material_from_dict",
    "material_to_dict",
    "parse_material_csv",
    "stack_from_config",
    "grid_from_config",
    "validate_config",
    "run_simulation",
    "result_to_dict",
    "analyze_result",
    "make_thickness_objective",
    "expand_optimization_config",
    "expand_optimization_variables",
    "enumerate_expanded_layers",
    "validate_opt_stack_config",
    "run_resonance_optimization",
    "run_thickness_optimization",
    "make_export_prefix",
    "build_spectra_text",
    "build_optimized_spectra_text",
    "build_optimization_parameters_text",
    "build_simulation_parameters_text",
    "build_export_zip_bytes",
]


# ===========================================================================
# i18n error / label catalog (ARCHITECTURE §12.1-D3, §12.2b)
#
# state.py owns its own catalog (mirroring plots.py's _PLOT_TRANSLATIONS) so it
# stays Dash-free AND config-free except for the DEFAULT_LANG constant. English
# is the canonical baseline; Italian mirrors the historic inline strings. The
# two key sets are IDENTICAL (gui-qa asserts set(_ERRORS["en"]) == _ERRORS["it"]).
#
# Templates use str.format(**fmt) placeholders ({where}, {index}, {kind}, ...)
# resolved by _e(key, lang, **fmt). _err(key, lang) is the no-format accessor.
# ===========================================================================
_ERRORS: dict[str, dict[str, str]] = {
    # ── English (canonical baseline) ─────────────────────────────────────────
    "en": {
        # ---- material ----
        "err_material_invalid": "Invalid material definition (expected a dict).",
        "err_material_csv_missing_cols": "CSV material missing the wavelength_nm/n columns.",
        "err_material_kind_unsupported": "Unsupported material type: {kind!r}. Use 'constant' or 'csv'.",
        "err_material_not_serializable": "Material is not serializable (kind={kind!r}); callable is not supported.",
        # ---- CSV parsing ----
        "err_csv_no_content": "No content to upload.",
        "err_csv_decode": "Could not decode the uploaded file (base64).",
        "err_csv_no_header": "The CSV file has no header row.",
        "err_csv_missing_cols": "Missing columns in the CSV: {missing}.",
        "err_csv_non_numeric_row": "Non-numeric value in the CSV at row {line}.",
        "err_csv_no_data_rows": "The CSV file contains no data rows.",
        "err_csv_fallback_name": "csv_material",
        # ---- stack config ----
        "err_stack_config_invalid": "Invalid stack configuration.",
        "err_incident_missing": "Incident medium missing.",
        "err_substrate_missing": "Substrate missing.",
        "err_layers_invalid": "Invalid layer list.",
        "err_layer_invalid": "Layer {index}: invalid definition.",
        "err_thickness_negative": "Layer {index}: thickness cannot be negative.",
        "err_thickness_non_numeric": "Layer {index}: non-numeric thickness.",
        # ---- per-material field errors (where-prefixed) ----
        "err_field_material_invalid": "{where}: invalid material definition.",
        "err_field_value_non_numeric": "{where}: non-numeric {label} value.",
        "err_field_csv_no_data": "{where}: CSV material has no data (wavelength_nm/n).",
        "err_field_csv_min_points": "{where}: the CSV material requires at least 2 points.",
        "err_field_kind_unsupported": "{where}: unsupported material type ({kind!r}).",
        # ---- grid / angle / polarization ----
        "err_grid_missing": "Wavelength grid missing.",
        "err_grid_min_points": "The grid requires at least 2 points (num >= 2).",
        "err_grid_start_ge_stop": "Start wavelength must be less than stop wavelength.",
        "err_grid_params_invalid": "Invalid grid parameters.",
        "err_angle_non_numeric": "Non-numeric incidence angle.",
        "err_polarization_invalid": "Invalid polarization: {pol!r}.",
        # ---- channel / analysis / optimization rejection ----
        "err_channel_invalid": "The channel must be \"R\", \"T\" or \"A\".",
        "err_resonance_needs_single_pol": (
            "Resonance analysis requires a single polarization "
            "(select 's' or 'p', not 'both')."
        ),
        "err_optimize_needs_single_pol": (
            "Optimization requires a single polarization "
            "(select 's' or 'p', not 'both')."
        ),
        # ---- grouped/cavity expansion ----
        "err_opt_stack_invalid": "Invalid stack configuration (groups/cavity).",
        "err_group_invalid": "{where}: invalid group definition.",
        "err_group_period_layer_invalid": "{where}: period layer {index} is invalid.",
        "err_group_definition_invalid": "{label}: invalid definition.",
        "err_group_layers_invalid": "{label}: invalid layer list.",
        "err_group_period_empty": "{label}: period layers cannot be empty.",
        "err_group_layer_invalid": "{label} layer {index}: invalid definition.",
        "err_group_thickness_negative": "{label} layer {index}: thickness cannot be negative.",
        "err_group_thickness_non_numeric": "{label} layer {index}: non-numeric thickness.",
        "err_repeat_negative": "The number of repetitions (M, K) cannot be negative.",
        "err_repeat_not_int": "{label}: the number of repetitions must be an integer.",
        "err_repeat_ge_zero": "{label}: the number of repetitions must be >= 0.",
        "err_stack_empty": (
            "The stack is empty: provide at least one group with repeat >= 1 "
            "or enable the cavity."
        ),
        "err_cavity_thickness_negative": "Cavity: thickness cannot be negative.",
        "err_cavity_thickness_non_numeric": "Cavity: non-numeric thickness.",
        "err_cavity_disabled_but_variable": (
            "The cavity is selected as a variable but is disabled."
        ),
        "err_variable_selector_invalid": "Invalid variable selector.",
        "err_variable_mode_invalid": (
            "Invalid variable mode: {mode!r} (use 'tied' or 'independent')."
        ),
        "err_select_one_variable": "Select at least one variable thickness.",
        "err_select_one_variable_tied": (
            "Select at least one variable thickness (cavity, input or output layer)."
        ),
        "err_flat_index_invalid": "Invalid expanded-stack layer index: {value!r}.",
        "err_flat_index_out_of_range": "Expanded-stack layer index out of range: {value}.",
        "err_input_layer_out_of_range": (
            "Input-group variable layer index out of range: {index}."
        ),
        "err_output_layer_out_of_range": (
            "Output-group variable layer index out of range: {index}."
        ),
        "err_expanded_index_invalid": "Invalid variable layer index after expansion.",
        # ---- where / label fragments (reused as message components) ----
        "where_incident": "Incident medium",
        "where_substrate": "Substrate",
        "where_layer": "Layer {index}",
        "where_cavity": "Cavity",
        "where_input_group": "Input group",
        "where_output_group": "Output group",
        # ---- enumerate_expanded_layers labels (§12.2b lbl_*) ----
        "lbl_input_layer": "Input · per. {r} · layer {p} ({name})",
        "lbl_cavity": "Cavity ({name})",
        "lbl_output_layer": "Output · per. {r} · layer {p} ({name})",
        "lbl_fallback_input": "input_{p}",
        "lbl_fallback_output": "output_{p}",
        "lbl_fallback_cavity": "cavity",
        "lbl_fallback_layer": "layer",
    },
    # ── Italian (mirror of the historic inline strings) ──────────────────────
    "it": {
        # ---- material ----
        "err_material_invalid": "Definizione materiale non valida (atteso un dizionario).",
        "err_material_csv_missing_cols": "Materiale CSV privo delle colonne wavelength_nm/n.",
        "err_material_kind_unsupported": "Tipo di materiale non supportato: {kind!r}. Usare 'constant' o 'csv'.",
        "err_material_not_serializable": "Materiale non serializzabile (kind={kind!r}); callable non supportato.",
        # ---- CSV parsing ----
        "err_csv_no_content": "Nessun contenuto da caricare.",
        "err_csv_decode": "Impossibile decodificare il file caricato (base64).",
        "err_csv_no_header": "Il file CSV non ha una riga di intestazione.",
        "err_csv_missing_cols": "Colonne mancanti nel CSV: {missing}.",
        "err_csv_non_numeric_row": "Valore non numerico nel CSV alla riga {line}.",
        "err_csv_no_data_rows": "Il file CSV non contiene righe di dati.",
        "err_csv_fallback_name": "materiale_csv",
        # ---- stack config ----
        "err_stack_config_invalid": "Configurazione dello stack non valida.",
        "err_incident_missing": "Mezzo incidente mancante.",
        "err_substrate_missing": "Substrato mancante.",
        "err_layers_invalid": "Elenco strati non valido.",
        "err_layer_invalid": "Strato {index}: definizione non valida.",
        "err_thickness_negative": "Strato {index}: lo spessore non puo essere negativo.",
        "err_thickness_non_numeric": "Strato {index}: spessore non numerico.",
        # ---- per-material field errors (where-prefixed) ----
        "err_field_material_invalid": "{where}: definizione materiale non valida.",
        "err_field_value_non_numeric": "{where}: valore {label} non numerico.",
        "err_field_csv_no_data": "{where}: materiale CSV privo di dati (wavelength_nm/n).",
        "err_field_csv_min_points": "{where}: il materiale CSV richiede almeno 2 punti.",
        "err_field_kind_unsupported": "{where}: tipo di materiale non supportato ({kind!r}).",
        # ---- grid / angle / polarization ----
        "err_grid_missing": "Griglia di lunghezze d'onda mancante.",
        "err_grid_min_points": "La griglia richiede almeno 2 punti (num >= 2).",
        "err_grid_start_ge_stop": "Lambda iniziale deve essere minore di lambda finale.",
        "err_grid_params_invalid": "Parametri della griglia non validi.",
        "err_angle_non_numeric": "Angolo di incidenza non numerico.",
        "err_polarization_invalid": "Polarizzazione non valida: {pol!r}.",
        # ---- channel / analysis / optimization rejection ----
        "err_channel_invalid": "Il canale deve essere \"R\", \"T\" o \"A\".",
        "err_resonance_needs_single_pol": (
            "L'analisi di risonanza richiede una singola polarizzazione "
            "(selezionare 's' o 'p', non 'both')."
        ),
        "err_optimize_needs_single_pol": (
            "L'ottimizzazione richiede una singola polarizzazione "
            "(selezionare 's' o 'p', non 'both')."
        ),
        # ---- grouped/cavity expansion ----
        "err_opt_stack_invalid": "Configurazione dello stack (gruppi/cavità) non valida.",
        "err_group_invalid": "{where}: definizione del gruppo non valida.",
        "err_group_period_layer_invalid": "{where}: strato di periodo {index} non valido.",
        "err_group_definition_invalid": "{label}: definizione non valida.",
        "err_group_layers_invalid": "{label}: elenco strati non valido.",
        "err_group_period_empty": "{label}: gli strati del periodo non possono essere vuoti.",
        "err_group_layer_invalid": "{label} strato {index}: definizione non valida.",
        "err_group_thickness_negative": "{label} strato {index}: lo spessore non puo essere negativo.",
        "err_group_thickness_non_numeric": "{label} strato {index}: spessore non numerico.",
        "err_repeat_negative": "Il numero di ripetizioni (M, K) non puo essere negativo.",
        "err_repeat_not_int": "{label}: il numero di ripetizioni deve essere intero.",
        "err_repeat_ge_zero": "{label}: il numero di ripetizioni deve essere >= 0.",
        "err_stack_empty": (
            "Lo stack è vuoto: indicare almeno un gruppo con ripetizione >= 1 "
            "oppure abilitare la cavità."
        ),
        "err_cavity_thickness_negative": "Cavità: lo spessore non puo essere negativo.",
        "err_cavity_thickness_non_numeric": "Cavità: spessore non numerico.",
        "err_cavity_disabled_but_variable": (
            "La cavità è selezionata come variabile ma è disabilitata."
        ),
        "err_variable_selector_invalid": "Selettore delle variabili non valido.",
        "err_variable_mode_invalid": (
            "Modalità delle variabili non valida: {mode!r} (usare 'tied' o 'independent')."
        ),
        "err_select_one_variable": "Selezionare almeno uno spessore variabile.",
        "err_select_one_variable_tied": (
            "Selezionare almeno uno spessore variabile (cavità, strato di "
            "ingresso o di uscita)."
        ),
        "err_flat_index_invalid": "Indice di strato (stack espanso) non valido: {value!r}.",
        "err_flat_index_out_of_range": "Indice di strato (stack espanso) fuori intervallo: {value}.",
        "err_input_layer_out_of_range": (
            "Indice strato variabile del gruppo di ingresso fuori intervallo: {index}."
        ),
        "err_output_layer_out_of_range": (
            "Indice strato variabile del gruppo di uscita fuori intervallo: {index}."
        ),
        "err_expanded_index_invalid": "Indice di strato variabile non valido dopo l'espansione.",
        # ---- where / label fragments (reused as message components) ----
        "where_incident": "Mezzo incidente",
        "where_substrate": "Substrato",
        "where_layer": "Strato {index}",
        "where_cavity": "Cavità",
        "where_input_group": "Gruppo di ingresso",
        "where_output_group": "Gruppo di uscita",
        # ---- enumerate_expanded_layers labels (§12.2b lbl_*) ----
        "lbl_input_layer": "Ingresso · per. {r} · strato {p} ({name})",
        "lbl_cavity": "Cavità ({name})",
        "lbl_output_layer": "Uscita · per. {r} · strato {p} ({name})",
        "lbl_fallback_input": "ingresso_{p}",
        "lbl_fallback_output": "uscita_{p}",
        "lbl_fallback_cavity": "cavità",
        "lbl_fallback_layer": "strato",
    },
}


def _err(key: str, lang: str = DEFAULT_LANG) -> str:
    """Return a localized message template (no interpolation), EN fallback.

    Unknown ``lang`` -> EN; unknown ``key`` -> the key itself (never KeyErrors).
    """

    table = _ERRORS.get(lang) if lang in _ERRORS else _ERRORS[DEFAULT_LANG]
    return table.get(key, _ERRORS[DEFAULT_LANG].get(key, key))


def _e(key: str, lang: str = DEFAULT_LANG, **fmt) -> str:
    """Return a localized message with ``str.format(**fmt)`` interpolation.

    Falls back to EN, then to the key itself. If formatting fails (a template
    referencing a key absent from ``fmt``), the raw template is returned.
    """

    template = _err(key, lang)
    if not fmt:
        return template
    try:
        return template.format(**fmt)
    except (KeyError, IndexError, ValueError):
        return template


# ===========================================================================
# Small JSON-safe conversion helpers (the JAX -> plain-Python boundary)
# ===========================================================================
def _to_float(value) -> float:
    """Convert a 0-d JAX/NumPy/Python value to a plain ``float``."""

    return float(np.asarray(value).item())


def _to_list(array) -> list:
    """Convert any array-like to a JSON-safe nested list of plain floats.

    Complex values are not expected on the output channels (R/T/A are real),
    so a real cast is applied defensively.
    """

    np_array = np.asarray(array)
    if np.iscomplexobj(np_array):
        np_array = np_array.real
    return np_array.astype(float).tolist()


# ===========================================================================
# Material <-> dict (ARCHITECTURE §2.1, §6.1)
# ===========================================================================
def material_from_dict(d: dict, lang: str = DEFAULT_LANG) -> Material:
    """Build a :class:`Material` from a §2.1 material dict.

    Supports ``kind == "constant"`` (-> :meth:`Material.constant`) and
    ``kind == "csv"`` (-> :meth:`Material.from_table`). Raises ``ValueError``
    for any other kind (notably ``"callable"`` is rejected — no code-eval).
    Error messages are localized by ``lang`` (default English).
    """

    if not isinstance(d, dict):
        raise ValueError(_e("err_material_invalid", lang))
    kind = d.get("kind")
    name = d.get("name")

    if kind == "constant":
        n = float(d.get("n", 0.0))
        k = float(d.get("k", 0.0))
        return Material.constant(complex(n, k), name=name)

    if kind == "csv":
        wavelength_nm = d.get("wavelength_nm")
        n = d.get("n")
        k = d.get("k", 0.0)
        if wavelength_nm is None or n is None:
            raise ValueError(_e("err_material_csv_missing_cols", lang))
        return Material.from_table(
            wavelength_nm=np.asarray(wavelength_nm, dtype=float),
            n=np.asarray(n, dtype=float),
            k=np.asarray(k, dtype=float) if not np.isscalar(k) else float(k),
            name=name,
        )

    raise ValueError(_e("err_material_kind_unsupported", lang, kind=kind))


def material_to_dict(m: Material, lang: str = DEFAULT_LANG) -> dict:
    """Serialize a :class:`Material` to a JSON-safe §2.1 dict.

    Handles ``constant`` and ``tabulated`` materials. Raises ``ValueError`` for
    callable materials (they are never produced by the GUI). Error messages are
    localized by ``lang`` (default English).
    """

    if m.kind == "constant":
        value = complex(np.asarray(m.data).item())
        return {
            "kind": "constant",
            "n": float(value.real),
            "k": float(value.imag),
            "name": m.name,
        }

    if m.kind == "tabulated":
        wavelength_nm, n, k = m.data
        return {
            "kind": "csv",
            "wavelength_nm": _to_list(wavelength_nm),
            "n": _to_list(n),
            "k": _to_list(k),
            "name": m.name,
        }

    raise ValueError(_e("err_material_not_serializable", lang, kind=m.kind))


def parse_material_csv(
    contents: str,
    filename: str | None = None,
    name: str | None = None,
    lang: str = DEFAULT_LANG,
) -> dict:
    """Parse a ``dcc.Upload`` base64 data URL into a §2.1 csv-material dict.

    ``contents`` is the ``dcc.Upload.contents`` value, typically of the form
    ``"data:text/csv;base64,<payload>"`` (a bare base64 payload is also
    accepted). The CSV must carry ``wavelength_nm,n,k`` columns (mirroring
    :func:`multilayer_tmm.io.read_material_csv`). Parsing happens fully
    in-memory — no temp file, no path round-trip through ``Material.from_csv``.

    Raises ``ValueError`` (localized by ``lang``) on missing columns,
    non-numeric cells, or an empty file.
    """

    if not contents:
        raise ValueError(_e("err_csv_no_content", lang))

    # Strip an optional "data:<mime>;base64," prefix.
    payload = contents
    if "," in contents and contents.strip().lower().startswith("data:"):
        payload = contents.split(",", 1)[1]

    try:
        decoded = base64.b64decode(payload)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise ValueError(_e("err_csv_decode", lang)) from exc

    text = decoded.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(_e("err_csv_no_header", lang))

    required = {"wavelength_nm", "n", "k"}
    missing = required - set(reader.fieldnames)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(_e("err_csv_missing_cols", lang, missing=missing_list))

    wavelengths: list[float] = []
    n_values: list[float] = []
    k_values: list[float] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            wavelengths.append(float(row["wavelength_nm"]))
            n_values.append(float(row["n"]))
            k_values.append(float(row["k"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                _e("err_csv_non_numeric_row", lang, line=line_number)
            ) from exc

    if not wavelengths:
        raise ValueError(_e("err_csv_no_data_rows", lang))

    resolved_name = name or filename or _err("err_csv_fallback_name", lang)
    return {
        "kind": "csv",
        "wavelength_nm": wavelengths,
        "n": n_values,
        "k": k_values,
        "name": resolved_name,
    }


# ===========================================================================
# Stack + grid (ARCHITECTURE §2.2, §6.1)
# ===========================================================================
def stack_from_config(config: dict, lang: str = DEFAULT_LANG) -> Stack:
    """Build a :class:`Stack` from a §2.2 stack-config dict.

    Error messages are localized by ``lang`` (default English).
    """

    if not isinstance(config, dict):
        raise ValueError(_e("err_stack_config_invalid", lang))

    incident = material_from_dict(config["incident"], lang=lang)
    substrate = material_from_dict(config["substrate"], lang=lang)

    layers: list[Layer] = []
    for index, raw_layer in enumerate(config.get("layers", [])):
        material = material_from_dict(raw_layer["material"], lang=lang)
        thickness = float(raw_layer["thickness_nm"])
        layers.append(Layer(material=material, thickness_nm=thickness))

    return Stack(incident=incident, layers=layers, substrate=substrate)


def grid_from_config(config: dict) -> "jnp.ndarray":
    """Build the wavelength grid from ``config["grid"]`` via ``wavelength_grid``."""

    grid = config["grid"]
    return wavelength_grid(
        start_nm=float(grid["start_nm"]),
        stop_nm=float(grid["stop_nm"]),
        num=int(grid["num"]),
    )


def _validate_material(d, where: str, errors: list[str], lang: str = DEFAULT_LANG) -> None:
    if not isinstance(d, dict):
        errors.append(_e("err_field_material_invalid", lang, where=where))
        return
    kind = d.get("kind")
    if kind == "constant":
        for field, label in (("n", "n"), ("k", "k")):
            try:
                float(d.get(field, 0.0))
            except (TypeError, ValueError):
                errors.append(
                    _e("err_field_value_non_numeric", lang, where=where, label=label)
                )
    elif kind == "csv":
        wl = d.get("wavelength_nm")
        n = d.get("n")
        if not wl or not n:
            errors.append(_e("err_field_csv_no_data", lang, where=where))
        elif len(wl) < 2:
            errors.append(_e("err_field_csv_min_points", lang, where=where))
    else:
        errors.append(_e("err_field_kind_unsupported", lang, where=where, kind=kind))


def validate_config(config: dict, lang: str = DEFAULT_LANG) -> list[str]:
    """Validate a §2.2 stack-config dict.

    Returns localized error strings ([] if ok); default English.
    """

    errors: list[str] = []
    if not isinstance(config, dict):
        return [_e("err_stack_config_invalid", lang)]

    # Materials.
    if "incident" not in config:
        errors.append(_e("err_incident_missing", lang))
    else:
        _validate_material(config["incident"], _err("where_incident", lang), errors, lang)
    if "substrate" not in config:
        errors.append(_e("err_substrate_missing", lang))
    else:
        _validate_material(config["substrate"], _err("where_substrate", lang), errors, lang)

    layers = config.get("layers", [])
    if not isinstance(layers, list):
        errors.append(_e("err_layers_invalid", lang))
        layers = []
    for index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, dict) or "material" not in raw_layer:
            errors.append(_e("err_layer_invalid", lang, index=index))
            continue
        _validate_material(
            raw_layer["material"], _e("where_layer", lang, index=index), errors, lang
        )
        try:
            thickness = float(raw_layer.get("thickness_nm"))
            if thickness < 0:
                errors.append(_e("err_thickness_negative", lang, index=index))
        except (TypeError, ValueError):
            errors.append(_e("err_thickness_non_numeric", lang, index=index))

    # Grid.
    grid = config.get("grid")
    if not isinstance(grid, dict):
        errors.append(_e("err_grid_missing", lang))
    else:
        try:
            start = float(grid["start_nm"])
            stop = float(grid["stop_nm"])
            num = int(grid["num"])
            if num < 2:
                errors.append(_e("err_grid_min_points", lang))
            if start >= stop:
                errors.append(_e("err_grid_start_ge_stop", lang))
        except (TypeError, ValueError, KeyError):
            errors.append(_e("err_grid_params_invalid", lang))

    # Angle.
    if "angle_deg" in config:
        try:
            float(config["angle_deg"])
        except (TypeError, ValueError):
            errors.append(_e("err_angle_non_numeric", lang))

    # Polarization.
    pol = config.get("polarization", "s")
    if pol not in ("s", "p", "both"):
        errors.append(_e("err_polarization_invalid", lang, pol=pol))

    return errors


# ===========================================================================
# Simulation (ARCHITECTURE §6.1)
# ===========================================================================
def result_to_dict(result: SimulationResult) -> dict:
    """Serialize a :class:`SimulationResult` to a JSON-safe dict.

    Schema (ARCHITECTURE §6.1)::

        {
          "wavelength_nm": [...],
          "R": <1-D list> | [<s list>, <p list>],
          "T": ...,
          "A": ...,
          "polarizations": ["s"] | ["s", "p"],
        }

    ``r`` and ``t`` (complex amplitudes) are intentionally omitted — they are
    not plotted. For ``polarization="both"`` the R/T/A arrays keep their leading
    axis of size 2 in ``(s, p)`` order, so each becomes a list of two lists.
    """

    return {
        "wavelength_nm": _to_list(result.wavelength_nm),
        "R": _to_list(result.R),
        "T": _to_list(result.T),
        "A": _to_list(result.A),
        "polarizations": [str(p) for p in result.polarizations],
    }


def run_simulation(config: dict, lang: str = DEFAULT_LANG) -> dict:
    """Build stack + grid from ``config``, run ``simulate_spectrum``, serialize.

    Raises ``ValueError`` (joined localized messages) if ``validate_config``
    reports problems, so callbacks can surface a single clear error string.
    Default English.
    """

    errors = validate_config(config, lang=lang)
    if errors:
        raise ValueError(" ".join(errors))

    stack = stack_from_config(config, lang=lang)
    wavelengths = grid_from_config(config)
    result = simulate_spectrum(
        stack,
        wavelengths_nm=wavelengths,
        angle_deg=float(config.get("angle_deg", 0.0)),
        polarization=config.get("polarization", "s"),
    )
    return result_to_dict(result)


# ===========================================================================
# Analysis (ARCHITECTURE §6.1)
# ===========================================================================
def _channel_array(result_dict: dict, channel: str, lang: str = DEFAULT_LANG) -> np.ndarray:
    key = channel.upper()
    if key not in ("R", "T", "A"):
        raise ValueError(_e("err_channel_invalid", lang))
    return np.asarray(result_dict[key], dtype=float)


def analyze_result(
    result_dict: dict,
    channel: str = "R",
    feature: str = "peak",
    lang: str = DEFAULT_LANG,
) -> dict:
    """Run resonance analysis on one channel of a serialized result.

    Operates on 1-D spectra only. A 2-row ("both") result raises ``ValueError``
    — the caller must pick a single polarization before analyzing (the library
    enforces the same 1-D requirement). Error messages localized by ``lang``.
    """

    polarizations = result_dict.get("polarizations", ["s"])
    values = _channel_array(result_dict, channel, lang=lang)
    if values.ndim != 1 or len(polarizations) > 1:
        raise ValueError(_e("err_resonance_needs_single_pol", lang))

    wavelengths = np.asarray(result_dict["wavelength_nm"], dtype=float)
    resonance = analyze_resonance(wavelengths, values, feature=feature)
    return _resonance_to_dict(resonance)


def _resonance_to_dict(resonance: ResonanceResult) -> dict:
    """Serialize a :class:`ResonanceResult` to a JSON-safe dict."""

    return {
        "resonance_wavelength_nm": _to_float(resonance.resonance_wavelength_nm),
        "linewidth_nm": _to_float(resonance.linewidth_nm),
        "quality_factor": _to_float(resonance.quality_factor),
        "extremum_value": _to_float(resonance.extremum_value),
        "half_level": _to_float(resonance.half_level),
        "left_wavelength_nm": _to_float(resonance.left_wavelength_nm),
        "right_wavelength_nm": _to_float(resonance.right_wavelength_nm),
        "feature": str(resonance.feature),
    }


# ===========================================================================
# Optimization (workflow 2, ARCHITECTURE §6.1)
# ===========================================================================
def _reject_both_for_optimization(config: dict, lang: str = DEFAULT_LANG) -> None:
    if config.get("polarization", "s") == "both":
        raise ValueError(_e("err_optimize_needs_single_pol", lang))


def make_thickness_objective(config: dict, channel: str = "R", lang: str = DEFAULT_LANG):
    """Build a mean-channel objective closure for ``optimize_thicknesses``.

    Returns ``(objective, initial_thicknesses_nm)`` where ``objective`` maps a
    thickness vector (one entry per finite layer) to the mean of the requested
    spectral channel over the grid. The closure captures the stack, grid, angle,
    and polarization from ``config``. ``initial_thicknesses_nm`` is a
    ``jnp.ndarray`` (the optimizer differentiates through it).

    Rejects ``polarization="both"`` (the library cannot optimize 2-row spectra).
    Error messages localized by ``lang`` (default English).
    """

    _reject_both_for_optimization(config, lang=lang)
    key = channel.upper()
    if key not in ("R", "T", "A"):
        raise ValueError(_e("err_channel_invalid", lang))

    base_stack = stack_from_config(config, lang=lang)
    wavelengths = grid_from_config(config)
    angle_deg = float(config.get("angle_deg", 0.0))
    polarization = config.get("polarization", "s")
    initial_thicknesses = stack_thicknesses(base_stack)

    def objective(thicknesses_nm):
        candidate = stack_with_thicknesses(base_stack, thicknesses_nm)
        if key == "R":
            # mean_reflectivity is the library's optimized convenience path.
            return mean_reflectivity(
                candidate,
                wavelengths_nm=wavelengths,
                angle_deg=angle_deg,
                polarization=polarization,
            )
        result = simulate_spectrum(
            candidate,
            wavelengths_nm=wavelengths,
            angle_deg=angle_deg,
            polarization=polarization,
        )
        values = result.T if key == "T" else result.A
        import jax.numpy as _jnp

        return _jnp.mean(values)

    return objective, initial_thicknesses


# ===========================================================================
# Grouped/cavity optimization model (ARCHITECTURE §9)
#
# The Ottimizzazione tab models a resonant-cavity stack:
#   incident | input_group ×M | cavity (if enabled) | output_group ×K | substrate
# `expand_optimization_variables` is the single bridge from this grouped dict
# (§9.1) to the flat library Stack + variable GROUPS (§11): one inner list per
# optimization variable, holding the flat indices that share it. It replicates
# the Python-loop expansion of examples/optimize_resonance_target.py
# (M=K=3, Lin=Lout=2, cavity-only => 13 layers, cavity_index=6). The "tied"
# mode broadcasts a period-layer variable across ALL its repeats; "independent"
# mode treats each selected expanded flat layer as its own variable.
# `expand_optimization_config` is a compat shim flattening groups -> tuple.
# ===========================================================================
def _group_period_layers(group: dict, where: str, lang: str = DEFAULT_LANG) -> list[Layer]:
    """Build the period :class:`Layer` list of a mirror group (one repeat)."""

    if not isinstance(group, dict):
        raise ValueError(_e("err_group_invalid", lang, where=where))
    period: list[Layer] = []
    for index, raw_layer in enumerate(group.get("layers", [])):
        if not isinstance(raw_layer, dict) or "material" not in raw_layer:
            raise ValueError(
                _e("err_group_period_layer_invalid", lang, where=where, index=index)
            )
        material = material_from_dict(raw_layer["material"], lang=lang)
        thickness = float(raw_layer["thickness_nm"])
        period.append(Layer(material=material, thickness_nm=thickness))
    return period


class _ExpandedGeometry:
    """Internal container for the flat expansion + geometry of a grouped config.

    Shared by :func:`expand_optimization_variables` and
    :func:`enumerate_expanded_layers` so the flat layer list and the index
    arithmetic stay identical between the two public helpers.
    """

    __slots__ = (
        "incident",
        "substrate",
        "layers",
        "input_period",
        "output_period",
        "len_in",
        "len_out",
        "repeat_m",
        "repeat_k",
        "cavity_enabled",
        "cavity_index",
        "out_start",
    )

    def __init__(
        self,
        incident,
        substrate,
        layers,
        input_period,
        output_period,
        repeat_m,
        repeat_k,
        cavity_enabled,
        cavity_index,
        out_start,
    ):
        self.incident = incident
        self.substrate = substrate
        self.layers = layers
        self.input_period = input_period
        self.output_period = output_period
        self.len_in = len(input_period)
        self.len_out = len(output_period)
        self.repeat_m = repeat_m
        self.repeat_k = repeat_k
        self.cavity_enabled = cavity_enabled
        self.cavity_index = cavity_index
        self.out_start = out_start


def _expand_geometry(opt_stack_config: dict, lang: str = DEFAULT_LANG) -> _ExpandedGeometry:
    """Build the flat layer list + geometry from a grouped §9.1 config.

    Expansion order (ARCHITECTURE §9.2) — exactly the example's loops::

        input_group.layers × M | cavity (iff enabled) | output_group.layers × K

    Raises ``ValueError`` (localized by ``lang``) on a malformed config or an
    empty stack.
    """

    if not isinstance(opt_stack_config, dict):
        raise ValueError(_e("err_opt_stack_invalid", lang))

    incident = material_from_dict(opt_stack_config["incident"], lang=lang)
    substrate = material_from_dict(opt_stack_config["substrate"], lang=lang)

    input_group = opt_stack_config.get("input_group", {}) or {}
    output_group = opt_stack_config.get("output_group", {}) or {}
    cavity = opt_stack_config.get("cavity", {}) or {}

    repeat_m = int(input_group.get("repeat", 0))
    repeat_k = int(output_group.get("repeat", 0))
    if repeat_m < 0 or repeat_k < 0:
        raise ValueError(_e("err_repeat_negative", lang))

    input_period = _group_period_layers(input_group, _err("where_input_group", lang), lang)
    output_period = _group_period_layers(output_group, _err("where_output_group", lang), lang)
    len_in = len(input_period)

    cavity_enabled = bool(cavity.get("enabled", False))

    # --- Flat layer list (the example's Python loops). --------------------
    layers: list[Layer] = []
    for _ in range(repeat_m):
        layers.extend(input_period)

    cavity_index = repeat_m * len_in
    if cavity_enabled:
        cavity_material = material_from_dict(cavity["material"], lang=lang)
        cavity_thickness = float(cavity["thickness_nm"])
        layers.append(Layer(material=cavity_material, thickness_nm=cavity_thickness))

    out_start = repeat_m * len_in + (1 if cavity_enabled else 0)
    for _ in range(repeat_k):
        layers.extend(output_period)

    if not layers:
        raise ValueError(_e("err_stack_empty", lang))

    return _ExpandedGeometry(
        incident=incident,
        substrate=substrate,
        layers=layers,
        input_period=input_period,
        output_period=output_period,
        repeat_m=repeat_m,
        repeat_k=repeat_k,
        cavity_enabled=cavity_enabled,
        cavity_index=cavity_index,
        out_start=out_start,
    )


def _material_name(material_dict, fallback: str) -> str:
    """Return a material dict's ``name`` or a generated fallback label."""

    if isinstance(material_dict, dict):
        name = material_dict.get("name")
        if name:
            return str(name)
    return fallback


def enumerate_expanded_layers(
    opt_stack_config: dict, lang: str = DEFAULT_LANG
) -> list[dict]:
    """One entry per fully-expanded flat layer (ARCHITECTURE §11.4).

    Walks the SAME expansion order as :func:`expand_optimization_variables`
    (input_group.layers × M | cavity iff enabled | output_group.layers × K) and
    returns, per flat layer (in physical order), a JSON-safe dict::

        {
          "flat_index": int,            # index into the flat Stack.layers
          "label": str,                 # localized, 1-based period & layer numbers
          "material_name": str,
          "thickness_nm": float,        # current period/cavity thickness
        }

    Labels are localized by ``lang`` (default English). Pure; no Dash, no
    ``app.plots`` import. Raises ``ValueError`` (localized) on a malformed
    config, mirroring :func:`expand_optimization_variables`.
    """

    geometry = _expand_geometry(opt_stack_config, lang=lang)

    input_group = opt_stack_config.get("input_group", {}) or {}
    output_group = opt_stack_config.get("output_group", {}) or {}
    cavity = opt_stack_config.get("cavity", {}) or {}

    input_layer_dicts = input_group.get("layers", []) or []
    output_layer_dicts = output_group.get("layers", []) or []

    entries: list[dict] = []
    flat_index = 0

    # --- Input mirror block (M repeats of the input period). --------------
    for r in range(geometry.repeat_m):
        for p in range(geometry.len_in):
            raw = input_layer_dicts[p]
            material_name = _material_name(
                raw.get("material"), _e("lbl_fallback_input", lang, p=p + 1)
            )
            entries.append(
                {
                    "flat_index": flat_index,
                    "label": _e(
                        "lbl_input_layer", lang, r=r + 1, p=p + 1, name=material_name
                    ),
                    "material_name": material_name,
                    "thickness_nm": float(raw.get("thickness_nm", 0.0)),
                }
            )
            flat_index += 1

    # --- Cavity (single layer, iff enabled). ------------------------------
    if geometry.cavity_enabled:
        material_name = _material_name(
            cavity.get("material"), _err("lbl_fallback_cavity", lang)
        )
        entries.append(
            {
                "flat_index": flat_index,
                "label": _e("lbl_cavity", lang, name=material_name),
                "material_name": material_name,
                "thickness_nm": float(cavity.get("thickness_nm", 0.0)),
            }
        )
        flat_index += 1

    # --- Output mirror block (K repeats of the output period). ------------
    for r in range(geometry.repeat_k):
        for p in range(geometry.len_out):
            raw = output_layer_dicts[p]
            material_name = _material_name(
                raw.get("material"), _e("lbl_fallback_output", lang, p=p + 1)
            )
            entries.append(
                {
                    "flat_index": flat_index,
                    "label": _e(
                        "lbl_output_layer", lang, r=r + 1, p=p + 1, name=material_name
                    ),
                    "material_name": material_name,
                    "thickness_nm": float(raw.get("thickness_nm", 0.0)),
                }
            )
            flat_index += 1

    return entries


def expand_optimization_variables(
    opt_stack_config: dict, lang: str = DEFAULT_LANG
) -> tuple[Stack, list[list[int]]]:
    """Expand a grouped §9.1 config into a flat :class:`Stack` + variable GROUPS.

    The flat layer list is built IDENTICALLY to §9.2; only the variable-set
    construction changes by mode (``opt_stack_config["variable"]["mode"]``,
    default ``"tied"``).

    Returns ``(stack, groups)`` where ``groups: list[list[int]]`` — one inner
    list per optimization VARIABLE; each inner list holds the flat indices that
    SHARE that variable. Invariants: every inner list is non-empty and sorted
    ascending; ``groups`` is sorted by each group's first flat index; the union
    of all groups is the selected, de-duplicated variable set; ``groups`` is
    non-empty (raises ``ValueError``, Italian, if no selection).

    MODE == "tied": for an input period-layer ``i`` the group spans all M
    repeats ``[i, i+Lin, ...]``; for an output period-layer ``j`` it spans all K
    repeats ``[out_start+j, out_start+j+Lout, ...]``; ``variable.cavity``
    (requires the cavity enabled) is the singleton ``[cavity_index]``.

    MODE == "independent": each selected flat index in ``variable.flat_layers``
    becomes its own singleton group; ``cavity``/``input_layers``/``output_layers``
    are ignored.

    Pure; no Dash. Error messages localized by ``lang`` (default English).
    """

    geometry = _expand_geometry(opt_stack_config, lang=lang)
    stack = Stack(
        incident=geometry.incident,
        layers=geometry.layers,
        substrate=geometry.substrate,
    )
    num_layers = len(geometry.layers)

    variable = opt_stack_config.get("variable", {}) or {}
    if not isinstance(variable, dict):
        raise ValueError(_e("err_variable_selector_invalid", lang))
    mode = variable.get("mode", "tied")
    if mode not in ("tied", "independent"):
        raise ValueError(_e("err_variable_mode_invalid", lang, mode=mode))

    groups: list[list[int]] = []

    if mode == "independent":
        flat_layers = variable.get("flat_layers", []) or []
        selected: set[int] = set()
        for raw in flat_layers:
            try:
                n = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    _e("err_flat_index_invalid", lang, value=raw)
                ) from exc
            if not (0 <= n < num_layers):
                raise ValueError(_e("err_flat_index_out_of_range", lang, value=n))
            selected.add(n)
        groups = [[n] for n in sorted(selected)]
    else:  # tied
        # cavity -> singleton [cavity_index]
        if variable.get("cavity"):
            if not geometry.cavity_enabled:
                raise ValueError(_e("err_cavity_disabled_but_variable", lang))
            groups.append([geometry.cavity_index])

        # input period-layer i -> all M repeats
        for raw in variable.get("input_layers", []) or []:
            i = int(raw)
            if not (0 <= i < geometry.len_in):
                raise ValueError(_e("err_input_layer_out_of_range", lang, index=i))
            group = [i + r * geometry.len_in for r in range(geometry.repeat_m)]
            if group:
                groups.append(sorted(group))

        # output period-layer j -> all K repeats
        for raw in variable.get("output_layers", []) or []:
            j = int(raw)
            if not (0 <= j < geometry.len_out):
                raise ValueError(_e("err_output_layer_out_of_range", lang, index=j))
            group = [
                geometry.out_start + j + r * geometry.len_out
                for r in range(geometry.repeat_k)
            ]
            if group:
                groups.append(sorted(group))

    if not groups:
        raise ValueError(_e("err_select_one_variable", lang))

    # Dedupe identical groups, sort inner lists, then sort groups by first index.
    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for group in groups:
        ordered = sorted(set(group))
        key = tuple(ordered)
        if key not in seen:
            seen.add(key)
            deduped.append(ordered)
    deduped.sort(key=lambda g: g[0])

    # Internal sanity check: every index must address a real flat layer.
    if any(idx < 0 or idx >= num_layers for group in deduped for idx in group):
        raise ValueError(_e("err_expanded_index_invalid", lang))

    return stack, deduped


def expand_optimization_config(
    opt_stack_config: dict, lang: str = DEFAULT_LANG
) -> tuple[Stack, tuple[int, ...]]:
    """DEPRECATED compat shim (ARCHITECTURE §11.4).

    Delegates to :func:`expand_optimization_variables` and flattens the groups
    back to the legacy sorted, de-duplicated ``variable_layer_indices`` tuple
    (the union of all groups). Retained so any caller/test on the old signature
    keeps working; the ``run_*`` paths consume groups directly. ``lang`` is
    threaded through for localized error messages.
    """

    stack, groups = expand_optimization_variables(opt_stack_config, lang=lang)
    flat = sorted({idx for group in groups for idx in group})
    return stack, tuple(flat)


def validate_opt_stack_config(opt_stack_config: dict, lang: str = DEFAULT_LANG) -> list[str]:
    """Validate a grouped §9.1 config.

    Returns localized error strings ([] if ok); default English. Covers §9.5:
    ``repeat`` int >= 0; non-empty expanded stack; group layers non-empty when
    its repeat >= 1; valid grid/angle/polarization; ``variable`` must select
    >= 1 expandable thickness and ``variable.cavity`` requires an enabled
    cavity. Per-layer materials/thicknesses reuse the §5 checks.
    """

    errors: list[str] = []
    if not isinstance(opt_stack_config, dict):
        return [_e("err_opt_stack_invalid", lang)]

    if "incident" not in opt_stack_config:
        errors.append(_e("err_incident_missing", lang))
    else:
        _validate_material(
            opt_stack_config["incident"], _err("where_incident", lang), errors, lang
        )
    if "substrate" not in opt_stack_config:
        errors.append(_e("err_substrate_missing", lang))
    else:
        _validate_material(
            opt_stack_config["substrate"], _err("where_substrate", lang), errors, lang
        )

    def _validate_group(group, label, repeat_value):
        if not isinstance(group, dict):
            errors.append(_e("err_group_definition_invalid", lang, label=label))
            return
        glayers = group.get("layers", [])
        if not isinstance(glayers, list):
            errors.append(_e("err_group_layers_invalid", lang, label=label))
            glayers = []
        if repeat_value >= 1 and not glayers:
            errors.append(_e("err_group_period_empty", lang, label=label))
        for index, raw_layer in enumerate(glayers):
            if not isinstance(raw_layer, dict) or "material" not in raw_layer:
                errors.append(
                    _e("err_group_layer_invalid", lang, label=label, index=index)
                )
                continue
            _validate_material(
                raw_layer["material"], f"{label} strato {index}", errors, lang
            )
            try:
                thickness = float(raw_layer.get("thickness_nm"))
                if thickness < 0:
                    errors.append(
                        _e("err_group_thickness_negative", lang, label=label, index=index)
                    )
            except (TypeError, ValueError):
                errors.append(
                    _e("err_group_thickness_non_numeric", lang, label=label, index=index)
                )

    input_group = opt_stack_config.get("input_group", {}) or {}
    output_group = opt_stack_config.get("output_group", {}) or {}
    input_label = _err("where_input_group", lang)
    output_label = _err("where_output_group", lang)

    def _int_repeat(group, label):
        raw = group.get("repeat", 0) if isinstance(group, dict) else 0
        try:
            value = int(raw)
        except (TypeError, ValueError):
            errors.append(_e("err_repeat_not_int", lang, label=label))
            return 0
        if value < 0:
            errors.append(_e("err_repeat_ge_zero", lang, label=label))
            return 0
        return value

    repeat_m = _int_repeat(input_group, input_label)
    repeat_k = _int_repeat(output_group, output_label)
    _validate_group(input_group, input_label, repeat_m)
    _validate_group(output_group, output_label, repeat_k)

    cavity = opt_stack_config.get("cavity", {}) or {}
    cavity_enabled = bool(cavity.get("enabled", False)) if isinstance(cavity, dict) else False
    if cavity_enabled:
        _validate_material(cavity.get("material"), _err("where_cavity", lang), errors, lang)
        try:
            if float(cavity.get("thickness_nm")) < 0:
                errors.append(_e("err_cavity_thickness_negative", lang))
        except (TypeError, ValueError):
            errors.append(_e("err_cavity_thickness_non_numeric", lang))

    # Non-empty expanded stack.
    len_in = len(input_group.get("layers", []) or []) if isinstance(input_group, dict) else 0
    len_out = len(output_group.get("layers", []) or []) if isinstance(output_group, dict) else 0
    if (repeat_m * len_in + repeat_k * len_out + (1 if cavity_enabled else 0)) == 0:
        errors.append(_e("err_stack_empty", lang))

    # Variable selector (mode-aware, §11.4).
    variable = opt_stack_config.get("variable", {}) or {}
    if not isinstance(variable, dict):
        errors.append(_e("err_variable_selector_invalid", lang))
        variable = {}
    mode = variable.get("mode", "tied")
    if mode not in ("tied", "independent"):
        errors.append(_e("err_variable_mode_invalid", lang, mode=mode))

    if mode == "independent":
        # Independent mode reads only flat_layers (every group is a singleton).
        num_flat = repeat_m * len_in + repeat_k * len_out + (1 if cavity_enabled else 0)
        flat_layers = variable.get("flat_layers", [])
        if not isinstance(flat_layers, list) or not flat_layers:
            errors.append(_e("err_select_one_variable", lang))
        else:
            for raw in flat_layers:
                try:
                    n = int(raw)
                except (TypeError, ValueError):
                    errors.append(_e("err_flat_index_invalid", lang, value=raw))
                    continue
                if not (0 <= n < num_flat):
                    errors.append(_e("err_flat_index_out_of_range", lang, value=n))
    else:
        # Tied mode reads cavity / input_layers / output_layers.
        selected_count = 0
        if variable.get("cavity"):
            if not cavity_enabled:
                errors.append(_e("err_cavity_disabled_but_variable", lang))
            else:
                selected_count += 1
        selected_count += len(variable.get("input_layers", []) or [])
        selected_count += len(variable.get("output_layers", []) or [])
        if selected_count == 0:
            errors.append(_e("err_select_one_variable_tied", lang))

    # Grid / angle / polarization (reuse §5 rules; "both" allowed here but
    # rejected later by the run_* entry points for optimization).
    grid = opt_stack_config.get("grid")
    if not isinstance(grid, dict):
        errors.append(_e("err_grid_missing", lang))
    else:
        try:
            start = float(grid["start_nm"])
            stop = float(grid["stop_nm"])
            num = int(grid["num"])
            if num < 2:
                errors.append(_e("err_grid_min_points", lang))
            if start >= stop:
                errors.append(_e("err_grid_start_ge_stop", lang))
        except (TypeError, ValueError, KeyError):
            errors.append(_e("err_grid_params_invalid", lang))

    if "angle_deg" in opt_stack_config:
        try:
            float(opt_stack_config["angle_deg"])
        except (TypeError, ValueError):
            errors.append(_e("err_angle_non_numeric", lang))

    pol = opt_stack_config.get("polarization", "s")
    if pol not in ("s", "p", "both"):
        errors.append(_e("err_polarization_invalid", lang, pol=pol))

    return errors


def _select_channel_1d(result: SimulationResult, channel: str, lang: str = DEFAULT_LANG):
    """Select one R/T/A channel as a 1-D array, rejecting 2-row "both" spectra.

    Mirrors the library's internal ``_select_spectrum`` (optimize.py) but raises
    the GUI's localized error so the message is consistent at the UI boundary.
    """

    key = channel.upper()
    if key == "R":
        values = result.R
    elif key == "T":
        values = result.T
    elif key == "A":
        values = result.A
    else:
        raise ValueError(_e("err_channel_invalid", lang))
    if getattr(values, "ndim", 1) == 2:
        raise ValueError(_e("err_optimize_needs_single_pol", lang))
    return values


def run_resonance_optimization(
    opt_stack_config: dict, opt_config: dict, lang: str = DEFAULT_LANG
) -> dict:
    """Optimize toward a target resonance wavelength + Q on a grouped §9.1 config.

    Expands ``opt_stack_config`` via :func:`expand_optimization_variables` into a
    flat ``Stack`` + variable ``groups`` (one variable shared across all flat
    copies of each group), reads grid/angle/polarization from the grouped config,
    and builds a custom objective on the PUBLIC ``resonance_target_loss`` (the
    library's ``optimize_resonance_target`` only supports a 1:1 index map and
    cannot share one variable across copies — §11.4). The objective mirrors the
    library's internal one (``optimize.py:160-179``): candidate thicknesses ->
    ``stack_with_thicknesses`` -> ``simulate_spectrum`` -> select the chosen 1-D
    channel -> ``resonance_target_loss``; driven by ``optimize_thicknesses``.

    Scalar targets/weights/steps come from ``opt_config`` (§2.3).
    ``polarization="both"`` is rejected (Italian error).

    Output schema (ARCHITECTURE §6.1)::

        {
          "thicknesses_nm": [...],            # full optimized stack thicknesses
          "variable_thicknesses_nm": [...],   # one value PER GROUP/variable
          "history": [...],                   # loss per step
          "resonance": <ResonanceResult-as-dict>,
          "final_result": <result_to_dict of the optimized stack>,
        }
    """

    import jax.numpy as _jnp

    errors = validate_opt_stack_config(opt_stack_config, lang=lang)
    if errors:
        raise ValueError(" ".join(errors))
    _reject_both_for_optimization(opt_stack_config, lang=lang)

    stack, groups = expand_optimization_variables(opt_stack_config, lang=lang)
    wavelengths = grid_from_config(opt_stack_config)
    polarization = opt_stack_config.get("polarization", "s")
    angle_deg = float(opt_stack_config.get("angle_deg", 0.0))

    channel = opt_config.get("spectrum", "R")
    feature = opt_config.get("feature", "peak")
    target_wavelength_nm = float(opt_config["target_wavelength_nm"])
    target_q = float(opt_config["target_q"])
    wavelength_weight = float(opt_config.get("wavelength_weight", 1.0))
    q_weight = float(opt_config.get("q_weight", 1.0))
    sharpness = float(opt_config.get("sharpness", 20.0))

    base = stack_thicknesses(stack)
    group_arrays = [_jnp.asarray(g, dtype=_jnp.int32) for g in groups]
    initial = _jnp.asarray([base[g[0]] for g in groups])

    def to_full(v):
        full = base
        for gi, idx in enumerate(group_arrays):
            full = full.at[idx].set(v[gi])
        return full

    def objective(v):
        candidate = stack_with_thicknesses(stack, to_full(v))
        result = simulate_spectrum(
            candidate,
            wavelengths_nm=wavelengths,
            angle_deg=angle_deg,
            polarization=polarization,
        )
        values = _select_channel_1d(result, channel, lang=lang)
        return resonance_target_loss(
            wavelength_nm=wavelengths,
            spectrum_values=values,
            target_wavelength_nm=target_wavelength_nm,
            target_q=target_q,
            feature=feature,
            wavelength_weight=wavelength_weight,
            q_weight=q_weight,
            sharpness=sharpness,
        )

    optimized: OptimizationResult = optimize_thicknesses(
        objective,
        initial_thicknesses_nm=initial,
        steps=int(opt_config.get("steps", 100)),
        learning_rate=float(opt_config.get("learning_rate", 0.1)),
        lower_bound_nm=float(opt_config.get("lower_bound_nm", 0.0)),
    )

    final_thicknesses = to_full(optimized.thicknesses_nm)
    optimized_stack = stack_with_thicknesses(stack, final_thicknesses)
    final_simulation = simulate_spectrum(
        optimized_stack,
        wavelengths_nm=wavelengths,
        angle_deg=angle_deg,
        polarization=polarization,
    )
    final_channel = _select_channel_1d(final_simulation, channel, lang=lang)
    resonance = analyze_resonance(wavelengths, final_channel, feature=feature)

    return {
        "thicknesses_nm": _to_list(final_thicknesses),
        "variable_thicknesses_nm": _to_list(optimized.thicknesses_nm),
        "history": _to_list(optimized.history),
        "resonance": _resonance_to_dict(resonance),
        "final_result": result_to_dict(final_simulation),
    }


def run_thickness_optimization(
    opt_stack_config: dict, opt_config: dict, lang: str = DEFAULT_LANG
) -> dict:
    """Generic-loss thickness optimization over ONLY the selected variables.

    Expands the grouped §9.1 config via :func:`expand_optimization_variables`
    into a flat ``Stack`` + variable ``groups``, builds an objective that moves
    one variable per group (scatter-broadcast to all of the group's flat copies,
    so the period stays uniform in tied mode and the M/K replication is never
    treated as a free variable), and runs the UNCHANGED library
    ``optimize_thicknesses`` (mean-channel loss; ``mean_reflectivity`` for R).
    ``polarization="both"`` is rejected.

    Output schema (ARCHITECTURE §6.1)::

        {
          "thicknesses_nm": [...],
          "variable_thicknesses_nm": [...],   # one value PER GROUP/variable
          "history": [...],
          "final_result": <result_to_dict of the optimized stack>,
        }
    """

    import jax.numpy as _jnp

    errors = validate_opt_stack_config(opt_stack_config, lang=lang)
    if errors:
        raise ValueError(" ".join(errors))
    _reject_both_for_optimization(opt_stack_config, lang=lang)

    stack, groups = expand_optimization_variables(opt_stack_config, lang=lang)
    wavelengths = grid_from_config(opt_stack_config)
    polarization = opt_stack_config.get("polarization", "s")
    angle_deg = float(opt_stack_config.get("angle_deg", 0.0))

    channel = opt_config.get("spectrum", "R")
    key = channel.upper()
    if key not in ("R", "T", "A"):
        raise ValueError(_e("err_channel_invalid", lang))

    base = stack_thicknesses(stack)
    group_arrays = [_jnp.asarray(g, dtype=_jnp.int32) for g in groups]
    initial = _jnp.asarray([base[g[0]] for g in groups])

    def to_full(v):
        full = base
        for gi, idx in enumerate(group_arrays):
            full = full.at[idx].set(v[gi])
        return full

    def objective(v):
        candidate = stack_with_thicknesses(stack, to_full(v))
        if key == "R":
            return mean_reflectivity(
                candidate,
                wavelengths_nm=wavelengths,
                angle_deg=angle_deg,
                polarization=polarization,
            )
        result = simulate_spectrum(
            candidate,
            wavelengths_nm=wavelengths,
            angle_deg=angle_deg,
            polarization=polarization,
        )
        values = _select_channel_1d(result, key, lang=lang)
        return _jnp.mean(values)

    optimized: OptimizationResult = optimize_thicknesses(
        objective,
        initial_thicknesses_nm=initial,
        steps=int(opt_config.get("steps", 100)),
        learning_rate=float(opt_config.get("learning_rate", 0.1)),
        lower_bound_nm=float(opt_config.get("lower_bound_nm", 0.0)),
    )

    full_thicknesses = to_full(optimized.thicknesses_nm)
    optimized_stack = stack_with_thicknesses(stack, full_thicknesses)
    final_simulation = simulate_spectrum(
        optimized_stack,
        wavelengths_nm=wavelengths,
        angle_deg=angle_deg,
        polarization=polarization,
    )

    return {
        "thicknesses_nm": _to_list(full_thicknesses),
        "variable_thicknesses_nm": _to_list(optimized.thicknesses_nm),
        "history": _to_list(optimized.history),
        "final_result": result_to_dict(final_simulation),
    }


# ===========================================================================
# Export (workflow 2): optimized spectra + parameters as connected .txt files
#
# The Ottimizzazione "Export" button delivers two files sharing one prefix:
#   <prefix>_spectra.txt      tab/space columns: wavelength_nm R T A
#   <prefix>_parameters.txt   human-readable optimization + structure dump
# where <prefix> = "simulation_<YYYYMMDD_HHMMSS>". Builders are pure (str in /
# str out) so they unit-test without Dash; the callback only wraps them in
# dcc.send_string. Mirrors the column convention of multilayer_tmm.io.
# ===========================================================================
def make_export_prefix(
    simulation_name: str = "simulation", timestamp: str | None = None
) -> tuple[str, str]:
    """Return ``(file_prefix, timestamp_text)`` for a connected export pair.

    ``file_prefix`` is ``"<simulation_name>_<timestamp>"`` (e.g.
    ``"simulation_20260616_143022"``); both export files share it so their
    names stay connected (``..._spectra.txt`` / ``..._parameters.txt``).
    """

    timestamp_text = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{simulation_name}_{timestamp_text}", timestamp_text


def _export_float(value) -> str:
    """Format a scalar for export text, leaving non-numbers untouched."""

    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _material_summary(material: dict | None) -> str:
    """One-line material description for the parameters dump."""

    material = material or {}
    name = material.get("name") or "(unnamed)"
    if material.get("kind") == "csv":
        return f"{name} (csv table)"
    n = _export_float(material.get("n", 0.0))
    k = _export_float(material.get("k", 0.0))
    return f"{name} (n={n}, k={k})"


def build_spectra_text(
    spectrum: dict, *, file_prefix: str, timestamp: str,
    simulation_name: str = "simulation",
) -> str:
    """Build a ``<prefix>_spectra.txt`` body from a spectrum dict.

    ``spectrum`` is a :func:`result_to_dict` payload (``wavelength_nm`` + R/T/A
    + ``polarizations``). Columns are ``wavelength_nm R T A`` for one
    polarization, or grouped per-polarization columns for ``"both"``. Raises
    ``ValueError`` when no spectrum is available.
    """

    if not spectrum or "wavelength_nm" not in spectrum:
        raise ValueError("No spectrum to export.")

    wavelengths = np.asarray(spectrum["wavelength_nm"], dtype=float)
    polarizations = spectrum.get("polarizations") or ["s"]

    def _rows(values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array[np.newaxis, :] if array.ndim == 1 else array

    R, T, A = _rows(spectrum["R"]), _rows(spectrum["T"]), _rows(spectrum["A"])

    columns = ["wavelength_nm"]
    arrays = [wavelengths]
    for index, polarization in enumerate(polarizations):
        suffix = "" if len(polarizations) == 1 else f"_{polarization}"
        columns.extend([f"R{suffix}", f"T{suffix}", f"A{suffix}"])
        arrays.extend([R[index], T[index], A[index]])

    header = "\n".join(
        (
            f"simulation_name: {simulation_name}",
            f"timestamp: {timestamp}",
            f"file_prefix: {file_prefix}",
            "columns: " + " ".join(columns),
        )
    )
    buffer = io.StringIO()
    np.savetxt(buffer, np.column_stack(arrays), header=header, comments="# ", fmt="%.12g")
    return buffer.getvalue()


def build_optimized_spectra_text(
    result_dict: dict, *, file_prefix: str, timestamp: str,
    simulation_name: str = "simulation",
) -> str:
    """Build the ``<prefix>_spectra.txt`` body from an optimization result.

    Thin wrapper over :func:`build_spectra_text` for the ``final_result`` of an
    optimization-result dict. Raises ``ValueError`` if absent.
    """

    final = (result_dict or {}).get("final_result")
    if not final:
        raise ValueError("No optimized spectrum to export.")
    return build_spectra_text(
        final, file_prefix=file_prefix, timestamp=timestamp,
        simulation_name=simulation_name,
    )


def build_simulation_parameters_text(
    stack_config: dict, result_dict: dict, *,
    file_prefix: str, timestamp: str, simulation_name: str = "simulated",
) -> str:
    """Build the ``<prefix>_parameters.txt`` body for a flat-stack simulation.

    Captures the grid/incidence and the finite-layer stack structure (incident,
    each finite layer with material + thickness, substrate) from the §2.2 flat
    ``stack_config``. ``result_dict`` is accepted for symmetry / future use.
    """

    cfg = stack_config or {}
    grid = cfg.get("grid", {}) or {}
    lines: list[str] = [
        "# multilayer_tmm — simulation parameters",
        f"simulation_name: {simulation_name}",
        f"timestamp: {timestamp}",
        f"file_prefix: {file_prefix}",
        "",
        "[Grid / incidence]",
        f"wavelength_start_nm: {_export_float(grid.get('start_nm'))}",
        f"wavelength_stop_nm: {_export_float(grid.get('stop_nm'))}",
        f"num_points: {grid.get('num')}",
        f"angle_deg: {_export_float(cfg.get('angle_deg', 0.0))}",
        f"polarization: {cfg.get('polarization', 's')}",
        "",
        "[Structure]",
        f"incident: {_material_summary(cfg.get('incident'))}",
        "finite_layers (top -> bottom):",
    ]
    layers = cfg.get("layers", []) or []
    if layers:
        for index, layer in enumerate(layers, start=1):
            thickness = _export_float((layer or {}).get("thickness_nm"))
            lines.append(
                f"  layer {index}: {_material_summary((layer or {}).get('material'))}"
                f" thickness={thickness} nm"
            )
    else:
        lines.append("  (none)")
    lines.append(f"substrate: {_material_summary(cfg.get('substrate'))}")

    return "\n".join(lines) + "\n"


def build_optimization_parameters_text(
    opt_stack_config: dict, opt_config: dict, result_dict: dict, *,
    file_prefix: str, timestamp: str, simulation_name: str = "simulation",
) -> str:
    """Build the ``<prefix>_parameters.txt`` body (settings + structure + result).

    Captures the optimization settings, the grid/incidence, the grouped stack
    structure (with its initial thicknesses and M/K repeats), the variable
    selection, and the achieved result (optimized thicknesses + resonance).
    """

    cfg = opt_stack_config or {}
    opt = opt_config or {}
    result = result_dict or {}
    lines: list[str] = [
        "# multilayer_tmm — optimization parameters",
        f"simulation_name: {simulation_name}",
        f"timestamp: {timestamp}",
        f"file_prefix: {file_prefix}",
        "",
        "[Optimization settings]",
        f"mode: {opt.get('mode', 'resonance')}",
        f"spectral_channel: {opt.get('spectrum', 'R')}",
        f"feature: {opt.get('feature', 'peak')}",
        f"target_wavelength_nm: {_export_float(opt.get('target_wavelength_nm'))}",
        f"target_q: {_export_float(opt.get('target_q'))}",
        f"wavelength_weight: {_export_float(opt.get('wavelength_weight'))}",
        f"q_weight: {_export_float(opt.get('q_weight'))}",
        f"sharpness: {_export_float(opt.get('sharpness'))}",
        f"steps: {opt.get('steps')}",
        f"learning_rate: {_export_float(opt.get('learning_rate'))}",
        f"minimum_thickness_nm: {_export_float(opt.get('lower_bound_nm'))}",
    ]

    variable = cfg.get("variable", {}) or {}
    lines.extend(
        [
            f"variable_mode: {variable.get('mode', 'tied')}",
            f"variable_cavity: {bool(variable.get('cavity', False))}",
            f"variable_input_layers: {list(variable.get('input_layers', []))}",
            f"variable_output_layers: {list(variable.get('output_layers', []))}",
            f"variable_flat_layers: {list(variable.get('flat_layers', []))}",
        ]
    )

    grid = cfg.get("grid", {}) or {}
    lines.extend(
        [
            "",
            "[Grid / incidence]",
            f"wavelength_start_nm: {_export_float(grid.get('start_nm'))}",
            f"wavelength_stop_nm: {_export_float(grid.get('stop_nm'))}",
            f"num_points: {grid.get('num')}",
            f"angle_deg: {_export_float(cfg.get('angle_deg', 0.0))}",
            f"polarization: {cfg.get('polarization', 's')}",
        ]
    )

    input_group = cfg.get("input_group", {}) or {}
    output_group = cfg.get("output_group", {}) or {}
    cavity = cfg.get("cavity", {}) or {}
    lines.extend(["", "[Structure]", f"incident: {_material_summary(cfg.get('incident'))}"])

    def _period_lines(group: dict, role: str) -> list[str]:
        repeat = group.get("repeat", 0)
        out = [f"{role}_mirror_period (x{repeat}):"]
        for index, layer in enumerate(group.get("layers", []) or []):
            thickness = _export_float((layer or {}).get("thickness_nm"))
            out.append(
                f"  layer {index}: {_material_summary((layer or {}).get('material'))}"
                f" thickness={thickness} nm"
            )
        return out

    lines.extend(_period_lines(input_group, "input"))
    cavity_state = "enabled" if cavity.get("enabled", False) else "disabled"
    lines.append(
        f"cavity ({cavity_state}): {_material_summary(cavity.get('material'))}"
        f" thickness={_export_float(cavity.get('thickness_nm'))} nm"
    )
    lines.extend(_period_lines(output_group, "output"))
    lines.append(f"substrate: {_material_summary(cfg.get('substrate'))}")

    lines.extend(["", "[Optimized result]"])
    thicknesses = result.get("thicknesses_nm")
    if thicknesses is not None:
        formatted = ", ".join(_export_float(value) for value in thicknesses)
        lines.append(f"optimized_thicknesses_nm: [{formatted}]")
    variable_thicknesses = result.get("variable_thicknesses_nm")
    if variable_thicknesses is not None:
        formatted = ", ".join(_export_float(value) for value in variable_thicknesses)
        lines.append(f"variable_thicknesses_nm: [{formatted}]")
    resonance = result.get("resonance")
    if resonance:
        lines.extend(
            [
                f"resonance_wavelength_nm: {_export_float(resonance.get('resonance_wavelength_nm'))}",
                f"quality_factor_Q: {_export_float(resonance.get('quality_factor'))}",
                f"linewidth_nm: {_export_float(resonance.get('linewidth_nm'))}",
                f"extremum_value: {_export_float(resonance.get('extremum_value'))}",
            ]
        )

    return "\n".join(lines) + "\n"


def build_export_zip_bytes(
    file_prefix: str, *, spectra_text: str, params_text: str
) -> bytes:
    """Pack the two export texts into one ZIP, preserving the connected names.

    Returns the bytes of a ZIP archive containing ``<prefix>_spectra.txt`` and
    ``<prefix>_parameters.txt``. A single download avoids the browser dropping
    the second of two simultaneous programmatic downloads.
    """

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{file_prefix}_spectra.txt", spectra_text)
        archive.writestr(f"{file_prefix}_parameters.txt", params_text)
    return buffer.getvalue()
