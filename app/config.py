"""Constants: defaults, cache dir, option lists, and i18n translation catalog.

No Dash imports — this module is safe to import from anywhere (including the
pure :mod:`app.state`). Default UI language is English; Italian is selectable
at runtime via the language selector (ARCHITECTURE §12). Option *values* are
English snake_case so they map directly onto the library API.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Cache directory for the DiskcacheManager (ARCHITECTURE §4)
# ---------------------------------------------------------------------------
CACHE_DIR = os.environ.get(
    "TMM_GUI_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".dash_cache"),
)

# ---------------------------------------------------------------------------
# Option *values* — English snake_case; feed the library unchanged.
# Display text has moved to the TRANSLATIONS catalog (§12.6).
# ---------------------------------------------------------------------------
POLARIZATION_VALUES = ("s", "p", "both")
FEATURE_VALUES = ("peak", "dip")
SPECTRUM_VALUES = ("R", "T", "A")
MATERIAL_KIND_VALUES = ("constant", "csv")
OPTIMIZE_MODE_VALUES = ("resonance", "mean_r")

# Legacy (value, label) option tuples — RETAINED for callers that still iterate
# (value, text). Labels are in Italian here; use options_for() for i18n.
POLARIZATION_OPTIONS: list[tuple[str, str]] = [
    ("s", "s (TE)"),
    ("p", "p (TM)"),
    ("both", "Entrambe (s e p)"),
]
FEATURE_OPTIONS: list[tuple[str, str]] = [
    ("peak", "Picco (massimo)"),
    ("dip", "Avvallamento (minimo)"),
]
SPECTRUM_OPTIONS: list[tuple[str, str]] = [
    ("R", "Riflettanza (R)"),
    ("T", "Trasmittanza (T)"),
    ("A", "Assorbanza (A)"),
]
MATERIAL_KIND_OPTIONS: list[tuple[str, str]] = [
    ("constant", "Indice costante (n, k)"),
    ("csv", "File CSV (n, k tabulati)"),
]
OPTIMIZE_MODE_OPTIONS: list[tuple[str, str]] = [
    ("resonance", "Risonanza mirata (lambda + Q)"),
    ("mean_r", "Minimizza R media"),
]

# ---------------------------------------------------------------------------
# Defaults for a fresh stack-config dict (ARCHITECTURE §2.2)
# ---------------------------------------------------------------------------
DEFAULT_GRID = {"start_nm": 400.0, "stop_nm": 800.0, "num": 401}
DEFAULT_ANGLE_DEG = 0.0
DEFAULT_POLARIZATION = "s"

DEFAULT_INCIDENT_MATERIAL = {"kind": "constant", "n": 1.0, "k": 0.0, "name": "Aria"}
DEFAULT_SUBSTRATE_MATERIAL = {"kind": "constant", "n": 1.46, "k": 0.0, "name": "SiO2"}
DEFAULT_LAYER_MATERIAL = {"kind": "constant", "n": 2.0, "k": 0.0, "name": "Strato"}
DEFAULT_LAYER_THICKNESS_NM = 120.0


def default_stack_config() -> dict:
    """Return a fresh, valid §2.2 stack-config dict."""

    return {
        "incident": dict(DEFAULT_INCIDENT_MATERIAL),
        "layers": [
            {"material": dict(DEFAULT_LAYER_MATERIAL), "thickness_nm": DEFAULT_LAYER_THICKNESS_NM}
        ],
        "substrate": dict(DEFAULT_SUBSTRATE_MATERIAL),
        "grid": dict(DEFAULT_GRID),
        "angle_deg": DEFAULT_ANGLE_DEG,
        "polarization": DEFAULT_POLARIZATION,
    }


# ---------------------------------------------------------------------------
# Defaults for the grouped/cavity optimize-stack-config dict (ARCHITECTURE §9.1)
#
# These mirror examples/optimize_resonance_target.py exactly: M = K = 3,
# Lin = Lout = 2, cavity enabled, cavity-only selected as variable. The
# expansion (state.expand_optimization_config) therefore yields a 13-layer flat
# Stack with cavity at flat index 6 and variable_layer_indices == (6,).
# ---------------------------------------------------------------------------
DEFAULT_OPT_GRID = {"start_nm": 520.0, "stop_nm": 720.0, "num": 241}
DEFAULT_OPT_REPEAT_M = 3  # input_group.repeat
DEFAULT_OPT_REPEAT_K = 3  # output_group.repeat

# Materials matching the example (air incident/substrate; high/low mirrors).
DEFAULT_OPT_INCIDENT_MATERIAL = {"kind": "constant", "n": 1.0, "k": 0.0, "name": "air"}
DEFAULT_OPT_SUBSTRATE_MATERIAL = {"kind": "constant", "n": 1.0, "k": 0.0, "name": "air"}
DEFAULT_OPT_HIGH_MATERIAL = {"kind": "constant", "n": 2.1, "k": 0.0, "name": "high_index"}
DEFAULT_OPT_LOW_MATERIAL = {"kind": "constant", "n": 1.45, "k": 0.0, "name": "low_index"}
DEFAULT_OPT_CAVITY_MATERIAL = {"kind": "constant", "n": 1.6, "k": 0.0, "name": "cavity"}
DEFAULT_OPT_CAVITY_THICKNESS_NM = 190.0


def default_opt_stack_config() -> dict:
    """Return a fresh, valid §9.1 grouped/cavity stack-config dict.

    Matches ``examples/optimize_resonance_target.py``: one input period of
    [high(72), low(103)] ×M=3, a cavity layer (1.6, 190 nm) enabled, one output
    period of [low(103), high(72)] ×K=3, air incident/substrate, default to the
    cavity-only variable selection.
    """

    return {
        "incident": dict(DEFAULT_OPT_INCIDENT_MATERIAL),
        "input_group": {
            "layers": [
                {"material": dict(DEFAULT_OPT_HIGH_MATERIAL), "thickness_nm": 72.0},
                {"material": dict(DEFAULT_OPT_LOW_MATERIAL), "thickness_nm": 103.0},
            ],
            "repeat": DEFAULT_OPT_REPEAT_M,
        },
        "cavity": {
            "material": dict(DEFAULT_OPT_CAVITY_MATERIAL),
            "thickness_nm": DEFAULT_OPT_CAVITY_THICKNESS_NM,
            "enabled": True,
        },
        "output_group": {
            "layers": [
                {"material": dict(DEFAULT_OPT_LOW_MATERIAL), "thickness_nm": 103.0},
                {"material": dict(DEFAULT_OPT_HIGH_MATERIAL), "thickness_nm": 72.0},
            ],
            "repeat": DEFAULT_OPT_REPEAT_K,
        },
        "substrate": dict(DEFAULT_OPT_SUBSTRATE_MATERIAL),
        "grid": dict(DEFAULT_OPT_GRID),
        "angle_deg": DEFAULT_ANGLE_DEG,
        "polarization": DEFAULT_POLARIZATION,
        "variable": {
            "mode": "tied",       # §11.1 "tied" | "independent" (default "tied")
            "cavity": True,       # default selection = cavity only
            "input_layers": [],   # indices i into input_group.layers (period def)
            "output_layers": [],  # indices j into output_group.layers
            "flat_layers": [],    # §11.1 selected expanded flat indices (independent mode)
        },
    }


# ---------------------------------------------------------------------------
# Defaults for the workflow-2 optimize-config dict (ARCHITECTURE §2.3)
# ---------------------------------------------------------------------------
DEFAULT_OPTIMIZE_STEPS = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_LOWER_BOUND_NM = 0.0
DEFAULT_WAVELENGTH_WEIGHT = 1.0
DEFAULT_Q_WEIGHT = 1.0
DEFAULT_SHARPNESS = 20.0
DEFAULT_TARGET_WAVELENGTH_NM = 600.0
DEFAULT_TARGET_Q = 50.0


def default_optimize_config() -> dict:
    """Return a fresh workflow-2 optimize-config dict (ARCHITECTURE §2.3)."""

    return {
        "mode": "resonance",
        "spectrum": "R",
        "feature": "peak",
        "target_wavelength_nm": DEFAULT_TARGET_WAVELENGTH_NM,
        "target_q": DEFAULT_TARGET_Q,
        "variable_layer_indices": None,  # None => all finite layers
        "steps": DEFAULT_OPTIMIZE_STEPS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "lower_bound_nm": DEFAULT_LOWER_BOUND_NM,
        "wavelength_weight": DEFAULT_WAVELENGTH_WEIGHT,
        "q_weight": DEFAULT_Q_WEIGHT,
        "sharpness": DEFAULT_SHARPNESS,
    }


# ---------------------------------------------------------------------------
# i18n translation catalog (ARCHITECTURE §12, D1)
#
# DEFAULT_LANG is "en" — English is the canonical baseline.
# SUPPORTED_LANGS defines the set of valid language codes.
# TRANSLATIONS["en"] and TRANSLATIONS["it"] must have IDENTICAL key sets.
# Invariant: set(TRANSLATIONS["en"]) == set(TRANSLATIONS["it"])
# ---------------------------------------------------------------------------
DEFAULT_LANG: str = "en"
SUPPORTED_LANGS: tuple[str, ...] = ("en", "it")

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── English (canonical baseline) ─────────────────────────────────────────
    "en": {
        # ---- general / app / tabs ----
        "app_title": "Multilayer TMM Simulator",
        "tab_simulate": "Simulation",
        "tab_optimize": "Optimization",
        "lang_label": "Language",
        "lang_en": "EN",
        "lang_it": "IT",

        # ---- stack builder (shared) ----
        "incident": "Incident medium",
        "substrate": "Substrate",
        "layers": "Finite layers",
        "add_layer": "Add layer",
        "material_kind": "Material type",
        "refractive_index_n": "Index n",
        "extinction_k": "Extinction k",
        "material_name": "Material name",
        "upload_csv": "Upload CSV",
        "upload_csv_hint": "Drag here or select a CSV file (wavelength_nm, n, k)",
        "thickness_nm": "Thickness (nm)",
        "grid_start": "Start wavelength (nm)",
        "grid_stop": "Stop wavelength (nm)",
        "grid_num": "Number of points",
        "angle_deg": "Angle (deg)",
        "polarization": "Polarization",
        "grid_section_legend": "Wavelength grid, angle & polarization",

        # ---- option-list display text (§12.6) ----
        "pol_s": "s (TE)",
        "pol_p": "p (TM)",
        "pol_both": "Both (s and p)",
        "feat_peak": "Peak (maximum)",
        "feat_dip": "Dip (minimum)",
        "ch_R": "Reflectance (R)",
        "ch_T": "Transmittance (T)",
        "ch_A": "Absorptance (A)",
        "matkind_constant": "Constant index (n, k)",
        "matkind_csv": "CSV file (tabulated n, k)",
        "optmode_resonance": "Target resonance (λ + Q)",
        "optmode_mean_r": "Minimize mean R",

        # ---- simulate panel ----
        "simulate_run": "Simulate",
        "simulate_channels": "Channels to plot",
        "simulate_status_ready": "Ready.",
        "simulate_status_running": "Simulation running...",
        "simulate_status_done": "Simulation complete.",

        # ---- optimize panel (chrome) ----
        "optimize_mode": "Mode",
        "optimize_target_wavelength": "Target wavelength (nm)",
        "optimize_target_q": "Target Q",
        "optimize_feature": "Feature",
        "optimize_spectrum": "Spectral channel",
        "optimize_variable_layers": "Variable layer indices (e.g. 0,2)",
        "optimize_steps": "Iterations",
        "optimize_learning_rate": "Learning rate",
        "optimize_lower_bound": "Minimum thickness (nm)",
        "optimize_wavelength_weight": "Wavelength weight",
        "optimize_q_weight": "Q weight",
        "optimize_sharpness": "Sharpness",
        "optimize_run": "Optimize",
        "optimize_status_ready": "Ready.",
        "optimize_status_running": "Optimization running...",
        "optimize_status_done": "Optimization complete.",
        "optimize_history": "Loss history",
        "optimize_thicknesses": "Optimized thicknesses (nm)",
        # grouped/cavity stack editor (§9.4)
        "opt_stack_title": "Cavity structure",
        "opt_input_group": "Input mirror (period)",
        "opt_output_group": "Output mirror (period)",
        "opt_input_repeat": "Repetitions M (input)",
        "opt_output_repeat": "Repetitions K (output)",
        "opt_cavity": "Cavity",
        "opt_cavity_enabled": "Cavity active",
        "opt_cavity_thickness": "Cavity thickness (nm)",
        "opt_variable_section": "Variable (optimizable) thicknesses",
        "opt_variable_cavity": "Cavity",
        "opt_variable_input_layers": "Variable input-period layers",
        "opt_variable_output_layers": "Variable output-period layers",
        # §11.3 two-mode variable selector
        "opt_variable_mode_tied": "Per period",
        "opt_variable_mode_independent": "Per individual layer",
        "opt_variable_flat_layers": "Optimizable layers (expanded stack)",
        "sketch_title": "Multilayer schematic",

        # ---- results / resonance readout ----
        "results": "Results",
        "resonance": "Resonance analysis",
        "export": "Export",
        "optimize_export": "Export optimized spectra",
        "optimize_export_empty": "Run an optimization before exporting.",
        "simulate_export_empty": "Run a simulation before exporting.",
        "export_done": "Saved {prefix}.zip — contains {prefix}_spectra.txt and {prefix}_parameters.txt.",
        "resonance_wavelength": "Resonance wavelength (nm)",
        "linewidth": "Linewidth (nm)",
        "quality_factor": "Quality factor (Q)",
        "extremum_value": "Extremum value",
        "empty_plot": "Run a simulation to display the spectrum.",
        "res_table_metric": "Metric",
        "res_table_value": "Value",
        "res_table_warning": "Warning",

        # ---- tooltips (tip_*) — optimization controls only (§12.2) ----
        "tip_mode": (
            "Choose 'Target resonance' to tune a spectral peak/dip to a desired "
            "wavelength and Q-factor, or 'Minimize mean R' to reduce average "
            "reflectance across the whole wavelength grid."
        ),
        "tip_spectrum": (
            "Spectral channel (R, T, or A) on which the feature detection and "
            "loss function operate."
        ),
        "tip_feature": (
            "Select 'Peak' to target a reflectance maximum (e.g. high-reflector) "
            "or 'Dip' to target a transmission minimum/absorption maximum."
        ),
        "tip_target_wavelength": (
            "Desired centre wavelength (nm) of the resonant feature; the optimizer "
            "minimizes the distance between this value and the detected spectral "
            "peak/dip position."
        ),
        "tip_target_q": (
            "Desired quality factor Q = λ_res / FWHM; higher Q means a sharper, "
            "narrower resonance. The optimizer balances this with the wavelength "
            "weight."
        ),
        "tip_wavelength_weight": (
            "Relative weight of the wavelength-mismatch term in the loss function; "
            "increase to prioritize hitting the target wavelength over achieving "
            "the target Q."
        ),
        "tip_q_weight": (
            "Relative weight of the Q-factor mismatch term in the loss function; "
            "increase to prioritize sharpness over wavelength accuracy."
        ),
        "tip_sharpness": (
            "Softmax sharpness parameter controlling how the differentiable spectral "
            "moments approximate the discrete peak/dip; higher values produce a "
            "sharper (but less smooth) approximation."
        ),
        "tip_steps": (
            "Number of gradient-descent iterations; more steps may improve "
            "convergence but increase runtime."
        ),
        "tip_learning_rate": (
            "Gradient-descent step size; too large causes oscillation, too small "
            "causes slow convergence — typical values: 0.01–0.5."
        ),
        "tip_lower_bound": (
            "Hard lower bound (nm) on every optimizable thickness; prevents "
            "layers from collapsing to zero during optimization."
        ),
        "tip_variable_mode_tied": (
            "Tied mode: each selected period layer is a single shared variable "
            "broadcast to ALL its repetitions, keeping the period uniform."
        ),
        "tip_variable_mode_independent": (
            "Independent mode: each selected expanded-stack layer is its own "
            "free variable, allowing the periods to become non-uniform."
        ),
    },

    # ── Italian ───────────────────────────────────────────────────────────────
    "it": {
        # ---- general / app / tabs ----
        "app_title": "Simulatore TMM multistrato",
        "tab_simulate": "Simulazione",
        "tab_optimize": "Ottimizzazione",
        "lang_label": "Lingua",
        "lang_en": "EN",
        "lang_it": "IT",

        # ---- stack builder (shared) ----
        "incident": "Mezzo incidente",
        "substrate": "Substrato",
        "layers": "Strati finiti",
        "add_layer": "Aggiungi strato",
        "material_kind": "Tipo di materiale",
        "refractive_index_n": "Indice n",
        "extinction_k": "Coefficiente k",
        "material_name": "Nome materiale",
        "upload_csv": "Carica CSV",
        "upload_csv_hint": "Trascina qui o seleziona un file CSV (wavelength_nm, n, k)",
        "thickness_nm": "Spessore (nm)",
        "grid_start": "Lambda iniziale (nm)",
        "grid_stop": "Lambda finale (nm)",
        "grid_num": "Numero di punti",
        "angle_deg": "Angolo (gradi)",
        "polarization": "Polarizzazione",
        "grid_section_legend": "Griglia lambda, angolo e polarizzazione",

        # ---- option-list display text (§12.6) ----
        "pol_s": "s (TE)",
        "pol_p": "p (TM)",
        "pol_both": "Entrambe (s e p)",
        "feat_peak": "Picco (massimo)",
        "feat_dip": "Avvallamento (minimo)",
        "ch_R": "Riflettanza (R)",
        "ch_T": "Trasmittanza (T)",
        "ch_A": "Assorbanza (A)",
        "matkind_constant": "Indice costante (n, k)",
        "matkind_csv": "File CSV (n, k tabulati)",
        "optmode_resonance": "Risonanza mirata (lambda + Q)",
        "optmode_mean_r": "Minimizza R media",

        # ---- simulate panel ----
        "simulate_run": "Simula",
        "simulate_channels": "Canali da tracciare",
        "simulate_status_ready": "Pronto.",
        "simulate_status_running": "Simulazione in corso...",
        "simulate_status_done": "Simulazione completata.",

        # ---- optimize panel (chrome) ----
        "optimize_mode": "Modalità",
        "optimize_target_wavelength": "Lambda obiettivo (nm)",
        "optimize_target_q": "Q obiettivo",
        "optimize_feature": "Caratteristica",
        "optimize_spectrum": "Canale spettrale",
        "optimize_variable_layers": "Indici strati variabili (es. 0,2)",
        "optimize_steps": "Iterazioni",
        "optimize_learning_rate": "Tasso di apprendimento",
        "optimize_lower_bound": "Spessore minimo (nm)",
        "optimize_wavelength_weight": "Peso lambda",
        "optimize_q_weight": "Peso Q",
        "optimize_sharpness": "Nitidezza (sharpness)",
        "optimize_run": "Ottimizza",
        "optimize_status_ready": "Pronto.",
        "optimize_status_running": "Ottimizzazione in corso...",
        "optimize_status_done": "Ottimizzazione completata.",
        "optimize_history": "Storico della perdita",
        "optimize_thicknesses": "Spessori ottimizzati (nm)",
        # grouped/cavity stack editor (§9.4)
        "opt_stack_title": "Struttura della cavità",
        "opt_input_group": "Specchio di ingresso (periodo)",
        "opt_output_group": "Specchio di uscita (periodo)",
        "opt_input_repeat": "Ripetizioni M (ingresso)",
        "opt_output_repeat": "Ripetizioni K (uscita)",
        "opt_cavity": "Cavità",
        "opt_cavity_enabled": "Cavità attiva",
        "opt_cavity_thickness": "Spessore cavità (nm)",
        "opt_variable_section": "Spessori variabili (ottimizzabili)",
        "opt_variable_cavity": "Cavità",
        "opt_variable_input_layers": "Strati periodo ingresso variabili",
        "opt_variable_output_layers": "Strati periodo uscita variabili",
        # §11.3 two-mode variable selector
        "opt_variable_mode_tied": "Per periodo",
        "opt_variable_mode_independent": "Per singolo strato",
        "opt_variable_flat_layers": "Strati ottimizzabili (stack espanso)",
        "sketch_title": "Schema del multistrato",

        # ---- results / resonance readout ----
        "results": "Risultati",
        "resonance": "Analisi di risonanza",
        "export": "Esporta",
        "optimize_export": "Esporta spettri ottimizzati",
        "optimize_export_empty": "Esegui un'ottimizzazione prima di esportare.",
        "simulate_export_empty": "Esegui una simulazione prima di esportare.",
        "export_done": "Salvato {prefix}.zip — contiene {prefix}_spectra.txt e {prefix}_parameters.txt.",
        "resonance_wavelength": "Lambda di risonanza (nm)",
        "linewidth": "Larghezza di riga (nm)",
        "quality_factor": "Fattore di qualità (Q)",
        "extremum_value": "Valore estremo",
        "empty_plot": "Esegui una simulazione per visualizzare lo spettro.",
        "res_table_metric": "Grandezza",
        "res_table_value": "Valore",
        "res_table_warning": "Avviso",

        # ---- tooltips (tip_*) — optimization controls only ----
        "tip_mode": (
            "Scegli 'Risonanza mirata' per centrare un picco/avvallamento "
            "spettrale su una lunghezza d'onda e un fattore Q desiderati, "
            "oppure 'Minimizza R media' per ridurre la riflettanza media "
            "sull'intera griglia."
        ),
        "tip_spectrum": (
            "Canale spettrale (R, T o A) su cui operano il rilevamento della "
            "caratteristica e la funzione di perdita."
        ),
        "tip_feature": (
            "Seleziona 'Picco' per puntare a un massimo di riflettanza (es. "
            "specchio ad alta riflessione) o 'Avvallamento' per un minimo di "
            "trasmissione o massimo di assorbimento."
        ),
        "tip_target_wavelength": (
            "Lunghezza d'onda centrale desiderata (nm) della caratteristica "
            "risonante; l'ottimizzatore minimizza la distanza tra questo valore "
            "e la posizione del picco/avvallamento rilevato."
        ),
        "tip_target_q": (
            "Fattore di qualità desiderato Q = λ_ris / FWHM; un Q più alto "
            "corrisponde a una risonanza più stretta e nitida."
        ),
        "tip_wavelength_weight": (
            "Peso relativo del termine di disaccordo in lunghezza d'onda nella "
            "funzione di perdita; aumentarlo privilegia il raggiungimento della "
            "lambda obiettivo rispetto al Q."
        ),
        "tip_q_weight": (
            "Peso relativo del termine di disaccordo del fattore Q nella "
            "funzione di perdita; aumentarlo privilegia la nitidezza rispetto "
            "alla precisione in lunghezza d'onda."
        ),
        "tip_sharpness": (
            "Parametro di nitidezza del softmax che controlla come i momenti "
            "spettrali differenziabili approssimano il picco/avvallamento "
            "discreto; valori più alti producono un'approssimazione più netta."
        ),
        "tip_steps": (
            "Numero di iterazioni di discesa del gradiente; più passi possono "
            "migliorare la convergenza ma aumentano il tempo di calcolo."
        ),
        "tip_learning_rate": (
            "Dimensione del passo della discesa del gradiente; troppo grande "
            "causa oscillazioni, troppo piccolo causa convergenza lenta — "
            "valori tipici: 0,01–0,5."
        ),
        "tip_lower_bound": (
            "Limite inferiore rigido (nm) su ogni spessore ottimizzabile; "
            "impedisce che gli strati collassino a zero durante l'ottimizzazione."
        ),
        "tip_variable_mode_tied": (
            "Modalità vincolata: ogni strato di periodo selezionato è "
            "un'unica variabile condivisa trasmessa a TUTTE le sue ripetizioni, "
            "mantenendo il periodo uniforme."
        ),
        "tip_variable_mode_independent": (
            "Modalità indipendente: ogni strato selezionato nello stack espanso "
            "è una variabile libera propria, consentendo ai periodi di diventare "
            "non uniformi."
        ),
    },
}

# ---------------------------------------------------------------------------
# Accessor functions (ARCHITECTURE §12.1, D1)
# ---------------------------------------------------------------------------

def labels_for(lang: str = DEFAULT_LANG) -> dict:
    """Return the full label dict for a language, EN-overlaid so a missing key can never KeyError.

    Unknown lang -> EN. Returns a NEW merged dict (EN under the requested language).
    """
    base = TRANSLATIONS[DEFAULT_LANG]
    if lang == DEFAULT_LANG or lang not in TRANSLATIONS:
        return dict(base)
    return {**base, **TRANSLATIONS[lang]}


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """Single-key accessor: t('app_title', 'it'). EN fallback then key-as-text for a missing key."""
    return labels_for(lang).get(key, TRANSLATIONS[DEFAULT_LANG].get(key, key))


def options_for(values: tuple[str, ...], key_prefix: str, lang: str = DEFAULT_LANG) -> list[dict]:
    """Build a Dash options list from value tuple + catalog key prefix.

    Components call e.g. ``config.options_for(config.POLARIZATION_VALUES, "pol_", lang)``.
    Each value ``v`` maps to ``{"label": labels[f"{key_prefix}{v}"], "value": v}``.
    """
    labels = labels_for(lang)
    return [{"label": labels[f"{key_prefix}{v}"], "value": v} for v in values]


# ---------------------------------------------------------------------------
# Legacy alias — retained ONE revision for zero-flag-day compatibility.
# REMOVAL IS THE DELIVERABLE: any remaining config.LABELS[k] usage should
# be migrated to config.labels_for(lang)[k] or config.t(key, lang).
# ---------------------------------------------------------------------------
LABELS: dict[str, str] = labels_for("en")
