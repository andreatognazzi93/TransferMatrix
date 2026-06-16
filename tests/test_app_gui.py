"""QA verification suite for the Dash GUI in ``app/``.

These tests are the final gate for the GUI. They verify:

* ``app/state.py`` pure functions (dict <-> Stack/Material, grid, validation,
  CSV round-trip, "both"-polarization rejection in analyze/optimize paths).
* ``app/plots.py`` figure builders (trace counts for one polarization and for
  "both", serialization via ``.to_dict()``).
* A cross-check that the GUI simulation path produces R/T/A identical to a
  direct ``multilayer_tmm.simulate_spectrum`` call on an equivalent stack.
* A smoke test that ``app.main.create_app()`` constructs and registers callbacks.
* i18n invariants: key-set parity, EN-default, IT switch, unknown-lang fallback,
  serve_layout mechanism, language selector presence, help tooltips, CSS asset.

They never edit ``multilayer_tmm/`` or ``app/`` feature code.
"""

from __future__ import annotations

import base64
import os

import numpy as np
import plotly.graph_objects as go
import pytest

from multilayer_tmm import (
    Layer,
    Material,
    Stack,
    simulate_spectrum,
    wavelength_grid,
)

from app import config, ids, plots, state


# ===========================================================================
# Fixtures / helpers
# ===========================================================================
def _constant(n, k=0.0, name=None):
    return {"kind": "constant", "n": n, "k": k, "name": name}


def _base_config(polarization="s", num=51):
    """A small, well-formed §2.2 stack-config dict (air / TiO2 / SiO2 / glass)."""

    return {
        "incident": _constant(1.0, name="air"),
        "layers": [
            {"material": _constant(2.35, name="TiO2"), "thickness_nm": 120.0},
            {"material": _constant(1.46, name="SiO2"), "thickness_nm": 90.0},
        ],
        "substrate": _constant(1.52, name="glass"),
        "grid": {"start_nm": 400.0, "stop_nm": 800.0, "num": num},
        "angle_deg": 0.0,
        "polarization": polarization,
    }


def _equivalent_stack():
    return Stack(
        incident=Material.constant(1.0),
        layers=[
            Layer(Material.constant(2.35), 120.0),
            Layer(Material.constant(1.46), 90.0),
        ],
        substrate=Material.constant(1.52),
    )


# ===========================================================================
# state.py — material <-> dict
# ===========================================================================
def test_app_material_from_dict_constant_complex_index():
    m = state.material_from_dict(_constant(1.46, 0.1, name="abs"))
    assert m.kind == "constant"
    value = complex(np.asarray(m.data).item())
    assert value.real == pytest.approx(1.46)
    assert value.imag == pytest.approx(0.1)


def test_app_material_from_dict_rejects_callable_kind():
    with pytest.raises(ValueError):
        state.material_from_dict({"kind": "callable", "name": "evil"})


def test_app_material_constant_round_trip():
    src = _constant(2.0, 0.25, name="X")
    m = state.material_from_dict(src)
    out = state.material_to_dict(m)
    assert out["kind"] == "constant"
    assert out["n"] == pytest.approx(2.0)
    assert out["k"] == pytest.approx(0.25)
    assert out["name"] == "X"


def test_app_material_to_dict_rejects_callable_material():
    # Material.from_callable is the only path to a callable material; if the
    # library exposes it, a callable material must not serialize. If it does
    # not, the constant/csv-only contract already holds.
    if not hasattr(Material, "from_callable"):
        pytest.skip("library has no from_callable; callable path not constructible")
    m = Material.from_callable(lambda wl: 1.5 + 0j, name="cb")
    with pytest.raises(ValueError):
        state.material_to_dict(m)


# ===========================================================================
# state.py — CSV parse + tabulated round-trip
# ===========================================================================
def _csv_data_url(rows):
    header = "wavelength_nm,n,k\n"
    body = "\n".join(f"{w},{n},{k}" for w, n, k in rows)
    raw = (header + body).encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")


def test_app_parse_material_csv_round_trip():
    rows = [(400.0, 1.50, 0.00), (600.0, 1.48, 0.01), (800.0, 1.46, 0.02)]
    url = _csv_data_url(rows)
    d = state.parse_material_csv(url, filename="glass.csv", name="Glass")

    assert d["kind"] == "csv"
    assert d["name"] == "Glass"
    assert d["wavelength_nm"] == [400.0, 600.0, 800.0]
    assert d["n"] == [1.50, 1.48, 1.46]
    assert d["k"] == [0.00, 0.01, 0.02]

    # dict -> Material -> dict round trips losslessly.
    m = state.material_from_dict(d)
    assert m.kind == "tabulated"
    back = state.material_to_dict(m)
    assert back["kind"] == "csv"
    np.testing.assert_allclose(back["wavelength_nm"], rows_col(rows, 0))
    np.testing.assert_allclose(back["n"], rows_col(rows, 1))
    np.testing.assert_allclose(back["k"], rows_col(rows, 2))


def rows_col(rows, i):
    return [r[i] for r in rows]


def test_app_parse_material_csv_missing_column_raises():
    raw = b"wavelength_nm,n\n400,1.5\n"
    url = "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError):
        state.parse_material_csv(url, filename="bad.csv")


def test_app_parse_material_csv_non_numeric_raises():
    raw = b"wavelength_nm,n,k\n400,foo,0.0\n"
    url = "data:text/csv;base64," + base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError):
        state.parse_material_csv(url, filename="bad.csv")


def test_app_parse_material_csv_empty_raises():
    with pytest.raises(ValueError):
        state.parse_material_csv("", filename="empty.csv")


# ===========================================================================
# state.py — stack_from_config / grid_from_config
# ===========================================================================
def test_app_stack_from_config_structure():
    stack = state.stack_from_config(_base_config())
    assert isinstance(stack, Stack)
    assert stack.num_layers == 2
    # incident + 2 layers + substrate = 4 media.
    assert len(stack.materials) == 4


def test_app_grid_from_config_matches_library():
    cfg = _base_config(num=11)
    grid = np.asarray(state.grid_from_config(cfg))
    expected = np.asarray(wavelength_grid(400.0, 800.0, 11))
    assert grid.shape == (11,)
    np.testing.assert_allclose(grid, expected)


# ===========================================================================
# state.py — validate_config  (EN-default + IT twins)
# ===========================================================================
def test_app_validate_config_ok():
    assert state.validate_config(_base_config()) == []


def test_app_validate_config_bad_grid_num():
    """Default (EN) error contains English text."""
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    errors = state.validate_config(cfg)
    assert any("2 points" in e for e in errors)


def test_app_validate_config_bad_grid_num_it():
    """Italian error contains Italian text."""
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    errors = state.validate_config(cfg, lang="it")
    assert any("2 punti" in e for e in errors)


def test_app_validate_config_start_ge_stop():
    """Default (EN) error for start >= stop."""
    cfg = _base_config()
    cfg["grid"]["start_nm"] = 900.0  # >= stop
    errors = state.validate_config(cfg)
    assert any("less than" in e or "Start wavelength" in e for e in errors)


def test_app_validate_config_start_ge_stop_it():
    """Italian error for start >= stop."""
    cfg = _base_config()
    cfg["grid"]["start_nm"] = 900.0
    errors = state.validate_config(cfg, lang="it")
    assert any("iniziale" in e for e in errors)


def test_app_validate_config_negative_thickness():
    """Default (EN) error for negative thickness."""
    cfg = _base_config()
    cfg["layers"][0]["thickness_nm"] = -5.0
    errors = state.validate_config(cfg)
    assert any("negative" in e for e in errors)


def test_app_validate_config_negative_thickness_it():
    """Italian error for negative thickness."""
    cfg = _base_config()
    cfg["layers"][0]["thickness_nm"] = -5.0
    errors = state.validate_config(cfg, lang="it")
    assert any("negativo" in e for e in errors)


def test_app_validate_config_bad_polarization():
    """Default (EN) error for invalid polarization."""
    cfg = _base_config()
    cfg["polarization"] = "circular"
    errors = state.validate_config(cfg)
    assert any("polarization" in e.lower() for e in errors)


def test_app_validate_config_bad_polarization_it():
    """Italian error for invalid polarization."""
    cfg = _base_config()
    cfg["polarization"] = "circular"
    errors = state.validate_config(cfg, lang="it")
    assert any("Polarizzazione" in e for e in errors)


# ===========================================================================
# state.py — "both"-polarization rejection in analyze + optimize paths
# ===========================================================================
def test_app_analyze_result_rejects_both():
    cfg = _base_config(polarization="both")
    result = state.run_simulation(cfg)
    assert result["polarizations"] == ["s", "p"]
    with pytest.raises(ValueError):
        state.analyze_result(result, channel="R")


def test_app_make_thickness_objective_rejects_both():
    with pytest.raises(ValueError):
        state.make_thickness_objective(_base_config(polarization="both"))


def test_app_run_resonance_optimization_rejects_both():
    cfg = _base_config(polarization="both")
    opt_config = {"target_wavelength_nm": 600.0, "target_q": 50.0, "steps": 2}
    with pytest.raises(ValueError):
        state.run_resonance_optimization(cfg, opt_config)


# ===========================================================================
# state.py — run_simulation shapes
# ===========================================================================
def test_app_run_simulation_single_pol_is_1d():
    result = state.run_simulation(_base_config(polarization="s", num=21))
    assert result["polarizations"] == ["s"]
    assert np.asarray(result["R"]).shape == (21,)
    assert np.asarray(result["T"]).shape == (21,)
    assert np.asarray(result["A"]).shape == (21,)


def test_app_run_simulation_both_is_2byN():
    result = state.run_simulation(_base_config(polarization="both", num=21))
    assert result["polarizations"] == ["s", "p"]
    assert np.asarray(result["R"]).shape == (2, 21)
    assert np.asarray(result["A"]).shape == (2, 21)


def test_app_run_simulation_invalid_config_raises():
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    with pytest.raises(ValueError):
        state.run_simulation(cfg)


# ===========================================================================
# plots.py — builders return Figures with expected trace counts
# ===========================================================================
def test_app_spectrum_figure_single_pol_trace_count():
    result = state.run_simulation(_base_config(polarization="s", num=31))
    fig = plots.spectrum_figure(result, channels=("R", "T", "A"))
    assert isinstance(fig, go.Figure)
    # one trace per channel for a single polarization.
    assert len(fig.data) == 3
    assert isinstance(fig.to_dict(), dict)


def test_app_spectrum_figure_both_pol_trace_count():
    result = state.run_simulation(_base_config(polarization="both", num=31))
    fig = plots.spectrum_figure(result, channels=("R", "T", "A"))
    assert isinstance(fig, go.Figure)
    # one trace per (channel, polarization) -> 3 channels x 2 pols = 6.
    assert len(fig.data) == 6
    assert isinstance(fig.to_dict(), dict)


def test_app_spectrum_figure_subset_channels():
    result = state.run_simulation(_base_config(polarization="s", num=31))
    fig = plots.spectrum_figure(result, channels=("R",))
    assert len(fig.data) == 1


def test_app_history_figure_is_figure():
    fig = plots.history_figure([1.0, 0.5, 0.25, 0.1])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.to_dict(), dict)


def test_app_resonance_overlay_figure_traces():
    result = state.run_simulation(_base_config(polarization="s", num=201))
    resonance = state.analyze_result(result, channel="R", feature="peak")
    fig = plots.resonance_overlay_figure(result, resonance, channel="R")
    assert isinstance(fig, go.Figure)
    # spectrum trace + resonance marker + half-level crossings = 3 scatter traces.
    assert len(fig.data) == 3
    assert isinstance(fig.to_dict(), dict)


def test_app_resonance_overlay_summary_in_legend():
    """Optimized-structure overlay carries λ_res and Q in the legend label."""
    result = state.run_simulation(_base_config(polarization="s", num=201))
    resonance = state.analyze_result(result, channel="R", feature="peak")
    fig = plots.resonance_overlay_figure(
        result, resonance, channel="R", summary_in_legend=True
    )
    res_names = [t.name for t in fig.data if t.name and "λ" in t.name]
    assert res_names, "expected a resonance marker trace"
    name = res_names[0]
    assert "<br>" in name, "legend name must use <br> to split onto two lines"
    assert "λ" in name and "Q" in name
    # Default (analysis) path keeps the plain EN label "Resonance".
    plain = plots.resonance_overlay_figure(result, resonance, channel="R")
    plain_names = [t.name for t in plain.data if t.name == "Resonance"]
    assert plain_names == ["Resonance"]


def test_app_resonance_overlay_default_name_is_resonance_it():
    """Italian: default resonance trace name is 'Risonanza' with lang='it'."""
    result = state.run_simulation(_base_config(polarization="s", num=201))
    resonance = state.analyze_result(result, channel="R", feature="peak")
    fig = plots.resonance_overlay_figure(result, resonance, channel="R", lang="it")
    plain_names = [t.name for t in fig.data if t.name == "Risonanza"]
    assert plain_names == ["Risonanza"]


def test_app_empty_figure_is_figure():
    fig = plots.empty_figure("Nessun dato")
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.to_dict(), dict)


# ===========================================================================
# plots.py — sketch (color encodes n AND k; grouped bracket spans the period)
# ===========================================================================
def _const(n, k=0.0, name=None):
    d = {"kind": "constant", "n": n, "k": k}
    if name:
        d["name"] = name
    return d


def test_app_sketch_same_n_different_k_distinct_colors():
    """Two films with equal n but different k must get different fill colors."""
    cfg = {
        "incident": _const(1.0, name="air"),
        "substrate": _const(1.0, name="air"),
        "layers": [
            {"thickness_nm": 100.0, "material": _const(2.0, 0.0, "lossless")},
            {"thickness_nm": 100.0, "material": _const(2.0, 0.5, "lossy")},
        ],
    }
    fig = plots.sketch_figure(cfg, grouped=False)
    rect_colors = [s.fillcolor for s in fig.layout.shapes if s.type == "rect"]
    assert len(rect_colors) == 2
    assert rect_colors[0] != rect_colors[1]


def test_app_sketch_lossless_color_unchanged_by_k_machinery():
    """A fully lossless stack (kmax == 0) keeps the pure Re(n) color."""
    assert plots._material_fill_color(
        _const(2.0, 0.0), 1.0, 2.0, 0.0
    ) == plots._re_n_to_color(2.0, 1.0, 2.0)


def test_app_text_on_picks_contrasting_color():
    """Dark fills get white text; light fills get near-black text (#2)."""
    assert plots._text_on((68, 1, 84)) == "#ffffff"     # deep Viridis purple
    assert plots._text_on((37, 133, 142)) == "#ffffff"  # mid Viridis teal
    assert plots._text_on((253, 231, 37)) == "#111111"  # bright Viridis yellow


def test_app_sketch_arrow_sits_above_structure():
    """The incidence arrow lives above y=0, not overlapping the medium (#1)."""
    cfg = {
        "incident": _const(1.0, name="air"),
        "substrate": _const(1.5, name="glass"),
        "layers": [{"thickness_nm": 120.0, "material": _const(2.0, name="film")}],
    }
    fig = plots.sketch_figure(cfg, angle_deg=0.0, grouped=False)
    arrows = [a for a in fig.layout.annotations if a.showarrow]
    assert len(arrows) == 1
    arrow = arrows[0]
    assert arrow.y == 0.0          # tip touches the top of the structure
    assert arrow.ay < 0.0          # tail is above the structure (negative depth)
    # y-axis top extends into the negative (above-structure) zone for the arrow.
    assert fig.layout.yaxis.range[1] < 0.0


def test_app_sketch_grouped_bracket_spans_whole_period():
    """The ×M bracket must span every layer of the repeated period, not one.

    Mirrors examples/optimize_resonance_target.py: the period is a LAYER PAIR
    (e.g. [high 72nm, low 103nm]) and the whole pair repeats.
    """
    period_in = [
        {"thickness_nm": 72.0, "material": _const(2.1, name="high")},
        {"thickness_nm": 103.0, "material": _const(1.45, name="low")},
    ]
    period_out = [
        {"thickness_nm": 103.0, "material": _const(1.45, name="low")},
        {"thickness_nm": 72.0, "material": _const(2.1, name="high")},
    ]
    cfg = {
        "incident": _const(1.0, name="air"),
        "substrate": _const(1.0, name="air"),
        "input_group": {"repeat": 20, "layers": period_in},
        "cavity": {
            "enabled": True,
            "thickness_nm": 190.0,
            "material": _const(1.6, name="cav"),
        },
        "output_group": {"repeat": 3, "layers": period_out},
    }
    fig = plots.sketch_figure(cfg, grouped=True)

    brackets = [
        s
        for s in fig.layout.shapes
        if s.type == "line" and abs(s.x0 - s.x1) < 1e-9 and abs(s.x0 - 1.04) < 1e-9
    ]
    # one vertical bracket per repeated group (input + output).
    assert len(brackets) == 2
    # each spans the full 175 nm period (72 + 103), not a single 72/103 layer.
    for b in brackets:
        assert abs(abs(b.y1 - b.y0) - 175.0) < 1e-6

    labels = {a.text for a in fig.layout.annotations if a.text.startswith("×")}
    assert labels == {"×20", "×3"}


# ===========================================================================
# CROSS-CHECK: GUI path R/T/A == direct simulate_spectrum (tight tolerance)
# ===========================================================================
@pytest.mark.parametrize("polarization", ["s", "p"])
def test_app_cross_check_gui_matches_direct(polarization):
    cfg = _base_config(polarization=polarization, num=101)
    gui = state.run_simulation(cfg)

    stack = _equivalent_stack()
    wl = wavelength_grid(400.0, 800.0, 101)
    direct = simulate_spectrum(
        stack, wavelengths_nm=wl, angle_deg=0.0, polarization=polarization
    )

    for ch in ("R", "T", "A"):
        gui_arr = np.asarray(gui[ch], dtype=float)
        direct_arr = np.asarray(getattr(direct, ch), dtype=float)
        assert gui_arr.shape == direct_arr.shape
        np.testing.assert_allclose(gui_arr, direct_arr, rtol=1e-10, atol=1e-12)


def test_app_cross_check_both_matches_direct():
    cfg = _base_config(polarization="both", num=61)
    gui = state.run_simulation(cfg)

    stack = _equivalent_stack()
    wl = wavelength_grid(400.0, 800.0, 61)
    direct = simulate_spectrum(
        stack, wavelengths_nm=wl, angle_deg=0.0, polarization="both"
    )

    for ch in ("R", "T", "A"):
        gui_arr = np.asarray(gui[ch], dtype=float)
        direct_arr = np.asarray(getattr(direct, ch), dtype=float)
        assert gui_arr.shape == (2, 61)
        np.testing.assert_allclose(gui_arr, direct_arr, rtol=1e-10, atol=1e-12)


def test_app_cross_check_oblique_angle():
    cfg = _base_config(polarization="p", num=41)
    cfg["angle_deg"] = 35.0
    gui = state.run_simulation(cfg)

    stack = _equivalent_stack()
    wl = wavelength_grid(400.0, 800.0, 41)
    direct = simulate_spectrum(
        stack, wavelengths_nm=wl, angle_deg=35.0, polarization="p"
    )
    np.testing.assert_allclose(
        np.asarray(gui["R"], dtype=float),
        np.asarray(direct.R, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


# ===========================================================================
# Smoke: app.main.create_app() constructs and callbacks are registered
# ===========================================================================
def test_app_main_create_app_constructs_with_callbacks():
    import app.main as main

    app_obj = main.create_app()
    assert app_obj is not None
    # register_callbacks ran -> the callback map is populated.
    assert len(app_obj.callback_map) > 0
    # the server (WSGI) object is exposed for gunicorn.
    assert app_obj.server is not None


# ===========================================================================
# i18n — key-set parity invariants (§12.2, §12.3, §12.2b)
# ===========================================================================
def test_i18n_config_translations_key_parity():
    """set(TRANSLATIONS['en']) == set(TRANSLATIONS['it'])."""
    assert set(config.TRANSLATIONS["en"]) == set(config.TRANSLATIONS["it"]), (
        "TRANSLATIONS key sets differ between 'en' and 'it'; "
        f"extra EN={set(config.TRANSLATIONS['en'])-set(config.TRANSLATIONS['it'])}, "
        f"extra IT={set(config.TRANSLATIONS['it'])-set(config.TRANSLATIONS['en'])}"
    )


def test_i18n_plot_translations_key_parity():
    """set(_PLOT_TRANSLATIONS['en']) == set(_PLOT_TRANSLATIONS['it'])."""
    en_keys = set(plots._PLOT_TRANSLATIONS["en"])
    it_keys = set(plots._PLOT_TRANSLATIONS["it"])
    assert en_keys == it_keys, (
        f"_PLOT_TRANSLATIONS key sets differ; "
        f"extra EN={en_keys - it_keys}, extra IT={it_keys - en_keys}"
    )


def test_i18n_state_errors_key_parity():
    """set(_ERRORS['en']) == set(_ERRORS['it'])."""
    en_keys = set(state._ERRORS["en"])
    it_keys = set(state._ERRORS["it"])
    assert en_keys == it_keys, (
        f"_ERRORS key sets differ; "
        f"extra EN={en_keys - it_keys}, extra IT={it_keys - en_keys}"
    )


# ===========================================================================
# i18n — default is English
# ===========================================================================
def test_i18n_default_lang_is_en():
    assert config.DEFAULT_LANG == "en"


def test_i18n_labels_for_default_is_english():
    """labels_for() with no arg returns English strings."""
    labels = config.labels_for()
    assert labels["tab_optimize"] == "Optimization"


def test_i18n_spectrum_figure_default_xaxis_english():
    """spectrum_figure() without lang= uses English x-axis label."""
    cfg = _base_config(polarization="s", num=21)
    result = state.run_simulation(cfg)
    fig = plots.spectrum_figure(result)
    assert fig.layout.xaxis.title.text == "Wavelength (nm)"


def test_i18n_validate_config_default_english():
    """validate_config() without lang= returns English error messages."""
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    errors = state.validate_config(cfg)
    assert any("2 points" in e for e in errors)


# ===========================================================================
# i18n — language switch works
# ===========================================================================
def test_i18n_labels_for_it_differs_from_en():
    """labels_for('it') returns different strings for a sample key."""
    en_val = config.labels_for("en")["tab_optimize"]
    it_val = config.labels_for("it")["tab_optimize"]
    assert en_val != it_val


def test_i18n_sketch_figure_it_title():
    """sketch_figure(cfg, lang='it') has an Italian title."""
    cfg = {
        "incident": _const(1.0, name="air"),
        "substrate": _const(1.52, name="glass"),
        "layers": [{"thickness_nm": 120.0, "material": _const(2.35, name="TiO2")}],
    }
    fig = plots.sketch_figure(cfg, grouped=False, lang="it")
    assert fig.layout.title.text == "Schema del multistrato"


def test_i18n_validate_config_lang_it_returns_italian():
    """validate_config(bad, lang='it') returns Italian error messages."""
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    errors = state.validate_config(cfg, lang="it")
    assert any("2 punti" in e for e in errors)


def test_i18n_labels_for_unknown_lang_falls_back_to_en():
    """labels_for('xx') falls back to EN without error."""
    labels = config.labels_for("xx")
    assert labels["tab_optimize"] == "Optimization"


def test_i18n_spectrum_figure_unknown_lang_falls_back_to_en():
    """spectrum_figure with unknown lang falls back to EN x-axis label."""
    cfg = _base_config(polarization="s", num=21)
    result = state.run_simulation(cfg)
    fig = plots.spectrum_figure(result, lang="xx")
    assert fig.layout.xaxis.title.text == "Wavelength (nm)"


def test_i18n_validate_config_unknown_lang_falls_back_to_en():
    """validate_config with unknown lang falls back to EN errors."""
    cfg = _base_config()
    cfg["grid"]["num"] = 1
    errors = state.validate_config(cfg, lang="xx")
    assert any("2 points" in e for e in errors)


def test_i18n_t_missing_key_returns_key():
    """config.t('missing_key_xyz', 'it') returns the key itself."""
    result = config.t("missing_key_xyz", "it")
    assert result == "missing_key_xyz"


# ===========================================================================
# i18n — serve_layout mechanism (§12, D4)
# ===========================================================================
def test_i18n_app_layout_is_callable():
    """app.main.app.layout is a callable (per-request function)."""
    import app.main as main
    assert callable(main.app.layout)


def test_i18n_build_layout_en_constructs():
    """build_layout('en') builds without error."""
    from app.layout import build_layout
    layout = build_layout("en")
    assert layout is not None


def test_i18n_build_layout_it_constructs():
    """build_layout('it') builds without error."""
    from app.layout import build_layout
    layout = build_layout("it")
    assert layout is not None


def test_i18n_serve_layout_lang_it_seeds_store():
    """serve_layout() with ?lang=it seeds LANGUAGE_STORE with 'it'."""
    import app.main as main
    app_obj = main.app
    with app_obj.server.test_request_context("/?lang=it"):
        layout = main.serve_layout()
        layout_str = str(layout)
        # LANGUAGE_STORE should carry data='it'
        idx = layout_str.find(ids.LANGUAGE_STORE)
        assert idx != -1, "LANGUAGE_STORE id not found in rendered layout"
        context = layout_str[idx:idx + 80]
        assert "data='it'" in context, (
            f"LANGUAGE_STORE should have data='it'; got: {context!r}"
        )


def test_i18n_serve_layout_unknown_lang_falls_back_en():
    """serve_layout() with ?lang=xx seeds LANGUAGE_STORE with 'en'."""
    import app.main as main
    app_obj = main.app
    with app_obj.server.test_request_context("/?lang=xx"):
        layout = main.serve_layout()
        layout_str = str(layout)
        idx = layout_str.find(ids.LANGUAGE_STORE)
        assert idx != -1
        context = layout_str[idx:idx + 80]
        assert "data='en'" in context, (
            f"LANGUAGE_STORE should fall back to 'en'; got: {context!r}"
        )


def test_i18n_lang_resolved_from_referrer_when_args_empty():
    """Regression: Dash fetches the layout via /_dash-layout WITHOUT the page's
    ?lang query, carrying it only as the referrer. serve_layout must read the
    language from the referrer when request.args has none (otherwise the live
    language switch silently does nothing)."""
    import app.main as main
    app_obj = main.app
    # No ?lang in the request itself; the page URL is the referrer.
    with app_obj.server.test_request_context(
        "/_dash-layout", headers={"Referer": "http://localhost:8050/?lang=it"}
    ):
        assert main._lang_from_request() == "it"
        layout_str = str(main.serve_layout())
        idx = layout_str.find(ids.LANGUAGE_STORE)
        assert idx != -1 and "data='it'" in layout_str[idx:idx + 80]
    # Bad referrer lang falls back to English.
    with app_obj.server.test_request_context(
        "/_dash-layout", headers={"Referer": "http://localhost:8050/?lang=xx"}
    ):
        assert main._lang_from_request() == "en"


# ===========================================================================
# i18n — language selector present in layout
# ===========================================================================
def test_i18n_language_selector_in_rendered_layout():
    """Rendered layout contains a component with id == ids.LANGUAGE_SELECTOR."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.LANGUAGE_SELECTOR in layout_str, (
        f"Rendered layout does not contain LANGUAGE_SELECTOR id={ids.LANGUAGE_SELECTOR!r}"
    )


# ===========================================================================
# Help tooltips — optimize panel emits help-icon spans (§12, D5)
# ===========================================================================
def test_i18n_help_icons_present_in_optimize_panel():
    """Rendered layout contains help-icon spans for optimize control fields."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.HELP_ICON_TYPE in layout_str, (
        f"HELP_ICON_TYPE={ids.HELP_ICON_TYPE!r} not found in rendered layout"
    )
    # Verify a few specific field keys exist
    for field in ("mode", "target_wavelength", "sharpness"):
        assert f"'field': '{field}'" in layout_str or f'"field": "{field}"' in layout_str, (
            f"help icon for field={field!r} not found in rendered layout"
        )


def test_i18n_help_icon_tip_texts_are_nonempty():
    """Every tip_* key in the EN catalog has non-empty tooltip text."""
    en = config.TRANSLATIONS["en"]
    tip_keys = [k for k in en if k.startswith("tip_")]
    assert len(tip_keys) > 0, "no tip_* keys found in TRANSLATIONS['en']"
    for k in tip_keys:
        assert en[k], f"tip key {k!r} has empty tooltip text in EN catalog"


def test_i18n_help_icon_tip_texts_it_nonempty():
    """Every tip_* key in the IT catalog has non-empty tooltip text."""
    it = config.TRANSLATIONS["it"]
    tip_keys = [k for k in it if k.startswith("tip_")]
    for k in tip_keys:
        assert it[k], f"tip key {k!r} has empty tooltip text in IT catalog"


# ===========================================================================
# CSS asset exists with tooltip and contrast rules (§12.7)
# ===========================================================================
def test_i18n_style_css_exists():
    """app/assets/style.css exists."""
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "assets", "style.css",
    )
    assert os.path.isfile(css_path), f"style.css not found at {css_path}"


def test_i18n_style_css_contains_help_rules():
    """style.css contains the .help and .help-text tooltip rules."""
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "assets", "style.css",
    )
    css = open(css_path).read()
    assert ".help" in css, "style.css missing .help rule"
    assert ".help-text" in css, "style.css missing .help-text rule"


def test_i18n_style_css_contains_label_contrast_rule():
    """style.css contains the .app-root label dark-theme contrast rule."""
    css_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "assets", "style.css",
    )
    css = open(css_path).read()
    assert ".app-root label" in css, "style.css missing .app-root label contrast rule"


# ===========================================================================
# Smoke (extended): callback_map and layout component ids
# ===========================================================================
def test_i18n_smoke_callback_map_has_flat_layer_options():
    """callback_map includes opt_variable_flat_layers_input.options."""
    import app.main as main
    app_obj = main.create_app()
    target = f"{ids.OPT_VARIABLE_FLAT_LAYERS_INPUT}.options"
    assert target in app_obj.callback_map, (
        f"callback_map missing {target!r}"
    )


def test_i18n_smoke_layout_contains_opt_variable_mode_tabs():
    """Rendered layout (via build_layout) contains OPT_VARIABLE_MODE_TABS id."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.OPT_VARIABLE_MODE_TABS in layout_str, (
        f"Rendered layout missing '{ids.OPT_VARIABLE_MODE_TABS}'"
    )


def test_i18n_smoke_layout_contains_opt_variable_flat_layers_input():
    """Rendered layout (via build_layout) contains OPT_VARIABLE_FLAT_LAYERS_INPUT id."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.OPT_VARIABLE_FLAT_LAYERS_INPUT in layout_str, (
        f"Rendered layout missing '{ids.OPT_VARIABLE_FLAT_LAYERS_INPUT}'"
    )


# ===========================================================================
# Export (workflow 2): optimized spectra + parameters .txt files
# ===========================================================================
def _fake_optimization_result() -> dict:
    """Minimal optimization-result dict shaped like state.run_*_optimization."""
    return {
        "thicknesses_nm": [72.0, 103.0, 190.5, 103.0, 72.0],
        "variable_thicknesses_nm": [190.5],
        "history": [1.0, 0.5, 0.1],
        "final_result": {
            "wavelength_nm": [500.0, 550.0, 600.0, 650.0],
            "R": [0.9, 0.8, 0.2, 0.85],
            "T": [0.1, 0.2, 0.8, 0.15],
            "A": [0.0, 0.0, 0.0, 0.0],
            "polarizations": ["s"],
        },
        "resonance": {
            "resonance_wavelength_nm": 600.0,
            "linewidth_nm": 12.0,
            "quality_factor": 50.0,
            "extremum_value": 0.2,
        },
    }


def test_make_export_prefix_connects_filenames():
    """Both files share one prefix: simulation_<timestamp>."""
    prefix, ts = state.make_export_prefix(timestamp="20260101_010203")
    assert prefix == "simulation_20260101_010203"
    assert ts == "20260101_010203"
    assert f"{prefix}_spectra.txt".startswith(prefix)
    assert f"{prefix}_parameters.txt".startswith(prefix)


def test_build_optimized_spectra_text_columns_and_rows():
    """Spectra text has the wavelength/R/T/A header and one row per grid point."""
    result = _fake_optimization_result()
    text = state.build_optimized_spectra_text(
        result, file_prefix="simulation_X", timestamp="X"
    )
    assert "columns: wavelength_nm R T A" in text
    assert "file_prefix: simulation_X" in text
    data_rows = [
        line for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(data_rows) == 4  # one per wavelength
    # first data row begins with the first wavelength
    assert data_rows[0].split()[0].startswith("500")


def test_build_optimized_spectra_text_raises_without_result():
    """No final_result => ValueError (callback turns this into a status msg)."""
    with pytest.raises(ValueError):
        state.build_optimized_spectra_text({}, file_prefix="p", timestamp="t")


def test_build_optimization_parameters_text_contains_sections():
    """Parameters dump captures settings, structure, and optimized result."""
    cfg = config.default_opt_stack_config()
    opt = config.default_optimize_config()
    result = _fake_optimization_result()
    text = state.build_optimization_parameters_text(
        cfg, opt, result, file_prefix="simulation_X", timestamp="X"
    )
    for marker in (
        "[Optimization settings]",
        "[Grid / incidence]",
        "[Structure]",
        "[Optimized result]",
        "mode: resonance",
        "input_mirror_period",
        "cavity (enabled)",
        "optimized_thicknesses_nm:",
        "resonance_wavelength_nm: 600",
        "quality_factor_Q: 50",
    ):
        assert marker in text, f"parameters text missing {marker!r}"


def test_export_button_and_downloads_in_layout():
    """Rendered layout contains the export button + single ZIP download target."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    for component_id in (
        ids.OPTIMIZE_EXPORT_BUTTON,
        ids.OPTIMIZE_EXPORT_DOWNLOAD,
        ids.OPTIMIZE_EXPORT_STATUS,
    ):
        assert component_id in layout_str, f"layout missing {component_id!r}"


def test_export_callback_registered():
    """callback_map wires the single ZIP download data output to the callback."""
    import app.main as main
    app_obj = main.create_app()
    joined_keys = "\n".join(app_obj.callback_map)
    assert f"{ids.OPTIMIZE_EXPORT_DOWNLOAD}.data" in joined_keys, (
        f"callback_map missing {ids.OPTIMIZE_EXPORT_DOWNLOAD}.data"
    )


def test_build_export_zip_contains_both_named_txt_files():
    """The export ZIP holds both connected .txt files with the right names."""
    import io as _io
    import zipfile

    zip_bytes = state.build_export_zip_bytes(
        "simulation_X", spectra_text="spectra-body", params_text="params-body"
    )
    with zipfile.ZipFile(_io.BytesIO(zip_bytes)) as archive:
        names = set(archive.namelist())
        assert names == {"simulation_X_spectra.txt", "simulation_X_parameters.txt"}
        assert archive.read("simulation_X_spectra.txt").decode() == "spectra-body"
        assert archive.read("simulation_X_parameters.txt").decode() == "params-body"


def test_export_done_label_formats_with_prefix():
    """export_done is a shared template that embeds the file prefix."""
    for lang in ("en", "it"):
        msg = config.labels_for(lang)["export_done"].format(prefix="simulation_X")
        assert "simulation_X_spectra.txt" in msg
        assert "simulation_X_parameters.txt" in msg


# ---------------------------------------------------------------------------
# Simulation-tab export (mirrors the optimize-tab export, ZIP of two .txt)
# ---------------------------------------------------------------------------
def test_make_export_prefix_simulated_name():
    """The simulate tab uses the 'simulated' prefix per the user's naming."""
    prefix, ts = state.make_export_prefix("simulated", timestamp="20260101_010203")
    assert prefix == "simulated_20260101_010203"
    assert f"{prefix}_spectra.txt" == "simulated_20260101_010203_spectra.txt"
    assert f"{prefix}_parameters.txt" == "simulated_20260101_010203_parameters.txt"


def test_build_spectra_text_from_simulation_result():
    """build_spectra_text consumes a run_simulation dict directly (no wrapper)."""
    result = state.run_simulation(config.default_stack_config(), lang="en")
    text = state.build_spectra_text(result, file_prefix="simulated_X", timestamp="X")
    assert "columns: wavelength_nm R T A" in text
    data_rows = [
        line for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(data_rows) == config.default_stack_config()["grid"]["num"]


def test_build_simulation_parameters_text_contains_structure():
    """Simulation parameters dump captures grid, incidence, and finite layers."""
    cfg = config.default_stack_config()
    result = state.run_simulation(cfg, lang="en")
    text = state.build_simulation_parameters_text(
        cfg, result, file_prefix="simulated_X", timestamp="X"
    )
    for marker in (
        "[Grid / incidence]",
        "[Structure]",
        "wavelength_start_nm: 400",
        "polarization: s",
        "finite_layers",
        "layer 1:",
        "substrate:",
    ):
        assert marker in text, f"simulation parameters text missing {marker!r}"


def test_simulate_export_button_and_download_in_layout():
    """Rendered layout has the simulate export button + ZIP download target."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    for component_id in (
        ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_BUTTON_SUFFIX),
        ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_DOWNLOAD_SUFFIX),
        ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_STATUS_SUFFIX),
    ):
        assert component_id in layout_str, f"layout missing {component_id!r}"


def test_simulate_export_callback_registered():
    """callback_map wires the simulate export download data output."""
    import app.main as main
    app_obj = main.create_app()
    joined_keys = "\n".join(app_obj.callback_map)
    target = (
        f"{ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_DOWNLOAD_SUFFIX)}.data"
    )
    assert target in joined_keys, f"callback_map missing {target!r}"
