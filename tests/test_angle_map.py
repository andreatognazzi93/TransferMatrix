"""GUI-level QA tests for the angle-sweep map feature (ANGLE_MAP_CONTRACT §2–§9).

Covers (per the contract §9 and the task spec):
1. State: run_angle_map / run_simulation with sim_mode="angle_map" returns §2.2 schema.
2. validate_config: all new sweep rules, EN+IT messages.
3. plots.angle_map_figure: heatmap structure, channels, zmin/zmax, serialization,
   z orientation.
4. i18n parity: new keys in config.TRANSLATIONS, state._ERRORS, plots._PLOT_TRANSLATIONS.
5. Layout/callback smoke: ids present in rendered layout; callback_map wires show/hide
   and polarization-disable callbacks.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import pytest

from app import config, ids, plots, state


# ===========================================================================
# Helpers
# ===========================================================================

def _constant(n, k=0.0, name=None):
    return {"kind": "constant", "n": n, "k": k, "name": name or f"mat{n}"}


def _base_config(polarization="s", num=21, sim_mode="single"):
    """Minimal valid §2.2 stack-config dict."""
    return {
        "incident": _constant(1.0, name="air"),
        "layers": [
            {"material": _constant(2.35, name="TiO2"), "thickness_nm": 120.0},
        ],
        "substrate": _constant(1.52, name="glass"),
        "grid": {"start_nm": 400.0, "stop_nm": 700.0, "num": num},
        "angle_deg": 0.0,
        "polarization": polarization,
        "sim_mode": sim_mode,
        "angle_sweep": {"start_deg": 0.0, "stop_deg": 30.0, "step_deg": 10.0},
    }


def _angle_map_config(
    polarization="s",
    start_deg=0.0,
    stop_deg=30.0,
    step_deg=10.0,
    num=11,
):
    """Valid angle-map config. Default gives 0, 10, 20, 30 (4 angles)."""
    cfg = _base_config(polarization=polarization, num=num, sim_mode="angle_map")
    cfg["angle_sweep"] = {
        "start_deg": start_deg,
        "stop_deg": stop_deg,
        "step_deg": step_deg,
    }
    return cfg


# ===========================================================================
# §2.1: run_simulation single mode returns mode="single"
# ===========================================================================

def test_run_simulation_single_mode_has_mode_key():
    """Single mode injects mode='single' into the result dict (§2.1)."""
    cfg = _base_config(sim_mode="single")
    result = state.run_simulation(cfg)
    assert result.get("mode") == "single"


def test_run_simulation_single_mode_has_polarizations_list():
    """Single-mode dict has 'polarizations' list, not 'polarization' string (§2.1)."""
    cfg = _base_config(sim_mode="single")
    result = state.run_simulation(cfg)
    assert "polarizations" in result
    assert isinstance(result["polarizations"], list)


# ===========================================================================
# §2.2: run_angle_map / run_simulation(angle_map) schema
# ===========================================================================

def test_run_angle_map_mode_discriminator():
    """angle_map result has mode='angle_map' (§2.2)."""
    result = state.run_angle_map(_angle_map_config())
    assert result["mode"] == "angle_map"


def test_run_angle_map_schema_keys():
    """Result contains all required §2.2 keys."""
    result = state.run_angle_map(_angle_map_config())
    for key in ("mode", "wavelength_nm", "angle_deg", "R", "T", "A", "polarization"):
        assert key in result, f"Missing key {key!r}"


def test_run_angle_map_no_polarizations_list():
    """angle_map dict uses 'polarization' (str), not 'polarizations' (list) (§2.2)."""
    result = state.run_angle_map(_angle_map_config())
    assert "polarization" in result
    assert isinstance(result["polarization"], str)
    # Must NOT carry the single-mode key
    assert "polarizations" not in result


def test_run_angle_map_polarization_values():
    """'polarization' is 's' when 's' is requested, 'p' when 'p' is (§2.2)."""
    for pol in ("s", "p"):
        result = state.run_angle_map(_angle_map_config(polarization=pol))
        assert result["polarization"] == pol


def test_run_angle_map_2d_channel_shape():
    """R/T/A are 2-D nested lists of shape (num_angles, num_wavelengths) (§2.2)."""
    # sweep 0..30 step 10 => angles [0, 10, 20, 30] => 4 angles
    cfg = _angle_map_config(start_deg=0.0, stop_deg=30.0, step_deg=10.0, num=11)
    result = state.run_angle_map(cfg)

    R = np.asarray(result["R"], dtype=float)
    T = np.asarray(result["T"], dtype=float)
    A = np.asarray(result["A"], dtype=float)

    expected_shape = (4, 11)
    assert R.shape == expected_shape, f"R.shape={R.shape} != {expected_shape}"
    assert T.shape == expected_shape
    assert A.shape == expected_shape


def test_run_angle_map_wavelength_axis_length():
    """'wavelength_nm' length equals num grid points (§2.2)."""
    num = 15
    result = state.run_angle_map(_angle_map_config(num=num))
    assert len(result["wavelength_nm"]) == num


def test_run_angle_map_angle_deg_axis_length():
    """'angle_deg' length equals num_angles (§2.2)."""
    # 0..40 step 10 => [0, 10, 20, 30, 40] => 5 angles
    cfg = _angle_map_config(start_deg=0.0, stop_deg=40.0, step_deg=10.0)
    result = state.run_angle_map(cfg)
    assert len(result["angle_deg"]) == 5


def test_run_angle_map_angle_deg_values_exact():
    """angle_deg values are exact float64 degrees, not float32 JAX noise (§2.2 + §5.2)."""
    requested = [0.0, 10.0, 20.0, 30.0]
    cfg = _angle_map_config(start_deg=0.0, stop_deg=30.0, step_deg=10.0)
    result = state.run_angle_map(cfg)
    got = result["angle_deg"]
    assert len(got) == len(requested)
    for expected, actual in zip(requested, got):
        # The run_angle_map override writes exact float64 Python floats (§5.2
        # final overwrite). Tolerance 1e-9 catches any floating-point step
        # accumulation but not float32 quantization (~1e-6 for 30 degrees).
        assert abs(actual - expected) < 1e-9, (
            f"angle_deg value {actual} differs from expected {expected}"
        )


def test_run_simulation_dispatches_to_angle_map():
    """run_simulation with sim_mode='angle_map' returns an angle_map schema (§5.2 branch)."""
    cfg = _angle_map_config()
    result = state.run_simulation(cfg)
    assert result["mode"] == "angle_map"
    assert "angle_deg" in result


def test_run_simulation_dispatches_to_single():
    """run_simulation with sim_mode='single' returns a single schema (§5.2 branch)."""
    cfg = _base_config(sim_mode="single")
    result = state.run_simulation(cfg)
    assert result["mode"] == "single"


# ===========================================================================
# §5.2: Inclusive num_angles formula
# ===========================================================================

def test_num_angles_inclusive_default_sweep():
    """Default sweep 0..80 step 1 => 81 angles (§3.1, §5.2)."""
    # The formula: floor((80-0)/1 + 1e-9) + 1 = 80 + 1 = 81
    cfg = {
        "incident": _constant(1.0, name="air"),
        "layers": [{"material": _constant(2.35, name="TiO2"), "thickness_nm": 120.0}],
        "substrate": _constant(1.52, name="glass"),
        "grid": {"start_nm": 400.0, "stop_nm": 700.0, "num": 5},
        "angle_deg": 0.0,
        "polarization": "s",
        "sim_mode": "angle_map",
        "angle_sweep": {"start_deg": 0.0, "stop_deg": 80.0, "step_deg": 1.0},
    }
    result = state.run_angle_map(cfg)
    assert len(result["angle_deg"]) == 81


def test_num_angles_step_10_gives_9():
    """0..80 step 10 => 9 angles: 0, 10, 20, 30, 40, 50, 60, 70, 80."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=80.0, step_deg=10.0, num=5)
    result = state.run_angle_map(cfg)
    assert len(result["angle_deg"]) == 9


def test_num_angles_0_to_0_step_excluded_by_start_lt_stop_rule():
    """A range with start >= stop fails validation before reaching the formula."""
    cfg = _angle_map_config(start_deg=30.0, stop_deg=30.0, step_deg=1.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("range" in e.lower() for e in errors)


# ===========================================================================
# §7: validate_config new sweep rules
# ===========================================================================

def test_validate_config_angle_map_clean_config():
    """A well-formed angle_map config passes validation with no errors."""
    errors = state.validate_config(_angle_map_config(), lang="en")
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_config_sweep_missing_key():
    """Missing angle_sweep entirely triggers err_sweep_missing (§7 rule 1)."""
    cfg = _angle_map_config()
    del cfg["angle_sweep"]
    errors = state.validate_config(cfg, lang="en")
    assert any("sweep" in e.lower() or "missing" in e.lower() for e in errors)


def test_validate_config_sweep_missing_key_it():
    """Italian: missing angle_sweep returns Italian message."""
    cfg = _angle_map_config()
    del cfg["angle_sweep"]
    errors = state.validate_config(cfg, lang="it")
    assert any("scansione" in e.lower() for e in errors)


def test_validate_config_sweep_not_dict():
    """angle_sweep must be a dict (§7 rule 1)."""
    cfg = _angle_map_config()
    cfg["angle_sweep"] = "bad"
    errors = state.validate_config(cfg, lang="en")
    assert errors  # any error is acceptable; the key is wrong type


def test_validate_config_sweep_params_invalid():
    """Non-numeric sweep params trigger err_sweep_params_invalid (§7 rule 2)."""
    cfg = _angle_map_config()
    cfg["angle_sweep"]["start_deg"] = "bad"
    errors = state.validate_config(cfg, lang="en")
    assert any("param" in e.lower() or "invalid" in e.lower() for e in errors)


def test_validate_config_sweep_range_start_ge_stop():
    """start_deg >= stop_deg triggers err_sweep_range_invalid (§7 rule 3)."""
    cfg = _angle_map_config(start_deg=50.0, stop_deg=30.0, step_deg=1.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("range" in e.lower() or "0 <=" in e for e in errors)


def test_validate_config_sweep_range_start_ge_stop_it():
    """Italian: start >= stop gives Italian range error."""
    cfg = _angle_map_config(start_deg=50.0, stop_deg=30.0, step_deg=1.0)
    errors = state.validate_config(cfg, lang="it")
    assert any("intervallo" in e.lower() for e in errors)


def test_validate_config_sweep_range_negative_start():
    """start_deg < 0 triggers range error (§7 rule 3: 0 <= start)."""
    cfg = _angle_map_config(start_deg=-5.0, stop_deg=30.0, step_deg=5.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("range" in e.lower() or "0 <=" in e for e in errors)


def test_validate_config_sweep_range_stop_gt_90():
    """stop_deg > 90 triggers range error (§7 rule 3: stop <= 90)."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=91.0, step_deg=1.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("range" in e.lower() or "90" in e for e in errors)


def test_validate_config_sweep_step_zero():
    """step_deg == 0 triggers err_sweep_step_invalid (§7 rule 4)."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=30.0, step_deg=0.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("step" in e.lower() for e in errors)


def test_validate_config_sweep_step_negative():
    """step_deg < 0 triggers err_sweep_step_invalid (§7 rule 4)."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=30.0, step_deg=-1.0)
    errors = state.validate_config(cfg, lang="en")
    assert any("step" in e.lower() for e in errors)


def test_validate_config_sweep_too_many_angles():
    """num_angles > 361 triggers err_sweep_too_many with {num} resolved (§7 rule 5)."""
    # 0..90 step 0.24 => floor(90/0.24 + 1e-9)+1 = 375+1 = 376 > 361
    cfg = _angle_map_config(start_deg=0.0, stop_deg=90.0, step_deg=0.24)
    errors = state.validate_config(cfg, lang="en")
    assert any("many" in e.lower() or "361" in e for e in errors)
    # {num} placeholder must be resolved to the actual number
    error_text = " ".join(errors)
    assert "{num}" not in error_text, "Placeholder {num} was not resolved"


def test_validate_config_sweep_too_many_angles_it():
    """Italian: too-many-angles error returns Italian text with resolved {num}."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=90.0, step_deg=0.24)
    errors = state.validate_config(cfg, lang="it")
    assert any("troppi" in e.lower() or "angoli" in e.lower() for e in errors)
    error_text = " ".join(errors)
    assert "{num}" not in error_text, "Placeholder {num} was not resolved in IT"


def test_validate_config_sweep_exactly_361_angles_ok():
    """Exactly 361 angles (0..360 step 1 -- but step>0 and stop<=90 limit applies,
    so use 0..90 step 0.25 which yields 361 angles) must pass."""
    # floor((90-0)/0.25 + 1e-9)+1 = 360+1 = 361
    cfg = _angle_map_config(start_deg=0.0, stop_deg=90.0, step_deg=0.25)
    errors = state.validate_config(cfg, lang="en")
    # Should NOT have a too-many-angles error
    assert not any("many" in e.lower() or "Too many" in e for e in errors), (
        f"Unexpected error for 361 angles: {errors}"
    )


def test_validate_config_sweep_both_polarization_rejected():
    """angle_map + polarization='both' triggers err_angle_map_needs_single_pol (§7 rule 6)."""
    cfg = _angle_map_config(polarization="s")
    cfg["polarization"] = "both"
    errors = state.validate_config(cfg, lang="en")
    assert any("single" in e.lower() or "both" in e.lower() for e in errors)


def test_validate_config_sweep_both_polarization_rejected_it():
    """Italian: 'both' in angle_map mode gives Italian error."""
    cfg = _angle_map_config(polarization="s")
    cfg["polarization"] = "both"
    errors = state.validate_config(cfg, lang="it")
    assert any("singola" in e.lower() or "polarizzazione" in e.lower() for e in errors)


def test_validate_config_single_mode_no_sweep_rules():
    """Sweep rules do NOT apply in single mode (§7 contract)."""
    # Even with a broken angle_sweep dict, single mode must not error on it.
    cfg = _base_config(sim_mode="single")
    cfg["angle_sweep"] = {"start_deg": 99.0, "stop_deg": 10.0, "step_deg": -5.0}
    errors = state.validate_config(cfg, lang="en")
    # No sweep-related errors should appear
    sweep_errors = [e for e in errors if "sweep" in e.lower() or "step" in e.lower()]
    assert sweep_errors == [], f"Sweep rules fired in single mode: {sweep_errors}"


# ===========================================================================
# §6: plots.angle_map_figure
# ===========================================================================

def _make_angle_map_result_dict(
    num_angles=4,
    num_wl=11,
    channels=("R", "T", "A"),
    polarization="s",
):
    """Build a synthetic §2.2 angle-map result dict for figure-builder tests."""
    wl = [400.0 + i * (300.0 / (num_wl - 1)) for i in range(num_wl)]
    angles = [float(i * 10) for i in range(num_angles)]
    # Fill with plausible values summing to 1
    R = [[0.3] * num_wl for _ in range(num_angles)]
    T = [[0.6] * num_wl for _ in range(num_angles)]
    A = [[0.1] * num_wl for _ in range(num_angles)]
    d = {
        "mode": "angle_map",
        "wavelength_nm": wl,
        "angle_deg": angles,
        "polarization": polarization,
    }
    if "R" in channels:
        d["R"] = R
    if "T" in channels:
        d["T"] = T
    if "A" in channels:
        d["A"] = A
    return d


def test_angle_map_figure_returns_figure():
    """angle_map_figure returns a go.Figure (§6.1)."""
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d)
    assert isinstance(fig, go.Figure)


def test_angle_map_figure_3_channels_3_heatmaps():
    """3 channels => 3 Heatmap traces (§6.1)."""
    d = _make_angle_map_result_dict(channels=("R", "T", "A"))
    fig = plots.angle_map_figure(d, channels=("R", "T", "A"))
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 3, f"Expected 3 Heatmap traces, got {len(heatmaps)}"


def test_angle_map_figure_1_channel_1_heatmap():
    """1 channel => 1 Heatmap trace (§6.1)."""
    d = _make_angle_map_result_dict(channels=("R",))
    fig = plots.angle_map_figure(d, channels=("R",))
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 1, f"Expected 1 Heatmap trace, got {len(heatmaps)}"


def test_angle_map_figure_heatmap_zmin_zmax():
    """Each Heatmap has zmin=0, zmax=1, zauto=False (§6.1)."""
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d, channels=("R", "T", "A"))
    for trace in fig.data:
        if isinstance(trace, go.Heatmap):
            assert trace.zmin == 0, f"zmin={trace.zmin} != 0"
            assert trace.zmax == 1, f"zmax={trace.zmax} != 1"
            assert trace.zauto == False, "zauto must be False"


def test_angle_map_figure_heatmap_colorscale():
    """Each Heatmap uses Viridis colorscale (§6.1).

    Plotly expands the string 'Viridis' into a tuple of (stop, color) pairs
    when it stores the value on the trace object.  We verify the colorscale is
    Viridis by checking the first stop color, which is the deep purple
    '#440154' that is the known Viridis start color.
    """
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d, channels=("R",))
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    for h in heatmaps:
        cs = h.colorscale
        # Plotly returns a tuple/list of (stop, color) pairs after expansion.
        # The first entry of the Viridis colorscale is (0.0, '#440154').
        assert cs is not None, "colorscale is None"
        if isinstance(cs, str):
            # If Plotly does not expand (future version), the string must be "Viridis"
            assert cs.lower() == "viridis", f"colorscale string {cs!r} != 'Viridis'"
        else:
            # Expanded form: first entry is (0.0, deep-purple)
            first_stop = cs[0]
            assert float(first_stop[0]) == pytest.approx(0.0), (
                f"First colorscale stop is not 0.0: {first_stop}"
            )
            # The first color of Viridis is a dark purple hex
            assert "#44" in first_stop[1].lower() or "440" in first_stop[1].lower(), (
                f"First colorscale color {first_stop[1]!r} doesn't look like Viridis start"
            )


def test_angle_map_figure_z_orientation():
    """z shape is (num_angles, num_wavelengths) — angle is axis 0 (§6.1 + §2.2)."""
    num_angles, num_wl = 5, 13
    d = _make_angle_map_result_dict(num_angles=num_angles, num_wl=num_wl)
    fig = plots.angle_map_figure(d, channels=("R",))
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    z = np.asarray(heatmaps[0].z, dtype=float)
    assert z.shape == (num_angles, num_wl), (
        f"z.shape={z.shape} != ({num_angles}, {num_wl})"
    )


def test_angle_map_figure_serializes():
    """.to_dict() on the figure must not raise (JSON-safe, §6.1)."""
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d)
    out = fig.to_dict()
    assert isinstance(out, dict)


def test_angle_map_figure_empty_channels_returns_empty():
    """No channels present => empty figure (§6.1)."""
    d = {"mode": "angle_map", "wavelength_nm": [400.0], "angle_deg": [0.0], "polarization": "s"}
    fig = plots.angle_map_figure(d, channels=("R", "T", "A"))
    # No heatmaps; figure should still be a Figure.
    assert isinstance(fig, go.Figure)
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert heatmaps == []


def test_angle_map_figure_in_dunder_all():
    """angle_map_figure is in plots.__all__ (§6.1)."""
    assert "angle_map_figure" in plots.__all__


def test_angle_map_figure_template_white():
    """template='plotly_white' is applied (§6.1)."""
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d)
    assert fig.layout.template.layout.paper_bgcolor == "white" or \
           "white" in str(fig.layout.template)


def test_angle_map_figure_italian_axis_labels():
    """With lang='it' the y-axis title is Italian (§6.2 i18n)."""
    d = _make_angle_map_result_dict()
    fig = plots.angle_map_figure(d, lang="it")
    # The first subplot's yaxis title should be the Italian translation
    assert "Angolo" in str(fig.layout)


def test_angle_map_figure_channels_order():
    """The order of heatmap traces follows the 'channels' argument (§6.1)."""
    d = _make_angle_map_result_dict(channels=("R", "T", "A"))
    fig = plots.angle_map_figure(d, channels=("A", "R"))  # A first, then R
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 2
    # The z data of the first trace should match 'A', not 'R'
    z_first = np.asarray(heatmaps[0].z, dtype=float)
    z_A = np.asarray(d["A"], dtype=float)
    z_R = np.asarray(d["R"], dtype=float)
    # A=0.1 everywhere, R=0.3 everywhere in our synthetic dict
    np.testing.assert_allclose(z_first, z_A, atol=1e-9, err_msg="First trace should be A")


# ===========================================================================
# Cross-check: run_angle_map output through angle_map_figure
# ===========================================================================

def test_angle_map_figure_roundtrip_from_run_angle_map():
    """Full pipeline: run_angle_map -> angle_map_figure produces a valid figure."""
    cfg = _angle_map_config(start_deg=0.0, stop_deg=30.0, step_deg=10.0, num=11)
    result_dict = state.run_angle_map(cfg)
    fig = plots.angle_map_figure(result_dict, channels=("R", "T", "A"))
    assert isinstance(fig, go.Figure)
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    assert len(heatmaps) == 3
    # z shape must be (num_angles=4, num_wl=11)
    z = np.asarray(heatmaps[0].z, dtype=float)
    assert z.shape == (4, 11)
    assert isinstance(fig.to_dict(), dict)


# ===========================================================================
# §2.2 + §1: Cross-check: angle_map_figure z values match direct simulation
# ===========================================================================

def test_angle_map_figure_z_values_match_kernel():
    """R z-values in the figure match direct simulate_spectrum calls (§cover item 4 × §1)."""
    from multilayer_tmm import Layer, Material, Stack, simulate_spectrum, wavelength_grid

    stack = Stack(
        incident=Material.constant(1.0),
        layers=[Layer(Material.constant(2.35), 120.0)],
        substrate=Material.constant(1.52),
    )
    angles = [0.0, 15.0, 30.0]
    wl = wavelength_grid(400.0, 700.0, 11)

    cfg = {
        "incident": {"kind": "constant", "n": 1.0, "k": 0.0, "name": "air"},
        "layers": [{"material": {"kind": "constant", "n": 2.35, "k": 0.0}, "thickness_nm": 120.0}],
        "substrate": {"kind": "constant", "n": 1.52, "k": 0.0, "name": "glass"},
        "grid": {"start_nm": 400.0, "stop_nm": 700.0, "num": 11},
        "angle_deg": 0.0,
        "polarization": "s",
        "sim_mode": "angle_map",
        "angle_sweep": {"start_deg": 0.0, "stop_deg": 30.0, "step_deg": 15.0},
    }
    result_dict = state.run_angle_map(cfg)
    fig = plots.angle_map_figure(result_dict, channels=("R",))
    heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
    z_fig = np.asarray(heatmaps[0].z, dtype=float)

    for i, angle in enumerate(angles):
        direct = simulate_spectrum(stack, wl, angle_deg=angle, polarization="s")
        np.testing.assert_allclose(
            z_fig[i], np.asarray(direct.R, dtype=float),
            rtol=1e-5, atol=1e-7,
            err_msg=f"Figure R row {i} (angle={angle}) does not match direct simulation",
        )


# ===========================================================================
# §3 / §12: i18n parity for new keys
# ===========================================================================

def test_i18n_config_translations_new_angle_map_keys_en():
    """All ANGLE_MAP_CONTRACT §3.2 keys exist in TRANSLATIONS['en']."""
    en = config.TRANSLATIONS["en"]
    required_keys = [
        "sim_mode_label",
        "sim_mode_single",
        "sim_mode_angle_map",
        "angle_sweep_section",
        "angle_start",
        "angle_stop",
        "angle_step",
        "angle_map_pol_hint",
        "res_na_angle_map",
    ]
    for key in required_keys:
        assert key in en, f"Missing TRANSLATIONS['en'] key: {key!r}"


def test_i18n_config_translations_new_angle_map_keys_it():
    """All ANGLE_MAP_CONTRACT §3.2 keys exist in TRANSLATIONS['it']."""
    it = config.TRANSLATIONS["it"]
    required_keys = [
        "sim_mode_label",
        "sim_mode_single",
        "sim_mode_angle_map",
        "angle_sweep_section",
        "angle_start",
        "angle_stop",
        "angle_step",
        "angle_map_pol_hint",
        "res_na_angle_map",
    ]
    for key in required_keys:
        assert key in it, f"Missing TRANSLATIONS['it'] key: {key!r}"


def test_i18n_config_translations_parity_still_holds():
    """Key sets of TRANSLATIONS['en'] and ['it'] remain identical (regression guard)."""
    assert set(config.TRANSLATIONS["en"]) == set(config.TRANSLATIONS["it"])


def test_i18n_state_errors_new_sweep_keys_en():
    """All ANGLE_MAP_CONTRACT §7 error keys exist in state._ERRORS['en']."""
    en = state._ERRORS["en"]
    required_keys = [
        "err_sweep_missing",
        "err_sweep_params_invalid",
        "err_sweep_range_invalid",
        "err_sweep_step_invalid",
        "err_sweep_too_many",
        "err_angle_map_needs_single_pol",
    ]
    for key in required_keys:
        assert key in en, f"Missing _ERRORS['en'] key: {key!r}"


def test_i18n_state_errors_new_sweep_keys_it():
    """All ANGLE_MAP_CONTRACT §7 error keys exist in state._ERRORS['it']."""
    it = state._ERRORS["it"]
    required_keys = [
        "err_sweep_missing",
        "err_sweep_params_invalid",
        "err_sweep_range_invalid",
        "err_sweep_step_invalid",
        "err_sweep_too_many",
        "err_angle_map_needs_single_pol",
    ]
    for key in required_keys:
        assert key in it, f"Missing _ERRORS['it'] key: {key!r}"


def test_i18n_state_errors_parity_still_holds():
    """Key sets of _ERRORS['en'] and ['it'] remain identical (regression guard)."""
    assert set(state._ERRORS["en"]) == set(state._ERRORS["it"])


def test_i18n_plot_translations_new_angle_map_keys_en():
    """All ANGLE_MAP_CONTRACT §6.2 _PLOT_TRANSLATIONS keys exist in ['en']."""
    en = plots._PLOT_TRANSLATIONS["en"]
    required_keys = ["angle_axis", "map_colorbar", "map_hover_angle"]
    for key in required_keys:
        assert key in en, f"Missing _PLOT_TRANSLATIONS['en'] key: {key!r}"


def test_i18n_plot_translations_new_angle_map_keys_it():
    """All ANGLE_MAP_CONTRACT §6.2 _PLOT_TRANSLATIONS keys exist in ['it']."""
    it = plots._PLOT_TRANSLATIONS["it"]
    required_keys = ["angle_axis", "map_colorbar", "map_hover_angle"]
    for key in required_keys:
        assert key in it, f"Missing _PLOT_TRANSLATIONS['it'] key: {key!r}"


def test_i18n_plot_translations_parity_still_holds():
    """Key sets of _PLOT_TRANSLATIONS['en'] and ['it'] remain identical (regression guard)."""
    en_keys = set(plots._PLOT_TRANSLATIONS["en"])
    it_keys = set(plots._PLOT_TRANSLATIONS["it"])
    assert en_keys == it_keys, (
        f"_PLOT_TRANSLATIONS key sets differ; "
        f"extra EN={en_keys - it_keys}, extra IT={it_keys - en_keys}"
    )


def test_i18n_config_translations_angle_map_en_values():
    """Spot-check a few EN translation values match the contract exactly (§3.2)."""
    en = config.TRANSLATIONS["en"]
    assert en["sim_mode_label"] == "Simulation mode"
    assert en["sim_mode_single"] == "Single angle"
    assert en["sim_mode_angle_map"] == "Angle sweep (map)"
    assert en["res_na_angle_map"] == "N/A for angle map"


def test_i18n_config_translations_angle_map_it_values():
    """Spot-check a few IT translation values match the contract exactly (§3.2)."""
    it = config.TRANSLATIONS["it"]
    assert it["sim_mode_label"] == "Modalità di simulazione"
    assert it["sim_mode_single"] == "Angolo singolo"
    assert it["sim_mode_angle_map"] == "Scansione angolare (mappa)"
    assert it["res_na_angle_map"] == "N/D per la mappa angolare"


# ===========================================================================
# §3.2: SIM_MODE_VALUES and options_for
# ===========================================================================

def test_config_sim_mode_values_exists():
    """config.SIM_MODE_VALUES = ('single', 'angle_map') (§3.2)."""
    assert hasattr(config, "SIM_MODE_VALUES")
    assert config.SIM_MODE_VALUES == ("single", "angle_map")


def test_config_sim_mode_options_for_en():
    """options_for(SIM_MODE_VALUES, 'sim_mode_', 'en') returns correct labels (§3.2)."""
    options = config.options_for(config.SIM_MODE_VALUES, "sim_mode_", "en")
    assert len(options) == 2
    labels = {o["value"]: o["label"] for o in options}
    assert labels["single"] == "Single angle"
    assert labels["angle_map"] == "Angle sweep (map)"


def test_config_sim_mode_options_for_it():
    """options_for works in Italian (§3.2)."""
    options = config.options_for(config.SIM_MODE_VALUES, "sim_mode_", "it")
    labels = {o["value"]: o["label"] for o in options}
    assert labels["single"] == "Angolo singolo"
    assert labels["angle_map"] == "Scansione angolare (mappa)"


# ===========================================================================
# §3.1: DEFAULT_ANGLE_SWEEP / default_angle_sweep / default_stack_config
# ===========================================================================

def test_config_default_angle_sweep_values():
    """DEFAULT_ANGLE_SWEEP has start=0, stop=80, step=1 (§3.1)."""
    assert config.DEFAULT_ANGLE_SWEEP == {"start_deg": 0.0, "stop_deg": 80.0, "step_deg": 1.0}


def test_config_default_angle_sweep_function():
    """default_angle_sweep() returns a fresh dict with the contract values (§3.1)."""
    sweep = config.default_angle_sweep()
    assert sweep == {"start_deg": 0.0, "stop_deg": 80.0, "step_deg": 1.0}
    # Must be a fresh copy, not the same object
    sweep["start_deg"] = 999.0
    assert config.DEFAULT_ANGLE_SWEEP["start_deg"] == 0.0


def test_config_default_stack_config_has_sim_mode():
    """default_stack_config() includes 'sim_mode': 'single' (§3.1)."""
    cfg = config.default_stack_config()
    assert cfg.get("sim_mode") == "single"


def test_config_default_stack_config_has_angle_sweep():
    """default_stack_config() includes 'angle_sweep' dict (§3.1)."""
    cfg = config.default_stack_config()
    assert "angle_sweep" in cfg
    assert cfg["angle_sweep"]["start_deg"] == 0.0


# ===========================================================================
# §4: ids.py new constants
# ===========================================================================

def test_ids_simulate_mode_input_exists():
    """ids.SIMULATE_MODE_INPUT is defined (§4)."""
    assert hasattr(ids, "SIMULATE_MODE_INPUT")
    assert ids.SIMULATE_MODE_INPUT == "simulate_mode_input"


def test_ids_simulate_angle_start_input_exists():
    """ids.SIMULATE_ANGLE_START_INPUT is defined (§4)."""
    assert hasattr(ids, "SIMULATE_ANGLE_START_INPUT")
    assert ids.SIMULATE_ANGLE_START_INPUT == "simulate_angle_start_input"


def test_ids_simulate_angle_stop_input_exists():
    """ids.SIMULATE_ANGLE_STOP_INPUT is defined (§4)."""
    assert hasattr(ids, "SIMULATE_ANGLE_STOP_INPUT")
    assert ids.SIMULATE_ANGLE_STOP_INPUT == "simulate_angle_stop_input"


def test_ids_simulate_angle_step_input_exists():
    """ids.SIMULATE_ANGLE_STEP_INPUT is defined (§4)."""
    assert hasattr(ids, "SIMULATE_ANGLE_STEP_INPUT")
    assert ids.SIMULATE_ANGLE_STEP_INPUT == "simulate_angle_step_input"


def test_ids_simulate_single_angle_container_exists():
    """ids.SIMULATE_SINGLE_ANGLE_CONTAINER is defined (§4)."""
    assert hasattr(ids, "SIMULATE_SINGLE_ANGLE_CONTAINER")
    assert ids.SIMULATE_SINGLE_ANGLE_CONTAINER == "simulate_single_angle_container"


def test_ids_simulate_angle_sweep_container_exists():
    """ids.SIMULATE_ANGLE_SWEEP_CONTAINER is defined (§4)."""
    assert hasattr(ids, "SIMULATE_ANGLE_SWEEP_CONTAINER")
    assert ids.SIMULATE_ANGLE_SWEEP_CONTAINER == "simulate_angle_sweep_container"


# ===========================================================================
# §5.1: angle_map_to_dict in state.__all__
# ===========================================================================

def test_state_angle_map_to_dict_in_all():
    """angle_map_to_dict and run_angle_map are in state.__all__ (§5.1)."""
    assert "angle_map_to_dict" in state.__all__
    assert "run_angle_map" in state.__all__


# ===========================================================================
# Layout/callback smoke: component ids present in rendered layout (§8)
# ===========================================================================

def test_layout_contains_simulate_mode_input():
    """Rendered layout contains SIMULATE_MODE_INPUT id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_MODE_INPUT in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_MODE_INPUT}'"
    )


def test_layout_contains_simulate_angle_start_input():
    """Rendered layout contains SIMULATE_ANGLE_START_INPUT id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_ANGLE_START_INPUT in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_ANGLE_START_INPUT}'"
    )


def test_layout_contains_simulate_angle_stop_input():
    """Rendered layout contains SIMULATE_ANGLE_STOP_INPUT id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_ANGLE_STOP_INPUT in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_ANGLE_STOP_INPUT}'"
    )


def test_layout_contains_simulate_angle_step_input():
    """Rendered layout contains SIMULATE_ANGLE_STEP_INPUT id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_ANGLE_STEP_INPUT in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_ANGLE_STEP_INPUT}'"
    )


def test_layout_contains_single_angle_container():
    """Rendered layout contains SIMULATE_SINGLE_ANGLE_CONTAINER id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_SINGLE_ANGLE_CONTAINER in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_SINGLE_ANGLE_CONTAINER}'"
    )


def test_layout_contains_angle_sweep_container():
    """Rendered layout contains SIMULATE_ANGLE_SWEEP_CONTAINER id (§4, §8)."""
    from app.layout import build_layout
    layout_str = str(build_layout("en"))
    assert ids.SIMULATE_ANGLE_SWEEP_CONTAINER in layout_str, (
        f"Rendered layout missing '{ids.SIMULATE_ANGLE_SWEEP_CONTAINER}'"
    )


def test_callback_map_wires_show_hide():
    """callback_map contains the show/hide callback output (§8.4).

    Dash encodes multi-output callbacks with the key pattern
    ``..out1.prop...out2.prop..`` (double dots around each output).  We check
    that the callback map has ONE key that contains both container ids so that
    the show/hide toggle is actually registered.
    """
    import app.main as main
    app_obj = main.create_app()
    # The multi-output key contains both container ids joined by the Dash pattern.
    single_id = ids.SIMULATE_SINGLE_ANGLE_CONTAINER
    sweep_id = ids.SIMULATE_ANGLE_SWEEP_CONTAINER
    matching = [
        k for k in app_obj.callback_map
        if single_id in k and sweep_id in k
    ]
    assert matching, (
        f"No callback_map key contains both '{single_id}' and '{sweep_id}'. "
        f"Available keys with 'angle': "
        f"{[k for k in app_obj.callback_map if 'angle' in k.lower()]}"
    )


def test_callback_map_wires_angle_sweep_container():
    """callback_map contains SIMULATE_ANGLE_SWEEP_CONTAINER.style output (§8.4).

    Checks the same multi-output callback as test_callback_map_wires_show_hide;
    kept as a separate targeted assertion for clarity.
    """
    import app.main as main
    app_obj = main.create_app()
    sweep_id = ids.SIMULATE_ANGLE_SWEEP_CONTAINER
    matching = [k for k in app_obj.callback_map if sweep_id in k]
    assert matching, (
        f"No callback_map key contains '{sweep_id}'; "
        f"available: {list(app_obj.callback_map.keys())[:5]}"
    )


def test_callback_map_wires_polarization_disable():
    """callback_map contains the polarization-disable callback (§8.4).

    The multi-output key contains both POLARIZATION_INPUT.options and
    POLARIZATION_INPUT.value (both are outputs of the same callback).
    """
    import app.main as main
    app_obj = main.create_app()
    pol_id = ids.POLARIZATION_INPUT
    # The key contains the polarization_input id (for both .options and .value)
    matching = [k for k in app_obj.callback_map if pol_id in k]
    assert matching, (
        f"No callback_map key contains '{pol_id}'; "
        f"available keys sample: {list(app_obj.callback_map.keys())[:5]}"
    )


# ===========================================================================
# Direct callback invocation: toggle_sweep_inputs
# ===========================================================================

def _get_toggle_callback_fn():
    """Return the bare toggle_sweep_inputs function via __wrapped__ for direct testing."""
    import app.main as main
    app_obj = main.create_app()
    single_id = ids.SIMULATE_SINGLE_ANGLE_CONTAINER
    sweep_id = ids.SIMULATE_ANGLE_SWEEP_CONTAINER
    toggle_key = next(
        (k for k in app_obj.callback_map if single_id in k and sweep_id in k),
        None,
    )
    assert toggle_key is not None, "show/hide callback not found in callback_map"
    wrapped = app_obj.callback_map[toggle_key]["callback"]
    # __wrapped__ is the inner function WITHOUT Dash's context injection
    return wrapped.__wrapped__


def test_toggle_sweep_inputs_single_mode():
    """toggle_sweep_inputs('single') => single container block, sweep container none (§8.4)."""
    fn = _get_toggle_callback_fn()
    single_style, sweep_style = fn("single")
    assert single_style.get("display") == "block", (
        f"single mode: single container should be 'block', got {single_style}"
    )
    assert sweep_style.get("display") == "none", (
        f"single mode: sweep container should be 'none', got {sweep_style}"
    )


def test_toggle_sweep_inputs_angle_map_mode_direct():
    """toggle_sweep_inputs('angle_map') => single hidden, sweep visible (§8.4)."""
    fn = _get_toggle_callback_fn()
    single_style, sweep_style = fn("angle_map")
    assert single_style.get("display") == "none", (
        f"angle_map mode: single container should be 'none', got {single_style}"
    )
    assert sweep_style.get("display") == "block", (
        f"angle_map mode: sweep container should be 'block', got {sweep_style}"
    )


def test_toggle_sweep_inputs_none_acts_as_single():
    """toggle_sweep_inputs(None) should act like 'single' (default guard, §8.4)."""
    fn = _get_toggle_callback_fn()
    single_style, sweep_style = fn(None)
    assert single_style.get("display") == "block"
    assert sweep_style.get("display") == "none"
