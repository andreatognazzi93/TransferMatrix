"""QA tests for §11 (Addition C): two optimization modes, enumerate_expanded_layers,
expand_optimization_variables, compat shim, tied-keeps-repeats-uniform, plots features.

These tests are the QA gate for the feature implemented in:
  app/state.py, app/ids.py, app/config.py, app/components/optimize_panel.py,
  app/callbacks/stack_callbacks.py, app/plots.py

Rules:
- No edits to app/ or multilayer_tmm/; defects are reported precisely.
- Runs via: PYTHONPATH=. python3 -m pytest -q
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import pytest

from app import config, ids, plots, state


# ===========================================================================
# Canonical example config helper
# ===========================================================================
def _const(n, k=0.0, name=None):
    d = {"kind": "constant", "n": n, "k": k}
    if name:
        d["name"] = name
    return d


def _canonical_config(*, mode="tied", cavity=True, input_layers=None,
                      output_layers=None, flat_layers=None,
                      repeat_m=3, repeat_k=3) -> dict:
    """Build the canonical example: [high 72nm, low 103nm] ×M, cavity, [low, high] ×K.

    Matches examples/optimize_resonance_target.py and config.default_opt_stack_config().
    """
    return {
        "incident": _const(1.0, name="air"),
        "input_group": {
            "layers": [
                {"material": _const(2.1, name="high_index"), "thickness_nm": 72.0},
                {"material": _const(1.45, name="low_index"), "thickness_nm": 103.0},
            ],
            "repeat": repeat_m,
        },
        "cavity": {
            "material": _const(1.6, name="cavity"),
            "thickness_nm": 190.0,
            "enabled": True,
        },
        "output_group": {
            "layers": [
                {"material": _const(1.45, name="low_index"), "thickness_nm": 103.0},
                {"material": _const(2.1, name="high_index"), "thickness_nm": 72.0},
            ],
            "repeat": repeat_k,
        },
        "substrate": _const(1.0, name="air"),
        "grid": {"start_nm": 520.0, "stop_nm": 720.0, "num": 61},
        "angle_deg": 0.0,
        "polarization": "s",
        "variable": {
            "mode": mode,
            "cavity": cavity,
            "input_layers": input_layers if input_layers is not None else [],
            "output_layers": output_layers if output_layers is not None else [],
            "flat_layers": flat_layers if flat_layers is not None else [],
        },
    }


# ===========================================================================
# expand_optimization_variables — tied mode
# ===========================================================================
class TestExpandVariablesTied:
    def test_cavity_only_default(self):
        """Default config (cavity only, tied): one group = [6] (singleton)."""
        cfg = _canonical_config(mode="tied", cavity=True,
                                input_layers=[], output_layers=[])
        stack, groups = state.expand_optimization_variables(cfg)
        # flat layout: 3*2=6 input layers, cavity at 6, 3*2=6 output -> 13 total
        assert stack.num_layers == 13
        assert groups == [[6]]

    def test_input_period_layer_0_with_m3(self):
        """Input period-layer 0 with M=3 → group [0, 2, 4] (all 3 repeats, Lin=2)."""
        cfg = _canonical_config(mode="tied", cavity=False,
                                input_layers=[0], output_layers=[])
        # cavity disabled; need at least one group
        cfg["cavity"]["enabled"] = False
        stack, groups = state.expand_optimization_variables(cfg)
        # Lin=2, so layer 0 repeats at 0, 2, 4
        assert [0, 2, 4] in groups
        input_group = [g for g in groups if 0 in g][0]
        assert sorted(input_group) == [0, 2, 4]

    def test_cavity_and_input_period_layer_0(self):
        """Input period-layer 0 selected + cavity → two groups, sorted by first index."""
        cfg = _canonical_config(mode="tied", cavity=True, input_layers=[0])
        stack, groups = state.expand_optimization_variables(cfg)
        # groups sorted by first element: [0,2,4] first, then [6]
        assert len(groups) == 2
        first_elements = [g[0] for g in groups]
        assert first_elements == sorted(first_elements)
        # cavity singleton
        cavity_group = [g for g in groups if 6 in g]
        assert len(cavity_group) == 1
        assert cavity_group[0] == [6]
        # input-layer-0 group spans all 3 repeats
        in_group = [g for g in groups if 0 in g]
        assert len(in_group) == 1
        assert sorted(in_group[0]) == [0, 2, 4]

    def test_inner_lists_are_sorted_ascending(self):
        """Every inner list in groups is sorted ascending (invariant)."""
        cfg = _canonical_config(mode="tied", cavity=True, input_layers=[0, 1])
        stack, groups = state.expand_optimization_variables(cfg)
        for g in groups:
            assert g == sorted(g), f"group {g} is not sorted"

    def test_groups_nonempty(self):
        """groups is non-empty when at least one variable is selected."""
        cfg = _canonical_config(mode="tied", cavity=True)
        _, groups = state.expand_optimization_variables(cfg)
        assert len(groups) > 0
        for g in groups:
            assert len(g) > 0

    def test_groups_sorted_by_first_index(self):
        """groups is sorted by each group's first flat index."""
        cfg = _canonical_config(mode="tied", cavity=True, input_layers=[0, 1])
        _, groups = state.expand_optimization_variables(cfg)
        first_elements = [g[0] for g in groups]
        assert first_elements == sorted(first_elements)

    def test_output_period_layer(self):
        """Output period-layer 1 with K=3, Lout=2 → repeats at out_start+1, out_start+3, out_start+5."""
        # With M=3, Lin=2, cavity enabled: cavity_index=6, out_start=7
        # Lout=2, j=1 → [7+1, 7+1+2, 7+1+4] = [8, 10, 12]
        cfg = _canonical_config(mode="tied", cavity=False,
                                output_layers=[1])
        cfg["cavity"]["enabled"] = False
        # out_start = 3*2 = 6 (no cavity)
        stack, groups = state.expand_optimization_variables(cfg)
        out_group = groups[0]
        # out_start=6 (no cavity), j=1, Lout=2 → [6+1, 6+1+2, 6+1+4] = [7, 9, 11]
        assert sorted(out_group) == [7, 9, 11]

    def test_tied_rejects_empty_selection(self):
        """Tied mode with nothing selected raises ValueError (English default)."""
        cfg = _canonical_config(mode="tied", cavity=False,
                                input_layers=[], output_layers=[])
        cfg["cavity"]["enabled"] = False
        # Add at least one output layer to make the stack non-empty
        cfg["output_group"]["repeat"] = 1
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg)
        # Default is English — check English words
        msg = str(exc_info.value)
        assert any(word in msg for word in ("variable", "Select", "select", "least"))

    def test_tied_rejects_empty_selection_it(self):
        """Tied mode with nothing selected raises ValueError (Italian when lang='it')."""
        cfg = _canonical_config(mode="tied", cavity=False,
                                input_layers=[], output_layers=[])
        cfg["cavity"]["enabled"] = False
        cfg["output_group"]["repeat"] = 1
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        msg = str(exc_info.value)
        assert any(word in msg for word in ("variabile", "selezionare", "Selezionare"))

    def test_tied_rejects_cavity_when_disabled(self):
        """Tied mode: selecting cavity as variable when it's disabled → ValueError (Italian)."""
        cfg = _canonical_config(mode="tied", cavity=True)
        cfg["cavity"]["enabled"] = False
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg)
        assert "cavità" in str(exc_info.value).lower() or "cav" in str(exc_info.value).lower()


# ===========================================================================
# expand_optimization_variables — independent mode
# ===========================================================================
class TestExpandVariablesIndependent:
    def test_singleton_groups(self):
        """Independent mode: flat_layers=[1, 7] → [[1], [7]] (singleton groups)."""
        cfg = _canonical_config(mode="independent", flat_layers=[1, 7])
        stack, groups = state.expand_optimization_variables(cfg)
        assert groups == [[1], [7]]

    def test_empty_flat_layers_raises(self):
        """Independent mode with flat_layers=[] raises ValueError (English default)."""
        cfg = _canonical_config(mode="independent", flat_layers=[])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg)
        msg = str(exc_info.value)
        assert any(word in msg for word in ("variable", "Select", "select", "least"))

    def test_empty_flat_layers_raises_it(self):
        """Independent mode with flat_layers=[] raises ValueError (Italian)."""
        cfg = _canonical_config(mode="independent", flat_layers=[])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        msg = str(exc_info.value)
        assert any(word in msg for word in ("variabile", "selezionare", "Selezionare"))

    def test_out_of_range_flat_index_raises(self):
        """Independent mode: flat index >= num_layers raises ValueError (English default)."""
        cfg = _canonical_config(mode="independent", flat_layers=[100])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg)
        assert "range" in str(exc_info.value).lower() or "out of" in str(exc_info.value).lower()

    def test_out_of_range_flat_index_raises_it(self):
        """Independent mode: flat index >= num_layers raises ValueError (Italian)."""
        cfg = _canonical_config(mode="independent", flat_layers=[100])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        assert "intervallo" in str(exc_info.value) or "fuori" in str(exc_info.value)

    def test_deduplication(self):
        """Independent mode deduplicates repeated indices."""
        cfg = _canonical_config(mode="independent", flat_layers=[1, 1, 7])
        _, groups = state.expand_optimization_variables(cfg)
        assert groups == [[1], [7]]

    def test_period_selectors_ignored_in_independent(self):
        """Independent mode ignores cavity/input_layers/output_layers."""
        cfg = _canonical_config(mode="independent", cavity=True,
                                input_layers=[0], output_layers=[1],
                                flat_layers=[3])
        _, groups = state.expand_optimization_variables(cfg)
        # Only flat_layers=[3] matters → [[3]]
        assert groups == [[3]]

    def test_result_schema_has_correct_keys(self):
        """run_thickness_optimization returns dict with expected keys (short run)."""
        cfg = _canonical_config(mode="independent", flat_layers=[6])
        opt_config = {
            "spectrum": "R",
            "steps": 2,
            "learning_rate": 0.01,
            "lower_bound_nm": 0.0,
        }
        result = state.run_thickness_optimization(cfg, opt_config)
        assert "thicknesses_nm" in result
        assert "variable_thicknesses_nm" in result
        assert "history" in result
        assert "final_result" in result

    def test_variable_thicknesses_nm_length_equals_selected(self):
        """In independent mode, variable_thicknesses_nm length == len(selected flat layers)."""
        cfg = _canonical_config(mode="independent", flat_layers=[1, 6])
        opt_config = {"spectrum": "R", "steps": 2, "learning_rate": 0.01, "lower_bound_nm": 0.0}
        result = state.run_thickness_optimization(cfg, opt_config)
        assert len(result["variable_thicknesses_nm"]) == 2


# ===========================================================================
# expand_optimization_config — compat shim
# ===========================================================================
class TestCompatShim:
    def test_cavity_only_returns_tuple_6(self):
        """Default (cavity-only, tied, M=K=3, Lin=Lout=2): compat shim → (6,)."""
        cfg = _canonical_config(mode="tied", cavity=True,
                                input_layers=[], output_layers=[])
        stack, indices = state.expand_optimization_config(cfg)
        assert isinstance(indices, tuple)
        assert indices == (6,)

    def test_compat_shim_returns_sorted_flat_tuple(self):
        """Compat shim flattens groups to a sorted de-duplicated tuple."""
        cfg = _canonical_config(mode="tied", cavity=True, input_layers=[0])
        _, indices = state.expand_optimization_config(cfg)
        assert isinstance(indices, tuple)
        assert list(indices) == sorted(set(indices))
        assert 6 in indices    # cavity
        assert 0 in indices    # input layer 0 (first repeat)

    def test_compat_shim_union_of_all_groups(self):
        """Compat shim union equals the flat union of all group members."""
        cfg = _canonical_config(mode="tied", cavity=True, input_layers=[0, 1])
        stack, groups = state.expand_optimization_variables(cfg)
        _, flat_tuple = state.expand_optimization_config(cfg)
        expected = sorted({idx for g in groups for idx in g})
        assert list(flat_tuple) == expected


# ===========================================================================
# enumerate_expanded_layers
# ===========================================================================
class TestEnumerateExpandedLayers:
    def test_length_canonical(self):
        """Length = M*Lin + 1(cavity) + K*Lout for default M=K=3, Lin=Lout=2."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        # 3*2 + 1 + 3*2 = 13
        assert len(entries) == 13

    def test_cavity_flat_index(self):
        """Cavity entry has flat_index = M*Lin (= 6 for M=3, Lin=2).

        Default (EN) uses 'Cavity'; IT uses 'Cavità'.
        """
        cfg = _canonical_config()
        # Test EN default: label contains 'Cavity'
        entries_en = state.enumerate_expanded_layers(cfg)
        cavity_entries_en = [e for e in entries_en if "Cavity" in e["label"]]
        assert len(cavity_entries_en) == 1
        assert cavity_entries_en[0]["flat_index"] == 6
        # Test IT: label contains 'Cavità'
        entries_it = state.enumerate_expanded_layers(cfg, lang="it")
        cavity_entries_it = [e for e in entries_it if "Cavità" in e["label"]]
        assert len(cavity_entries_it) == 1
        assert cavity_entries_it[0]["flat_index"] == 6

    def test_flat_indices_contiguous_from_zero(self):
        """flat_index values are 0..N-1 in physical order."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        for i, e in enumerate(entries):
            assert e["flat_index"] == i

    def test_labels_are_english_and_nonempty(self):
        """Default (EN) labels are non-empty strings containing English words."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        for e in entries:
            assert isinstance(e["label"], str)
            assert len(e["label"]) > 0
            # English labels contain one of: Input, Cavity, Output
            assert any(word in e["label"] for word in ("Input", "Cavity", "Output"))

    def test_labels_are_italian_and_nonempty(self):
        """IT labels (lang='it') are non-empty strings containing Italian words."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg, lang="it")
        for e in entries:
            assert isinstance(e["label"], str)
            assert len(e["label"]) > 0
            # Italian labels contain one of: Ingresso, Cavità, Uscita
            assert any(word in e["label"] for word in ("Ingresso", "Cavità", "Uscita"))

    def test_input_entries_labeled_input(self):
        """Input mirror entries have EN labels containing 'Input' (default)."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        input_entries = entries[:6]  # first M*Lin = 6 entries
        for e in input_entries:
            assert "Input" in e["label"]

    def test_input_entries_labeled_ingresso(self):
        """Input mirror entries have IT labels containing 'Ingresso' (lang='it')."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg, lang="it")
        input_entries = entries[:6]  # first M*Lin = 6 entries
        for e in input_entries:
            assert "Ingresso" in e["label"]

    def test_output_entries_labeled_output(self):
        """Output mirror entries have EN labels containing 'Output' (default)."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        output_entries = entries[7:]  # after cavity (index 6)
        for e in output_entries:
            assert "Output" in e["label"]

    def test_output_entries_labeled_uscita(self):
        """Output mirror entries have IT labels containing 'Uscita' (lang='it')."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg, lang="it")
        output_entries = entries[7:]  # after cavity (index 6)
        for e in output_entries:
            assert "Uscita" in e["label"]

    def test_without_cavity(self):
        """With cavity disabled, length = M*Lin + K*Lout (no cavity row)."""
        cfg = _canonical_config()
        cfg["cavity"]["enabled"] = False
        entries = state.enumerate_expanded_layers(cfg)
        # 3*2 + 3*2 = 12 (no cavity)
        assert len(entries) == 12
        # EN default: no "Cavity" entries; IT: no "Cavità" entries
        cavity_entries = [e for e in entries if "Cavity" in e["label"] or "Cavità" in e["label"]]
        assert cavity_entries == []

    def test_thickness_matches_period_def(self):
        """First input layer's thickness_nm matches the period definition."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        # First layer is input period-layer 0: high_index 72nm
        assert entries[0]["thickness_nm"] == pytest.approx(72.0)
        # Second layer is input period-layer 1: low_index 103nm
        assert entries[1]["thickness_nm"] == pytest.approx(103.0)

    def test_material_name_in_entry(self):
        """material_name field is non-empty for entries with named materials."""
        cfg = _canonical_config()
        entries = state.enumerate_expanded_layers(cfg)
        for e in entries:
            assert "material_name" in e
            assert isinstance(e["material_name"], str)
            assert len(e["material_name"]) > 0

    def test_parametric_length(self):
        """Length matches M*Lin + cavity_count + K*Lout for varied M, K."""
        for m, k in [(1, 1), (5, 2), (0, 3)]:
            if m == 0:
                # stack needs something: use cavity-only
                cfg = _canonical_config(repeat_m=m, repeat_k=k)
                cfg["output_group"]["repeat"] = k
                cfg["input_group"]["repeat"] = m
            else:
                cfg = _canonical_config(repeat_m=m, repeat_k=k)
            cfg["variable"]["cavity"] = True
            entries = state.enumerate_expanded_layers(cfg)
            expected = m * 2 + 1 + k * 2  # Lin=Lout=2, cavity=1
            assert len(entries) == expected, (
                f"M={m}, K={k}: expected {expected}, got {len(entries)}"
            )


# ===========================================================================
# Tied mode: run keeps repeats uniform
# ===========================================================================
class TestTiedKeepsRepeatsUniform:
    """After a short tied-mode run the optimized thicknesses of all
    copies of a period-layer are equal (within floating-point tolerance).
    """

    def _run(self, mode="tied", input_layers=None, flat_layers=None):
        cfg = _canonical_config(mode=mode,
                                cavity=False,
                                input_layers=input_layers or [0],
                                flat_layers=flat_layers or [])
        cfg["cavity"]["enabled"] = False
        opt_config = {
            "spectrum": "R",
            "steps": 3,
            "learning_rate": 0.01,
            "lower_bound_nm": 0.0,
        }
        return state.run_thickness_optimization(cfg, opt_config)

    def test_tied_input_layer0_all_copies_equal(self):
        """Tied mode, input layer 0 with M=3: t[0] == t[2] == t[4] after opt."""
        result = self._run(mode="tied", input_layers=[0])
        t = result["thicknesses_nm"]
        # input layer 0 repeats at indices 0, 2, 4 (Lin=2, M=3)
        assert t[0] == pytest.approx(t[2], rel=1e-6, abs=1e-8), (
            f"t[0]={t[0]} != t[2]={t[2]} — tied mode did not broadcast"
        )
        assert t[0] == pytest.approx(t[4], rel=1e-6, abs=1e-8), (
            f"t[0]={t[0]} != t[4]={t[4]} — tied mode did not broadcast"
        )

    def test_tied_other_layers_unchanged(self):
        """Tied mode: the OTHER period-layer (index 1) stays at its initial value."""
        cfg_before = _canonical_config(mode="tied", cavity=False, input_layers=[0])
        cfg_before["cavity"]["enabled"] = False
        # initial thickness of period-layer 1 (low_index) = 103 nm
        initial_low = 103.0

        result = self._run(mode="tied", input_layers=[0])
        t = result["thicknesses_nm"]
        # period-layer 1 copies at flat indices 1, 3, 5 — should be unchanged
        for idx in (1, 3, 5):
            assert t[idx] == pytest.approx(initial_low, rel=1e-5), (
                f"t[{idx}]={t[idx]} changed but layer 1 not selected"
            )

    def test_resonance_run_tied_keeps_uniform(self):
        """run_resonance_optimization in tied mode also keeps repeats uniform."""
        cfg = _canonical_config(mode="tied", cavity=True)
        opt_config = {
            "target_wavelength_nm": 620.0,
            "target_q": 50.0,
            "spectrum": "R",
            "feature": "peak",
            "steps": 3,
            "learning_rate": 0.01,
            "lower_bound_nm": 0.0,
            "wavelength_weight": 1.0,
            "q_weight": 1.0,
            "sharpness": 20.0,
        }
        result = state.run_resonance_optimization(cfg, opt_config)
        # cavity-only: t[6] is the only free variable; no uniformity to check
        # but the run must succeed and return the required keys
        assert "thicknesses_nm" in result
        assert "resonance" in result
        assert "final_result" in result
        # variable_thicknesses_nm is ONE per group (cavity group only → 1 entry)
        assert len(result["variable_thicknesses_nm"]) == 1


# ===========================================================================
# Validation: empty selection and out-of-range
# ===========================================================================
class TestValidation:
    def test_tied_empty_selection_raises_italian(self):
        """Tied mode with nothing selected raises ValueError with Italian message (lang='it')."""
        cfg = _canonical_config(mode="tied", cavity=False, input_layers=[], output_layers=[])
        cfg["cavity"]["enabled"] = False
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        msg = str(exc_info.value)
        # Must contain Italian words
        assert any(word in msg for word in ("Selezionare", "selezionare", "variabile"))

    def test_independent_empty_flat_layers_raises_italian(self):
        """Independent mode with empty flat_layers raises ValueError with Italian message (lang='it')."""
        cfg = _canonical_config(mode="independent", flat_layers=[])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        msg = str(exc_info.value)
        assert any(word in msg for word in ("Selezionare", "selezionare", "variabile"))

    def test_independent_out_of_range_raises_italian(self):
        """Independent mode with out-of-range flat index raises ValueError (Italian, lang='it')."""
        cfg = _canonical_config(mode="independent", flat_layers=[99])
        with pytest.raises(ValueError) as exc_info:
            state.expand_optimization_variables(cfg, lang="it")
        msg = str(exc_info.value)
        assert "intervallo" in msg or "fuori" in msg

    def test_validate_opt_stack_config_ok(self):
        """Default canonical config passes validate_opt_stack_config."""
        cfg = _canonical_config()
        errors = state.validate_opt_stack_config(cfg)
        assert errors == []

    def test_validate_opt_stack_config_bad_mode_reports_error(self):
        """Invalid mode string surfaces as an Italian validation error."""
        cfg = _canonical_config()
        cfg["variable"]["mode"] = "magic"
        errors = state.validate_opt_stack_config(cfg)
        assert any("modalit" in e.lower() or "mode" in e.lower() or "valid" in e.lower()
                   for e in errors)

    def test_validate_independent_empty_flat_layers_reports_error(self):
        """Independent mode + empty flat_layers is caught by validate_opt_stack_config."""
        cfg = _canonical_config(mode="independent", flat_layers=[])
        errors = state.validate_opt_stack_config(cfg)
        assert len(errors) > 0

    def test_validate_independent_out_of_range_reports_error(self):
        """Independent mode + OOB flat index is caught by validate_opt_stack_config."""
        cfg = _canonical_config(mode="independent", flat_layers=[999])
        errors = state.validate_opt_stack_config(cfg)
        assert len(errors) > 0


# ===========================================================================
# plots.py — resonance_overlay_figure summary_in_legend
# ===========================================================================
class TestResonanceOverlaySummaryInLegend:
    def _result_and_resonance(self):
        from app import state as _st
        cfg = {
            "incident": _const(1.0, name="air"),
            "layers": [
                {"material": _const(2.35, name="TiO2"), "thickness_nm": 120.0},
                {"material": _const(1.46, name="SiO2"), "thickness_nm": 90.0},
            ],
            "substrate": _const(1.52, name="glass"),
            "grid": {"start_nm": 400.0, "stop_nm": 800.0, "num": 201},
            "angle_deg": 0.0,
            "polarization": "s",
        }
        result = _st.run_simulation(cfg)
        resonance = _st.analyze_result(result, channel="R", feature="peak")
        return result, resonance

    def test_summary_in_legend_has_br_and_lambda_and_q(self):
        """With summary_in_legend=True the resonance trace name contains <br>, λ and Q."""
        result, resonance = self._result_and_resonance()
        fig = plots.resonance_overlay_figure(result, resonance, channel="R",
                                             summary_in_legend=True)
        # find the resonance marker trace (the one that carries wavelength)
        res_names = [t.name for t in fig.data if t.name and "λ" in t.name]
        assert len(res_names) >= 1, (
            "expected at least one resonance trace with λ in name; "
            f"trace names: {[t.name for t in fig.data]}"
        )
        name = res_names[0]
        assert "<br>" in name, f"legend name missing <br>: {name!r}"
        assert "λ" in name, f"legend name missing λ: {name!r}"
        assert "Q" in name, f"legend name missing Q: {name!r}"

    def test_default_path_name_is_resonance(self):
        """Without summary_in_legend the default (EN) resonance trace name is 'Resonance'."""
        result, resonance = self._result_and_resonance()
        fig = plots.resonance_overlay_figure(result, resonance, channel="R")
        plain_names = [t.name for t in fig.data if t.name == "Resonance"]
        assert plain_names == ["Resonance"], (
            f"expected exactly one 'Resonance' trace; trace names: {[t.name for t in fig.data]}"
        )

    def test_default_path_name_is_risonanza_it(self):
        """With lang='it' the resonance trace name is exactly 'Risonanza'."""
        result, resonance = self._result_and_resonance()
        fig = plots.resonance_overlay_figure(result, resonance, channel="R", lang="it")
        plain_names = [t.name for t in fig.data if t.name == "Risonanza"]
        assert plain_names == ["Risonanza"], (
            f"expected exactly one 'Risonanza' trace; trace names: {[t.name for t in fig.data]}"
        )

    def test_figure_serializes(self):
        """Figure produced with summary_in_legend=True serializes via to_dict()."""
        result, resonance = self._result_and_resonance()
        fig = plots.resonance_overlay_figure(result, resonance, channel="R",
                                             summary_in_legend=True)
        d = fig.to_dict()
        assert isinstance(d, dict)
        assert "data" in d


# ===========================================================================
# plots.py — sketch_figure colorbar placement
# ===========================================================================
class TestSketchColorbarPlacement:
    def _simple_cfg(self):
        return {
            "incident": _const(1.0, name="air"),
            "substrate": _const(1.52, name="glass"),
            "layers": [
                {"thickness_nm": 120.0, "material": _const(2.35, name="TiO2")},
            ],
        }

    def test_sketch_builds_and_serializes(self):
        """sketch_figure builds without error and serializes via to_dict()."""
        cfg = self._simple_cfg()
        fig = plots.sketch_figure(cfg, grouped=False)
        assert isinstance(fig, go.Figure)
        d = fig.to_dict()
        assert isinstance(d, dict)

    def _colorbar_traces(self, fig):
        """Return traces that have showscale=True (the Re(n) colorbar trace)."""
        # Use the serialized dict form to reliably check showscale; the
        # go.Scatter object's .marker.colorbar.x may be None on non-colorbar
        # legend marker traces (the material swatch entries).
        return [
            t for t in fig.data
            if hasattr(t, "marker") and t.marker is not None
            and getattr(t.marker, "showscale", False) is True
        ]

    def test_colorbar_marker_trace_exists(self):
        """A colorbar marker trace (showscale=True) is present in the figure's data."""
        cfg = self._simple_cfg()
        fig = plots.sketch_figure(cfg, grouped=False)
        colorbar_traces = self._colorbar_traces(fig)
        assert len(colorbar_traces) >= 1, (
            "no colorbar marker trace (showscale=True) found in sketch figure; "
            f"traces: {[(type(t).__name__, getattr(t, 'name', None)) for t in fig.data]}"
        )

    def test_colorbar_x_is_negative(self):
        """Colorbar moved left (x < 0) per §11 sketch requirement."""
        cfg = self._simple_cfg()
        fig = plots.sketch_figure(cfg, grouped=False)
        colorbar_traces = self._colorbar_traces(fig)
        assert len(colorbar_traces) >= 1
        cb_x = colorbar_traces[0].marker.colorbar.x
        assert cb_x is not None, "colorbar.x is None — value was never set"
        assert cb_x < 0, (
            f"colorbar.x={cb_x} is not negative — it should be moved left "
            "to avoid overlapping the right-side materials legend"
        )

    def test_colorbar_x_exact_value(self):
        """Colorbar x is -0.12 per the sketch implementation."""
        cfg = self._simple_cfg()
        fig = plots.sketch_figure(cfg, grouped=False)
        colorbar_traces = self._colorbar_traces(fig)
        assert len(colorbar_traces) >= 1
        cb_x = colorbar_traces[0].marker.colorbar.x
        assert cb_x is not None, "colorbar.x is None — value was never set"
        assert cb_x == pytest.approx(-0.12, abs=0.01), (
            f"colorbar.x={cb_x}, expected -0.12"
        )

    def test_grouped_sketch_builds(self):
        """Grouped sketch builds and serializes for the canonical opt config."""
        cfg = _canonical_config()
        fig = plots.sketch_figure(cfg, grouped=True)
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.to_dict(), dict)


# ===========================================================================
# UI smoke: app.main.create_app() structure
# ===========================================================================
class TestAppSmokeExtended:
    def _app(self):
        import app.main as main
        return main.create_app()

    def test_create_app_constructs(self):
        """app.main.create_app() returns a non-None Dash object."""
        app_obj = self._app()
        assert app_obj is not None

    def test_callback_map_has_flat_layer_options_output(self):
        """callback_map contains an output for opt_variable_flat_layers_input.options."""
        app_obj = self._app()
        # Dash stores callback outputs keyed by a string like
        # "opt_variable_flat_layers_input.options"
        target_key = f"{ids.OPT_VARIABLE_FLAT_LAYERS_INPUT}.options"
        assert target_key in app_obj.callback_map, (
            f"callback_map missing key {target_key!r}; "
            f"available keys: {sorted(app_obj.callback_map.keys())}"
        )

    def test_layout_contains_opt_variable_mode_tabs(self):
        """Rendered layout (via build_layout) contains the OPT_VARIABLE_MODE_TABS component id.

        app.layout is now a function (serve_layout, §12 D4), so str(app.layout) returns
        the function repr — not the component tree. Use build_layout() to render the tree.
        """
        from app.layout import build_layout
        layout_str = str(build_layout("en"))
        assert ids.OPT_VARIABLE_MODE_TABS in layout_str, (
            f"rendered layout does not contain '{ids.OPT_VARIABLE_MODE_TABS}'"
        )

    def test_layout_contains_opt_variable_flat_layers_input(self):
        """Rendered layout (via build_layout) contains OPT_VARIABLE_FLAT_LAYERS_INPUT id.

        app.layout is now a function (serve_layout, §12 D4); use build_layout() to render.
        """
        from app.layout import build_layout
        layout_str = str(build_layout("en"))
        assert ids.OPT_VARIABLE_FLAT_LAYERS_INPUT in layout_str, (
            f"rendered layout does not contain '{ids.OPT_VARIABLE_FLAT_LAYERS_INPUT}'"
        )

    def test_server_attribute_present(self):
        """app.server is present (needed for gunicorn app.main:server)."""
        app_obj = self._app()
        assert app_obj.server is not None


# ===========================================================================
# ids.py — new §11.2 ids are defined
# ===========================================================================
class TestIdsExist:
    def test_opt_variable_mode_tabs_id_exists(self):
        assert hasattr(ids, "OPT_VARIABLE_MODE_TABS")
        assert ids.OPT_VARIABLE_MODE_TABS == "opt_variable_mode_tabs"

    def test_opt_variable_mode_tied_id_exists(self):
        assert hasattr(ids, "OPT_VARIABLE_MODE_TIED")
        assert ids.OPT_VARIABLE_MODE_TIED == "tied"

    def test_opt_variable_mode_independent_id_exists(self):
        assert hasattr(ids, "OPT_VARIABLE_MODE_INDEPENDENT")
        assert ids.OPT_VARIABLE_MODE_INDEPENDENT == "independent"

    def test_opt_variable_flat_layers_input_id_exists(self):
        assert hasattr(ids, "OPT_VARIABLE_FLAT_LAYERS_INPUT")
        assert ids.OPT_VARIABLE_FLAT_LAYERS_INPUT == "opt_variable_flat_layers_input"


# ===========================================================================
# config.py — §11.3 labels and default_opt_stack_config variable schema
# ===========================================================================
class TestConfigLabels:
    def test_mode_tied_label_exists(self):
        assert "opt_variable_mode_tied" in config.LABELS
        assert config.LABELS["opt_variable_mode_tied"]  # non-empty

    def test_mode_independent_label_exists(self):
        assert "opt_variable_mode_independent" in config.LABELS
        assert config.LABELS["opt_variable_mode_independent"]

    def test_flat_layers_label_exists(self):
        assert "opt_variable_flat_layers" in config.LABELS
        assert config.LABELS["opt_variable_flat_layers"]

    def test_default_opt_stack_config_has_mode(self):
        cfg = config.default_opt_stack_config()
        assert "mode" in cfg["variable"]
        assert cfg["variable"]["mode"] == "tied"

    def test_default_opt_stack_config_has_flat_layers(self):
        cfg = config.default_opt_stack_config()
        assert "flat_layers" in cfg["variable"]
        assert cfg["variable"]["flat_layers"] == []
