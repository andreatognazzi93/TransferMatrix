"""Workflow-1 callbacks: simulate spectrum and render it.

Thin adapter (ARCHITECTURE §1, §6.1, §6.2): on "Simula", read
``stack_config_store`` -> ``state.run_simulation`` -> write
``simulation_result_store`` -> build a figure via ``plots.spectrum_figure`` and
(when a single polarization) a resonance readout via ``state.analyze_result``.
"""

from __future__ import annotations

from dash import Input, Output, State, dcc, no_update

from app import config, ids, plots, state


def register(app) -> None:
    """Register the simulate callbacks."""

    @app.callback(
        Output(ids.SIMULATION_RESULT_STORE, "data"),
        Output(ids.SIMULATE_STATUS, "children"),
        Input(ids.SIMULATE_BUTTON, "n_clicks"),
        State(ids.STACK_CONFIG_STORE, "data"),
        # Angle-sweep map widgets (ANGLE_MAP_CONTRACT §8.1). The flat-stack sync
        # callback (stack_callbacks.sync_stack_config) does NOT write sim_mode /
        # angle_sweep into STACK_CONFIG_STORE, so we read those widgets directly
        # here and merge them into the config before dispatching. This keeps the
        # plumbing entirely within gui-core's owned files.
        State(ids.SIMULATE_MODE_INPUT, "value"),
        State(ids.SIMULATE_ANGLE_START_INPUT, "value"),
        State(ids.SIMULATE_ANGLE_STOP_INPUT, "value"),
        State(ids.SIMULATE_ANGLE_STEP_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def run_simulation(
        n_clicks,
        stack_config,
        sim_mode,
        angle_start,
        angle_stop,
        angle_step,
        lang,
    ):
        """Run the simulation (single or angle map) and store the result dict."""

        if not n_clicks or not stack_config:
            return no_update, no_update
        lang = lang or config.DEFAULT_LANG

        # Merge the mode + sweep widgets into the config so the §5.2 branch in
        # state.run_simulation fires. Fall back to the config defaults / "single".
        config_dict = dict(stack_config)
        config_dict["sim_mode"] = sim_mode or config_dict.get("sim_mode", "single")
        if config_dict["sim_mode"] == "angle_map":
            base_sweep = config_dict.get("angle_sweep") or config.default_angle_sweep()
            sweep = dict(base_sweep)
            if angle_start is not None:
                sweep["start_deg"] = angle_start
            if angle_stop is not None:
                sweep["stop_deg"] = angle_stop
            if angle_step is not None:
                sweep["step_deg"] = angle_step
            config_dict["angle_sweep"] = sweep

        try:
            result_dict = state.run_simulation(config_dict, lang=lang)
        except ValueError as exc:
            return no_update, str(exc)
        return result_dict, config.labels_for(lang)["simulate_status_done"]

    @app.callback(
        Output(ids.SIMULATE_GRAPH, "figure"),
        Input(ids.SIMULATION_RESULT_STORE, "data"),
        Input(ids.SIMULATE_CHANNELS_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def render_spectrum(result_dict, channels, lang):
        """Build the spectrum / angle-map figure from the stored result.

        Branches on ``result_dict["mode"]`` (ANGLE_MAP_CONTRACT §8.2): an
        ``"angle_map"`` result renders stacked heatmaps via
        ``plots.angle_map_figure``; ``"single"`` (or a missing ``mode``, treated
        as single for backward compatibility) renders line spectra.
        """

        lang = lang or config.DEFAULT_LANG
        if not result_dict:
            return plots.empty_figure(config.labels_for(lang)["empty_plot"], lang=lang)
        selected = tuple(channels) if channels else ("R", "T", "A")
        if result_dict.get("mode") == "angle_map":
            return plots.angle_map_figure(result_dict, channels=selected, lang=lang)
        return plots.spectrum_figure(result_dict, channels=selected, lang=lang)

    @app.callback(
        Output(ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_RESONANCE_TABLE_SUFFIX), "data"),
        Input(ids.SIMULATION_RESULT_STORE, "data"),
        State(ids.SIMULATE_CHANNELS_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def render_resonance_readout(result_dict, channels, lang):
        """Compute the resonance readout for a single-polarization result.

        Returns rows for a small key/value DataTable. For a 2-row ("both")
        result, ``state.analyze_result`` raises and a single explanatory row is
        returned instead. Row keys + warning label localized via the language store.
        """

        if not result_dict:
            return []
        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        # Angle-sweep map has no single resonance value (ANGLE_MAP_CONTRACT §8.3).
        if result_dict.get("mode") == "angle_map":
            return [{"grandezza": labels["resonance"], "valore": labels["res_na_angle_map"]}]
        channel = (channels[0] if channels else "R")
        try:
            resonance = state.analyze_result(
                result_dict, channel=channel, feature="peak", lang=lang
            )
        except ValueError as exc:
            return [{"grandezza": labels["res_table_warning"], "valore": str(exc)}]
        return [
            {"grandezza": labels["resonance_wavelength"],
             "valore": round(resonance["resonance_wavelength_nm"], 3)},
            {"grandezza": labels["linewidth"],
             "valore": round(resonance["linewidth_nm"], 3)},
            {"grandezza": labels["quality_factor"],
             "valore": round(resonance["quality_factor"], 3)},
            {"grandezza": labels["extremum_value"],
             "valore": round(resonance["extremum_value"], 5)},
        ]

    @app.callback(
        Output(ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_DOWNLOAD_SUFFIX), "data"),
        Output(ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_STATUS_SUFFIX), "children"),
        Input(ids.results_id(ids.SIMULATE_RESULTS_PREFIX, ids.RESULTS_EXPORT_BUTTON_SUFFIX), "n_clicks"),
        State(ids.SIMULATION_RESULT_STORE, "data"),
        State(ids.STACK_CONFIG_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def export_simulation(n_clicks, result_dict, stack_config, lang):
        """Save the simulated spectra + parameters as one ZIP of two .txt files.

        One click downloads ``<prefix>.zip`` containing ``<prefix>_spectra.txt``
        (wavelength/R/T/A) and ``<prefix>_parameters.txt`` (grid + structure),
        with ``<prefix> = simulated_<YYYYMMDD_HHMMSS>``. A single ZIP is used
        because browsers drop the 2nd of two simultaneous downloads. Requires a
        completed simulation; otherwise it shows a localized "run first" message.
        """

        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        if not n_clicks:
            return no_update, no_update
        if not result_dict or "wavelength_nm" not in result_dict:
            return no_update, labels["simulate_export_empty"]

        file_prefix, timestamp = state.make_export_prefix(simulation_name="simulated")
        spectra_text = state.build_spectra_text(
            result_dict, file_prefix=file_prefix, timestamp=timestamp,
            simulation_name="simulated",
        )
        params_text = state.build_simulation_parameters_text(
            stack_config or {}, result_dict,
            file_prefix=file_prefix, timestamp=timestamp, simulation_name="simulated",
        )
        zip_bytes = state.build_export_zip_bytes(
            file_prefix, spectra_text=spectra_text, params_text=params_text
        )
        return (
            dcc.send_bytes(zip_bytes, f"{file_prefix}.zip"),
            labels["export_done"].format(prefix=file_prefix),
        )

    # --- Angle-sweep map mode toggle (ANGLE_MAP_CONTRACT §8.4) ---------------
    @app.callback(
        Output(ids.SIMULATE_SINGLE_ANGLE_CONTAINER, "style"),
        Output(ids.SIMULATE_ANGLE_SWEEP_CONTAINER, "style"),
        Input(ids.SIMULATE_MODE_INPUT, "value"),
    )
    def toggle_sweep_inputs(sim_mode):
        """Show the single-angle inputs OR the angle-sweep inputs by mode.

        ``"angle_map"`` reveals the sweep container and hides the single-angle
        container; any other value (``"single"`` / ``None``) does the reverse.
        """

        if sim_mode == "angle_map":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output(ids.POLARIZATION_INPUT, "options"),
        Output(ids.POLARIZATION_INPUT, "value"),
        Input(ids.SIMULATE_MODE_INPUT, "value"),
        State(ids.POLARIZATION_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
    )
    def disable_both_for_angle_map(sim_mode, current_value, lang):
        """Disable the ``both`` polarization option in angle_map mode (§8.4).

        In ``"angle_map"`` mode the ``both`` entry is marked disabled and the
        value is coerced off ``both`` to ``"s"``. In ``"single"`` mode the full
        enabled option list is restored and the value is left untouched.
        """

        lang = lang or config.DEFAULT_LANG
        options = config.options_for(config.POLARIZATION_VALUES, "pol_", lang)
        if sim_mode == "angle_map":
            for option in options:
                if option["value"] == "both":
                    option["disabled"] = True
            value = "s" if current_value == "both" else no_update
            return options, value
        return options, no_update
