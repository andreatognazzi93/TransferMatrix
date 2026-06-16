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
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, stack_config, lang):
        """Run the simulation and store the JSON-safe result dict."""

        if not n_clicks or not stack_config:
            return no_update, no_update
        lang = lang or config.DEFAULT_LANG
        try:
            result_dict = state.run_simulation(stack_config, lang=lang)
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
        """Build the spectrum figure from the stored result."""

        lang = lang or config.DEFAULT_LANG
        if not result_dict:
            return plots.empty_figure(config.labels_for(lang)["empty_plot"], lang=lang)
        selected = tuple(channels) if channels else ("R", "T", "A")
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
