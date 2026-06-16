"""Workflow-2 callbacks: long-running optimization as a Dash background callback.

ARCHITECTURE §4: the "Ottimizza" handler is registered with ``background=True``
and the app's ``DiskcacheManager`` (constructed in ``app.main`` and passed via
``register_callbacks``). The library runs its full ``steps`` loop internally and
exposes **no per-step hook**, so progress is necessarily coarse — the callback
reports an indeterminate "in corso" status while the loop runs, then writes the
final result. The loss-per-step curve comes from ``OptimizationResult.history``
after completion (plotted by ``plots.history_figure``).

Thin adapter (ARCHITECTURE §9.4): read ``opt_stack_config_store`` (the grouped/
cavity stack structure, §9.1) + ``optimize_config_store`` (the scalar workflow-2
inputs, §2.3) -> ``state.run_resonance_optimization`` /
``state.run_thickness_optimization`` -> write ``optimization_result_store`` ->
render history / overlay figures. The grouped store carries grid/angle/
polarization and the ``variable`` selector; the scalar config carries only
target wl/Q, feature, spectrum, steps, lr, bounds, weights, sharpness.
"""

from __future__ import annotations

from dash import Input, Output, State, dcc, no_update

from app import config, ids, plots, state


def _collect_opt_config(
    mode,
    spectrum,
    feature,
    target_wavelength,
    target_q,
    steps,
    learning_rate,
    lower_bound,
    wavelength_weight,
    q_weight,
    sharpness,
) -> dict:
    """Assemble the §2.3 scalar optimize-config dict from the panel widgets.

    The variable-layer selection no longer lives here (§9): it is part of the
    grouped ``opt_stack_config_store["variable"]`` selector.
    """

    cfg = config.default_optimize_config()
    cfg.update(
        {
            "mode": mode or "resonance",
            "spectrum": spectrum or "R",
            "feature": feature or "peak",
            "target_wavelength_nm": target_wavelength,
            "target_q": target_q,
            "steps": steps,
            "learning_rate": learning_rate,
            "lower_bound_nm": lower_bound,
            "wavelength_weight": wavelength_weight,
            "q_weight": q_weight,
            "sharpness": sharpness,
        }
    )
    return cfg


def register(app, cache) -> None:
    """Register the optimize callbacks, including the background run callback.

    Args:
        app: the ``dash.Dash`` instance.
        cache: the ``DiskcacheManager`` for the background callback (§4).
    """

    @app.callback(
        Output(ids.OPTIMIZATION_RESULT_STORE, "data"),
        Output(ids.OPTIMIZE_STATUS, "children"),
        Input(ids.OPTIMIZE_BUTTON, "n_clicks"),
        State(ids.OPT_STACK_CONFIG_STORE, "data"),
        State(ids.OPTIMIZE_MODE_INPUT, "value"),
        State(ids.OPTIMIZE_SPECTRUM_INPUT, "value"),
        State(ids.OPTIMIZE_FEATURE_INPUT, "value"),
        State(ids.OPTIMIZE_TARGET_WAVELENGTH_INPUT, "value"),
        State(ids.OPTIMIZE_TARGET_Q_INPUT, "value"),
        State(ids.OPTIMIZE_STEPS_INPUT, "value"),
        State(ids.OPTIMIZE_LEARNING_RATE_INPUT, "value"),
        State(ids.OPTIMIZE_LOWER_BOUND_INPUT, "value"),
        State(ids.OPTIMIZE_WAVELENGTH_WEIGHT_INPUT, "value"),
        State(ids.OPTIMIZE_Q_WEIGHT_INPUT, "value"),
        State(ids.OPTIMIZE_SHARPNESS_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        background=True,
        manager=cache,
        running=[
            (Output(ids.OPTIMIZE_BUTTON, "disabled"), True, False),
            # ARCHITECTURE §12.5 caveat: the two running-status strings are baked
            # in at registration time (before any request) and cannot read
            # LANGUAGE_STORE, so the transient "running…" toast is fixed at the
            # DEFAULT_LANG (English). The SETTLED status returned by the body
            # below IS localized and overrides this transient.
            (Output(ids.OPTIMIZE_STATUS, "children"),
             config.labels_for(config.DEFAULT_LANG)["optimize_status_running"],
             config.labels_for(config.DEFAULT_LANG)["optimize_status_done"]),
        ],
        progress=[Output(ids.OPTIMIZATION_PROGRESS_STORE, "data")],
        prevent_initial_call=True,
    )
    def run_optimization(
        set_progress,
        n_clicks,
        opt_stack_config,
        mode,
        spectrum,
        feature,
        target_wavelength,
        target_q,
        steps,
        learning_rate,
        lower_bound,
        wavelength_weight,
        q_weight,
        sharpness,
        lang,
    ):
        """Run the chosen optimization and store the JSON-safe result dict.

        Reads the grouped/cavity stack store (§9.1); ``state.run_*`` expand it
        internally. Coarse progress only (library has no step hook): emit a
        single "started" progress payload, then return the completed result.
        The settled status/error text is localized via ``LANGUAGE_STORE`` (the
        transient running toast stays English — §12.5 caveat).
        """

        if not n_clicks or not opt_stack_config:
            return no_update, no_update
        lang = lang or config.DEFAULT_LANG

        # Coarse, indeterminate progress (ARCHITECTURE §4).
        set_progress([{"status": "running", "step": 0, "total": steps}])

        opt_config = _collect_opt_config(
            mode, spectrum, feature, target_wavelength, target_q,
            steps, learning_rate, lower_bound, wavelength_weight, q_weight, sharpness,
        )
        try:
            if opt_config["mode"] == "mean_r":
                result_dict = state.run_thickness_optimization(
                    opt_stack_config, opt_config, lang=lang
                )
            else:
                result_dict = state.run_resonance_optimization(
                    opt_stack_config, opt_config, lang=lang
                )
        except ValueError as exc:
            return no_update, str(exc)

        return result_dict, config.labels_for(lang)["optimize_status_done"]

    @app.callback(
        Output(ids.OPTIMIZE_HISTORY_GRAPH, "figure"),
        Input(ids.OPTIMIZATION_RESULT_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def render_history(result_dict, lang):
        """Plot the loss-per-step curve from ``OptimizationResult.history``."""

        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        if not result_dict or "history" not in result_dict:
            return plots.empty_figure(labels["empty_plot"], lang=lang)
        return plots.history_figure(
            result_dict["history"], title=labels["optimize_history"], lang=lang
        )

    @app.callback(
        Output(ids.OPTIMIZE_RESULT_GRAPH, "figure"),
        Input(ids.OPTIMIZATION_RESULT_STORE, "data"),
        State(ids.OPTIMIZE_SPECTRUM_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def render_final_spectrum(result_dict, spectrum, lang):
        """Render the optimized spectrum; overlay resonance markers when present."""

        lang = lang or config.DEFAULT_LANG
        if not result_dict or "final_result" not in result_dict:
            return plots.empty_figure(config.labels_for(lang)["empty_plot"], lang=lang)
        channel = spectrum or "R"
        resonance = result_dict.get("resonance")
        if resonance:
            # Optimized structure: show achieved λ_res and Q in the legend.
            return plots.resonance_overlay_figure(
                result_dict["final_result"],
                resonance,
                channel=channel,
                summary_in_legend=True,
                lang=lang,
            )
        return plots.spectrum_figure(
            result_dict["final_result"], channels=(channel,), lang=lang
        )

    @app.callback(
        Output(ids.OPTIMIZE_THICKNESS_READOUT, "children"),
        Input(ids.OPTIMIZATION_RESULT_STORE, "data"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def render_thicknesses(result_dict, lang):
        """Render the optimized layer thicknesses as a short text readout."""

        if not result_dict or "thicknesses_nm" not in result_dict:
            return ""
        lang = lang or config.DEFAULT_LANG
        thicknesses = result_dict["thicknesses_nm"]
        formatted = ", ".join(f"{value:.2f}" for value in thicknesses)
        return f"{config.labels_for(lang)['optimize_thicknesses']}: [{formatted}]"

    @app.callback(
        Output(ids.OPTIMIZE_EXPORT_DOWNLOAD, "data"),
        Output(ids.OPTIMIZE_EXPORT_STATUS, "children"),
        Input(ids.OPTIMIZE_EXPORT_BUTTON, "n_clicks"),
        State(ids.OPTIMIZATION_RESULT_STORE, "data"),
        State(ids.OPT_STACK_CONFIG_STORE, "data"),
        State(ids.OPTIMIZE_MODE_INPUT, "value"),
        State(ids.OPTIMIZE_SPECTRUM_INPUT, "value"),
        State(ids.OPTIMIZE_FEATURE_INPUT, "value"),
        State(ids.OPTIMIZE_TARGET_WAVELENGTH_INPUT, "value"),
        State(ids.OPTIMIZE_TARGET_Q_INPUT, "value"),
        State(ids.OPTIMIZE_STEPS_INPUT, "value"),
        State(ids.OPTIMIZE_LEARNING_RATE_INPUT, "value"),
        State(ids.OPTIMIZE_LOWER_BOUND_INPUT, "value"),
        State(ids.OPTIMIZE_WAVELENGTH_WEIGHT_INPUT, "value"),
        State(ids.OPTIMIZE_Q_WEIGHT_INPUT, "value"),
        State(ids.OPTIMIZE_SHARPNESS_INPUT, "value"),
        State(ids.LANGUAGE_STORE, "data"),
        prevent_initial_call=True,
    )
    def export_optimization(
        n_clicks,
        result_dict,
        opt_stack_config,
        mode,
        spectrum,
        feature,
        target_wavelength,
        target_q,
        steps,
        learning_rate,
        lower_bound,
        wavelength_weight,
        q_weight,
        sharpness,
        lang,
    ):
        """Save the optimized spectra + parameters as one ZIP of two .txt files.

        One click downloads ``<prefix>.zip`` containing ``<prefix>_spectra.txt``
        (wavelength/R/T/A) and ``<prefix>_parameters.txt`` (settings + structure
        + result), sharing the prefix ``simulation_<YYYYMMDD_HHMMSS>``. A ZIP is
        used because browsers drop the 2nd of two simultaneous downloads, which
        previously lost the parameters file. Requires a completed run; otherwise
        it reports a localized "run first" message and emits nothing.
        """

        lang = lang or config.DEFAULT_LANG
        labels = config.labels_for(lang)
        if not n_clicks:
            return no_update, no_update
        if not result_dict or "final_result" not in result_dict:
            return no_update, labels["optimize_export_empty"]

        opt_config = _collect_opt_config(
            mode, spectrum, feature, target_wavelength, target_q,
            steps, learning_rate, lower_bound, wavelength_weight, q_weight, sharpness,
        )
        file_prefix, timestamp = state.make_export_prefix(simulation_name="optimized")
        spectra_text = state.build_optimized_spectra_text(
            result_dict, file_prefix=file_prefix, timestamp=timestamp,
            simulation_name="optimized",
        )
        params_text = state.build_optimization_parameters_text(
            opt_stack_config or {}, opt_config, result_dict,
            file_prefix=file_prefix, timestamp=timestamp, simulation_name="optimized",
        )
        zip_bytes = state.build_export_zip_bytes(
            file_prefix, spectra_text=spectra_text, params_text=params_text
        )
        return (
            dcc.send_bytes(zip_bytes, f"{file_prefix}.zip"),
            labels["export_done"].format(prefix=file_prefix),
        )
