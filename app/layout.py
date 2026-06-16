"""Top-level page skeleton (ARCHITECTURE §6.4, §12).

Declares all ``dcc.Store`` components (§2.4), embeds the **single** shared
stack-builder, and lays out the two tabs ("Simulazione" / "Ottimizzazione").
The stack-builder is instantiated once and placed above the tabs so both
workflows feed the one ``stack_config_store`` — the single-stack-builder
requirement. Component ids are unique app-wide, so the editor cannot be
duplicated per tab.

i18n (§12): ``build_layout(lang)`` threads the active language code into every
component builder, seeds ``LANGUAGE_STORE`` so callbacks can localize their
output via ``State(LANGUAGE_STORE, "data")``, and places the header language
selector. The default and canonical language is English.

Contains no business logic; all wiring is in :mod:`app.callbacks` (plus the
material-upload + layer-dropdown helpers and the clientside language toggle
registered in :mod:`app.main`).
"""

from __future__ import annotations

from dash import dcc, html

from app import config, ids
from app.components import (
    build_optimize_panel,
    build_results_panel,
    build_simulate_panel,
    build_stack_builder,
)
from app.components.header import build_language_selector


def _stores(lang: str = "en"):
    """All dcc.Store components for the app-state model (ARCHITECTURE §2.4).

    Adds ``LANGUAGE_STORE`` (seeded with the active language) so every callback
    can read the current language via ``State(LANGUAGE_STORE, "data")``.
    """

    return [
        # §12.4 active language code — read by callbacks to localize output.
        dcc.Store(id=ids.LANGUAGE_STORE, data=lang),
        # §2.2 flat stack dict — source of truth for the Simulazione tab.
        dcc.Store(id=ids.STACK_CONFIG_STORE, data=config.default_stack_config()),
        # §9.1 grouped/cavity stack dict — source of truth for the
        # Ottimizzazione tab's structure (mirror groups, cavity, grid, angle,
        # polarization, and the `variable` selector). Distinct from the flat
        # STACK_CONFIG_STORE (the two tabs own separate stack stores).
        dcc.Store(id=ids.OPT_STACK_CONFIG_STORE, data=config.default_opt_stack_config()),
        dcc.Store(id=ids.OPTIMIZE_CONFIG_STORE, data=config.default_optimize_config()),
        dcc.Store(id=ids.SIMULATION_RESULT_STORE),
        dcc.Store(id=ids.OPTIMIZATION_RESULT_STORE),
        dcc.Store(id=ids.OPTIMIZATION_PROGRESS_STORE),
        # Uploaded-CSV materials keyed by name; populated by the upload callback
        # in app.main and read by stack_callbacks.sync_stack_config.
        dcc.Store(id=ids.MATERIAL_LIBRARY_STORE, data={}),
    ]


def build_layout(lang: str = "en"):
    """Build the full page layout (ARCHITECTURE §6.4, §12).

    Args:
        lang: active UI language code (``"en"`` or ``"it"``). Threaded into every
            component builder and seeded into ``LANGUAGE_STORE``. Default English.
    """

    labels = config.labels_for(lang)
    return html.Div(
        className="app-root",
        children=[
            *_stores(lang),
            # Hidden output target for the clientside language-toggle callback
            # (main.py). It must NOT output back to LANGUAGE_SELECTOR.value
            # (circular) — the dummy children absorb the no-op return.
            html.Div(id="lang-reload-dummy", style={"display": "none"}),
            html.Div(
                className="app-header",
                children=[
                    html.H1(labels["app_title"]),
                    build_language_selector(lang),
                ],
            ),
            dcc.Tabs(
                id=ids.APP_TABS,
                value=ids.TAB_SIMULATE,
                children=[
                    dcc.Tab(
                        label=labels["tab_simulate"],
                        value=ids.TAB_SIMULATE,
                        children=[
                            # Flat stack-builder feeding STACK_CONFIG_STORE. Lives
                            # in the Simulazione tab only; its ids are unique
                            # app-wide so it cannot be duplicated. The
                            # Ottimizzazione tab owns its own grouped editor
                            # (§9.4) feeding OPT_STACK_CONFIG_STORE.
                            build_stack_builder(lang),
                            build_simulate_panel(lang),
                            build_results_panel(
                                ids.SIMULATE_RESULTS_PREFIX,
                                graph_id=ids.SIMULATE_GRAPH,
                                lang=lang,
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label=labels["tab_optimize"],
                        value=ids.TAB_OPTIMIZE,
                        children=[build_optimize_panel(lang)],
                    ),
                ],
            ),
        ],
    )
