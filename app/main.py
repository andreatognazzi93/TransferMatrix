"""App factory + entrypoint (ARCHITECTURE §6.3).

Builds the Dash app with a :class:`~dash.DiskcacheManager` background-callback
manager (§4), installs the layout, and registers the gui-core callbacks via
``register_callbacks(app, cache)``. It additionally registers the two
frontend-owned wiring callbacks that gui-core's submodules do not cover:

* CSV material uploads -> ``material_library_store`` (keyed by material name),
* propagating the library names into the layer-table ``csv_name`` dropdown.

Run with ``python -m app`` / ``python app/main.py`` (dev server) or
``gunicorn app.main:server`` (WSGI).
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import dash
import flask
from dash import Input, Output, State, html, no_update
from dash import DiskcacheManager
from diskcache import Cache

from app import config, ids, state
from app.callbacks import register_callbacks
from app.layout import build_layout


def _register_material_uploads(app) -> None:
    """Wire each material CSV upload into ``material_library_store``.

    Parses the base64 upload in-memory via ``state.parse_material_csv`` and
    stores the resulting csv-material dict keyed by its name. A per-input status
    line reports success or the localized validation error (language read from
    ``LANGUAGE_STORE``; default English).
    """

    prefixes = (
        ids.INCIDENT_MATERIAL_PREFIX,
        ids.SUBSTRATE_MATERIAL_PREFIX,
        # grouped/cavity tab material inputs (§9.4)
        ids.OPT_INCIDENT_MATERIAL_PREFIX,
        ids.OPT_SUBSTRATE_MATERIAL_PREFIX,
        ids.OPT_CAVITY_MATERIAL_PREFIX,
    )
    for index, prefix in enumerate(prefixes):
        upload_id = ids.material_id(prefix, ids.MATERIAL_UPLOAD_SUFFIX)
        status_id = ids.material_id(prefix, ids.MATERIAL_UPLOAD_STATUS_SUFFIX)

        @app.callback(
            Output(ids.MATERIAL_LIBRARY_STORE, "data", allow_duplicate=True),
            Output(status_id, "children"),
            Input(upload_id, "contents"),
            State(upload_id, "filename"),
            State(ids.MATERIAL_LIBRARY_STORE, "data"),
            State(ids.LANGUAGE_STORE, "data"),
            prevent_initial_call=True,
        )
        def store_uploaded_material(contents, filename, library, lang):
            if not contents:
                return no_update, no_update
            lang = lang or config.DEFAULT_LANG
            try:
                material = state.parse_material_csv(contents, filename=filename, lang=lang)
            except ValueError as exc:
                return no_update, str(exc)
            library = dict(library or {})
            name = material.get("name") or filename or f"material_{len(library)}"
            material["name"] = name
            library[name] = material
            return library, f"{config.labels_for(lang)['upload_csv']}: {name}"


def _register_layer_dropdown_options(app) -> None:
    """Refresh each layer-table ``csv_name`` dropdown from the material library.

    Covers the flat finite-layer table plus the two grouped period-definition
    tables on the Ottimizzazione tab (§9.4).
    """

    tables = (ids.LAYER_TABLE, ids.OPT_INPUT_GROUP_TABLE, ids.OPT_OUTPUT_GROUP_TABLE)
    for table_id in tables:

        @app.callback(
            Output(table_id, "dropdown"),
            Input(ids.MATERIAL_LIBRARY_STORE, "data"),
            State(table_id, "dropdown"),
            prevent_initial_call=True,
        )
        def update_csv_name_options(library, dropdown):
            dropdown = dict(dropdown or {})
            names = sorted((library or {}).keys())
            dropdown[ids.LAYER_COL_CSV_NAME] = {
                "options": [{"label": n, "value": n} for n in names]
            }
            return dropdown


def _register_language_toggle(app) -> None:
    """Clientside callback: reload with ``?lang=<value>`` on selector change.

    The layout is a function (``serve_layout``) that reads ``?lang`` per request,
    so switching language is a full page reload. The JS compares the selected
    value to the CURRENT ``?lang`` and only navigates if they differ (avoids a
    reload loop). The output is a DUMMY hidden div — NOT ``LANGUAGE_SELECTOR``
    (that would be circular). ``prevent_initial_call=True`` keeps the first
    server-rendered load from triggering a navigation.
    """

    app.clientside_callback(
        """
        function(v) {
            if (!v) { return window.dash_clientside.no_update; }
            var u = new URL(window.location);
            if (u.searchParams.get('lang') !== v) {
                u.searchParams.set('lang', v);
                window.location.assign(u.toString());
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("lang-reload-dummy", "children"),
        Input(ids.LANGUAGE_SELECTOR, "value"),
        prevent_initial_call=True,
    )


def _lang_from_request() -> str:
    """Resolve the UI language for the current request.

    Dash fetches the layout via a separate ``/_dash-layout`` request that does
    NOT carry the page's ``?lang`` query string, so ``flask.request.args`` is
    usually empty inside :func:`serve_layout`. The page URL is available as the
    request *referrer*, so we read ``?lang`` from the args first (covers the
    index ``GET /?lang=..`` request) and fall back to the referrer's query
    string (covers the ``/_dash-layout`` fetch). Unknown / missing languages
    fall back to ``config.DEFAULT_LANG`` (English).
    """

    lang = flask.request.args.get("lang")
    if not lang and flask.request.referrer:
        referrer_qs = parse_qs(urlparse(flask.request.referrer).query)
        values = referrer_qs.get("lang")
        if values:
            lang = values[0]
    if lang not in config.SUPPORTED_LANGS:
        lang = config.DEFAULT_LANG
    return lang


def serve_layout():
    """Per-request layout: read the language (via :func:`_lang_from_request`)
    and build the tree.

    Dash calls this on every page load (``app.layout`` is assigned the function
    object, not its value), so the URL query parameter selects the language.
    """

    return build_layout(_lang_from_request())


def create_app() -> dash.Dash:
    """Construct the configured Dash app (ARCHITECTURE §6.3, §12)."""

    cache = DiskcacheManager(Cache(config.CACHE_DIR))
    app = dash.Dash(
        __name__,
        title=config.labels_for(config.DEFAULT_LANG)["app_title"],
        background_callback_manager=cache,
        suppress_callback_exceptions=True,
    )
    # Assign the FUNCTION (not its value) so the language is read per request.
    app.layout = serve_layout

    # gui-core callbacks (stack sync, simulate, background optimize).
    register_callbacks(app, cache)
    # frontend-owned wiring (uploads + layer dropdown options).
    _register_material_uploads(app)
    _register_layer_dropdown_options(app)
    # clientside language toggle (full reload via ?lang=).
    _register_language_toggle(app)
    return app


app = create_app()
server = app.server


def main() -> None:
    """Run the Dash development server."""

    app.run(debug=True)


if __name__ == "__main__":
    main()
