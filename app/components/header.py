"""App header component: title + language selector (ARCHITECTURE §12.4).

The language selector is a small ``dcc.RadioItems`` placed beside the app
title. Switching language triggers a full-page reload with ``?lang=<code>``
(clientside callback wired by gui-core in ``app.main``). This module only
builds the widget; it has no callbacks of its own.

Import path: ``from app.components.header import build_language_selector``

Signature::

    build_language_selector(lang: str = "en") -> dash.development.base_component.Component

The returned component has ``id=ids.LANGUAGE_SELECTOR`` and ``value=lang``.
"""

from __future__ import annotations

from dash import dcc, html

from app import config, ids


def build_language_selector(lang: str = "en") -> html.Div:
    """Return the EN/IT radio-items language switcher.

    Args:
        lang: currently active language code (``"en"`` or ``"it"``).
              Defaults to ``"en"`` so the app stays importable before
              gui-core threads the per-request language value.

    Returns:
        A small ``html.Div`` containing a label and a ``dcc.RadioItems``
        with ``id=ids.LANGUAGE_SELECTOR`` and ``value=lang``.
        gui-core's ``build_layout(lang)`` embeds this in the header area
        and wires the clientside reload callback.
    """
    labels = config.labels_for(lang)
    return html.Div(
        className="language-selector",
        children=[
            html.Span(
                labels["lang_label"] + ":",
                className="language-selector-label",
            ),
            dcc.RadioItems(
                id=ids.LANGUAGE_SELECTOR,
                options=[
                    {"label": labels["lang_en"], "value": "en"},
                    {"label": labels["lang_it"], "value": "it"},
                ],
                value=lang,
                inline=True,
                className="language-radio",
                inputClassName="language-radio-input",
                labelClassName="language-radio-label",
            ),
        ],
    )
