"""Dash/Plotly GUI package wrapping the ``multilayer_tmm`` library.

This package is a thin presentation/integration layer. The pure domain<->dict
boundary lives in :mod:`app.state`; Plotly figure builders in :mod:`app.plots`;
Dash wiring in :mod:`app.layout`, :mod:`app.components`, and :mod:`app.callbacks`.

The ``multilayer_tmm`` library is never edited — only its public API is consumed.
"""
