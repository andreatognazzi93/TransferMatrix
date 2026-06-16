"""Pure Plotly figure builders for the Dash GUI.

This module is the visualization seam described in ``app/ARCHITECTURE.md`` (§6.2).
It is intentionally *pure*: it imports only :mod:`plotly` and :mod:`numpy`, never
``multilayer_tmm`` and never Dash. ``gui-core`` produces JSON-safe result dicts
(via ``state.result_to_dict``) and the callback layer hands them here to obtain
:class:`plotly.graph_objects.Figure` objects.

Result-dict schema consumed (ARCHITECTURE.md §6.1 ``result_to_dict``)::

    {
        "wavelength_nm": [float, ...],            # shape (N,)
        "R": <1-D list> | [<s list>, <p list>],   # (N,) or (2, N) in (s, p) order
        "T": ...,                                  # same shape as R
        "A": ...,                                  # A = 1 - R - T
        "polarizations": ["s"] | ["s", "p"],
    }

For a single polarization each channel array is shape ``(N,)``. For
``polarization="both"`` each channel is ``(2, N)`` with the leading axis ordered
``("s", "p")`` — rendered as two separate traces / legend entries.

Resonance dict consumed by :func:`resonance_overlay_figure` (ARCHITECTURE.md
``analyze_result`` -> ``ResonanceResult`` fields as dict)::

    {
        "resonance_wavelength_nm": float,
        "linewidth_nm": float,
        "quality_factor": float,
        "extremum_value": float,
        "half_level": float,
        "left_wavelength_nm": float,
        "right_wavelength_nm": float,
        "feature": "peak" | "valley" | ...,
    }

Language / i18n (ARCHITECTURE.md §12, §12.D):
    Default and canonical language is English (``lang="en"``).  Pass
    ``lang="it"`` to every public builder for Italian output.  The module
    carries its own catalog (``_PLOT_TRANSLATIONS``) so it remains pure
    (no ``config`` import).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

__all__ = [
    "spectrum_figure",
    "history_figure",
    "resonance_overlay_figure",
    "sketch_figure",
    "empty_figure",
]

# ---------------------------------------------------------------------------
# i18n catalog (ARCHITECTURE.md §12.3)
# Default / canonical language: English.  Italian is the alternative.
# Invariant: set(_PLOT_TRANSLATIONS["en"]) == set(_PLOT_TRANSLATIONS["it"])
# ---------------------------------------------------------------------------
_PLOT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Spectrum / shared axes
        "x_axis": "Wavelength (nm)",
        "y_value": "Value",
        # Channel display names
        "ch_R": "Reflectance",
        "ch_T": "Transmittance",
        "ch_A": "Absorptance",
        # Hover labels
        "hover_value": "Value",
        # History figure
        "loss": "Loss",
        "opt_step": "Optimization step",
        # Resonance overlay
        "resonance": "Resonance",
        "resonance_feature": "Resonance",
        "half_max": "Half maximum",
        "half_max_inline": "half maximum",
        "q_label": "Q",
        "linewidth_label": "Linewidth",
        "na": "n/a",
        # Sketch
        "sketch_title": "Multilayer schematic",
        "sketch_incident": "incident medium",
        "sketch_substrate": "substrate",
        "sketch_colorbar": "Re(n)",
        "sketch_materials": "Materials",
        "sketch_angle": "angle θ",
        # Resonance summary (summary_in_legend=True)
        "lambda_res": "λ_res",
        "q_factor": "Q",
    },
    "it": {
        # Spectrum / shared axes
        "x_axis": "Lunghezza d'onda (nm)",
        "y_value": "Valore",
        # Channel display names
        "ch_R": "Riflettività",
        "ch_T": "Trasmissione",
        "ch_A": "Assorbimento",
        # Hover labels
        "hover_value": "Valore",
        # History figure
        "loss": "Perdita",
        "opt_step": "Passo di ottimizzazione",
        # Resonance overlay
        "resonance": "Risonanza",
        "resonance_feature": "Risonanza",
        "half_max": "Mezza altezza",
        "half_max_inline": "mezza altezza",
        "q_label": "Q",
        "linewidth_label": "Larghezza di riga",
        "na": "n/d",
        # Sketch
        "sketch_title": "Schema del multistrato",
        "sketch_incident": "mezzo incidente",
        "sketch_substrate": "substrato",
        "sketch_colorbar": "Re(n)",
        "sketch_materials": "Materiali",
        "sketch_angle": "angolo θ",
        # Resonance summary (summary_in_legend=True)
        "lambda_res": "λ_ris",
        "q_factor": "Q",
    },
}


def _plot_labels(lang: str = "en") -> dict:
    """Return the display-string dict for *lang*, with EN fallback for missing keys.

    Unknown or unsupported ``lang`` values fall back to English so callers
    never get a ``KeyError`` from a missing translation.
    """
    base = _PLOT_TRANSLATIONS["en"]
    if lang == "en" or lang not in _PLOT_TRANSLATIONS:
        return dict(base)
    return {**base, **_PLOT_TRANSLATIONS[lang]}


# ---------------------------------------------------------------------------
# Stable colors / dash styles — NOT translated.
# ---------------------------------------------------------------------------

# Stable colors per channel so R/T/A keep the same hue across figures.
_CHANNEL_COLORS: dict[str, str] = {
    "R": "#1f77b4",  # blue
    "T": "#2ca02c",  # green
    "A": "#d62728",  # red
}

# Dash style per polarization so s/p are distinguishable beyond color.
_POL_DASH: dict[str, str] = {"s": "solid", "p": "dash"}


def _x(result_dict: dict) -> np.ndarray:
    """Extract the wavelength axis as a 1-D float array."""
    return np.asarray(result_dict["wavelength_nm"], dtype=float).ravel()


def _channel_array(result_dict: dict, channel: str) -> np.ndarray:
    """Return the channel data as a float ndarray, preserving (N,) or (2, N)."""
    return np.asarray(result_dict[channel], dtype=float)


def _polarizations(result_dict: dict) -> list[str]:
    """Return the polarization list, defaulting sensibly if absent."""
    pol = result_dict.get("polarizations")
    if not pol:
        return ["s"]
    return list(pol)


def _hovertemplate(channel: str, L: dict) -> str:
    """Build the hover template for a spectrum channel using localized labels *L*."""
    _ch_key = {"R": "ch_R", "T": "ch_T", "A": "ch_A"}
    label = L.get(_ch_key.get(channel, ""), channel)
    return (
        f"{label}<br>"
        f"{L['x_axis']}: %{{x:.3f}}<br>"
        f"{L['hover_value']}: %{{y:.4f}}<extra></extra>"
    )


def _iter_channel_series(result_dict: dict, channel: str):
    """Yield ``(pol, y)`` series for a channel, handling 1-D and (2, N).

    For one polarization the channel is 1-D -> a single ``(pol, y)`` pair.
    For "both" the channel is ``(2, N)`` in (s, p) order -> two pairs.
    """
    data = _channel_array(result_dict, channel)
    pols = _polarizations(result_dict)

    if data.ndim == 1:
        pol = pols[0] if pols else "s"
        yield pol, data
        return

    # data.ndim >= 2 -> leading axis is polarization (s, p) order.
    n_series = data.shape[0]
    for i in range(n_series):
        pol = pols[i] if i < len(pols) else f"pol{i}"
        yield pol, data[i]


def _trace_name(channel: str, pol: str, multi_pol: bool, L: dict) -> str:
    """Localized trace legend name for a channel/polarization combination."""
    _ch_key = {"R": "ch_R", "T": "ch_T", "A": "ch_A"}
    label = L.get(_ch_key.get(channel, ""), channel)
    if multi_pol:
        return f"{label} ({pol})"
    return label


def _channel_label(channel: str, L: dict) -> str:
    """Localized display name for a single channel."""
    _ch_key = {"R": "ch_R", "T": "ch_T", "A": "ch_A"}
    return L.get(_ch_key.get(channel, ""), channel)


def spectrum_figure(
    result_dict: dict,
    channels: tuple[str, ...] = ("R", "T", "A"),
    title: str | None = None,
    lang: str = "en",
) -> go.Figure:
    """Interactive R/T/A spectrum.

    One figure with wavelength on the x-axis and one trace per
    ``(channel, polarization)``. Handles both 1-D arrays (single polarization)
    and ``(2, N)`` arrays (``polarization="both"``, rendered as separate s/p
    traces grouped per channel in the legend). Hover shows the value at each λ.

    Parameters
    ----------
    result_dict:
        JSON-safe result dict from ``state.result_to_dict``.
    channels:
        Which of ``("R", "T", "A")`` to include.
    title:
        Optional figure title; ``None`` omits the title bar.
    lang:
        Display language — ``"en"`` (default) or ``"it"``.
    """
    L = _plot_labels(lang)
    fig = go.Figure()
    x = _x(result_dict)
    pols = _polarizations(result_dict)
    multi_pol = len(pols) > 1

    for channel in channels:
        if channel not in result_dict:
            continue
        color = _CHANNEL_COLORS.get(channel)
        for pol, y in _iter_channel_series(result_dict, channel):
            y = np.asarray(y, dtype=float).ravel()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=_trace_name(channel, pol, multi_pol, L),
                    legendgroup=channel,
                    line=dict(
                        color=color,
                        dash=_POL_DASH.get(pol, "solid") if multi_pol else "solid",
                    ),
                    hovertemplate=_hovertemplate(channel, L),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=L["x_axis"],
        yaxis_title=L["y_value"],
        hovermode="x unified",
        legend=dict(title=None),
        template="plotly_white",
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
    )
    return fig


def history_figure(history, title: str | None = None, lang: str = "en") -> go.Figure:
    """Loss-vs-step curve for an optimization run.

    ``history`` is the per-step loss sequence from ``OptimizationResult.history``
    (a list of floats). The x-axis is the optimization step index.

    Parameters
    ----------
    history:
        Sequence of per-step loss values.
    title:
        Optional figure title.
    lang:
        Display language — ``"en"`` (default) or ``"it"``.
    """
    L = _plot_labels(lang)
    y = np.asarray(history, dtype=float).ravel()
    x = np.arange(y.size)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=L["loss"],
            line=dict(color="#1f77b4"),
            hovertemplate=(
                f"{L['opt_step']}: %{{x}}<br>{L['loss']}: %{{y:.6g}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=L["opt_step"],
        yaxis_title=L["loss"],
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
    )
    return fig


def resonance_overlay_figure(
    result_dict: dict,
    resonance: dict,
    channel: str = "R",
    summary_in_legend: bool = False,
    lang: str = "en",
) -> go.Figure:
    """Single-channel spectrum with resonance markers overlaid.

    Draws the spectrum trace for ``channel`` plus markers for the resonance
    wavelength (extremum), the half-level horizontal reference, and the
    left/right half-level crossings (linewidth edges). ``resonance`` is a
    ``ResonanceResult``-as-dict (see module docstring).

    When ``summary_in_legend`` is True the resonance marker's **legend label**
    carries the final resonant wavelength and Q-factor (used for the optimized
    structure on the Ottimizzazione tab, §workflow-2), so the achieved λ_res and
    Q are readable straight from the legend, not only on hover.

    The resonance analysis is defined on a 1-D spectrum; if the channel holds
    two polarizations only the first (s) series is overlaid, matching the GUI
    boundary rule that resonance analysis rejects 2-row spectra.

    Parameters
    ----------
    result_dict:
        JSON-safe result dict from ``state.result_to_dict``.
    resonance:
        ``ResonanceResult``-as-dict from ``state.analyze_result``.
    channel:
        Which channel to display (``"R"``, ``"T"``, or ``"A"``).
    summary_in_legend:
        When ``True``, the resonance marker legend label carries
        ``λ_res = … nm<br>Q = …``.
    lang:
        Display language — ``"en"`` (default) or ``"it"``.
    """
    L = _plot_labels(lang)
    fig = go.Figure()
    x = _x(result_dict)
    label = _channel_label(channel, L)

    # Pick the 1-D spectrum series for this channel.
    series = list(_iter_channel_series(result_dict, channel))
    pol, y = series[0]
    y = np.asarray(y, dtype=float).ravel()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=_CHANNEL_COLORS.get(channel)),
            hovertemplate=_hovertemplate(channel, L),
        )
    )

    # --- resonance markers ---------------------------------------------
    res_wl = resonance.get("resonance_wavelength_nm")
    extremum = resonance.get("extremum_value")
    half_level = resonance.get("half_level")
    left_wl = resonance.get("left_wavelength_nm")
    right_wl = resonance.get("right_wavelength_nm")
    q = resonance.get("quality_factor")
    linewidth = resonance.get("linewidth_nm")
    feature = resonance.get("feature", "peak")

    if res_wl is not None and extremum is not None:
        na = L["na"]
        q_txt = f"{q:.3g}" if q is not None else na
        lw_txt = f"{linewidth:.4g}" if linewidth is not None else na
        marker_name = L["resonance"]
        if summary_in_legend:
            # Surface the achieved λ_res and Q directly in the legend label,
            # on two separate lines via Plotly's <br> support in trace names.
            lambda_sym = L["lambda_res"]
            q_sym = L["q_factor"]
            marker_name = f"{lambda_sym} = {res_wl:.1f} nm<br>{q_sym} = {q_txt}"
        fig.add_trace(
            go.Scatter(
                x=[res_wl],
                y=[extremum],
                mode="markers",
                name=marker_name,
                marker=dict(color="#d62728", size=11, symbol="x"),
                hovertemplate=(
                    f"{L['resonance_feature']} ({feature})<br>"
                    f"{L['x_axis']}: %{{x:.3f}}<br>"
                    f"{L['hover_value']}: %{{y:.4f}}<br>"
                    f"{L['q_label']}: {q_txt}<br>"
                    f"{L['linewidth_label']}: {lw_txt} nm<extra></extra>"
                ),
            )
        )

    # Half-level crossings (linewidth edges).
    if half_level is not None:
        cross_x = [v for v in (left_wl, right_wl) if v is not None]
        if cross_x:
            fig.add_trace(
                go.Scatter(
                    x=cross_x,
                    y=[half_level] * len(cross_x),
                    mode="markers",
                    name=L["half_max"],
                    marker=dict(color="#ff7f0e", size=9, symbol="circle"),
                    hovertemplate=(
                        f"{L['half_max']}<br>"
                        f"{L['x_axis']}: %{{x:.3f}}<br>"
                        f"{L['hover_value']}: %{{y:.4f}}<extra></extra>"
                    ),
                )
            )
        # Horizontal half-level reference line.
        fig.add_hline(
            y=half_level,
            line=dict(color="#ff7f0e", dash="dot", width=1),
            annotation_text=L["half_max_inline"],
            annotation_position="top left",
        )

    # Vertical guide at the resonance wavelength.
    if res_wl is not None:
        fig.add_vline(
            x=res_wl,
            line=dict(color="#d62728", dash="dot", width=1),
        )

    fig.update_layout(
        title=None,
        xaxis_title=L["x_axis"],
        yaxis_title=label,
        hovermode="closest",
        template="plotly_white",
        showlegend=True,
        margin=dict(l=60, r=20, t=20, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Mini-sketch of the multilayer (ARCHITECTURE.md §10)
# ---------------------------------------------------------------------------
#
# A schematic, NOT a spectral plot: stacked rectangles whose height is
# proportional to the real thickness_nm, fill color encodes Re(n) over a shared
# Viridis colorscale, semi-infinite incident/substrate drawn as hatched/dashed
# bands top & bottom, and an incidence-angle arrow hitting the top interface.
#
# Re(n) is extracted GUI-side from the §2.1 material dict so that this module
# keeps its purity (plotly + numpy only, no multilayer_tmm import):
#   kind == "constant" -> float(real part of d["n"])
#   kind == "csv"      -> mean(d["n"])  (representative band value, schematic)

# Sketch display strings are localized via ``_PLOT_TRANSLATIONS`` (§12.3):
# ``sketch_incident`` / ``sketch_substrate`` / ``sketch_colorbar``.

# Fixed "thickness" (in the same arbitrary units as the finite-layer heights)
# given to the semi-infinite bands so they read as distinct fixed-size slabs.
_SEMI_INF_BAND_FRACTION = 0.18  # fraction of the total finite-stack height

# Sampling resolution of the Viridis colorscale used GUI-side to map Re(n) to a
# concrete fill color (Plotly shapes need a literal color, not a colorscale ref).
_VIRIDIS = [
    (0.0, (68, 1, 84)),
    (0.1, (72, 40, 120)),
    (0.2, (62, 73, 137)),
    (0.3, (49, 104, 142)),
    (0.4, (38, 130, 142)),
    (0.5, (31, 158, 137)),
    (0.6, (53, 183, 121)),
    (0.7, (110, 206, 88)),
    (0.8, (181, 222, 43)),
    (0.9, (226, 228, 39)),
    (1.0, (253, 231, 37)),
]


def _material_re_n(material: dict) -> float:
    """Extract a representative Re(n) from a §2.1 material dict (GUI-side).

    Kind-aware per ARCHITECTURE.md §10.3. ``constant`` uses the real part of the
    stored ``n``; ``csv`` uses the mean of its tabulated ``n`` array. Any complex
    values are reduced to their real part. Falls back to ``1.0`` when ``n`` is
    missing or unusable, keeping the schematic robust to partial config.
    """
    if not isinstance(material, dict):
        return 1.0
    kind = material.get("kind")
    n = material.get("n")
    try:
        if kind == "csv":
            arr = np.asarray(n, dtype=complex).ravel()
            if arr.size == 0:
                return 1.0
            return float(np.mean(arr.real))
        # "constant" (and any unknown kind) -> scalar real part of n.
        return float(np.asarray(n, dtype=complex).ravel()[0].real)
    except (TypeError, ValueError, IndexError):
        return 1.0


def _material_k(material: dict) -> float:
    """Extract a representative k (absorption) from a §2.1 material dict.

    Mirror of :func:`_material_re_n` for the absorption coefficient, stored in
    the SEPARATE ``"k"`` field (``state.material_to_dict``) rather than baked
    into ``n``. ``csv`` uses the mean of the tabulated ``k`` array; ``constant``
    (and any unknown kind) uses the scalar. Falls back to ``0.0`` when ``k`` is
    missing or unusable, so lossless materials read as non-absorbing.
    """
    if not isinstance(material, dict):
        return 0.0
    kind = material.get("kind")
    k = material.get("k", 0.0)
    try:
        if kind == "csv":
            arr = np.asarray(k, dtype=float).ravel()
            if arr.size == 0:
                return 0.0
            return float(np.mean(np.abs(arr)))
        return float(abs(np.asarray(k, dtype=float).ravel()[0]))
    except (TypeError, ValueError, IndexError):
        return 0.0


def _material_name(material: dict, fallback: str) -> str:
    """Display name for a material, falling back to a generated label."""
    if isinstance(material, dict):
        name = material.get("name")
        if name:
            return str(name)
    return fallback


# Maximum brightness reduction applied to the Viridis hue at the largest k in
# the figure: an absorbing film reads darker than a lossless one of equal Re(n).
_K_DARKEN_MAX = 0.55


def _viridis_rgb(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Map a value to an ``(r, g, b)`` Viridis triple over ``[vmin, vmax]``.

    Normalizes ``value`` over the shared range, then linearly interpolates the
    sampled Viridis stops. Deterministic for a given value/range.
    """
    if vmax > vmin:
        t = (value - vmin) / (vmax - vmin)
    else:
        t = 0.5
    t = float(min(1.0, max(0.0, t)))

    for i in range(len(_VIRIDIS) - 1):
        t0, c0 = _VIRIDIS[i]
        t1, c1 = _VIRIDIS[i + 1]
        if t <= t1:
            span = t1 - t0
            f = 0.0 if span == 0 else (t - t0) / span
            r = round(c0[0] + (c1[0] - c0[0]) * f)
            g = round(c0[1] + (c1[1] - c0[1]) * f)
            b = round(c0[2] + (c1[2] - c0[2]) * f)
            return (r, g, b)
    return _VIRIDIS[-1][1]


def _re_n_to_color(value: float, vmin: float, vmax: float) -> str:
    """Map a Re(n) value to an ``rgb(...)`` string via the Viridis stops."""
    r, g, b = _viridis_rgb(value, vmin, vmax)
    return f"rgb({r}, {g}, {b})"


def _material_rgb(
    material: dict, vmin: float, vmax: float, kmax: float
) -> tuple[int, int, int]:
    """``(r, g, b)`` fill encoding BOTH Re(n) (hue) and k (darkness).

    The hue comes from ``Re(n)`` via Viridis over the shared ``[vmin, vmax]``
    range; the absorption ``k`` then darkens that hue, normalized over
    ``[0, kmax]``, so two materials with equal ``n`` but different ``k`` get
    visibly different colors. With a lossless stack (``kmax == 0``) this reduces
    exactly to the pure Re(n) color, preserving prior behaviour.
    """
    r, g, b = _viridis_rgb(_material_re_n(material), vmin, vmax)
    if kmax > 0.0:
        k = _material_k(material)
        factor = 1.0 - _K_DARKEN_MAX * min(1.0, max(0.0, k / kmax))
        r, g, b = round(r * factor), round(g * factor), round(b * factor)
    return (r, g, b)


def _material_fill_color(
    material: dict, vmin: float, vmax: float, kmax: float
) -> str:
    """``rgb(...)`` string form of :func:`_material_rgb`."""
    r, g, b = _material_rgb(material, vmin, vmax, kmax)
    return f"rgb({r}, {g}, {b})"


def _text_on(rgb: tuple[int, int, int]) -> str:
    """Readable label color (near-black or white) for a given fill ``rgb``.

    Uses the sRGB relative-luminance weights so dark fills (e.g. the deep
    Viridis purples/teals of low Re(n) or strongly absorbing films) get white
    text and light fills (the yellows of high Re(n)) get near-black text.
    """
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 140.0 else "#ffffff"


def _sketch_finite_blocks(stack_config: dict, grouped: bool):
    """Return the ordered list of finite blocks to draw, top -> bottom.

    Each block is a dict with keys:
      ``thickness`` (float, real nm), ``material`` (dict), ``label`` (str shown
      inside the block), ``multiplier`` (None or int M/K to render as "×M"),
      and on the first layer of a grouped period ``group_span`` (the number of
      layers in that period, so the renderer can bracket the whole period).

    For ``grouped=False`` (flat §2.2) every layer is one block. For
    ``grouped=True`` (grouped §9.1) every layer of each period is drawn as its
    own block; the period's ``repeat`` multiplier and ``group_span`` are tagged
    onto its FIRST layer so the ×M bracket can span the entire period. The
    cavity block is included only when ``enabled``.
    """
    blocks: list[dict] = []

    if not grouped:
        for idx, layer in enumerate(stack_config.get("layers", []) or []):
            blocks.append(
                {
                    "thickness": float(layer.get("thickness_nm", 0.0) or 0.0),
                    "material": layer.get("material", {}),
                    "label": _material_name(
                        layer.get("material", {}), f"strato {idx + 1}"
                    ),
                    "multiplier": None,
                }
            )
        return blocks

    # grouped=True -> §9.1 shape. input_group ×M | cavity? | output_group ×K.
    def _group_blocks(group: dict, multiplier_label: str):
        if not isinstance(group, dict):
            return
        repeat = int(group.get("repeat", 0) or 0)
        layers = group.get("layers", []) or []
        if repeat < 1 or not layers:
            return
        for idx, layer in enumerate(layers):
            blocks.append(
                {
                    "thickness": float(layer.get("thickness_nm", 0.0) or 0.0),
                    "material": layer.get("material", {}),
                    "label": _material_name(
                        layer.get("material", {}),
                        f"{multiplier_label} {idx + 1}",
                    ),
                    # Multiplier annotated once per collapsed group: tag the
                    # first period-layer of the group with the repeat count.
                    "multiplier": repeat if idx == 0 else None,
                    "group_span": len(layers) if idx == 0 else None,
                }
            )

    _group_blocks(stack_config.get("input_group", {}), "ingresso")

    cavity = stack_config.get("cavity", {})
    if isinstance(cavity, dict) and cavity.get("enabled"):
        blocks.append(
            {
                "thickness": float(cavity.get("thickness_nm", 0.0) or 0.0),
                "material": cavity.get("material", {}),
                "label": _material_name(cavity.get("material", {}), "cavità"),
                "multiplier": None,
            }
        )

    _group_blocks(stack_config.get("output_group", {}), "uscita")

    return blocks


def sketch_figure(
    stack_config: dict,
    angle_deg: float = 0.0,
    grouped: bool = False,
    title: str | None = None,
    lang: str = "en",
) -> go.Figure:
    """Schematic mini-sketch of the multilayer stack (ARCHITECTURE.md §10).

    Draws the stack as stacked rectangles (top -> bottom in physical order),
    each rectangle's height proportional to its real ``thickness_nm`` on a shared
    linear scale, filled by a Viridis color encoding ``Re(n)`` normalized over
    all media in the figure. Semi-infinite incident/substrate media are drawn as
    hatched/dashed bands at the top and bottom. An incidence-angle arrow enters
    the top interface at ``angle_deg`` from the normal with a ``θ`` label.

    ``grouped=False`` consumes the flat §2.2 config (``incident``, ``layers[]``,
    ``substrate``). ``grouped=True`` consumes the grouped §9.1 config: repeated
    mirror groups are collapsed to a single period block carrying a ``"×M"`` /
    ``"×K"`` bracket label, and the cavity block is drawn only if ``enabled``.

    Pure: imports only plotly + numpy; no ``multilayer_tmm`` import.
    """
    fig = go.Figure()
    L = _plot_labels(lang)

    incident = stack_config.get("incident", {})
    substrate = stack_config.get("substrate", {})
    blocks = _sketch_finite_blocks(stack_config, grouped)

    # --- shared Re(n) range over ALL media (incident, substrate, finite) -----
    re_n_values = [_material_re_n(incident), _material_re_n(substrate)]
    for blk in blocks:
        re_n_values.append(_material_re_n(blk["material"]))
    vmin = float(min(re_n_values))
    vmax = float(max(re_n_values))

    # --- shared k range over ALL media: drives the per-material darkening so
    # equal-n / different-k films are distinguishable (0 when the stack is
    # lossless -> colours collapse back to the pure Re(n) encoding).
    k_values = [_material_k(incident), _material_k(substrate)]
    for blk in blocks:
        k_values.append(_material_k(blk["material"]))
    kmax = float(max(k_values))

    # --- vertical layout (y grows downward as physical depth) ----------------
    total_finite = sum(max(b["thickness"], 0.0) for b in blocks)
    if total_finite <= 0.0:
        # Degenerate stack (no finite thickness): give every block equal height
        # so the sketch still renders meaningfully.
        for b in blocks:
            b["_h"] = 1.0
        total_finite = float(len(blocks)) or 1.0
    else:
        for b in blocks:
            b["_h"] = max(b["thickness"], 0.0)

    band_h = total_finite * _SEMI_INF_BAND_FRACTION
    x0, x1 = 0.0, 1.0  # horizontal extent of every slab

    shapes: list[dict] = []
    annotations: list[dict] = []

    band_traces: list[go.Scatter] = []

    def _hatched_band(y_top: float, y_bot: float, fill: str, hatch: str):
        # layout.Shape has no fillpattern, so hatched semi-infinite bands are
        # drawn as filled Scatter traces (which support fillpattern) with a
        # dashed border. Returns the trace; appended after finite shapes.
        band_traces.append(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y_top, y_top, y_bot, y_bot, y_top],
                mode="lines",
                fill="toself",
                fillcolor=fill,
                fillpattern=dict(shape=hatch, fgcolor="#ffffff", bgcolor=fill),
                line=dict(color="#444", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Top semi-infinite incident band (hatched + dashed border).
    y_cursor = 0.0
    incident_rgb = _material_rgb(incident, vmin, vmax, kmax)
    incident_color = f"rgb({incident_rgb[0]}, {incident_rgb[1]}, {incident_rgb[2]})"
    _hatched_band(y_cursor, y_cursor + band_h, incident_color, "/")
    annotations.append(
        dict(
            x=(x0 + x1) / 2,
            y=y_cursor + band_h / 2,
            text=(
                f"{L['sketch_incident']}<br>"
                f"{_material_name(incident, 'incidente')} "
                f"(n={_material_re_n(incident):.3g})"
            ),
            showarrow=False,
            font=dict(size=11, color=_text_on(incident_rgb)),
        )
    )
    top_interface_y = y_cursor + band_h
    y_cursor = top_interface_y

    # Finite layers.
    for i, blk in enumerate(blocks):
        h = blk["_h"]
        rgb = _material_rgb(blk["material"], vmin, vmax, kmax)
        color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y_cursor,
                y1=y_cursor + h,
                line=dict(color="#222", width=1),
                fillcolor=color,
                layer="below",
            )
        )
        # In-block label: material name + thickness. Text color adapts to the
        # fill luminance so it stays legible on dark and light boxes alike.
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=y_cursor + h / 2,
                text=f"{blk['label']} ({blk['thickness']:.1f} nm)",
                showarrow=False,
                font=dict(size=10, color=_text_on(rgb)),
            )
        )

        # Period multiplier bracket + "×M" label (grouped mode).
        mult = blk.get("multiplier")
        if mult is not None:
            # A period is drawn as ALL its layers (e.g. ingresso 1 + ingresso 2),
            # and the whole period repeats ×M (state.expand_optimization_config
            # extends input_period/output_period M/K times). So the bracket must
            # span every layer of the period, not just this first one.
            span = blk.get("group_span") or 1
            span_h = sum(
                blocks[j]["_h"] for j in range(i, min(i + span, len(blocks)))
            )
            bracket_x = x1 + 0.04
            y_top = y_cursor
            y_bot = y_cursor + span_h
            shapes.append(
                dict(
                    type="line",
                    x0=bracket_x,
                    x1=bracket_x,
                    y0=y_top,
                    y1=y_bot,
                    line=dict(color="#444", width=2),
                )
            )
            for yy in (y_top, y_bot):
                shapes.append(
                    dict(
                        type="line",
                        x0=bracket_x,
                        x1=bracket_x - 0.02,
                        y0=yy,
                        y1=yy,
                        line=dict(color="#444", width=2),
                    )
                )
            annotations.append(
                dict(
                    x=bracket_x + 0.03,
                    y=(y_top + y_bot) / 2,
                    text=f"×{mult}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=13, color="#444"),
                )
            )

        y_cursor += h

    # Bottom semi-infinite substrate band.
    substrate_rgb = _material_rgb(substrate, vmin, vmax, kmax)
    substrate_color = f"rgb({substrate_rgb[0]}, {substrate_rgb[1]}, {substrate_rgb[2]})"
    _hatched_band(y_cursor, y_cursor + band_h, substrate_color, "\\")
    annotations.append(
        dict(
            x=(x0 + x1) / 2,
            y=y_cursor + band_h / 2,
            text=(
                f"{L['sketch_substrate']}<br>"
                f"{_material_name(substrate, 'substrato')} "
                f"(n={_material_re_n(substrate):.3g})"
            ),
            showarrow=False,
            font=dict(size=11, color=_text_on(substrate_rgb)),
        )
    )
    total_height = y_cursor + band_h

    for _band in band_traces:
        fig.add_trace(_band)

    # --- incidence-angle arrow (sits ABOVE the structure, not overlapping it) --
    # The arrow lives in a reserved zone above y=0 (the top of the incident
    # band). Its tip touches the top of the structure; the tail and the θ label
    # stay in the margin above, so nothing overlaps the medium labels.
    theta = np.deg2rad(float(angle_deg))
    arrow_zone = band_h  # reserved headroom above the structure
    tip_x = (x0 + x1) / 2
    tip_y = 0.0  # top edge of the incident band == top of the drawn structure
    arrow_len = arrow_zone * 0.85
    # Normal is vertical (up = decreasing y -> negative, above the structure).
    # Tail offset by the angle from normal: horizontal ∝ sin(θ), vertical ∝ cos(θ).
    tail_x = tip_x - 0.30 * np.sin(theta)
    tail_y = tip_y - arrow_len * np.cos(theta)
    annotations.append(
        dict(
            x=tip_x,
            y=tip_y,
            ax=tail_x,
            ay=tail_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#d62728",
            text="",
        )
    )
    # Normal reference (dotted, in the headroom above the surface) + θ label.
    shapes.append(
        dict(
            type="line",
            x0=tip_x,
            x1=tip_x,
            y0=-arrow_len,
            y1=tip_y,
            line=dict(color="#888", width=1, dash="dot"),
        )
    )
    annotations.append(
        dict(
            x=tail_x,
            y=tail_y,
            text=f"{L['sketch_angle']} = {float(angle_deg):.1f}°",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#d62728"),
        )
    )

    # --- discrete material legend (name + Re(n) [+ k]) -----------------------
    # One invisible scatter trace per distinct material so it shows in the
    # legend with its swatch color; deterministic order: incident, finite, sub.
    # Distinctness keys on (name, n, k) so equal-n / different-k materials get
    # separate entries with their own (darkened) swatch.
    seen: set[tuple[str, float, float]] = set()

    def _add_legend_entry(material: dict, fallback: str):
        name = _material_name(material, fallback)
        rn = _material_re_n(material)
        kv = _material_k(material)
        key = (name, round(rn, 6), round(kv, 6))
        if key in seen:
            return
        seen.add(key)
        label = f"{name} (n={rn:.3g}"
        if kv > 0:
            label += f", k={kv:.3g}"
        label += ")"
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=12, color=_material_fill_color(material, vmin, vmax, kmax)
                ),
                name=label,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    _add_legend_entry(incident, "incidente")
    for blk in blocks:
        _add_legend_entry(blk["material"], blk["label"])
    _add_legend_entry(substrate, "substrato")

    # --- colorbar (Re(n)) via an invisible marker-color trace ----------------
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                color=[vmin],
                showscale=True,
                colorbar=dict(
                    title=L["sketch_colorbar"],
                    thickness=14,
                    len=0.7,
                    x=-0.12,
                    xanchor="right",
                    y=0.5,
                    yanchor="middle",
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title or L["sketch_title"],
        template="plotly_white",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            visible=False,
            range=[-0.15, 1.45],
            fixedrange=True,
        ),
        yaxis=dict(
            visible=False,
            # reversed: depth grows down. Top extended to fit the incidence
            # arrow zone that now sits above the structure (#1).
            range=[total_height, -(arrow_zone + band_h * 0.2)],
            fixedrange=True,
            scaleanchor=None,
        ),
        legend=dict(
            title=L["sketch_materials"],
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
        ),
        margin=dict(l=80, r=40, t=50 if (title or True) else 20, b=20),
    )
    return fig


def empty_figure(message: str = "", lang: str = "en") -> go.Figure:
    """Placeholder figure shown before the first simulation/optimization run.

    ``lang`` is accepted for API symmetry with the other builders (§12.D); the
    ``message`` is already localized by the caller, so it is not used here.
    """
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    if message:
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#666"),
        )
    return fig
