# Angle-Sweep Map — Interface Contract

Status: design only. No feature code in this document. Every name below is the
exact string/identifier the team codes against. Every default is a literal.

This contract extends the Simulation tab with a mode toggle:

- `"single"` — current behavior (single angle, polarization may be `s`/`p`/`both`).
- `"angle_map"` — angle sweep. User sets angle start/stop (deg) + step. Output is
  up to three stacked heatmap subplots (R, T, A) in the existing
  `SIMULATE_GRAPH` slot: wavelength on x, angle on y, value as color
  (`Viridis`, fixed `zmin=0`, `zmax=1`). Polarization is SINGLE only (`s` or
  `p`); `both` is disabled. The existing `SIMULATE_CHANNELS_INPUT` checklist
  selects which of R/T/A appear.

Already-decided (do NOT reopen): outer `jax.vmap` over angle wrapped in one
`jax.jit`, no Python loop over angles; materials evaluated once; stacked
subplots in `SIMULATE_GRAPH`; single polarization in sweep; angle count capped
at `361`.

Pin: the kernel `multilayer_tmm/tmm.py::_simulate_one_polarization` already
`jax.vmap`s `_coherent_tmm_single` over wavelength with a SCALAR `angle_rad`.
The angle map adds ONE more `jax.vmap` over an angle vector around it. Both axes
live under a single `jax.jit`. This is purely additive — the existing kernel and
its `@partial(jax.jit, static_argnames=("polarization",))` decorator are NOT
modified.

---

## 1. Core API (`multilayer_tmm/tmm.py`) — owner: Forge

### 1.1 `AngleMapResult` dataclass

Add a new frozen dataclass next to `SimulationResult`:

```python
@dataclass(frozen=True)
class AngleMapResult:
    """Angle-resolved optical response for a single polarization."""

    wavelength_nm: jnp.ndarray   # shape (num_wavelengths,)
    angle_deg: jnp.ndarray       # shape (num_angles,)
    R: jnp.ndarray               # shape (num_angles, num_wavelengths)
    T: jnp.ndarray               # shape (num_angles, num_wavelengths)
    A: jnp.ndarray               # shape (num_angles, num_wavelengths)
    polarization: str            # "s" or "p" (single only)
```

Field order is fixed: `wavelength_nm`, `angle_deg`, `R`, `T`, `A`,
`polarization`. The 2-D arrays are `(num_angles, num_wavelengths)` — angle is
axis 0 (rows, y), wavelength is axis 1 (columns, x). `r` and `t` (complex
amplitudes) are intentionally NOT included (not plotted; keeps the map result
JSON-safe-friendly and small). `polarization` is a single `str`, NOT a tuple —
this is the structural discriminator from `SimulationResult.polarizations`.

### 1.2 `simulate_angle_map_arrays(...)` — functional/JAX interface

```python
def simulate_angle_map_arrays(
    n_by_wavelength: Any,
    thicknesses_nm: Any,
    wavelengths_nm: Any,
    angles_rad: Any,
    polarization: str = "s",
) -> AngleMapResult:
    ...
```

Contract:

- `n_by_wavelength` shape `(num_wavelengths, num_layers + 2)` — identical to
  `simulate_spectrum_arrays`. Materials are angle-independent, so this array is
  computed ONCE and shared across all angles (no per-angle re-evaluation).
- `angles_rad` is a 1-D array (radians), length `num_angles`.
- `polarization` MUST normalize to exactly one of `("s",)` / `("p",)` via the
  existing `_normalize_polarization`. If it normalizes to `("s", "p")` (i.e.
  the caller passed `"both"`/`"sp"`), raise `ValueError` with message
  `"angle map requires a single polarization (\"s\" or \"p\")."`. The stored
  `AngleMapResult.polarization` is the normalized single code (`"s"` or `"p"`).
- Reuse the SAME input validation as `simulate_spectrum_arrays`: `n_array.ndim
  == 2`, first dim matches `wavelengths_nm` length, second dim equals
  `len(thicknesses_nm) + 2`. Raise the existing `ValueError` strings.
- `angles_rad` must be 1-D; raise `ValueError("angles_rad must be a
  one-dimensional array.")` otherwise.

vmap/jit structure (exact, do NOT change):

```python
@partial(jax.jit, static_argnames=("polarization",))
def _simulate_angle_map_one_polarization(
    n_by_wavelength, thicknesses_nm, wavelengths_nm, angles_rad, polarization
):
    # OUTER vmap over angle; INNER call is the existing per-wavelength path.
    return jax.vmap(
        lambda angle_rad: _simulate_one_polarization(
            n_by_wavelength=n_by_wavelength,
            thicknesses_nm=thicknesses_nm,
            wavelengths_nm=wavelengths_nm,
            angle_rad=angle_rad,
            polarization=polarization,
        )
    )(angles_rad)
```

`_simulate_one_polarization` returns `(R, T, A, r, t)` each shape
`(num_wavelengths,)`; the outer `jax.vmap` stacks them to
`(num_angles, num_wavelengths)`. `simulate_angle_map_arrays` calls this jitted
helper, takes the first three returns (R, T, A), discards `r`/`t`, and packs an
`AngleMapResult`. The whole angle×wavelength computation runs under the single
`jax.jit` on `_simulate_angle_map_one_polarization` — no Python loop over
angles.

### 1.3 `simulate_angle_map(...)` — friendly interface

```python
def simulate_angle_map(
    stack: Stack,
    wavelengths_nm: Any,
    angles_deg: Any,
    polarization: str = "s",
) -> AngleMapResult:
    ...
```

Contract:

- Mirrors `simulate_spectrum`: `wavelengths = _as_1d_wavelengths(...)`,
  `n_by_wavelength, thicknesses_nm = stack_to_arrays(stack, wavelengths)`.
- `angles_deg` -> radians via `jnp.deg2rad(jnp.asarray(angles_deg,
  dtype=wavelengths.dtype))`. Must be 1-D (raise as in 1.2 if not).
- No `angle_rad` parameter and no scalar-angle path — this entry point is
  always a sweep. The single-angle workflow stays on `simulate_spectrum`.
- Delegates to `simulate_angle_map_arrays`.

### 1.4 Re-export (`multilayer_tmm/__init__.py`) — owner: Forge

Add to the `from multilayer_tmm.tmm import (...)` block and to `__all__`:
`AngleMapResult`, `simulate_angle_map`, `simulate_angle_map_arrays`. Keep
`__all__` alphabetized as the file already is.

---

## 2. Result-dict JSON schema (`SIMULATION_RESULT_STORE`)

`SIMULATION_RESULT_STORE` already holds the single-angle schema. A `mode`
discriminator is added so downstream readers branch unambiguously.

### 2.1 Single-angle schema (unchanged behavior, explicit discriminator)

`state.result_to_dict(result)` keeps its current keys and gains `mode:
"single"`:

```json
{
  "mode": "single",
  "wavelength_nm": [float, ...],
  "R": "<1-D list> | [<s list>, <p list>]",
  "T": "...",
  "A": "...",
  "polarizations": ["s"]
}
```

Readers that omit `mode` MUST treat a missing `mode` as `"single"` (backward
compatible).

### 2.2 Angle-map schema (new)

`state.angle_map_to_dict(result)` writes:

```json
{
  "mode": "angle_map",
  "wavelength_nm": [float, ...],
  "angle_deg": [float, ...],
  "R": [[float, ...], ...],
  "T": [[float, ...], ...],
  "A": [[float, ...], ...],
  "polarization": "s"
}
```

Key contract:

- `mode` is the literal string `"angle_map"`.
- `wavelength_nm` length is `num_wavelengths` (x axis).
- `angle_deg` length is `num_angles` (y axis).
- `R`, `T`, `A` are 2-D nested lists, shape `(num_angles, num_wavelengths)` —
  outer list indexed by angle, inner list by wavelength. Same orientation as
  `AngleMapResult` arrays. Built with the existing `_to_list` helper (it already
  produces JSON-safe nested lists of plain floats and casts complex->real
  defensively).
- `polarization` is a single string `"s"` or `"p"` (NOT the `polarizations`
  list key used in single mode). This is how readers distinguish the two schemas
  even without `mode`.

---

## 3. `app/config.py` keys — owner: gui-core

gui-core owns ALL of `config.py` (defaults AND labels). Frontend/viz only
REFERENCE these names; they never edit `config.py`.

### 3.1 Sweep defaults

Add the constant and accessor:

```python
DEFAULT_ANGLE_SWEEP = {"start_deg": 0.0, "stop_deg": 80.0, "step_deg": 1.0}

def default_angle_sweep() -> dict:
    """Return a fresh, valid angle-sweep dict (degrees)."""
    return dict(DEFAULT_ANGLE_SWEEP)
```

Literal values: `start_deg = 0.0`, `stop_deg = 80.0`, `step_deg = 1.0`. With
these defaults `num_angles == 81` (see §7 for the inclusive count formula).

The sweep dict lives under `stack_config["angle_sweep"]` (see §5.2). The
existing `angle_deg` key is unchanged and remains the single-angle source.
Add the mode discriminator default to `default_stack_config()`:
`"sim_mode": "single"` (new key; `config.SIM_MODE_VALUES = ("single",
"angle_map")`).

### 3.2 New label keys (added to BOTH `TRANSLATIONS["en"]` and `["it"]`)

Key sets must stay identical between `en` and `it` (the existing invariant).
Exact key strings and literal values:

| key | EN | IT |
|---|---|---|
| `sim_mode_label` | `Simulation mode` | `Modalità di simulazione` |
| `sim_mode_single` | `Single angle` | `Angolo singolo` |
| `sim_mode_angle_map` | `Angle sweep (map)` | `Scansione angolare (mappa)` |
| `angle_sweep_section` | `Angle sweep` | `Scansione angolare` |
| `angle_start` | `Start angle (deg)` | `Angolo iniziale (gradi)` |
| `angle_stop` | `Stop angle (deg)` | `Angolo finale (gradi)` |
| `angle_step` | `Step (deg)` | `Passo (gradi)` |
| `angle_map_pol_hint` | `Angle maps use a single polarization (s or p).` | `Le mappe angolari usano una singola polarizzazione (s o p).` |

Mode-toggle option lists are built with `config.options_for`. Add:

```python
SIM_MODE_VALUES = ("single", "angle_map")
```

and the option labels reuse the prefix `"sim_mode_"` so
`config.options_for(config.SIM_MODE_VALUES, "sim_mode_", lang)` yields
`{"label": "Single angle", "value": "single"}` and
`{"label": "Angle sweep (map)", "value": "angle_map"}`. (Therefore the keys
`sim_mode_single` / `sim_mode_angle_map` above double as both the toggle labels
and the `options_for` lookups — do not invent separate keys.)

### 3.3 Sweep validation message keys (added to `state._ERRORS`, see §7)

These live in `state.py::_ERRORS` (NOT `config.TRANSLATIONS`), because all
validation messages already live there. Listed here for the EN/IT text; the
exact keys are in §7.

---

## 4. `app/ids.py` — owner: gui-core

gui-core owns ALL of `ids.py`. Add to the "Simulate panel (workflow 1)"
section. Exact id strings:

```python
SIMULATE_MODE_INPUT = "simulate_mode_input"            # dcc.RadioItems: "single" | "angle_map"
SIMULATE_ANGLE_START_INPUT = "simulate_angle_start_input"
SIMULATE_ANGLE_STOP_INPUT = "simulate_angle_stop_input"
SIMULATE_ANGLE_STEP_INPUT = "simulate_angle_step_input"
#: container wrapping the single-angle ANGLE_INPUT + POLARIZATION "both" option.
SIMULATE_SINGLE_ANGLE_CONTAINER = "simulate_single_angle_container"
#: container wrapping the three angle-sweep inputs (start/stop/step). Shown only
#: in angle_map mode; toggled via style.display by a show/hide callback.
SIMULATE_ANGLE_SWEEP_CONTAINER = "simulate_angle_sweep_container"
```

Mode values are the literal strings `"single"` and `"angle_map"` (matching
`config.SIM_MODE_VALUES` and the schema `mode` discriminator). No new Store id
is needed — the angle-map result reuses `SIMULATION_RESULT_STORE`.

---

## 5. App-state model (`app/state.py`) — owner: gui-core

### 5.1 New serializer

```python
def angle_map_to_dict(result: AngleMapResult) -> dict:
    ...
```

Produces the §2.2 schema. Uses `_to_list` for the 2-D channel arrays,
`_to_list` for `wavelength_nm` and `angle_deg`, and `str(result.polarization)`.
Add `AngleMapResult` and `simulate_angle_map` to the `from multilayer_tmm
import (...)` block; add `angle_map_to_dict` and `run_angle_map` to `__all__`.

### 5.2 New runner — `run_simulation` branch

`run_simulation(config, lang)` branches on `config.get("sim_mode", "single")`:

- `"single"`: existing path, unchanged. Returns the §2.1 dict with `mode:
  "single"` injected.
- `"angle_map"`: validate (including §7 sweep rules), build the stack + grid as
  today, build the angle vector (degrees) from `config["angle_sweep"]`, call
  `simulate_angle_map(stack, wavelengths_nm=grid, angles_deg=angles,
  polarization=config["polarization"])`, return `angle_map_to_dict(result)`.

Factor the angle-map path into a helper `run_angle_map(config, lang)` so the
branch in `run_simulation` stays thin, OR inline it — either is acceptable, but
`run_angle_map` MUST exist as a public function (in `__all__`) so gui-qa can
test it directly.

Angle-vector construction (degrees), inclusive of `stop` within tolerance:

```python
sweep = config["angle_sweep"]
start, stop, step = float(sweep["start_deg"]), float(sweep["stop_deg"]), float(sweep["step_deg"])
num_angles = int(np.floor((stop - start) / step + 1e-9)) + 1
angles_deg = start + np.arange(num_angles) * step
```

This is the SAME `num_angles` formula validation uses in §7 — keep them
identical so a config that validates always builds.

### 5.3 Mapping table (UI widget -> config key -> Stack/sweep)

| UI widget (id) | config key | consumed by |
|---|---|---|
| `SIMULATE_MODE_INPUT` | `config["sim_mode"]` (`"single"`/`"angle_map"`) | `run_simulation` branch |
| `ANGLE_INPUT` | `config["angle_deg"]` (single mode only) | `simulate_spectrum(angle_deg=...)` |
| `POLARIZATION_INPUT` | `config["polarization"]` | both paths; `both` rejected in angle_map |
| `SIMULATE_ANGLE_START_INPUT` | `config["angle_sweep"]["start_deg"]` | `run_angle_map` angle vector |
| `SIMULATE_ANGLE_STOP_INPUT` | `config["angle_sweep"]["stop_deg"]` | `run_angle_map` angle vector |
| `SIMULATE_ANGLE_STEP_INPUT` | `config["angle_sweep"]["step_deg"]` | `run_angle_map` angle vector |
| `GRID_*_INPUT` | `config["grid"]` | wavelength grid (shared) |
| `SIMULATE_CHANNELS_INPUT` | (not stored) `render_spectrum`/`angle_map_figure` arg | figure builders |

---

## 6. `app/plots.py` — owner: gui-viz

### 6.1 New builder

```python
def angle_map_figure(
    result_dict: dict,
    channels: tuple[str, ...] = ("R", "T", "A"),
    title: str | None = None,
    lang: str = "en",
) -> go.Figure:
    ...
```

Behavior contract:

- Reads the §2.2 angle-map schema: `wavelength_nm` (x), `angle_deg` (y), and the
  selected 2-D channels.
- One `go.Heatmap` per selected channel that is present in `result_dict`,
  ordered as in `channels` (default `("R", "T", "A")`). Iterate `channels`
  filtered by `if channel in result_dict`, mirroring `spectrum_figure`.
- STACKED subplots via `plotly.subplots.make_subplots(rows=n_selected, cols=1,
  shared_xaxes=True, vertical_spacing=...)`. gui-viz MUST add
  `from plotly.subplots import make_subplots` (not currently imported in
  `plots.py`).
- Per heatmap: `z` shape `(num_angles, num_wavelengths)` taken directly from the
  schema (no transpose — angle is already axis 0 / rows / y), `x=wavelength_nm`,
  `y=angle_deg`, `colorscale="Viridis"`, `zmin=0`, `zmax=1`, `zauto=False`.
- Per-subplot colorbar titled with the channel name (`L["ch_R"]` etc.). Use a
  distinct `colorbar` per trace positioned for its row (e.g. set `colorbar` `len`
  and `y` per row, or rely on `coloraxis` per subplot). Each subplot's y-axis
  title is `L["angle_axis"]`; only the bottom subplot carries the x-axis title
  `L["x_axis"]` (shared x). Add a per-row subplot title equal to the channel
  display name so users can tell maps apart.
- `template="plotly_white"`. `title` optional (default `None`).
- If `channels` selects nothing present, return `empty_figure(L["empty_plot"]
  fallback...)` — but the empty-message string is the caller's responsibility in
  the single-mode path today; for the map, return an `empty_figure("")` so the
  callback layer stays the message owner. (Match the existing convention:
  `render_spectrum` passes the localized empty message; the builder does not own
  it.)

### 6.2 New `_PLOT_TRANSLATIONS` keys (BOTH `en` and `it`)

Add (key sets must stay identical):

| key | EN | IT |
|---|---|---|
| `angle_axis` | `Angle (deg)` | `Angolo (gradi)` |
| `map_colorbar` | `Value` | `Valore` |
| `map_hover_angle` | `Angle` | `Angolo` |

Existing keys reused: `x_axis` (`Wavelength (nm)` / `Lunghezza d'onda (nm)`),
`ch_R`/`ch_T`/`ch_A` (channel display names for subplot titles + colorbar
titles). No new colors — `Viridis` is fixed, not part of `_CHANNEL_COLORS`.

Hover template for each heatmap (use localized labels):
`f"{L['x_axis']}: %{{x:.3f}}<br>{L['map_hover_angle']}: %{{y:.3f}}<br>{L['map_colorbar']}: %{{z:.4f}}<extra></extra>"`.

Add `"angle_map_figure"` to `plots.__all__`.

---

## 7. `validate_config` additions (`app/state.py`) — owner: gui-core

`validate_config(config, lang)` gains a sweep block, evaluated only when
`config.get("sim_mode") == "angle_map"`:

Rules (all on `config["angle_sweep"]`, values cast to float/int):

1. `angle_sweep` present and a dict, else `err_sweep_missing`.
2. `start_deg`, `stop_deg`, `step_deg` numeric, else `err_sweep_params_invalid`.
3. `0 <= start_deg < stop_deg <= 90`, else `err_sweep_range_invalid`.
4. `step_deg > 0`, else `err_sweep_step_invalid`.
5. `num_angles` (computed by the §5.2 formula) must satisfy
   `1 <= num_angles <= 361`, else `err_sweep_too_many` (when `> 361`). Because
   rule 3 guarantees `start < stop` and rule 4 guarantees `step > 0`,
   `num_angles >= 1` always holds once 3 and 4 pass, so the lower bound never
   fires on its own; only the upper cap `> 361` is reported.
6. When `sim_mode == "angle_map"` and `config["polarization"] == "both"`, append
   `err_angle_map_needs_single_pol`. (Belt-and-suspenders: the UI disables
   `both` in sweep mode, the kernel also rejects it, but validation surfaces a
   localized message before the kernel call.)

The single-angle `angle_deg` numeric check is unchanged and still runs
regardless of mode.

New `_ERRORS` keys (added to BOTH `en` and `it`; key sets stay identical):

| key | EN | IT |
|---|---|---|
| `err_sweep_missing` | `Angle-sweep parameters missing.` | `Parametri della scansione angolare mancanti.` |
| `err_sweep_params_invalid` | `Invalid angle-sweep parameters.` | `Parametri della scansione angolare non validi.` |
| `err_sweep_range_invalid` | `Angle range must satisfy 0 <= start < stop <= 90 degrees.` | `L'intervallo angolare deve soddisfare 0 <= iniziale < finale <= 90 gradi.` |
| `err_sweep_step_invalid` | `Angle step must be greater than zero.` | `Il passo angolare deve essere maggiore di zero.` |
| `err_sweep_too_many` | `Too many angles ({num}); reduce the range or increase the step (max 361).` | `Troppi angoli ({num}); riduci l'intervallo o aumenta il passo (max 361).` |
| `err_angle_map_needs_single_pol` | `Angle maps require a single polarization (select 's' or 'p', not 'both').` | `Le mappe angolari richiedono una singola polarizzazione (selezionare 's' o 'p', non 'both').` |

`err_sweep_too_many` uses the `{num}` placeholder resolved by `_e(key, lang,
num=num_angles)`. The cap literal is `361`.

---

## 8. Callback wiring (`app/callbacks/simulate_callbacks.py`) — owner: gui-core

All Input/Output/State tuples are exact (`id`, `prop`).

### 8.1 `run_simulation` — extend State

Add the four new inputs as State so the existing button-driven callback reads
them. The `config` written to / read from `STACK_CONFIG_STORE` already carries
`sim_mode` and `angle_sweep` (gui-core's stack-store sync callback owns writing
them from `SIMULATE_MODE_INPUT` + the three angle inputs). The simplest
contract: the stack-store callback (gui-core, existing) keeps
`STACK_CONFIG_STORE` authoritative, so `run_simulation` reads only
`STACK_CONFIG_STORE` + `LANGUAGE_STORE` as today and the branch in §5.2 fires on
`config["sim_mode"]`. No new State on `run_simulation` is required IF the store
already holds `sim_mode`/`angle_sweep`.

Required outcome regardless of plumbing choice: by the time the button callback
runs, `stack_config["sim_mode"]` and `stack_config["angle_sweep"]` reflect the
current `SIMULATE_MODE_INPUT` / `SIMULATE_ANGLE_*_INPUT` widget values. gui-core
guarantees this through the existing config-sync callback (the same one that
already syncs grid/angle/polarization into `STACK_CONFIG_STORE`).

### 8.2 `render_spectrum` — branch on mode

```
Output(SIMULATE_GRAPH, "figure")
Input(SIMULATION_RESULT_STORE, "data")
Input(SIMULATE_CHANNELS_INPUT, "value")
State(LANGUAGE_STORE, "data")
```

Body:

- empty/None result -> `plots.empty_figure(labels["empty_plot"], lang=lang)`
  (unchanged).
- `result_dict.get("mode") == "angle_map"` -> `plots.angle_map_figure(
  result_dict, channels=selected, lang=lang)`.
- else (`"single"` or missing) -> `plots.spectrum_figure(result_dict,
  channels=selected, lang=lang)` (unchanged).

`selected = tuple(channels) if channels else ("R", "T", "A")` (unchanged).

### 8.3 `render_resonance_readout` — N/A in sweep mode

Same callback signature as today. When `result_dict.get("mode") ==
"angle_map"`, return a SINGLE row:

```python
[{"grandezza": labels["resonance"], "valore": labels["res_na_angle_map"]}]
```

The DataTable column ids are the existing `"grandezza"` / `"valore"` (see
`results_panel.py`). Add a new label key (gui-core, in `config.TRANSLATIONS`,
both langs):

| key | EN | IT |
|---|---|---|
| `res_na_angle_map` | `N/A for angle map` | `N/D per la mappa angolare` |

The single-mode resonance path is unchanged.

### 8.4 Show/hide callback for the sweep inputs (NEW callback)

```
Output(SIMULATE_SINGLE_ANGLE_CONTAINER, "style")
Output(SIMULATE_ANGLE_SWEEP_CONTAINER, "style")
Input(SIMULATE_MODE_INPUT, "value")
```

Body returns two `style` dicts toggling `display`:

- mode `"single"`: single-angle container `{"display": "block"}`, sweep
  container `{"display": "none"}`.
- mode `"angle_map"`: single-angle container `{"display": "none"}`, sweep
  container `{"display": "block"}`.

A second clientside-or-server callback MUST disable the `both` option in
`POLARIZATION_INPUT` when mode is `"angle_map"` (and re-enable it for
`"single"`):

```
Output(POLARIZATION_INPUT, "options")
Output(POLARIZATION_INPUT, "value")
Input(SIMULATE_MODE_INPUT, "value")
State(POLARIZATION_INPUT, "value")
State(LANGUAGE_STORE, "data")
```

In `"angle_map"` mode, rebuild options from `config.POLARIZATION_VALUES` with
the `"both"` entry marked `"disabled": True`; if the current value is `"both"`,
coerce it to `"s"`. In `"single"` mode, restore the full enabled option list and
leave the value untouched (`no_update` on the value). gui-core owns this
callback; the option list comes from `config.options_for(config.POLARIZATION_
VALUES, "pol_", lang)` post-processed to set `disabled` on the `both` entry.

---

## 9. File ownership table (parallel team — no overlapping edits)

| File | Owner | Edits in this contract |
|---|---|---|
| `multilayer_tmm/tmm.py` | Forge | §1.1–1.3: `AngleMapResult`, `simulate_angle_map_arrays`, `simulate_angle_map`, `_simulate_angle_map_one_polarization` |
| `multilayer_tmm/__init__.py` | Forge | §1.4: re-export the three new names |
| `app/state.py` | gui-core | §5: `angle_map_to_dict`, `run_angle_map`, `run_simulation` branch, `mode:"single"` injection; §7: sweep validation + new `_ERRORS` keys |
| `app/callbacks/simulate_callbacks.py` | gui-core | §8: `render_spectrum` branch, resonance N/A row, show/hide callback, polarization-disable callback |
| `app/config.py` | gui-core | §3: `DEFAULT_ANGLE_SWEEP`/`default_angle_sweep`, `SIM_MODE_VALUES`, `sim_mode` default, all new `TRANSLATIONS` keys (both langs), `res_na_angle_map` |
| `app/ids.py` | gui-core | §4: five new id constants |
| `app/components/simulate_panel.py` | gui-frontend | mode toggle (`SIMULATE_MODE_INPUT`), single-angle + sweep containers, three angle inputs; references config/ids names only |
| `app/assets/style.css` | gui-frontend | styling for the mode toggle + sweep container layout |
| `app/plots.py` | gui-viz | §6: `angle_map_figure`, `make_subplots` import, new `_PLOT_TRANSLATIONS` keys, `__all__` entry |
| `tests/` | gui-qa | kernel shape/jit tests, schema round-trip, validation rules, figure structure |

Hard rule: gui-core owns ALL of `config.py` (defaults AND labels) and ALL of
`ids.py`. gui-frontend and gui-viz REFERENCE those names exactly as written in
this contract and NEVER edit `config.py` or `ids.py`. If a name here is wrong or
missing, raise it against this doc — do not patch the owned file from another
seat.

---

## 10. Build order

1. Forge — `multilayer_tmm/tmm.py` + `__init__.py` (§1). No app dependency;
   unblocks gui-core's `run_angle_map` and gui-qa's kernel tests.
2. gui-core — `ids.py` + `config.py` names/defaults/labels (§3, §4). Pure
   constants; unblocks gui-frontend and gui-viz (they only reference these).
3. gui-core — `state.py` `angle_map_to_dict` / `run_angle_map` /
   `validate_config` (§5, §7), against Forge's API.
4. gui-viz — `plots.angle_map_figure` (§6), against the §2.2 schema (can start
   from the schema doc in parallel with step 3; only needs config label names
   from step 2 if it reuses them — it carries its own `_PLOT_TRANSLATIONS`).
5. gui-frontend — `simulate_panel.py` toggle + containers + inputs (§4 ids, §3
   labels), `style.css`.
6. gui-core — `simulate_callbacks.py` wiring (§8): branch render, resonance N/A,
   show/hide, polarization disable. Needs steps 2–5 in place.
7. gui-qa — `tests/` (§9 last row) once 1–6 land; assert EN/IT key-set
   invariants for the new `_ERRORS`, `TRANSLATIONS`, and `_PLOT_TRANSLATIONS`
   keys.

---

## 11. Library-gap notes (no core patching beyond the additive §1)

- The existing kernel already supports the full computation; §1 is purely
  additive (new public functions + dataclass + re-export). No existing
  `multilayer_tmm` function signature changes. This respects the
  library-is-pure constraint while giving the GUI exactly the angle-resolved
  entry point it needs.
- `_normalize_polarization` is reused as-is for the single-polarization guard;
  no new polarization codes are introduced.
- `_to_list` (state.py) already handles 2-D arrays and complex->real casting, so
  no new serialization helper is required for the map channels.
