# `app/` — Dash/Plotly GUI Architecture

Architecture decisions and interface contracts for the Dash/Plotly GUI wrapping the
`multilayer_tmm` JAX library. This document is binding for the downstream agents
(`gui-frontend`, `gui-core`, `gui-viz`). It contains **no feature implementation** — only
module seams, the app-state model, decisions with rationale, and exact function signatures.

> **Hard rule:** `multilayer_tmm/` is a pure library and is **never edited**. The GUI only
> *calls* its public API (everything re-exported from `multilayer_tmm/__init__.py`).

---

## 0. Library API the GUI is allowed to call

Verified against the source. These are the only entry points the GUI depends on.

**Construction**
- `Material.constant(refractive_index: complex | float, name: str | None = None) -> Material`
- `Material.from_table(wavelength_nm, n, k=0.0, name=None) -> Material` — used to build CSV-uploaded materials *in memory* (no filesystem path needed).
- `Material.from_csv(path, wavelength_col="wavelength_nm", n_col="n", k_col="k", name=None)` — **not used** by the GUI; uploads are parsed in-memory and routed through `from_table`. (`from_callable` is **never** used — see §5.)
- `Layer(material: Material, thickness_nm)` — frozen dataclass.
- `Stack(incident: Material, layers: Sequence[Layer], substrate: Material)`; properties `.num_layers`, `.materials`.
- `stack_with_thicknesses(stack, thicknesses_nm) -> Stack`, `stack_thicknesses(stack, dtype=None) -> jnp.ndarray`.

**Grid**
- `wavelength_grid(start_nm: float, stop_nm: float, num: int) -> jnp.ndarray`

**Simulation (workflow 1)**
- `simulate_spectrum(stack, wavelengths_nm, angle_deg=None, angle_rad=None, polarization="s") -> SimulationResult`
- `SimulationResult` fields: `wavelength_nm, R, T, A, r, t, polarizations: tuple[str, ...]`.
  For `polarization="both"`, `R/T/A/r/t` gain a **leading axis of size 2** ordered `("s","p")`;
  for a single polarization they are 1-D.

**Optimization (workflow 2)**
- `optimize_thicknesses(objective, initial_thicknesses_nm, steps=100, learning_rate=0.1, lower_bound_nm=0.0) -> OptimizationResult` — fields `thicknesses_nm, history`.
- `optimize_resonance_target(stack, wavelengths_nm, target_wavelength_nm, target_q, spectrum="R", feature="peak", variable_layer_indices=None, angle_deg=0.0, angle_rad=None, polarization="s", steps=100, learning_rate=0.1, lower_bound_nm=0.0, wavelength_weight=1.0, q_weight=1.0, sharpness=20.0) -> ResonanceOptimizationResult` — fields `thicknesses_nm, variable_thicknesses_nm, history, resonance`.
- `mean_reflectivity(stack, wavelengths_nm, angle_deg=None, angle_rad=None, polarization="s") -> jnp.ndarray` — convenience objective.
- `resonance_target_loss(...)` — differentiable loss, used internally by `optimize_resonance_target`.

**Analysis**
- `analyze_resonance(wavelength_nm, spectrum, feature="peak", fraction=0.5, baseline=None) -> ResonanceResult`
  (`spectrum` must be **1-D**; reject 2-row "both" spectra at the GUI boundary).
  Fields: `resonance_wavelength_nm, linewidth_nm, quality_factor, extremum_value, half_level, left_wavelength_nm, right_wavelength_nm, feature`.

**Export (optional, server-side)**
- `export_simulation(stack, result, output_dir, simulation_name="simulation", reference_wavelength_nm=None, timestamp=None, show=False) -> ExportPaths`.

### Library gaps to respect (do NOT patch the core)
1. **No "build Stack from plain dict" helper.** The GUI must own its own serialization (the App-State Model, §2). State lives as JSON-serializable dicts in `dcc.Store`; `state.py` converts dicts → `Stack`.
2. **CSV materials need in-memory construction.** `Material.from_csv` only takes a path; uploads arrive as base64 in the browser. `state.py` parses bytes → arrays → `Material.from_table`. Do not add a "from bytes" method to the core.
3. **Optimization objective for `optimize_thicknesses` is a closure.** Building it requires capturing the stack + grid; `state.py` owns that closure factory so callbacks stay thin.
4. **`SimulationResult` arrays are `jnp.ndarray`.** Plot/serialization layers must `np.asarray(...)` before handing to Plotly/JSON. Never put `jnp` arrays in a `dcc.Store`.
5. **No tied/shared-variable support in `optimize_resonance_target`.** It only supports a 1:1 `variable_layer_indices` map (one free variable per flat layer). Sharing one variable across N repeats (tied mode, §11) is a GUI-side objective built on the public `resonance_target_loss` + `optimize_thicknesses`. Do not add grouping to the core.

---

## 1. Module map of `app/`

```
app/
  __init__.py          # package marker; no logic
  main.py              # app factory + entrypoint: builds Dash app, DiskcacheManager,
                       #   registers layout + callbacks, exposes `app` / `server`, __main__ runner
  config.py            # constants: defaults, cache dir, polarization/feature/spectrum option
                       #   lists, Italian label strings. No Dash imports.
  state.py             # PURE domain<->dict boundary. Dict (Store) <-> Stack/Material/grid.
                       #   Objective-closure factory. Result -> JSON-safe dict. NO Dash, NO Plotly.
  ids.py               # canonical component-id + Store-key string constants (English snake_case).
  layout.py            # top-level page skeleton: tabs (Simulazione / Ottimizzazione),
                       #   wires components together, declares all dcc.Store. No business logic.
  components/
    __init__.py
    stack_builder.py   # shared stack editor: incident/substrate material inputs +
                       #   dynamic finite-layer DataTable + grid/angle/polarization controls.
    material_input.py  # one material editor (constant n,k OR CSV upload) reused for
                       #   incident/substrate and inside layer rows where applicable.
    simulate_panel.py  # workflow-1 controls (run button, spectrum selection, status).
    optimize_panel.py  # workflow-2 controls (target wl/Q, feature, variable layers,
                       #   steps/lr/bounds, run button, progress, history).
    results_panel.py   # shared output area: graph(s) + resonance readout table + export.
  callbacks/
    __init__.py        # register_callbacks(app, cache) -> None; calls the submodules below
    stack_callbacks.py # add/remove/edit finite layers; keep Stack-state Store in sync.
    simulate_callbacks.py  # workflow 1: state -> simulate_spectrum -> result Store -> figures.
    optimize_callbacks.py  # workflow 2: background callback; progress + history -> UI.
  plots.py             # PURE Plotly figure builders. Input: JSON-safe result dicts. Output:
                       #   plotly.graph_objects.Figure. NO Dash, NO library imports.
  assets/              # Dash auto-served CSS (optional styling). No Python.
```

**Seam summary.** `state.py` and `plots.py` are pure and import-light: `state.py` imports
only `multilayer_tmm` + numpy; `plots.py` imports only `plotly` + numpy. Callbacks are thin
adapters: read Stores → call `state.py` → call library → call `plots.py` → write Stores.
Layout/components contain zero business logic.

---

## 2. App-state model

All UI state is held in **JSON-serializable dicts** inside `dcc.Store` components. `jnp`/`np`
arrays never enter a Store. `state.py` is the single translator between this dict shape and
the library's `Stack` / `Material` / grid / `SimulationResult`.

### 2.1 Material dict
```python
# constant material
{"kind": "constant", "n": 1.46, "k": 0.0, "name": "SiO2"}
# csv-uploaded material (parsed in-memory; arrays inlined, already sorted-safe via from_table)
{"kind": "csv", "wavelength_nm": [...], "n": [...], "k": [...], "name": "Ag_meas"}
```
`kind` is the discriminator. Only `"constant"` and `"csv"` are valid (no `"callable"`).

### 2.2 Stack-config dict (the canonical app state)
```python
{
  "incident":  <material dict>,                 # semi-infinite
  "layers": [                                   # ordered, finite layers
     {"material": <material dict>, "thickness_nm": 120.0},
     ...
  ],
  "substrate": <material dict>,                 # semi-infinite
  "grid":  {"start_nm": 400.0, "stop_nm": 800.0, "num": 401},
  "angle_deg": 0.0,
  "polarization": "s"                            # "s" | "p" | "both"
}
```

### 2.3 Mapping UI → library objects
| App-state field | UI widget (component) | Library target |
|---|---|---|
| `incident` | `material_input` (top of `stack_builder`) | `Stack.incident` (`Material`) |
| `layers[*].material` | per-row material cell in finite-layer editor | `Layer.material` |
| `layers[*].thickness_nm` | per-row thickness cell (editable DataTable) | `Layer.thickness_nm` |
| `substrate` | `material_input` (bottom of `stack_builder`) | `Stack.substrate` |
| `grid` | start/stop/num numeric inputs | `wavelength_grid(start_nm, stop_nm, num)` |
| `angle_deg` | numeric input (deg) | `simulate_spectrum(..., angle_deg=...)` |
| `polarization` | radio/dropdown | `simulate_spectrum(..., polarization=...)` |

Optimization adds **two** Stores: `optimize_config_store` for the workflow-2 scalar inputs
(target wl/Q, feature, spectrum channel, steps, lr, bounds, weights, sharpness) and
`opt_stack_config_store` for the **grouped/cavity stack structure** (§9), which also carries
the `variable` selector for which expanded thicknesses are free.

> **Contract change (explicit).** The earlier statement that "the Stack config Store is shared
> by both workflows" is **superseded** for the optimization tab. The optimization tab uses its
> own grouped stack store (`opt_stack_config_store`, §9) because its structure
> (`incident | input_group ×M | cavity | output_group ×K | substrate`) is not expressible as a
> flat layer list in the editor. The Simulazione tab continues to use the flat
> `stack_config_store` unchanged. The "single shared stack-builder" goal is preserved at the
> *component* level: §9.4 reuses `material_input` and the same material-library Store; only the
> top-level structure differs.

### 2.4 Store inventory (declared in `layout.py`, keys in `ids.py`)
- `stack_config_store` — §2.2 **flat** stack dict. Source of truth for the **Simulazione** tab.
- `opt_stack_config_store` — §9 **grouped/cavity** stack dict. Source of truth for the
  **Ottimizzazione** tab's *structure* (incident, input/output mirror groups, cavity,
  substrate, grid, angle, polarization, and the `variable` selector). Added in §9; it does
  **not** replace `stack_config_store` — the two tabs own distinct stack stores.
- `optimize_config_store` — §2.3 workflow-2 *scalar* inputs (target wl/Q, feature, spectrum
  channel, steps, lr, bounds, weights, sharpness). The `variable_layer_indices` selection now
  lives inside `opt_stack_config_store["variable"]` (§9), not here.
- `simulation_result_store` — last `SimulationResult` as JSON dict (§ contract `result_to_dict`).
- `optimization_result_store` — last optimization outcome as JSON dict (history + final thicknesses + resonance).
- `optimization_progress_store` — incremental progress for the running background job (step/total + partial history) when `progress=` is used.

> **Scope note (locked).** The grouped/cavity model (§9) applies to the **Ottimizzazione tab
> only**. The **Simulazione tab keeps the existing flat layer list** (§2.2 + §3 `DataTable`).
> Nothing in §2.2 / §3 changes.

---

## 3. Decision: dynamic finite-layer editor → **Dash `DataTable`**

**Chosen:** `dash_table.DataTable` (editable, `row_deletable=True`, an "Aggiungi strato"
button appending rows) for the ordered finite-layer list.

**Rationale.**
- The finite layers are a homogeneous **ordered table** (material + thickness per row) — the
  exact shape `DataTable` is built for. One callback reads `derived_virtual_data` / `data`
  and rewrites `stack_config_store["layers"]`; no per-row callback explosion.
- Pattern-matching callbacks (`ALL`/`MATCH` dynamic component ids) are powerful but introduce
  N components × callback wiring, ordering/reindex bugs on delete, and far more state to keep
  consistent — overkill for a flat list where order = stack order.
- `DataTable` gives row reordering-by-delete/insert, inline edit, and deletion essentially for
  free, keeping `stack_callbacks.py` thin.

**Constraint that shapes the table.** A `DataTable` cell cannot host a file-upload widget. So:
- **Thickness** and **constant (n, k)** and a **material-kind selector** live as editable
  columns in the table (numeric + `dropdown` columns).
- **CSV-backed layer materials** are handled out-of-band: a small "libreria materiali" Store
  holds uploaded materials by name; a table dropdown column references them by `name`. CSV
  upload itself uses a dedicated `dcc.Upload` in `material_input.py`, not a table cell.
  This keeps incident/substrate and layer materials on the same upload path.

Downstream (`gui-frontend`) owns the exact column set, but the table's `data` must round-trip
losslessly into `stack_config_store["layers"]` via the §6 contracts.

---

## 4. Decision: long-running optimization → **Dash background callbacks + `DiskcacheManager`**

**Chosen:** the optimize "Esegui" callback is registered with `background=True` and a
`DiskcacheManager` constructed in `main.py`. (Locked by the user; this records the wiring.)

**Why DiskcacheManager (not Celery):** single-process desktop/local app, no broker to run,
zero external infra. `diskcache` is the documented zero-dependency manager for Dash background
callbacks and is sufficient for one user running one optimization at a time.

**Progress + history to the UI:**
- The background callback declares `running=[...]` to disable the run button / show a spinner
  and `progress=[Output(...)]` to push incremental updates into `optimization_progress_store`.
- **Library limitation to design around:** `optimize_thicknesses` / `optimize_resonance_target`
  run their full `steps` loop internally and only return the final `OptimizationResult` /
  `ResonanceOptimizationResult` (with the complete `history` array). They do **not** expose a
  per-step callback. We therefore do **not** patch the core. The GUI's progress strategy:
  - Coarse progress: the callback reports indeterminate/"in corso" status while the library
    loop runs, then a single completion event.
  - **Final history is plotted from `OptimizationResult.history`** (the loss-per-step curve)
    once the job returns — this is the "history reaching the UI", via `optimization_result_store`.
  - If finer live progress is wanted later, the documented gap (no step hook) tells a future
    library change; the GUI stays unchanged in shape.
- On completion the callback writes `optimization_result_store`, and a downstream display
  callback renders: final thicknesses, the loss-history figure (`history_figure`), and — for
  resonance runs — the `ResonanceResult` readout plus a final-spectrum overlay obtained by one
  follow-up `simulate_spectrum` on the optimized stack.

---

## 5. Material-input policy

- **Allowed:** constant `n + i k` (→ `Material.constant`) and CSV upload (→ parsed in-memory
  → `Material.from_table`).
- **Excluded:** callable materials (`Material.from_callable`). The GUI **never** evaluates
  user-supplied Python/expressions. There is no code-eval path anywhere in `app/`.
- **CSV upload handling.** `dcc.Upload` yields a base64 data URL. `state.py.parse_material_csv`
  decodes bytes, parses `wavelength_nm,n,k` columns (default headers, mirroring
  `read_material_csv`), and returns a CSV-material dict (§2.1). Parsing happens in `app/`, not
  via a temp file through `Material.from_csv`, so no path round-trip and no library change.
- **Validation** (raise `ValueError` with an Italian message, surfaced in the UI): missing
  columns, non-numeric cells, empty file, `num < 2` for the grid, `start_nm >= stop_nm`,
  negative thickness.

---

## 6. Interface contracts (binding)

These signatures are committed. Downstream agents implement to these exactly. Types: dicts are
plain JSON-safe Python (`float`/`int`/`str`/`list`/`bool`), figures are
`plotly.graph_objects.Figure`.

### 6.1 `state.py` (owner: `gui-core`) — pure, imports only `multilayer_tmm` + numpy
```python
# ---- material ----
def material_from_dict(d: dict) -> Material: ...
def material_to_dict(m: Material) -> dict: ...          # constant/tabulated only; raises on callable
def parse_material_csv(contents: str, filename: str | None = None,
                       name: str | None = None) -> dict: ...   # base64 dcc.Upload -> csv-material dict

# ---- stack + grid ----
def stack_from_config(config: dict) -> Stack: ...        # config = §2.2 dict -> Stack
def grid_from_config(config: dict) -> "jnp.ndarray": ...  # uses wavelength_grid(start,stop,num)
def validate_config(config: dict) -> list[str]: ...      # [] if valid, else Italian error strings

# ---- simulation ----
def run_simulation(config: dict) -> dict: ...            # builds stack+grid, calls simulate_spectrum,
                                                         #   returns result_to_dict(...) (JSON-safe)
def result_to_dict(result: SimulationResult) -> dict: ...
    # {"wavelength_nm": [...], "R": <1d or [s,p]>, "T": ..., "A": ...,
    #  "polarizations": ["s"] | ["s","p"]}   (r,t omitted; not plotted)

# ---- analysis ----
def analyze_result(result_dict: dict, channel: str = "R",
                   feature: str = "peak") -> dict: ...   # 1-D only; -> ResonanceResult fields as dict;
                                                         #   raises if result has 2 polarizations

# ---- optimization (workflow 2) ----
def make_thickness_objective(config: dict, channel: str = "R"):
    # returns (objective: Callable[[jnp.ndarray], jnp.ndarray], initial_thicknesses_nm: jnp.ndarray)
    # closure captures stack+grid+angle+polarization; for optimize_thicknesses path.
    ...

# Grouped/cavity expansion (§9) — pure, no Dash. The single bridge from the grouped
# optimization config to the flat library Stack + real variable indices.
def expand_optimization_config(opt_stack_config: dict) -> tuple[Stack, tuple[int, ...]]:
    # opt_stack_config = §9.1 dict. Replicates the example's Python-loop expansion:
    #   incident | input_group.layers * repeat | cavity (if enabled) | output_group.layers * repeat | substrate
    # Returns (flat Stack, variable_layer_indices) where variable_layer_indices are the REAL
    # flat indices (post-expansion) of the thicknesses selected by opt_stack_config["variable"]
    # (§9.2). Tuple is sorted, de-duplicated, and non-empty (raises ValueError if empty).
    # NOTE (§11): now a compat shim delegating to expand_optimization_variables (flattens groups).

# Optimization entry points now take the GROUPED stack config (§9.1) instead of the flat §2.2
# config. They expand internally via expand_optimization_variables (§11), then call the library.
def run_resonance_optimization(opt_stack_config: dict, opt_config: dict) -> dict: ...
    # expand_optimization_variables -> (stack, groups); reads grid/angle/polarization from
    #   opt_stack_config; optimizes one variable per group (scatter-broadcast, §11.4). Returns:
    # {"thicknesses_nm":[...], "variable_thicknesses_nm":[...], "history":[...],
    #  "resonance": <ResonanceResult-as-dict>, "final_result": <result_to_dict of optimized stack>}
    # variable_thicknesses_nm is now ONE value per variable/group (§11.4). "both" REJECTED.
def run_thickness_optimization(opt_stack_config: dict, opt_config: dict) -> dict: ...
    # generic loss path. expand_optimization_variables -> (stack, groups); builds an objective
    #   over the per-group variables (scatter-broadcast) + optimize_thicknesses. Returns:
    # {"thicknesses_nm":[...], "variable_thicknesses_nm":[...], "history":[...],
    #  "final_result": <result_to_dict>}.  polarization="both" REJECTED.
```

> **Signature change (explicit).** `run_resonance_optimization` / `run_thickness_optimization`
> previously took `(config, opt_config)` where `config` was the flat §2.2 dict. They now take
> `(opt_stack_config, opt_config)` where `opt_stack_config` is the grouped §9.1 dict. This is a
> *replacement*, not a new entry point — the optimization tab only ever produces grouped
> configs. `make_thickness_objective(config, channel)` (flat-config helper) is **retained
> unchanged** for any future flat-stack optimization but is **not** on the optimization tab's
> path. Both "both"-polarization rejections are preserved (and made explicit above).

### 6.2 `plots.py` (owner: `gui-viz`) — pure, imports only `plotly` + numpy
```python
def spectrum_figure(result_dict: dict, channels: tuple[str, ...] = ("R", "T", "A"),
                    title: str | None = None) -> go.Figure: ...
    # one figure, wavelength on x; one trace per (channel, polarization).
    # handles both 1-D and [s,p] result arrays.

def history_figure(history: list[float], title: str | None = None) -> go.Figure: ...
    # loss-vs-step line for optimization runs.

def resonance_overlay_figure(result_dict: dict, resonance: dict,
                             channel: str = "R") -> go.Figure: ...
    # spectrum trace + markers for resonance wavelength / half-level / left&right crossings.

def empty_figure(message: str = "") -> go.Figure: ...      # placeholder before first run.

# Mini-sketch of the multilayer (§10). One function, `grouped` flag selects input dict shape.
def sketch_figure(stack_config: dict, angle_deg: float = 0.0,
                  grouped: bool = False, title: str | None = None) -> go.Figure: ...
    # grouped=False -> consumes the FLAT §2.2 dict (Simulazione tab).
    # grouped=True  -> consumes the GROUPED §9.1 dict (Ottimizzazione tab); repeated groups
    #                  drawn ONCE with a "×M" / "×K" bracket label; disabled cavity omitted.
    # Stacked rectangles, height ∝ real thickness_nm; incident/substrate as hatched/dashed
    # semi-infinite bands top & bottom; fill color encodes Re(n) (§10.3); incidence-angle arrow
    # drawn from the top per angle_deg. Pure: imports only plotly + numpy.
```

### 6.3 `main.py` (owner: `gui-frontend`)
```python
def create_app() -> dash.Dash: ...
    # builds DiskcacheManager(Cache(config.CACHE_DIR)), constructs Dash(background_callback_manager=...),
    # sets app.layout = build_layout(), calls callbacks.register_callbacks(app, cache).
server = ...        # create_app().server   (for `gunicorn app.main:server`)
# __main__: app.run(debug=...)
```

### 6.4 `layout.py` (owner: `gui-frontend`)
```python
def build_layout() -> "dash.development.base_component.Component": ...
    # tabs (Simulazione / Ottimizzazione), embeds component builders, declares all dcc.Store from §2.4.
```

### 6.5 `callbacks/__init__.py` (owner: `gui-frontend`)
```python
def register_callbacks(app: dash.Dash, cache) -> None: ...
    # delegates to register in stack_callbacks / simulate_callbacks / optimize_callbacks.
```

### 6.6 `components/*` (owner: `gui-frontend`)
Each module exposes one builder returning a Dash component; ids come from `ids.py`:
```python
# stack_builder.py
def build_stack_builder() -> Component: ...
# material_input.py
def build_material_input(id_prefix: str, label: str) -> Component: ...
# simulate_panel.py
def build_simulate_panel() -> Component: ...
# optimize_panel.py
def build_optimize_panel() -> Component: ...
# results_panel.py
def build_results_panel(id_prefix: str) -> Component: ...
```

**Conventions everyone honors:** UI text/labels in **Italian**; all component ids and Store
keys in **English snake_case**, declared once in `ids.py` and imported (never inline string
literals). Callbacks contain no domain math — they only read Stores, call `state.py`, call
`plots.py`, and write Stores.

> **Superseded by §12 (i18n).** The "UI text/labels in Italian" convention above is replaced:
> the default and canonical UI language is **English**, with Italian selectable. See §12 for the
> catalog, the `lang` parameters added to `plots.*`/`state.*`/component builders, and the
> runtime mechanism.

---

## 7. Editable-install recommendation

**Recommendation: switch to an editable install (`pip install -e ".[dev,opt,gui]"`).**

- The repo runs app/tests via `PYTHONPATH=.` today. That works for a flat `import
  multilayer_tmm`, but the GUI introduces a second importable package (`app`) and a
  console/server entrypoint (`gunicorn app.main:server`, `python -m app.main`). Editable
  install makes both packages importable from any cwd without `PYTHONPATH` juggling and makes
  the `[gui]` extra installable in one step.
- It also lets `app` be packaged/declared in `pyproject.toml` cleanly.

**`pyproject.toml` change to request (owner: `gui-frontend`, applied to pyproject — NOT to
`multilayer_tmm/`):** add a `[project.optional-dependencies]` group:
```toml
gui = [
    "dash>=2.16",
    "plotly>=5.20",
    "diskcache>=5.6",       # DiskcacheManager backend for background callbacks
    "pandas>=2.0",          # dash DataTable data handling / CSV parsing convenience
    "multiprocess>=0.70",   # required by Dash background callbacks with diskcache
]
```
If editable install is declined, everything still runs with `PYTHONPATH=.`; the only cost is
manual env management and no installed entrypoint. Editable install is strictly the lower-risk
path and does not touch the library source.

---

## 8. Build order for downstream agents

This now spans the original flat-simulation work **and** the two additions: the grouped/cavity
optimization model (§9) and the mini-sketch (§10).

1. **`gui-core`** — `state.py` (§6.1) + `config.py` constants + `ids.py`.
2. **`gui-viz`** — `plots.py` (§6.2).
3. **`gui-frontend`** — `pyproject.toml` `[gui]` extra (§7), `components/*`, `layout.py`,
   `main.py`, then `callbacks/*` (§6.3–6.6).
4. **`gui-qa`** — tests last, against the frozen contracts.

Contract changes after this point require updating this document first.

---

## 9. Grouped/cavity stack model — Ottimizzazione tab (Addition A)

The optimization tab models a resonant-cavity stack whose structure mirrors
`examples/optimize_resonance_target.py` exactly:

```
incident (semi-inf) | input_mirror_group ×M | cavity (single layer) | output_mirror_group ×K | substrate (semi-inf)
```

**Locked decisions (not re-litigated):**
- Grouped+cavity model applies to the **Ottimizzazione tab only**. Simulazione stays flat (§2.2/§3).
- `M`, `K` are user-set **integers, FIXED during optimization** — they are expansion counts
  (Python-loop replication, exactly like the example), **not** differentiable variables.
- Cavity is a **single layer** (material + thickness) with an **on/off toggle** (`enabled`).
- Optimizable thicknesses are selectable among `{cavity, input-group layer i, output-group
  layer j}`. **Default = cavity only** (matching the example's `variable_layer_indices=(cavity_index,)`).

### 9.1 Grouped optimization stack-config dict (`opt_stack_config_store`)

JSON-safe; all values `float`/`int`/`str`/`bool`/`list`/dict. Material dicts are the §2.1 shape.

```python
{
  "incident":  <material dict>,                         # semi-infinite, top
  "input_group": {
     "layers": [                                        # one period of the input mirror
        {"material": <material dict>, "thickness_nm": 72.0},
        {"material": <material dict>, "thickness_nm": 103.0}
     ],
     "repeat": 3                                         # M (int >= 0)
  },
  "cavity": {"material": <material dict>, "thickness_nm": 190.0, "enabled": true},
  "output_group": {
     "layers": [                                        # one period of the output mirror
        {"material": <material dict>, "thickness_nm": 103.0},
        {"material": <material dict>, "thickness_nm": 72.0}
     ],
     "repeat": 3                                         # K (int >= 0)
  },
  "substrate": <material dict>,                          # semi-infinite, bottom

  "grid":  {"start_nm": 520.0, "stop_nm": 720.0, "num": 241},
  "angle_deg": 0.0,
  "polarization": "s",                                  # "s" | "p"  ("both" rejected, §6.1)

  "variable": { ... }                                   # see §11.1 (mode + selectors)
}
```

> **`variable` schema is now defined by §11.1** (it gained `mode` + `flat_layers`). The
> period-level keys (`cavity`, `input_layers`, `output_layers`) keep their §9.2 meaning and are
> used by the "tied" mode.

### 9.2 Expansion + real variable indices — `expand_optimization_config`

**Expansion (replicates the example's loops):** build the flat `layers` list in order
1. `input_group.layers` repeated `input_group.repeat` (M) times;
2. the `cavity` single layer **iff** `cavity.enabled`;
3. `output_group.layers` repeated `output_group.repeat` (K) times.
Then `Stack(incident, layers, substrate)` with materials built via `material_from_dict`.

**Real flat-index computation.** Let `Lin = len(input_group.layers)`, `Lout =
len(output_group.layers)`. Flat layout offsets:
- input block occupies flat indices `[0, M*Lin)`; period-layer `i` in the **first repeat** →
  flat index `i`.
- cavity (if enabled) → flat index `cavity_index = M*Lin`.
- output block starts at `out_start = M*Lin + (1 if cavity.enabled else 0)`; period-layer `j`
  in the **first repeat** → flat index `out_start + j`.

> **Superseded by §11:** the original §9.2 rule mapped a selected period-layer to **only its
> first repeat**, leaving the other M-1/K-1 copies frozen. §11 replaces this with the **tied**
> mode (broadcast across all repeats) and adds an **independent** mode. `expand_optimization_config`
> is now a compat shim over `expand_optimization_variables` (§11.4).

### 9.3 Feeding the existing optimizers
See §11.4 for the current (group-based) wiring.

### 9.4 Component/UI mapping (gui-frontend, `optimize_panel.py`)

| Grouped-config field | UI widget | Notes |
|---|---|---|
| `incident`, `substrate` | `material_input` (reused) | same path as flat tab |
| `input_group.layers` / `output_group.layers` | two small `DataTable`s (period definition) | material-kind + n,k + thickness columns, like §3 |
| `input_group.repeat` / `output_group.repeat` | integer inputs (M, K) | fixed during a run |
| `cavity.material` / `cavity.thickness_nm` | `material_input` + numeric | single layer |
| `cavity.enabled` | toggle/checkbox | off ⇒ omit cavity from expansion + sketch |
| `variable` | Tabs ("Per periodo" / "Per singolo strato") hosting the §11 selectors | default = "Per periodo", cavity only |

### 9.5 Validation additions (`validate_opt_stack_config` scope)

Grouped config adds (Italian messages): `repeat` must be int ≥ 0; at least one group with
repeat ≥ 1 *or* an enabled cavity (non-empty stack); group `layers` non-empty if its `repeat`
≥ 1; reuse §5 grid/thickness/material checks per layer. **Variable-selector checks are now in
§11.4** (mode-aware).

---

## 10. Mini-sketch of the multilayer — both tabs (Addition B)

A Plotly schematic giving an at-a-glance view of the stack. Pure builder in `plots.py`.
(Unchanged by §11.) Signature, visual encoding, n→color mapping, and placement as previously
specified: one `sketch_figure(stack_config, angle_deg, grouped, title)` with `grouped=False`
for the flat §2.2 dict and `grouped=True` for the grouped §9.1 dict (repeated groups collapsed
with ×M/×K labels, disabled cavity omitted, Re(n)→Viridis colorscale + colorbar).

---

## 11. Two optimization modes: "Per periodo" (tied) vs "Per singolo strato" (independent) — Addition C

The old §9.2 rule made only the **first repeat** of a selected period-layer variable
(`selected.add(i)`), so the other `M-1` / `K-1` copies stayed frozen — a silent footgun. §11
replaces the single period-level scheme with **two explicit, user-switchable optimization
modes**, chosen by a Tabs widget placed *inside* the existing "Spessori variabili
(ottimizzabili)" fieldset. The active tab is the mode that runs when "Ottimizza" is clicked.

1. **"Per periodo" (tied).** Reuses the existing period-level selectors (cavity checkbox +
   input/output period-layer checklists). Each selected period-layer is **one optimization
   variable shared across ALL its repetitions** (scatter-broadcast 1:N). The period stays
   uniform. This is the physically correct generalization of the old default (cavity-only tied
   → one variable).
2. **"Per singolo strato" (independent).** A single checklist over **every layer of the
   fully-expanded stack** (e.g. `2 layers × 10 periods = 20` entries, plus the cavity if
   enabled). Each selected flat layer is **its own variable**, optimized independently.

**Locked feasibility (no library change).** Both modes are expressible as a
`groups: list[list[int]]` where each inner list is a set of flat indices sharing one variable.
**Independent = every group is a singleton; tied = each group holds all flat repeats of a
period-layer.** The objective maps a reduced variable vector `v` (length `len(groups)`) to the
full thickness array by scatter-broadcast, then calls the unchanged library. This is the exact
`base.at[idx].set(...)` pattern already in `run_thickness_optimization`, generalized from a flat
index array to groups.

### 11.1 Store schema: `OPT_STACK_CONFIG_STORE["variable"]` (extended)

```python
"variable": {
   "mode": "tied",            # "tied" | "independent"  (default "tied")
   # --- tied-mode selectors (period-level; unchanged meaning) ---
   "cavity": true,            # default selection = cavity only (as today)
   "input_layers":  [],       # indices i into input_group.layers (period definition)
   "output_layers": [],       # indices j into output_group.layers (period definition)
   # --- independent-mode selector (NEW) ---
   "flat_layers": []          # selected EXPANDED flat indices (ints into the flat Stack.layers)
}
```

- **Defaults** (`config.default_opt_stack_config`): `mode="tied"`, `cavity=True`,
  `input_layers=[]`, `output_layers=[]`, `flat_layers=[]`. Identical runtime behavior to today
  (cavity-only, one variable).
- **Backward compatibility.** A stored `variable` dict lacking `"mode"` is read as `"tied"`; a
  dict lacking `"flat_layers"` is read as `[]`. `state.py` reads with `.get(..., default)` so old
  Stores keep working.
- **Per-mode key honoring.** `tied` reads `cavity`/`input_layers`/`output_layers`, ignores
  `flat_layers`. `independent` reads `flat_layers`, ignores the three period selectors. Both key
  sets are always persisted (toggling the tab does not lose the other mode's selection).

### 11.2 New ids (in `ids.py`)

```python
# Mode Tabs inside the "Spessori variabili" fieldset.
OPT_VARIABLE_MODE_TABS  = "opt_variable_mode_tabs"     # dcc.Tabs `value` = active mode
OPT_VARIABLE_MODE_TIED        = "tied"                 # tab value (matches store "mode")
OPT_VARIABLE_MODE_INDEPENDENT = "independent"          # tab value (matches store "mode")
# Independent-mode checklist over every expanded flat layer.
OPT_VARIABLE_FLAT_LAYERS_INPUT = "opt_variable_flat_layers_input"
```

**Mode source of truth (decided): mirror the Tabs `value` into the store; do not read the tab
directly in the run path.** The mode lives in `OPT_STACK_CONFIG_STORE["variable"]["mode"]`,
written by the **existing** `sync_opt_stack_config` callback (`stack_callbacks.py`), which gains
`Input(OPT_VARIABLE_MODE_TABS, "value")` and `Input(OPT_VARIABLE_FLAT_LAYERS_INPUT, "value")`.
Rationale: the run path (`optimize_callbacks.py`) and the expansion (`state.py`) already consume
**only** the grouped store; keeping mode + selections inside that one store preserves the
single-producer / single-consumer seam and means the background callback's existing
`State(OPT_STACK_CONFIG_STORE)` needs no new inputs.

### 11.3 New LABELS (in `config.py`, Italian)

```python
"opt_variable_mode_tied":        "Per periodo",
"opt_variable_mode_independent": "Per singolo strato",
"opt_variable_flat_layers":      "Strati ottimizzabili (stack espanso)",
```

(The existing `opt_variable_section`, `opt_variable_cavity`, `opt_variable_input_layers`,
`opt_variable_output_layers` labels are retained and shown inside the "Per periodo" tab.)

### 11.4 `state.py` contract (owner: `gui-core`)

Two new pure functions; `expand_optimization_config` is **refactored to delegate** to the new
`expand_optimization_variables` (kept as a thin compat shim so nothing else breaks).

```python
def enumerate_expanded_layers(opt_stack_config: dict) -> list[dict]:
    """One entry per fully-expanded flat layer (for the independent checklist options).

    Walks the SAME expansion order as expand_optimization_variables
    (input_group.layers × M | cavity iff enabled | output_group.layers × K) and
    returns, per flat layer, a JSON-safe dict:
        {
          "flat_index": int,            # index into the flat Stack.layers
          "label": str,                 # human label (Italian), e.g.
                                        #   "Ingresso · per. 2 · strato 1 (high_index)"
                                        #   "Cavità (cavity)"
                                        #   "Uscita · per. 1 · strato 2 (low_index)"
          "material_name": str | None,  # material dict "name"
          "thickness_nm": float,        # current period/cavity thickness
        }
    Label convention (1-based periods AND 1-based layer-in-period for the UI;
    flat_index stays 0-based for the backend):
        f"Ingresso · per. {m+1} · strato {p+1} ({name})"
        f"Cavità ({name})"
        f"Uscita · per. {k+1} · strato {p+1} ({name})"
    Pure; no Dash. Raises ValueError (Italian) on a malformed config, mirroring
    expand_optimization_variables' material/group checks.
    """

def expand_optimization_variables(
    opt_stack_config: dict,
) -> tuple[Stack, list[list[int]]]:
    """Expand the grouped §9.1 config into a flat Stack + variable GROUPS.

    The flat layer list is built IDENTICALLY to §9.2 (input × M | cavity iff
    enabled | output × K); only the variable-set construction changes by mode
    (opt_stack_config["variable"]["mode"], default "tied").

    Returns (stack, groups) where groups: list[list[int]] — one inner list per
    optimization VARIABLE; each inner list holds the flat indices that SHARE that
    variable. Invariants:
      * every inner list is non-empty and sorted ascending;
      * groups is sorted by each group's first flat index;
      * the union of all groups is the selected, de-duplicated variable set;
      * groups is non-empty (raise ValueError, Italian, if no selection).

    MODE == "tied":
      Let Lin = len(input_group.layers), Lout = len(output_group.layers),
      M = input_group.repeat, K = output_group.repeat,
      cavity_index = M*Lin, out_start = M*Lin + (1 if cavity.enabled else 0).
      For a selected INPUT period-layer i: group = [i, i+Lin, i+2*Lin, ...]  (M copies).
      For a selected OUTPUT period-layer j: group =
            [out_start+j, out_start+j+Lout, ...]                             (K copies).
      For variable.cavity (requires cavity.enabled): group = [cavity_index]  (singleton).
      => tied reproduces today's selection PLUS broadcasts to ALL repeats (the fix).

    MODE == "independent":
      Each selected flat index in variable.flat_layers becomes its OWN singleton
      group: groups = [[f] for f in sorted(set(flat_layers))].
      Validate every f in [0, num_flat_layers); raise ValueError (Italian) otherwise.
      variable.cavity / input_layers / output_layers are IGNORED in this mode.

    Pure; no Dash. Raises ValueError (Italian) for: empty selection; out-of-range
    flat index (independent); variable.cavity set while cavity disabled (tied);
    period-layer index out of range (tied).
    """

def expand_optimization_config(
    opt_stack_config: dict,
) -> tuple[Stack, tuple[int, ...]]:
    """DEPRECATED compat shim. Delegates to expand_optimization_variables and
    flattens groups back to the legacy sorted, de-duplicated variable_layer_indices
    tuple (union of all groups). Retained so any caller/test on the old signature
    keeps working; the run_* paths switch to groups directly."""
```

**How `run_*` consume `groups` (scatter-broadcast).** Both run functions switch from
`variable_layer_indices` to `groups`. The reduced→full mapping (JAX, inside the objective):

```python
import jax.numpy as jnp
base = stack_thicknesses(stack)                       # full base thickness vector
group_arrays = [jnp.asarray(g, dtype=jnp.int32) for g in groups]
initial_variable = jnp.asarray(                       # one initial per group
    [base[g[0]] for g in groups]                      # value of the group's first flat layer
)

def to_full(v):                                       # v: shape (len(groups),)
    full = base
    for gi, idx in enumerate(group_arrays):
        full = full.at[idx].set(v[gi])                # broadcast scalar v[gi] to all copies
    return full
```

- `run_thickness_optimization`: objective = `to_full(v) → stack_with_thicknesses → mean channel`
  (reuse `mean_reflectivity` for R), optimized via
  `optimize_thicknesses(objective, initial_variable, ...)`. Final full thicknesses =
  `to_full(optimized.thicknesses_nm)`.
- `run_resonance_optimization`: the library's `optimize_resonance_target` supports only the 1:1
  `variable_layer_indices` map (no sharing), so gui-core builds its **own** objective using the
  public `resonance_target_loss` over `to_full(v)` (identical math to the library's internal
  objective, `optimize.py:160`), then drives it with `optimize_thicknesses`; finally
  `analyze_resonance` on the optimized spectrum for the `resonance` field. (Independent mode with
  all-singleton groups MAY delegate to `optimize_resonance_target(..., variable_layer_indices=
  flat_tuple)` as a fast path, but the shared-objective path is the single required
  implementation and covers both modes.) Output dict shape is unchanged (§6.1).
- `variable_thicknesses_nm` now means **one value per variable/group** (length `len(groups)`),
  not one per flat layer — note this in `render_thicknesses` readout text.
- `state.py.__all__` gains `expand_optimization_variables` and `enumerate_expanded_layers`.

**Validation additions** (`validate_opt_stack_config`, §9.5 scope, Italian messages):
`variable.mode` must be `"tied"` or `"independent"` (default tied if absent); in independent mode
`variable.flat_layers` must be a non-empty list of in-range ints; in tied mode the existing
≥1-selection + cavity-enabled checks apply.

### 11.5 Ownership / sequence (no file collisions)

| File | Owner | §11 change |
|---|---|---|
| `app/state.py` | **gui-core** | add `enumerate_expanded_layers`, `expand_optimization_variables`; refactor `expand_optimization_config` to delegate; switch `run_resonance_optimization` / `run_thickness_optimization` to `groups`; extend `validate_opt_stack_config`; update `__all__`. |
| `app/ids.py` | **gui-frontend** | add `OPT_VARIABLE_MODE_TABS`, `OPT_VARIABLE_MODE_TIED`, `OPT_VARIABLE_MODE_INDEPENDENT`, `OPT_VARIABLE_FLAT_LAYERS_INPUT`. |
| `app/config.py` | **gui-frontend** | add the three §11.3 LABELS; add `mode`/`flat_layers` to `default_opt_stack_config()["variable"]`. |
| `app/components/optimize_panel.py` | **gui-frontend** | wrap the "Spessori variabili" body in `dcc.Tabs(id=OPT_VARIABLE_MODE_TABS, value="tied")` with two `dcc.Tab`s: "Per periodo" hosting the existing cavity/input/output checklists, "Per singolo strato" hosting `dcc.Checklist(id=OPT_VARIABLE_FLAT_LAYERS_INPUT, options=[])` (options filled by callback). |
| `app/callbacks/stack_callbacks.py` | **gui-core** | extend `sync_opt_stack_config`: add `Input(OPT_VARIABLE_MODE_TABS,"value")` + `Input(OPT_VARIABLE_FLAT_LAYERS_INPUT,"value")`; write `variable.mode` + `variable.flat_layers`. **Owns the dynamic-options callback below.** |
| `app/callbacks/optimize_callbacks.py` | **gui-frontend** | no signature change — already reads `OPT_STACK_CONFIG_STORE`; only the readout-label note. |

**Dynamic options for the independent checklist (single owner: gui-core).** A dedicated callback
in `stack_callbacks.py` regenerates the flat-layer checklist options whenever the expanded
structure changes:

```python
@app.callback(
    Output(OPT_VARIABLE_FLAT_LAYERS_INPUT, "options"),
    Input(OPT_STACK_CONFIG_STORE, "data"),     # fires after sync_opt_stack_config writes it
)
def _flat_layer_options(opt_stack_config):
    entries = state.enumerate_expanded_layers(opt_stack_config or {})
    return [{"label": e["label"], "value": e["flat_index"]} for e in entries]
```

gui-core owns this because the labels come from `state.enumerate_expanded_layers`; the callback
is the only place coupling that helper to a widget, and it lives in the same file as the store's
single producer (`sync_opt_stack_config`). It is driven by the **store** (not raw widgets) so it
always reflects the post-expansion truth (M, K, cavity on/off, table edits). gui-frontend only
declares the empty `dcc.Checklist` placeholder; it never builds option labels. No cross-file race
on the same store/widget.

**Sequence:** gui-core lands `state.py` (§11.4) + the two callbacks first (testable headless).
gui-frontend then adds ids, LABELS, and the Tabs layout against the frozen ids. gui-qa validates:
tied mode produces M-copy / K-copy groups (the fix); independent mode produces singleton groups
over the right flat indices; `enumerate_expanded_layers` labels match the example's 13-layer
expansion (cavity at flat_index 6); empty-selection and out-of-range errors raise Italian
`ValueError`; `expand_optimization_config` shim still returns the legacy tuple.

---

## 12. Internationalization (EN/IT, default ENGLISH) + help tooltips — Addition D

Full-app i18n with **English as the default and canonical baseline** (today everything is
Italian), plus a CSS-only help-tooltip system on the optimization panel, plus a dark-theme
label-contrast fix. This section is binding; it supersedes the "UI text is Italian" notes in
§0/§1/§6.6 and in the `config.py`/`ids.py`/`plots.py`/`state.py` docstrings (those docstrings get
reworded to "default English, IT available" by the owning agent when it touches the file).

Scope of translation (full coverage): all UI chrome (titles, tab labels, headings, field
labels, buttons, placeholders, dropdown/radio/checklist option text), **plus** plot axis /
legend / title / hover labels, sketch labels/annotations, the resonance-readout row keys, and
**validation/error messages** returned from `state.py`.

### 12.1 Decisions (the load-bearing choices)

**D1 — Catalog shape.** Replace the flat `config.LABELS` with a two-language nested catalog:

```python
# app/config.py  (owner: gui-frontend)
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": { ... },   # canonical baseline; authored first
    "it": { ... },   # must have the IDENTICAL key set as "en"
}
DEFAULT_LANG = "en"
SUPPORTED_LANGS = ("en", "it")

def labels_for(lang: str = DEFAULT_LANG) -> dict:
    """Return the full label dict for a language, EN-overlaid so a missing key can never KeyError.

    Unknown lang -> EN. Returns a NEW merged dict (EN under the requested language)."""
    base = TRANSLATIONS[DEFAULT_LANG]
    if lang == DEFAULT_LANG or lang not in TRANSLATIONS:
        return dict(base)
    return {**base, **TRANSLATIONS[lang]}

def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """Single-key accessor: t('app_title', 'it'). EN fallback then key-as-text for a missing key."""
    return labels_for(lang).get(key, TRANSLATIONS[DEFAULT_LANG].get(key, key))
```

- **Key-naming convention:** flat snake_case string keys, grouped by area via prefix
  (`general_*`/`tab_*`, `sim_*`, `opt_*`, `mat_*`, `grid_*`, `res_*`, `err_*`, `tip_*`). Existing
  keys keep their current names where possible to minimize churn (`app_title`, `tab_simulate`,
  `incident`, `optimize_run`, …). New keys use the prefixes above.
  **Invariant: `set(TRANSLATIONS["en"]) == set(TRANSLATIONS["it"])`** — gui-qa asserts this; CI
  fails on key drift.
- **`config.LABELS` is removed.** Every component that did `config.LABELS[k]` switches to a
  per-build `labels = config.labels_for(lang)` then `labels[k]` (§12.5). A short-lived alias
  `LABELS = labels_for("en")` MAY exist for one commit to avoid a flag-day, but its removal is the
  deliverable.

**D2 — Plot/sketch strings stay INSIDE `plots.py` (keep it pure).** `plots.py` must not import
`config` (that breaks its §6.2 purity invariant). It gets its OWN tiny catalog plus a `lang`
parameter on every public builder:

```python
# app/plots.py  (owner: gui-viz) — imports only plotly + numpy, unchanged purity
_PLOT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "x_axis": "Wavelength (nm)", "y_value": "Value",
        "ch_R": "Reflectance", "ch_T": "Transmittance", "ch_A": "Absorptance",
        "loss": "Loss", "opt_step": "Optimization step",
        "resonance": "Resonance", "half_max": "Half maximum", "half_max_inline": "half maximum",
        "sketch_title": "Multilayer schematic", "sketch_incident": "incident medium",
        "sketch_substrate": "substrate", "sketch_materials": "Materials",
        "sketch_angle": "angle θ", "lambda_res": "λ_res", "q_factor": "Q",
        # ... full set, see §12.3
    },
    "it": { ...identical key set, Italian values... },
}
```

Every builder gains a trailing keyword `lang: str = "en"`; each function resolves
`L = _PLOT_TRANSLATIONS.get(lang, _PLOT_TRANSLATIONS["en"])` at the top and replaces the current
module-level constants (`_X_LABEL`, `_CHANNEL_LABELS`, `_POL_LABELS`, `_SKETCH_*`) and the inline
strings (`"Risonanza"`, `"Mezza altezza"`, `"mezza altezza"`, `"Schema del multistrato"`,
`"Materiali"`, `"angolo θ"`, `"Valore"`, `"Perdita"`, `"Passo di ottimizzazione"`, `"λ_ris"`,
`"Q"`, the `"n/d"` token). `_CHANNEL_COLORS`, `_POL_DASH`, `_VIRIDIS`, and all geometry are NOT
translated. The current "Italian labels (hard requirement)" comment is replaced by "default
English; `lang` selects EN/IT".

*Rationale:* keeping `plots.py` self-contained preserves §6.2 purity (no new import, testable
headless), and a 1-parameter signature change is the smallest possible seam.

**D3 — `state.py` errors use a STATE-INTERNAL catalog keyed by `lang` (NOT key-passing to the
caller).** Lower-risk choice and justification:

- `state.py` returns *human strings today* and callbacks pass them straight to status `children`
  (`return no_update, str(exc)`; `" ".join(errors)`). Returning message *keys* instead would force
  every callback to translate and would break every `" ".join(errors)` aggregation in
  `run_simulation`/`run_resonance_optimization`/`run_thickness_optimization` — far more call sites,
  higher risk.
- Instead, the public message-producing functions gain a trailing `lang: str = "en"` and resolve
  against a private `_ERRORS = {"en": {...}, "it": {...}}` via a helper `_e(key, lang, **fmt)` that
  applies f-string-style interpolation (indices, ranges, field names). `state.py` stays Dash-free
  and **config-free** — it owns its own error catalog (mirrors D2's pattern for plots, so the two
  pure modules are consistent and neither imports `config`).
- Functions gaining `lang` (default `"en"`, **placed LAST** so existing positional calls keep
  working): `validate_config`, `validate_opt_stack_config`, `run_simulation`,
  `run_thickness_optimization`, `run_resonance_optimization`, `analyze_result`,
  `make_thickness_objective`, `material_from_dict`, `material_to_dict`, `parse_material_csv`,
  `stack_from_config`, `expand_optimization_variables`, `enumerate_expanded_layers`,
  `expand_optimization_config`. Private helpers (`_validate_material`, `_group_period_layers`,
  `_expand_geometry`, `_select_channel_1d`, `_reject_both_for_optimization`) take `lang` threaded
  from their public caller.
- `enumerate_expanded_layers` labels (e.g. `"Ingresso · per. 1 · strato 1 (...)"`) are user-facing
  and therefore localized via the same `_ERRORS`/label mechanism keyed by `lang`.

**D4 — Runtime mechanism: URL-query language + `serve_layout()` function + clientside reload.**
Chosen approach (the user's recommended baseline; validated against the code below).

- `app.layout` becomes a **function** `serve_layout()` (Dash calls it on every page load):

  ```python
  # app/main.py  (owner: gui-core)
  import flask
  def serve_layout():
      lang = flask.request.args.get("lang", config.DEFAULT_LANG)
      if lang not in config.SUPPORTED_LANGS:
          lang = config.DEFAULT_LANG
      return build_layout(lang)
  app.layout = serve_layout          # assign the FUNCTION object, NOT serve_layout()
  ```

  Today `app.layout = build_layout()` is a *value*; this changes to assigning the function so the
  language is read per-request. `suppress_callback_exceptions=True` is already set in
  `create_app()` (required when layout is a function), so no extra config is needed.
- `build_layout(lang: str = "en")` and EVERY `build_*` builder take `lang`, resolve
  `labels = config.labels_for(lang)` once, and thread `labels`/`lang` into children. `build_layout`
  also:
  - seeds `dcc.Store(id=ids.LANGUAGE_STORE, data=lang)` so callbacks read the active language via
    `State(LANGUAGE_STORE, "data")` without re-parsing the URL;
  - sets the header selector's `value` to `lang`;
  - passes `lang` into the initial `plots.sketch_figure(..., lang=lang)` / `plots.empty_figure(...)`
    calls embedded in components.
- **Header language selector** (`dcc.RadioItems`/`dcc.Dropdown`, EN/IT, default EN) lives in the
  header beside `H1`. Switching is a **clientside callback** that reloads with `?lang=`:

  ```python
  app.clientside_callback(
      """function(lang){
            if(!lang) return window.dash_clientside.no_update;
            var u = new URL(window.location.href);
            if(u.searchParams.get('lang') === lang) return window.dash_clientside.no_update;
            u.searchParams.set('lang', lang);
            window.location.href = u.toString();   // full reload -> serve_layout() re-runs
            return lang;
      }""",
      Output(ids.LANGUAGE_STORE, "data"),
      Input(ids.LANGUAGE_SELECTOR, "value"),
      prevent_initial_call=True,
  )
  ```

- **Accepted trade-off:** switching language is a full reload, so any in-progress, unsaved form
  edits reset to defaults (the layout is rebuilt fresh). Language switching is a rare action; this
  is acceptable and avoids a fragile re-translate-in-place machine. It does **not** lose
  `simulation_result_store` / `optimization_result_store` data *spuriously mid-run* — a reload is
  an explicit user action, equivalent to a page refresh. **All component ids and all existing
  callbacks are preserved**; only `lang` parameters are added and three new ids introduced.

  *Validated against the code:* `build_layout()` already builds the whole tree from `config.LABELS`
  + builders with no per-request state, so parameterizing it on `lang` is mechanical; `main.py`
  already sets `suppress_callback_exceptions=True` and builds the app in `create_app()` (the
  natural place to assign `app.layout = serve_layout`); no callback currently reads the URL, so the
  query-driven function layout introduces no conflict.

**D5 — Help tooltips: CSS-only, optimization panel ONLY, no new dependency.** Each optimize-field
label gets a small "?"-in-a-circle after the text; hovering the icon shows a CSS tooltip. No
`dash-bootstrap-components`, no JS — a single `app/assets/style.css` (Dash auto-serves
`app/assets/`). Tooltip text is translatable via the `tip_*` keys.

### 12.2 Complete key list (grouped by area) — `config.TRANSLATIONS`

`TRANSLATIONS["en"]` / `["it"]` must contain EXACTLY these keys (identical sets). Existing keys
retained; new keys marked **(new)**.

**general / app / tabs**
`app_title`, `tab_simulate`, `tab_optimize`, `lang_label` **(new)**, `lang_en` **(new)**,
`lang_it` **(new)**.

**stack builder (shared)**
`incident`, `substrate`, `layers`, `add_layer`, `material_kind`, `refractive_index_n`,
`extinction_k`, `material_name`, `upload_csv`, `upload_csv_hint`, `thickness_nm`, `grid_start`,
`grid_stop`, `grid_num`, `angle_deg`, `polarization`, `grid_section_legend` **(new — replaces the
mislabeled `LABELS["polarization"]` currently used as the `<Legend>` of the grid fieldset)**.

**option-list display text** (today the 2nd tuple element of `*_OPTIONS`; see §12.6)
`pol_s`, `pol_p`, `pol_both`, `feat_peak`, `feat_dip`, `ch_R_opt`, `ch_T_opt`, `ch_A_opt`,
`matkind_constant`, `matkind_csv`, `optmode_resonance`, `optmode_mean_r` — all **(new key names)**.

**simulate panel**
`simulate_run`, `simulate_channels`, `simulate_status_ready`, `simulate_status_running`,
`simulate_status_done`.

**optimize panel (chrome)**
`optimize_mode`, `optimize_target_wavelength`, `optimize_target_q`, `optimize_feature`,
`optimize_spectrum`, `optimize_variable_layers`, `optimize_steps`, `optimize_learning_rate`,
`optimize_lower_bound`, `optimize_wavelength_weight`, `optimize_q_weight`, `optimize_sharpness`,
`optimize_run`, `optimize_status_ready`, `optimize_status_running`, `optimize_status_done`,
`optimize_history`, `optimize_thicknesses`, `opt_stack_title`, `opt_input_group`,
`opt_output_group`, `opt_input_repeat`, `opt_output_repeat`, `opt_cavity`, `opt_cavity_enabled`,
`opt_cavity_thickness`, `opt_variable_section`, `opt_variable_cavity`, `opt_variable_input_layers`,
`opt_variable_output_layers`, `opt_variable_mode_tied`, `opt_variable_mode_independent`,
`opt_variable_flat_layers`, `sketch_title`.

**results / resonance readout**
`results`, `resonance`, `export`, `resonance_wavelength`, `linewidth`, `quality_factor`,
`extremum_value`, `empty_plot`, `res_table_metric` **(new — left column header of the resonance
readout, currently hardcoded as `config.LABELS["resonance"]`)**, `res_table_value` **(new —
replaces the hardcoded `"Valore"` column header in `results_panel.py`)**, `res_table_warning`
**(new — replaces the hardcoded `"Avviso"` row key in `simulate_callbacks.py`)**.

**tooltips (`tip_*`, NEW; in `config.TRANSLATIONS`)** — optimize fields in scope:
`tip_mode`, `tip_spectrum`, `tip_feature`, `tip_target_wavelength`, `tip_target_q`,
`tip_wavelength_weight`, `tip_q_weight`, `tip_sharpness`, `tip_steps`, `tip_learning_rate`,
`tip_lower_bound`, and the mode tabs `tip_variable_mode_tied` / `tip_variable_mode_independent`
(the eleven required fields plus the tied/independent mode tabs).

### 12.2b Error + expanded-layer keys — `state._ERRORS` (owned by gui-core, NOT in `config`)

**errors (`err_*`)** — one key per distinct message in `state.py` today (EN canonical, IT mirror).
Minimum set, named by content:
`err_stack_config_invalid`, `err_material_invalid`, `err_material_csv_missing_cols`,
`err_material_kind_unsupported`, `err_csv_no_content`, `err_csv_decode`, `err_csv_no_header`,
`err_csv_missing_cols`, `err_csv_non_numeric_row`, `err_csv_no_data_rows`, `err_incident_missing`,
`err_substrate_missing`, `err_layers_invalid`, `err_layer_invalid`, `err_thickness_negative`,
`err_thickness_non_numeric`, `err_grid_missing`, `err_grid_min_points`, `err_grid_start_ge_stop`,
`err_grid_params_invalid`, `err_angle_non_numeric`, `err_polarization_invalid`,
`err_channel_invalid`, `err_resonance_needs_single_pol`, `err_optimize_needs_single_pol`,
`err_group_invalid`, `err_group_period_layer_invalid`, `err_group_period_empty`,
`err_repeat_negative`, `err_repeat_not_int`, `err_stack_empty`, `err_cavity_disabled_but_variable`,
`err_variable_selector_invalid`, `err_variable_mode_invalid`, `err_select_one_variable`,
`err_select_one_variable_tied`, `err_flat_index_invalid`, `err_flat_index_out_of_range`,
`err_input_layer_out_of_range`, `err_output_layer_out_of_range`, `err_expanded_index_invalid`,
`err_material_not_serializable`. Interpolated values (layer index, group label, range bounds) go
through `_e(key, lang, **fmt)`.

**expanded-layer labels (`lbl_*`)** — used by `enumerate_expanded_layers`:
`lbl_input_layer` (EN template `"Input · per. {r} · layer {p} ({name})"`, IT
`"Ingresso · per. {r} · strato {p} ({name})"`), `lbl_output_layer`, `lbl_cavity`, plus fallback
name stems `lbl_fallback_input`, `lbl_fallback_output`, `lbl_fallback_cavity`, `lbl_fallback_layer`.

**gui-core authors both EN and IT for `_ERRORS`** (it owns `state.py`); gui-frontend does NOT touch
`_ERRORS`. The same-key-set invariant applies: `set(_ERRORS["en"]) == set(_ERRORS["it"])`.

### 12.3 `plots.py` `_PLOT_TRANSLATIONS` key list (gui-viz, self-contained)

`x_axis`, `y_value`, `ch_R`, `ch_T`, `ch_A`, `hover_value`, `loss`, `opt_step`, `resonance`,
`resonance_feature`, `half_max`, `half_max_inline`, `q_label`, `linewidth_label`, `na`,
`sketch_title`, `sketch_incident`, `sketch_substrate`, `sketch_colorbar` (`Re(n)` — kept for
symmetry), `sketch_materials`, `sketch_angle`, `lambda_res`, `q_factor`. Same key set for `"en"`
and `"it"`.

### 12.4 New ids (in `ids.py`, owner: gui-frontend)

```python
LANGUAGE_STORE = "language_store"            # dcc.Store seeded by build_layout(lang)
LANGUAGE_SELECTOR = "language_selector"      # header dcc.RadioItems / dcc.Dropdown (EN/IT)

# Help-icon ids: pattern-matching dict ids so one CSS rule + one factory cover all fields.
HELP_ICON_TYPE = "opt-help-icon"
def help_icon_id(field: str) -> dict:
    """Canonical id for an optimize-field help icon: {'type': 'opt-help-icon', 'field': field}."""
    return {"type": HELP_ICON_TYPE, "field": field}
```

The help icon is purely presentational (CSS `:hover` shows the tooltip); it needs **no callback**,
so the dict id is for uniqueness/styling only — never an `Input`/`Output`. `field` values match the
`tip_*` suffixes (e.g. `"target_wavelength"`, `"variable_mode_tied"`).

### 12.5 Threading `lang` into plots & state — exact wiring

**Component builders (gui-frontend)** — every builder gains `lang: str = "en"`, resolves
`labels = config.labels_for(lang)` once at the top, and replaces `config.LABELS[k]` with
`labels[k]`: `build_layout`, `build_stack_builder`, `build_material_input`, `build_simulate_panel`,
`build_optimize_panel`, `build_results_panel`, and the private helpers (`_grouped_stack_editor`,
`_optimize_controls`, `_num`, `_group_table`, `_variable_layer_options`, `_layer_table`). Initial
embedded figures pass `lang=lang` into `plots.sketch_figure` / `plots.empty_figure`.
`build_material_input(id_prefix, label)` keeps its explicit `label` arg (caller already passes a
localized string); option text inside it uses `config.options_for(..., lang)`.

**Callbacks gaining `State(ids.LANGUAGE_STORE, "data")`** (owner: gui-core), each then passing
`lang` into the `plots.*` / `state.*` call (default `"en"` if the store is `None`):

| Callback (file) | Adds State(LANGUAGE_STORE) | Threads lang into |
|---|---|---|
| `simulate_callbacks.run_simulation` | yes | `state.run_simulation(cfg, lang=lang)`; status text from `labels` |
| `simulate_callbacks.render_spectrum` | yes | `plots.spectrum_figure(..., lang=lang)`, `plots.empty_figure(...)` |
| `simulate_callbacks.render_resonance_readout` | yes | `state.analyze_result(..., lang=lang)`; row keys from `labels`; warning key |
| `optimize_callbacks.run_optimization` | yes | `state.run_resonance_optimization` / `run_thickness_optimization(..., lang=lang)`; see background caveat below |
| `optimize_callbacks.render_history` | yes | `plots.history_figure(..., lang=lang)`, title from `labels` |
| `optimize_callbacks.render_final_spectrum` | yes | `plots.resonance_overlay_figure(..., lang=lang)` / `plots.spectrum_figure(..., lang=lang)` |
| `optimize_callbacks.render_thicknesses` | yes | readout label from `labels` |
| `sketch_callbacks.update_simulate_sketch` | yes | `plots.sketch_figure(..., lang=lang)` |
| `sketch_callbacks.update_optimize_sketch` | yes | `plots.sketch_figure(..., lang=lang)` |
| `main._register_material_uploads.store_uploaded_material` | yes | `state.parse_material_csv(..., lang=lang)`; success label from `labels` |

> **Background-callback caveat (gui-core).** `run_optimization` declares
> `running=[(Output(OPTIMIZE_STATUS,"children"), <start>, <done>)]` whose two strings are baked in
> at *registration* time, before any request — they cannot read `LANGUAGE_STORE`. Decision: at
> registration the running-status strings default to **English** (`labels_for("en")`); the final
> localized "done"/error status is written by the callback body's return value (which DOES read
> `LANGUAGE_STORE`) and overrides the transient. Documented known minor seam: the transient
> "running…" toast is always English; the settled status is localized. (Adding a library step hook
> is out of scope and forbidden.)

**`plots.*` signatures (gui-viz)** — append `lang: str = "en"`:
`spectrum_figure(result_dict, channels=("R","T","A"), title=None, lang="en")`,
`history_figure(history, title=None, lang="en")`,
`resonance_overlay_figure(result_dict, resonance, channel="R", summary_in_legend=False, lang="en")`,
`sketch_figure(stack_config, angle_deg=0.0, grouped=False, title=None, lang="en")`,
`empty_figure(message="", lang="en")` (lang reserved; `message` is already passed translated by the
caller, so `empty_figure` may ignore `lang` — but accepts it for signature symmetry).

**`state.*` signatures (gui-core)** — append `lang: str = "en"` LAST on every public
message-producing function in §12.1-D3 plus `enumerate_expanded_layers`/
`expand_optimization_variables` (so labels/messages localize). `result_to_dict`,
`grid_from_config`, `_to_float`, `_to_list` do NOT change (no human strings).

### 12.6 Option-list display text

`POLARIZATION_OPTIONS` / `FEATURE_OPTIONS` / `SPECTRUM_OPTIONS` / `MATERIAL_KIND_OPTIONS` /
`OPTIMIZE_MODE_OPTIONS` currently bake Italian display text into the second tuple element. Change:
keep the `*_VALUES` tuples as the single source of option *values* (English snake_case, unchanged —
they feed the library), but move display text into the catalog under the `pol_*` / `feat_*` /
`ch_*_opt` / `matkind_*` / `optmode_*` keys. Provide a helper:

```python
# config.py
def options_for(values: tuple[str, ...], key_prefix: str, lang: str = "en") -> list[dict]:
    labels = labels_for(lang)
    return [{"label": labels[f"{key_prefix}{v}"], "value": v} for v in values]
```

Components call e.g. `config.options_for(config.POLARIZATION_VALUES, "pol_", lang)`. DataTable
column `name`s and the kind-dropdown options inside the tables localize the same way (`labels[...]`
/ `options_for`).

### 12.7 `app/assets/style.css` (owner: gui-frontend) — tooltips + dark-theme label contrast

Dash auto-serves any file in `app/assets/`. This single new file covers BOTH requirements.

**(a) Help tooltip (CSS-only).** Markup the optimize-field rows render (gui-frontend) per field:

```html
<span class="opt-label">{label}
  <span class="help" tabindex="0">?
    <span class="help-text" role="tooltip">{tip_text}</span>
  </span>
</span>
```

CSS contract:
```css
.help { display:inline-flex; align-items:center; justify-content:center;
        width:1.05em; height:1.05em; margin-left:.4em; border-radius:50%;
        border:1px solid currentColor; font-size:.75em; cursor:help; position:relative; }
.help-text { visibility:hidden; opacity:0; position:absolute; z-index:1000;
             left:50%; bottom:135%; transform:translateX(-50%);
             min-width:180px; max-width:280px; padding:.5em .6em;
             background:#1f2733; color:#f3f5f7; border:1px solid #3a4658;
             border-radius:6px; font-size:.85rem; line-height:1.25; white-space:normal;
             box-shadow:0 4px 14px rgba(0,0,0,.4); transition:opacity .12s ease; }
.help:hover .help-text, .help:focus .help-text,
.help:focus-within .help-text { visibility:visible; opacity:1; }
```
`tabindex` + `:focus`/`:focus-within` gives keyboard/touch access without JS. The tooltip text is
the translated `tip_*` string injected by the builder; only the icon glyph `?` is static.

**(b) Dark-theme label/legend/heading contrast fix.** The pending bug: `<label>`/`<legend>`/
headings render dark-on-dark. **Selector strategy (documented, exact):** scope to the app root and
target the structural form/heading elements (not per-component classes) so future components
inherit the fix automatically:

```css
.app-root label,
.app-root legend,
.app-root h1, .app-root h2, .app-root h3, .app-root h4, .app-root h5,
.app-root .status-text,
.app-root .thickness-readout { color:#e8ebef; }
.app-root legend { font-weight:600; }
/* DataTable + dropdown menus render their own light surfaces; do NOT override their text color
   (they are dark-on-light by design). Scope ONLY bare label/legend/headings. */
```
`.app-root` is the existing wrapper class on `build_layout`'s root `html.Div`. This is the single
contrast authority; components must not set inline label colors. In-figure plot/sketch text colors
are handled inside `plots.py` and are out of CSS scope.

### 12.8 File ownership / build order (no collisions)

| File | Owner | §12 responsibility |
|---|---|---|
| `app/config.py` | **gui-frontend** | replace `LABELS` with `TRANSLATIONS` + `DEFAULT_LANG`/`SUPPORTED_LANGS` + `labels_for`/`t`/`options_for`; author full EN **and** IT catalogs incl. all `tip_*`; move option display text out of `*_OPTIONS` into catalog keys. **Does NOT touch `state._ERRORS`.** |
| `app/ids.py` | **gui-frontend** | add `LANGUAGE_STORE`, `LANGUAGE_SELECTOR`, `HELP_ICON_TYPE`, `help_icon_id()`. |
| `app/components/*` | **gui-frontend** | add `lang` param to every builder; resolve `labels_for(lang)`; render the header language selector (header builder) and the `?`-icon + tooltip markup on the eleven optimize fields + the two mode tabs; pass `lang` into embedded `plots.*` calls. |
| `app/assets/style.css` | **gui-frontend** | NEW: tooltip CSS (§12.7a) + label/legend/heading contrast fix (§12.7b). |
| `app/layout.py` | **gui-core** | `build_layout(lang="en")` (thread `lang`); seed `LANGUAGE_STORE`; place the header language-selector slot. (gui-frontend authors the selector widget; gui-core wires it into the signature + slot.) |
| `app/main.py` | **gui-core** | add `serve_layout()` reading `flask.request.args`; `app.layout = serve_layout`; register the clientside language-toggle callback; thread `lang` into `_register_material_uploads`. |
| `app/callbacks/*` | **gui-core** | add `State(LANGUAGE_STORE,"data")` to the callbacks in §12.5; pass `lang` into `state.*`/`plots.*`; resolve status strings via `labels_for(lang)`; handle the background `running=[...]` EN-default caveat. |
| `app/plots.py` | **gui-viz** | add `_PLOT_TRANSLATIONS` (EN+IT, §12.3); add `lang="en"` to all builders; replace module-level Italian constants/inline strings with per-call lookups; default EN. |
| `app/state.py` | **gui-core** | add private `_ERRORS = {"en":..,"it":..}` + `_e()`; author EN+IT for every `err_*`/`lbl_*`; add `lang="en"` LAST to all public message-producing functions and thread to private helpers; default EN. |
| `tests/test_app_gui.py`, `tests/test_app_feature_c.py` | **gui-qa** | see §12.9. |

> **Single-writer per file.** The only file two roles "share" is `layout.py`: gui-core owns the
> `lang`-threaded signature + `LANGUAGE_STORE` + selector slot; gui-frontend authors the selector
> widget and passes it into the reserved slot. Resolve by gui-core landing the skeleton first,
> gui-frontend filling the slot. No file is edited by two agents concurrently.

**Build order:**
1. **gui-frontend** lands `config.py` (`TRANSLATIONS` EN+IT, accessors, `options_for`, `tip_*`) and
   `ids.py` (new ids) — the frozen catalog + id contract everyone references.
2. **gui-viz** lands `plots.py` (`_PLOT_TRANSLATIONS` + `lang` params, §12.3) — testable headless,
   no dependency on config.
3. **gui-core** lands `state.py` (`_ERRORS` + `lang` params) — testable headless; then
   `layout.py`/`main.py` (`serve_layout`, `LANGUAGE_STORE`, clientside toggle) and the
   `callbacks/*` `lang` threading.
4. **gui-frontend** lands `components/*` (`lang` params, header selector, `?` icons) and
   `app/assets/style.css` against the frozen ids/catalog.
5. **gui-qa** lands tests last (§12.9).

### 12.9 Test adaptation (owner: gui-qa)

English is now the default, so any test calling `state.*` / `plots.*` without `lang=` gets
**English** strings. The existing Italian-substring assertions WILL FAIL unmodified
(`"negativo"`, `"iniziale"`, `"Polarizzazione"`, `"2 punti"`, `"variabile"`/`"Selezionare"`, plot
`"Risonanza"`, sketch `"mezzo incidente"`/`"substrato"`, status `"in corso"`/`"Pronto"`, …).
Binding policy:

- **Make existing assertions language-explicit, EN-canonical.** Rewrite each Italian-substring
  assertion to the **English** baseline, e.g.
  `assert any("negative" in e for e in state.validate_config(cfg))`,
  `… start" / "less than" …` for the start≥stop check, `"polarization"` for the pol check,
  `"2 points"` for the grid check, and plot trace `name == "Resonance"`.
- **Add IT-coverage twins** by passing `lang="it"`, e.g. `state.validate_config(cfg, lang="it")` →
  `assert any("negativo" in e ...)`; `plots.resonance_overlay_figure(..., lang="it")` → name
  `"Risonanza"`. Keeps Italian regression-covered without default-language ambiguity.
- **`"Risonanza"` exact-name tests** (in `test_app_gui.py` and `test_app_feature_c.py`, which assert
  a trace named exactly `"Risonanza"`): update to the EN default `"Resonance"`, plus an IT twin.
- **New invariant tests:** `set(config.TRANSLATIONS["en"]) == set(config.TRANSLATIONS["it"])`, and
  the same for `plots._PLOT_TRANSLATIONS` and `state._ERRORS`/`lbl_*` — fail on key drift.
- **New mechanism tests (headless, no browser):** `config.labels_for("it")["app_title"]` is
  Italian; `labels_for("xx")` falls back to EN; `t("missing_key","it")` returns the key;
  `serve_layout()` under a faked `flask.request` with `?lang=it` yields a tree whose
  `LANGUAGE_STORE` data is `"it"` and whose `H1` is the Italian title.
- **Tooltip/contrast** are CSS-only and not unit-tested for rendering; gui-qa may assert the
  optimize panel emits a help node per in-scope field (presence of `HELP_ICON_TYPE` dict ids) and
  that `app/assets/style.css` exists. Visual verification is via Dash **Preview** (not Interceptor).
