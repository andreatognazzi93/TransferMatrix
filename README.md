# multilayer-tmm

`multilayer-tmm` is a small JAX package for coherent transfer-matrix simulations of multilayer thin-film stacks. It computes wavelength-dependent reflectivity `R`, transmission `T`, absorption `A = 1 - R - T`, and the complex amplitudes `r` and `t`.

## Features

- Arbitrary incident medium, finite layers, and substrate
- Complex refractive indices with the `n + i k` convention
- Constant, callable, and CSV-tabulated dispersive materials
- `s`, `p`, or both polarizations
- Complex Snell-law angles in absorbing media
- JAX `vmap` over wavelength and `jit` compiled transfer-matrix kernels
- Differentiable layer thicknesses and differentiable callable materials
- Simple optimization helper using Optax when installed, otherwise gradient descent

## Install

```bash
pip install -e ".[dev,opt]"
```

`optax` is optional. Without it, optimization falls back to basic gradient descent.

## Graphical Interface (GUI)

The project ships with an interactive [Dash](https://dash.plotly.com/) app for building stacks, running simulations, and tuning resonances in the browser — no Python scripting required.

### 1. Get the project

```bash
git clone https://github.com/andreatognazzi93/TransferMatrix.git
cd TransferMatrix
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install with the GUI extras

```bash
pip install -e ".[gui]"
```

This installs the library plus the GUI dependencies (`dash`, `plotly`, `diskcache`, `pandas`, `multiprocess`). Add `opt` if you also want Optax-backed optimization: `pip install -e ".[gui,opt]"`.

### 4. Launch the app

```bash
python -m app
```

Then open the URL printed in the terminal — by default <http://127.0.0.1:8050>. The server runs locally on your own machine; nothing is uploaded anywhere.

To stop it, press `Ctrl+C` in the terminal.

### Using the GUI

- **Build a stack** — set the incident medium and substrate, then add finite layers (material + thickness in nm). Materials can be constant `n + i k`, or uploaded as a CSV (`wavelength_nm,n,k`).
- **Simulate** — choose the wavelength range, angle, and polarization, then run to see interactive `R`, `T`, and `A` spectra.
- **Optimize** — on the optimization tab, target a resonance wavelength and Q and let the app tune selected layer thicknesses. Long optimization runs use a background worker, so the UI stays responsive.
- Plots are interactive (zoom, pan, hover); results can be exported from the app.

> **Production deployment:** `python -m app` starts Dash's development server, which is fine for local single-user use. To serve it to others, run it behind a WSGI server, e.g. `gunicorn app.main:server` inside a container (Render, Railway, Fly.io, or Hugging Face Spaces are good fits).

## Basic Usage

```python
import jax.numpy as jnp

from multilayer_tmm import Layer, Material, Stack, export_simulation, simulate_spectrum

air = Material.constant(1.0 + 0.0j)
glass = Material.constant(1.5 + 0.0j)
metal = Material.from_csv("gold.csv")

stack = Stack(
    incident=air,
    layers=[
        Layer(material=metal, thickness_nm=50.0),
        Layer(material=glass, thickness_nm=100.0),
    ],
    substrate=glass,
)

wavelengths = jnp.linspace(400.0, 900.0, 1000)
result = simulate_spectrum(
    stack,
    wavelengths_nm=wavelengths,
    angle_deg=30.0,
    polarization="s",
)

paths = export_simulation(
    stack=stack,
    result=result,
    output_dir="results",
    simulation_name="gold_glass_stack",
    show=True,
)
```

You can also create a wavelength range with:

```python
from multilayer_tmm import wavelength_grid

wavelengths = wavelength_grid(start_nm=400.0, stop_nm=900.0, num=1000)
```

For `polarization="both"`, `R`, `T`, `A`, `r`, and `t` have shape `(2, num_wavelengths)` in `("s", "p")` order.

## Exported Outputs

`export_simulation()` writes six connected files with the same simulation prefix:

- `gold_glass_stack_YYYYMMDD_HHMMSS_structure.txt`
- `gold_glass_stack_YYYYMMDD_HHMMSS_spectra.txt`
- `gold_glass_stack_YYYYMMDD_HHMMSS_reflectivity.png`
- `gold_glass_stack_YYYYMMDD_HHMMSS_transmission.png`
- `gold_glass_stack_YYYYMMDD_HHMMSS_absorption.png`
- `gold_glass_stack_YYYYMMDD_HHMMSS_RTA.png`

For one polarization, the spectra file has four columns:

```text
wavelength_nm R T A
```

The structure file records the incident medium, each finite layer, the substrate, finite-layer thicknesses, and refractive indices evaluated at a reference wavelength.

Pass `show=True` to display a combined figure with three stacked subplots for `R`, `T`, and `A` while still saving all output files. Matplotlib must be using an interactive backend such as `macosx`, `QtAgg`, or `TkAgg`; non-interactive backends such as `Agg` will save files only.

## Material Models

Constant material:

```python
mgf2 = Material.constant(1.38 + 0.0j)
```

JAX-callable material:

```python
def cauchy(wavelength_nm):
    wavelength_um = wavelength_nm / 1000.0
    return 1.45 + 0.004 / wavelength_um**2

coating = Material.from_callable(cauchy)
```

CSV material:

```csv
wavelength_nm,n,k
400,1.47,1.95
500,0.97,1.87
600,0.24,3.07
```

```python
gold = Material.from_csv("gold.csv")
```

CSV interpolation uses `jax.numpy.interp`, so the simulation path remains backend-compatible after the file has been loaded.

## Gradients

Layer thickness can be differentiated directly:

```python
import jax
import jax.numpy as jnp

from multilayer_tmm import Layer, Material, Stack, simulate_spectrum

air = Material.constant(1.0)
mgf2 = Material.constant(1.38)
glass = Material.constant(1.52)
wavelengths = jnp.linspace(400.0, 700.0, 101)

def reflectivity_at_550(thickness_nm):
    stack = Stack(
        incident=air,
        layers=[Layer(mgf2, thickness_nm)],
        substrate=glass,
    )
    return simulate_spectrum(stack, jnp.array([550.0])).R[0]

print(jax.grad(reflectivity_at_550)(100.0))
```

Mean reflectivity over a wavelength band:

```python
def mean_reflectivity(thickness_nm):
    stack = Stack(
        incident=air,
        layers=[Layer(mgf2, thickness_nm)],
        substrate=glass,
    )
    return jnp.mean(simulate_spectrum(stack, wavelengths).R)

value, gradient = jax.value_and_grad(mean_reflectivity)(100.0)
```

## Resonance And Q

Use `analyze_resonance()` to calculate the dominant peak or dip wavelength and quality factor from a reflectivity, transmission, or absorption spectrum:

```python
from multilayer_tmm import analyze_resonance

resonance = analyze_resonance(
    result.wavelength_nm,
    result.R,
    feature="peak",
)

print(resonance.resonance_wavelength_nm)
print(resonance.linewidth_nm)
print(resonance.quality_factor)
```

By default, the linewidth is measured at half-prominence using the sampled spectral baseline. If the off-resonance baseline is known, pass it explicitly:

```python
resonance = analyze_resonance(wavelengths, transmission, feature="peak", baseline=0.0)
```

For optimization, use `optimize_resonance_target()` to tune one or more layer thicknesses toward a target resonance wavelength and Q. Internally it uses differentiable smooth spectral moments so JAX can compute gradients:

```python
from multilayer_tmm import optimize_resonance_target

optimization = optimize_resonance_target(
    stack,
    wavelengths_nm=wavelengths,
    target_wavelength_nm=620.0,
    target_q=15.0,
    spectrum="T",
    feature="peak",
    variable_layer_indices=(0,),
    steps=60,
    learning_rate=0.2,
)
```

See [examples/dbr_benchmark.py](examples/dbr_benchmark.py) for the supplied distributed Bragg reflector benchmark and [examples/optimize_resonance_target.py](examples/optimize_resonance_target.py) for target-resonance optimization.

## Physics Notes

The core implements the coherent characteristic-matrix method. Complex angles are computed from Snell's law,

```text
n0 sin(theta0) = nj sin(thetaj)
```

with a forward-propagating square-root branch for `cos(thetaj)`. Optical admittance is

```text
Y_s = n cos(theta)
Y_p = cos(theta) / n
```

and transmission uses the power-flux correction

```text
T = Re(Y_substrate) / Re(Y_incident) * |t|^2
```

Reflectivity is `R = |r|^2`.

## JAX Devices

```python
from multilayer_tmm import print_jax_devices

print_jax_devices()
```

JAX will use available CPU, GPU, or accelerator devices according to the installed JAX backend.

## Run Tests

```bash
python -m pytest
```
