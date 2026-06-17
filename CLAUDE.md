# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev,opt]"   # editable install with pytest + optax
python -m pytest               # full test suite (testpaths=tests, addopts=-q)
python -m pytest tests/test_gradients.py::test_name   # single test
python examples/basic_stack.py # run an example end-to-end
```

`optax` is optional; without it `optimize_thicknesses` falls back to plain gradient descent.

## Architecture

JAX package for coherent transfer-matrix (TMM) simulation of multilayer thin films. Computes wavelength-dependent `R`, `T`, `A = 1 - R - T`, and complex amplitudes `r`, `t`. All public API is re-exported from `multilayer_tmm/__init__.py`.

Data flow, user-facing → kernel:

1. **`materials.py`** — `Material` is a frozen dataclass with three kinds (`constant`, `callable`, `tabulated`). Calling a `Material` evaluates `n + i k` at given wavelengths. Material evaluation happens *outside* the jitted kernel so the kernel only ever sees plain arrays.
2. **`layers.py`** — `Layer` (material + thickness) and `Stack` (incident medium, finite `layers`, substrate; incident/substrate are semi-infinite). `stack_with_thicknesses` rebuilds a stack with new thicknesses — the seam used by optimization.
3. **`tmm.py`** — core. `simulate_spectrum(stack, ...)` is the friendly entry; it calls `stack_to_arrays` then `simulate_spectrum_arrays(n_by_wavelength, thicknesses_nm, ...)`, the functional JAX-compatible interface for optimization loops. `n_by_wavelength` has shape `(num_wavelengths, num_layers + 2)`. The jitted kernel `_simulate_one_polarization` is `jax.vmap`'d over wavelength; `_coherent_tmm_single` builds characteristic matrices and folds them with `jax.lax.scan`.
4. **`resonance.py`** — two paths that must stay consistent: `analyze_resonance` (NumPy, discrete FWHM, for *reporting*) and `smooth_resonance_metrics` (JAX, soft spectral moments, *differentiable*, for optimization objectives).
5. **`optimize.py`** — `optimize_thicknesses` (generic gradient loop) and `optimize_resonance_target` (tunes selected `variable_layer_indices` toward target wavelength + Q via `resonance_target_loss`, which uses the smooth metrics).
6. **`io.py`** — `export_simulation` writes 6 timestamped files (structure.txt, spectra.txt, 4 PNGs). `utils.py` — JAX device inspection + `wavelength_grid`.

## Conventions and invariants

- **Refractive index convention is `n + i k`** with positive `k` lossy under `exp(i ω t)`. The negative imaginary sign in `_characteristic_matrices` keeps absorbing films passive — do not flip it.
- **Differentiability is load-bearing.** Layer thicknesses and callable materials are differentiable through `simulate_spectrum`. Keep new core code JAX-traceable (no Python branching on traced values, no NumPy inside the kernel). `_safe_complex` / `_safe_real` guard divisions to keep gradients finite.
- **Polarization** `"s"`/`"p"`/`"both"`. For `"both"`, `R/T/A/r/t` gain a leading axis of size 2 in `("s","p")` order; optimization helpers reject 2-row spectra (pass a single polarization).
- Transmission uses the power-flux correction `T = Re(Y_sub)/Re(Y_inc) * |t|^2`, not just `|t|^2`.
- Everything is in nanometers. Wavelength arrays must be 1-D.

Not a git repo. `docs/codex-history.md` is a build log, not API docs.
