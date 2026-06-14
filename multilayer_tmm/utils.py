"""Utilities for inspecting the JAX runtime."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def available_devices() -> list[jax.Device]:
    """Return devices visible to JAX."""

    return list(jax.devices())


def print_jax_devices() -> None:
    """Print available JAX devices."""

    for device in available_devices():
        print(device)


def wavelength_grid(start_nm: float, stop_nm: float, num: int) -> jnp.ndarray:
    """Create an evenly spaced wavelength grid in nanometers."""

    return jnp.linspace(start_nm, stop_nm, num)
