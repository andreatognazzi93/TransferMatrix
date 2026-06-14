"""Material models for wavelength-dependent complex refractive index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax.numpy as jnp

ArrayLike = Any
MaterialKind = Literal["constant", "callable", "tabulated"]
IndexCallable = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class Material:
    """Complex refractive-index model evaluated in nanometers.

    The core simulation receives already-evaluated JAX arrays, so material
    evaluation stays outside the jitted transfer-matrix kernel. Callable
    materials can still be differentiable when they are written with JAX
    operations and close over differentiable parameters.
    """

    kind: MaterialKind
    data: Any
    name: str | None = None

    @classmethod
    def constant(cls, refractive_index: complex | float, name: str | None = None) -> "Material":
        """Create a material with wavelength-independent ``n + i k``."""

        return cls(kind="constant", data=refractive_index, name=name)

    @classmethod
    def from_callable(cls, fn: IndexCallable, name: str | None = None) -> "Material":
        """Create a material from ``fn(wavelength_nm) -> complex n``."""

        return cls(kind="callable", data=fn, name=name)

    @classmethod
    def from_table(
        cls,
        wavelength_nm: ArrayLike,
        n: ArrayLike,
        k: ArrayLike | float = 0.0,
        name: str | None = None,
    ) -> "Material":
        """Create a tabulated material with linear interpolation.

        Values outside the tabulated wavelength range use JAX's endpoint
        behavior for ``jnp.interp``.
        """

        wavelength_nm_array = jnp.asarray(wavelength_nm)
        n_array = jnp.asarray(n)
        k_array = jnp.asarray(k)
        if k_array.ndim == 0:
            k_array = jnp.full_like(n_array, k_array)
        order = jnp.argsort(wavelength_nm_array)
        return cls(
            kind="tabulated",
            data=(
                wavelength_nm_array[order],
                n_array[order],
                k_array[order],
            ),
            name=name,
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        wavelength_col: str = "wavelength_nm",
        n_col: str = "n",
        k_col: str = "k",
        name: str | None = None,
    ) -> "Material":
        """Load a tabulated material from a CSV file.

        The CSV must include columns named ``wavelength_nm``, ``n``, and ``k``
        by default. Use the column-name arguments for files with different
        headers.
        """

        from multilayer_tmm.io import read_material_csv

        wavelength_nm, n, k = read_material_csv(
            path,
            wavelength_col=wavelength_col,
            n_col=n_col,
            k_col=k_col,
        )
        return cls.from_table(wavelength_nm=wavelength_nm, n=n, k=k, name=name)

    def refractive_index(self, wavelength_nm: ArrayLike) -> jnp.ndarray:
        """Evaluate complex refractive index at one or more wavelengths."""

        wavelengths = jnp.asarray(wavelength_nm)
        if self.kind == "constant":
            value = jnp.asarray(self.data)
            values = jnp.ones_like(wavelengths, dtype=jnp.result_type(value, 1.0j)) * value
            return values

        if self.kind == "callable":
            raw = self.data(wavelengths)
            values = jnp.asarray(raw)
            if values.ndim == 0:
                values = jnp.broadcast_to(values, wavelengths.shape)
            return values

        if self.kind == "tabulated":
            table_wavelength_nm, table_n, table_k = self.data
            n = jnp.interp(wavelengths, table_wavelength_nm, table_n)
            k = jnp.interp(wavelengths, table_wavelength_nm, table_k)
            return n + 1.0j * k

        raise ValueError(f"Unknown material kind: {self.kind!r}")

    def __call__(self, wavelength_nm: ArrayLike) -> jnp.ndarray:
        return self.refractive_index(wavelength_nm)
