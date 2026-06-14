"""User-facing layer and stack containers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import jax.numpy as jnp

from multilayer_tmm.materials import Material


@dataclass(frozen=True)
class Layer:
    """A finite layer in a coherent multilayer stack."""

    material: Material
    thickness_nm: Any


@dataclass(frozen=True)
class Stack:
    """A multilayer thin-film stack.

    ``incident`` and ``substrate`` are semi-infinite media. Only entries in
    ``layers`` are finite and require thicknesses.
    """

    incident: Material
    layers: Sequence[Layer]
    substrate: Material

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def materials(self) -> tuple[Material, ...]:
        return (
            self.incident,
            *(layer.material for layer in self.layers),
            self.substrate,
        )


def stack_thicknesses(stack: Stack, dtype: Any | None = None) -> jnp.ndarray:
    """Return finite-layer thicknesses as a JAX array."""

    return jnp.asarray([layer.thickness_nm for layer in stack.layers], dtype=dtype)


def stack_with_thicknesses(stack: Stack, thicknesses_nm: Any) -> Stack:
    """Return a new stack with replaced finite-layer thicknesses."""

    thicknesses = tuple(jnp.ravel(jnp.asarray(thicknesses_nm)))
    if len(thicknesses) != stack.num_layers:
        raise ValueError(
            f"Expected {stack.num_layers} thickness values, got {len(thicknesses)}."
        )
    return Stack(
        incident=stack.incident,
        layers=[
            replace(layer, thickness_nm=thickness)
            for layer, thickness in zip(stack.layers, thicknesses, strict=True)
        ],
        substrate=stack.substrate,
    )
