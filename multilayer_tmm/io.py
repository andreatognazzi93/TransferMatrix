"""Input and output helpers for material data and simulation results."""

from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from multilayer_tmm.layers import Stack
from multilayer_tmm.tmm import SimulationResult


@dataclass(frozen=True)
class ExportPaths:
    """Paths written by ``export_simulation``."""

    structure_txt: Path
    spectra_txt: Path
    reflectivity_plot: Path
    transmission_plot: Path
    absorption_plot: Path
    combined_plot: Path


def read_material_csv(
    path: str | Path,
    wavelength_col: str = "wavelength_nm",
    n_col: str = "n",
    k_col: str = "k",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Read ``wavelength_nm,n,k`` columns from a CSV file."""

    rows: list[tuple[float, float, float]] = []
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path!s} has no header row.")

        missing = {
            column
            for column in (wavelength_col, n_col, k_col)
            if column not in reader.fieldnames
        }
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"CSV file {path!s} is missing columns: {missing_list}")

        for row in reader:
            rows.append(
                (
                    float(row[wavelength_col]),
                    float(row[n_col]),
                    float(row[k_col]),
                )
            )

    if not rows:
        raise ValueError(f"CSV file {path!s} contains no material rows.")

    wavelength_nm, n, k = zip(*rows, strict=True)
    return jnp.asarray(wavelength_nm), jnp.asarray(n), jnp.asarray(k)


def read_spectrum_csv(
    path: str | Path,
    wavelength_col: str | None = None,
    value_col: str | None = None,
    wavelength_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Read a two-column spectrum CSV.

    If column names are omitted, the first column is treated as wavelength and
    the second as the spectral value. Use ``wavelength_scale=1000`` to convert
    micrometers to nanometers.
    """

    wavelengths: list[float] = []
    values: list[float] = []
    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or len(reader.fieldnames) < 2:
            raise ValueError(f"CSV file {path!s} needs at least two columns.")
        wavelength_name = wavelength_col or reader.fieldnames[0]
        value_name = value_col or reader.fieldnames[1]
        if wavelength_name not in reader.fieldnames:
            raise ValueError(f"CSV file {path!s} is missing column {wavelength_name!r}.")
        if value_name not in reader.fieldnames:
            raise ValueError(f"CSV file {path!s} is missing column {value_name!r}.")

        for row in reader:
            wavelengths.append(float(row[wavelength_name]) * wavelength_scale)
            values.append(float(row[value_name]))

    if not wavelengths:
        raise ValueError(f"CSV file {path!s} contains no spectral rows.")

    wavelength_array = jnp.asarray(wavelengths)
    value_array = jnp.asarray(values)
    order = jnp.argsort(wavelength_array)
    return wavelength_array[order], value_array[order]


def export_simulation(
    stack: Stack,
    result: SimulationResult,
    output_dir: str | Path,
    simulation_name: str = "simulation",
    reference_wavelength_nm: float | None = None,
    timestamp: str | None = None,
    show: bool = False,
) -> ExportPaths:
    """Write connected structure, spectra, and plot files for a simulation.

    Files share the same sanitized prefix:

    - ``<prefix>_<YYYYMMDD_HHMMSS>_structure.txt``
    - ``<prefix>_<YYYYMMDD_HHMMSS>_spectra.txt``
    - ``<prefix>_<YYYYMMDD_HHMMSS>_reflectivity.png``
    - ``<prefix>_<YYYYMMDD_HHMMSS>_transmission.png``
    - ``<prefix>_<YYYYMMDD_HHMMSS>_absorption.png``
    - ``<prefix>_<YYYYMMDD_HHMMSS>_RTA.png``

    For one polarization, the spectra text has exactly four columns:
    wavelength, reflectivity, transmission, absorption. For ``"both"``, the
    same file includes grouped columns for each polarization.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(simulation_name)
    timestamp_text = _safe_prefix(timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    file_prefix = f"{prefix}_{timestamp_text}"

    paths = ExportPaths(
        structure_txt=output_path / f"{file_prefix}_structure.txt",
        spectra_txt=output_path / f"{file_prefix}_spectra.txt",
        reflectivity_plot=output_path / f"{file_prefix}_reflectivity.png",
        transmission_plot=output_path / f"{file_prefix}_transmission.png",
        absorption_plot=output_path / f"{file_prefix}_absorption.png",
        combined_plot=output_path / f"{file_prefix}_RTA.png",
    )

    wavelengths = np.asarray(result.wavelength_nm)
    if reference_wavelength_nm is None:
        reference_wavelength_nm = float(wavelengths[len(wavelengths) // 2])

    paths.structure_txt.write_text(
        _structure_text(
            stack=stack,
            simulation_name=prefix,
            file_prefix=file_prefix,
            timestamp=timestamp_text,
            result=result,
            reference_wavelength_nm=reference_wavelength_nm,
        )
    )
    _write_spectra(
        paths.spectra_txt,
        simulation_name=prefix,
        file_prefix=file_prefix,
        timestamp=timestamp_text,
        result=result,
    )
    _plot_combined_rta(
        paths.combined_plot,
        result=result,
        title=f"{prefix} R/T/A",
        show=show,
    )
    _plot_quantity(
        paths.reflectivity_plot,
        result=result,
        values=result.R,
        ylabel="Reflectivity",
        title=f"{prefix} reflectivity",
    )
    _plot_quantity(
        paths.transmission_plot,
        result=result,
        values=result.T,
        ylabel="Transmission",
        title=f"{prefix} transmission",
    )
    _plot_quantity(
        paths.absorption_plot,
        result=result,
        values=result.A,
        ylabel="Absorption",
        title=f"{prefix} absorption",
    )
    return paths


def _write_spectra(
    path: Path,
    simulation_name: str,
    file_prefix: str,
    timestamp: str,
    result: SimulationResult,
) -> None:
    wavelengths = np.asarray(result.wavelength_nm)
    polarizations = result.polarizations
    R = _as_polarization_rows(result.R)
    T = _as_polarization_rows(result.T)
    A = _as_polarization_rows(result.A)

    columns = ["wavelength_nm"]
    arrays = [wavelengths]
    for index, polarization in enumerate(polarizations):
        suffix = "" if len(polarizations) == 1 else f"_{polarization}"
        columns.extend([f"R{suffix}", f"T{suffix}", f"A{suffix}"])
        arrays.extend([R[index], T[index], A[index]])

    data = np.column_stack(arrays)
    header = "\n".join(
        (
            f"simulation_name: {simulation_name}",
            f"timestamp: {timestamp}",
            f"file_prefix: {file_prefix}",
            "columns: " + " ".join(columns),
        )
    )
    np.savetxt(path, data, header=header, comments="# ", fmt="%.12g")


def _structure_text(
    stack: Stack,
    simulation_name: str,
    file_prefix: str,
    timestamp: str,
    result: SimulationResult,
    reference_wavelength_nm: float,
) -> str:
    lines = [
        "# multilayer_tmm structure",
        f"simulation_name: {simulation_name}",
        f"timestamp: {timestamp}",
        f"file_prefix: {file_prefix}",
        f"reference_wavelength_nm: {_format_float(reference_wavelength_nm)}",
        f"polarizations: {', '.join(result.polarizations)}",
        "role,layer_index,material,thickness_nm,refractive_index",
    ]
    lines.append(
        _structure_row(
            role="incident",
            layer_index="",
            material_name=stack.incident.name or "incident",
            thickness="semi-infinite",
            refractive_index=_material_index_text(stack.incident, reference_wavelength_nm),
        )
    )
    for index, layer in enumerate(stack.layers, start=1):
        lines.append(
            _structure_row(
                role="finite",
                layer_index=str(index),
                material_name=layer.material.name or f"layer_{index}",
                thickness=_format_float(float(jnp.asarray(layer.thickness_nm))),
                refractive_index=_material_index_text(
                    layer.material,
                    reference_wavelength_nm,
                ),
            )
        )
    lines.append(
        _structure_row(
            role="substrate",
            layer_index="",
            material_name=stack.substrate.name or "substrate",
            thickness="semi-infinite",
            refractive_index=_material_index_text(stack.substrate, reference_wavelength_nm),
        )
    )
    return "\n".join(lines) + "\n"


def _plot_quantity(
    path: Path,
    result: SimulationResult,
    values: jnp.ndarray,
    ylabel: str,
    title: str,
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    wavelengths = np.asarray(result.wavelength_nm)
    value_rows = _as_polarization_rows(values)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for row, polarization in zip(value_rows, result.polarizations, strict=True):
        label = polarization if len(result.polarizations) > 1 else None
        ax.plot(wavelengths, row, label=label, linewidth=2.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(result.polarizations) > 1:
        ax.legend(title="Polarization")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_combined_rta(
    path: Path,
    result: SimulationResult,
    title: str,
    show: bool,
) -> None:
    in_ipython = _in_ipython_kernel()
    has_explicit_display = any(
        os.environ.get(name)
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "MPLBACKEND")
    )
    _configure_matplotlib(interactive=show and (in_ipython or has_explicit_display))
    import matplotlib
    import matplotlib.pyplot as plt

    show_method = _show_method(
        show,
        matplotlib.get_backend(),
        in_ipython=in_ipython,
    )
    wavelengths = np.asarray(result.wavelength_nm)
    quantities = (
        ("Reflectivity", result.R),
        ("Transmission", result.T),
        ("Absorption", result.A),
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle(title)
    for axis, (ylabel, values) in zip(axes, quantities, strict=True):
        value_rows = _as_polarization_rows(values)
        for row, polarization in zip(value_rows, result.polarizations, strict=True):
            label = polarization if len(result.polarizations) > 1 else None
            axis.plot(wavelengths, row, label=label, linewidth=2.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if len(result.polarizations) > 1:
            axis.legend(title="Polarization")
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.savefig(path, dpi=180)
    if show_method == "ipython":
        _display_in_ipython(path)
        plt.close(fig)
    elif show_method == "pyplot":
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)


def _configure_matplotlib(interactive: bool = False) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "multilayer_tmm_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    if not interactive and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg", force=True)


def _show_method(show: bool, backend: str, in_ipython: bool | None = None) -> str:
    """Return the display method for a Matplotlib backend."""

    if not show:
        return "none"
    if in_ipython is None:
        in_ipython = _in_ipython_kernel()
    if in_ipython:
        return "ipython"
    normalized = backend.lower()
    if "inline" in normalized or "ipympl" in normalized or "widget" in normalized:
        return "ipython"
    noninteractive_backends = {"agg", "pdf", "ps", "svg", "template"}
    if normalized in noninteractive_backends or normalized.endswith("backend_agg"):
        return "none"
    if normalized == "macosx" and not os.environ.get("MPLBACKEND"):
        return "none"
    return "pyplot"


def _display_in_ipython(path: Path) -> None:
    """Display the saved PNG in IPython/Jupyter/VS Code interactive."""

    try:
        from IPython.display import Image, display
    except ImportError:
        return
    display(Image(filename=str(path)))


def _in_ipython_kernel() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    return shell is not None


def _as_polarization_rows(values: jnp.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim == 2:
        return array
    raise ValueError("Expected a one- or two-dimensional spectrum array.")


def _structure_row(
    role: str,
    layer_index: str,
    material_name: str,
    thickness: str,
    refractive_index: str,
) -> str:
    return ",".join(
        (
            role,
            layer_index,
            material_name.replace(",", " "),
            thickness,
            refractive_index,
        )
    )


def _material_index_text(material: object, wavelength_nm: float) -> str:
    values = material(jnp.asarray([wavelength_nm]))
    value = complex(np.asarray(values)[0])
    return _format_complex(value)


def _format_complex(value: complex) -> str:
    real = _format_float(value.real)
    imag = _format_float(abs(value.imag))
    sign = "+" if value.imag >= 0 else "-"
    return f"{real}{sign}{imag}j"


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _safe_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return prefix.strip("._-") or "simulation"
