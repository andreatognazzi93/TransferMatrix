"""Matplotlib backend setup for VS Code/Jupyter interactive runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def use_inline_backend_if_available() -> None:
    """Use the inline backend when a script is run inside IPython."""

    cache_dir = Path(tempfile.gettempdir()) / "multilayer_tmm_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    try:
        get_ipython  # type: ignore[name-defined]
    except NameError:
        return
    import matplotlib

    matplotlib.use("module://matplotlib_inline.backend_inline", force=True)
