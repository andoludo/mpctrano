"""Set a fresh Google Colab runtime up for the tutorials (a no-op anywhere else).

The first cell of every notebook clones this repository on Colab and calls :func:`setup`, which installs mpctrano from
the clone with its notebook extra (rumoca, CasADi and FastSmoothBuildings come as wheels from PyPI), then moves into the
clone for the tutorial files and the cached runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/andoludo/mpctrano"
RUMOCA = "rumoca>=0.10.3,<0.11"
CLONE = Path("/content/mpctrano")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _pip(*arguments: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *arguments], check=False).returncode


def _rumoca_exports_building_models() -> bool:
    try:
        from mpctrano.model import rumoca_exports_building_models

        return bool(rumoca_exports_building_models())
    except ImportError:
        return False


def setup() -> None:
    """Install the tutorials and their toolchain on Colab; do nothing elsewhere."""
    if not in_colab():
        return
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPOSITORY, str(CLONE)], check=True)
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    if not _rumoca_exports_building_models():
        _pip(f"{CLONE}[notebook]")
        if not _rumoca_exports_building_models():
            # a runtime that kept another build of rumoca 0.10.3 (the wheel of an earlier release of the tutorials)
            _pip("--force-reinstall", "--no-deps", RUMOCA)
    os.chdir(CLONE)
    print(f"ready: {CLONE}")
