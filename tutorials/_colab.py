"""Set a fresh Google Colab runtime up for the tutorials (a no-op anywhere else).

The first cell of every notebook clones this repository on Colab and calls :func:`setup`, which installs the
rumoca wheel the FastModel models need (from the GitHub release of this repository, or built from source when the
release has no wheel for the runtime's Python: a Rust toolchain is installed first, which takes about ten minutes)
and mpctrano from the clone with its notebook extra, then moves into the clone for the tutorial files and the
cached runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "https://github.com/andoludo/mpctrano"
RUMOCA_TAG = "rumoca-0.10.3-c74cddf"
RUMOCA_SOURCE = (
    "rumoca @ git+https://github.com/CogniPilot/rumoca@c74cddff6d7669fc2c3603d43529ab62e0964072"
    "#subdirectory=crates/rumoca-bind-python"
)
CLONE = Path("/content/mpctrano")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _pip(*arguments: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *arguments], check=False).returncode


def _rumoca_wheel_url() -> str | None:
    """The wheel of the release matching this Python and platform, if the release lists one."""
    url = f"https://api.github.com/repos/andoludo/mpctrano/releases/tags/{RUMOCA_TAG}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:  # the GitHub API over https
            assets = json.load(response).get("assets", [])
    except Exception:  # no release, no network: build from source
        return None
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for asset in assets:
        name = str(asset["name"])
        if name.endswith(".whl") and "manylinux" in name and "x86_64" in name and (tag in name or "abi3" in name):
            return str(asset["browser_download_url"])
    return None


def _rumoca_exports_fast_models() -> bool:
    try:
        from mpctrano.model import rumoca_exports_fast_models

        return bool(rumoca_exports_fast_models())
    except ImportError:
        return False


def setup() -> None:
    """Install the toolchain and the tutorials on Colab; do nothing elsewhere."""
    if not in_colab():
        return
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPOSITORY, str(CLONE)], check=True)
    if not _rumoca_exports_fast_models():
        wheel = _rumoca_wheel_url()
        if wheel is None or _pip(wheel) != 0:
            print("No rumoca wheel for this runtime: installing Rust and building rumoca (about ten minutes)...")
            subprocess.run("curl -sSf https://sh.rustup.rs | sh -s -- -y", shell=True, check=True)
            os.environ["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{os.environ['PATH']}"
            _pip(RUMOCA_SOURCE)
        _pip(f"{CLONE}[notebook]")
    os.chdir(CLONE)
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    print(f"ready: {CLONE}")
