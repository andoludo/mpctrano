"""The tutorial scripts stay runnable (in quick mode)."""

import os
import runpy
import sys
from pathlib import Path

import pytest

TUTORIALS = sorted((Path(__file__).parent.parent / "tutorials").glob("0*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", TUTORIALS, ids=[script.stem for script in TUTORIALS])
def test_tutorial_runs(script: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MPCTRANO_QUICK", "1")
    monkeypatch.syspath_prepend(str(script.parent.parent))
    import tutorials._common as common

    monkeypatch.setattr(common, "OUTPUT", tmp_path)
    monkeypatch.setattr(common, "QUICK", True)
    sys.modules.pop(f"tutorials.{script.stem}", None)
    runpy.run_path(str(script), run_name="__main__")
    assert os.environ["MPCTRANO_QUICK"] == "1"
