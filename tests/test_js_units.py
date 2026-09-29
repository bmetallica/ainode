"""Run the dashboard's JavaScript unit tests (tests/js) as part of the suite,
so CI and a local `pytest` cover them without a second command."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_the_javascript_units_pass():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    done = subprocess.run([node, "--test", str(ROOT / "tests" / "js")],
                          capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]


def test_the_scripts_parse():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    for name in ("app.js", "lib.js", "topology.js"):
        path = ROOT / "ainode" / "web" / "static" / "js" / name
        done = subprocess.run([node, "--check", str(path)],
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, f"{name}: {done.stderr}"
