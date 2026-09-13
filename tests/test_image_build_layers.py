"""A code change must not re-download a gigabyte of wheels.

The dependency install pulls torch (454 MB) and cuDNN (651 MB) with the
embeddings extra. That layer used to sit BELOW `COPY ainode /src/ainode`, so
editing one Python file invalidated it and the build downloaded everything
again — an hour on a domestic line, for a one-line change. Observed:

    => [ 7/11] RUN pip install --no-cache-dir "/src[embeddings]"   3455.3s
    => => # Downloading torch-2.14.0-...-aarch64.whl (454.0 MB)
    => => # Downloading nvidia_cudnn_cu13-...-aarch64.whl (651.0 MB)
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = (REPO / "scripts" / "Dockerfile.ainode").read_text()


def _index(needle: str) -> int:
    position = DOCKERFILE.find(needle)
    assert position != -1, f"{needle!r} is not in the Dockerfile"
    return position


class TestLayerOrder:
    def test_dependencies_are_installed_before_the_source_is_copied(self):
        assert _index('pip install "/src${AINODE_EXTRAS}"') < _index(
            "COPY ainode /src/ainode")

    def test_the_source_install_resolves_nothing(self):
        # --no-deps, or the expensive layer happens twice.
        assert "pip install --no-deps /src" in DOCKERFILE

    def test_only_pyproject_is_copied_for_the_dependency_layer(self):
        before = DOCKERFILE[:_index('pip install "/src${AINODE_EXTRAS}"')]
        assert "COPY pyproject.toml README.md /src/" in before
        assert "COPY ainode /src/ainode" not in before

    def test_the_pip_cache_is_mounted(self):
        # So a dependency change reuses what is already downloaded instead of
        # fetching 1.1 GB again.
        assert DOCKERFILE.count("--mount=type=cache,target=/root/.cache/pip") >= 2

    def test_no_no_cache_dir_where_the_cache_is_mounted(self):
        # The two contradict each other: --no-cache-dir tells pip not to use
        # the very directory the mount provides.
        for block in re.findall(r"RUN --mount=type=cache,target=/root/\.cache/pip.*?(?=\nRUN |\nCOPY |\nARG |\Z)",
                                DOCKERFILE, re.S):
            assert "--no-cache-dir" not in block, block[:200]


class TestTheStubIsEnough:
    """pip must be able to resolve dependencies from pyproject alone."""

    def test_a_wheel_builds_from_pyproject_and_an_empty_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            (src / "ainode").mkdir(parents=True)
            (src / "ainode" / "__init__.py").write_text("")
            for name in ("pyproject.toml", "README.md"):
                (src / name).write_text((REPO / name).read_text())
            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps",
                 "-w", str(Path(tmp) / "out"), str(src)],
                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    def test_the_version_is_not_read_from_the_package(self):
        # A dynamic version read from ainode/__init__.py would make the stub
        # build a wheel with the wrong version, or fail.
        pyproject = (REPO / "pyproject.toml").read_text()
        assert re.search(r'^version = "\d', pyproject, re.M)
        assert "dynamic" not in pyproject.split("[tool.")[0]


class TestTheLeanBuildIsStillOffered:
    def test_the_extra_is_a_build_arg(self):
        assert 'ARG AINODE_EXTRAS="[embeddings]"' in DOCKERFILE

    def test_and_it_is_documented_as_the_lean_option(self):
        assert "AINODE_EXTRAS" in (REPO / "scripts" / "Dockerfile.ainode").read_text()
        assert "lean orchestrator" in DOCKERFILE
