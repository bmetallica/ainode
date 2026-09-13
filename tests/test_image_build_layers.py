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
import sys
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
        assert _index("pip install -r /src/requirements.txt") < _index(
            "COPY ainode /src/ainode")

    def test_the_source_install_resolves_nothing(self):
        # --no-deps, or the expensive layer happens twice.
        assert "pip install --no-deps --no-cache-dir /src" in DOCKERFILE

    def test_the_dependency_layer_never_sees_our_source(self):
        before = DOCKERFILE[:_index("pip install -r /src/requirements.txt")]
        assert "COPY ainode /src/ainode" not in before
        assert "COPY pyproject.toml scripts/_deps_from_pyproject.py /src/" in before

    def test_no_stub_package_is_built(self):
        # A package with our name and version but different contents can be
        # handed back by pip's wheel cache OR Docker's layer cache. One was,
        # and the image shipped an empty ainode module.
        assert "printf '' > /src/ainode/__init__.py" not in DOCKERFILE
        assert "mkdir -p /src/ainode" not in DOCKERFILE

    def test_the_pip_cache_is_mounted_for_the_expensive_layer(self):
        # So a dependency change reuses what is already downloaded instead of
        # fetching 1.1 GB again. Only there: the source install must not see
        # the cache.
        assert DOCKERFILE.count("--mount=type=cache,target=/root/.cache/pip") == 1
        source_install = re.search(r"RUN pip install --no-deps[^\n]*", DOCKERFILE)
        assert source_install and "--no-cache-dir" in source_install.group(0)

    def test_the_build_verifies_what_it_shipped(self):
        assert "_verify_image.py" in DOCKERFILE
        assert _index("COPY ainode /src/ainode") < _index("_verify_image.py")


class TestTheRequirementsExtractor:
    """It replaces building a stub package, so it has to be exactly right."""

    def _deps(self, extras=""):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            from _deps_from_pyproject import dependencies
            return dependencies(REPO / "pyproject.toml", extras)
        finally:
            sys.path.pop(0)

    def test_it_lists_the_runtime_dependencies(self):
        deps = self._deps()
        assert any(d.startswith("aiohttp") for d in deps)
        assert any(d.startswith("paho-mqtt") for d in deps)

    def test_an_extra_adds_its_group(self):
        assert any(d.startswith("sentence-transformers")
                   for d in self._deps("[embeddings]"))

    def test_no_extra_leaves_torch_out(self):
        # The lean build's entire point.
        assert not any("sentence-transformers" in d for d in self._deps(""))

    @pytest.mark.parametrize("spelling", ["[embeddings]", "embeddings", " embeddings "])
    def test_the_extras_argument_is_forgiving(self, spelling):
        assert any("sentence-transformers" in d for d in self._deps(spelling))

    def test_several_extras(self):
        deps = self._deps("[embeddings,training]")
        assert any("sentence-transformers" in d for d in deps)
        assert any(d.startswith("torch") for d in deps)

    def test_an_unknown_extra_fails_the_build(self):
        # Silently installing nothing would surface as a missing import at
        # runtime, on the node.
        with pytest.raises(SystemExit) as excinfo:
            self._deps("[embeddingz]")
        assert "embeddingz" in str(excinfo.value)

    def test_it_matches_what_pip_would_resolve(self):
        # The list is the contract with pyproject; drift means the image
        # installs something the package does not declare.
        import tomllib

        project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        expected = list(project["dependencies"]) + list(
            project["optional-dependencies"]["embeddings"])
        assert self._deps("[embeddings]") == expected


class TestTheLeanBuildIsStillOffered:
    def test_the_extra_is_a_build_arg(self):
        assert 'ARG AINODE_EXTRAS="[embeddings]"' in DOCKERFILE

    def test_and_it_is_documented_as_the_lean_option(self):
        assert "AINODE_EXTRAS" in (REPO / "scripts" / "Dockerfile.ainode").read_text()
        assert "lean orchestrator" in DOCKERFILE
