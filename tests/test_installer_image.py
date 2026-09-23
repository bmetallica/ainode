"""How the installer decides which image to install.

Re-running `scripts/install.sh` is the documented way to change what ends up
in ExecStart — a new mount, a new flag. On a fork whose GHCR package is
private that re-run used to die at `docker pull`, on a node that was already
installed and running perfectly well. These tests pin the way out: the image
recorded by the last install is used when the registry cannot be reached.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

INSTALL = Path(__file__).resolve().parent.parent / "scripts" / "install.sh"
SOURCE = INSTALL.read_text()


def _function(name: str) -> str:
    """The shell source of one function, so it can be run on its own."""
    start = SOURCE.index(f"{name}() {{")
    end = SOURCE.index("\n}\n", start) + 3
    return SOURCE[start:end]


def _run(script: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, env={"PATH": "/usr/bin:/bin", **env})


class TestInstalledImage:
    """`installed_image` reads the EnvironmentFile the last install wrote."""

    def _harness(self, tmp_path, *, known: str = "") -> str:
        # A docker stub: `docker image inspect X` succeeds only for `known`.
        stub = tmp_path / "bin"
        stub.mkdir(exist_ok=True)
        (stub / "docker").write_text(
            '#!/bin/sh\n[ "$1" = image ] || exit 1\n'
            f'[ "$3" = "{known}" ] && exit 0\nexit 1\n')
        (stub / "docker").chmod(0o755)
        return (f'export PATH="{stub}:$PATH"\nAINODE_HOME="{tmp_path}"\n'
                + _function("installed_image")
                + '\ninstalled_image\n')

    def test_it_echoes_the_recorded_image(self, tmp_path):
        (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/x/ainode:0.5.1\n")
        out = _run(self._harness(tmp_path, known="ghcr.io/x/ainode:0.5.1"))
        assert out.returncode == 0
        assert out.stdout.strip() == "ghcr.io/x/ainode:0.5.1"

    def test_no_file_means_no_answer(self, tmp_path):
        assert _run(self._harness(tmp_path, known="anything")).returncode != 0

    def test_a_recorded_image_that_is_gone_does_not_count(self, tmp_path):
        # Pruned, or the disk was rebuilt. Naming an image that is not there
        # would only move the failure to `docker run`, at service start.
        (tmp_path / "image.env").write_text("AINODE_IMAGE=ghcr.io/x/ainode:0.5.1\n")
        out = _run(self._harness(tmp_path, known="something-else"))
        assert out.returncode != 0
        assert out.stdout.strip() == ""

    def test_an_empty_file_does_not_count(self, tmp_path):
        (tmp_path / "image.env").write_text("\n")
        assert _run(self._harness(tmp_path, known="")).returncode != 0


class TestResolutionOrder:
    """Registry first, installed image second, :latest last."""

    BLOCK = SOURCE.split('if [ -z "$AINODE_IMAGE" ]; then')[1].split("\nfi\n")[0]

    def test_an_explicit_image_skips_all_of_it(self):
        assert 'if [ -z "$AINODE_IMAGE" ]; then' in SOURCE

    def test_the_registry_is_still_asked_first(self):
        # The fallback must not cost anyone their upgrade: a reachable
        # registry still decides, exactly as before.
        assert self.BLOCK.index("resolve_latest_tag") < self.BLOCK.index("installed_image")

    def test_the_installed_image_comes_before_latest(self):
        assert self.BLOCK.index("installed_image") < self.BLOCK.index(":latest")

    def test_it_says_which_image_it_kept(self):
        assert "keeping the image this node already runs" in self.BLOCK

    def test_it_names_how_to_upgrade_anyway(self):
        # Otherwise the node is pinned to this image with no visible way off.
        assert "ainode update" in self.BLOCK
        assert "AINODE_IMAGE=" in self.BLOCK


class TestMessagesRenderTheirLineBreaks:
    r"""The advice on a failed pull is a list of commands, one per line. With
    printf %s it arrived as a single line with literal \n in it — unreadable
    exactly when the reader most needs to follow it."""

    @pytest.mark.parametrize("fn", ["log", "warn", "die"])
    def test_the_printer_interprets_escapes(self, fn):
        body = _function(fn) if f"{fn}() {{\n" in SOURCE else \
            re.search(rf"^{fn}\(\) \{{.*$", SOURCE, re.M).group(0)
        assert "%b" in body, f"{fn} would print backslash-n literally"

    def test_a_multi_line_message_really_comes_out_multi_line(self):
        printer = re.search(r"^die\(\) \{.*$", SOURCE, re.M).group(0)
        script = printer + '\ndie "one\\n  two" 2>&1 || true\n'
        out = _run(script)
        assert "one" in out.stdout
        assert "  two" in out.stdout.splitlines()[-1]
        assert "\\n" not in out.stdout

    def test_the_pull_failure_points_at_the_recorded_image(self):
        failure = SOURCE.split("Could not pull")[1][:900]
        assert "image.env" in failure


class TestBuildingWithoutBuildKit:
    r"""The UI's update button builds the orchestrator image from inside the
    orchestrator container, and that build died:

        Step 16/28 : RUN --mount=type=cache,target=/root/.cache/pip ...
        the --mount option requires BuildKit
        [ainode] FAILED: scripts/update-cluster.sh exited with 1

    `--mount` is a BuildKit directive and the legacy builder stops on it
    rather than ignoring it. On a host this never comes up — docker-ce ships
    buildx and uses it by default — but the container's docker-ce-cli was
    installed with --no-install-recommends, and buildx is a *recommended*
    package.

    Both halves are needed. The image installs buildx so the next build has
    it; the script copes without it, because a fix that only works on images
    built after itself is no use to the node running the older one.
    """

    BUILD = (Path(__file__).resolve().parent.parent / "scripts" /
             "build-ainode-image.sh").read_text()
    DOCKERFILE = (Path(__file__).resolve().parent.parent / "scripts" /
                  "Dockerfile.ainode").read_text()

    def test_the_image_installs_buildx(self):
        assert "docker-buildx-plugin" in self.DOCKERFILE

    def test_the_script_checks_for_it(self):
        assert "docker buildx version" in self.BUILD

    def test_it_uses_buildkit_when_it_is_there(self):
        assert "DOCKER_BUILDKIT=1" in self.BUILD

    def test_it_strips_the_mounts_when_it_is_not(self):
        assert "s/--mount=type=" in self.BUILD

    def test_the_strip_leaves_a_valid_dockerfile(self):
        # The flag goes, the RUN stays: `RUN --mount=... \` must not become
        # a bare continuation with no command.
        stripped = re.sub(r"--mount=type=[^ ]+ *", "", self.DOCKERFILE)
        # The directive is gone from the instructions. The word may still
        # appear in a comment explaining why, which the builder ignores.
        assert "--mount=type=" not in stripped
        for line in stripped.splitlines():
            assert not line.strip().startswith("--")
        # Every RUN still starts a command or a continuation.
        assert "RUN \\\n" in stripped or "RUN " in stripped

    def test_the_fallback_says_what_it_costs(self):
        # Silently building differently is how two images that should be
        # identical stop being identical.
        assert "pip cache disabled" in self.BUILD

    def test_the_temporary_dockerfile_is_cleaned_up(self):
        assert "trap 'rm -f" in self.BUILD

    def test_only_this_dockerfile_needs_the_dance(self):
        # If another one grows a cache mount, this test says so before a
        # build on a buildx-less node does.
        scripts = Path(__file__).resolve().parent.parent / "scripts"
        with_mounts = [f.name for f in scripts.glob("Dockerfile*")
                       if "--mount=type=" in f.read_text()]
        assert with_mounts == ["Dockerfile.ainode"]


class TestTheLayerOrderDoesNotCostAGigabyte:
    """What a one-line commit is allowed to rebuild.

    Reported from the cluster, watching an update build:

        warum wird hier alles neu runtergeladen, der kram müsste doch
        gecacht sein oder nicht?

    It should have been. `ARG AINODE_GIT_SHA` changes on every commit and a
    changed ARG invalidates every layer after it — and it sat above the apt
    step and the pip step. So every commit re-installed gcc and re-downloaded
    torch (454 MB), cuDNN (651 MB) and the CUDA runtime. The fix is ordering,
    not caching: the volatile inputs go last.
    """

    DOCKERFILE = (Path(__file__).resolve().parent.parent / "scripts" /
                  "Dockerfile.ainode").read_text()

    def _at(self, needle: str) -> int:
        index = self.DOCKERFILE.find(needle)
        assert index > 0, f"{needle!r} is not in the Dockerfile"
        return index

    def test_the_commit_sha_comes_after_the_dependency_install(self):
        assert self._at("pip install -r /src/requirements.txt") < \
            self._at("ARG AINODE_GIT_SHA")

    def test_it_comes_after_the_compiler_too(self):
        assert self._at("gcc libc6-dev") < self._at("ARG AINODE_GIT_SHA")

    def test_it_still_reaches_the_image(self):
        # Later, but not gone: the update check compares this against the
        # fork's branch.
        assert "ENV AINODE_GIT_SHA=${AINODE_GIT_SHA}" in self.DOCKERFILE
        assert "org.opencontainers.image.revision" in self.DOCKERFILE

    def test_only_source_follows_it(self):
        after = self.DOCKERFILE[self._at("ARG AINODE_GIT_SHA"):]
        # No apt, no dependency resolution: those must be above.
        assert "apt-get install" not in after
        assert "pip install -r" not in after

    def test_pyproject_arrives_just_before_it_is_needed(self):
        # It changes on a version bump, so everything expensive above it
        # survives one.
        assert self._at("gcc libc6-dev") < \
            self._at("COPY pyproject.toml scripts/_deps_from_pyproject.py")

    def test_the_source_copy_is_last_of_the_three(self):
        assert self._at("COPY pyproject.toml scripts/_deps_from_pyproject.py") < \
            self._at("COPY ainode /src/ainode")
