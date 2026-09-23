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
