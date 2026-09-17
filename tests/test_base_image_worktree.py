"""The second --base build of a clone must work as well as the first.

Reported from the cluster:

    ==> Preparing eugr worktree at /opt/ainode/scripts/_eugr (commit 346dc04)
    error: Ihre lokalen Änderungen in den folgenden Dateien würden beim
    Auschecken überschrieben werden:  Dockerfile
    Abbruch

The script patches eugr's Dockerfile itself, a few lines below the checkout —
the NCCL pin. So the first build on a fresh clone succeeds and every build
after it fails on the script's own leftovers, which reads as a broken machine
rather than a script that cannot run twice.

Discarding is correct here and only here: _eugr is a scratch checkout this
script owns, re-created from scratch when absent, and every local change in
it was put there by the patch step that runs again immediately afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts" /
          "build-base-image.sh").read_text()


class TestTheWorktreeIsReclaimed:
    def test_the_reused_checkout_is_forced(self):
        reuse = SCRIPT[SCRIPT.index('if [ -d "$WORKTREE/.git" ]'):]
        reuse = reuse[:reuse.index("else")]
        assert re.search(r'checkout -q -f "\$EUGR_COMMIT"', reuse)

    def test_the_fresh_clone_path_needs_no_force(self):
        """Nothing has patched it yet, and forcing there would hide a real
        failure to check out the pinned commit."""
        fresh = SCRIPT[SCRIPT.index("rm -rf \"$WORKTREE\""):]
        fresh = fresh[:fresh.index("# -- Patch eugr's Dockerfile")]
        assert 'checkout -q "$EUGR_COMMIT"' in fresh
        assert "checkout -q -f" not in fresh

    def test_the_script_still_patches_the_dockerfile(self):
        """If it ever stops, the force above is no longer justified — it would
        then be discarding someone else's edit rather than its own."""
        assert "# -- Patch eugr's Dockerfile" in SCRIPT

    def test_why_discarding_is_safe_is_written_down(self):
        assert "scratch checkout this script owns" in SCRIPT


def _patch_program() -> str:
    """The python block the uv-override patch runs, extracted at test time."""
    match = re.search(
        r"OVERRIDES=\"\$AINODE_UV_OVERRIDES\" python3 - \"\$DOCKERFILE\" <<'PATCH'\n"
        r"(.*?)\nPATCH\n", SCRIPT, re.S)
    assert match, "the uv-override patch is no longer where the test expects it"
    return match.group(1)


def _run_patch(dockerfile_text: str):
    import subprocess
    import sys
    import tempfile

    target = Path(tempfile.mkdtemp()) / "Dockerfile"
    target.write_text(dockerfile_text)
    result = subprocess.run(
        [sys.executable, "-c", _patch_program(), str(target)],
        env={"OVERRIDES": "quack-kernels>=0.6.5", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30)
    return result, target.read_text()


class TestTheUvOverridePatch:
    """It failed the build the first time the eugr pin moved forward.

        ==> Patching eugr Dockerfile: uv overrides (quack-kernels>=0.6.5)
        !! expected wheel-install line not found; re-audit this patch

    Upstream had taken the problem over: eugr's own Dockerfile now writes
    nvidia-cutlass-dsl into /tmp/wheel-override.txt and installs the wheels
    with --override, which is exactly what this patch existed to add. Stopping
    a 25-minute build over a problem that is already solved is the wrong
    answer; so is patching on top of it and pinning the constraint twice.

    A patch against someone else's file has three outcomes, not two: applied,
    unnecessary, and no-longer-applicable. Only the third is an error.
    """

    UPSTREAM = (
        'RUN echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" '
        '>> /tmp/wheel-override.txt && \\\n'
        "    uv pip install /workspace/vllm-wheels/*.whl "
        "--override /tmp/wheel-override.txt\n"
    )
    OLD = "        uv pip install /workspace/wheels/*.whl; \\\n"

    def test_an_upstream_override_is_left_alone(self):
        result, after = _run_patch(self.UPSTREAM)
        assert result.returncode == 0, result.stderr
        assert "skipped" in result.stdout
        assert after == self.UPSTREAM

    def test_the_old_shape_is_still_patched(self):
        """A build pinned back to an older eugr commit must still work."""
        result, after = _run_patch(self.OLD)
        assert result.returncode == 0, result.stderr
        assert "ainode-override.txt" in after
        assert "quack-kernels>=0.6.5" in after

    def test_re_running_does_not_patch_twice(self):
        _, once = _run_patch(self.OLD)
        result, twice = _run_patch(once)
        assert result.returncode == 0
        assert twice == once

    def test_an_unrecognisable_dockerfile_still_stops_the_build(self):
        """The case the error was written for: the file changed shape and
        nobody has looked. Silently building an unpatched image would be
        worse than failing."""
        result, _ = _run_patch("FROM scratch\nRUN true\n")
        assert result.returncode != 0
        assert "re-audit" in result.stderr
