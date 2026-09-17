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
