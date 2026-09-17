"""--base rebuilds the engine image here; the peers have to get it too.

Reported straight after a --base run:

    Error: Cluster launch aborted because image 'vllm-node' is not in sync.
    Sync it with: ./build-and-copy.sh --no-build -t vllm-node --copy-to <hosts>

eugr's launcher compares image ids across the cluster before starting
anything, and it is right to: ranks on different builds fail later and far
less clearly. But update-cluster.sh distributed only ainode:dev, so --base
left the cluster unable to launch anything distributed until someone noticed.

AINode's launch-time image distribution does not cover this one either. It
fires only for a model whose recipe PINS an engine image, on the reasoning
that the launcher default is built locally on every node — true when every
node ran the build, false the moment it moved to the head alone.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts" /
          "update-cluster.sh").read_text()


def _section() -> str:
    start = SCRIPT.index("# --- 5b. the engine image")
    return SCRIPT[start:SCRIPT.index("# --- 6. restart")]


class TestItIsDistributed:
    def test_there_is_a_step_for_it(self):
        assert "Distributing the engine image" in SCRIPT

    def test_only_after_base_rebuilt_it(self):
        """A normal update leaves the engine image alone, so pushing 20 GB on
        every run would be pure waste."""
        assert 'if [[ $DO_BASE -eq 1 && ${#NODE_LIST[@]} -gt 0 ]]; then' in _section()

    def test_it_goes_through_the_registry_when_there_is_one(self):
        section = _section()
        assert 'if [[ -n "$REGISTRY" ]]' in section
        assert "docker push" in section

    def test_and_over_ssh_when_there_is_not(self):
        assert "docker save" in _section() and "docker load" in _section()

    def test_a_missing_image_warns_instead_of_failing_the_run(self):
        """The models already on the nodes keep working; a hard exit here
        would take the rest of the update down with it."""
        section = _section()
        assert "is not here; skipping" in section

    def test_check_mode_does_not_move_anything(self):
        assert 'if [[ $CHECK -eq 1 ]]; then' in _section()


class TestItVerifiesWhatTheLauncherWillCheck:
    def test_the_ids_are_compared_not_the_tags(self):
        """The launcher compares ids. A peer holding a different build under
        the same tag passes any tag check and fails the launch."""
        section = _section()
        assert "{{.Id}}" in section
        assert re.search(r'peer_id.*==.*head_id|"\$peer_id" == "\$head_id"', section)

    def test_a_mismatch_is_reported_here_rather_than_at_the_next_launch(self):
        assert "a distributed launch will abort" in _section()

    def test_the_header_mentions_it(self):
        """Someone reading --help should learn that --base moves ~20 GB."""
        assert "distributes the ENGINE image too" in SCRIPT
