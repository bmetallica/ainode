"""The updater rewrites a host's Docker configuration. That deserves a test.

Everything else in scripts/update-cluster.sh is orchestration that shows its
work and can be run with --check. The one part that edits a file the operator
did not write, on machines the operator cares about, is the daemon.json merge
— and it has to be idempotent, non-destructive, and honest about failing.

The merge program is extracted from the shell script at test time rather than
copied here: a copy would drift, and the thing worth testing is what actually
ships.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SETUP = REPO / "scripts" / "setup-registry-cache.sh"
UPDATE = REPO / "scripts" / "update-cluster.sh"

MIRROR = "http://192.168.1.2:5000"
INSECURE = "192.168.1.2:5000"
LOCAL = "192.168.1.2:5001"

#: The merge program's contract with the shell around it.
UNCHANGED, CHANGED, REFUSED = 0, 10, 2


@pytest.fixture(scope="module")
def merge_program(tmp_path_factory) -> Path:
    text = SETUP.read_text()
    match = re.search(r"read -r -d '' MERGE_PY <<'PY' \|\| true\n(.*?)\nPY\n",
                      text, re.S)
    assert match, "the merge program is no longer where the test expects it"
    path = tmp_path_factory.mktemp("merge") / "merge.py"
    path.write_text(match.group(1))
    return path


def _merge(merge_program: Path, daemon_json: Path, check_only: bool = False):
    env = {"DAEMON_JSON": str(daemon_json), "PATH": "/usr/bin:/bin"}
    if check_only:
        env["CHECK_ONLY"] = "1"
    return subprocess.run(
        [sys.executable, str(merge_program), MIRROR, INSECURE, LOCAL],
        capture_output=True, text=True, env=env,
    )


class TestDaemonJsonMerge:
    def test_a_missing_file_is_created(self, merge_program, tmp_path):
        target = tmp_path / "daemon.json"
        assert _merge(merge_program, target).returncode == CHANGED
        config = json.loads(target.read_text())
        assert config["registry-mirrors"] == [MIRROR]
        assert INSECURE in config["insecure-registries"]
        assert LOCAL in config["insecure-registries"]

    def test_an_empty_file_is_filled(self, merge_program, tmp_path):
        target = tmp_path / "daemon.json"
        target.write_text("   \n")
        assert _merge(merge_program, target).returncode == CHANGED
        assert json.loads(target.read_text())["registry-mirrors"] == [MIRROR]

    def test_existing_settings_are_kept(self, merge_program, tmp_path):
        # People put things in daemon.json. Losing a log driver or a data-root
        # because a monitoring feature wanted a mirror would be unforgivable.
        target = tmp_path / "daemon.json"
        target.write_text(json.dumps({
            "log-driver": "json-file",
            "data-root": "/mnt/docker",
            "default-runtime": "nvidia",
        }))
        assert _merge(merge_program, target).returncode == CHANGED
        config = json.loads(target.read_text())
        assert config["log-driver"] == "json-file"
        assert config["data-root"] == "/mnt/docker"
        assert config["default-runtime"] == "nvidia"
        assert config["registry-mirrors"] == [MIRROR]

    def test_an_existing_mirror_is_not_dropped(self, merge_program, tmp_path):
        target = tmp_path / "daemon.json"
        target.write_text(json.dumps({"registry-mirrors": ["http://other:5000"]}))
        _merge(merge_program, target)
        mirrors = json.loads(target.read_text())["registry-mirrors"]
        assert mirrors == [MIRROR, "http://other:5000"]

    def test_running_twice_changes_nothing_the_second_time(self, merge_program, tmp_path):
        # The whole point: this is the update command, run whenever.
        target = tmp_path / "daemon.json"
        assert _merge(merge_program, target).returncode == CHANGED
        first = target.read_text()
        assert _merge(merge_program, target).returncode == UNCHANGED
        assert target.read_text() == first

    def test_unchanged_means_docker_is_not_restarted(self, merge_program, tmp_path):
        # Exit 0 is what the shell reads as "leave the daemon alone" — and a
        # needless daemon restart bounces every container on the node.
        target = tmp_path / "daemon.json"
        _merge(merge_program, target)
        assert _merge(merge_program, target).returncode == UNCHANGED

    def test_a_backup_is_written_before_an_edit(self, merge_program, tmp_path):
        target = tmp_path / "daemon.json"
        target.write_text(json.dumps({"log-driver": "journald"}))
        _merge(merge_program, target)
        backups = list(tmp_path.glob("daemon.json.ainode-*"))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text())["log-driver"] == "journald"

    def test_invalid_json_is_refused_not_overwritten(self, merge_program, tmp_path):
        # A hand-edited file with a trailing comma is a file someone cares
        # about. Replacing it with our two keys would destroy their config.
        target = tmp_path / "daemon.json"
        target.write_text('{"log-driver": "json-file",}')
        result = _merge(merge_program, target)
        assert result.returncode == REFUSED
        assert target.read_text() == '{"log-driver": "json-file",}'
        assert "not valid JSON" in result.stderr

    def test_check_mode_reports_without_writing(self, merge_program, tmp_path):
        target = tmp_path / "daemon.json"
        result = _merge(merge_program, target, check_only=True)
        assert result.returncode == CHANGED
        assert not target.exists()
        assert MIRROR in result.stdout


class TestScriptsAreSane:
    @pytest.mark.parametrize("script", [SETUP, UPDATE])
    def test_shell_syntax(self, script):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0

    @pytest.mark.parametrize("script", [SETUP, UPDATE])
    def test_executable_and_strict(self, script):
        assert script.stat().st_mode & 0o111, f"{script.name} is not executable"
        assert "set -euo pipefail" in script.read_text()

    def test_check_mode_touches_nothing(self):
        # --check has to be safe to run on a live cluster, so it must not
        # reach systemctl, docker build or ssh other than to look.
        result = subprocess.run(
            ["bash", str(UPDATE), "--check", "--skip-pull", "--skip-build"],
            capture_output=True, text=True, cwd=REPO, timeout=120)
        assert result.returncode == 0, result.stderr
        assert "nothing was changed" in result.stdout
        assert "would run: sudo systemctl restart ainode" in result.stdout

    def test_help_explains_repeat_runs(self):
        result = subprocess.run(["bash", str(UPDATE), "--help"],
                                capture_output=True, text=True, timeout=30)
        assert "again" in result.stdout.lower()

    def test_the_updater_can_reach_the_registry_setup(self):
        assert "setup-registry-cache.sh" in UPDATE.read_text()

    def test_peers_are_restarted_before_the_head(self):
        # The head builds its picture from what it discovers, so it comes up
        # last and sees a complete cluster.
        text = UPDATE.read_text()
        peer = text.index('say "${node}: restarting ainode"')
        head = text.index('say "this node: restarting ainode"')
        assert peer < head

    def test_preflight_runs_before_the_build(self):
        # A run that builds and then finds node 2 unreachable leaves the
        # cluster on two versions — worse than where it started.
        text = UPDATE.read_text()
        assert text.index("cannot ssh to") < text.index("build-ainode-image.sh")
