"""`ssh <peer>` from inside the container must use the operator's account.

eugr's launcher checks connectivity with `ssh <ip> true` — no user — so it
inherits the container's, which is root, while the keys belong to the install
user. The mapping lives in /root/.ssh/config, and it used to be written only
when config.json already listed peer_ips.

peer_ips is empty until a distributed launch has SUCCEEDED. So the mapping
appeared only after it was no longer needed, and the first distributed launch
on a fresh install died with:

    Error: Passwordless SSH to 192.168.1.3 failed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend

REPO = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO / "scripts" / "docker-entrypoint.sh"


def _entrypoint_ssh_program() -> str:
    """The python block the entrypoint runs, extracted at test time."""
    match = re.search(r"python3 - <<'PY'\n(.*?)\nPY\n", ENTRYPOINT.read_text(), re.S)
    assert match, "the ssh-config block is no longer where the test expects it"
    return match.group(1)


class TestTheEntrypointMapping:
    def _run(self, tmp_path, config: dict) -> str:
        home = tmp_path / "ainode"
        home.mkdir()
        (home / "config.json").write_text(json.dumps(config))
        ssh_dir = tmp_path / "ssh"
        ssh_dir.mkdir()
        program = _entrypoint_ssh_program().replace("/root/.ssh", str(ssh_dir))
        subprocess.run([sys.executable, "-c", program],
                       env={"AINODE_HOME": str(home), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=30)
        target = ssh_dir / "config"
        return target.read_text() if target.exists() else ""

    def test_it_writes_the_user_without_any_peers(self, tmp_path):
        # The whole bug: peer_ips is empty before the first successful launch.
        written = self._run(tmp_path, {"ssh_user": "admin", "peer_ips": []})
        assert "User admin" in written
        assert "Host *" in written

    def test_it_still_works_with_peers(self, tmp_path):
        written = self._run(tmp_path, {"ssh_user": "admin",
                                       "peer_ips": ["192.168.1.3"]})
        assert "User admin" in written

    def test_no_user_means_no_block(self, tmp_path):
        assert self._run(tmp_path, {"peer_ips": ["192.168.1.3"]}) == ""

    def test_a_broken_config_is_survivable(self, tmp_path):
        home = tmp_path / "ainode"
        home.mkdir()
        (home / "config.json").write_text("{ not json")
        result = subprocess.run(
            [sys.executable, "-c", _entrypoint_ssh_program()],
            env={"AINODE_HOME": str(home), "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0


class TestTheBackendWritesItToo:
    """The entrypoint runs once, at container start. ssh_user can be set after."""

    def _backend(self, tmp_path, ssh_user="admin"):
        backend = EugrBackend.__new__(EugrBackend)
        backend.config = NodeConfig(model="org/model", ssh_user=ssh_user)
        return backend

    def test_it_writes_the_mapping(self, tmp_path, monkeypatch):
        target = tmp_path / "config"
        monkeypatch.setattr("ainode.engine.backends.eugr.Path",
                            lambda p: target if p == "/root/.ssh/config" else Path(p))
        self._backend(tmp_path)._ensure_ssh_user()
        assert "User admin" in target.read_text()

    def test_it_is_idempotent(self, tmp_path, monkeypatch):
        target = tmp_path / "config"
        monkeypatch.setattr("ainode.engine.backends.eugr.Path",
                            lambda p: target if p == "/root/.ssh/config" else Path(p))
        backend = self._backend(tmp_path)
        backend._ensure_ssh_user()
        backend._ensure_ssh_user()
        assert target.read_text().count("User admin") == 1

    def test_it_keeps_what_was_already_there(self, tmp_path, monkeypatch):
        target = tmp_path / "config"
        target.write_text("Host github.com\n    IdentityFile ~/.ssh/gh\n")
        monkeypatch.setattr("ainode.engine.backends.eugr.Path",
                            lambda p: target if p == "/root/.ssh/config" else Path(p))
        self._backend(tmp_path)._ensure_ssh_user()
        written = target.read_text()
        assert "Host github.com" in written and "User admin" in written

    @pytest.mark.parametrize("bad", ["", "  ", "ad min", "admin; rm -rf /", "-oProxyCommand=x"])
    def test_a_junk_user_is_refused(self, tmp_path, monkeypatch, bad):
        # It lands in an ssh config file, which is a directive language.
        target = tmp_path / "config"
        monkeypatch.setattr("ainode.engine.backends.eugr.Path",
                            lambda p: target if p == "/root/.ssh/config" else Path(p))
        self._backend(tmp_path, ssh_user=bad)._ensure_ssh_user()
        assert not target.exists()

    def test_it_runs_before_the_launcher(self):
        # After would be too late: the launcher's first act is the ssh check.
        src = (REPO / "ainode" / "engine" / "backends" / "eugr.py").read_text()
        assert src.index("self._ensure_ssh_user()") < src.index(
            'cmd = [str(EUGR_LAUNCHER), "--no-cache-dirs"')


class TestTheHintMatchesTheEvidence:
    def test_it_names_the_compile_cache_first(self):
        # The traceback put the illegal instruction in a Triton kernel loaded
        # from ~/.cache/vllm, not in FlashInfer's prefill kernel. That cache
        # lives on the host and survives an engine-image change.
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("RuntimeError: CUDA driver error: an illegal instruction was encountered")
        tracker.fail("the launcher exited (code 1)")
        reason = tracker.failure_reason()
        assert "compile cache" in reason
        assert "FlashInfer" not in reason
        # And the fallbacks, in order.
        assert reason.index("compile cache") < reason.index("--enforce-eager")
        assert "compilation-config" in reason


class TestAbsoluteIdentityPaths:
    """The host's ssh config names keys by a path that does not exist here.

    nvidia-sync writes `IdentityFile /home/admin/.ssh/id_ed25519_nvsync_...`
    into the operator's ~/.ssh/config. The entrypoint copies the keys into
    /root/.ssh, but /home/admin is not mounted, so ssh skipped the key
    entirely:

        no such identity: /home/admin/.ssh/id_ed25519_nvsync_cluster_assistant:
            No such file or directory
        admin@10.100.36.2: Permission denied (publickey,password)

    Every weight transfer to a peer failed on authentication, which surfaced
    as "Failed to distribute <model> to <ip> (rc=255)".
    """

    def _run(self, tmp_path, config_text: str, keys=("id_ed25519_nvsync_cluster_assistant",),
             cfg: dict | None = None) -> str:
        home = tmp_path / "ainode"
        home.mkdir()
        (home / "config.json").write_text(json.dumps(cfg if cfg is not None else {}))
        ssh_dir = tmp_path / "ssh"
        ssh_dir.mkdir()
        (ssh_dir / "config").write_text(config_text)
        for key in keys:
            (ssh_dir / key).write_text("PRIVATE KEY")
        program = _entrypoint_ssh_program().replace("/root/.ssh", str(ssh_dir))
        result = subprocess.run([sys.executable, "-c", program],
                                env={"AINODE_HOME": str(home), "PATH": "/usr/bin:/bin"},
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return (ssh_dir / "config").read_text()

    def test_an_unresolvable_path_is_pointed_at_the_copy(self, tmp_path):
        written = self._run(
            tmp_path,
            "Host 10.100.36.2\n"
            "    IdentityFile /home/admin/.ssh/id_ed25519_nvsync_cluster_assistant\n",
        )
        assert "/home/admin" not in written
        assert written.rstrip().endswith("/ssh/id_ed25519_nvsync_cluster_assistant")
        assert "Host 10.100.36.2" in written        # the rest is untouched

    def test_indentation_survives(self, tmp_path):
        # ssh does not require it, but a config the operator opens later
        # should still read like the one they wrote.
        written = self._run(
            tmp_path,
            "Host peer\n\tIdentityFile /home/admin/.ssh/id_ed25519_nvsync_cluster_assistant\n",
        )
        assert "\n\tIdentityFile " in written

    def test_a_quoted_path_is_handled(self, tmp_path):
        written = self._run(
            tmp_path,
            'Host peer\n    IdentityFile "/home/admin/.ssh/id_ed25519_nvsync_cluster_assistant"\n',
        )
        assert "/home/admin" not in written

    def test_a_key_we_do_not_have_is_left_alone(self, tmp_path):
        """Rewriting it would be a lie: there is no copy to point at, and the
        original path is the only clue the operator gets about what is missing.
        """
        written = self._run(
            tmp_path,
            "Host peer\n    IdentityFile /home/admin/.ssh/some_other_key\n",
        )
        assert "/home/admin/.ssh/some_other_key" in written

    def test_a_resolvable_path_is_left_alone(self, tmp_path):
        # An IdentityFile that ssh can already open is none of our business,
        # wherever it lives.
        elsewhere = tmp_path / "mounted_key"
        elsewhere.write_text("PRIVATE KEY")
        written = self._run(tmp_path, f"Host peer\n    IdentityFile {elsewhere}\n")
        assert f"IdentityFile {elsewhere}" in written

    def test_it_happens_before_the_user_block_is_prepended(self, tmp_path):
        """Order matters only because the block is prepended: rewriting after
        would have to skip the freshly written lines. Doing it first keeps both
        halves independent."""
        written = self._run(
            tmp_path,
            "Host peer\n    IdentityFile /home/admin/.ssh/id_ed25519_nvsync_cluster_assistant\n",
            cfg={"ssh_user": "admin"},
        )
        assert "User admin" in written
        assert "/home/admin" not in written
        assert written.index("User admin") < written.index("IdentityFile")
