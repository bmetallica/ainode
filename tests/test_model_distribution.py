"""Model weights reach every peer's local disk before a distributed launch.

Each rank reads the model from its own node. AINode does not use shared
storage for this — /mnt/shared-models carries only a 3 KB NCCL script — so the
head copies the weights, over a direct RoCE cable where one exists.

Before this, the eugr backend (the default) copied nothing: a peer without the
weights downloaded them from Hugging Face independently, N times over the WAN.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from ainode.core.config import NodeConfig, host_path
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.distribute import (
    DistributionError,
    ensure_peer_has_dir,
    hf_cache_dir_name,
)


class TestHfCacheDirName:
    def test_repo_id_becomes_the_cache_dir(self):
        assert hf_cache_dir_name("org/model-x") == "models--org--model-x"

    def test_only_the_first_slash_matters(self):
        assert hf_cache_dir_name("a/b") == "models--a--b"


class TestEnsurePeerHasDir:
    def _src(self, tmp_path, name="models--org--m"):
        d = tmp_path / "hub" / name
        d.mkdir(parents=True)
        (d / "weights.safetensors").write_bytes(b"x" * 64)
        return str(tmp_path / "hub"), name

    def test_missing_source_returns_false_without_touching_the_peer(self, tmp_path):
        with mock.patch("subprocess.run") as run:
            got = ensure_peer_has_dir(
                ssh_user="u", transfer_ip="10.0.0.2",
                source_parent=str(tmp_path), dir_name="models--nope",
                target_parent=str(tmp_path),
            )
        assert got is False
        run.assert_not_called()

    def test_peer_that_already_has_it_is_not_copied_to(self, tmp_path):
        parent, name = self._src(tmp_path)
        calls = []

        def fake(cmd, *a, **k):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="present\n", stderr="")

        with mock.patch("subprocess.run", side_effect=fake):
            got = ensure_peer_has_dir(
                ssh_user="u", transfer_ip="10.0.0.2", source_parent=parent,
                dir_name=name, target_parent=parent,
            )
        assert got is True
        assert len(calls) == 1  # the probe only

    def test_rsync_is_preferred_when_available(self, tmp_path):
        parent, name = self._src(tmp_path)
        calls = []

        def fake(cmd, *a, **k):
            calls.append(cmd)
            if "test -d" in " ".join(cmd):
                return mock.Mock(returncode=0, stdout="missing\n", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake), \
             mock.patch("shutil.which", return_value="/usr/bin/rsync"):
            assert ensure_peer_has_dir(
                ssh_user="u", transfer_ip="10.0.0.2", source_parent=parent,
                dir_name=name, target_parent=parent,
            )
        assert any(c[0] == "rsync" for c in calls)
        rsync = next(c for c in calls if c[0] == "rsync")
        assert "--partial" in rsync          # resumable across a re-launch

    def test_tar_fallback_without_rsync(self, tmp_path):
        parent, name = self._src(tmp_path)
        calls = []

        def fake(cmd, *a, **k):
            calls.append(cmd)
            if "test -d" in " ".join(cmd):
                return mock.Mock(returncode=0, stdout="missing\n", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake), \
             mock.patch("shutil.which", return_value=None):
            assert ensure_peer_has_dir(
                ssh_user="u", transfer_ip="10.0.0.2", source_parent=parent,
                dir_name=name, target_parent=parent,
            )
        assert any(c[0] == "bash" and "tar -C" in c[2] for c in calls)

    def test_a_failed_transfer_raises(self, tmp_path):
        parent, name = self._src(tmp_path)

        def fake(cmd, *a, **k):
            if "test -d" in " ".join(cmd):
                return mock.Mock(returncode=0, stdout="missing\n", stderr="")
            return mock.Mock(returncode=23, stdout="", stderr="rsync: link failed")

        with mock.patch("subprocess.run", side_effect=fake), \
             mock.patch("shutil.which", return_value="/usr/bin/rsync"), \
             pytest.raises(DistributionError, match="rsync: link failed"):
            ensure_peer_has_dir(
                ssh_user="u", transfer_ip="10.0.0.2", source_parent=parent,
                dir_name=name, target_parent=parent,
            )


class TestEugrDistribution:
    def _backend(self, tmp_path, **overrides):
        cfg = dict(
            node_id="head", model="org/m", distributed_mode="head",
            peer_ips=["192.168.1.3", "192.168.1.4"],
            peer_transfer_ips={"192.168.1.3": "10.100.36.2"},
            models_dir=str(tmp_path / "models"), ssh_user="admin",
        )
        cfg.update(overrides)
        hub = tmp_path / "models" / "hub" / "models--org--m"
        hub.mkdir(parents=True)
        (hub / "w.safetensors").write_bytes(b"x" * 32)
        return EugrBackend(NodeConfig(**cfg))

    def test_direct_link_is_used_where_one_exists(self, tmp_path):
        backend = self._backend(tmp_path)
        seen = []
        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_dir",
                        side_effect=lambda **kw: seen.append(kw) or True):
            backend._distribute_model_to_peers()
        targets = [k["transfer_ip"] for k in seen]
        # Peer with a cable gets the RoCE address; the other keeps coordination.
        assert targets == ["10.100.36.2", "192.168.1.4"]
        assert all(k["dir_name"] == "models--org--m" for k in seen)
        assert all(k["ssh_user"] == "admin" for k in seen)

    def test_nothing_is_sent_when_the_head_lacks_the_weights(self, tmp_path):
        backend = self._backend(tmp_path, model="org/other")
        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_dir",
                        return_value=False) as send:
            backend._distribute_model_to_peers()
        # Probes the first peer, learns the source is missing, stops.
        assert send.call_count == 1

    def test_an_unreachable_peer_does_not_abort_the_launch(self, tmp_path):
        backend = self._backend(tmp_path)
        calls = []

        def flaky(**kw):
            calls.append(kw["transfer_ip"])
            if kw["transfer_ip"] == "10.100.36.2":
                raise OSError("no route to host")
            return True

        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_dir",
                        side_effect=flaky):
            backend._distribute_model_to_peers()   # must not raise
        assert calls == ["10.100.36.2", "192.168.1.4"]

    def test_a_failed_transfer_does_abort(self, tmp_path):
        """A half-copied checkpoint is worse than none."""
        backend = self._backend(tmp_path)
        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_dir",
                        side_effect=DistributionError("disk full")), \
             pytest.raises(DistributionError):
            backend._distribute_model_to_peers()

    def test_solo_model_unset_is_a_no_op(self, tmp_path):
        backend = self._backend(tmp_path, model="")
        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_dir") as send:
            backend._distribute_model_to_peers()
        send.assert_not_called()


class TestHostPathTranslation:
    def test_no_op_when_not_containerised(self, monkeypatch):
        monkeypatch.delenv("AINODE_HOST_HOME", raising=False)
        assert host_path("/root/.ainode/models") == "/root/.ainode/models"

    def test_rewrites_paths_under_ainode_home(self, monkeypatch):
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/admin/.ainode")
        monkeypatch.setattr("ainode.core.config.AINODE_HOME", Path("/root/.ainode"))
        assert host_path("/root/.ainode/models") == "/home/admin/.ainode/models"

    def test_leaves_unrelated_paths_alone(self, monkeypatch):
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/admin/.ainode")
        monkeypatch.setattr("ainode.core.config.AINODE_HOME", Path("/root/.ainode"))
        assert host_path("/mnt/shared-models") == "/mnt/shared-models"

    def test_a_sibling_prefix_is_not_rewritten(self, monkeypatch):
        """/root/.ainode-backup must not be caught by a naive startswith."""
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/admin/.ainode")
        monkeypatch.setattr("ainode.core.config.AINODE_HOME", Path("/root/.ainode"))
        assert host_path("/root/.ainode-backup") == "/root/.ainode-backup"
