"""Unloading a distributed model has to free every node, not just the head.

Reported: after unloading GLM 5.3 Flash (TP=2), the `vllm_node` container was
still running on BOTH nodes with the weights still resident, and the operator
had to ssh to each machine and run `docker stop vllm_node` by hand.

The teardown did ask eugr's launch-cluster.sh to stop the cluster. Reading it
(MIT, read-only reference) shows why that is not enough:

  * `cleanup()` returns immediately when CLUSTER_WAS_RUNNING=true,
  * it runs after node autodetection, which exits 1 on failure — before
    cleanup is ever reached,
  * and for a peer it runs `docker stop` only, never `docker rm`, so the next
    launch inherits a stale container.

None of that is a bug in the launcher: it is written for an operator at a
shell who reads the message it prints. We know the peers and the container
name, so the teardown says it directly.
"""

from __future__ import annotations

from unittest import mock


from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend


def _backend(**kw) -> EugrBackend:
    config = NodeConfig(node_id="head", model="org/model", ssh_user="admin",
                        distributed_mode="head", **kw)
    backend = EugrBackend(config)
    return backend


def _commands(run) -> list[str]:
    return [" ".join(call[0][0]) for call in run.call_args_list]


class TestPeerContainers:
    def test_every_peer_is_stopped_and_removed(self):
        backend = _backend(peer_ips=["10.0.0.2", "10.0.0.3"])
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend._stop_peer_containers()
        cmds = _commands(run)
        for peer in ("10.0.0.2", "10.0.0.3"):
            assert any(f"admin@{peer}" in c and "docker stop -t 30 vllm_node" in c
                       for c in cmds), peer
            assert any(f"admin@{peer}" in c and "docker rm -f vllm_node" in c
                       for c in cmds), peer

    def test_it_uses_the_transfer_address(self):
        # Same reasoning as the weight copy: the direct link, not the
        # coordination path, and it is the address ssh actually reaches.
        backend = _backend(peer_ips=["192.168.1.3"],
                           peer_transfer_ips={"192.168.1.3": "10.100.36.2"})
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend._stop_peer_containers()
        assert all("10.100.36.2" in c for c in _commands(run))

    def test_a_stacked_instance_stops_its_own_container(self):
        """Stacked instances have their own container name. Stopping the
        primary's would take down a model nobody asked to unload."""
        config = NodeConfig(node_id="head", model="m", ssh_user="admin",
                            distributed_mode="head", peer_ips=["10.0.0.2"])
        backend = EugrBackend(config, instance_id="abc123")
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend._stop_peer_containers()
        cmds = _commands(run)
        assert all("vllm_node-abc123" in c for c in cmds)

    def test_solo_touches_no_peer(self):
        backend = _backend(peer_ips=[])
        with mock.patch("subprocess.run") as run:
            backend._stop_peer_containers()
        run.assert_not_called()

    def test_without_an_ssh_user_it_says_so_instead_of_guessing(self):
        backend = EugrBackend(NodeConfig(node_id="head", model="m",
                                         ssh_user="", distributed_mode="head",
                                         peer_ips=["10.0.0.2"]))
        with mock.patch("subprocess.run") as run:
            backend._stop_peer_containers()
        run.assert_not_called()

    def test_a_peer_that_is_unreachable_does_not_stop_the_others(self):
        backend = _backend(peer_ips=["10.0.0.2", "10.0.0.3"])

        def run(cmd, *a, **k):
            if "10.0.0.2" in " ".join(cmd):
                raise OSError("host unreachable")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=run) as patched:
            backend._stop_peer_containers()
        assert any("10.0.0.3" in c for c in _commands(patched))

    def test_no_such_container_is_not_worth_a_warning(self, caplog):
        backend = _backend(peer_ips=["10.0.0.2"])
        with mock.patch("subprocess.run", return_value=mock.Mock(
                returncode=1, stdout="", stderr="Error: No such container: vllm_node")):
            backend._stop_peer_containers()
        assert "No such container" not in caplog.text


class TestStopOrdersItRight:
    def test_stop_reaches_peers_and_this_node(self):
        backend = _backend(peer_ips=["10.0.0.2"])
        order = []
        with mock.patch.object(type(backend), "_stop_peer_containers",
                               lambda s: order.append("peers")), \
             mock.patch.object(type(backend), "_stop_container",
                               lambda s: order.append("local")), \
             mock.patch("ainode.engine.backends.eugr.EUGR_LAUNCHER") as launcher:
            launcher.exists.return_value = False
            backend.stop()
        assert order == ["peers", "local"]

    def test_the_phase_is_reset(self):
        """Otherwise the next card for this backend starts from the last
        launch's phase — a stopped instance reading "ready · 100%"."""
        backend = _backend(peer_ips=[])
        backend._phase.observe("Loading model weights")
        backend._ready = True
        with mock.patch.object(type(backend), "_stop_container", lambda s: None), \
             mock.patch("ainode.engine.backends.eugr.EUGR_LAUNCHER") as launcher:
            launcher.exists.return_value = False
            backend.stop()
        assert backend._phase.current(ready_latch=False) != "ready"


class TestUnloadDoesNotSkipTheTeardown:
    def test_it_does_not_gate_on_the_launcher_being_alive(self):
        """is_running() asks about the LAUNCHER process; the containers
        outlive it. Gating on it skipped the teardown entirely."""

        import inspect

        from ainode.models.api_routes import (
            handle_model_load,
            handle_model_unload,
        )

        for handler in (handle_model_unload, handle_model_load):
            source = inspect.getsource(handler)
            assert "engine.stop()" in source, handler.__name__
            assert "if engine.is_running():" not in source, handler.__name__


class TestTheCardStopsClaimingProgress:
    def test_a_dead_instance_that_reached_ready_is_not_drawn_at_100(self):
        """It read "SIZING THE KV CACHE · 100%" — the last thing the load was
        doing, at a full bar, on a model that had died."""
        from pathlib import Path

        source = Path("ainode/web/static/js/app.js").read_text()
        # The check moved into instanceState(), which decides the card's one
        # coloured word. The rule is the same: 'ready' as a PHASE on an
        # instance that is not answering means it stopped, not that it is at
        # 100%.
        assert "if (phase === 'ready') return { label: 'STOPPED ANSWERING'" in source
