"""Unloading a model has to free the machine, not just the bookkeeping.

stop() killed the launcher process, and the launcher is not the engine:
`vllm serve` runs inside the engine container, started by `docker exec`. So an
unloaded model kept its container up and its weights resident — the node
stayed full, and the next launch found the port taken and the container name
in use by a model the operator had already dismissed. The hour-old vllm_node
that hijacked a later launch came from exactly this.
"""

from __future__ import annotations

from unittest import mock


from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend


def _backend(mode="solo", instance_id=""):
    backend = EugrBackend(NodeConfig(model="org/model", distributed_mode=mode,
                                     peer_ips=["10.0.0.2"] if mode == "head" else []),
                          instance_id=instance_id)
    return backend


def _docker_calls(run):
    return [c.args[0] for c in run.call_args_list
            if c.args and isinstance(c.args[0], list) and c.args[0][:1] == ["docker"]]


class TestStopRemovesTheContainer:
    def test_a_solo_stop_stops_and_removes_it(self, tmp_path):
        backend = _backend()
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend.stop()
        calls = _docker_calls(run)
        assert ["docker", "stop", "-t", "30", "vllm_node"] in calls
        assert ["docker", "rm", "-f", "vllm_node"] in calls

    def test_a_stacked_instance_removes_its_own(self):
        backend = _backend(instance_id="8001")
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend.stop()
        calls = _docker_calls(run)
        assert ["docker", "rm", "-f", "vllm_node-8001"] in calls
        # And emphatically not the primary's.
        assert not any(c[-1] == "vllm_node" for c in calls)

    def test_a_missing_container_is_not_an_error(self):
        # The normal case after a launch that failed before the container
        # existed, or when a previous stop already won.
        backend = _backend()
        with mock.patch("subprocess.run", return_value=mock.Mock(
                returncode=1, stdout="", stderr="Error: No such container: vllm_node")):
            backend.stop()  # must not raise

    def test_docker_being_unreachable_does_not_raise(self):
        backend = _backend()
        with mock.patch("subprocess.run", side_effect=OSError("docker is gone")):
            backend.stop()


class TestDistributedStop:
    def test_the_launcher_is_asked_to_stop_the_peers(self, tmp_path, monkeypatch):
        # Only the launcher can reach the peers' containers over SSH.
        import ainode.engine.backends.eugr as E

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr(E, "EUGR_LAUNCHER", launcher)
        backend = _backend(mode="head")
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend.stop()
        launcher_calls = [c.args[0] for c in run.call_args_list
                          if c.args and str(launcher) in str(c.args[0])]
        assert launcher_calls and launcher_calls[0][1] == "stop"

    def test_and_the_local_container_is_removed_anyway(self, tmp_path, monkeypatch):
        # The launcher's stop is best effort, and a solo launch never went
        # through it at all.
        import ainode.engine.backends.eugr as E

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr(E, "EUGR_LAUNCHER", launcher)
        backend = _backend(mode="head")
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend.stop()
        assert ["docker", "rm", "-f", "vllm_node"] in _docker_calls(run)

    def test_the_stop_names_the_container(self, tmp_path, monkeypatch):
        import ainode.engine.backends.eugr as E

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr(E, "EUGR_LAUNCHER", launcher)
        backend = _backend(mode="head", instance_id="8001")
        with mock.patch("subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr="")) as run:
            backend.stop()
        launcher_call = next(c.args[0] for c in run.call_args_list
                             if c.args and str(launcher) in str(c.args[0]))
        assert "--name" in launcher_call
        assert "vllm_node-8001" in launcher_call


class TestTheCrashIsExplained:
    def test_an_illegal_instruction_names_the_remedy(self):
        # Seen on the cluster: a model died during CUDA-graph capture with
        # RuntimeError: cudaErrorIllegalInstruction, which says nothing about
        # what to do. This repo has known the answer since 2026-06-17.
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("Capturing CUDA graphs (FULL):  40%")
        tracker.observe("RuntimeError: cudaErrorIllegalInstruction")
        tracker.fail("the launcher exited (code 1)")
        reason = tracker.failure_reason()
        assert "cudaErrorIllegalInstruction" in reason
        assert "--enforce-eager" in reason
