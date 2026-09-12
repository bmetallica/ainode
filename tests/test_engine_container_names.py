"""Two models on one node need two engine containers.

The eugr backend passed no name to the launcher, so every instance used its
default: `vllm_node`. Two consequences, both seen on hardware.

A second model launched on the same node landed in the FIRST model's
container — the launcher reuses a running container by name and execs into
it. And an engine that outlived an orchestrator restart silently owned both
the name and port 8000, so the next launch was allocated port 8001 and exec'd
into the hour-old container, while the instance list showed nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import get_backend
from ainode.engine.backends.eugr import EugrBackend, EugrBackendError


def _backend(instance_id=""):
    return EugrBackend(NodeConfig(model="org/model"), instance_id=instance_id)


class TestNaming:
    def test_the_primary_keeps_the_familiar_name(self):
        # Docs, runbooks and muscle memory all say vllm_node.
        assert _backend().container_name == "vllm_node"
        assert _backend()._container_args() == []

    def test_a_stacked_instance_gets_its_own(self):
        backend = _backend("8001")
        assert backend.container_name == "vllm_node-8001"
        assert backend._container_args() == ["--name", "vllm_node-8001"]

    def test_the_name_is_checked_before_it_reaches_a_shell(self):
        backend = _backend("8001; rm -rf /")
        with pytest.raises(EugrBackendError):
            backend._container_args()


class TestTheFactoryPassesItOn:
    def test_instance_id_reaches_the_eugr_backend(self):
        # It was accepted by get_backend and quietly dropped for this backend.
        backend = get_backend(NodeConfig(model="org/model", engine_backend="eugr"),
                              instance_id="8001")
        assert backend.container_name == "vllm_node-8001"

    def test_a_primary_still_gets_the_default(self):
        backend = get_backend(NodeConfig(model="org/model", engine_backend="eugr"))
        assert backend.container_name == "vllm_node"


class TestGeneratedScripts:
    def test_instances_do_not_share_a_script_file(self, tmp_path, monkeypatch):
        # Both wrote examples/ainode-solo.sh: two launches racing would have
        # one execute the other's command line.
        from ainode.engine.parallelism import ParallelPlan

        first = _backend()
        second = _backend("8001")
        paths = set()
        for backend in (first, second):
            paths.add(Path(backend._write_launch_script(ParallelPlan(), solo=True)).name)
        assert paths == {"ainode-solo.sh", "ainode-solo-8001.sh"}
