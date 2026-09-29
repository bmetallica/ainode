"""A model spread across nodes outlives a restart of AINode.

B1 in upgrade-fixes.md, which turned out to be three bugs behind one line:

    def is_running(self):
        return self._process is not None and self._process.poll() is None

The eugr backend only believed in the launcher process it had spawned itself.
That process lives inside the AINode container, so after an update, a crash or
`systemctl restart ainode`, every engine read as stopped while it went on
serving. From that:

* the boot path called start_distributed() on a cluster that was running, and
  the launcher's first act is to stop what it finds — so every update of the
  orchestrator cost a full reload of the model across the nodes;
* the adoption of engines that outlived AINode (#203) adopted the image server,
  whose backend asks docker, and never an LLM;
* a distributed head was not seeded into the instance list at all, so the model
  vanished from INSTANCES, the router and the planner while it held the memory.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from ainode.core.config import NodeConfig
from ainode.engine.backends import eugr


def _backend(**config):
    cfg = NodeConfig(model="org/model", api_port=8000, **config)
    return eugr.EugrBackend(cfg)


class TestIsRunningAsksWhatIsActuallyThere:
    def test_the_launcher_it_spawned_counts(self):
        backend = _backend()

        class _Alive:
            def poll(self):
                return None

        backend._process = _Alive()
        assert backend.is_running() is True

    def test_a_container_that_answers_counts_too(self):
        with patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=True):
            assert _backend().is_running() is True

    def test_a_container_whose_engine_is_gone_does_not(self):
        """The launcher execs vLLM into a long-lived container. A container
        that outlived its engine must not read as a working model."""
        with patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=False):
            assert _backend().is_running() is False

    def test_no_container_is_not_running(self):
        with patch.object(eugr, "_container_up", return_value=False), \
                patch.object(eugr, "_answers", return_value=True):
            assert _backend().is_running() is False

    def test_an_adopted_engine_reads_as_ready(self):
        backend = _backend()
        with patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=True):
            backend.is_running()
        assert backend._ready is True

    def test_the_answer_is_cached_briefly(self):
        backend = _backend()
        calls = []

        def _up(name):
            calls.append(name)
            return True

        with patch.object(eugr, "_container_up", side_effect=_up), \
                patch.object(eugr, "_answers", return_value=True):
            backend.is_running()
            backend.is_running()
        assert len(calls) == 1

    def test_it_asks_about_its_own_container(self):
        seen = []
        with patch.object(eugr, "_container_up",
                          side_effect=lambda name: seen.append(name) or False):
            eugr.EugrBackend(NodeConfig(model="m", api_port=8001),
                             instance_id="8001").is_running()
        assert seen == ["vllm_node-8001"]


class TestAStartOfARunningClusterIsAnAdoption:
    def test_start_distributed_does_not_relaunch(self):
        backend = _backend(distributed_mode="head", peer_ips=["10.0.0.2"])
        with patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=True), \
                patch.object(eugr.subprocess, "Popen",
                             side_effect=AssertionError("relaunched")):
            assert backend.start_distributed() is True

    def test_start_solo_does_not_relaunch_either(self):
        backend = _backend()
        with patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=True), \
                patch.object(eugr.subprocess, "Popen",
                             side_effect=AssertionError("relaunched")):
            assert backend.start_solo() is True


class TestTheHeadIsOnTheBooksAfterABoot:
    def _app(self, **config):
        from ainode.api.server import create_app

        cfg = NodeConfig(node_id="head", model="org/big", api_port=8000,
                         onboarded=True, **config)
        return create_app(cfg, engine=_backend(**config))

    def test_a_distributed_primary_is_seeded(self):
        app = self._app(distributed_mode="head", peer_ips=["10.0.0.2"],
                        tensor_parallel_size=2)
        record = app["instances"].by_model("org/big").record
        assert record.peer_ips == ["10.0.0.2"]
        assert record.tensor_parallel_size == 2

    def test_a_solo_primary_still_is(self):
        app = self._app()
        record = app["instances"].by_model("org/big").record
        assert record.peer_ips == [] and record.tensor_parallel_size == 1

    def test_a_node_with_no_primary_still_has_a_manager(self):
        from ainode.api.server import create_app

        app = create_app(NodeConfig(node_id="n", onboarded=True), engine=None)
        assert app["instances"] is not None


class TestTheManifestRecordsTheSplit:
    def test_a_distributed_instance_is_written_with_its_peers(self, tmp_path):
        from ainode.models import api_routes

        class _Rec:
            model, api_port = "org/big", 8001

        class _Inst:
            record = _Rec()
            backend = type("B", (), {"config": NodeConfig(
                distributed_mode="head", peer_ips=["10.0.0.2"],
                tensor_parallel_size=2, parallel_strategy="tensor")})()

        class _Manager:
            def instances(self):
                return [_Inst()]

        path = tmp_path / "instances.json"
        with patch.object(api_routes, "_manifest_path", lambda: path):
            api_routes.save_instance_manifest({"instances": _Manager()})
        entry = json.loads(path.read_text())["instances"][0]
        assert entry["distributed"] is True
        assert entry["peer_ips"] == ["10.0.0.2"]
        assert entry["tensor_parallel_size"] == 2


class TestAStackedDistributedEngineIsAdopted:
    def test_vllm_node_with_a_port_suffix_is_adoptable(self):
        from ainode.models.api_routes import _running_engine_containers

        with patch("subprocess.run") as run:
            run.return_value.stdout = "ainode\nvllm_node-8001\nvllm_node\n"
            # The unsuffixed one is the primary, owned by the boot engine.
            assert _running_engine_containers() == [("vllm_node-8001", 8001)]

    def test_it_comes_back_with_its_peers_and_split(self, tmp_path):
        from ainode.engine.instance_manager import InstanceManager
        from ainode.models import api_routes

        path = tmp_path / "instances.json"
        path.write_text(json.dumps({"instances": [{
            "model": "org/big", "api_port": 8001, "distributed": True,
            "peer_ips": ["10.0.0.2"], "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1}]}))
        config = NodeConfig(node_id="head", api_port=8000)
        app = {"config": config,
               "instances": InstanceManager(base_port=8000)}
        with patch.object(api_routes, "_manifest_path", lambda: path), \
                patch.object(api_routes, "_running_engine_containers",
                             lambda: [("vllm_node-8001", 8001)]), \
                patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=True):
            assert api_routes.adopt_running_engines(app) == 1
        instance = app["instances"].by_model("org/big")
        assert instance.record.peer_ips == ["10.0.0.2"]
        assert instance.record.tensor_parallel_size == 2
        assert instance.backend.config.distributed_mode == "head"

    def test_a_dead_engine_in_a_live_container_is_not(self, tmp_path):
        from ainode.engine.instance_manager import InstanceManager
        from ainode.models import api_routes

        path = tmp_path / "instances.json"
        path.write_text(json.dumps({"instances": [{
            "model": "org/big", "api_port": 8001}]}))
        app = {"config": NodeConfig(node_id="head", api_port=8000),
               "instances": InstanceManager(base_port=8000)}
        with patch.object(api_routes, "_manifest_path", lambda: path), \
                patch.object(api_routes, "_running_engine_containers",
                             lambda: [("vllm_node-8001", 8001)]), \
                patch.object(eugr, "_container_up", return_value=True), \
                patch.object(eugr, "_answers", return_value=False):
            assert api_routes.adopt_running_engines(app) == 0
