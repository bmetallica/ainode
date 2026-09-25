"""An engine that outlived the orchestrator is still an engine.

Reported from the cluster:

    wenn ainode neu gestartet wird erkennt es das laufende diffusion immage
    nicht mehr als laufend … INSTANCES im UI bleibt leer

        CONTAINER ID   IMAGE                     STATUS          NAMES
        0d72f7ccf15f   ainode:dev                Up 49 seconds   ainode
        d20d57843dca   ainode-diffusers:latest   Up 15 hours     ainode_image-8001

Restarting AINode does not stop the engines — they are separate containers on
purpose, so that updating the orchestrator does not take a fifteen-hour-old
image server down with it. Nothing put them back on the books, though. The
only path that noticed was the manifest replay, which relaunches from disk and
adopts by accident because a backend's start() returns early when its
container is up — and that path sleeps ten seconds and then waits up to five
minutes for the PRIMARY port to answer, which on a node whose only engine is a
stacked one never happens.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ainode.core.config import NodeConfig


class _Rec:
    def __init__(self, model, port, kind=""):
        self.model = model
        self.api_port = port
        self.kind = kind


class _Inst:
    def __init__(self, record, backend):
        self.record = record
        self.backend = backend


class _Manager:
    def __init__(self):
        self.added = []

    def instances(self):
        return list(self.added)

    def by_model(self, model):
        for i in self.added:
            if i.record.model == model:
                return i
        return None

    def add(self, record, backend):
        self.added.append(_Inst(record, backend))


class _Backend:
    """A backend whose container is up, like the real one after a restart."""
    running = True

    def __init__(self, config, on_ready=None, instance_id=""):
        self.config = config
        self.instance_id = instance_id
        self.started = False

    def is_running(self):
        return type(self).running

    def start(self):
        self.started = True
        return True


def _app(tmp_path, manifest=None):
    config = NodeConfig(node_id="head", api_port=8000)
    config.models_dir = str(tmp_path / "models")
    app = {"config": config, "instances": _Manager()}
    entries = manifest if manifest is not None else []
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"instances": entries}))
    return app, path


def _adopt(app, path, containers, served=""):
    from ainode.models import api_routes

    with patch.object(api_routes, "_manifest_path", lambda: path), \
            patch.object(api_routes, "_running_engine_containers",
                         lambda: containers), \
            patch.object(api_routes, "_model_on_port", lambda p: served), \
            patch("ainode.engine.backends.get_backend", _Backend):
        return api_routes.adopt_running_engines(app)


class TestTheImageServerComesBack:
    def test_it_is_adopted_from_the_manifest(self, tmp_path):
        app, path = _app(tmp_path, [
            {"model": "Qwen/Qwen-Image-2.1", "api_port": 8001,
             "engine_backend": "diffusers"},
        ])
        assert _adopt(app, path, [("ainode_image-8001", 8001)]) == 1
        record = app["instances"].added[0].record
        assert record.model == "Qwen/Qwen-Image-2.1"
        assert record.api_port == 8001
        assert record.status == "serving"

    def test_it_is_marked_as_an_image_instance(self, tmp_path):
        # The card, the profile and the telemetry read this rather than asking
        # the backend what class it happens to be.
        app, path = _app(tmp_path, [
            {"model": "Qwen/Qwen-Image-2.1", "api_port": 8001,
             "engine_backend": "diffusers"},
        ])
        _adopt(app, path, [("ainode_image-8001", 8001)])
        assert app["instances"].added[0].record.kind == "image"

    def test_an_llm_is_not_mislabelled(self, tmp_path):
        app, path = _app(tmp_path, [{"model": "org/llm", "api_port": 8002}])
        _adopt(app, path, [("ainode-vllm-node-solo-8002", 8002)])
        assert app["instances"].added[0].record.kind == "llm"

    def test_nothing_is_launched(self, tmp_path):
        # Adoption asks; it does not start. A start() here would relaunch a
        # fifteen-hour-old server that was serving perfectly well.
        app, path = _app(tmp_path, [{"model": "org/llm", "api_port": 8002}])
        _adopt(app, path, [("ainode-vllm-node-solo-8002", 8002)])
        backend = app["instances"].added[0].backend
        assert backend.started is False


class TestItAsksTheEngineWhenTheManifestCannot:
    def test_the_port_answers_for_itself(self, tmp_path):
        # A manifest written before api_port was recorded, or none at all.
        app, path = _app(tmp_path, [])
        assert _adopt(app, path, [("ainode_image-8001", 8001)],
                      served="Qwen/Qwen-Image-2.1") == 1
        assert app["instances"].added[0].record.model == "Qwen/Qwen-Image-2.1"

    def test_an_engine_that_will_not_say_is_left_alone(self, tmp_path):
        # Still loading, or wedged. The replay can deal with it; guessing a
        # model name here would put a wrong record in front of the operator.
        app, path = _app(tmp_path, [])
        assert _adopt(app, path, [("ainode_image-8001", 8001)], served="") == 0
        assert app["instances"].added == []


class TestItDoesNotDoubleUp:
    def test_a_model_already_registered_is_skipped(self, tmp_path):
        app, path = _app(tmp_path, [{"model": "org/llm", "api_port": 8002}])
        app["instances"].add(_Rec("org/llm", 8002), object())
        assert _adopt(app, path, [("ainode-vllm-node-solo-8002", 8002)]) == 0
        assert len(app["instances"].added) == 1

    def test_a_container_whose_engine_is_gone_is_skipped(self, tmp_path):
        app, path = _app(tmp_path, [{"model": "org/llm", "api_port": 8002}])
        _Backend.running = False
        try:
            assert _adopt(app, path, [("ainode-vllm-node-solo-8002", 8002)]) == 0
        finally:
            _Backend.running = True


class TestWhichContainersCount:
    @pytest.mark.parametrize("name,port", [
        ("ainode_image-8001", 8001),
        ("ainode-vllm-node-solo-8002", 8002),
    ])
    def test_a_suffixed_engine_container_is_adoptable(self, name, port):
        from ainode.models.api_routes import _running_engine_containers

        with patch("subprocess.run") as run:
            run.return_value.stdout = f"ainode\n{name}\nainode-registry-cache\n"
            assert _running_engine_containers() == [(name, port)]

    def test_the_orchestrator_and_the_registries_are_not_engines(self):
        from ainode.models.api_routes import _running_engine_containers

        with patch("subprocess.run") as run:
            run.return_value.stdout = ("ainode\nainode-registry-local\n"
                                       "ainode-registry-cache\n")
            assert _running_engine_containers() == []

    def test_the_primary_keeps_its_unsuffixed_name_and_is_not_adopted(self):
        # It is owned by the boot engine, which claims it from config.model.
        from ainode.models.api_routes import _running_engine_containers

        with patch("subprocess.run") as run:
            run.return_value.stdout = "ainode_image\nainode-vllm-node-solo\n"
            assert _running_engine_containers() == []


class TestTheManifestRecordsThePort:
    def test_it_is_written(self, tmp_path):
        from ainode.models import api_routes

        manager = _Manager()
        backend = _Backend(NodeConfig(distributed_mode="solo"))
        manager.add(_Rec("org/llm", 8002), backend)
        app = {"instances": manager}
        path = tmp_path / "instances.json"
        with patch.object(api_routes, "_manifest_path", lambda: path):
            api_routes.save_instance_manifest(app)
        assert json.loads(path.read_text())["instances"][0]["api_port"] == 8002


class TestItRunsBeforeAnythingElse:
    def test_the_startup_path_adopts_first(self):
        source = (__import__("pathlib").Path(__file__).resolve().parent.parent
                  / "ainode" / "api" / "server.py").read_text()
        adopt = source.index("adopt_running_engines, app")
        restore = source.index("await startup_restore(app)")
        replay = source.index("await replay_instances_on_startup(app)")
        assert adopt < restore < replay
