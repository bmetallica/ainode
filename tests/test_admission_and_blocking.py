"""Two gates and one thread rule.

The admission gate is what should have refused the launch that took two nodes
down; the executor is why a node stops disappearing from the cluster while it
loads. They are tested together because they are the two halves of the same
failure: nothing checked on the way in, and nothing could have answered while
it was happening.
"""

from __future__ import annotations

import asyncio
import json
import time

from ainode.planner.facts import facts_from_config
from ainode.safety.admission import check_admission

MINIMAX = {
    "num_hidden_layers": 62, "num_attention_heads": 48,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 6144,
    "max_position_embeddings": 196608, "torch_dtype": "bfloat16",
}


class _Node:
    def __init__(self, node_id, free_mb=120000.0):
        self.node_id = node_id
        self.node_name = node_id
        self.status = "online"
        self.gpu_memory_gb = 122.0
        self.gpu_memory_total_mb = 124928.0
        self.gpu_memory_used_mb = 124928.0 - free_mb
        self.model = ""
        self.instances = []
        self.embedding_models = []


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


class _Config:
    node_id = "n1"
    api_port = 8000


class _Manager:
    def _catalog_lookup(self, model_id):
        return None

    def model_dirs_for_repo(self, repo):
        return []


class _Guard:
    def __init__(self, refusal="", warn_mb=8192):
        self._refusal = refusal
        self.warn_mb = warn_mb

    def accepting_loads(self):
        return self._refusal

    def read(self):
        return type("R", (), {"warn_mb": self.warn_mb})()


def _app(free_mb=120000.0, refusal="", weights_gb=130.0):
    app = {
        "config": _Config(),
        "cluster_state": _Cluster([_Node("n1", free_mb), _Node("n2", free_mb),
                                   _Node("n3", free_mb)]),
        "model_manager": _Manager(),
        "memory_guard": _Guard(refusal),
    }
    app["_facts"] = facts_from_config(MINIMAX, "org/m", int(weights_gb * 1e9))
    return app


def _with_facts(monkeypatch, app):
    import ainode.safety.admission as admission

    monkeypatch.setattr(admission, "local_facts",
                        lambda manager, model: app["_facts"], raising=False)
    import ainode.planner.facts as facts_module
    monkeypatch.setattr(facts_module, "local_facts",
                        lambda manager, model: app["_facts"])


class TestTheGate:
    def test_the_guard_refuses_first(self, monkeypatch):
        # No arithmetic needed: the host is already short.
        app = _app(refusal="Refusing to start: only 900 MB …")
        assert "900 MB" in check_admission(app, "org/m")

    def test_a_model_that_fits_passes(self, monkeypatch):
        app = _app()
        _with_facts(monkeypatch, app)
        assert check_admission(app, "org/m", max_model_len=65536) == ""

    def test_a_model_that_cannot_fit_is_refused_before_it_starts(self, monkeypatch):
        # The point of the whole exercise: the engine used to discover this
        # after loading the weights, and on this hardware that discovery takes
        # the node down.
        app = _app(weights_gb=500.0)
        _with_facts(monkeypatch, app)
        refusal = check_admission(app, "org/m")
        assert refusal
        assert "refusing the launch" in refusal
        assert "force" in refusal

    def test_force_overrides_everything(self, monkeypatch):
        # The planner is conservative on purpose, and an operator who knows
        # what a launch needs is allowed to be right.
        app = _app(weights_gb=500.0, refusal="host is short")
        _with_facts(monkeypatch, app)
        assert check_admission(app, "org/m", force=True) == ""

    def test_a_model_not_on_disk_is_not_refused(self, monkeypatch):
        # Nothing to compute from, and the launch may be a distributed one
        # whose weights live on a peer.
        app = _app()
        app["_facts"] = facts_from_config(MINIMAX, "org/m", weight_bytes=0)
        _with_facts(monkeypatch, app)
        assert check_admission(app, "org/m") == ""

    def test_an_unreadable_checkpoint_is_not_refused(self, monkeypatch):
        # A gate that refuses what it does not understand would block every
        # checkpoint whose config.json it cannot parse.
        app = _app()
        app["_facts"] = facts_from_config({}, "org/m", int(10e9))
        _with_facts(monkeypatch, app)
        assert check_admission(app, "org/m") == ""

    def test_it_plans_into_the_reserve_the_guard_enforces(self, monkeypatch):
        # The two numbers have to say the same thing, or the gate lets through
        # exactly what the guard then kills.
        loose = _app(free_mb=100000.0)
        loose["memory_guard"] = _Guard(warn_mb=4096)
        _with_facts(monkeypatch, loose)
        tight = _app(free_mb=100000.0, weights_gb=130.0)
        tight["memory_guard"] = _Guard(warn_mb=60000)
        tight["_facts"] = loose["_facts"]
        _with_facts(monkeypatch, tight)
        assert check_admission(loose, "org/m", max_model_len=65536) == ""
        assert check_admission(tight, "org/m", max_model_len=65536) != ""

    def test_no_guard_at_all_is_not_a_refusal(self, monkeypatch):
        app = _app()
        app.pop("memory_guard")
        _with_facts(monkeypatch, app)
        assert check_admission(app, "org/m", max_model_len=65536) == ""


class TestTheLaunchLeavesTheLoopAlone:
    """A node must not vanish from the cluster because it is loading."""

    def test_the_solo_path_runs_the_launch_in_a_worker_thread(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_model_load)
        assert "run_in_executor" in source
        assert "append_solo_instance(request.app" not in source

    def test_the_distributed_path_does_too(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "run_in_executor" in source
        assert "started = backend.start_distributed()" not in source

    def test_the_unload_path_does_too(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_model_unload)
        assert "run_in_executor" in source

    def test_a_slow_launch_does_not_stop_the_loop(self):
        # The behaviour all three of the above exist for: while the launch is
        # running, this loop keeps ticking — which is where the UDP
        # announcement lives, and the API, and the memory guard's API surface.
        ticks = []

        async def _scenario():
            async def _announcer():
                for _ in range(20):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            def _slow_launch():
                time.sleep(0.15)
                return "launched"

            task = asyncio.get_event_loop().create_task(_announcer())
            result = await asyncio.get_event_loop().run_in_executor(
                None, _slow_launch)
            await task
            return result

        assert asyncio.run(_scenario()) == "launched"
        # Without the executor the announcer would have produced one tick.
        assert len(ticks) >= 10


class _Req:
    def __init__(self, app, body):
        self.app = app
        self._body = body

    async def json(self):
        return self._body


class TestTheRefusalReachesTheCaller:
    def test_the_solo_route_answers_507_with_the_reason(self, monkeypatch):
        from ainode.models import api_routes

        app = _app(refusal="Refusing to start: only 900 MB of host memory")
        app["engine"] = None
        monkeypatch.setattr(api_routes, "check_admission",
                            lambda *a, **k: "no room", raising=False)
        import ainode.safety.admission as admission
        monkeypatch.setattr(admission, "check_admission", lambda *a, **k: "no room")

        resp = asyncio.run(api_routes.handle_model_load(
            _Req(app, {"model": "org/m"})))
        assert resp.status == 507
        assert json.loads(resp.body)["error"] == "no room"
        assert json.loads(resp.body)["refused_by"] == "admission"
