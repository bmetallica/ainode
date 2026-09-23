"""Keeping the host alive when an engine asks for more than there is.

Two nodes were lost launching a model whose memory did not fit. On GB10 that
is not an ordinary out-of-memory: the GPU's memory IS the host's memory, so an
engine that over-allocates does not get a CUDA error — it starves the kernel,
and the node has to be power-cycled.

Three things are tested here, and the third is the one that is easy to get
wrong: the guard must keep working while the event loop is blocked, because
the launch path blocks it for minutes and that is exactly the window the guard
exists to cover.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from ainode.safety.memory_guard import (
    MAX_RESERVE_SHARE,
    MemoryGuard,
    PRESETS,
    host_available_mb,
)


def _meminfo(tmp_path, available_mb, total_mb=128000):
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {int(total_mb * 1024)} kB\n"
        f"MemFree:        1024 kB\n"
        f"MemAvailable:   {int(available_mb * 1024)} kB\n")
    return path


class _Record:
    def __init__(self, model):
        self.model = model
        self.load_error = ""
        self.load_phase = ""
        self.status = "serving"


class _Backend:
    def __init__(self):
        self.killed = False
        self.stopped = False

    def kill(self):
        self.killed = True

    def stop(self):
        self.stopped = True


class _Instance:
    def __init__(self, model):
        self.record = _Record(model)
        self.backend = _Backend()


class _Manager:
    def __init__(self, models):
        self._instances = [_Instance(m) for m in models]

    def instances(self):
        return list(self._instances)


def _guard(tmp_path, available_mb, models=("org/a", "org/b"), **kw):
    app = {"instances": _Manager(list(models))}
    return MemoryGuard(app, meminfo=_meminfo(tmp_path, available_mb),
                       poll_seconds=0.01, **kw)


class TestReadingTheHost:
    def test_it_reads_memavailable_not_memfree(self, tmp_path):
        # MemFree on a node that has just read forty gigabytes of weights is
        # near zero while the machine is perfectly healthy. Refusing a launch
        # on that reading would refuse every launch.
        path = _meminfo(tmp_path, available_mb=40000)
        assert round(host_available_mb(path)) == 40000

    def test_an_unreadable_file_is_unknown_not_zero(self, tmp_path):
        assert host_available_mb(tmp_path / "nope") is None

    def test_unknown_never_blocks_a_launch(self, tmp_path):
        # A safety net, not a gate that fails closed on a platform whose /proc
        # looks different.
        guard = MemoryGuard({}, meminfo=tmp_path / "nope")
        assert guard.read().readable is False
        assert guard.accepting_loads() == ""


class TestRefusingALaunch:
    def test_it_refuses_below_the_warning_line(self, tmp_path):
        guard = _guard(tmp_path, available_mb=6000, warn_gb=8, critical_gb=4)
        refusal = guard.accepting_loads()
        assert refusal
        assert "6000 MB" in refusal and "8192 MB" in refusal

    def test_it_allows_above_it(self, tmp_path):
        assert _guard(tmp_path, available_mb=20000).accepting_loads() == ""

    def test_the_refusal_explains_the_hardware(self, tmp_path):
        # "Out of memory" is not why this matters here.
        refusal = _guard(tmp_path, available_mb=1000).accepting_loads()
        assert "one pool" in refusal and "takes the node down" in refusal

    def test_disabled_means_disabled(self, tmp_path):
        guard = _guard(tmp_path, available_mb=100, enabled=False)
        assert guard.accepting_loads() == ""


class TestKillingTheNewest:
    def test_below_the_critical_line_it_acts_at_once(self, tmp_path):
        # It used to wait for a second sample. On this hardware the second
        # sample is taken on a node that is already gone: 4 GB of headroom is
        # a fraction of a second at memory bandwidth.
        guard = _guard(tmp_path, available_mb=1000, critical_gb=4, warn_gb=8)
        guard._tick()
        assert guard._app["instances"].instances()[-1].backend.killed is True

    def test_the_warning_band_alone_is_not_a_verdict(self, tmp_path):
        # Below warn, above critical, and steady: a node serving two models
        # sits here legitimately.
        guard = _guard(tmp_path, available_mb=6000, critical_gb=4, warn_gb=8)
        guard._tick()
        guard._tick()
        assert guard._app["instances"].instances()[-1].backend.killed is False

    def test_it_recovers_without_acting(self, tmp_path):
        # In and out of the warning band at a survivable rate: the page cache
        # being reclaimed behind a load looks like this.
        ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
        guard = _guard(tmp_path, available_mb=6000, critical_gb=4, warn_gb=8,
                       clock=lambda: next(ticks))
        guard._tick()
        guard._meminfo = _meminfo(tmp_path, available_mb=9000)
        guard._tick()
        guard._meminfo = _meminfo(tmp_path, available_mb=6500)
        guard._tick()
        assert guard._app["instances"].instances()[-1].backend.killed is False


class TestFallingFast:
    """The line that was never observed being crossed.

    An engine sizing its KV cache goes from "plenty" to "none" between two
    samples. Waiting for a reading below the critical line means waiting for
    a reading that is never taken — the node stops answering first.
    """

    def _clock(self, values):
        it = iter(values)
        return lambda: next(it)

    def test_a_steep_drop_inside_the_warning_band_acts(self, tmp_path):
        guard = _guard(tmp_path, available_mb=60000, critical_gb=4, warn_gb=8,
                       clock=self._clock([0.0, 1.0, 2.0, 3.0]))
        guard._tick()                                    # 60 GB, calm
        guard._meminfo = _meminfo(tmp_path, available_mb=7000)
        guard._tick()                                    # 53 GB gone in a second
        assert guard._app["instances"].instances()[-1].backend.killed is True

    def test_a_steep_drop_that_lands_safely_does_not(self, tmp_path):
        # The same slope, but still far above the warning line: a model
        # finished loading, which is what that looks like.
        guard = _guard(tmp_path, available_mb=120000, critical_gb=4, warn_gb=8,
                       clock=self._clock([0.0, 1.0, 2.0, 3.0]))
        guard._tick()
        guard._meminfo = _meminfo(tmp_path, available_mb=60000)
        guard._tick()
        assert guard._app["instances"].instances()[-1].backend.killed is False

    def test_a_slow_slide_into_the_band_does_not(self, tmp_path):
        guard = _guard(tmp_path, available_mb=9000, critical_gb=4, warn_gb=8,
                       clock=self._clock([0.0, 1.0, 2.0, 3.0]))
        guard._tick()
        guard._meminfo = _meminfo(tmp_path, available_mb=7000)
        guard._tick()
        assert guard._app["instances"].instances()[-1].backend.killed is False

    def test_it_takes_the_newest_and_nothing_else(self, tmp_path):
        guard = _guard(tmp_path, available_mb=500, models=("org/old", "org/new"))
        guard.act(guard.read())
        instances = guard._app["instances"].instances()
        assert instances[-1].backend.killed is True
        assert instances[0].backend.killed is False

    def test_it_kills_rather_than_stops(self, tmp_path):
        # A graceful teardown takes a minute per container. A node this close
        # to the edge does not have a minute.
        guard = _guard(tmp_path, available_mb=500)
        guard.act(guard.read())
        newest = guard._app["instances"].instances()[-1]
        assert newest.backend.killed is True
        assert newest.backend.stopped is False

    def test_the_instance_says_what_happened(self, tmp_path):
        guard = _guard(tmp_path, available_mb=500, critical_gb=4)
        guard.act(guard.read())
        record = guard._app["instances"].instances()[-1].record
        assert "host memory guard" in record.load_error
        assert "500 MB" in record.load_error
        assert record.load_phase == "failed"

    def test_nothing_to_stop_is_not_a_crash(self, tmp_path):
        guard = MemoryGuard({"instances": _Manager([])},
                            meminfo=_meminfo(tmp_path, 500))
        assert guard.act(guard.read()) is None

    def test_a_backend_that_refuses_to_die_falls_back_to_stop(self, tmp_path):
        guard = _guard(tmp_path, available_mb=500)
        newest = guard._app["instances"].instances()[-1]

        def _boom():
            raise RuntimeError("docker is gone")
        newest.backend.kill = _boom
        guard.act(guard.read())
        assert newest.backend.stopped is True


class TestItSurvivesABlockedEventLoop:
    def test_the_guard_runs_in_its_own_thread(self, tmp_path):
        # The whole point. The launch path blocks the event loop for minutes
        # (docker stop, an rsync of the weights, the launcher), and a guard
        # living in that loop would be deaf during exactly that window.
        guard = _guard(tmp_path, available_mb=500, critical_gb=4)

        async def _blocked_loop():
            guard.start()
            # Block the loop the way append_solo_instance used to.
            time.sleep(0.2)

        asyncio.run(_blocked_loop())
        deadline = time.time() + 3
        while time.time() < deadline:
            if guard._app["instances"].instances()[-1].backend.killed:
                break
            time.sleep(0.02)
        guard.stop()
        assert guard._app["instances"].instances()[-1].backend.killed is True
        assert threading.active_count() >= 1


class TestTheReserveFitsTheMachine:
    def test_a_small_machine_does_not_get_a_128gb_reserve(self, tmp_path):
        # Applied unchanged to a laptop or an 8 GB CI runner, the Spark
        # preset would refuse every launch and read as a broken product.
        guard = MemoryGuard({}, warn_gb=8, critical_gb=4,
                            meminfo=_meminfo(tmp_path, available_mb=3500,
                                             total_mb=8000))
        reading = guard.read()
        assert reading.warn_mb == pytest.approx(8000 * MAX_RESERVE_SHARE)
        assert reading.blocking is False

    def test_a_spark_keeps_the_full_reserve(self, tmp_path):
        guard = MemoryGuard({}, warn_gb=8, critical_gb=4,
                            meminfo=_meminfo(tmp_path, available_mb=60000,
                                             total_mb=128000))
        # 15% of 128 GB is 19 GB, well above the 8 GB default: unchanged.
        assert guard.read().warn_mb == 8 * 1024

    def test_the_two_lines_keep_their_distance_when_squeezed(self, tmp_path):
        guard = MemoryGuard({}, warn_gb=8, critical_gb=4,
                            meminfo=_meminfo(tmp_path, available_mb=100,
                                             total_mb=4000))
        reading = guard.read()
        assert reading.critical_mb < reading.warn_mb


class _Req:
    def __init__(self, app, body=None):
        self.app = app
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _Config:
    node_id = "n"
    host_memory_warn_gb = 8.0
    host_memory_critical_gb = 4.0
    host_memory_guard = True
    saved = False

    def save(self):
        type(self).saved = True


class TestTheSettings:
    def _app(self, tmp_path):
        guard = _guard(tmp_path, available_mb=60000)
        return {"memory_guard": guard, "config": _Config()}

    def test_it_reports_the_reserve_and_the_reading(self, tmp_path):
        from ainode.safety.api_routes import handle_get

        body = json.loads(asyncio.run(handle_get(_Req(self._app(tmp_path)))).body)
        assert body["enabled"] is True
        assert body["warn_gb"] == 8.0
        assert body["available_mb"] == 60000
        assert "dgx-spark" in body["presets"]

    def test_a_preset_can_be_applied(self, tmp_path):
        from ainode.safety.api_routes import handle_put

        app = self._app(tmp_path)
        body = json.loads(asyncio.run(
            handle_put(_Req(app, {"preset": "generic"}))).body)
        assert body["warn_gb"] == PRESETS["generic"]["warn_gb"]

    def test_the_reserve_is_persisted(self, tmp_path):
        # Or it resets on the next restart — which is the moment right after a
        # node was rebooted for running out.
        from ainode.safety.api_routes import handle_put

        app = self._app(tmp_path)
        _Config.saved = False
        asyncio.run(handle_put(_Req(app, {"warn_gb": 12, "critical_gb": 6})))
        assert app["config"].host_memory_warn_gb == 12
        assert _Config.saved is True

    def test_a_critical_line_above_the_warning_line_is_corrected(self, tmp_path):
        # Otherwise engines would be killed without a load ever being refused,
        # which is the wrong order of defence.
        guard = _guard(tmp_path, available_mb=60000)
        guard.configure(warn_gb=2, critical_gb=6)
        assert guard.warn_mb >= guard.critical_mb

    def test_an_unknown_preset_is_a_400(self, tmp_path):
        from ainode.safety.api_routes import handle_put

        resp = asyncio.run(handle_put(_Req(self._app(tmp_path),
                                           {"preset": "laptop"})))
        assert resp.status == 400

    def test_the_routes_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/safety/memory" in paths


class TestTheMemberNodeCanDefendItself:
    """The half of a distributed launch that had no defence at all.

    A member's engine container is started over SSH by the head's launcher.
    AINode on that node never creates an instance record, so the guard there
    found nothing to stop and logged that it had found nothing — while the
    node it was watching filled up and died. Reported from the cluster:

        wenn ich z.b. minimax3 auf node 1 und node2 starte läuft der speicher
        sofort auf 122gb voll und die sparks schmieren komplett ab
    """

    def test_the_names_match_the_backends(self):
        from ainode.engine.backends.diffusers import DiffusersBackend
        from ainode.engine.backends.eugr import EugrBackend
        from ainode.safety.memory_guard import ENGINE_CONTAINERS

        assert EugrBackend.CONTAINER_BASENAME in ENGINE_CONTAINERS
        assert DiffusersBackend.CONTAINER_BASENAME in ENGINE_CONTAINERS

    def _guard_with_docker(self, tmp_path, monkeypatch, running):
        calls = []

        class _Done:
            def __init__(self, stdout=""):
                self.stdout = stdout
                self.returncode = 0

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[:2] == ["docker", "ps"]:
                return _Done("\n".join(running))
            return _Done()

        monkeypatch.setattr("subprocess.run", fake_run)
        guard = MemoryGuard({"instances": None},
                            meminfo=_meminfo(tmp_path, available_mb=900),
                            critical_gb=4, warn_gb=8, poll_seconds=0.01)
        return guard, calls

    def test_it_kills_the_engine_container(self, tmp_path, monkeypatch):
        guard, calls = self._guard_with_docker(
            tmp_path, monkeypatch, ["vllm_node", "ainode", "some-database"])
        assert guard.act(guard.read()) == "vllm_node"
        assert ["docker", "kill", "vllm_node"] in calls

    def test_it_does_not_kill_ainode_itself(self, tmp_path, monkeypatch):
        # That would stop the guard along with everything else, and systemd
        # would bring the node back up into the same launch.
        guard, calls = self._guard_with_docker(
            tmp_path, monkeypatch, ["ainode", "ainode-db"])
        assert guard.act(guard.read()) is None
        assert not any(c[:2] == ["docker", "kill"] for c in calls)

    def test_a_stacked_instances_container_counts(self, tmp_path, monkeypatch):
        guard, calls = self._guard_with_docker(
            tmp_path, monkeypatch, ["vllm_node-8001", "ainode_image-8002"])
        stopped = guard.act(guard.read())
        assert "vllm_node-8001" in stopped and "ainode_image-8002" in stopped

    def test_the_stop_is_recorded_like_any_other(self, tmp_path, monkeypatch):
        guard, _ = self._guard_with_docker(tmp_path, monkeypatch, ["vllm_node"])
        guard.act(guard.read())
        assert guard.stops == 1
        assert "member of a launch the head started" in guard.read().actions[-1]["reason"]

    def test_an_instance_record_still_wins(self, tmp_path, monkeypatch):
        # On the head there IS a record, and stopping the right instance beats
        # killing every engine container on the node.
        calls = []
        monkeypatch.setattr("subprocess.run",
                            lambda cmd, **kw: calls.append(cmd))
        guard = _guard(tmp_path, available_mb=900, critical_gb=4)
        guard.act(guard.read())
        assert guard._app["instances"].instances()[-1].backend.killed is True
        assert calls == []
