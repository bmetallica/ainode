"""Telemetry: what gets measured, what gets published, and what never breaks.

A monitoring feature has two ways to fail badly. It can publish numbers that
are wrong — an average speed divided by uptime instead of by generation time,
a network percentage of a link whose speed is unknown. Or it can take the node
down with it, which is worse than having no monitoring at all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest

from ainode.core.config import NodeConfig
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.system import SystemSampler, interface_is_interesting
from ainode.telemetry import mqtt as mqtt_mod
from ainode.telemetry.api_routes import (
    handle_get_settings,
    handle_put_settings,
)
from ainode.telemetry.payloads import build_payloads, node_identity

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class _Req:
    def __init__(self, app, body=None):
        self.app = app
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _call(handler, app, body=None):
    return asyncio.run(handler(_Req(app, body)))


def _json(resp):
    return json.loads(resp.body)


class TestSystemMetrics:
    def test_the_first_sample_reports_no_cpu_percent(self):
        # psutil's first interval=None call compares against nothing and
        # returns 0.0 — an idle CPU and an unmeasured one are different things.
        first = SystemSampler().sample()
        assert "percent" not in first.get("cpu", {})

    def test_a_later_sample_reports_one(self):
        sampler = SystemSampler()
        sampler.sample()
        assert "percent" in sampler.sample().get("cpu", {})

    def test_memory_and_disk_are_present(self):
        sample = SystemSampler().sample()
        assert sample["memory"]["total_mb"] > 0
        assert sample["disk"]["/"]["total_gb"] > 0

    def test_loopback_and_container_plumbing_are_skipped(self):
        for name in ("lo", "docker0", "veth1234", "br-abc"):
            assert interface_is_interesting(name) is False
        for name in ("enP7s7", "wlP9s9", "eth0", "enP2p1s0f0np0"):
            assert interface_is_interesting(name) is True

    def test_network_rates_appear_only_after_a_baseline(self):
        sampler = SystemSampler()
        first = sampler.sample()
        for entry in first.get("network", {}).values():
            assert "tx_mbit_s" not in entry
        for entry in sampler.sample().get("network", {}).values():
            assert "tx_mbit_s" in entry

    def test_a_percentage_needs_a_known_link_speed(self):
        # An interface reporting speed 0 (RoCE often does) gets a rate but no
        # percentage: a percentage of an unknown maximum is a made-up number.
        sampler = SystemSampler()
        counters = {"test0": mock.Mock(bytes_sent=1000, bytes_recv=2000)}
        stats = {"test0": mock.Mock(isup=True, speed=0)}
        with mock.patch("psutil.net_io_counters", return_value=counters), \
             mock.patch("psutil.net_if_stats", return_value=stats):
            sampler._network()
            counters["test0"] = mock.Mock(bytes_sent=2000, bytes_recv=4000)
            out = sampler._network()
        assert "tx_mbit_s" in out["test0"]
        assert "tx_percent" not in out["test0"]

    def test_a_known_link_speed_gives_a_percentage(self):
        sampler = SystemSampler()
        counters = {"eth9": mock.Mock(bytes_sent=0, bytes_recv=0)}
        stats = {"eth9": mock.Mock(isup=True, speed=10000)}
        with mock.patch("psutil.net_io_counters", return_value=counters), \
             mock.patch("psutil.net_if_stats", return_value=stats):
            sampler._network()
            sampler._last_net_at -= 1.0            # pretend a second passed
            counters["eth9"] = mock.Mock(bytes_sent=125_000_000, bytes_recv=0)
            out = sampler._network()
        # 125 MB/s = 1000 Mbit/s = 10% of a 10G link.
        assert out["eth9"]["tx_percent"] == pytest.approx(10.0, abs=0.5)

    def test_a_down_interface_is_not_reported(self):
        sampler = SystemSampler()
        with mock.patch("psutil.net_io_counters",
                        return_value={"eth9": mock.Mock(bytes_sent=0, bytes_recv=0)}), \
             mock.patch("psutil.net_if_stats",
                        return_value={"eth9": mock.Mock(isup=False, speed=10000)}):
            assert sampler._network() == {}

    def test_a_broken_metric_does_not_break_the_sample(self):
        sampler = SystemSampler()
        with mock.patch.object(SystemSampler, "_memory", side_effect=OSError("nope")):
            sample = sampler.sample()
        assert "memory" not in sample and "disk" in sample


class TestPerModelStats:
    def test_speed_is_tokens_over_generation_time_not_uptime(self):
        # A model that served one request an hour ago is idle, not slow.
        collector = MetricsCollector()
        collector.record_request("a/b", latency_ms=2000, tokens_generated=40)
        stats = collector.model_stats()["a/b"]
        assert stats["avg_tokens_per_second"] == 20.0

    def test_models_are_kept_apart(self):
        collector = MetricsCollector()
        collector.record_request("chat", latency_ms=1000, tokens_generated=10)
        collector.record_request("code", latency_ms=1000, tokens_generated=50)
        stats = collector.model_stats()
        assert stats["chat"]["avg_tokens_per_second"] == 10.0
        assert stats["code"]["avg_tokens_per_second"] == 50.0

    def test_a_model_with_no_token_counts_reports_no_speed(self):
        collector = MetricsCollector()
        collector.record_request("a/b", latency_ms=500)
        stats = collector.model_stats()["a/b"]
        assert stats["requests"] == 1
        assert "avg_tokens_per_second" not in stats

    def test_errors_are_counted_per_model(self):
        collector = MetricsCollector()
        collector.record_request("a/b", latency_ms=10, error=True)
        assert collector.model_stats()["a/b"]["errors"] == 1


class _Instance:
    def __init__(self, model, port, peers=()):
        self.record = mock.Mock(model=model, api_port=port, peer_ips=list(peers),
                                status="running")
        self.backend = mock.Mock(config=NodeConfig(model=model,
                                                   gpu_memory_utilization=0.4),
                                 load_phase="ready")


class _Manager:
    def __init__(self, instances):
        self._i = list(instances)

    def instances(self):
        return self._i


@pytest.fixture()
def app():
    collector = MetricsCollector()
    collector.record_request("a/b", latency_ms=1000, tokens_generated=30)
    return {
        "config": NodeConfig(node_id="head", node_name="spark-1"),
        "metrics_collector": collector,
        "instances": _Manager([_Instance("a/b", 8000)]),
        "embedding_manager": None,
        "cluster_state": None,
        "secrets_manager": None,
    }


class TestPayloads:
    def test_every_payload_identifies_its_node(self, app):
        payloads = build_payloads(app, SystemSampler())
        for name, payload in payloads.items():
            assert payload["node_id"] == "head", name
            assert payload["timestamp"] > 0, name

    def test_the_groups_are_system_and_models(self, app):
        payloads = build_payloads(app, SystemSampler())
        assert "system" in payloads and "models" in payloads

    def test_loaded_models_and_their_speed_are_published(self, app):
        models = build_payloads(app, SystemSampler())["models"]
        assert models["loaded"][0]["model"] == "a/b"
        assert models["per_model"]["a/b"]["avg_tokens_per_second"] == 30.0

    def test_a_member_does_not_publish_the_cluster_view(self, app):
        # Two publishers on one topic means the last writer wins, and half the
        # time that is the node with the partial picture.
        app["config"].distributed_mode = "member"
        app["cluster_state"] = mock.Mock(members=lambda: [])
        assert "cluster" not in build_payloads(app, SystemSampler())

    def test_the_head_does(self, app):
        node = mock.Mock(node_id="n2", node_name="spark-2", status="online",
                         model="a/b", gpu_memory_gb=128.0, gpu_memory_used_pct=40)
        app["cluster_state"] = mock.Mock(members=lambda: [node])
        cluster = build_payloads(app, SystemSampler())["cluster"]
        assert cluster["nodes_total"] == 1 and cluster["nodes_online"] == 1
        assert cluster["vram_total_gb"] == 128.0

    def test_a_failing_collector_does_not_lose_the_rest(self, app):
        app["metrics_collector"] = mock.Mock(
            get_gpu_metrics=mock.Mock(side_effect=RuntimeError("nvml gone")),
            get_snapshot=mock.Mock(side_effect=RuntimeError("nvml gone")),
            model_stats=mock.Mock(return_value={}))
        payloads = build_payloads(app, SystemSampler())
        assert "system" in payloads and "gpu" not in payloads

    def test_identity_carries_the_version(self, app):
        assert node_identity(app["config"])["version"]


class TestTopics:
    def test_node_topics_are_namespaced_by_node(self):
        config = NodeConfig(node_id="head", mqtt_topic_prefix="ainode")
        assert mqtt_mod._topic(config, "system") == "ainode/head/system"

    def test_the_cluster_topic_is_not(self):
        # The fleet view is not a property of whichever node publishes it.
        config = NodeConfig(node_id="head", mqtt_topic_prefix="ainode")
        assert mqtt_mod._topic(config, "cluster") == "ainode/cluster"

    def test_a_prefix_with_slashes_is_tolerated(self):
        config = NodeConfig(node_id="head", mqtt_topic_prefix="/lab/ai/")
        assert mqtt_mod._topic(config, "gpu") == "lab/ai/head/gpu"


class TestSettingsApi:
    @pytest.fixture()
    def app(self, tmp_path):
        config = NodeConfig(node_id="head")
        config.save = lambda *a, **k: None
        return {"config": config, "secrets_manager": _Secrets(), "mqtt_publisher": None}

    def test_defaults_are_off(self, app):
        data = _json(_call(handle_get_settings, app))
        assert data["settings"]["mqtt_enabled"] is False
        assert data["password_set"] is False

    def test_saving_settings(self, app):
        resp = _call(handle_put_settings, app, {
            "mqtt_enabled": True, "mqtt_host": "192.168.1.50",
            "mqtt_port": 8883, "mqtt_username": "ainode",
            "mqtt_password": "secret", "mqtt_interval": 15,
        })
        assert resp.status == 200
        settings = _json(resp)["settings"]
        assert settings["mqtt_host"] == "192.168.1.50"
        assert settings["mqtt_port"] == 8883
        assert settings["mqtt_interval"] == 15
        assert app["secrets_manager"].get("mqtt_password") == "secret"

    def test_the_password_is_never_returned(self, app):
        _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_password": "secret"})
        body = _call(handle_get_settings, app).body.decode()
        assert "secret" not in body
        assert _json(_call(handle_get_settings, app))["password_set"] is True

    def test_an_empty_password_keeps_the_stored_one(self, app):
        # A form that round-trips its fields must not wipe a working credential.
        _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_password": "secret"})
        _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_password": ""})
        assert app["secrets_manager"].get("mqtt_password") == "secret"

    def test_clearing_is_explicit(self, app):
        _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_password": "secret"})
        _call(handle_put_settings, app,
              {"mqtt_host": "h", "mqtt_password": "", "clear_password": True})
        assert app["secrets_manager"].get("mqtt_password") is None

    def test_enabling_without_a_broker_is_refused(self, app):
        resp = _call(handle_put_settings, app, {"mqtt_enabled": True, "mqtt_host": ""})
        assert resp.status == 400

    def test_one_second_is_allowed(self, app):
        """Asked for: an operator watching a launch or a transfer wants to see
        it move. The old floor of 5 s was a guess about sampling cost, not a
        measurement of it."""
        resp = _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_interval": 1})
        assert _json(resp)["settings"]["mqtt_interval"] == 1

    def test_below_one_is_still_clamped(self, app):
        """0 would spin the publish loop without yielding."""
        resp = _call(handle_put_settings, app, {"mqtt_host": "h", "mqtt_interval": 0})
        assert _json(resp)["settings"]["mqtt_interval"] == mqtt_mod.MIN_INTERVAL

    def test_the_upper_bound_still_holds(self, app):
        resp = _call(handle_put_settings, app, {"mqtt_host": "h",
                                                "mqtt_interval": 99999})
        assert _json(resp)["settings"]["mqtt_interval"] == mqtt_mod.MAX_INTERVAL

    def test_the_publisher_honours_one_second(self, app):
        """The clamp in the loop is separate from the one in the route, and a
        floor left behind there would silently override the setting."""
        config = type("C", (), {"mqtt_interval": 1})()
        publisher = mqtt_mod.MqttPublisher.__new__(mqtt_mod.MqttPublisher)
        assert publisher._interval(config) == 1

    def test_junk_is_not_a_500(self, app):
        assert _call(handle_put_settings, app, None).status == 400


class _Secrets:
    def __init__(self):
        self._d = {}

    def has(self, key):
        return key in self._d

    def get(self, key):
        return self._d.get(key)

    def set(self, key, value):
        self._d[key] = value

    def delete(self, key):
        return self._d.pop(key, None) is not None


class TestPublisherLifecycle:
    def test_disabled_means_not_started(self):
        app = {"config": NodeConfig(mqtt_enabled=False)}
        assert mqtt_mod.MqttPublisher(app).start() is False

    def test_enabled_without_a_host_does_not_start(self):
        app = {"config": NodeConfig(mqtt_enabled=True, mqtt_host="")}
        assert mqtt_mod.MqttPublisher(app).start() is False

    def test_status_is_readable_before_starting(self):
        app = {"config": NodeConfig()}
        status = mqtt_mod.MqttPublisher(app).status()
        assert status["running"] is False and status["published"] == 0

    def test_a_missing_broker_is_an_error_not_a_crash(self):
        config = NodeConfig(mqtt_host="")
        assert mqtt_mod.test_connection(config, "")["ok"] is False


class TestUi:
    def test_the_section_exists(self):
        assert 'data-section="monitoring"' in INDEX
        assert "case 'monitoring':" in APP_JS
        assert "renderConfigMonitoring" in APP_JS

    @pytest.mark.parametrize("element", [
        "cfg-mqtt-enabled", "cfg-f-mqtt_host", "cfg-f-mqtt_port",
        "cfg-f-mqtt_username", "cfg-f-mqtt_password", "cfg-f-mqtt_topic_prefix",
        "cfg-f-mqtt_interval", "cfg-mqtt-tls", "cfg-mqtt-retain",
    ])
    def test_every_setting_has_a_field(self, element):
        assert element in APP_JS

    def test_test_and_publish_buttons_exist(self):
        assert "/api/telemetry/mqtt/test" in APP_JS
        assert "/api/telemetry/mqtt/publish" in APP_JS

    def test_the_payload_can_be_previewed(self):
        # Building a dashboard needs the message shape before the publishing
        # works, not after.
        assert "/api/telemetry/preview" in APP_JS


class TestClusterWideSettings:
    """Each node publishes its own vital signs; no other node can see them.

    Configuring the broker on the head alone produced telemetry from one of
    three machines — no CPU, memory, disk or network from the peers, because
    the discovery announcement carries GPU and status and nothing else.
    """

    def test_the_route_exists(self):
        from ainode.api.server import create_app

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/telemetry/mqtt/apply-to-cluster" in paths

    def test_it_sends_the_password_too(self):
        # Settings without the password would leave every peer unable to
        # connect, which looks like the feature not working.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "telemetry" /
               "api_routes.py").read_text()
        block = src[src.index("async def handle_apply_to_cluster"):]
        block = block[:block.index("async def handle_preview")]
        assert 'payload["mqtt_password"] = password' in block
        # and it says so — the docstring wraps, so compare on words
        assert "clear" in " ".join(block.split())

    def test_it_reports_per_node(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "telemetry" /
               "api_routes.py").read_text()
        assert '"results": results' in src

    def test_the_ui_warns_before_copying_the_password(self):
        assert "cfg-mqtt-cluster" in APP_JS
        assert "in the clear" in APP_JS


class TestTheClusterPayloadIsTruthful:
    def test_gpu_use_is_computed_from_megabytes(self):
        # gpu_memory_used_pct does not exist on a cluster node; reading it
        # returned 0.0 for every node on a cluster with two models loaded.
        from unittest import mock

        from ainode.telemetry.payloads import _cluster

        node = mock.Mock(node_id="n2", node_name="spark-2", status="online",
                         model="a/b", gpu_memory_gb=121.7,
                         gpu_memory_used_mb=78000, gpu_memory_total_mb=124610,
                         gpu_utilization=42.0, instances=[])
        app = {"config": NodeConfig(node_id="head"),
               "cluster_state": mock.Mock(members=lambda: [node])}
        entry = _cluster(app)["nodes"][0]
        assert entry["gpu_memory_used_percent"] == pytest.approx(62.6, abs=0.2)
        assert entry["gpu_memory_used_mb"] == 78000
        assert entry["gpu_utilization_percent"] == 42.0

    def test_a_node_without_readings_omits_them(self):
        from unittest import mock

        from ainode.telemetry.payloads import _cluster

        node = mock.Mock(node_id="n2", node_name="spark-2", status="online",
                         model="", gpu_memory_gb=121.7, gpu_memory_used_mb=0,
                         gpu_memory_total_mb=0, gpu_utilization=None,
                         instances=[])
        app = {"config": NodeConfig(node_id="head"),
               "cluster_state": mock.Mock(members=lambda: [node])}
        entry = _cluster(app)["nodes"][0]
        assert "gpu_memory_used_percent" not in entry

    def test_the_instances_come_along(self):
        from unittest import mock

        from ainode.telemetry.payloads import _cluster

        node = mock.Mock(node_id="n2", node_name="spark-2", status="online",
                         model="a/b", gpu_memory_gb=121.7, gpu_memory_used_mb=1,
                         gpu_memory_total_mb=2, gpu_utilization=0,
                         instances=[{"model": "a/b", "api_port": 8000,
                                     "status": "serving"}])
        app = {"config": NodeConfig(node_id="head"),
               "cluster_state": mock.Mock(members=lambda: [node])}
        assert _cluster(app)["nodes"][0]["instances"][0]["model"] == "a/b"


class TestTheSamplerIsShared:
    """A rate needs two readings, and a fresh sampler has one.

    "Publish now" built its own SystemSampler, so its first — and only —
    sample had no baseline and the payload carried no tx_mbit_s, no
    rx_mbit_s, no cpu percent. Absent fields read as a broken metric rather
    than as a missing baseline, and the operator has no way to tell which.

    The publish loop already keeps a sampler alive across ticks. Sharing it
    is the whole fix.
    """

    def test_the_loop_publishes_its_sampler(self):
        import inspect

        from ainode.telemetry.mqtt import MqttPublisher

        source = inspect.getsource(MqttPublisher)
        assert '_app["_telemetry_sampler"] = sampler' in source

    def test_publish_now_prefers_it(self):
        import inspect

        from ainode.telemetry import api_routes

        source = inspect.getsource(api_routes.handle_publish_now)
        assert 'request.app.get("_telemetry_sampler")' in source

    def test_a_second_sample_carries_rates(self):
        """The property that was missing: sample twice on one sampler and the
        rate fields appear."""
        from ainode.metrics.system import SystemSampler

        sampler = SystemSampler()
        first = sampler.sample()["network"]
        second = sampler.sample()["network"]
        assert first, "no interfaces to measure"
        name = next(iter(first))
        assert "tx_mbit_s" not in first[name]
        assert "tx_mbit_s" in second[name]
