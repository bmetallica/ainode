"""The smaller finds: latency tails, transfers, version drift, launch events.

Each is a field that existed somewhere in the process and never reached the
broker, and each answers a question that came up on this cluster for real.
"""

from __future__ import annotations

import json

from ainode.core.config import NodeConfig
from ainode.telemetry.events import LaunchWatcher
from ainode.telemetry.payloads import build_payloads


class _Sampler:
    def sample(self):
        return {}


def _config():
    return NodeConfig(node_id="spark-1", node_name="SPARK1",
                      mqtt_topic_prefix="ainode")


class _Collector:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def get_snapshot(self):
        return self._snapshot

    def get_gpu_metrics(self):
        return {"error": "no gpu"}

    def model_stats(self):
        return {}


class TestLatencyTails:
    def test_the_percentiles_reach_the_broker(self):
        # The collector has computed these all along and nothing carried them
        # out. An average hides the tail, and the tail is what a user notices.
        collector = _Collector({"requests": {"total": 10, "errors": 1,
                                             "latency_ms": {"p50": 120.0,
                                                            "p95": 900.0,
                                                            "p99": 2300.0}},
                                "uptime_seconds": 5.0})
        payload = build_payloads({"config": _config(),
                                  "metrics_collector": collector}, _Sampler())
        assert payload["models"]["latency_ms"]["p95"] == 900.0

    def test_no_measurements_means_no_field(self):
        collector = _Collector({"requests": {"total": 0, "errors": 0},
                                "uptime_seconds": 1.0})
        payload = build_payloads({"config": _config(),
                                  "metrics_collector": collector}, _Sampler())
        assert "latency_ms" not in payload["models"]


class TestTransfers:
    """Hours of work that was visible only to whoever had the browser open."""

    def test_a_running_download_is_reported(self):
        app = {"config": _config(), "download_jobs": {
            "j1": {"hf_repo": "org/big", "status": "downloading",
                   "progress": 41.5, "downloaded_bytes": 100,
                   "total_bytes": 240}}}
        transfers = build_payloads(app, _Sampler())["transfers"]
        assert transfers["downloads"][0]["model"] == "org/big"
        assert transfers["downloads"][0]["percent"] == 41.5

    def test_a_finished_one_is_not(self):
        app = {"config": _config(), "download_jobs": {
            "j1": {"hf_repo": "org/big", "status": "completed",
                   "progress": 100}}}
        assert "transfers" not in build_payloads(app, _Sampler())

    def test_a_mirror_run_is_reported(self):
        app = {"config": _config(),
               "mirror_jobs": {"running": True, "current": "org/big",
                               "done": 1, "total": 3}}
        mirror = build_payloads(app, _Sampler())["transfers"]["mirror"]
        assert mirror["model"] == "org/big" and mirror["total"] == 3

    def test_nothing_in_flight_publishes_no_topic(self):
        assert "transfers" not in build_payloads({"config": _config()},
                                                 _Sampler())


class _Node:
    def __init__(self, node_id, version=""):
        self.node_id = node_id
        self.node_name = node_id
        self.status = "online"
        self.gpu_memory_gb = 122.0
        self.gpu_memory_used_mb = 0
        self.gpu_memory_total_mb = 0
        self.model = ""
        self.instances = []
        self.version = version


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


class TestVersionDrift:
    """A fleet that does not agree about its build is a distributed launch
    waiting to fail: the launcher compares engine images and aborts when they
    differ, minutes in."""

    def _cluster_payload(self, versions):
        app = {"config": _config(),
               "cluster_state": _Cluster(
                   [_Node(f"n{i}", v) for i, v in enumerate(versions)])}
        return build_payloads(app, _Sampler())["cluster"]

    def test_agreement_is_stated_plainly(self):
        payload = self._cluster_payload(["0.6.0", "0.6.0", "0.6.0"])
        assert payload["versions_agree"] is True
        assert "versions" not in payload

    def test_disagreement_names_the_versions(self):
        payload = self._cluster_payload(["0.6.0", "0.6.0", "0.5.9"])
        assert payload["versions_agree"] is False
        assert payload["versions"] == ["0.5.9", "0.6.0"]

    def test_a_node_carries_its_own(self):
        payload = self._cluster_payload(["0.6.0"])
        assert payload["nodes"][0]["version"] == "0.6.0"

    def test_an_older_peer_that_sends_none_is_not_a_disagreement(self):
        # It simply does not say, and inventing a mismatch would fire an
        # alert about an upgrade that already happened.
        payload = self._cluster_payload(["0.6.0", ""])
        assert payload["versions_agree"] is True

    def test_the_version_crosses_the_wire(self):
        from ainode.discovery.broadcast import NodeAnnouncement
        from ainode.discovery.cluster import ClusterNode, NodeStatus

        announcement = NodeAnnouncement(
            node_id="n", node_name="n", gpu_name="GB10", gpu_memory_gb=128.0,
            unified_memory=True, model="", status="idle", api_port=8000,
            web_port=3000, version="0.6.0")
        restored = NodeAnnouncement.from_json(announcement.to_json())
        assert restored.version == "0.6.0"
        node = ClusterNode.from_announcement(restored, NodeStatus.ONLINE)
        assert node.version == "0.6.0"


class _Backend:
    def __init__(self, phase, timeline=None, error=""):
        self.load_phase = phase
        self.load_timeline = timeline or []
        self.load_error = error
        self.load_seconds = 0


class _Record:
    def __init__(self, model):
        self.model = model


class _Instance:
    def __init__(self, model, backend):
        self.record = _Record(model)
        self.backend = backend


class _Manager:
    def __init__(self, instances):
        self._instances = instances

    def instances(self):
        return list(self._instances)


class TestLaunchEvents:
    def _app(self, backend):
        return {"instances": _Manager([_Instance("org/m", backend)])}

    def test_reaching_ready_is_an_event(self):
        backend = _Backend("loading_weights")
        watcher = LaunchWatcher(self._app(backend))
        assert watcher.poll() == []
        backend.load_phase = "ready"
        backend.load_timeline = [{"phase": "starting", "seconds": 40.0},
                                 {"phase": "profiling", "seconds": 180.0}]
        event = watcher.poll()[0]
        assert event["outcome"] == "ready"
        assert event["seconds"] == 220.0
        assert event["timeline"][0]["phase"] == "starting"

    def test_failing_is_an_event_with_its_reason(self):
        backend = _Backend("loading_weights")
        watcher = LaunchWatcher(self._app(backend))
        watcher.poll()
        backend.load_phase = "failed"
        backend.load_error = "ValueError: Unsupported weight_bits: 16"
        event = watcher.poll()[0]
        assert event["outcome"] == "failed"
        assert "weight_bits" in event["error"]

    def test_the_reason_is_truncated(self):
        # An event is not a log; the log topic carries the rest.
        backend = _Backend("starting")
        watcher = LaunchWatcher(self._app(backend))
        watcher.poll()
        backend.load_phase = "failed"
        backend.load_error = "x" * 5000
        assert len(watcher.poll()[0]["error"]) == 1000

    def test_it_fires_once(self):
        backend = _Backend("starting")
        watcher = LaunchWatcher(self._app(backend))
        watcher.poll()
        backend.load_phase = "ready"
        assert len(watcher.poll()) == 1
        assert watcher.poll() == []

    def test_an_instance_already_finished_when_first_seen_is_not_an_event(self):
        # A restart, or telemetry switched on later. It did not happen now,
        # and dating it now would be a lie.
        watcher = LaunchWatcher(self._app(_Backend("ready")))
        assert watcher.poll() == []

    def test_an_unloaded_instance_is_forgotten(self):
        # Or reloading the same model would never fire again.
        backend = _Backend("ready")
        app = self._app(backend)
        watcher = LaunchWatcher(app)
        watcher.poll()
        app["instances"] = _Manager([])
        watcher.poll()
        app["instances"] = _Manager([_Instance("org/m", _Backend("starting"))])
        watcher.poll()
        app["instances"].instances()[0].backend.load_phase = "ready"
        assert len(watcher.poll()) == 1

    def test_a_broken_manager_is_silence_not_a_crash(self):
        class _Boom:
            def instances(self):
                raise RuntimeError("no")

        assert LaunchWatcher({"instances": _Boom()}).poll() == []


class _Client:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


class TestTheEventIsPublished:
    def test_it_goes_to_its_own_topic_and_is_not_retained(self):
        # A retained event would replay every time a dashboard reconnects,
        # reporting a launch that happened last week as news.
        from ainode.telemetry.mqtt import publish_launch

        client = _Client()
        app = {"config": _config(),
               "mqtt_publisher": type("P", (), {"_client": client})()}
        assert publish_launch(app, "org/m", "ready", seconds=220.0) is True
        topic, payload, _, retain = client.published[0]
        assert topic == "ainode/spark-1/events/launch"
        assert retain is False
        assert json.loads(payload)["seconds"] == 220.0

    def test_no_broker_is_not_a_failure(self):
        from ainode.telemetry.mqtt import publish_launch

        assert publish_launch({"config": _config()}, "org/m", "ready") is False

    def test_the_loop_polls_the_watcher(self):
        import inspect

        from ainode.telemetry.mqtt import MqttPublisher

        source = inspect.getsource(MqttPublisher._run)
        assert "launches.poll()" in source
        assert "publish_launch(" in source
