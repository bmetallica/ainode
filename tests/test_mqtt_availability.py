"""Two things whose whole value is in their timing.

"Is that node alive" could only be answered by noticing that messages had
stopped — which needs a timeout in every dashboard, and which retained
messages defeat entirely: the last reading of a node that died an hour ago
looks exactly as fresh as a live one.

And the host memory guard could kill a running engine while leaving no trace
anywhere but a log file, up to thirty seconds before the next telemetry
interval would have hinted at it.
"""

from __future__ import annotations

import json

from ainode.core.config import NodeConfig
from ainode.telemetry.mqtt import (
    availability_payload,
    publish_event,
    set_last_will,
)


class _Client:
    def __init__(self):
        self.will = None
        self.published = []

    def will_set(self, topic, payload, qos=0, retain=False):
        self.will = (topic, payload, qos, retain)

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


def _config(**kw):
    return NodeConfig(node_id="spark-1", node_name="SPARK1",
                      mqtt_topic_prefix="ainode", **kw)


class TestTheBrokerAnnouncesTheDeath:
    def test_the_will_is_set_before_connecting(self):
        # It has to be registered on the CONNECT packet; setting it later is
        # too late for the connection it describes.
        client = _Client()
        set_last_will(client, _config())
        topic, payload, _, retain = client.will
        assert topic == "ainode/spark-1/status"
        assert json.loads(payload)["status"] == "offline"
        assert retain is True

    def test_the_will_names_the_node(self):
        # A dashboard receiving it has only this message to go on.
        client = _Client()
        set_last_will(client, _config())
        body = json.loads(client.will[1])
        assert body["node_id"] == "spark-1"
        assert body["node_name"] == "SPARK1"

    def test_coming_up_is_announced_and_retained(self):
        # Retained, so a dashboard that subscribes later still learns the node
        # is up instead of waiting for the next interval.
        payload = json.loads(availability_payload(_config(), online=True))
        assert payload["status"] == "online"
        assert "timestamp" in payload

    def test_the_publisher_wires_both(self):
        import inspect

        from ainode.telemetry.mqtt import MqttPublisher

        connect = inspect.getsource(MqttPublisher._connect)
        assert "set_last_will(client, config)" in connect
        assert "availability_payload(config, online=True)" in connect
        assert "retain=True" in connect

    def test_a_deliberate_shutdown_says_so_itself(self):
        # Both are "offline"; only one of them is a reason to get out of bed,
        # and leaving a planned restart to the last will makes it look like a
        # failure.
        import inspect

        from ainode.telemetry.mqtt import MqttPublisher

        assert "availability_payload(config, online=False)" in \
            inspect.getsource(MqttPublisher._disconnect)


class _Guard:
    def __init__(self, reading, enabled=True, stops=0):
        self._reading = reading
        self.enabled = enabled
        self.stops = stops

    def read(self):
        return self._reading


def _reading(**kw):
    from ainode.safety.memory_guard import MemoryReading

    defaults = dict(available_mb=60000, total_mb=128000, warn_mb=8192,
                    critical_mb=4096)
    defaults.update(kw)
    return MemoryReading(**defaults)


class _Sampler:
    def sample(self):
        return {}


class TestTheGuardIsVisible:
    def _payload(self, guard):
        from ainode.telemetry.payloads import build_payloads

        return build_payloads({"config": _config(), "memory_guard": guard},
                              _Sampler()).get("safety")

    def test_it_reports_what_the_guard_sees(self):
        payload = self._payload(_Guard(_reading()))
        assert payload["host_available_mb"] == 60000
        assert payload["warn_mb"] == 8192
        assert payload["critical_mb"] == 4096
        assert payload["blocking_launches"] is False
        assert payload["memory_guard_enabled"] is True

    def test_it_reports_the_enforced_lines_not_the_configured_ones(self):
        # The reserve is capped against the size of the machine. An alert
        # built on a number the guard is not actually using fires at the
        # wrong moment.
        payload = self._payload(_Guard(_reading(warn_mb=1228, critical_mb=614)))
        assert payload["warn_mb"] == 1228

    def test_blocking_and_critical_are_separate(self):
        payload = self._payload(
            _Guard(_reading(available_mb=3000, warn_mb=8192, critical_mb=4096)))
        assert payload["blocking_launches"] is False   # set by the reading
        assert payload["below_critical"] is False

    def test_a_node_without_a_guard_publishes_no_safety_topic(self):
        from ainode.telemetry.payloads import build_payloads

        assert "safety" not in build_payloads({"config": _config()}, _Sampler())

    def test_the_last_stop_is_carried(self):
        reading = _reading()
        reading.actions = [{"at": 1.0, "model": "org/m", "reason": "…",
                            "available_mb": 3200}]
        payload = self._payload(_Guard(reading, stops=2))
        assert payload["stops_total"] == 2
        assert payload["last_stop"]["model"] == "org/m"

    def test_a_guard_that_cannot_read_memory_says_so(self):
        payload = self._payload(_Guard(_reading(readable=False)))
        assert payload["host_memory_readable"] is False


class TestTheStopIsAnnouncedImmediately:
    def test_the_guard_calls_back_when_it_acts(self):
        from ainode.safety.memory_guard import MemoryGuard

        seen = []

        class _Backend:
            def kill(self):
                pass

        class _Instance:
            record = type("R", (), {"model": "org/m", "load_error": "",
                                    "load_phase": "", "status": ""})()
            backend = _Backend()

        class _Manager:
            def instances(self):
                return [_Instance()]

        guard = MemoryGuard({"instances": _Manager()},
                            on_action=lambda action: seen.append(action))
        guard.act(_reading(available_mb=500))
        assert seen and seen[0]["model"] == "org/m"
        assert guard.stops == 1

    def test_a_callback_that_raises_does_not_stop_the_guard(self):
        # The guard is the last line of defence; a broken broker must not be
        # able to disarm it.
        from ainode.safety.memory_guard import MemoryGuard

        def _boom(action):
            raise RuntimeError("no broker")

        class _Instance:
            record = type("R", (), {"model": "m", "load_error": "",
                                    "load_phase": "", "status": ""})()
            backend = type("B", (), {"kill": lambda self: None})()

        class _Manager:
            def instances(self):
                return [_Instance()]

        guard = MemoryGuard({"instances": _Manager()}, on_action=_boom)
        assert guard.act(_reading(available_mb=500)) == "m"

    def test_publishing_an_event_needs_no_broker_to_be_safe(self):
        # An event that cannot be sent must not become a second fault.
        assert publish_event({"config": _config()}, "safety") is False

    def test_it_publishes_on_the_live_client(self):
        client = _Client()
        app = {"config": _config(),
               "mqtt_publisher": type("P", (), {"_client": client})(),
               "memory_guard": _Guard(_reading()),
               "_telemetry_sampler": _Sampler()}
        assert publish_event(app, "safety") is True
        topic, payload, _, _ = client.published[0]
        assert topic == "ainode/spark-1/safety"
        assert json.loads(payload)["host_available_mb"] == 60000

    def test_the_server_wires_the_guard_to_the_broker(self):
        import inspect

        from ainode.api import server

        source = inspect.getsource(server._on_startup)
        assert "guard.on_action = _announce_stop" in source
        assert 'publish_event(app, "safety")' in source
