"""Publish node and cluster telemetry to an MQTT broker.

Why MQTT rather than only the Prometheus endpoint that already exists: a
scrape needs the scraper to reach every node, which on a switchless mesh means
the monitoring host has to be on the coordination network and know each node's
address. A publish only needs each node to reach one broker — and most homes
and labs already have one, with Home Assistant, Node-RED or Grafana behind it.

Design points that are not obvious:

* **The loop owns the client.** paho reconnects on its own once
  ``loop_start()`` is running, so a broker that is down at boot or restarts at
  night needs no logic here beyond letting it.
* **Failures are logged once, not every interval.** A broker that is
  unreachable for a day would otherwise write 2880 identical lines.
* **Telemetry never breaks serving.** Every publish is wrapped; an exception
  here is a log line, not a failed request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["MqttPublisher", "MqttUnavailable", "publish_once", "test_connection"]

#: One second, because an operator watching a launch or a transfer wants to
#: see it move, and 5 s was a guess about cost rather than a measurement of
#: it. The sampler reads /proc and sysfs; at one-second intervals that is
#: noise next to an idle node, let alone a busy one.
#:
#: It is still a floor rather than no limit: 0 or a negative value would spin
#: the publish loop without yielding, and a broker does not thank anyone for
#: that. Rates measured over a one-second window are correspondingly
#: coarser — a counter that ticks a few times a second reads as a step
#: function — which is a property of the window, not a fault to fix.
MIN_INTERVAL = 1
MAX_INTERVAL = 3600
CONNECT_TIMEOUT = 10


class MqttUnavailable(RuntimeError):
    """paho-mqtt is not installed, or the broker cannot be reached."""


def _client(config, password: str, *, client_id: str = ""):
    """A configured, not-yet-connected paho client."""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:  # pragma: no cover - present in the image
        raise MqttUnavailable(
            "paho-mqtt is not installed in this image. It ships with AINode; "
            "a build without it can add it with `pip install paho-mqtt`."
        ) from exc

    node_id = getattr(config, "node_id", "") or socket.gethostname()
    # Client ids must be unique per broker connection: two nodes sharing one
    # would disconnect each other in a loop that looks like a flapping network.
    name = client_id or f"ainode-{node_id}"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=name)
    except (AttributeError, TypeError):
        # paho 1.x — no CallbackAPIVersion. Same call, older signature.
        client = mqtt.Client(client_id=name)

    username = (getattr(config, "mqtt_username", "") or "").strip()
    if username:
        client.username_pw_set(username, password or None)
    if getattr(config, "mqtt_tls", False):
        client.tls_set()
    return client


def availability_payload(config, online: bool) -> str:
    """The retained message on <prefix>/<node>/status."""
    return json.dumps({
        "node_id": getattr(config, "node_id", "") or "",
        "node_name": getattr(config, "node_name", "") or "",
        "status": "online" if online else "offline",
        "timestamp": round(time.time(), 3),
    })


def set_last_will(client, config) -> None:
    """Have the BROKER announce this node's death.

    Until now "is that node alive" could only be answered by noticing that
    messages had stopped — which needs a timeout in every dashboard, and which
    retained messages defeat entirely: the last reading of a node that died an
    hour ago looks exactly as fresh as a live one.

    A last will is the protocol's own answer. The broker holds the message and
    publishes it the moment the connection drops, however it drops — a killed
    process, a pulled cable, a node that locked up. Nothing on this side has
    to still be running for it to be sent, which is the entire point.
    """
    client.will_set(_topic(config, "status"),
                    availability_payload(config, online=False),
                    qos=int(getattr(config, "mqtt_qos", 0) or 0),
                    retain=True)


def _topic(config, suffix: str) -> str:
    prefix = (getattr(config, "mqtt_topic_prefix", "") or "ainode").strip("/")
    node_id = getattr(config, "node_id", "") or "node"
    if suffix == "cluster":
        # The fleet view is not a property of the node that happens to publish
        # it, so it does not live under that node's id.
        return f"{prefix}/cluster"
    return f"{prefix}/{node_id}/{suffix}"


def _broker(config) -> tuple:
    host = (getattr(config, "mqtt_host", "") or "").strip()
    try:
        port = int(getattr(config, "mqtt_port", 1883) or 1883)
    except (TypeError, ValueError):
        port = 1883
    return host, port


def test_connection(config, password: str) -> Dict[str, Any]:
    """Connect, publish nothing, disconnect. Returns a UI-shaped result.

    Exists because "did I type the password right" should not be answered by
    waiting for the next interval and then reading a log file.
    """
    host, port = _broker(config)
    if not host:
        return {"ok": False, "error": "No broker host configured."}
    try:
        client = _client(config, password)
    except MqttUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    try:
        client.connect(host, port, keepalive=CONNECT_TIMEOUT)
        client.disconnect()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "broker": f"{host}:{port}",
            "topic_example": _topic(config, "system")}


def publish_once(config, password: str, payloads: Dict[str, Dict[str, Any]]) -> int:
    """Connect, publish each payload, disconnect. Returns how many were sent.

    Used by the "Publish now" button. The running publisher keeps a persistent
    connection instead; this one is deliberately self-contained so it works
    whether or not telemetry is enabled.
    """
    host, port = _broker(config)
    if not host:
        raise MqttUnavailable("No broker host configured.")
    client = _client(config, password)
    client.connect(host, port, keepalive=CONNECT_TIMEOUT)
    sent = 0
    try:
        for suffix, payload in payloads.items():
            client.publish(
                _topic(config, suffix), json.dumps(payload),
                qos=int(getattr(config, "mqtt_qos", 0) or 0),
                retain=bool(getattr(config, "mqtt_retain", False)),
            )
            sent += 1
        # Without this the socket can close before paho has flushed QoS-0
        # messages, and nothing arrives at a broker that was reachable.
        client.loop_write()
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
    return sent


def publish_event(app, suffix: str) -> bool:
    """Publish one payload NOW, outside the interval. Returns whether it went.

    For the handful of things whose value is in their timing. The guard
    samples host memory every two seconds and the telemetry loop publishes
    every thirty; an engine killed to save the node would otherwise be news
    up to half a minute later — or never, if the node goes down in between.

    Best effort by design: no client, no broker, no publish, no exception. An
    event that cannot be sent must not become a second fault.
    """
    publisher = app.get("mqtt_publisher")
    client = getattr(publisher, "_client", None)
    if client is None:
        return False
    config = app.get("config")
    try:
        from ainode.metrics.system import SystemSampler
        from ainode.telemetry.payloads import build_payloads

        sampler = app.get("_telemetry_sampler") or SystemSampler()
        payload = build_payloads(app, sampler).get(suffix)
        if payload is None:
            return False
        client.publish(_topic(config, suffix), json.dumps(payload),
                       qos=int(getattr(config, "mqtt_qos", 0) or 0),
                       retain=bool(getattr(config, "mqtt_retain", False)))
        return True
    except Exception:
        logger.debug("could not publish %s as an event", suffix, exc_info=True)
        return False


class MqttPublisher:
    """Background task publishing telemetry on a timer."""

    def __init__(self, app):
        self._app = app
        self._task: Optional[asyncio.Task] = None
        self._client = None
        self._stopping = False
        self._last_error = ""
        self._published = 0

    # -- lifecycle ------------------------------------------------------

    def start(self) -> bool:
        """Begin publishing if it is enabled and configured. Idempotent."""
        config = self._app.get("config")
        if config is None or not getattr(config, "mqtt_enabled", False):
            return False
        if not (getattr(config, "mqtt_host", "") or "").strip():
            logger.warning("MQTT telemetry is enabled but no broker host is set")
            return False
        if self._task is not None and not self._task.done():
            return True
        self._stopping = False
        self._task = asyncio.get_event_loop().create_task(self._run())
        return True

    async def stop(self) -> None:
        self._stopping = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._disconnect()

    async def restart(self) -> None:
        """Apply changed settings. Called when the config is saved."""
        await self.stop()
        self.start()

    def status(self) -> Dict[str, Any]:
        config = self._app.get("config")
        running = self._task is not None and not self._task.done()
        host, port = _broker(config) if config is not None else ("", 0)
        return {
            "enabled": bool(getattr(config, "mqtt_enabled", False)) if config else False,
            "running": running,
            "broker": f"{host}:{port}" if host else "",
            "published": self._published,
            "last_error": self._last_error,
        }

    # -- internals ------------------------------------------------------

    def _password(self) -> str:
        secrets = self._app.get("secrets_manager")
        if secrets is None:
            return ""
        try:
            return secrets.get("mqtt_password") or ""
        except Exception:
            logger.debug("could not read the MQTT password", exc_info=True)
            return ""

    def _interval(self, config) -> int:
        try:
            value = int(getattr(config, "mqtt_interval", 30) or 30)
        except (TypeError, ValueError):
            value = 30
        return max(MIN_INTERVAL, min(MAX_INTERVAL, value))

    def _connect(self, config) -> None:
        host, port = _broker(config)
        client = _client(config, self._password())
        set_last_will(client, config)
        client.connect(host, port, keepalive=60)
        # paho's own loop thread handles reconnects, so a broker that is down
        # at boot or restarts overnight needs nothing from us.
        client.loop_start()
        # Retained, so a dashboard that subscribes later still learns the node
        # is up rather than waiting for the next interval.
        client.publish(_topic(config, "status"),
                       availability_payload(config, online=True),
                       qos=int(getattr(config, "mqtt_qos", 0) or 0), retain=True)
        self._client = client
        logger.info("MQTT telemetry connected to %s:%s", host, port)

    def _disconnect(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        # A deliberate shutdown says so, rather than leaving the last will to
        # report it as a failure. Both are "offline"; only one of them is a
        # reason to get out of bed.
        try:
            config = self._app.get("config")
            client.publish(_topic(config, "status"),
                           availability_payload(config, online=False),
                           qos=int(getattr(config, "mqtt_qos", 0) or 0),
                           retain=True)
        except Exception:
            logger.debug("could not publish the offline state", exc_info=True)
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            logger.debug("MQTT disconnect failed", exc_info=True)

    async def _run(self) -> None:
        from ainode.metrics.system import SystemSampler
        from ainode.telemetry.payloads import build_payloads

        config = self._app.get("config")
        models_dir = getattr(config, "models_dir", "") or ""
        sampler = SystemSampler(disk_paths=[models_dir] if models_dir else [])
        # Shared, so "publish now" measures against the loop's last sample
        # rather than against nothing. A rate needs two readings and a fresh
        # sampler has one, so a manual publish reported no network or CPU
        # rates at all — the fields were simply absent, which reads as a
        # broken metric rather than as a missing baseline.
        self._app["_telemetry_sampler"] = sampler
        from ainode.telemetry.logs import LogPublisher

        log_publisher = LogPublisher(self._app)
        self._app["_log_publisher"] = log_publisher
        loop = asyncio.get_event_loop()
        reported_error = ""

        while not self._stopping:
            config = self._app.get("config")
            interval = self._interval(config)
            try:
                if self._client is None:
                    await loop.run_in_executor(None, self._connect, config)
                payloads = await loop.run_in_executor(
                    None, build_payloads, self._app, sampler)
                # Logs are built here rather than in build_payloads because
                # they are stateful — each source publishes what has been
                # ADDED since the last cycle — and build_payloads is also
                # called by "publish now" and by the settings preview, which
                # must not consume anything.
                if log_publisher is not None:
                    try:
                        payloads.update(log_publisher.payloads())
                    except Exception:
                        logger.debug("log payloads failed", exc_info=True)
                # What the ENGINE reports, as opposed to what our proxy
                # measured. Async, so it cannot go in build_payloads — and it
                # should not: "publish now" and the settings preview would
                # then each scrape every instance.
                try:
                    from ainode.telemetry.engine_metrics import collect
                    from ainode.telemetry.payloads import node_identity

                    identity = node_identity(config)
                    for name, metrics in (await collect(self._app)).items():
                        payloads[f"engine/{name}"] = {
                            **identity, "instance": name, **metrics}
                except Exception:
                    logger.debug("engine metrics failed", exc_info=True)
                for suffix, payload in payloads.items():
                    self._client.publish(
                        _topic(config, suffix), json.dumps(payload),
                        qos=int(getattr(config, "mqtt_qos", 0) or 0),
                        retain=bool(getattr(config, "mqtt_retain", False)),
                    )
                self._published += len(payloads)
                self._last_error = ""
                reported_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._last_error = message
                # Once per distinct fault. A broker down for a day would
                # otherwise write a line every interval, all the same.
                if message != reported_error:
                    logger.warning("MQTT publish failed: %s", message)
                    reported_error = message
                self._disconnect()
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
