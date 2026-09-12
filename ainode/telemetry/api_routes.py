"""Settings and diagnostics for telemetry publishing."""

from __future__ import annotations

import logging

from aiohttp import web

from ainode.api.params import as_object, int_field, str_field

logger = logging.getLogger(__name__)

__all__ = ["register_telemetry_routes"]

_BOOL_FIELDS = ("mqtt_enabled", "mqtt_tls", "mqtt_retain")


def register_telemetry_routes(app: web.Application) -> None:
    app.router.add_get("/api/telemetry/mqtt", handle_get_settings)
    app.router.add_put("/api/telemetry/mqtt", handle_put_settings)
    app.router.add_post("/api/telemetry/mqtt/test", handle_test)
    app.router.add_post("/api/telemetry/mqtt/publish", handle_publish_now)
    app.router.add_get("/api/telemetry/preview", handle_preview)


def _settings(config) -> dict:
    return {
        "mqtt_enabled": bool(getattr(config, "mqtt_enabled", False)),
        "mqtt_host": getattr(config, "mqtt_host", "") or "",
        "mqtt_port": int(getattr(config, "mqtt_port", 1883) or 1883),
        "mqtt_username": getattr(config, "mqtt_username", "") or "",
        "mqtt_tls": bool(getattr(config, "mqtt_tls", False)),
        "mqtt_topic_prefix": getattr(config, "mqtt_topic_prefix", "") or "ainode",
        "mqtt_interval": int(getattr(config, "mqtt_interval", 30) or 30),
        "mqtt_retain": bool(getattr(config, "mqtt_retain", False)),
        "mqtt_qos": int(getattr(config, "mqtt_qos", 0) or 0),
    }


async def handle_get_settings(request: web.Request) -> web.Response:
    """Current settings, whether a password is stored, and the live status.

    The password itself is never returned — only whether one exists, which is
    the part a form needs in order to decide between "set" and "change".
    """
    config = request.app["config"]
    secrets = request.app.get("secrets_manager")
    publisher = request.app.get("mqtt_publisher")
    topics = _topic_examples(config)
    return web.json_response({
        "settings": _settings(config),
        "password_set": bool(secrets is not None and secrets.has("mqtt_password")),
        "status": publisher.status() if publisher is not None else {"running": False},
        "topics": topics,
    })


def _topic_examples(config) -> list:
    from ainode.telemetry.mqtt import _topic

    return [_topic(config, s) for s in ("system", "gpu", "models", "cluster")]


async def handle_put_settings(request: web.Request) -> web.Response:
    """Save settings. An empty password field leaves the stored one alone."""
    try:
        body = as_object(await request.json())
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    config = request.app["config"]
    for field in _BOOL_FIELDS:
        if field in body:
            setattr(config, field, bool(body[field]))

    host = str_field(body, "mqtt_host")
    if "mqtt_host" in body:
        config.mqtt_host = host
    if "mqtt_username" in body:
        config.mqtt_username = str_field(body, "mqtt_username")
    if "mqtt_topic_prefix" in body:
        config.mqtt_topic_prefix = str_field(
            body, "mqtt_topic_prefix", default="ainode").strip("/") or "ainode"

    port = int_field(body, "mqtt_port", minimum=1, maximum=65535)
    if port is not None:
        config.mqtt_port = port
    from ainode.telemetry.mqtt import MAX_INTERVAL, MIN_INTERVAL

    interval = int_field(body, "mqtt_interval",
                         minimum=MIN_INTERVAL, maximum=MAX_INTERVAL)
    if interval is not None:
        config.mqtt_interval = interval
    qos = int_field(body, "mqtt_qos", minimum=0, maximum=2)
    if qos is not None:
        config.mqtt_qos = qos

    if config.mqtt_enabled and not (config.mqtt_host or "").strip():
        return web.json_response(
            {"error": "Enabling MQTT needs a broker host."}, status=400)

    password = body.get("mqtt_password")
    secrets = request.app.get("secrets_manager")
    if isinstance(password, str) and secrets is not None:
        # "" means "leave it": a form that round-trips an empty password field
        # must not wipe a working credential.
        if password.strip():
            secrets.set("mqtt_password", password.strip())
        elif body.get("clear_password"):
            secrets.delete("mqtt_password")

    try:
        config.save()
    except Exception as exc:
        logger.exception("could not persist MQTT settings")
        return web.json_response({"error": f"Could not save: {exc}"}, status=500)

    publisher = request.app.get("mqtt_publisher")
    if publisher is not None:
        try:
            await publisher.restart()
        except Exception:
            logger.exception("could not restart the MQTT publisher")

    return web.json_response({
        "ok": True,
        "settings": _settings(config),
        "topics": _topic_examples(config),
        "status": publisher.status() if publisher is not None else {"running": False},
    })


def _password(request) -> str:
    secrets = request.app.get("secrets_manager")
    if secrets is None:
        return ""
    try:
        return secrets.get("mqtt_password") or ""
    except Exception:
        return ""


async def handle_test(request: web.Request) -> web.Response:
    """Connect and disconnect, so a typo is found now rather than in a log."""
    import asyncio

    from ainode.telemetry.mqtt import test_connection

    config = request.app["config"]
    result = await asyncio.get_event_loop().run_in_executor(
        None, test_connection, config, _password(request))
    return web.json_response(result, status=200 if result.get("ok") else 502)


async def handle_publish_now(request: web.Request) -> web.Response:
    """Publish one round immediately, enabled or not."""
    import asyncio

    from ainode.metrics.system import SystemSampler
    from ainode.telemetry.mqtt import MqttUnavailable, publish_once
    from ainode.telemetry.payloads import build_payloads

    config = request.app["config"]
    loop = asyncio.get_event_loop()
    sampler = request.app.get("_telemetry_sampler") or SystemSampler(
        disk_paths=[getattr(config, "models_dir", "") or ""])
    payloads = await loop.run_in_executor(None, build_payloads, request.app, sampler)
    try:
        sent = await loop.run_in_executor(
            None, publish_once, config, _password(request), payloads)
    except MqttUnavailable as exc:
        return web.json_response({"error": str(exc)}, status=503)
    except Exception as exc:
        return web.json_response(
            {"error": f"{type(exc).__name__}: {exc}"}, status=502)
    return web.json_response({"ok": True, "published": sent,
                              "topics": _topic_examples(config)})


async def handle_preview(request: web.Request) -> web.Response:
    """Exactly what would be published, without a broker.

    Someone building a dashboard needs the shape of the messages, and reading
    it off a live topic requires the publishing to work first.
    """
    import asyncio

    from ainode.metrics.system import SystemSampler
    from ainode.telemetry.payloads import build_payloads

    config = request.app["config"]
    sampler = SystemSampler(disk_paths=[getattr(config, "models_dir", "") or ""])
    loop = asyncio.get_event_loop()
    # Two samples: the first establishes the baseline the rates are measured
    # against, so a preview shows real numbers instead of an absent CPU.
    await loop.run_in_executor(None, build_payloads, request.app, sampler)
    await asyncio.sleep(1.0)
    payloads = await loop.run_in_executor(None, build_payloads, request.app, sampler)
    from ainode.telemetry.mqtt import _topic

    return web.json_response(
        {"payloads": {_topic(config, k): v for k, v in payloads.items()}})
