"""Checking for and running a source update, from the UI."""

from __future__ import annotations

import asyncio
import logging
import time

from aiohttp import web

from ainode.api.params import as_object, str_field, str_list_field
from ainode.update.runner import CONTAINER_SOURCE_DIR, UpdateRunner
from ainode.update.source import CACHE_SECONDS, built_from, check_for_updates

logger = logging.getLogger(__name__)

__all__ = ["register_update_routes", "get_update_runner"]


def get_update_runner(app) -> UpdateRunner:
    runner = app.get("update_runner")
    if runner is None:
        runner = UpdateRunner(app)
        app["update_runner"] = runner
    return runner


def register_update_routes(app: web.Application) -> None:
    app.router.add_get("/api/update/check", handle_check)
    app.router.add_get("/api/update/settings", handle_get_settings)
    app.router.add_put("/api/update/settings", handle_put_settings)
    app.router.add_post("/api/update/run", handle_run)
    app.router.add_get("/api/update/status", handle_status)


def _settings(config) -> dict:
    from ainode.update.runner import DEFAULT_SOURCE_DIR

    return {
        "source_repo": getattr(config, "source_repo", "") or "",
        "source_branch": getattr(config, "source_branch", "") or "main",
        "source_dir": getattr(config, "source_dir", "") or DEFAULT_SOURCE_DIR,
        "cluster_ssh_nodes": list(getattr(config, "cluster_ssh_nodes", []) or []),
    }


async def handle_check(request: web.Request) -> web.Response:
    """GET /api/update/check[?force=1] — is our fork's branch ahead of us?

    Cached for an hour. The UI asks on that cadence and the manual button
    passes force, because "check now" that returns a cached answer is not a
    check.
    """
    config = request.app["config"]
    force = str(request.query.get("force") or "").lower() in ("1", "true", "yes")
    cached = request.app.get("_update_state")
    if (cached is not None and not force
            and time.time() - cached.checked_at < CACHE_SECONDS):
        payload = cached.to_dict()
        payload["cached"] = True
    else:
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
            None, check_for_updates,
            _settings(config)["source_repo"], _settings(config)["source_branch"])
        request.app["_update_state"] = state
        payload = state.to_dict()
        payload["cached"] = False

    runner = get_update_runner(request.app)
    payload["can_run"] = not runner.why_not()
    payload["why_not"] = runner.why_not()
    payload["settings"] = _settings(config)
    return web.json_response(payload)


async def handle_get_settings(request: web.Request) -> web.Response:
    return web.json_response({
        "settings": _settings(request.app["config"]),
        "built_from": built_from(),
        "can_run": not get_update_runner(request.app).why_not(),
        "why_not": get_update_runner(request.app).why_not(),
        "container_source_dir": CONTAINER_SOURCE_DIR,
    })


async def handle_put_settings(request: web.Request) -> web.Response:
    try:
        body = as_object(await request.json())
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    config = request.app["config"]
    if "source_repo" in body:
        repo = str_field(body, "source_repo").strip().strip("/")
        if repo and repo.count("/") != 1:
            return web.json_response(
                {"error": f"{repo!r} is not an owner/name repository"},
                status=400)
        config.source_repo = repo
    if "source_branch" in body:
        config.source_branch = str_field(body, "source_branch",
                                         default="main").strip() or "main"
    if "source_dir" in body:
        config.source_dir = str_field(body, "source_dir").strip()
    if "cluster_ssh_nodes" in body:
        nodes = []
        for node in str_list_field(body, "cluster_ssh_nodes"):
            node = node.strip()
            # It is pasted straight into an ssh command line by the update
            # script, so a shell metacharacter here is a command on three
            # machines rather than a typo.
            if node and not _is_safe_ssh_target(node):
                return web.json_response(
                    {"error": f"{node!r} is not a plain ssh target "
                              f"(letters, digits, . _ - @ : and no spaces)"},
                    status=400)
            if node and node not in nodes:
                nodes.append(node)
        config.cluster_ssh_nodes = nodes

    try:
        config.save()
    except Exception as exc:
        return web.json_response({"error": f"could not save: {exc}"}, status=500)
    # The answer changes with the repo, so a stale one would be about the old
    # setting.
    request.app.pop("_update_state", None)
    return await handle_get_settings(request)


def _is_safe_ssh_target(value: str) -> bool:
    import re

    return bool(re.fullmatch(r"[A-Za-z0-9._@:-]{1,128}", value))


async def handle_run(request: web.Request) -> web.Response:
    """POST /api/update/run {nodes?, base?, images?} — git pull, then update."""
    try:
        body = as_object(await request.json()) if request.can_read_body else {}
    except Exception:
        body = {}
    config = request.app["config"]
    nodes = str_list_field(body, "nodes") or list(
        getattr(config, "cluster_ssh_nodes", []) or [])
    bad = [n for n in nodes if not _is_safe_ssh_target(n)]
    if bad:
        return web.json_response(
            {"error": f"unusable ssh target(s): {', '.join(bad)}"}, status=400)

    result = get_update_runner(request.app).start(
        nodes=nodes, base=bool(body.get("base")), images=bool(body.get("images")))
    if not result.get("ok"):
        return web.json_response({"error": result["error"]},
                                 status=result.get("status", 500))
    return web.json_response({"ok": True, "nodes": nodes})


async def handle_status(request: web.Request) -> web.Response:
    return web.json_response(get_update_runner(request.app).job)
