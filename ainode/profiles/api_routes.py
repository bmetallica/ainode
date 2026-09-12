"""HTTP surface for profiles.

Deliberately thin: every handler validates, then calls into
:mod:`ainode.profiles.store` or :mod:`ainode.profiles.apply`. A profile is not
a second way to launch a model — applying one goes through the same routes the
dashboard's buttons do.
"""

from __future__ import annotations

import json
import logging

from aiohttp import web

from ainode.api.params import as_object, str_field
from ainode.profiles.apply import apply_profile, capture_profile
from ainode.profiles.store import Profile, ProfileError, ProfileStore

logger = logging.getLogger(__name__)

__all__ = ["register_profile_routes", "get_store"]


def get_store(app) -> ProfileStore:
    """The app's profile store, created on first use."""
    store = app.get("profiles")
    if store is None:
        store = ProfileStore()
        app["profiles"] = store
    return store


def register_profile_routes(app: web.Application) -> None:
    app.router.add_get("/api/profiles", handle_list_profiles)
    app.router.add_post("/api/profiles", handle_create_profile)
    app.router.add_post("/api/profiles/capture", handle_capture_profile)
    app.router.add_get("/api/profiles/{name}", handle_get_profile)
    app.router.add_put("/api/profiles/{name}", handle_put_profile)
    app.router.add_delete("/api/profiles/{name}", handle_delete_profile)
    app.router.add_post("/api/profiles/{name}/apply", handle_apply_profile)
    app.router.add_post("/api/profiles/{name}/default", handle_set_default)


async def _body(request) -> dict:
    try:
        return as_object(await request.json())
    except Exception:
        return {}


async def handle_list_profiles(request: web.Request) -> web.Response:
    store = get_store(request.app)
    return web.json_response({
        "profiles": [p.to_dict() for p in store.all()],
        "default": store.default_name,
    })


async def handle_get_profile(request: web.Request) -> web.Response:
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    profile = store.get(name)
    if profile is None:
        return web.json_response({"error": f"No profile named {name!r}."}, status=404)
    return web.json_response({
        "profile": profile.to_dict(),
        "is_default": store.default_name == profile.name,
    })


async def handle_create_profile(request: web.Request) -> web.Response:
    store = get_store(request.app)
    body = await _body(request)
    try:
        profile = Profile.from_dict(body)
    except ProfileError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    if store.get(profile.name) is not None:
        return web.json_response(
            {"error": f"A profile named {profile.name!r} already exists; "
                      f"PUT it to replace it."},
            status=409)
    try:
        store.put(profile)
    except OSError as exc:
        return web.json_response({"error": f"Could not save: {exc}"}, status=500)
    return web.json_response({"ok": True, "profile": profile.to_dict()})


async def handle_put_profile(request: web.Request) -> web.Response:
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    body = await _body(request)
    body.setdefault("name", name)
    if str(body.get("name") or "") != name:
        return web.json_response(
            {"error": "The profile name in the URL and the body must match; "
                      "rename by creating the new one and deleting the old."},
            status=400)
    try:
        profile = Profile.from_dict(body)
        store.put(profile)
    except ProfileError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError as exc:
        return web.json_response({"error": f"Could not save: {exc}"}, status=500)
    return web.json_response({"ok": True, "profile": profile.to_dict()})


async def handle_delete_profile(request: web.Request) -> web.Response:
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    if not store.delete(name):
        return web.json_response({"error": f"No profile named {name!r}."}, status=404)
    return web.json_response({"ok": True, "deleted": name,
                              "default": store.default_name})


async def handle_set_default(request: web.Request) -> web.Response:
    """Set the profile applied at startup. ``{"default": false}`` clears it."""
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    body = await _body(request)
    clear = body.get("default") is False
    try:
        store.set_default("" if clear else name)
    except ProfileError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except OSError as exc:
        return web.json_response({"error": f"Could not save: {exc}"}, status=500)
    return web.json_response({"ok": True, "default": store.default_name})


async def handle_capture_profile(request: web.Request) -> web.Response:
    """Save what is running right now as a profile."""
    store = get_store(request.app)
    body = await _body(request)
    name = str_field(body, "name")
    if not name:
        return web.json_response({"error": "name field required"}, status=400)
    if store.get(name) is not None and not body.get("overwrite"):
        return web.json_response(
            {"error": f"A profile named {name!r} already exists; pass "
                      f"overwrite: true to replace it."},
            status=409)
    try:
        profile = capture_profile(request.app, name,
                                  str_field(body, "description"))
        store.put(profile)
    except ProfileError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError as exc:
        return web.json_response({"error": f"Could not save: {exc}"}, status=500)
    return web.json_response({"ok": True, "profile": profile.to_dict(),
                              "captured": len(profile.entries)})


async def handle_apply_profile(request: web.Request) -> web.Response:
    """Converge the node onto this profile.

    Long-running by nature — each model has to be answering before the next
    starts — so this holds the request open until the last entry is done. The
    UI shows the per-entry report it returns; there is no partial progress to
    poll because the instance list already shows models coming up.
    """
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    profile = store.get(name)
    if profile is None:
        return web.json_response({"error": f"No profile named {name!r}."}, status=404)

    body = await _body(request)
    wait = body.get("wait") is not False
    try:
        report = await apply_profile(request.app, profile, wait=wait)
    except Exception as exc:
        logger.exception("applying profile %s raised", name)
        return web.json_response({"error": f"Applying {name!r} failed: {exc}"},
                                 status=500)
    return web.json_response(report, status=200 if report.get("ok") else 207,
                             dumps=json.dumps)
