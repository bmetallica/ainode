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
    # Before the app starts: the apply jobs (profiles/jobs.py) live here.
    if app.get("profile_jobs") is None:
        app["profile_jobs"] = {}
    app.router.add_get("/api/profiles", handle_list_profiles)
    app.router.add_post("/api/profiles", handle_create_profile)
    app.router.add_post("/api/profiles/capture", handle_capture_profile)
    app.router.add_get("/api/profiles/{name}", handle_get_profile)
    app.router.add_put("/api/profiles/{name}", handle_put_profile)
    app.router.add_delete("/api/profiles/{name}", handle_delete_profile)
    app.router.add_post("/api/profiles/{name}/apply", handle_apply_profile)
    app.router.add_post("/api/profiles/{name}/default", handle_set_default)
    # Registered before the {name} routes would matter only for GET; this is a
    # POST on a path no profile route has.
    app.router.add_post("/api/profiles/converge", handle_converge)
    app.router.add_get("/api/profiles/jobs/current", handle_current_job)
    app.router.add_get("/api/profiles/jobs/{job_id}", handle_get_job)
    app.router.add_post("/api/profiles/jobs/{job_id}/cancel", handle_cancel_job)
    app.router.add_post("/api/profiles/jobs/{job_id}/restore", handle_restore_job)
    app.router.add_get("/api/profiles/{name}/opencode", handle_profile_opencode)


async def handle_current_job(request: web.Request) -> web.Response:
    """GET /api/profiles/jobs/current — the running apply, or the last one;
    ``{"job": null}`` when there has been none since this node started."""
    from ainode.profiles.jobs import current_job

    job = current_job(request.app)
    return web.json_response({"job": job.to_dict() if job else None})


async def handle_get_job(request: web.Request) -> web.Response:
    from ainode.profiles.jobs import get_job

    job = get_job(request.app, request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "no such job"}, status=404)
    return web.json_response({"job": job.to_dict()})


async def handle_cancel_job(request: web.Request) -> web.Response:
    """Stop after the entry that is starting now; the rest are skipped."""
    from ainode.profiles.jobs import get_job

    job = get_job(request.app, request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "no such job"}, status=404)
    if job.state == "running":
        job.cancel_requested = True
    return web.json_response({"job": job.to_dict()})


async def handle_restore_job(request: web.Request) -> web.Response:
    """Apply what ran before this job — its own capture — as a new job."""
    from ainode.profiles.jobs import get_job, start_apply_job

    job = get_job(request.app, request.match_info["job_id"])
    if job is None:
        return web.json_response({"error": "no such job"}, status=404)
    if job.before is None:
        return web.json_response({"error": "nothing was captured before this job"},
                                 status=409)
    try:
        restore = await start_apply_job(request.app, job.before,
                                        restore_of=job.profile.name)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=409)
    return web.json_response({"ok": True, "job": restore.to_dict()}, status=202)


async def handle_profile_opencode(request: web.Request) -> web.Response:
    """GET /api/profiles/{name}/opencode?base_url= — an OpenCode config for
    what the profile will serve (clients/opencode.py), not for what runs now."""
    import asyncio

    from ainode.clients.opencode import build_opencode_config_for_profile

    profile = get_store(request.app).get(request.match_info.get("name", ""))
    if profile is None:
        return web.json_response({"error": "no such profile"}, status=404)
    base = str(request.query.get("base_url") or "")
    if not base:
        config = request.app.get("config")
        host = getattr(config, "fabric_ip", "") or "127.0.0.1"
        base = f"http://{host}:{getattr(config, 'web_port', 3000)}"
    payload = await asyncio.get_event_loop().run_in_executor(
        None, build_opencode_config_for_profile, request.app, profile, base)
    return web.json_response(payload)


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
    """DELETE /api/profiles/{name}[?stop=1] — with ``stop``, the profile's own
    models are stopped on every node it names first (nothing else is)."""
    store = get_store(request.app)
    name = request.match_info.get("name", "")
    profile = store.get(name)
    if profile is None:
        return web.json_response({"error": f"No profile named {name!r}."}, status=404)
    stopped = []
    query = getattr(request, "query", None) or {}
    if query.get("stop") in ("1", "true", "yes"):
        stopped = await _stop_profile_models(request.app, profile)
    store.delete(name)
    return web.json_response({"ok": True, "deleted": name, "stopped": stopped,
                              "default": store.default_name})


async def _stop_profile_models(app, profile) -> list:
    from ainode.profiles.apply import _entry_runs_here, _peer_converge, _unload_listed

    own = str(getattr(app.get("config"), "node_id", "") or "")
    stopped = _unload_listed(app, [e for e in profile.entries
                                   if _entry_runs_here(app, e)])
    by_peer: dict = {}
    for entry in profile.entries:
        if not _entry_runs_here(app, entry):
            by_peer.setdefault(entry.node_ids[0], []).append(entry)
    for node_id, entries in by_peer.items():
        if node_id == own:
            continue
        answer = await _peer_converge(app, node_id, "unload", entries)
        stopped.extend(f"{m} ({node_id})" for m in (answer or {}).get("stopped") or [])
    return stopped


async def handle_converge(request: web.Request) -> web.Response:
    """POST /api/profiles/converge {phase, entries} — the head asking this
    node to converge onto its part of a profile (profiles/apply.py). Cluster
    key only: it stops models without asking anyone."""
    from ainode.auth.cluster_key import is_cluster_request
    from ainode.profiles.apply import converge_here
    from ainode.profiles.store import ProfileEntry

    if not is_cluster_request(request.headers):
        return web.json_response({"error": "cluster key required"}, status=403)
    body = await _body(request)
    phase = str(body.get("phase") or "")
    try:
        entries = [ProfileEntry.from_dict(e) for e in (body.get("entries") or [])]
        result = await converge_here(request.app, phase, entries)
    except (ProfileError, ValueError, TypeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True, **result})


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
    if body.get("background"):
        # As a job the page can follow, cancel and undo (profiles/jobs.py).
        from ainode.profiles.jobs import start_apply_job

        try:
            job = await start_apply_job(request.app, profile)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        return web.json_response({"ok": True, "job": job.to_dict()}, status=202)
    wait = body.get("wait") is not False
    try:
        report = await apply_profile(request.app, profile, wait=wait)
    except Exception as exc:
        logger.exception("applying profile %s raised", name)
        return web.json_response({"error": f"Applying {name!r} failed: {exc}"},
                                 status=500)
    return web.json_response(report, status=200 if report.get("ok") else 207,
                             dumps=json.dumps)
