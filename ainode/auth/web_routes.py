"""Sign-in for the web UI, and the middleware that asks for it.

See auth/web_login.py for what is protected and why, and auth/cluster_key.py
for how the nodes get past it when they call each other.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import quote, urlparse

from aiohttp import web

from ainode.auth.cluster_key import cluster_headers, is_cluster_request
from ainode.auth.web_login import (COOKIE_NAME, SESSION_SECONDS, LoginThrottle,
                                   WebLogin, password_problem)

logger = logging.getLogger(__name__)

__all__ = ["register_web_login", "web_login_middleware", "is_public"]

#: Reachable without signing in. Everything the login page itself needs, the
#: health check a load balancer or a script asks, and the Prometheus endpoint —
#: a scraper cannot sign in, and what it reads is already on the network as
#: MQTT telemetry.
_PUBLIC_EXACT = frozenset({
    "/login", "/api/login", "/api/logout", "/api/auth/web/status",
    "/api/health", "/favicon.ico", "/metrics",
})
#: ``/v1/`` is the OpenAI-compatible proxy: Open WebUI and opencode cannot sign
#: in to a web page. It keeps the optional API keys it had before
#: (auth/middleware.py), which apply to it alone now.
_PUBLIC_PREFIXES = ("/static/", "/v1/")

_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def is_public(path: str) -> bool:
    return path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIXES)


def _is_loopback(request: web.Request) -> bool:
    remote = str(request.remote or "")
    return remote in ("::1", "localhost") or remote.startswith("127.")


def _bearer_ok(request: web.Request) -> bool:
    """A valid API key also opens /api/*, for automation that has one."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    config = request.app.get("auth_config")
    try:
        return bool(config is not None and config.api_keys
                    and config.validate_token(header[7:]))
    except Exception:
        return False


def _same_origin(request: web.Request) -> bool:
    """A browser request that changes something must come from this UI.

    SameSite=Lax already keeps the cookie off a cross-site POST in every
    current browser; this is the second lock, for the one that does not. No
    Origin and no Referer means not a browser, and a non-browser with a valid
    cookie is not what CSRF is about.
    """
    source = request.headers.get("Origin") or request.headers.get("Referer")
    if not source or source == "null":
        return not source   # "null" Origin: sandboxed or file:// — refuse
    host = request.headers.get("Host", "")
    try:
        return urlparse(source).netloc == host
    except ValueError:
        return False


def _login(request: web.Request) -> WebLogin:
    return request.app["web_login"]


def _signed_in(request: web.Request) -> bool:
    try:
        return _login(request).check(request.cookies.get(COOKIE_NAME, ""))
    except Exception:
        return False


@web.middleware
async def web_login_middleware(request: web.Request, handler):
    path = request.path
    if request.app.get("web_login") is None or is_public(path):
        return await handler(request)
    if request.method == "OPTIONS":
        return await handler(request)
    # Not a browser, and trusted for reasons of their own. Loopback can be
    # switched off — the tests do, since their client is always 127.0.0.1, and
    # so should anyone who puts a reverse proxy on the same host, where every
    # request would otherwise arrive looking local.
    loopback = request.app.get("trust_loopback", True) and _is_loopback(request)
    if loopback or is_cluster_request(request.headers) or _bearer_ok(request):
        return await handler(request)

    if _signed_in(request):
        if request.method in _MUTATING and not _same_origin(request):
            return web.json_response(
                {"error": "cross-site request refused"}, status=403)
        return await handler(request)

    if path.startswith("/api/"):
        return web.json_response(
            {"error": "sign in required", "login": "/login"}, status=401)
    target = request.path_qs if request.method == "GET" else "/"
    raise web.HTTPFound(f"/login?next={quote(target, safe='')}")


# -- routes ---------------------------------------------------------------


def _safe_next(value: Optional[str]) -> str:
    """Only a path on this server. An open redirect after sign-in is a phishing
    link with a real login page in the middle of it."""
    value = str(value or "/")
    if not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value


def _set_cookie(response: web.StreamResponse, token: str) -> None:
    # Not Secure: the UI is served over plain HTTP on the LAN, and a Secure
    # cookie would never be sent back. Lax rather than Strict so following a
    # link to the dashboard from another page does not show the sign-in again.
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS,
                        httponly=True, samesite="Lax", path="/")


async def handle_login_page(request: web.Request) -> web.Response:
    if _signed_in(request):
        raise web.HTTPFound(_safe_next(request.query.get("next")))
    from ainode.web.serve import get_login_html

    return web.Response(text=get_login_html(), content_type="text/html",
                        headers={"Cache-Control": "no-store"})


async def handle_login(request: web.Request) -> web.Response:
    throttle: LoginThrottle = request.app["login_throttle"]
    address = str(request.remote or "?")
    wait = throttle.blocked(address)
    if wait:
        return web.json_response(
            {"error": f"Too many attempts. Try again in {int(wait) + 1} s."},
            status=429)
    try:
        body = await request.json()
    except Exception:
        body = {}
    user = str((body or {}).get("user") or "")
    password = str((body or {}).get("password") or "")
    login = _login(request)
    ok = await asyncio.get_event_loop().run_in_executor(
        None, login.verify, user, password)
    if not ok:
        throttle.failed(address)
        logger.warning("failed web sign-in for %r from %s", user, address)
        return web.json_response({"error": "Wrong user name or password."},
                                 status=401)
    throttle.succeeded(address)
    response = web.json_response({
        "ok": True, "next": _safe_next(body.get("next")),
        "is_default": login.is_default})
    _set_cookie(response, login.issue())
    return response


async def handle_logout(request: web.Request) -> web.Response:
    response = web.json_response({"ok": True})
    response.del_cookie(COOKIE_NAME, path="/")
    return response


async def handle_status(request: web.Request) -> web.Response:
    login = _login(request)
    signed_in = _signed_in(request)
    payload = {"signed_in": signed_in}
    if signed_in:
        payload.update(user=login.user, is_default=login.is_default)
    return web.json_response(payload)


async def handle_change_password(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    login = _login(request)
    current = str(body.get("current") or "")
    new = str(body.get("new") or "")
    user = str(body.get("user") or "").strip() or None
    problem = password_problem(new)
    if problem:
        return web.json_response({"error": problem}, status=400)
    changed, why = await asyncio.get_event_loop().run_in_executor(
        None, lambda: login.set_password(current, new, user))
    if not changed:
        return web.json_response({"error": why}, status=400)

    peers = {}
    if body.get("all_nodes", True):
        peers = await _push_to_peers(request.app, login.export_hash())
    response = web.json_response({"ok": True, "user": login.user,
                                  "nodes": peers})
    # The secret rotated with the password, so this browser needs a new cookie
    # to stay signed in; every other browser is signed out, as intended.
    _set_cookie(response, login.issue())
    return response


async def _push_to_peers(app, blob) -> dict:
    """Give every other node the same credentials. {node: "ok" | reason}."""
    cluster = app.get("cluster_state")
    session = app.get("client_session")
    own = str(getattr(app.get("config"), "node_id", "") or "")
    out = {}
    if cluster is None or session is None:
        return out
    headers = cluster_headers()
    if not headers:
        return {"*": "no cluster key on this node"}
    for node in cluster.members():
        if node.node_id == own:
            continue
        host = (getattr(node, "fabric_ip", "") or "").strip() or \
            (getattr(node, "peer_ip", "") or "").strip()
        if not host:
            out[node.node_id] = "no address"
            continue
        url = f"http://{host}:{getattr(node, 'web_port', 3000) or 3000}" \
              f"/api/auth/web/sync"
        try:
            async with session.post(url, json=blob, headers=headers,
                                    timeout=10) as resp:
                out[node.node_id] = "ok" if resp.status == 200 else \
                    f"HTTP {resp.status}"
        except Exception as exc:
            out[node.node_id] = f"unreachable: {exc.__class__.__name__}"
    return out


async def handle_sync(request: web.Request) -> web.Response:
    """Another node's new credentials. Cluster key only — not a session, not
    loopback: this is the one route that replaces the password without asking
    for the old one."""
    if not is_cluster_request(request.headers):
        return web.json_response({"error": "cluster key required"}, status=403)
    try:
        blob = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    if not _login(request).import_hash(blob):
        return web.json_response({"error": "incomplete credentials"},
                                 status=400)
    return web.json_response({"ok": True})


def register_web_login(app: web.Application, login: Optional[WebLogin] = None,
                       ui: bool = True) -> None:
    app["web_login"] = login or WebLogin()
    app["login_throttle"] = LoginThrottle()
    if ui:
        app.router.add_get("/login", handle_login_page)
    app.router.add_post("/api/login", handle_login)
    app.router.add_post("/api/logout", handle_logout)
    app.router.add_get("/api/auth/web/status", handle_status)
    app.router.add_post("/api/auth/web/password", handle_change_password)
    app.router.add_post("/api/auth/web/sync", handle_sync)
