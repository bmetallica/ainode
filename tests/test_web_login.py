"""The password in front of the web UI.

Asked for after the review found the API open to the whole LAN:

    bei K1 setze ein passwort nur für das webui (bitte im style passend zur
    anwendung) standarduser=admin passwort=admin das passwort soll im ui zu
    ändern sein, das ui darf im lan erreichbar bleiben

Every /api/* route was reachable without signing in — including the update
that runs git and restarts the node, and the ones that start containers. The
container holds the host's docker socket and the operator's SSH keys.

What these tests hold down: the UI and /api/* need a session; /v1/* does not
(its clients cannot sign in); the other nodes get past with the cluster key;
the default is admin/admin and says so; a changed password signs other browsers
out; and the old API keys no longer lock the dashboard out of itself.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ainode.auth import cluster_key as ck
from ainode.auth.middleware import AuthConfig, auth_middleware
from ainode.auth.web_login import (COOKIE_NAME, DEFAULT_PASSWORD, DEFAULT_USER,
                                   LoginThrottle, WebLogin, password_problem)
from ainode.auth.web_routes import (is_public, register_web_login,
                                    web_login_middleware)


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    """Nothing here may write to the real ~/.ainode."""
    monkeypatch.setattr(ck, "key_path", lambda: tmp_path / "cluster.key")
    ck._CACHE.update(mtime=None, path=None, key="")
    yield
    ck._CACHE.update(mtime=None, path=None, key="")


def _app(tmp_path, *, api_keys: AuthConfig = None) -> web.Application:
    async def ok(request):
        return web.json_response({"ok": True})

    async def page(request):
        return web.Response(text="<html>dashboard</html>",
                            content_type="text/html")

    app = web.Application(middlewares=[web_login_middleware, auth_middleware])
    # The test client always connects from 127.0.0.1, which the middleware
    # trusts in production. Off here, or nothing below would be tested.
    app["trust_loopback"] = False
    app["auth_config"] = api_keys or AuthConfig()
    register_web_login(app, WebLogin(tmp_path / "web-auth.json"), ui=False)
    app.router.add_get("/", page)
    app.router.add_get("/api/status", ok)
    app.router.add_post("/api/models/unload", ok)
    app.router.add_get("/v1/models", ok)
    app.router.add_get("/api/health", ok)
    app.router.add_get("/static/css/style.css", ok)
    return app


@pytest_asyncio.fixture
async def client(tmp_path):
    async with TestClient(TestServer(_app(tmp_path))) as c:
        yield c


async def _sign_in(client, user=DEFAULT_USER, password=DEFAULT_PASSWORD):
    return await client.post("/api/login",
                             json={"user": user, "password": password})


class TestTheCredentials:
    def test_a_fresh_node_is_admin_admin(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        assert login.verify("admin", "admin")
        assert login.is_default is True

    def test_nothing_is_written_until_someone_signs_in(self, tmp_path):
        # A node nobody has signed in to should not grow a credentials file,
        # and a test suite must not write one into the real home.
        WebLogin(tmp_path / "w.json")
        assert not (tmp_path / "w.json").exists()

    def test_the_first_sign_in_persists_it(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        token = login.issue()
        again = WebLogin(tmp_path / "w.json")
        # Same secret after a restart: the session survives it.
        assert again.check(token)

    def test_the_password_is_not_stored(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        login.issue()
        text = (tmp_path / "w.json").read_text()
        assert '"admin"' in text                  # the user name, yes
        assert "admin\"" in text and "password_hash" in text
        assert json.loads(text)["password_hash"] != "admin"

    def test_the_file_is_private(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        login.issue()
        assert oct((tmp_path / "w.json").stat().st_mode & 0o777) == "0o600"

    def test_wrong_user_or_password_fails(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        assert not login.verify("admin", "nope")
        assert not login.verify("root", "admin")

    def test_changing_it_needs_the_current_one(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        ok, why = login.set_password("wrong", "correct horse")
        assert not ok and "current password" in why

    def test_a_changed_password_works_and_the_old_does_not(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        assert login.set_password("admin", "correct horse")[0]
        assert login.verify("admin", "correct horse")
        assert not login.verify("admin", "admin")
        assert login.is_default is False

    def test_the_user_name_can_change_too(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        login.set_password("admin", "correct horse", user="ops")
        assert login.verify("ops", "correct horse")

    def test_changing_it_signs_other_browsers_out(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        before = login.issue()
        login.set_password("admin", "correct horse")
        assert not login.check(before)
        assert login.check(login.issue())

    @pytest.mark.parametrize("password,problem", [
        ("short", "at least 8"), ("admin", "at least 8"),
    ])
    def test_a_weak_new_password_is_refused(self, password, problem):
        assert problem in password_problem(password)

    def test_a_good_one_is_accepted(self):
        assert password_problem("correct horse") == ""


class TestSessions:
    def test_a_tampered_cookie_is_refused(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        body, mac = login.issue().split(".")
        assert not login.check(body + "." + mac[:-2] + "AA")

    def test_an_expired_cookie_is_refused(self, tmp_path):
        login = WebLogin(tmp_path / "w.json")
        token = login.issue(now=1_000.0)
        assert not login.check(token, now=1_000.0 + 31 * 24 * 3600)

    @pytest.mark.parametrize("junk", ["", "x", "a.b", "....", "a" * 400])
    def test_junk_is_refused_not_raised(self, tmp_path, junk):
        assert WebLogin(tmp_path / "w.json").check(junk) is False


class TestWhatIsProtected:
    @pytest.mark.asyncio
    async def test_the_api_needs_a_session(self, client):
        resp = await client.get("/api/status")
        assert resp.status == 401
        assert (await resp.json())["login"] == "/login"

    @pytest.mark.asyncio
    async def test_the_dashboard_redirects_to_the_sign_in(self, client):
        resp = await client.get("/", allow_redirects=False)
        assert resp.status == 302
        assert resp.headers["Location"].startswith("/login?next=")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/v1/models", "/api/health",
                                      "/static/css/style.css"])
    async def test_what_stays_open(self, client, path):
        """/v1/ because Open WebUI and opencode cannot sign in; the health
        check for scripts; static files for the sign-in page itself."""
        assert (await client.get(path)).status == 200

    def test_the_public_list_is_short(self):
        assert is_public("/v1/chat/completions")
        assert not is_public("/api/update/run")
        assert not is_public("/api/secrets/HF_TOKEN")
        assert not is_public("/onboarding")


class TestSigningIn:
    @pytest.mark.asyncio
    async def test_admin_admin_works_on_a_fresh_node(self, client):
        resp = await _sign_in(client)
        assert resp.status == 200
        assert (await resp.json())["is_default"] is True

    @pytest.mark.asyncio
    async def test_then_the_api_answers(self, client):
        await _sign_in(client)
        assert (await client.get("/api/status")).status == 200

    @pytest.mark.asyncio
    async def test_the_cookie_is_http_only_and_lax(self, client):
        resp = await _sign_in(client)
        cookie = resp.headers.get("Set-Cookie", "")
        assert COOKIE_NAME in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=Lax" in cookie

    @pytest.mark.asyncio
    async def test_a_wrong_password_is_a_401(self, client):
        assert (await _sign_in(client, password="nope")).status == 401

    @pytest.mark.asyncio
    async def test_signing_out_ends_it(self, client):
        await _sign_in(client)
        await client.post("/api/logout",
                          headers={"Origin": str(client.make_url("/")).rstrip("/")})
        client.session.cookie_jar.clear()
        assert (await client.get("/api/status")).status == 401

    @pytest.mark.asyncio
    async def test_the_next_page_cannot_leave_the_server(self, client):
        # An open redirect after sign-in is a phishing link with a real login
        # page in the middle of it.
        resp = await client.post("/api/login", json={
            "user": "admin", "password": "admin",
            "next": "//evil.example/steal"})
        assert (await resp.json())["next"] == "/"


class TestGuessing:
    def test_ten_failures_block_the_address(self):
        throttle = LoginThrottle(limit=10, window=300)
        for _ in range(10):
            throttle.failed("10.0.0.9", now=100.0)
        assert throttle.blocked("10.0.0.9", now=101.0) > 0
        assert throttle.blocked("10.0.0.8", now=101.0) == 0

    def test_the_block_lapses(self):
        throttle = LoginThrottle(limit=3, window=60)
        for _ in range(3):
            throttle.failed("a", now=0.0)
        assert throttle.blocked("a", now=61.0) == 0

    @pytest.mark.asyncio
    async def test_the_route_answers_429(self, client):
        client.app["login_throttle"].limit = 2
        for _ in range(2):
            await _sign_in(client, password="nope")
        assert (await _sign_in(client)).status == 429


class TestCrossSiteRequests:
    """K3: a signed-in browser must not be driven by another site."""

    @pytest.mark.asyncio
    async def test_a_foreign_origin_is_refused(self, client):
        await _sign_in(client)
        resp = await client.post("/api/models/unload", json={},
                                 headers={"Origin": "http://evil.example"})
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_this_ui_is_accepted(self, client):
        await _sign_in(client)
        origin = str(client.make_url("/")).rstrip("/")
        resp = await client.post("/api/models/unload", json={},
                                 headers={"Origin": origin})
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_a_null_origin_is_refused(self, client):
        await _sign_in(client)
        resp = await client.post("/api/models/unload", json={},
                                 headers={"Origin": "null"})
        assert resp.status == 403


class TestTheNodesGetPast:
    @pytest.mark.asyncio
    async def test_with_the_cluster_key(self, client):
        key = ck.cluster_key()
        resp = await client.get("/api/status",
                                headers={ck.CLUSTER_HEADER: key})
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_not_with_a_wrong_one(self, client):
        ck.cluster_key()
        resp = await client.get("/api/status",
                                headers={ck.CLUSTER_HEADER: "guess"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_not_when_this_node_has_none(self, client):
        # An empty key matches nothing — refusing is the safe default.
        resp = await client.get("/api/status",
                                headers={ck.CLUSTER_HEADER: ""})
        assert resp.status == 401

    def test_the_key_goes_only_to_cluster_hosts(self):
        """Added per request by destination: the shared session also talks to
        engines on localhost and, through some routes, to the outside."""
        class _Node:
            def __init__(self, node_id, fabric_ip):
                self.node_id, self.fabric_ip = node_id, fabric_ip
                self.peer_ip, self.node_name, self.ib_ips = "", "", []

        class _Cluster:
            def members(self):
                return [_Node("me", "10.0.0.1"), _Node("peer", "10.0.0.2")]

        class _Config:
            node_id = "me"

        hosts = ck.cluster_hosts({"cluster_state": _Cluster(),
                                  "config": _Config()})
        assert hosts == {"10.0.0.2"}
        assert "huggingface.co" not in hosts

    def test_the_key_file_is_private(self, tmp_path):
        ck.cluster_key()
        assert oct((tmp_path / "cluster.key").stat().st_mode & 0o777) == "0o600"


class TestAPIKeysNoLongerLockTheDashboardOut:
    """K2. Keys used to guard every route; the dashboard never sent one, so
    switching them on returned 401 to the switch that turns them off."""

    @pytest_asyncio.fixture
    async def keyed(self, tmp_path):
        keys = AuthConfig()
        keys.__class__.save = lambda self: None     # no disk writes
        keys.enable()
        async with TestClient(TestServer(_app(tmp_path, api_keys=keys))) as c:
            yield c

    @pytest.mark.asyncio
    async def test_a_signed_in_dashboard_reaches_the_api(self, keyed):
        await _sign_in(keyed)
        assert (await keyed.get("/api/status")).status == 200

    @pytest.mark.asyncio
    async def test_and_its_own_chat_on_v1(self, keyed):
        await _sign_in(keyed)
        assert (await keyed.get("/v1/models")).status == 200

    @pytest.mark.asyncio
    async def test_an_outside_client_still_needs_a_key(self, keyed):
        assert (await keyed.get("/v1/models")).status == 401


class TestPasswordChangeRoute:
    @pytest.mark.asyncio
    async def test_it_changes_and_keeps_this_browser_signed_in(self, client):
        await _sign_in(client)
        origin = str(client.make_url("/")).rstrip("/")
        resp = await client.post("/api/auth/web/password", json={
            "current": "admin", "new": "correct horse", "all_nodes": False},
            headers={"Origin": origin})
        assert resp.status == 200
        assert (await client.get("/api/status")).status == 200

    @pytest.mark.asyncio
    async def test_the_sync_route_wants_the_cluster_key(self, client):
        # The one route that replaces the password without the old one.
        await _sign_in(client)
        origin = str(client.make_url("/")).rstrip("/")
        resp = await client.post("/api/auth/web/sync", json={
            "user": "x", "salt": "00", "password_hash": "00"},
            headers={"Origin": origin})
        assert resp.status == 403


class TestTheUI:
    SOURCE = None

    @classmethod
    def setup_class(cls):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "ainode" / "web"
        cls.SOURCE = (root / "static" / "js" / "app.js").read_text()
        cls.LOGIN = (root / "templates" / "login.html").read_text()

    def test_the_sign_in_page_uses_the_design_system(self):
        assert "/static/css/style.css" in self.LOGIN
        assert "logo-wordmark" in self.LOGIN
        assert "argentos.ai" in self.LOGIN

    def test_every_fetch_sends_you_to_sign_in_on_401(self):
        assert "watchForSignOut" in self.SOURCE

    def test_the_default_password_is_mentioned_until_changed(self):
        assert "renderDefaultPasswordBanner" in self.SOURCE

    def test_it_can_be_changed_in_the_ui(self):
        assert "/api/auth/web/password" in self.SOURCE
        assert "cfg-web-new" in self.SOURCE
