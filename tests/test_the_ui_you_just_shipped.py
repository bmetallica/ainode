"""A new API behind the previous UI is the worst of the two possible wrongs.

Reported from the cluster, after an update that landed ten merged branches:

    wenn ich werte in den feldern ändere wird der rest aber nach wie vor nicht
    live angepasst

Everything else in that report had arrived — the decimal sizes, the 18 GB
cache, the tightest-node note, "-0.6 GB vs the plan". All of it server-side.
The two changes that live in app.js had not, because the template asked for
``/static/js/app.js`` with no version on it. aiohttp sends Last-Modified and
answers a conditional request, but a browser is not obliged to make one: with
no Cache-Control it may apply heuristic freshness — a tenth of the file's age —
and simply not ask.

The numbers looked new and the behaviour was old, which is the failure mode
that wastes an afternoon looking at the wrong code.
"""

from __future__ import annotations

import re

import pytest

from ainode.web.serve import (STATIC_DIR, VERSIONED, asset_token,
                              get_index_html, get_onboarding_html)


class TestEveryAssetCarriesAVersion:
    @pytest.mark.parametrize("asset", VERSIONED)
    def test_the_dashboard_stamps_it(self, asset):
        html = get_index_html()
        assert f'"{asset}?v=' in html
        assert f'"{asset}"' not in html

    def test_the_onboarding_page_too(self):
        # Same trap, and the page an operator sees first.
        html = get_onboarding_html()
        for asset in VERSIONED:
            assert f'"{asset}"' not in html

    def test_nothing_unversioned_is_left(self):
        html = get_index_html()
        bare = re.findall(r'"(/static/(?:js|css)/[\w.]+)"', html)
        assert bare == []

    def test_the_token_is_url_safe(self):
        for asset in VERSIONED:
            assert re.fullmatch(r"[0-9a-z]+", asset_token(asset))


class TestTheTokenTracksTheFile:
    def test_it_changes_when_the_file_does(self, monkeypatch, tmp_path):
        from ainode.web import serve

        target = tmp_path / "js"
        target.mkdir()
        (target / "app.js").write_text("// one")
        monkeypatch.setattr(serve, "STATIC_DIR", tmp_path)
        first = serve.asset_token("/static/js/app.js")

        import os

        os.utime(target / "app.js", (0, 1_700_000_000))
        second = serve.asset_token("/static/js/app.js")
        assert first != second

    def test_it_is_stable_when_the_file_is_not(self):
        assert asset_token(VERSIONED[0]) == asset_token(VERSIONED[0])

    def test_a_missing_file_does_not_raise(self, monkeypatch, tmp_path):
        from ainode.web import serve

        monkeypatch.setattr(serve, "STATIC_DIR", tmp_path)
        # A wrong token beats an exception on the one page that would explain
        # why the UI is stale.
        assert asset_token("/static/js/nothing.js")

    def test_the_real_assets_exist(self):
        for asset in VERSIONED:
            assert (STATIC_DIR / asset.split("/static/", 1)[1]).is_file(), asset


class TestTheBrowserIsAskedToRevalidate:
    @pytest.mark.asyncio
    async def test_static_answers_with_no_cache(self, tmp_path):
        import aiohttp
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.api.server import _revalidate_static

        static = tmp_path / "js"
        static.mkdir()
        (static / "app.js").write_text("// x")
        app = web.Application(middlewares=[_revalidate_static])
        app.router.add_static("/static", tmp_path)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/static/js/app.js")
            assert resp.status == 200
            # Not "do not store" — revalidate. The usual answer is a 304 and
            # nothing is re-sent.
            assert resp.headers["Cache-Control"] == "no-cache"
        del aiohttp

    @pytest.mark.asyncio
    async def test_other_routes_are_left_alone(self, tmp_path):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.api.server import _revalidate_static

        async def _handler(request):
            return web.json_response({"ok": True})

        app = web.Application(middlewares=[_revalidate_static])
        app.router.add_get("/api/thing", _handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/api/thing")
            assert "Cache-Control" not in resp.headers

    def test_the_middleware_is_registered_with_the_static_route(self):
        import inspect

        from ainode.api import server

        source = inspect.getsource(server.create_app)
        assert "_revalidate_static" in source
        assert source.index("add_static") < source.index("_revalidate_static")
