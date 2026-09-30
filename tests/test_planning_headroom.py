"""The planning headroom, set once for the cluster from the Memory Guard page.

Asked for: "punkt 4, ja bau das bitte ins UI ein" — after the reserve
arithmetic had been laid out: the guard's lines are the brake, the 6% on top
is only what a plan keeps back.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from ainode.core.config import NodeConfig
from ainode.planner.compute import plan_headroom_gb


class TestTheFigure:
    def test_automatic_is_six_percent(self):
        assert plan_headroom_gb(130.7) == 7.8
        assert plan_headroom_gb(130.7, None) == 7.8

    def test_configured_wins(self):
        assert plan_headroom_gb(130.7, 3.0) == 3.0
        assert plan_headroom_gb(130.7, 0.0) == 0.0

    def test_the_planner_holds_back_the_configured_figure(self):
        from ainode.planner.api_routes import held_back_gb

        auto = held_back_gb({"config": NodeConfig()}, 130.7)
        set3 = held_back_gb({"config": NodeConfig(plan_headroom_gb=3.0)}, 130.7)
        assert auto - set3 == pytest.approx(4.8, abs=0.05)


class _Guard:
    enabled = True
    warn_mb = 8192.0
    critical_mb = 4096.0

    def read(self):
        class _R:
            def to_dict(self):
                return {"total_mb": 124650, "available_mb": 100000, "warn_mb": 8192,
                        "critical_mb": 4096}
        return _R()

    def configure(self, **kw):
        self.configured = kw


class TestTheRoute:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        from ainode.safety.api_routes import register_safety_routes

        config = NodeConfig()
        monkeypatch.setattr(NodeConfig, "save", lambda self: None)
        app = web.Application()
        app["config"] = config
        app["memory_guard"] = _Guard()
        register_safety_routes(app)
        return app

    @pytest.mark.asyncio
    async def test_it_is_read_with_the_guard(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        data = await (await client.get("/api/safety/memory")).json()
        assert data["plan_headroom_gb"] is None
        assert data["plan_headroom_effective_gb"] == 7.8

    @pytest.mark.asyncio
    async def test_it_is_set_and_cleared(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.put("/api/safety/memory", json={"plan_headroom_gb": 3})
        assert resp.status == 200
        assert app["config"].plan_headroom_gb == 3.0
        assert (await resp.json())["plan_headroom_effective_gb"] == 3.0
        resp = await client.put("/api/safety/memory", json={"plan_headroom_gb": None})
        assert app["config"].plan_headroom_gb == -1.0

    @pytest.mark.asyncio
    async def test_setting_it_leaves_the_guard_alone(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        await client.put("/api/safety/memory", json={"plan_headroom_gb": 3})
        assert app["memory_guard"].configured == {"warn_gb": None, "critical_gb": None,
                                                  "enabled": None}

    @pytest.mark.asyncio
    async def test_nonsense_is_refused(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.put("/api/safety/memory", json={"plan_headroom_gb": 99})
        assert resp.status == 400


def test_the_page_offers_it_for_all_nodes():
    from pathlib import Path

    js = Path("ainode/web/static/js/views/config.js").read_text()
    assert 'id="mem-headroom"' in js
    assert "all: true, plan_headroom_gb:" in js
