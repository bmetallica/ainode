"""The three patterns every finding in update.md shared.

  * fetching data nobody needs, repeatedly;
  * fetching data once and never again;
  * showing this node's state where the cluster's was meant.

They were found in the Models page and the Server view. This file covers what
the sweep through the rest of the UI turned up — the largest of which was the
launch form, quietly walking the whole model store twice every five seconds
for as long as a browser tab stayed open.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class TestThePollDoesNotWalkTheDisk:
    def test_the_launch_list_is_not_refetched_every_five_seconds(self):
        # populateLaunchModels runs on every refresh() — every five seconds,
        # for as long as a tab is open — and fetched the catalog AND the
        # downloaded list each time. Both walk model directories on the
        # server: two full disk scans every five seconds per open tab.
        block = APP_JS.split("populateLaunchModels() {")[1].split("\n  },")[0]
        assert "fetchJSON('/api/models')" not in block
        assert "fetchJSON('/api/models/downloaded')" not in block
        assert "this.ensureModelLists()" in block

    def test_the_shared_lists_are_fetched_on_a_ttl(self):
        block = APP_JS.split("ensureModelLists() {")[1].split("\n  },")[0]
        assert "this._stale('modelLists')" in block

    def test_a_failed_poll_does_not_empty_the_list(self):
        # The launch form losing every model because one poll timed out would
        # read as the models having been unloaded.
        block = APP_JS.split("ensureModelLists() {")[1].split("\n  },")[0]
        assert "(self._modelLists || {}).catalog" in block
        assert "return self._modelLists ||" in block

    def test_an_action_still_shows_its_own_result_at_once(self):
        block = APP_JS.split("invalidate() {")[1].split("\n  },")[0]
        assert "this._modelLists = null;" in block


class TestWhereTheWeightsAre:
    def test_the_card_says_which_nodes_hold_it(self):
        # "On disk" meant this node's disk. The page lists the whole cluster
        # now, and a badge that does not say which node would be the same
        # half-truth in the other direction.
        assert "nodesHolding(model)" in APP_JS
        assert "whereBadge" in APP_JS

    def test_delete_reaches_the_disk_the_weights_are_on(self):
        # A Delete that could only reach the head would report "not
        # downloaded" about a model the page plainly shows as present.
        assert "fetch('/api/cluster/delete-repo'" in APP_JS
        assert "data-nodes=" in APP_JS

    def test_the_confirmation_names_the_nodes(self):
        block = APP_JS.split("confirmDeleteModel(hfRepo, nodeIds) {")[1][:1600]
        assert "nodeIds || []" in block

    def test_the_route_exists_and_dispatches(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/cluster/delete-repo" in paths

    def test_a_delete_without_a_node_still_means_here(self):
        # Back-compat: the old button sent no node and meant this one.
        import inspect

        from ainode.api import server

        source = inspect.getsource(server._cluster_dispatch)
        assert 'path.endswith("/delete-repo")' in source


class _Req:
    def __init__(self, app, body):
        self.app = app
        self._body = body

    async def json(self):
        return self._body


class TestTheDispatchPicksTheRightHandler:
    def test_delete_repo_goes_to_the_delete_handler(self, monkeypatch):
        from ainode.api import server
        from ainode.models import api_routes

        seen = {}

        async def _fake_delete(request):
            seen["called"] = (await request.json()).get("hf_repo")
            from aiohttp import web

            return web.json_response({"status": "deleted"})

        monkeypatch.setattr(api_routes, "handle_delete_repo", _fake_delete)

        class _Config:
            node_id = "head"

        app = {"config": _Config(), "cluster_state": None}
        resp = asyncio.run(server.handle_cluster_delete_repo(
            _Req(app, {"hf_repo": "org/m"})))
        assert seen["called"] == "org/m"
        assert json.loads(resp.body)["status"] == "deleted"
