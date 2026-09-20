"""The Models page: what it scans, how often, and whose disk.

Three findings, one page. It walked every model directory on every request,
it fetched its two lists exactly once per page load, and it showed only the
head's disk while the panel beside it showed the whole cluster.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ainode.models.registry import ModelManager

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class TestTheSizeCache:
    def _tree(self, tmp_path, size=1024):
        (tmp_path / "model.safetensors").write_bytes(b"x" * size)
        return tmp_path

    def test_a_directory_is_walked_once(self, tmp_path, monkeypatch):
        # It was walked per catalog entry per request, twice per model in
        # list_downloaded, from six call sites in the UI: hundreds of
        # gigabytes of stat() per page view, which also evicted the page cache
        # the next launch wanted.
        ModelManager.forget_size()
        tree = self._tree(tmp_path)
        walks = {"n": 0}
        real = Path.rglob

        def counting(self, pattern):
            walks["n"] += 1
            return real(self, pattern)

        monkeypatch.setattr(Path, "rglob", counting)
        assert ModelManager._dir_size_gb(tree) > 0
        assert ModelManager._dir_size_gb(tree) > 0
        assert ModelManager._dir_size_gb(tree) > 0
        assert walks["n"] == 1

    def test_the_answer_is_the_same(self, tmp_path):
        ModelManager.forget_size()
        tree = self._tree(tmp_path, size=4096)
        first = ModelManager._dir_size_gb(tree)
        assert ModelManager._dir_size_gb(tree) == first
        assert first == 4096 / (1024 ** 3)

    def test_an_entry_also_expires_on_time(self, tmp_path, monkeypatch):
        # mtime is a cheap hint, not a guarantee: a file rewritten inside an
        # existing tree can leave the parent untouched, and two writes inside
        # one filesystem clock tick are indistinguishable. The TTL is the
        # backstop for everything that changes a tree without going through
        # the download and delete paths.
        ModelManager.forget_size()
        tree = self._tree(tmp_path)
        before = ModelManager._dir_size_gb(tree)
        (tree / "second.safetensors").write_bytes(b"y" * 8192)
        monkeypatch.setattr(ModelManager, "_SIZE_TTL_SECONDS", -1)
        assert ModelManager._dir_size_gb(tree) > before

    def test_forgetting_covers_the_subtree(self, tmp_path):
        # A file rewritten inside an existing tree leaves the parent's mtime
        # untouched on some filesystems, so a download and a delete say so
        # explicitly rather than trusting it.
        ModelManager.forget_size()
        tree = self._tree(tmp_path)
        ModelManager._dir_size_gb(tree)
        ModelManager.forget_size(tree)
        assert str(tree) not in ModelManager._SIZE_CACHE

    def test_a_missing_directory_is_zero_not_a_crash(self, tmp_path):
        assert ModelManager._dir_size_gb(tmp_path / "gone") == 0.0

    def test_the_download_path_drops_it(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes._run_download_repo)
        assert "_forget_size(manager, target)" in source

    def test_dropping_it_never_fails_a_download(self):
        # The download path is handed stand-ins by callers that only need part
        # of the interface. A cache invalidation must not be why a download
        # reports failure.
        from ainode.models.api_routes import _forget_size

        _forget_size(object(), Path("/nowhere"))


class _Node:
    def __init__(self, node_id, fabric_ip="10.0.0.2"):
        self.node_id = node_id
        self.node_name = node_id.upper()
        self.status = "online"
        self.fabric_ip = fabric_ip
        self.web_port = 3000


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


class _Config:
    node_id = "head"
    node_name = "HEAD"


class _Manager:
    def __init__(self, models):
        self._models = models

    def list_downloaded(self):
        return self._models


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload


class _Session:
    def __init__(self, by_host):
        self._by_host = by_host
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        for host, payload in self._by_host.items():
            if host in url:
                return _Resp(payload)
        raise OSError("unreachable")


class _Req:
    def __init__(self, app):
        self.app = app


def _cluster_app(peer_payload=None, peer_ip="10.0.0.2"):
    return {
        "config": _Config(),
        "cluster_state": _Cluster([_Node("head", "10.0.0.1"),
                                   _Node("n3", peer_ip)]),
        "model_manager": _Manager([
            {"hf_repo": "org/here", "name": "Here", "local_size_gb": 12.0}]),
        "client_session": _Session({peer_ip: peer_payload or {"models": []}}),
    }


class TestTheClusterView:
    def test_a_model_on_a_peer_is_listed(self):
        # The page scanned this node's disk while the panel beside it showed
        # what the whole cluster was serving. A model downloaded to node 3 and
        # running there appeared in one and not the other.
        from ainode.api.server import handle_cluster_models

        app = _cluster_app({"models": [
            {"hf_repo": "org/there", "name": "There", "local_size_gb": 40.0}]})
        body = json.loads(asyncio.run(handle_cluster_models(_Req(app))).body)
        repos = [m["hf_repo"] for m in body["models"]]
        assert repos == ["org/here", "org/there"]

    def test_it_says_which_nodes_have_it(self):
        from ainode.api.server import handle_cluster_models

        app = _cluster_app({"models": [{"hf_repo": "org/here"}]})
        body = json.loads(asyncio.run(handle_cluster_models(_Req(app))).body)
        row = next(m for m in body["models"] if m["hf_repo"] == "org/here")
        assert sorted(row["nodes"]) == ["head", "n3"]

    def test_the_largest_copy_decides_the_size(self):
        # A partial mirror on one node must not make the model look smaller
        # than it is.
        from ainode.api.server import handle_cluster_models

        app = _cluster_app({"models": [
            {"hf_repo": "org/here", "local_size_gb": 0.4}]})
        body = json.loads(asyncio.run(handle_cluster_models(_Req(app))).body)
        row = next(m for m in body["models"] if m["hf_repo"] == "org/here")
        assert row["size_gb"] == 12.0

    def test_an_unreachable_peer_costs_its_row_not_the_page(self):
        from ainode.api.server import handle_cluster_models

        app = _cluster_app(peer_ip="10.9.9.9")
        app["client_session"] = _Session({})   # every get() raises
        body = json.loads(asyncio.run(handle_cluster_models(_Req(app))).body)
        assert [m["hf_repo"] for m in body["models"]] == ["org/here"]

    def test_the_route_exists(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/cluster/models" in paths


class TestThePageRefreshes:
    def test_neither_list_is_fetched_once_and_forgotten(self):
        # The exact shape of the bug: `if (!this.state.catalog)` meant a
        # download that finished, a model deleted or one launched afterwards
        # never appeared until the browser was reloaded.
        assert "if (!this.state.catalog) {" not in APP_JS
        assert "if (!this.state.downloadedModels) {" not in APP_JS
        assert "this._stale('catalog')" in APP_JS
        assert "this._stale('downloadedModels')" in APP_JS

    def test_the_page_asks_the_cluster_not_the_local_disk(self):
        assert "fetch('/api/cluster/models')" in APP_JS

    def test_an_action_makes_its_own_result_visible_at_once(self):
        # Waiting out the TTL after a delete shows the model as still there
        # for up to a minute after its own delete reported success.
        for marker in ("self.invalidate();", "this.invalidate();"):
            assert marker in APP_JS
        assert "invalidate() {" in APP_JS
