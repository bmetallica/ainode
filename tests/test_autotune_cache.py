"""Every rank of a distributed launch starts with the same FlashInfer tuning
results (engine/autotune_cache.py).

The Qwen3.8-Flash-Next launch that hung: only the leading node kept
autotune_configs.json from the first launch, the second launch found it on
rank 0 and not on rank 1, and rank 1 waited thirty minutes in the tuner's
all_reduce.
"""

from __future__ import annotations

import base64
import json

import pytest
from aiohttp import web

from ainode.engine import autotune_cache


def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


class TestTheFiles:
    def test_a_snapshot_carries_every_file(self, tmp_path):
        _tree(tmp_path, {"0.7.0/121a/h/autotune_configs.json": '{"a": 1}'})
        snap = autotune_cache.snapshot(tmp_path)
        assert list(snap) == ["0.7.0/121a/h/autotune_configs.json"]
        assert base64.b64decode(snap["0.7.0/121a/h/autotune_configs.json"]) == b'{"a": 1}'

    def test_replace_leaves_exactly_what_was_sent(self, tmp_path):
        _tree(tmp_path, {"old/stale.json": "x"})
        autotune_cache.replace({"new/a.json": base64.b64encode(b"1").decode()}, tmp_path)
        assert sorted(p.relative_to(tmp_path).as_posix()
                      for p in tmp_path.rglob("*") if p.is_file()) == ["new/a.json"]

    def test_nothing_sent_is_an_empty_directory(self, tmp_path):
        # The leader has no results: every rank tunes afresh, together.
        _tree(tmp_path, {"h/autotune_configs.json": "{}"})
        autotune_cache.replace({}, tmp_path)
        assert not any(p.is_file() for p in tmp_path.rglob("*"))

    @pytest.mark.parametrize("rel", ["../evil", "/etc/passwd", "a/../../b", ""])
    def test_a_path_out_of_the_directory_is_refused(self, tmp_path, rel):
        with pytest.raises(ValueError):
            autotune_cache.replace({rel: base64.b64encode(b"x").decode()}, tmp_path)


class TestTheRoute:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.auth.cluster_key.cluster_key", lambda create=False: "k")
        monkeypatch.setattr(autotune_cache, "autotune_dir", lambda: tmp_path / "at")
        app = web.Application()
        autotune_cache.register_autotune_routes(app)
        return app

    @pytest.mark.asyncio
    async def test_the_leader_can_set_it(self, app, aiohttp_client, tmp_path):
        client = await aiohttp_client(app)
        resp = await client.put("/api/engine/autotune-cache", headers={"X-AINode-Cluster-Key": "k"},
                                json={"files": {"h/a.json": base64.b64encode(b"{}").decode()}})
        assert resp.status == 200
        assert (tmp_path / "at" / "h" / "a.json").read_bytes() == b"{}"

    @pytest.mark.asyncio
    async def test_nobody_else_can(self, app, aiohttp_client):
        client = await aiohttp_client(app)
        resp = await client.put("/api/engine/autotune-cache", json={"files": {}})
        assert resp.status == 403


class TestTheLaunch:
    def test_every_peer_gets_the_leaders_results(self, monkeypatch):
        sent = []

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _urlopen(request, timeout=0):
            sent.append((request.full_url, json.loads(request.data)))
            return _Resp()

        monkeypatch.setattr("urllib.request.urlopen", _urlopen)
        failed = autotune_cache.push_to_peers(["10.0.0.2", "10.0.0.3"], 3000,
                                              files={"h/a.json": "e30="})
        assert failed == []
        assert [u for u, _ in sent] == ["http://10.0.0.2:3000/api/engine/autotune-cache",
                                        "http://10.0.0.3:3000/api/engine/autotune-cache"]
        assert sent[0][1] == {"files": {"h/a.json": "e30="}}

    def test_an_unreachable_peer_is_reported_not_fatal(self, monkeypatch):
        def _urlopen(request, timeout=0):
            raise OSError("no route")

        monkeypatch.setattr("urllib.request.urlopen", _urlopen)
        assert autotune_cache.push_to_peers(["10.0.0.9"], files={}) == ["10.0.0.9"]

    def test_the_distributed_launch_aligns_before_it_starts(self):
        import inspect

        from ainode.engine.backends.eugr import EugrBackend

        source = inspect.getsource(EugrBackend.start_distributed)
        assert source.index("_distribute_model_to_peers()") < source.index(
            "_align_autotune_cache()") < source.index("preparing the cluster launch")
