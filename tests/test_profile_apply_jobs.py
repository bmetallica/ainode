# ruff: noqa: F811  (the fixtures app/launches are imported and then requested by name)
"""Applying a profile as a job: progress, cancel, undo, measurements.

Phase 7 of wizzard.md.
"""

from __future__ import annotations

import asyncio

import pytest

import ainode.models.api_routes as api_routes
from ainode.profiles import apply as apply_module
from ainode.profiles import jobs as jobs_module
from ainode.profiles.store import Profile, ProfileStore
from tests.test_profile_apply import _Instance, _Manager  # noqa: F401
from tests.test_profile_apply import app, launches  # noqa: F401


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(api_routes, "save_instance_manifest", lambda app: None)

    async def _ready(port, timeout=0):
        return True

    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready)
    monkeypatch.setattr(jobs_module, "MEASURE_WAIT", 0.0)
    monkeypatch.setattr(apply_module, "capture_profile",
                        lambda app, name, desc="": Profile(name=name, entries=[
                            {"model": "was/running"}]))
    monkeypatch.setattr(jobs_module, "capture_profile", apply_module.capture_profile)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _finish(job):
    for _ in range(200):
        if job.state != "running":
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("the job never finished")


class TestTheJob:
    def test_every_entry_has_a_state(self, app, launches):
        async def go():
            job = await jobs_module.start_apply_job(
                app, Profile(name="p", entries=[{"model": "a/b"}, {"model": "c/d"}]))
            return await _finish(job)

        job = asyncio.run(go())
        data = job.to_dict()
        assert data["state"] == "done" and data["ok"]
        assert [e["state"] for e in data["entries"]] == ["ready", "ready"]
        assert data["can_restore"] is True

    def test_what_ran_before_is_captured(self, app, launches):
        async def go():
            job = await jobs_module.start_apply_job(
                app, Profile(name="p", entries=[{"model": "a/b"}]))
            await _finish(job)
            return job

        job = asyncio.run(go())
        assert [e.model for e in job.before.entries] == ["was/running"]

    def test_cancel_skips_what_has_not_started(self, app, launches, monkeypatch):
        started = []

        async def _slow_start(app, entry):
            started.append(entry.model)
            await asyncio.sleep(0.05)
            return apply_module.ApplyResult(entry.model, "launched", True, api_port=8001)

        monkeypatch.setattr(apply_module, "_start_llm_entry", _slow_start)

        async def go():
            job = await jobs_module.start_apply_job(
                app, Profile(name="p", entries=[{"model": "a/b"}, {"model": "c/d"},
                                                {"model": "e/f"}]))
            await asyncio.sleep(0.01)
            job.cancel_requested = True
            return await _finish(job)

        job = asyncio.run(go())
        states = [e["state"] for e in job.to_dict()["entries"]]
        assert job.state == "cancelled"
        assert states[0] == "ready" and states[1:] == ["skipped", "skipped"]
        assert started == ["a/b"]

    def test_one_job_at_a_time(self, app, launches, monkeypatch):
        async def _slow_start(app, entry):
            await asyncio.sleep(0.05)
            return apply_module.ApplyResult(entry.model, "launched", True)

        monkeypatch.setattr(apply_module, "_start_llm_entry", _slow_start)

        async def go():
            first = await jobs_module.start_apply_job(
                app, Profile(name="p", entries=[{"model": "a/b"}]))
            with pytest.raises(RuntimeError):
                await jobs_module.start_apply_job(
                    app, Profile(name="q", entries=[{"model": "c/d"}]))
            await _finish(first)

        asyncio.run(go())

    def test_what_it_took_is_written_into_the_profile(self, app, launches,
                                                      tmp_path, monkeypatch):
        store = ProfileStore(tmp_path / "p.json")
        profile = Profile(name="p", entries=[{"model": "a/b", "node_ids": ["head"]}])
        store.put(profile)
        app["profiles"] = store

        async def _gather(app):
            return {"a/b": [
                {"node_id": "head", "memory_gb": 81.4, "kv_tokens": 500000,
                 "last_ok": 9e12},
                {"node_id": "head", "memory_gb": 1.0, "last_ok": 1.0}]}

        import ainode.measure.api_routes as measure_routes

        monkeypatch.setattr(measure_routes, "gather_cluster_measurements", _gather)

        async def go():
            job = await jobs_module.start_apply_job(app, profile)
            return await _finish(job)

        asyncio.run(go())
        measured = store.get("p").measured
        assert measured["a/b@head"]["memory_gb"] == 81.4
        assert measured["a/b@head"]["kv_tokens"] == 500000


class TestTheRoutes:
    @pytest.mark.asyncio
    async def test_background_apply_and_the_job_routes(self, tmp_path, monkeypatch):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        from ainode.core.config import NodeConfig
        from ainode.profiles.api_routes import register_profile_routes

        async def _apply(app, profile, wait=True, progress=None):
            for entry in profile.entries:
                progress.entry(entry, "ready")
            return {"ok": True, "results": [], "stopped": ["old (s2)"]}

        monkeypatch.setattr(jobs_module, "apply_profile", _apply)
        web_app = web.Application()
        web_app["config"] = NodeConfig(node_id="head")
        store = ProfileStore(tmp_path / "p.json")
        store.put(Profile(name="P", entries=[{"model": "a/b"}]))
        web_app["profiles"] = store
        from ainode.measure.store import MeasurementStore

        web_app["measurement_store"] = MeasurementStore(tmp_path / "m.json")
        register_profile_routes(web_app)
        async with TestClient(TestServer(web_app)) as client:
            resp = await client.post("/api/profiles/P/apply", json={"background": True})
            assert resp.status == 202
            job_id = (await resp.json())["job"]["id"]
            for _ in range(100):
                data = await (await client.get("/api/profiles/jobs/current")).json()
                if data["job"]["state"] != "running":
                    break
                await asyncio.sleep(0.01)
            assert data["job"]["id"] == job_id and data["job"]["state"] == "done"
            assert data["job"]["stopped"] == ["old (s2)"]
            restore = await client.post(f"/api/profiles/jobs/{job_id}/restore")
            assert restore.status == 202
            assert (await restore.json())["job"]["restore_of"] == "P"
            assert (await client.get("/api/profiles/jobs/nope")).status == 404


class TestAnOpencodeConfigFromAProfile:
    def test_replicas_are_one_model_with_the_smaller_window(self, app):
        from ainode.clients.opencode import build_opencode_config_for_profile

        app["model_manager"] = None
        profile = Profile(name="p", entries=[
            {"model": "coder", "node_ids": ["s2"], "max_model_len": 131072,
             "extra_vllm_args": ["--max-num-seqs", "2"]},
            {"model": "coder", "node_ids": ["s3"], "max_model_len": 65536},
            {"model": "bge", "kind": "embedding", "node_ids": ["s3"]},
        ])
        out = build_opencode_config_for_profile(app, profile, "http://head:3000")
        models = out["config"]["provider"]["vllm"]["models"]
        assert list(models) == ["coder"]
        assert models["coder"]["limit"]["context"] < 65536
        assert out["profile"] == "p"
