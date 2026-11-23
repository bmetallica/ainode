"""Parallel slots per model on engine/<modell>: total, used, free."""

from __future__ import annotations

import asyncio

from ainode.telemetry.engine_metrics import collect, with_slots

SAMPLE = "vllm:num_requests_running 3\nvllm:num_requests_waiting 0\n"


class _Resp:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return SAMPLE


class _Session:
    def get(self, url, timeout=None):
        return _Resp()


class _Cfg:
    def __init__(self, args):
        self.extra_vllm_args = args


class _Instance:
    def __init__(self, model, port, args):
        self.record = type("R", (), {"model": model, "api_port": port})()
        self.backend = type("B", (), {"config": _Cfg(args)})()


class _Instances:
    def __init__(self, *items):
        self._items = items

    def instances(self):
        return list(self._items)


class _Info:
    extra_vllm_args = ["--max-num-seqs", "16"]


class _Catalog:
    def _find_catalog_by_hf_repo(self, repo):
        return _Info() if repo == "org/recipe" else None


def test_free_is_total_minus_running():
    out = with_slots({"requests_running": 3}, 16)
    assert (out["slots_total"], out["slots_used"], out["slots_free"]) == (16, 3, 13)


def test_never_negative():
    assert with_slots({"requests_running": 20}, 16)["slots_free"] == 0


def test_unknown_capacity_adds_nothing():
    # vLLM does not export --max-num-seqs; a guess would describe another engine.
    assert with_slots({"requests_running": 3}, 0) == {"requests_running": 3}
    assert with_slots({}, 16) == {}


def test_not_for_an_image_engine():
    metrics = {"requests_running": 1, "images_generated_total": 4}
    assert with_slots(metrics, 16) == metrics


def test_launch_args_then_recipe():
    app = {"client_session": _Session(), "model_manager": _Catalog(),
           "instances": _Instances(
               _Instance("org/typed", 8000, ["--max-num-seqs=8"]),
               _Instance("org/recipe", 8001, []),
               _Instance("org/none", 8002, []))}
    out = asyncio.run(collect(app))
    assert out["org_typed"]["slots_free"] == 5
    assert out["org_recipe"]["slots_total"] == 16
    assert "slots_total" not in out["org_none"]


def test_the_prefix_cache_hit_rate():
    from ainode.telemetry.engine_metrics import distil

    out = distil("vllm:prefix_cache_queries_total 7440000\n"
                 "vllm:prefix_cache_hits_total 6810000\n")
    assert out["prefix_cache_hit_rate"] == 0.915
    assert "prefix_cache_hit_rate" not in distil("vllm:prefix_cache_queries_total 0\n")
