"""The pool the engine is given, not the memory that happens to be free.

Reported from the cluster, Smaug-Flash at TP=2 driven from opencode:

    das modell hört aber immer wieder auf zu arbeiten, es sieht sehr so aus als
    würde er in einen cache overflow laufen obwohl die einstellungen aus ainode
    zu passen scheinen

They did seem to. Three numbers were involved and no two of them agreed:

    opencode.json   "context": 608512    (from a 684,032 window)
    Load tab        Context length 180.2K tokens
    launched with   --max-model-len 684032

The planner sized the KV cache from FREE memory while the engine is given a
FRACTION OF THE TOTAL. Those two agreed by construction, because the planner
derived its own gpu_memory_utilization from its own claim — and they stopped
agreeing the moment anything else set the fraction. Here the admission gate's
utilization cap lowered it to 0.66 on a node that had filled up since the plan
was made:

    plan:  1,048,576 context, 2,533,712 tokens of cache
    0.66 x 128 GB = 84.5 GB of pool, minus 87.6 GB of weights = nothing at all

vLLM started anyway and spent its time preempting and recomputing, which from a
client looks like a model that keeps stopping rather than like a memory error.
"""

from __future__ import annotations

import pytest

from ainode.planner.compute import NodeBudget, plan_for
from ainode.planner.facts import ModelFacts


def _smaug():
    return ModelFacts(
        repo="abacusai/Smaug-Flash", num_layers=43, attention_layers=43,
        kv_lora_rank=512, qk_rope_head_dim=64, num_attention_heads=64,
        num_kv_heads=1, head_dim=512, torch_dtype="bfloat16",
        weight_bytes=166_900_000_000, max_position_embeddings=1048576,
        num_experts=256)


NODES = [NodeBudget(node_id="a", name="13e1", total_gb=128.0, free_gb=126.0),
         NodeBudget(node_id="b", name="1432", total_gb=128.0, free_gb=126.0)]


def _plan(gmu=0.0, **kw):
    kw.setdefault("kv_cache_dtype", "fp8")
    kw.setdefault("concurrency", 2)
    return plan_for(_smaug(), NODES, recommended_gmu=gmu, **kw)


class TestTheCeilingBoundsTheCache:
    def test_a_lower_fraction_means_less_cache(self):
        assert _plan(0.85).kv_gb < _plan(0.95).kv_gb

    def test_and_therefore_a_shorter_window(self):
        assert _plan(0.85).max_model_len < _plan(0.95).max_model_len

    @pytest.mark.parametrize("gmu", [0.95, 0.85, 0.80])
    def test_the_cache_fits_inside_the_pool(self, gmu):
        plan = _plan(gmu)
        pool = gmu * 128.0
        per_node = (plan.weights_per_node_gb + plan.overhead_per_node_gb
                    + plan.kv_gb / max(1, len(plan.node_ids)))
        assert per_node <= pool + 0.5

    def test_the_reported_case_is_refused_outright(self):
        """0.66 of 128 is 84.5 GB, and the weights alone are 87.6. There is no
        window to offer, and offering one is what produced the stall."""
        plan = _plan(0.66)
        assert plan.fits is False
        assert plan.max_model_len == 0

    def test_no_ceiling_still_plans_from_free_memory(self):
        # Where nothing imposes a fraction, the planner derives its own from
        # its claim and the two agree — that path is unchanged.
        assert _plan(0.0).fits is True


class TestTheOldArithmeticIsGone:
    def test_the_cache_is_not_what_free_memory_would_give(self):
        """Free memory here is 126 GB, usable 122 after the system reserve. A
        cache sized from that at 0.73 would be 51 GB where the pool has 2.8."""
        plan = _plan(0.73)
        if plan.fits:
            assert plan.kv_gb < 20

    def test_a_ceiling_above_what_is_free_changes_nothing(self):
        # min(usable, ceiling x total): a generous fraction cannot invent
        # memory that is not free.
        tight = [NodeBudget(node_id="a", name="a", total_gb=128.0, free_gb=95.0),
                 NodeBudget(node_id="b", name="b", total_gb=128.0, free_gb=95.0)]
        loose = plan_for(_smaug(), tight, kv_cache_dtype="fp8", concurrency=2,
                         recommended_gmu=0.99)
        none = plan_for(_smaug(), tight, kv_cache_dtype="fp8", concurrency=2)
        assert loose.kv_gb == none.kv_gb


class TestTheLaunchBringsTheContextDownWithTheFraction:
    def test_the_clamp_exists_and_runs_after_the_cap(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "_clamp_context_to_utilization" in source
        assert source.index("cap_utilization") < \
            source.index("_clamp_context_to_utilization")

    def test_it_only_ever_lowers(self):
        import inspect

        from ainode.engine.sharding_routes import \
            _clamp_context_to_utilization

        source = inspect.getsource(_clamp_context_to_utilization)
        assert "if replanned.max_model_len >= wanted:" in source

    def test_it_says_what_it_did_and_why(self):
        import inspect

        from ainode.engine.sharding_routes import \
            _clamp_context_to_utilization

        source = inspect.getsource(_clamp_context_to_utilization)
        assert "context lowered from" in source
        assert "preempt every request" in source

    def test_a_failure_to_re_plan_leaves_the_context_alone(self):
        # The behaviour before this existed. Refusals belong to the admission
        # gate, not to a clamp.
        import inspect

        from ainode.engine.sharding_routes import \
            _clamp_context_to_utilization

        source = inspect.getsource(_clamp_context_to_utilization)
        assert 'return ""' in source


class TestTheLoadTabReadsTheInstance:
    def test_it_prefers_the_backend_config(self):
        import inspect

        from ainode.api import server_routes

        source = inspect.getsource(server_routes.handle_server_status)
        assert "_backend_config(manager, mid) or config" in source

    def test_the_reason_is_recorded(self):
        from pathlib import Path

        import ainode.api.server_routes as server_routes

        source = Path(server_routes.__file__).read_text()
        # A distributed launch never writes config.max_model_len, so the shared
        # config keeps a previous solo load's value.
        assert "never config.max_model_len" in source
