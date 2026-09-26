"""How much of the node will this take, and how much of that will it use.

Asked from the cluster:

    was mir noch fehlt ist beim modelladen über die seitenleiste, ist eine
    prognose wie viel vram das modell belegen wird, welche sich abhängig zu den
    einstellungen live aktualisiert

Two numbers, because either one alone is the misleading half.

The engine TAKES gpu_memory_utilization x the node's TOTAL — that fraction is
of the total, not of what is free — and vLLM fills it with cache blocks whether
the configured context needs them or not. That is what free(1) shows and what
the memory guard watches approach its line.

The launch USES the weights, the engine, and cache for
max_model_len x concurrency. On unified memory the difference between the two is
real memory held and not used, and it is the thing an operator reaches for the
wrong lever about: raising the fraction does not make the model bigger, it makes
the pool bigger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.planner.compute import (COMM_OVERHEAD_GB, ENGINE_OVERHEAD_GB,
                                    NodeBudget, plan_for)
from ainode.planner.facts import ModelFacts

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
          / "js" / "app.js").read_text()
CSS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
       / "css" / "style.css").read_text()


def _facts(weight_bytes=132_700_000_000):
    return ModelFacts(repo="org/model", num_layers=48, attention_layers=12,
                      num_kv_heads=2, head_dim=256, num_attention_heads=24,
                      torch_dtype="bfloat16", weight_bytes=weight_bytes,
                      max_position_embeddings=262144, num_experts=512)


NODES = [NodeBudget(node_id="a", name="1432", total_gb=128.0, free_gb=126.4),
         NodeBudget(node_id="b", name="659b", total_gb=128.0, free_gb=126.4)]


def _plan(**kw):
    kw.setdefault("kv_cache_dtype", "auto")
    return plan_for(_facts(), NODES, **kw)


class TestWhatTheEngineTakes:
    def test_reserved_is_the_fraction_of_the_total(self):
        plan = _plan(concurrency=2)
        assert plan.reserved_per_node_gb == pytest.approx(
            plan.gpu_memory_utilization * plan.node_total_gb, abs=0.05)

    def test_the_total_is_the_tightest_nodes(self):
        # Every rank gets the same share; the tightest node is what bounds it.
        assert _plan(concurrency=2).node_total_gb == 128.0

    def test_it_is_not_derived_from_what_is_free(self):
        """The fraction is of the total. Reading it as a share of free memory
        is how a launch gets planned into memory that is already in use."""
        tight = plan_for(_facts(), [
            NodeBudget(node_id="a", name="a", total_gb=128.0, free_gb=100.0),
            NodeBudget(node_id="b", name="b", total_gb=128.0, free_gb=100.0),
        ], kv_cache_dtype="auto", concurrency=2)
        assert tight.node_total_gb == 128.0


class TestWhatTheLaunchUses:
    def test_it_is_the_three_parts_added_up(self):
        plan = _plan(concurrency=4)
        assert plan.needed_per_node_gb == pytest.approx(
            plan.weights_per_node_gb + plan.overhead_per_node_gb
            + plan.cache_used_per_node_gb, abs=0.05)

    def test_the_overhead_includes_the_rank_buffers(self):
        assert _plan(concurrency=2).overhead_per_node_gb == pytest.approx(
            ENGINE_OVERHEAD_GB + COMM_OVERHEAD_GB, abs=0.01)

    def test_a_solo_plan_pays_no_rank_buffers(self):
        solo = plan_for(_facts(weight_bytes=40_000_000_000), NODES[:1],
                        kv_cache_dtype="auto", concurrency=1)
        assert solo.overhead_per_node_gb == pytest.approx(ENGINE_OVERHEAD_GB,
                                                          abs=0.01)

    def test_more_concurrency_uses_more(self):
        assert _plan(concurrency=8).needed_per_node_gb > \
            _plan(concurrency=2).needed_per_node_gb

    def test_a_shorter_window_uses_less(self):
        long = _plan(max_model_len=262144, concurrency=4)
        short = _plan(max_model_len=32768, concurrency=4)
        assert short.cache_used_per_node_gb < long.cache_used_per_node_gb

    def test_it_uses_the_concurrency_asked_for_not_the_one_that_fits(self):
        """plan.concurrent_requests answers "how many could run". Using it here
        would report the pool as fully used on every plan, which is the one
        thing this is meant to show is not true."""
        plan = _plan(concurrency=1)
        assert plan.concurrent_requests > 1          # many would fit
        assert plan.needed_per_node_gb < plan.reserved_per_node_gb


class TestTheGapIsThePoint:
    def test_a_low_concurrency_leaves_the_pool_mostly_idle(self):
        plan = _plan(concurrency=1)
        idle = plan.reserved_per_node_gb - plan.needed_per_node_gb
        assert idle > 20

    def test_filling_the_concurrency_closes_it(self):
        plan = _plan(concurrency=15)
        idle = plan.reserved_per_node_gb - plan.needed_per_node_gb
        assert idle < 5

    def test_needed_never_exceeds_the_node(self):
        for concurrency in (1, 2, 4, 8, 15):
            plan = _plan(concurrency=concurrency)
            assert plan.needed_per_node_gb <= plan.node_total_gb


class TestItIsCarriedToTheBrowser:
    @pytest.mark.parametrize("field", [
        "node_total_gb", "reserved_per_node_gb", "needed_per_node_gb",
        "cache_used_per_node_gb", "overhead_per_node_gb"])
    def test_the_payload_has_it(self, field):
        assert field in _plan(concurrency=2).to_dict()

    def test_the_sidebar_draws_it(self):
        assert "renderOccupancy" in APP_JS
        assert "Occupies" in APP_JS

    def test_it_is_drawn_inside_the_plan_hint(self):
        # Which is what makes it live: the hint is redrawn on every re-plan,
        # and every advanced field re-plans.
        assert "line + forecast + measured + warn" in APP_JS

    def test_it_names_the_idle_share_and_what_to_do(self):
        assert "reserved and will not be used" in APP_JS
        assert "lower the memory fraction" in APP_JS

    def test_the_bar_has_a_segment_per_part(self):
        for part in ("weights", "engine", "cache", "idle"):
            assert f".occupancy-seg.{part}" in CSS, part
