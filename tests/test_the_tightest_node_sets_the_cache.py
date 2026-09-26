"""One node's spare memory is not the cluster's headroom.

Asked from the cluster, looking at two nodes running one model at TP=2:

    auf node1 habe ich 91% vram voll und auf node2 72% — dann müsste ich den
    cache ja noch größer skalieren können

Reasonable, and no. gpu_memory_utilization gives every rank the same fraction
of its OWN total, and all ranks hold the SAME number of KV blocks — vLLM has
no per-rank cache size. So the cache is the tightest rank's capacity times the
rank count, and the roomier node's spare memory cannot be reached by raising
anything: raising the fraction pushes the tight node into the memory guard and
leaves the cache where it was.

The arithmetic has always used min(usable) and never said so, which left the
free memory on the other node looking like headroom.
"""

from __future__ import annotations

import pytest

from ainode.planner.compute import (ASYMMETRY_GB, NodeBudget, _asymmetry,
                                    _evaluate, plan_for)
from ainode.planner.facts import ModelFacts


def _mla(weight_bytes=166_900_000_000):
    return ModelFacts(
        repo="org/model", num_layers=43, attention_layers=43,
        kv_lora_rank=512, qk_rope_head_dim=64, num_attention_heads=64,
        num_kv_heads=1, head_dim=512, torch_dtype="bfloat16",
        weight_bytes=weight_bytes, max_position_embeddings=1048576,
        num_experts=256)


def _nodes(free_a, free_b):
    return [NodeBudget(node_id="n1", name="spark-13e1", total_gb=128,
                       free_gb=free_a),
            NodeBudget(node_id="n2", name="spark-1432", total_gb=128,
                       free_gb=free_b)]


class TestTheCacheFollowsTheTightestNode:
    def test_a_tighter_second_node_shrinks_the_whole_cache(self):
        even = plan_for(_mla(), _nodes(116, 116), kv_cache_dtype="fp8_ds_mla")
        uneven = plan_for(_mla(), _nodes(116, 100), kv_cache_dtype="fp8_ds_mla")
        assert uneven.kv_tokens < even.kv_tokens

    def test_the_spare_room_does_not_appear_in_the_total(self):
        # 16 GB more on one node buys nothing: the cache is min x ranks.
        uneven = _evaluate(_mla(), _nodes(116, 100), "tensor", 24768)
        matched = _evaluate(_mla(), _nodes(100, 100), "tensor", 24768)
        assert uneven.kv_gb == matched.kv_gb


class TestItSaysSo:
    def test_the_tight_node_is_named(self):
        note = _asymmetry(_evaluate(_mla(), _nodes(116, 100), "tensor", 24768))
        assert note and "spark-1432" in note[0]

    def test_the_stranded_amount_is_given(self):
        note = _asymmetry(_evaluate(_mla(), _nodes(116, 100), "tensor", 24768))
        assert "16.0 GB more that cannot be used" in note[0]

    def test_it_says_what_to_do_instead_of_raising_a_number(self):
        note = _asymmetry(_evaluate(_mla(), _nodes(116, 100), "tensor", 24768))
        assert "no setting that spends one node's spare room" in note[0]
        assert "free spark-1432" in note[0]

    def test_it_reaches_the_plan(self):
        plan = plan_for(_mla(), _nodes(116, 100), kv_cache_dtype="fp8_ds_mla")
        assert any("tightest node" in n for n in plan.notes)


class TestItStaysQuietWhenThereIsNothingToSay:
    @pytest.mark.parametrize("gap", [0.0, ASYMMETRY_GB - 0.5])
    def test_nodes_of_practically_the_same_size(self, gap):
        """Below the threshold two nodes are the same size for practical
        purposes, and saying this on every plan would be noise."""
        assert _asymmetry(_evaluate(_mla(), _nodes(116, 116 - gap), "tensor",
                                    24768)) == []

    def test_a_single_node_plan_has_no_tightest_node(self):
        candidate = _evaluate(_mla(weight_bytes=40_000_000_000),
                              _nodes(116, 116)[:1], "tensor", 24768)
        assert _asymmetry(candidate) == []

    def test_a_solo_plan_says_nothing_about_ranks(self):
        # 80 GB of weights fit one node with 112 usable, and then there is no
        # second rank to be tight.
        plan = plan_for(_mla(weight_bytes=80_400_000_000), _nodes(116, 100),
                        kv_cache_dtype="fp8_ds_mla")
        assert plan.strategy == "solo"
        assert not any("tightest node" in n for n in plan.notes)
