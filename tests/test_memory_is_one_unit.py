"""The weights were decimal and the memory was binary.

Reported from the cluster, planning Qwen3-Coder-Next bf16 across two nodes with
nothing loaded on either — one of them the head:

    Weights: 159.4 GB on disk, split 2 ways = 83.7 GB per node
    Left for KV: 4.1 GB across 2 node(s)
    KV per token: 24.0 KiB -> 165,774 tokens

    auf dem node ist nichts geladen, es ist aber der head

The head does carry a base load, and it is not 80 GB of it. 159.4 decimal GB is
148.5 GiB; per node at TP=2 that is 77.9 GiB against the 83.7 the planner
reserved. 5.7 GiB too much, on a plan with 2.05 GiB per node left for the
cache — so most of the cache was spent on a rounding convention rather than on
anything running.

facts.weights_gb came from weight_bytes_on_disk / 1e9. node_budgets came from
psutil, through MiB, divided by 1024. Both were called _gb and subtracted from
each other. The error is 7.4%, always in the direction that makes the model
look bigger than the node, and it lands entirely on the KV cache because the
cache is the remainder.
"""

from __future__ import annotations

import pytest

from ainode.core.units import (BYTES_PER_GB, BYTES_PER_GIB, gb_from_bytes,
                               gb_from_gib, gb_from_mib, gib_from_gb)


class TestTheConversions:
    def test_a_decimal_gigabyte_is_a_decimal_gigabyte(self):
        assert gb_from_bytes(10 ** 9) == 1.0

    def test_mib_to_gb(self):
        # What a 128 GB Spark actually reports.
        assert round(gb_from_mib(122070), 1) == 128.0

    def test_the_two_units_differ_by_the_expected_amount(self):
        assert round(BYTES_PER_GIB / BYTES_PER_GB, 4) == 1.0737

    def test_round_tripping_is_stable(self):
        assert round(gb_from_gib(gib_from_gb(128.0)), 6) == 128.0

    def test_a_guard_threshold_in_gib_is_larger_in_gb(self):
        # 8 GiB is 8.59 GB. Where a guard figure enters a plan it is converted
        # explicitly, because assuming they are the same is what this fixes.
        assert round(gb_from_gib(8), 2) == 8.59


class TestTheBudgetsAreDecimal:
    def _budgets(self, total_mib=122070, used_mib=14 * 1024):
        from ainode.planner.api_routes import node_budgets

        node = type("N", (), {
            "node_id": "n1", "node_name": "n1", "status": "online",
            "gpu_memory_total_mb": total_mib, "gpu_memory_used_mb": used_mib,
            "gpu_memory_gb": 0, "instances": [],
        })()
        cluster = type("C", (), {"members": lambda self: [node]})()
        return node_budgets({"cluster_state": cluster})

    def test_a_128_gb_node_reports_128(self):
        # It reported 119.2 and called that GB, so every plan compared decimal
        # weights against binary memory.
        assert round(self._budgets()[0].total_gb) == 128

    def test_free_is_total_minus_used_in_the_same_unit(self):
        budget = self._budgets()[0]
        assert round(budget.free_gb, 1) == round(gb_from_mib(122070 - 14 * 1024), 1)

    def test_it_is_the_unit_the_weights_are_in(self, tmp_path):
        from ainode.planner.facts import weight_bytes_on_disk

        directory = tmp_path / "org--model"
        directory.mkdir()
        with open(directory / "model.safetensors", "wb") as sink:
            sink.truncate(10 ** 9)
        # One decimal gigabyte of weights against one decimal gigabyte of
        # memory. That is the whole point.
        assert round(weight_bytes_on_disk(directory) / BYTES_PER_GB, 3) == 1.0
        # The same billion bytes, arriving through the memory path.
        assert round(gb_from_mib(10 ** 9 / (1024 * 1024)), 3) == 1.0


class TestWhatItDoesToThePlan:
    """The reported case, rebuilt: two 128 GB nodes, ~14 GiB base load on the
    head, nothing serving."""

    def _plan(self, kv="fp8"):
        from ainode.planner.compute import NodeBudget, plan_for
        from ainode.planner.facts import ModelFacts

        facts = ModelFacts(
            repo="Qwen/Qwen3-Coder-Next", num_layers=48, attention_layers=12,
            num_kv_heads=2, head_dim=256, num_attention_heads=16,
            torch_dtype="bfloat16", weight_bytes=159_400_000_000,
            max_position_embeddings=262144, num_experts=512)
        nodes = [
            NodeBudget(node_id="n1", name="spark-13e1",
                       total_gb=gb_from_mib(122070),
                       free_gb=gb_from_mib(122070 - 14 * 1024)),
            NodeBudget(node_id="n2", name="spark-1432",
                       total_gb=gb_from_mib(122070),
                       free_gb=gb_from_mib(122070 - 1024)),
        ]
        return plan_for(facts, nodes, kv_cache_dtype=kv, concurrency=2)

    def test_the_full_window_fits(self):
        # The panel offered 163,840 with one concurrent request.
        assert self._plan().max_model_len == 262144

    def test_there_is_room_for_more_than_one_request(self):
        assert self._plan().concurrent_requests >= 2

    def test_the_cache_is_tens_of_gigabytes_not_four(self):
        assert self._plan().kv_gb > 30

    @pytest.mark.parametrize("kv,expected_ratio", [("auto", 1), ("fp8", 2)])
    def test_fp8_doubles_the_tokens(self, kv, expected_ratio):
        base = self._plan("auto").kv_tokens
        assert self._plan(kv).kv_tokens == base * expected_ratio
