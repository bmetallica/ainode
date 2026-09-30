"""The coding model that actually fits, in the catalog.

Asked, with a link to the bf16 original:

    Bekommen wir das hier zum laufen? .../Qwen/Qwen3.8-Flash-Next

Not that one: 360 GB against about 300 GB usable across three nodes, and that
is before a byte of cache. NVIDIA's NVFP4 build is 132.7 GB and fits two nodes
with the full window.

What makes it worth an entry is how little it needs. Every trap this catalog has
collected — a missing quant_method, an architecture vLLM does not know, a
checkpoint laid out for another runtime, repository code that must be trusted —
this one avoids. The interesting part of the recipe is therefore what it does
NOT set, and each omission is a decision with a reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.models.registry import CURATED_CLUSTER_MODELS, MODEL_CATALOG

SOURCE = Path(__import__("ainode.models.registry", fromlist=["x"]).__file__
              ).read_text()


@pytest.fixture
def entry():
    found = (CURATED_CLUSTER_MODELS.get("qwen3.8-flash-next-nvfp4")
             or MODEL_CATALOG.get("qwen3.8-flash-next-nvfp4"))
    assert found is not None, "the entry is not in the catalog"
    return found


class TestItPointsAtTheOneThatFits:
    def test_the_nvfp4_build_not_the_original(self, entry):
        assert entry.hf_repo == "nvidia/Qwen3.8-Flash-Next-NVFP4"

    def test_the_size_is_the_hubs_own_figure(self, entry):
        assert entry.size_gb == 132.7

    def test_it_needs_more_than_one_node(self, entry):
        assert entry.min_memory_gb > 128
        assert entry.proven_tp == 2

    def test_the_original_is_not_in_the_catalog(self):
        # 360 GB. Listing it would offer a launch that cannot happen.
        for catalog in (CURATED_CLUSTER_MODELS, MODEL_CATALOG):
            for model in catalog.values():
                assert model.hf_repo != "Qwen/Qwen3.8-Flash-Next"


class TestWhatItClaims:
    def test_it_has_been_served_here_but_is_not_badged(self, entry):
        # Served across spark-1432 and spark-659b; the badge waits for a
        # judgement of its answers, not only a launch.
        assert entry.verified is False
        assert "NOT YET SERVED HERE" not in entry.description

    def test_it_drafts_with_its_own_mtp_layer(self, entry):
        args = entry.extra_vllm_args
        assert args[args.index("--speculative_config.method") + 1] == "qwen4_exp_mtp"
        assert args[args.index("--speculative_config.num_speculative_tokens") + 1] == "2"

    def test_it_does_not_use_instanttensor(self, entry):
        # Its 1.27 GB embedding is above the driver's ~1.2 GB budget.
        assert "instanttensor" not in " ".join(entry.extra_vllm_args)

    def test_it_is_a_vision_model(self, entry):
        assert "vision" in entry.capabilities

    def test_it_is_a_coding_model(self, entry):
        assert "code" in entry.capabilities and "tool_use" in entry.capabilities

    def test_the_licence_is_named(self, entry):
        assert "NVIDIA" in entry.license


class TestTheRecipeSplitsTheExperts:
    def test_expert_parallelism_is_on(self, entry):
        # 512 experts; without it 132.7 GB needs 132.7 per node, not 66.4.
        assert "--enable-expert-parallel" in entry.extra_vllm_args

    def test_the_window_is_the_models_own(self, entry):
        args = entry.extra_vllm_args
        assert args[args.index("--max-model-len") + 1] == "262144"
        assert entry.context_length == 262144


class TestWhatItDeliberatelyDoesNotSet:
    def test_no_trust_remote_code(self, entry):
        """The config has no auto_map and the architecture is in vLLM's own
        registry. Asking for it anyway would run repository code for nothing."""
        assert "--trust-remote-code" not in entry.extra_vllm_args

    def test_no_kv_cache_dtype(self, entry):
        """It has a vision_config, and fp8 KV corrupts vision generation on
        GB10. serve_args downgrades the fp8 default to auto for a multimodal
        model — an explicit value here would override that safety rule."""
        assert "--kv-cache-dtype" not in entry.extra_vllm_args

    def test_no_reasoning_parser(self, entry):
        """It thinks by default and a client will want the blocks separated,
        but vLLM's argparse rejects a name it does not know and kills the launch
        at second three. A recipe that has never run here must not guess."""
        assert "--reasoning-parser" not in entry.extra_vllm_args

    def test_each_omission_is_explained_in_the_source(self):
        for reason in ("no auto_map", "corrupts vision", "argparse rejects"):
            assert reason in SOURCE, reason

    def test_the_description_tells_the_operator_about_the_parser(self, entry):
        assert "reasoning parser" in entry.description


class TestTheMemoryFractionIsHeldBack:
    def test_it_is_below_what_the_planner_would_pick(self, entry):
        # The planner reaches 0.95, which reserves 121.6 GB of 128 per node for
        # a launch that uses 79 at two sessions.
        assert 0.7 <= entry.recommended_gmu <= 0.9

    def test_the_reason_is_written_down(self):
        assert "held and idle" in SOURCE


class TestItPlansTheWayTheEntryClaims:
    def _plan(self, concurrency=2):
        from ainode.planner.compute import NodeBudget, plan_for
        from ainode.planner.facts import ModelFacts

        facts = ModelFacts(
            repo="nvidia/Qwen3.8-Flash-Next-NVFP4", num_layers=48,
            attention_layers=12, num_kv_heads=2, head_dim=256,
            num_attention_heads=24, torch_dtype="bfloat16",
            weight_bytes=132_700_000_000, max_position_embeddings=262144,
            num_experts=512)
        nodes = [NodeBudget(node_id="a", name="1432", total_gb=128.0,
                            free_gb=126.4),
                 NodeBudget(node_id="b", name="659b", total_gb=128.0,
                            free_gb=126.4)]
        return plan_for(facts, nodes, kv_cache_dtype="auto",
                        concurrency=concurrency, recommended_gmu=0.85)

    def test_two_nodes_at_the_full_window(self):
        plan = self._plan()
        assert plan.fits and plan.tensor_parallel_size == 2
        assert plan.max_model_len == 262144

    def test_the_weights_per_node_match_the_description(self):
        assert round(self._plan().weights_per_node_gb) == 70

    def test_ten_or_more_requests_fit(self):
        assert self._plan().max_num_seqs >= 10

    def test_the_fraction_leaves_memory_on_the_node(self):
        plan = self._plan()
        assert plan.reserved_per_node_gb < plan.node_total_gb - 15
