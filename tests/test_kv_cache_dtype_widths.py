"""How wide is a cached element, really.

The planner knew two answers: "fp8" meant one byte, everything else meant the
model's own dtype. Measured against the engine image that actually serves
here — vllm-node, vLLM 0.3.1.dev19+g08633cb5c.d20260917 — there are eighteen:

    ('auto', 'float16', 'bfloat16', 'fp8', 'fp8_e4m3', 'fp8_e5m2', 'fp8_inc',
     'fp8_ds_mla', 'nvfp4_ds_mla', 'turboquant_k8v4', 'turboquant_4bit_nc',
     'turboquant_k3v4_nc', 'turboquant_3bit_nc', 'int4_per_token_head',
     'int8_per_token_head', 'fp8_per_token_head', 'nvfp4', 'nvfp4_4over6')

A 4-bit cache read as the model dtype is a factor of four, in the direction
that refuses a launch which fits. nvfp4_ds_mla is what the published
two-Spark recipe for DeepSeek-V4-Flash serves with, so this is a value an
operator can reach for today, through Advanced → extra vLLM args.
"""

from __future__ import annotations

import pytest

from ainode.planner.compute import (NodeBudget, kv_bytes_per_token,
                                    kv_dtype_bytes, plan_for)
from ainode.planner.facts import ModelFacts


def _facts(**kw):
    base = dict(repo="org/model", num_layers=60, attention_layers=60,
                num_kv_heads=4, head_dim=128, torch_dtype="bfloat16",
                weight_bytes=100_000_000_000)
    base.update(kw)
    return ModelFacts(**base)


class TestTheWidthOfEachDtype:
    @pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3", "fp8_e5m2",
                                       "fp8_inc", "fp8_ds_mla",
                                       "fp8_per_token_head",
                                       "int8_per_token_head"])
    def test_eight_bit_families_are_one_byte(self, dtype):
        assert kv_dtype_bytes(dtype, 2) == 1.0

    @pytest.mark.parametrize("dtype", ["nvfp4", "nvfp4_4over6",
                                       "int4_per_token_head"])
    def test_four_bit_families_are_half_a_byte(self, dtype):
        assert kv_dtype_bytes(dtype, 2) == 0.5

    def test_nvfp4_ds_mla_is_a_layout_and_not_a_width(self):
        """The one that looks four-bit and is not.

        On DeepSeek-V4 the ds_mla layout is 584 bytes per token per layer for
        both dtypes — the compressed latent at one byte per element plus its
        scales; only the kernel dispatch differs (MiaAI-Lab's DGX Spark
        recipe, docs/PATCHES.md issue #22, MIT). Reading four bits out of the
        name would halve the planned cache against a checkpoint that does not
        shrink, and that is the direction that takes a node down.
        """
        assert kv_dtype_bytes("nvfp4_ds_mla", 2) == kv_dtype_bytes("fp8_ds_mla", 2)
        assert kv_dtype_bytes("nvfp4_ds_mla", 2) == 1.0

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_an_explicit_full_width_is_two(self, dtype):
        assert kv_dtype_bytes(dtype, 1) == 2.0

    def test_auto_is_the_models_own_dtype(self):
        assert kv_dtype_bytes("auto", 2) == 2.0
        assert kv_dtype_bytes("", 2) == 2.0

    @pytest.mark.parametrize("dtype", ["turboquant_k8v4", "turboquant_4bit_nc",
                                       "turboquant_k3v4_nc",
                                       "turboquant_3bit_nc"])
    def test_turboquant_falls_back_rather_than_guessing(self, dtype):
        """The names read like per-tensor widths. They have not been verified
        against the kernels, and the safe direction is to over-reserve: a plan
        that holds back too much cache serves a shorter context than it could;
        one that holds back too little takes the node down."""
        assert kv_dtype_bytes(dtype, 2) == 2.0

    def test_case_and_padding_do_not_matter(self):
        assert kv_dtype_bytes("  NVFP4_4OVER6 ", 2) == 0.5


class TestItReachesTheTokenCost:
    def test_four_bit_halves_what_fp8_costs(self):
        facts = _facts()
        assert kv_bytes_per_token(facts, "nvfp4") == \
            kv_bytes_per_token(facts, "fp8") // 2

    def test_the_old_rule_was_four_times_off(self):
        # What this fixes, stated as the number it used to produce: anything
        # not starting with "fp8" fell through to the model dtype.
        facts = _facts()
        assert kv_bytes_per_token(facts, "nvfp4") * 4 == \
            kv_bytes_per_token(facts, "auto")

    def test_mla_is_costed_on_the_latent_not_the_heads(self):
        facts = _facts(kv_lora_rank=512, qk_rope_head_dim=64)
        assert kv_bytes_per_token(facts, "fp8_ds_mla") == 60 * (512 + 64)


class TestItReachesThePlan:
    NODES = [NodeBudget(node_id="n1", name="S1", total_gb=128, free_gb=116),
             NodeBudget(node_id="n2", name="S2", total_gb=128, free_gb=116)]

    def test_a_four_bit_cache_backs_more_context(self):
        facts = _facts(weight_bytes=160_000_000_000)
        fp8 = plan_for(facts, self.NODES, kv_cache_dtype="fp8", concurrency=2)
        nvfp4 = plan_for(facts, self.NODES, kv_cache_dtype="nvfp4",
                         concurrency=2)
        assert nvfp4.kv_tokens == fp8.kv_tokens * 2

    def test_the_ds_mla_pair_plan_the_same(self):
        facts = _facts(weight_bytes=160_000_000_000, kv_lora_rank=512,
                       qk_rope_head_dim=64)
        fp8 = plan_for(facts, self.NODES, kv_cache_dtype="fp8_ds_mla")
        nvfp4 = plan_for(facts, self.NODES, kv_cache_dtype="nvfp4_ds_mla")
        assert nvfp4.kv_tokens == fp8.kv_tokens

    def test_the_note_names_the_dtype_that_was_planned(self):
        facts = _facts(weight_bytes=160_000_000_000)
        plan = plan_for(facts, self.NODES, kv_cache_dtype="nvfp4_ds_mla")
        assert any("nvfp4_ds_mla" in note for note in plan.notes)

    def test_an_unknown_dtype_names_the_models_own(self):
        facts = _facts(weight_bytes=160_000_000_000)
        plan = plan_for(facts, self.NODES, kv_cache_dtype="turboquant_k8v4")
        assert any("bfloat16" in note for note in plan.notes)


class TestTheLatentGeometryIsRecognised:
    """DeepSeek-V4 writes MLA without naming it.

    No ``kv_lora_rank`` in the config — one KV "head" whose ``head_dim`` IS
    the compressed rank (512), beside ``qk_rope_head_dim`` 64. Read as
    grouped-query attention that costs 2 x heads x head_dim, which counts a K
    and a V where MLA has one latent: 1024 bytes per layer per token against
    a measured 584 (MiaAI-Lab's DGX Spark recipe, docs/PATCHES.md issue #22,
    MIT). Nearly twice the cache reserved, straight off the offered context.
    """

    DSV4 = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43, "num_attention_heads": 64,
        "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
        "hidden_size": 4096, "torch_dtype": "bfloat16",
        "max_position_embeddings": 1048576,
    }

    def _facts(self, **over):
        from ainode.planner.facts import facts_from_config

        config = dict(self.DSV4)
        config.update(over)
        return facts_from_config(config, repo="org/model", weight_bytes=1)

    def test_the_latent_rank_is_inferred(self):
        assert self._facts().kv_lora_rank == 512

    def test_the_per_layer_cost_lands_near_the_measured_584(self):
        facts = self._facts()
        per_layer = kv_bytes_per_token(facts, "fp8_ds_mla") / facts.num_layers
        assert per_layer == 576          # 584 minus the scale block

    def test_it_is_not_the_grouped_query_figure(self):
        facts = self._facts()
        naive = 2 * facts.num_layers * 1 * 512
        assert kv_bytes_per_token(facts, "fp8_ds_mla") < naive * 0.6

    def test_an_explicit_kv_lora_rank_still_wins(self):
        assert self._facts(kv_lora_rank=256).kv_lora_rank == 256

    @pytest.mark.parametrize("over", [
        {"num_key_value_heads": 4},        # ordinary GQA
        {"head_dim": 128},                 # a real head, not a latent
        {"qk_rope_head_dim": 0},           # no rope split: not this layout
    ])
    def test_grouped_query_models_are_left_alone(self, over):
        assert self._facts(**over).kv_lora_rank == 0
