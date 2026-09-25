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

    @pytest.mark.parametrize("dtype", ["nvfp4", "nvfp4_ds_mla",
                                       "nvfp4_4over6", "int4_per_token_head"])
    def test_four_bit_families_are_half_a_byte(self, dtype):
        assert kv_dtype_bytes(dtype, 2) == 0.5

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
        assert kv_dtype_bytes("  NVFP4_DS_MLA ", 2) == 0.5


class TestItReachesTheTokenCost:
    def test_four_bit_halves_what_fp8_costs(self):
        facts = _facts()
        assert kv_bytes_per_token(facts, "nvfp4_ds_mla") == \
            kv_bytes_per_token(facts, "fp8") // 2

    def test_the_old_rule_was_four_times_off(self):
        # What this fixes, stated as the number it used to produce: anything
        # not starting with "fp8" fell through to the model dtype.
        facts = _facts()
        assert kv_bytes_per_token(facts, "nvfp4_ds_mla") * 4 == \
            kv_bytes_per_token(facts, "auto")

    def test_mla_is_costed_on_the_latent_not_the_heads(self):
        facts = _facts(kv_lora_rank=512, qk_rope_head_dim=64)
        assert kv_bytes_per_token(facts, "nvfp4_ds_mla") == \
            int(60 * (512 + 64) * 0.5)


class TestItReachesThePlan:
    NODES = [NodeBudget(node_id="n1", name="S1", total_gb=128, free_gb=116),
             NodeBudget(node_id="n2", name="S2", total_gb=128, free_gb=116)]

    def test_a_four_bit_cache_backs_more_context(self):
        facts = _facts(weight_bytes=160_000_000_000)
        fp8 = plan_for(facts, self.NODES, kv_cache_dtype="fp8_ds_mla",
                       concurrency=2)
        nvfp4 = plan_for(facts, self.NODES, kv_cache_dtype="nvfp4_ds_mla",
                         concurrency=2)
        assert nvfp4.kv_tokens == fp8.kv_tokens * 2

    def test_the_note_names_the_dtype_that_was_planned(self):
        facts = _facts(weight_bytes=160_000_000_000)
        plan = plan_for(facts, self.NODES, kv_cache_dtype="nvfp4_ds_mla")
        assert any("nvfp4_ds_mla" in note for note in plan.notes)

    def test_an_unknown_dtype_names_the_models_own(self):
        facts = _facts(weight_bytes=160_000_000_000)
        plan = plan_for(facts, self.NODES, kv_cache_dtype="turboquant_k8v4")
        assert any("bfloat16" in note for note in plan.notes)
