"""A curated model must launch the same way solo and across the cluster.

The recipe used to be applied in exactly one place — the solo load route — so
clicking Qwen3.8 on one node produced a command with its engine image,
reasoning parser and speculative config, and clicking the same model on three
nodes produced a command with none of them. The distributed launch, the one
that most needs the proven configuration, was the one that ran without it.
"""

from ainode.engine.parallelism import Strategy, plan_for_model
from ainode.models.api_routes import (
    apply_catalog_recipe,
    catalog_proven_tp,
    catalog_recipe,
)

QWEN = "unsloth/Qwen3.8-27B-NVFP4"


class TestRecipeApplication:
    def test_fills_engine_image_and_flags_when_caller_said_nothing(self):
        overrides, gmu = apply_catalog_recipe(QWEN, {}, None)
        assert overrides["engine_image"] == catalog_recipe(QWEN)["engine_image"]
        assert overrides["extra_vllm_args"]
        assert gmu is not None and 0 < gmu <= 1

    def test_caller_wins_over_recipe(self):
        overrides, gmu = apply_catalog_recipe(
            QWEN, {"engine_image": "mine:1", "extra_vllm_args": []}, 0.5)
        assert overrides["engine_image"] == "mine:1"
        assert overrides["extra_vllm_args"] == []
        assert gmu == 0.5

    def test_uncurated_model_is_left_alone(self):
        overrides, gmu = apply_catalog_recipe("some/random-model", {}, None)
        assert overrides == {} and gmu is None

    def test_junk_model_id_does_not_raise(self):
        assert apply_catalog_recipe("", {}, None) == ({}, None)


class TestProvenTpPlanning:
    def test_tensor_beyond_proven_width_becomes_pipeline(self):
        # Qwen3.8 is proven at TP=1; two nodes must not re-shard its heads.
        plan, note = plan_for_model(Strategy.TENSOR, 2, proven_tp=1)
        assert plan.tensor_parallel_size == 1
        assert plan.pipeline_parallel_size == 2
        assert "proven" in note.lower()

    def test_auto_respects_proven_width_too(self):
        plan, note = plan_for_model("auto", 2, proven_tp=1)
        assert plan.strategy is Strategy.PIPELINE and note

    def test_no_limit_for_uncurated_models(self):
        plan, note = plan_for_model(Strategy.TENSOR, 2, proven_tp=0)
        assert plan.tensor_parallel_size == 2 and note == ""

    def test_model_proven_wider_than_the_node_set_is_not_downgraded(self):
        # proven_tp=4 on 2 nodes is a memory question, not a layout one.
        plan, note = plan_for_model(Strategy.TENSOR, 2, proven_tp=4)
        assert plan.tensor_parallel_size == 2 and note == ""

    def test_explicit_pipeline_is_never_rewritten(self):
        plan, note = plan_for_model(Strategy.PIPELINE, 3, proven_tp=1)
        assert plan.pipeline_parallel_size == 3 and note == ""

    def test_qwen_is_curated_as_proven_at_one(self):
        assert catalog_proven_tp(QWEN) == 1
        assert catalog_proven_tp("some/random-model") == 0
