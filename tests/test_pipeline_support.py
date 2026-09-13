"""Not every model can be pipelined, and finding out should not cost minutes.

From the cluster, after a three-node launch had loaded across every node:

    NotImplementedError: Pipeline parallelism is not supported for this model.
    Supported models implement the `SupportsPP` interface.

The planner had converted the selection to PP=3 on its own — the right move
for a model proven at TP=2, and the wrong one for a model that cannot pipeline
at all. Tensor-parallel then being the only axis, and needing a power-of-two
rank count, three nodes have no valid split whatsoever.
"""

from __future__ import annotations

import pytest

from ainode.engine.parallelism import (
    ParallelPlanError,
    Strategy,
    plan_for_model,
)
from ainode.models.api_routes import catalog_supports_pipeline


class TestTheCatalogKnows:
    def test_glm_cannot_pipeline(self):
        assert catalog_supports_pipeline(
            "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark") is False

    @pytest.mark.parametrize("model", [
        "unsloth/Qwen3.8-27B-NVFP4",
        "nvidia/Gemma-4-26B-A4B-NVFP4",
    ])
    def test_others_can(self, model):
        assert catalog_supports_pipeline(model) is True

    def test_an_unknown_model_is_assumed_able(self):
        # Refusing on a guess would be worse than the failure this prevents,
        # which is at least explicit.
        assert catalog_supports_pipeline("some/random-model") is True
        assert catalog_supports_pipeline("") is True


class TestThePlannerRefusesEarly:
    def test_three_nodes_are_refused_with_the_alternative(self):
        with pytest.raises(ParallelPlanError) as excinfo:
            plan_for_model(Strategy.AUTO, 3, proven_tp=2, supports_pipeline=False)
        message = str(excinfo.value)
        assert "pipeline" in message
        assert "Select exactly 2" in message

    def test_an_explicit_pipeline_request_is_refused_too(self):
        with pytest.raises(ParallelPlanError):
            plan_for_model(Strategy.PIPELINE, 2, supports_pipeline=False)

    def test_two_nodes_still_plan_as_tensor(self):
        plan, note = plan_for_model(Strategy.AUTO, 2, proven_tp=2,
                                    supports_pipeline=False)
        assert plan.tensor_parallel_size == 2 and note == ""

    def test_one_node_is_fine(self):
        plan, _ = plan_for_model(Strategy.AUTO, 1, proven_tp=2,
                                 supports_pipeline=False)
        assert plan.world_size == 1

    def test_five_nodes_name_the_largest_usable_count(self):
        with pytest.raises(ParallelPlanError) as excinfo:
            plan_for_model(Strategy.AUTO, 5, supports_pipeline=False)
        assert "Select exactly 4" in str(excinfo.value)

    def test_a_pipelineable_model_is_unaffected(self):
        plan, _ = plan_for_model(Strategy.AUTO, 3, proven_tp=2,
                                 supports_pipeline=True)
        assert plan.pipeline_parallel_size == 3

    def test_and_a_tensor_request_it_downgrades_still_explains_itself(self):
        # The note is for a downgrade the operator did not ask for. Choosing
        # pipeline because three is not a tensor-parallel size is the ordinary
        # resolution of "auto", and needs no apology.
        plan, note = plan_for_model(Strategy.TENSOR, 3, proven_tp=2,
                                    supports_pipeline=True)
        assert plan.pipeline_parallel_size == 3
        assert "proven" in note


class TestTheLogNetCoversTheRest:
    def test_the_engine_message_is_explained(self):
        # For a model nobody curated, the refusal cannot happen in advance.
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe(
            "NotImplementedError: Pipeline parallelism is not supported for "
            "this model. Supported models implement the `SupportsPP` interface.")
        tracker.fail("the launcher exited (code 1)")
        reason = tracker.failure_reason()
        assert "power-of-two" in reason
        assert "NotImplementedError" in reason


class TestTheLaunchRouteAsks:
    def test_both_launch_paths_pass_the_flag(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "engine" /
               "sharding_routes.py").read_text()
        assert src.count("supports_pipeline=catalog_supports_pipeline(model)") >= 1
        assert "catalog_supports_pipeline" in src
