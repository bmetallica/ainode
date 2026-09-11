"""Unit tests for ainode.engine.parallelism.

The central constraint: a 3-node mesh cannot run tensor parallelism, so
plan_for has to refuse TP=3 with a message that names the alternatives
rather than letting vLLM fail deep in engine startup.
"""

from __future__ import annotations

import pytest

from ainode.engine.parallelism import (
    TENSOR_PARALLEL_SIZES,
    ParallelPlan,
    ParallelPlanError,
    Strategy,
    plan_for,
    recommend_strategy,
    validate_plan,
)


class TestStrategyParse:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("tensor", Strategy.TENSOR),
            ("tensor_parallel", Strategy.TENSOR),
            ("tensor-parallel", Strategy.TENSOR),
            ("TP", Strategy.TENSOR),
            ("pipeline", Strategy.PIPELINE),
            ("pipeline_parallel", Strategy.PIPELINE),
            ("pp", Strategy.PIPELINE),
            ("data", Strategy.DATA),
            ("data_parallel", Strategy.DATA),
            ("dp", Strategy.DATA),
            ("auto", Strategy.AUTO),
            ("  Tensor  ", Strategy.TENSOR),
        ],
    )
    def test_accepts_every_spelling_in_circulation(self, value, expected):
        assert Strategy.parse(value) is expected

    def test_empty_means_auto(self):
        assert Strategy.parse(None) is Strategy.AUTO
        assert Strategy.parse("") is Strategy.AUTO

    def test_passthrough_of_an_enum_member(self):
        assert Strategy.parse(Strategy.PIPELINE) is Strategy.PIPELINE

    def test_unknown_raises_with_the_valid_set(self):
        with pytest.raises(ParallelPlanError, match="tensor, pipeline, data, auto"):
            Strategy.parse("megatron")


class TestRecommendStrategy:
    @pytest.mark.parametrize("nodes", [1, 2, 4, 8])
    def test_power_of_two_picks_tensor(self, nodes):
        assert recommend_strategy(nodes) is Strategy.TENSOR

    @pytest.mark.parametrize("nodes", [3, 5, 6, 7, 9])
    def test_other_counts_pick_pipeline(self, nodes):
        assert recommend_strategy(nodes) is Strategy.PIPELINE

    def test_never_picks_data_on_its_own(self):
        """Data parallelism changes what the cluster is for — capacity to
        concurrency — so it stays an explicit operator choice."""
        assert all(
            recommend_strategy(n) is not Strategy.DATA for n in range(1, 17)
        )


class TestPlanFor:
    def test_two_nodes_default_to_tp2(self):
        plan = plan_for("auto", 2)
        assert (plan.tensor_parallel_size, plan.pipeline_parallel_size,
                plan.data_parallel_size) == (2, 1, 1)
        assert plan.strategy is Strategy.TENSOR

    def test_four_nodes_default_to_tp4(self):
        assert plan_for("auto", 4).tensor_parallel_size == 4

    def test_three_nodes_default_to_pp3(self):
        plan = plan_for("auto", 3)
        assert (plan.tensor_parallel_size, plan.pipeline_parallel_size,
                plan.data_parallel_size) == (1, 3, 1)
        assert plan.strategy is Strategy.PIPELINE

    def test_explicit_tp_on_three_nodes_is_refused(self):
        with pytest.raises(ParallelPlanError) as excinfo:
            plan_for("tensor", 3)
        message = str(excinfo.value)
        # The message has to carry the way out, not just the refusal.
        assert "pipeline" in message
        assert "data" in message
        assert "2 nodes" in message

    def test_explicit_pipeline_on_three_nodes(self):
        assert plan_for("pipeline", 3).pipeline_parallel_size == 3

    def test_explicit_data_on_three_nodes(self):
        plan = plan_for("data", 3)
        assert plan.data_parallel_size == 3
        assert plan.tensor_parallel_size == 1

    def test_single_node_is_tp1(self):
        plan = plan_for("auto", 1)
        assert plan.world_size == 1
        assert not plan.is_distributed

    def test_zero_nodes_is_refused(self):
        with pytest.raises(ParallelPlanError, match="at least one node"):
            plan_for("auto", 0)

    @pytest.mark.parametrize("nodes", [5, 6, 7])
    def test_odd_counts_are_refused_for_tensor(self, nodes):
        with pytest.raises(ParallelPlanError, match="not supported"):
            plan_for("tensor", nodes)

    def test_pipeline_works_at_any_node_count(self):
        for nodes in range(1, 9):
            assert plan_for("pipeline", nodes).world_size == nodes


class TestValidatePlan:
    def test_accepts_an_exact_fill(self):
        validate_plan(ParallelPlan(tensor_parallel_size=2,
                                   pipeline_parallel_size=2), 4)

    def test_rejects_under_fill(self):
        with pytest.raises(ParallelPlanError, match="needs 2 GPU"):
            validate_plan(ParallelPlan(tensor_parallel_size=2), 3)

    def test_rejects_over_fill(self):
        with pytest.raises(ParallelPlanError, match="needs 4 GPU"):
            validate_plan(ParallelPlan(tensor_parallel_size=4), 3)

    def test_rejects_unsupported_tensor_size(self):
        with pytest.raises(ParallelPlanError, match="is not supported"):
            validate_plan(ParallelPlan(tensor_parallel_size=3), 3)

    def test_rejects_a_zero_size(self):
        with pytest.raises(ParallelPlanError, match="at least 1"):
            validate_plan(ParallelPlan(pipeline_parallel_size=0), 0)

    def test_combined_axes_are_allowed_when_they_multiply_out(self):
        """TP=2 x PP=2 across 4 nodes is legitimate, and TENSOR_PARALLEL_SIZES
        constrains only the TP factor, not the product."""
        validate_plan(ParallelPlan(tensor_parallel_size=2,
                                   pipeline_parallel_size=2), 4)
        validate_plan(ParallelPlan(tensor_parallel_size=2,
                                   data_parallel_size=3), 6)


class TestParallelPlanShape:
    def test_world_size_multiplies_every_axis(self):
        plan = ParallelPlan(tensor_parallel_size=2, pipeline_parallel_size=3,
                            data_parallel_size=2)
        assert plan.world_size == 12

    @pytest.mark.parametrize(
        "plan,expected",
        [
            (ParallelPlan(), "TP=1"),
            (ParallelPlan(tensor_parallel_size=4), "TP=4"),
            (ParallelPlan(pipeline_parallel_size=3), "PP=3"),
            (ParallelPlan(data_parallel_size=3), "DP=3"),
            (ParallelPlan(tensor_parallel_size=2, pipeline_parallel_size=2),
             "TP=2 · PP=2"),
        ],
    )
    def test_label_reads_as_a_badge(self, plan, expected):
        assert plan.label() == expected

    def test_round_trips_through_a_dict(self):
        plan = plan_for("pipeline", 3)
        assert ParallelPlan.from_dict(plan.to_dict()) == plan

    def test_from_dict_defaults_missing_axes_to_one(self):
        """A record written by an older build carries only the TP size."""
        plan = ParallelPlan.from_dict({"tensor_parallel_size": 4})
        assert plan.pipeline_parallel_size == 1
        assert plan.data_parallel_size == 1
        assert plan.world_size == 4

    def test_from_dict_survives_junk(self):
        plan = ParallelPlan.from_dict(
            {"tensor_parallel_size": "not-a-number", "pipeline_parallel_size": None}
        )
        assert plan.world_size == 1

    def test_to_dict_carries_everything_a_ui_needs(self):
        payload = plan_for("pipeline", 3).to_dict()
        assert payload == {
            "strategy": "pipeline",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 3,
            "data_parallel_size": 1,
            "world_size": 3,
            "label": "PP=3",
        }


def test_tensor_parallel_sizes_are_powers_of_two():
    """Guards the constant itself: the whole TP=3 refusal rests on it."""
    assert all(n & (n - 1) == 0 for n in TENSOR_PARALLEL_SIZES)
