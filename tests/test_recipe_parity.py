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
            QWEN, {"engine_image": "mine:1"}, 0.5)
        assert overrides["engine_image"] == "mine:1"
        assert gmu == 0.5

    def test_one_advanced_field_does_not_drop_the_whole_recipe(self):
        # Typing a concurrency limit for Qwen3.8 used to replace its recipe
        # wholesale — reasoning parser, tool-call parser and speculative
        # config gone — and the launch then died on the argparse error the
        # recipe exists to avoid.
        overrides, _ = apply_catalog_recipe(
            QWEN, {"extra_vllm_args": ["--max-num-seqs", "40"]}, None)
        args = overrides["extra_vllm_args"]
        assert args[:2] == ["--max-num-seqs", "40"]
        assert "--reasoning-parser" in args
        assert "--tool-call-parser" in args

    def test_a_caller_flag_overrides_the_recipe_flag_of_the_same_name(self):
        overrides, _ = apply_catalog_recipe(
            QWEN, {"extra_vllm_args": ["--reasoning-parser", "mine"]}, None)
        args = overrides["extra_vllm_args"]
        assert args.count("--reasoning-parser") == 1
        assert args[args.index("--reasoning-parser") + 1] == "mine"

    def test_caller_env_wins_but_the_rest_of_the_recipe_env_survives(self, monkeypatch):
        # No curated model needs extra_env today; the merge still has to be
        # right for the first one that does (the b12x FP4 path has no CLI
        # flags at all, only env).
        import ainode.models.api_routes as routes

        monkeypatch.setattr(routes, "catalog_recipe", lambda model: {
            "extra_env": {"KEEP": "1", "OVERRIDE": "recipe"}})
        overrides, _ = apply_catalog_recipe(
            "any/model", {"extra_env": {"OVERRIDE": "mine"}}, None)
        assert overrides["extra_env"] == {"KEEP": "1", "OVERRIDE": "mine"}

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


class TestGemma4Catalog:
    """The everyday chat model of the target deployment, from eugr's recipe."""

    GEMMA = "nvidia/Gemma-4-26B-A4B-NVFP4"

    def test_it_carries_its_parsers_and_drafter(self):
        args = catalog_recipe(self.GEMMA)["extra_vllm_args"]
        assert "--reasoning-parser" in args and "gemma4" in args
        assert "--tool-call-parser" in args
        assert any("gemma-4-26B-A4B-it-assistant" in a for a in args)

    def test_its_drafter_is_refused_with_the_base_named(self):
        # Loading the assistant on its own dies minutes later inside vLLM with
        # an AttributeError about draft_model_config.
        from ainode.models.api_routes import drafter_base_model

        assert drafter_base_model("google/gemma-4-26B-A4B-it-assistant") == self.GEMMA
        assert drafter_base_model(self.GEMMA) == ""

    def test_it_is_not_marked_verified(self):
        # It has not been served on this hardware by us; the badge must not
        # claim otherwise just because the recipe is upstream-proven.
        from ainode.models.registry import CURATED_CLUSTER_MODELS

        assert CURATED_CLUSTER_MODELS["gemma4-26b-a4b-nvfp4"].verified is False


class TestGemma31B:
    """The dense sibling, which is a different model from the curated MoE.

    Launched from the UI with no recipe at all — no parsers, no loader, no
    memory fraction — because the catalog only knew the 26B-A4B. It then died
    in CUDA-graph capture, which looked like the model being broken.
    """

    GEMMA31 = "nvidia/Gemma-4-31B-IT-NVFP4"

    def test_it_has_a_recipe_now(self):
        assert catalog_recipe(self.GEMMA31)["extra_vllm_args"]

    def test_it_carries_the_family_parsers(self):
        args = catalog_recipe(self.GEMMA31)["extra_vllm_args"]
        assert args[args.index("--tool-call-parser") + 1] == "gemma4"
        assert args[args.index("--reasoning-parser") + 1] == "gemma4"

    def test_it_does_not_inherit_the_moe_drafter(self):
        # The 26B-A4B recipe's speculative model belongs to that model. Pairing
        # it with this one fails in a way that reads like a broken model.
        args = catalog_recipe(self.GEMMA31)["extra_vllm_args"]
        assert not any("speculative" in a for a in args)
        assert not any("assistant" in a for a in args)

    def test_it_is_a_separate_entry_from_the_moe(self):
        from ainode.models.registry import CURATED_CLUSTER_MODELS

        moe = CURATED_CLUSTER_MODELS["gemma4-26b-a4b-nvfp4"]
        dense = CURATED_CLUSTER_MODELS["gemma4-31b-it-nvfp4"]
        assert moe.hf_repo != dense.hf_repo
        assert dense.verified is False

    def test_it_fits_one_node(self):
        from ainode.models.registry import CURATED_CLUSTER_MODELS

        dense = CURATED_CLUSTER_MODELS["gemma4-31b-it-nvfp4"]
        assert dense.proven_tp == 1
        assert dense.min_memory_gb < 122


class TestThePlanNoteReachesTheOperator:
    """A split that is not the one asked for has to be visible.

    Selecting three nodes for a model proven at two silently becomes a
    pipeline split. That is the right call and the wrong thing to keep in a
    log file: the operator is watching the dashboard, deciding whether the
    launch they just started is the one they meant.
    """

    def test_the_launch_response_carries_it(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "engine" /
               "sharding_routes.py").read_text()
        assert '"note": plan_note,' in src

    def test_the_ui_shows_it(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "if (data.note) this.toast(data.note" in app_js

    def test_a_normal_launch_has_no_note(self):
        # Only a downgrade is worth interrupting for.
        from ainode.engine.parallelism import Strategy, plan_for_model

        _, note = plan_for_model(Strategy.TENSOR, 2, proven_tp=2)
        assert note == ""
