"""The KV dtype a launch really used — recorded, planned and chosen by one rule.

Seen in an export: unsloth/Qwen3.8-27B-NVFP4 ran with its recipe's
``--kv-cache-dtype auto`` and was recorded as fp8 (the node's setting), and
the planner planned it at fp8 too — the node's setting came before the
recipe. A form choice, the other way round, lost to the recipe's flag at
launch.
"""

from __future__ import annotations

import json

from ainode.core.config import NodeConfig


def _vision_dir(tmp_path, model="org/vlm"):
    directory = tmp_path / model.replace("/", "--")
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"vision_config": {}}))
    return str(tmp_path)


class TestTheRecord:
    def test_a_flag_in_the_extra_args_is_what_ran(self):
        from ainode.measure.recorder import launched_kv_dtype

        config = NodeConfig(kv_cache_dtype="fp8",
                            extra_vllm_args=["--kv-cache-dtype", "auto", "--max-num-seqs", "5"])
        assert launched_kv_dtype(config) == "auto"

    def test_a_vision_model_ran_at_auto_unless_asked_for_fp8(self, tmp_path):
        from ainode.measure.recorder import launched_kv_dtype

        models = _vision_dir(tmp_path)
        assert launched_kv_dtype(NodeConfig(model="org/vlm", models_dir=models,
                                            kv_cache_dtype="fp8")) == "auto"
        assert launched_kv_dtype(NodeConfig(model="org/vlm", models_dir=models,
                                            kv_cache_dtype="fp8",
                                            kv_cache_dtype_explicit=True)) == "fp8"

    def test_a_text_model_keeps_the_nodes_setting(self):
        from ainode.measure.recorder import launched_kv_dtype

        assert launched_kv_dtype(NodeConfig(kv_cache_dtype="fp8")) == "fp8"

    def test_the_launch_record_agrees(self):
        from ainode.measure.recorder import _launch_of

        config = NodeConfig(kv_cache_dtype="fp8", extra_vllm_args=["--kv-cache-dtype=auto"])
        assert _launch_of(None, config)["kv_cache_dtype"] == "auto"


class _Recipe:
    extra_vllm_args = ["--enable-prefix-caching", "--kv-cache-dtype", "auto"]


class TestThePlan:
    def test_the_recipe_comes_before_the_nodes_setting(self):
        from ainode.planner.api_routes import planning_kv_dtype

        app = {"config": NodeConfig(kv_cache_dtype="fp8")}
        assert planning_kv_dtype(app, _Recipe()) == "auto"

    def test_an_explicit_choice_comes_first(self):
        from ainode.planner.api_routes import planning_kv_dtype

        app = {"config": NodeConfig(kv_cache_dtype="fp8")}
        assert planning_kv_dtype(app, _Recipe(), "fp8") == "fp8"

    def test_a_vision_model_is_planned_at_auto(self, tmp_path):
        from ainode.planner.api_routes import planning_kv_dtype

        app = {"config": NodeConfig(kv_cache_dtype="fp8", models_dir=_vision_dir(tmp_path))}
        assert planning_kv_dtype(app, None, "", "org/vlm") == "auto"
        assert planning_kv_dtype(app, None, "", "org/text") == "fp8"


class TestTheLaunch:
    def test_a_form_choice_replaces_the_recipes_flag(self):
        from ainode.models.api_routes import apply_catalog_recipe

        overrides, _ = apply_catalog_recipe(
            "unsloth/Qwen3.8-27B-NVFP4",
            {"kv_cache_dtype": "fp8", "kv_cache_dtype_explicit": True}, None)
        args = overrides["extra_vllm_args"]
        assert "--kv-cache-dtype" not in args
        assert "--reasoning-parser" in args        # the rest of the recipe stays

    def test_without_one_the_recipe_decides(self):
        from ainode.models.api_routes import apply_catalog_recipe

        overrides, _ = apply_catalog_recipe("unsloth/Qwen3.8-27B-NVFP4", {}, None)
        args = overrides["extra_vllm_args"]
        assert args[args.index("--kv-cache-dtype") + 1] == "auto"
