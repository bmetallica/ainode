"""Removing a recipe flag, not just overriding it.

Overriding was possible from the start. Removing was not — and there is no
value meaning "not set" for --quantization or --speculative-config. Diagnosing
a model whose recipe carries a flag its engine build cannot use needed exactly
that: GLM 5.3's launch died inside a helper that resolves the model string as
a Hugging Face repo id, and the only way to find out which flag called it was
to take them away one at a time.
"""

from __future__ import annotations

import pytest

from ainode.engine.serve_args import dropped_flags, merge_vllm_args
from ainode.models.api_routes import apply_catalog_recipe, parse_load_overrides

RECIPE = ["--quantization", "modelopt_mixed",
          "--speculative-config", '{"method":"mtp"}',
          "--moe-backend", "b12x"]


class TestDropping:
    def test_a_dropped_flag_and_its_value_are_gone(self):
        merged = merge_vllm_args(RECIPE, ["drop:--speculative-config"])
        assert "--speculative-config" not in merged
        assert '{"method":"mtp"}' not in merged

    def test_the_rest_of_the_recipe_survives(self):
        merged = merge_vllm_args(RECIPE, ["drop:--speculative-config"])
        assert merged == ["--quantization", "modelopt_mixed",
                          "--moe-backend", "b12x"]

    def test_the_marker_never_reaches_the_engine(self):
        # vLLM would reject it as an unknown argument, which is a worse
        # failure than the one being diagnosed.
        merged = merge_vllm_args(RECIPE, ["drop:--quantization"])
        assert not any(a.startswith("drop:") for a in merged)

    def test_several_at_once(self):
        merged = merge_vllm_args(
            RECIPE, ["drop:--speculative-config", "drop:--quantization",
                     "--load-format", "auto"])
        assert merged == ["--load-format", "auto", "--moe-backend", "b12x"]

    def test_dropping_and_setting_can_be_mixed(self):
        merged = merge_vllm_args(RECIPE, ["drop:--quantization",
                                          "--moe-backend", "marlin"])
        assert merged == ["--moe-backend", "marlin",
                          "--speculative-config", '{"method":"mtp"}']

    @pytest.mark.parametrize("spelling", [
        "drop:--speculative-config", "drop:speculative-config",
    ])
    def test_the_dashes_are_optional(self, spelling):
        assert "--speculative-config" in dropped_flags([spelling])

    def test_dropping_something_the_recipe_never_had_is_harmless(self):
        assert merge_vllm_args(RECIPE, ["drop:--enforce-eager"]) == RECIPE

    def test_it_does_nothing_without_a_recipe(self):
        assert merge_vllm_args([], ["drop:--quantization"]) == []


class TestThroughTheLaunchPath:
    QWEN = "unsloth/Qwen3.8-27B-NVFP4"

    def test_a_drop_reaches_the_merge(self):
        overrides, err = parse_load_overrides(
            {"extra_vllm_args": "drop:--speculative_config --enforce-eager"})
        assert err is None
        overrides, _ = apply_catalog_recipe(self.QWEN, overrides, None)
        args = overrides["extra_vllm_args"]
        assert "--enforce-eager" in args
        assert not any(a.startswith("drop:") for a in args)
        assert not any("qwen3_5_mtp" in a for a in args)

    def test_the_ui_documents_the_syntax(self):
        from pathlib import Path

        index = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                 "templates" / "index.html").read_text()
        assert "drop:--speculative-config" in index
