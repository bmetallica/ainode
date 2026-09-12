"""Tool calling has to work without the operator knowing vLLM's parser names.

Open WebUI sends ``tool_choice: "auto"`` whenever a tool is attached to a chat.
vLLM refuses that unless the engine was started with --enable-auto-tool-choice
and a --tool-call-parser matching how the model emits calls, and it says so
with an error about server configuration that no amount of clicking can fix.
Only three curated models carried those flags; everything downloaded from HF
had tool calls simply not work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.models.api_routes import apply_catalog_recipe, apply_tool_calling
from ainode.models.tool_parsers import AUTO, OFF, parser_for_model, tool_call_args

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestParserMapping:
    @pytest.mark.parametrize("model,parser", [
        # Read off eugr/spark-vllm-docker's recipes (MIT), one per family.
        ("unsloth/Qwen3.8-27B-NVFP4", "qwen3_xml"),
        ("nvidia/Qwen3.6-35B-A3B-NVFP4", "qwen3_xml"),
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "qwen3_xml"),
        ("Qwen/Qwen3-Coder-Next-FP8", "qwen3_coder"),
        ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3_coder"),
        ("nvidia/Gemma-4-26B-A4B-NVFP4", "gemma4"),
        ("nvidia/GLM-5.2-NVFP4", "glm47"),
        ("QuantTrio/MiniMax-M2-AWQ", "minimax_m2"),
        ("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4"),
        ("stepfun-ai/Step-3.7-Flash-FP8", "step3p5"),
        ("openai/gpt-oss-120b", "openai"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama3_json"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ("NousResearch/Hermes-3-Llama-3.1-8B", "hermes"),
    ])
    def test_family_is_recognised(self, model, parser):
        assert parser_for_model(model) == parser

    def test_the_coder_rule_beats_the_generation_rule(self):
        # Qwen3-Coder is qwen3_coder even though the generic Qwen rule exists.
        assert parser_for_model("Qwen/Qwen3-Coder-480B") == "qwen3_coder"

    def test_an_unknown_model_gets_nothing(self):
        # Guessing would turn a launch that works into one that does not.
        assert parser_for_model("microsoft/phi-3-mini") == ""
        assert parser_for_model("some/random-model") == ""
        assert parser_for_model("") == ""


class TestFlagAssembly:
    def test_both_flags_are_added_together(self):
        args = tool_call_args("nvidia/Gemma-4-26B-A4B-NVFP4", [], AUTO)
        assert args == ["--tool-call-parser", "gemma4", "--enable-auto-tool-choice"]

    def test_an_existing_parser_is_never_second_guessed(self):
        args = tool_call_args("unsloth/Qwen3.8-27B-NVFP4",
                              ["--tool-call-parser", "hermes",
                               "--enable-auto-tool-choice"], AUTO)
        assert args == []

    def test_a_parser_without_the_switch_gets_the_switch(self):
        # One without the other is inert — exactly the failure being fixed.
        args = tool_call_args("x/y", ["--tool-call-parser", "hermes"], AUTO)
        assert args == ["--enable-auto-tool-choice"]

    def test_off_adds_nothing_even_for_a_known_family(self):
        assert tool_call_args("nvidia/Gemma-4-26B-A4B-NVFP4", [], OFF) == []

    def test_a_forced_parser_overrides_the_mapping(self):
        args = tool_call_args("unsloth/Qwen3.8-27B-NVFP4", [], "qwen3_coder")
        assert args[:2] == ["--tool-call-parser", "qwen3_coder"]

    def test_an_unknown_model_still_gets_a_forced_parser(self):
        args = tool_call_args("some/random-model", [], "hermes")
        assert args[:2] == ["--tool-call-parser", "hermes"]


class TestAppliedToLaunches:
    def test_an_uncurated_qwen_gets_tool_calling(self):
        # The whole point: a model nobody curated, downloaded from HF, answers
        # Open WebUI's tool requests without anyone typing a flag.
        overrides = apply_tool_calling("Qwen/Qwen3.6-35B-A3B-FP8", {})
        assert "--enable-auto-tool-choice" in overrides["extra_vllm_args"]

    def test_a_curated_recipe_keeps_its_own_parser(self):
        overrides, _ = apply_catalog_recipe("unsloth/Qwen3.8-27B-NVFP4", {}, None)
        overrides = apply_tool_calling("unsloth/Qwen3.8-27B-NVFP4", overrides)
        args = overrides["extra_vllm_args"]
        assert args.count("--tool-call-parser") == 1
        assert args.count("--enable-auto-tool-choice") == 1

    def test_applying_twice_changes_nothing(self):
        # A profile re-applied must not grow its argument list each time.
        first = apply_tool_calling("nvidia/Gemma-4-26B-A4B-NVFP4", {})
        second = apply_tool_calling("nvidia/Gemma-4-26B-A4B-NVFP4", dict(first))
        assert first["extra_vllm_args"] == second["extra_vllm_args"]

    def test_an_unknown_model_is_launched_exactly_as_before(self):
        assert apply_tool_calling("some/random-model", {}) == {}

    def test_off_is_honoured_end_to_end(self):
        overrides = apply_tool_calling("nvidia/Gemma-4-26B-A4B-NVFP4", {}, "off")
        assert overrides == {}


class TestUi:
    def test_the_field_exists_and_is_sent(self):
        assert 'id="launch-tool-calling"' in INDEX
        assert "advanced.tool_calling" in APP_JS

    def test_automatic_is_the_default_choice(self):
        # The empty option must come first, so the server decides unless the
        # operator overrides.
        select = INDEX.split('id="launch-tool-calling"', 1)[1]
        first_option = select.split("<option", 2)[1]
        assert 'value=""' in first_option and "Automatic" in first_option

    def test_off_is_offered(self):
        assert 'value="off"' in INDEX
