"""Three numbers, three sources, none of them the launch.

Reported from the cluster, after loading Qwen3-Coder-Next with two concurrent
requests and a hand-set context:

    Server / Load:  Context length 4096 · GPU layers -1 · Parallel 1
    opencode.json:  "context": 14336, "output": 6144, "tool_call": false

    das passt doch alles irgendwie nicht zusammen. oder verstehe ich da was
    falsch?

No. Two of the three were wrong, in different ways:

* The Load tab was three literals in the template. 4096 and -1 were typed
  into the HTML and sat under a real model claiming to describe it. GPU layers
  is a llama.cpp knob that never applied to a vLLM at all.
* "tool_call": false came from the curated catalog being the only source of
  capabilities — while AINode had itself chosen --tool-call-parser qwen3_coder
  for that launch. Two parts of one program disagreeing, with the wrong one
  written into the client config.

The 14336/6144 pair was right: it is exactly what limits_for(24576) returns,
so the instance really was running a 24576 window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.clients.opencode import _capabilities, limits_for

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
          / "js" / "app.js").read_text()


class TestTheNumbersTheOperatorSaw:
    def test_the_reported_pair_comes_from_a_24576_window(self):
        # Which is how we know that half of the report was not a bug.
        assert limits_for(24576) == {"context": 14336, "output": 6144}

    def test_a_bigger_window_gives_a_bigger_limit(self):
        assert limits_for(262144)["context"] > limits_for(24576)["context"]


class TestNothingInTheLoadPanelIsTyped:
    @pytest.mark.parametrize("literal", ['value="4096"', 'value="-1"',
                                         'value="0.7"', 'value="0.95"',
                                         'value="40"'])
    def test_the_invented_values_are_gone(self, literal):
        assert literal not in APP_JS

    def test_gpu_layers_is_gone_rather_than_zeroed(self):
        # It is a llama.cpp knob. Showing -1 implied the engine had one.
        assert "GPU layers" not in APP_JS

    def test_it_reads_the_instance_instead(self):
        for field in ("max_model_len", "kv_cache_dtype",
                      "gpu_memory_utilization", "max_num_seqs"):
            assert field in APP_JS, field

    def test_an_unreported_value_is_a_dash_not_a_guess(self):
        assert "_loadRow" in APP_JS
        assert "'—'" in APP_JS

    def test_it_says_where_to_look_when_it_does_not_know(self):
        assert "serve command" in APP_JS

    def test_sampling_is_described_as_per_request(self):
        # The inference tab invented a temperature the engine does not hold.
        assert "Sampling is per request" in APP_JS


class TestTheServerPayloadCarriesTheLaunch:
    def test_load_params_reads_the_config(self):
        from ainode.api.server_routes import load_params

        class _Cfg:
            max_model_len = 24576
            kv_cache_dtype = "fp8"
            gpu_memory_utilization = 0.77
            trust_remote_code = True
            extra_vllm_args = ["--max-num-seqs", "2"]

        out = load_params(_Cfg())
        assert out["max_model_len"] == 24576
        assert out["kv_cache_dtype"] == "fp8"
        assert out["gpu_memory_utilization"] == 0.77
        assert out["max_num_seqs"] == 2
        assert out["trust_remote_code"] is True

    def test_a_flag_fills_in_for_a_missing_field(self):
        from ainode.api.server_routes import load_params

        class _Cfg:
            extra_vllm_args = ["--max-model-len=131072",
                               "--kv-cache-dtype", "fp8_ds_mla"]

        out = load_params(_Cfg())
        assert out["max_model_len"] == 131072
        assert out["kv_cache_dtype"] == "fp8_ds_mla"

    def test_nothing_known_reports_none_not_zero(self):
        from ainode.api.server_routes import load_params

        out = load_params(None)
        assert out["max_model_len"] is None
        assert out["kv_cache_dtype"] is None
        assert out["gpu_memory_utilization"] is None

    def test_the_split_comes_from_the_record(self):
        from ainode.api.server_routes import load_params

        class _Rec:
            tensor_parallel_size = 2
            pipeline_parallel_size = 1

        assert load_params(None, _Rec())["tensor_parallel_size"] == 2


class TestOpencodeAsksTheLaunchNotTheCatalog:
    class _Entry:
        extra_vllm_args = ["--tool-call-parser", "qwen3_coder",
                           "--enable-auto-tool-choice"]

    def test_a_tool_call_flag_makes_it_true(self):
        tool_call, _, _ = _capabilities(None, self._Entry(), "x/y")
        assert tool_call is True

    def test_a_reasoning_flag_makes_it_true(self):
        class _E:
            extra_vllm_args = ["--reasoning-parser", "deepseek_v4"]

        _, reasoning, _ = _capabilities(None, _E(), "x/y")
        assert reasoning is True

    def test_the_family_table_answers_when_no_flag_does(self):
        """models/tool_parsers.py matches qwen3-?coder and the launcher uses
        it. The client config used to say false for the same model."""
        tool_call, _, _ = _capabilities(None, None, "Qwen/Qwen3-Coder-Next")
        assert tool_call is True

    def test_a_model_nothing_knows_stays_false(self):
        # Claiming tool support a model does not have is the opposite failure.
        tool_call, _, _ = _capabilities(None, None, "acme/mystery-1b")
        assert tool_call is False

    def test_the_catalog_still_answers_for_vision(self):
        class _Info:
            capabilities = ["vision", "tool_use"]
            extra_vllm_args = []

        tool_call, _, attachment = _capabilities(_Info(), None, "x/y")
        assert tool_call is True and attachment is True
