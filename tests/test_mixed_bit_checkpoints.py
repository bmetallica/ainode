"""A checkpoint the engine cannot parse is refused before two nodes start.

From the cluster, after a TP=2 launch of
``aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx`` had formed a Ray cluster
and read the weights:

    Value error, Unsupported weight_bits: 16, currently only support
    {2, 3, 4, 5, 6, 7, 8}

The 16 is not a mistake in the checkpoint. A mixed-bit model can be written
two ways round, and only one of them loads:

* quantized default, unquantized exceptions — vLLM's per-layer lookup treats
  ``bits >= 16`` in ``extra_config`` as "leave this one alone";
* unquantized default, quantized exceptions — which is this one, and the
  global value is validated in the config constructor before any per-layer
  entry is read.

Verified against vLLM's own INCConfig: SUPPORTED_BITS = {2..8} at
construction, and get_quant_method's ``bits >= 16`` branch further down. The
model card confirms it too — it names a vLLM fork, a GPTQ plugin at a tag,
and a custom loader.
"""

from __future__ import annotations

from ainode.models.quantization import (
    mixed_bit_widths,
    name_suggests_mixed_bits,
    quantization_verdict,
)

# The real shape, abbreviated: bits=16 globally, the widths per module.
MINIMAX = {
    "quantization_config": {
        "bits": 16,
        "data_type": "int",
        "quant_method": "autoround",
        "autoround_version": "0.13.0",
        "block_name_to_quantize": "model.language_model.layers",
        "extra_config": {
            "model.language_model.layers.0.self_attn.q_proj": {"bits": 4},
            "model.language_model.layers.10.mlp.experts.7.down_proj": {"bits": 2},
            "model.language_model.layers.12.mlp.experts.33.down_proj": {"bits": 3},
            "model.language_model.layers.3.mlp.gate": {"bits": 16},
        },
    }
}

# The layout vLLM does handle: a quantized default, exceptions left alone.
ORDINARY_MIXED = {
    "quantization_config": {
        "bits": 4,
        "quant_method": "autoround",
        "extra_config": {"lm_head": {"bits": 16}},
    }
}

PLAIN_AWQ = {"quantization_config": {"bits": 4, "quant_method": "awq",
                                     "group_size": 128}}


class TestTheVerdict:
    def test_the_unquantized_default_is_refused(self):
        servable, reason = quantization_verdict(MINIMAX)
        assert servable is False
        assert "bits=16" in reason

    def test_it_says_which_widths_are_really_in_there(self):
        # The part that is actually informative: a checkpoint advertised as
        # "3.2 bit" holds 2-, 3- and 4-bit modules.
        _, reason = quantization_verdict(MINIMAX)
        assert "2, 3, 4-bit" in reason

    def test_it_says_it_is_not_a_flag(self):
        _, reason = quantization_verdict(MINIMAX)
        assert "No serve flag changes that" in reason
        assert "model card" in reason

    def test_it_names_the_layout_that_would_work(self):
        # So the next checkpoint can be recognised from its config alone.
        _, reason = quantization_verdict(MINIMAX)
        assert "other way round" in reason

    def test_the_supported_layout_is_not_refused(self):
        assert quantization_verdict(ORDINARY_MIXED) == (True, "")

    def test_a_plain_quantized_checkpoint_is_not_refused(self):
        assert quantization_verdict(PLAIN_AWQ) == (True, "")

    def test_an_unquantized_model_is_not_refused(self):
        assert quantization_verdict({"model_type": "qwen3"}) == (True, "")

    def test_a_method_it_does_not_know_is_silence(self):
        # Never refuse what it does not understand: the operator would have no
        # way to launch a model that works.
        config = {"quantization_config": {"bits": 16, "quant_method": "something-new",
                                          "extra_config": {"a": {"bits": 4}}}}
        assert quantization_verdict(config) == (True, "")

    def test_bits_16_without_per_layer_entries_is_silence(self):
        # Nothing to refuse: that is just an unquantized config that happens
        # to say so.
        assert quantization_verdict(
            {"quantization_config": {"bits": 16, "quant_method": "autoround"}}
        ) == (True, "")

    def test_rubbish_is_silence(self):
        assert quantization_verdict(None) == (True, "")
        assert quantization_verdict({"quantization_config": "yes"}) == (True, "")


class TestTheWidths:
    def test_it_collects_the_distinct_ones(self):
        assert mixed_bit_widths(
            MINIMAX["quantization_config"]["extra_config"]) == [2, 3, 4]

    def test_sixteen_is_not_a_width(self):
        assert mixed_bit_widths({"a": {"bits": 16}}) == []

    def test_junk_entries_are_skipped(self):
        assert mixed_bit_widths({"a": "nope", "b": {"bits": "four"}}) == []


class TestTheName:
    def test_a_fractional_width_is_a_hint(self):
        assert name_suggests_mixed_bits("x/MiniMax-M3-AutoRound-3.2bit-longctx")

    def test_a_whole_one_is_not(self):
        assert not name_suggests_mixed_bits("x/Qwen3-4bit-AWQ")


class TestTheGateAsksBeforeTheMemoryQuestions:
    def test_the_admission_gate_consults_it(self):
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission.check_admission)
        assert "_quantization_says" in source
        # Before the guard: there is no amount of free memory that makes an
        # unreadable quantization_config readable.
        assert source.index("_quantization_says") < source.index("memory_guard")

    def test_force_still_overrides_it(self):
        # The operator may have put the plugin in the engine image since.
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission.check_admission)
        assert source.index("if force:") < source.index("_quantization_says")

    def test_it_refuses_a_real_directory(self, tmp_path):
        import json

        from ainode.safety.admission import check_admission

        (tmp_path / "config.json").write_text(json.dumps(MINIMAX))

        class _Manager:
            def model_dirs_for_repo(self, repo):
                return [tmp_path]

        refusal = check_admission({"model_manager": _Manager()}, "x/m")
        assert "mixed-bit" in refusal

    def test_a_checkpoint_it_can_read_falls_through(self, tmp_path):
        import json

        from ainode.safety.admission import check_admission

        (tmp_path / "config.json").write_text(json.dumps(PLAIN_AWQ))

        class _Manager:
            def model_dirs_for_repo(self, repo):
                return [tmp_path]

        assert check_admission({"model_manager": _Manager()}, "x/m") == ""
