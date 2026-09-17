"""DeepSeek V4 Flash, as it actually serves on this cluster.

The launch that worked, from the engine's own non-default args:

    max_model_len 131072, tensor_parallel_size 2, enable_expert_parallel True,
    gpu_memory_utilization 0.87, kv_cache_dtype fp8, max_num_seqs 8,
    speculative_config {'method': 'dspark', 'num_speculative_tokens': 7,
                        'draft_sample_method': 'greedy'},
    tool_call_parser deepseek_v4

    GPU KV cache size: 1,226,910 tokens,
    Maximum concurrency for 131,072 tokens per request: 9.36x

Four things had to be got right first, and each is recorded beside the flag
it belongs to, because each cost a failed launch:

  * DeepGEMM has to load, which needed the engine image rebuilt;
  * expert parallelism, or the weights do not fit;
  * max_model_len 131072 rather than the model's advertised 1048576;
  * and the client has to know the model reasons, or answers arrive empty.
"""

from __future__ import annotations

from ainode.models.registry import CURATED_CLUSTER_MODELS

ENTRY = CURATED_CLUSTER_MODELS["deepseek-v4-flash-dspark"]
ARGS = ENTRY.extra_vllm_args


def _value(flag: str) -> str:
    return ARGS[ARGS.index(flag) + 1]


class TestTheProvenLaunch:
    def test_it_is_marked_served(self):
        """It answered on two Sparks; the flag was False until that was true."""
        assert ENTRY.verified is True
        assert ENTRY.proven_tp == 2

    def test_expert_parallelism_is_on(self):
        """256 experts with 6 active. Without it every rank holds every
        expert and 155 GiB does not fit two nodes."""
        assert "--enable-expert-parallel" in ARGS

    def test_the_context_is_pinned_below_the_model_maximum(self):
        """The model advertises 1M and cannot serve it here: at 1048576 the
        bookkeeping leaves 3.37 GiB of cache and the launch is refused."""
        assert _value("--max-model-len") == "131072"
        assert ENTRY.context_length == 1048576      # what the model claims

    def test_the_reason_for_that_gap_is_written_down(self):
        """Otherwise the next reader raises it back to the advertised figure."""
        import inspect

        from ainode.models import registry

        source = inspect.getsource(registry)
        assert "3.37 GiB" in source and "1,226,910 tokens" in source

    def test_the_drafter_ships_in_the_checkpoint(self):
        import json

        spec = json.loads(_value("--speculative-config"))
        assert spec["method"] == "dspark"
        assert spec["num_speculative_tokens"] == 7

    def test_remote_code_is_allowed(self):
        assert "--trust-remote-code" in ARGS

    def test_tool_calling_is_wired_to_its_own_parser(self):
        assert "--enable-auto-tool-choice" in ARGS
        assert _value("--tool-call-parser") == "deepseek_v4"


class TestWhatAnOperatorNeedsToKnow:
    def test_the_size_is_the_measured_one(self):
        """usedStorage, not the dtype estimate that called this 306 GB and
        hid it from the search."""
        assert ENTRY.size_gb == 166.9

    def test_the_description_warns_that_it_reasons(self):
        """A client that does not expect it shows empty answers — the same
        failure GLM produced, reported twice before it was recognised."""
        assert "reasons before answering" in ENTRY.description

    def test_the_licence_is_named(self):
        """MIT matters here: the alternative of similar capability is under a
        non-commercial community licence."""
        assert ENTRY.license == "MIT"

    def test_the_measured_capacity_is_stated(self):
        assert "9.36x" in ENTRY.description
