"""MiniMax-M2.7's catalog entry, checked against the checkpoint.

Three of its fields were wrong in ways that change a capacity plan:

  * hf_repo named demon-zombie/..., which now 307-redirects to et0dev/...
  * size_gb said 120.0 — 230B x 4 bits, the same arithmetic that put Gemma 4
    31B at half its real size. The Hub's file listing says 111.6 GB across 24
    shards.
  * context_length said 131072; the checkpoint's max_position_embeddings is
    196608.

The size is the one that decides whether it runs here at all: 111.6 GB over
two nodes is 55.8 GB of weights per node against ~106 GB addressable, which
leaves room for a real KV cache. GLM, at 175 GB, does not.
"""

from __future__ import annotations

from ainode.models.registry import CURATED_CLUSTER_MODELS

ENTRY = CURATED_CLUSTER_MODELS["minimax-m2.7-awq"]

# From config.json: 62 layers, 8 KV heads, head_dim 128, K and V.
KV_BYTES_PER_TOKEN_FP8 = 2 * 62 * 8 * 128


class TestTheFieldsMatchTheCheckpoint:
    def test_the_repo_is_the_one_that_answers(self):
        assert ENTRY.hf_repo == "et0dev/MiniMax-M2.7-AWQ-4bit"

    def test_the_size_is_the_measured_one(self):
        assert ENTRY.size_gb == 111.6

    def test_the_context_is_the_checkpoints(self):
        assert ENTRY.context_length == 196608


class TestItFitsTwoNodes:
    def test_the_weights_leave_room_for_a_cache(self):
        """The arithmetic the entry claims, so it cannot rot silently."""
        addressable_per_node = 122 * 0.87          # GB, at recommended_gmu
        weights_per_node = ENTRY.size_gb / 2
        overhead = 8                                # activations, graphs, NCCL
        kv_per_node = addressable_per_node - weights_per_node - overhead
        assert kv_per_node > 35, kv_per_node
        total_kv_bytes = 2 * kv_per_node * 1024 ** 3
        assert total_kv_bytes / KV_BYTES_PER_TOKEN_FP8 > 600_000

    def test_two_nodes_split_the_kv_heads_evenly(self):
        # 8 KV heads over TP=2 is 4 each. A rank count above the KV-head count
        # replicates instead of splitting, and the cache stops scaling.
        assert 8 % ENTRY.proven_tp == 0

    def test_it_is_not_claimed_as_verified(self):
        # Nobody has served it here yet.
        assert ENTRY.verified is False


class TestTheRecipe:
    def test_fp8_kv_is_asked_for(self):
        # Text-only, so the vision-model fp8 corruption does not apply, and
        # fp8 halves the 124 KiB per token this model needs.
        args = ENTRY.extra_vllm_args
        assert args[args.index("--kv-cache-dtype") + 1] == "fp8"

    def test_the_description_carries_the_kv_arithmetic(self):
        # A capacity claim without its inputs cannot be checked by the next
        # reader, and this catalog has already carried one that was wrong.
        for fact in ("124 KiB per token", "8 KV heads", "62 layers"):
            assert fact in ENTRY.description, fact
