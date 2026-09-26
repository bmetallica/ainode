"""The coding model this cluster can actually serve.

Asked, after MiniMax-M3-REAP50 turned out to be built for a patched SGLang:

    ich suche bessere (modernere) modelle welche auf den 2 nodes als
    programmier modelle (also eins) laufen könnten

Smaug-Flash is an agentic-coding finetune of DeepSeek-V4-Flash-0731 in the
same layout and the same quantization — which is the point: the base is
already served here, so the plan, the loader and the flags are known
quantities. That is not the same as having served THIS checkpoint, and the
catalog must not let the two blur.
"""

from __future__ import annotations

import pytest

from ainode.models.registry import CURATED_CLUSTER_MODELS, MODEL_CATALOG


def _entry(key):
    return CURATED_CLUSTER_MODELS.get(key) or MODEL_CATALOG.get(key)


@pytest.fixture
def smaug():
    entry = _entry("smaug-flash")
    assert entry is not None, "smaug-flash is not in the catalog"
    return entry


@pytest.fixture
def base():
    entry = _entry("deepseek-v4-flash-dspark")
    assert entry is not None
    return entry


class TestWhatItIs:
    def test_it_points_at_the_finetune_not_the_base(self, smaug):
        assert smaug.hf_repo == "abacusai/Smaug-Flash"

    def test_the_size_is_the_hubs_own_figure(self, smaug):
        # 166.9 GB, read from the file listing — the same as the base, which
        # the card says explicitly ("same layout, same quantization formats").
        assert smaug.size_gb == 166.9

    def test_it_needs_more_than_one_node(self, smaug):
        assert smaug.min_memory_gb > 128
        assert smaug.proven_tp == 2

    def test_it_is_a_coding_model(self, smaug):
        assert "code" in smaug.capabilities
        assert "tool_use" in smaug.capabilities


class TestItDoesNotClaimToHaveRun:
    def test_it_is_not_marked_verified(self, smaug):
        """The base has been served on this cluster. This has not, and the
        badge would say it had."""
        assert smaug.verified is False

    def test_the_description_says_so_in_words(self, smaug):
        assert "NOT YET SERVED HERE" in smaug.description

    def test_the_base_keeps_its_own_claim(self, base):
        # Nothing here may weaken what was actually measured.
        assert base.verified is True


class TestTheFlagsAreTheMeasuredOnes:
    def test_they_match_the_base_recipe(self, smaug, base):
        """Byte-compatible checkpoint, so the strongest available evidence is
        the base's measured recipe — not a recipe read off someone else's
        cluster running a different engine image."""
        assert smaug.extra_vllm_args == base.extra_vllm_args
        assert smaug.recommended_gmu == base.recommended_gmu

    def test_expert_parallelism_is_on(self, smaug):
        # 256 routed experts. Without it every rank holds every expert.
        assert "--enable-expert-parallel" in smaug.extra_vllm_args

    def test_the_kv_dtype_is_the_measured_one_not_the_read_one(self, smaug):
        # fp8_ds_mla is very likely better and is not the default, because
        # swapping a measured flag for a read one is how a recipe stops
        # meaning anything.
        args = smaug.extra_vllm_args
        assert args[args.index("--kv-cache-dtype") + 1] == "fp8"

    def test_the_slow_four_bit_layout_is_not_used(self, smaug):
        assert "nvfp4_ds_mla" not in smaug.extra_vllm_args


class TestTheTrapsAreWrittenDown:
    SOURCE = None

    @classmethod
    def setup_class(cls):
        from pathlib import Path

        import ainode.models.registry as registry

        cls.SOURCE = Path(registry.__file__).read_text()

    def test_the_ragged_drafter_batch_is_noted_where_max_num_seqs_is_set(self):
        assert "uniform flattened" in self.SOURCE

    def test_the_mla_kernel_note_is_on_the_glm_entry(self):
        assert "dense multi-head attention, not sparse MLA" in self.SOURCE

    def test_the_draft_attention_trap_is_noted(self):
        assert "TRITON_ATTN" in self.SOURCE
        assert "collapses acceptance" in self.SOURCE
