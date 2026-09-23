"""No curated model asks for the InstantTensor loader any more.

This file used to assert the opposite: that every model using the loader
capped its staging buffer at 64 MiB. The cap was added after

    RuntimeError: buffer_size (5086090240 B) exceeds device memory budget
    (825161728 B)

and it was reasoned from the GLM recipe, which carries the same variables
and serves with ``--load-format auto`` — so the cap had never been exercised
anywhere, and the claim it came with ("what has loaded a 175 GB checkpoint
here") described a launch that did not use the loader.

Measured since, on one node in one day, with every launch of Qwen3.8 that
had the loader on::

    07:08 ok   08:22 ok   09:26 ok
    13:41 .. 14:48   thirteen failures
    15:14 ok   15:25 ok   15:30 ok
    19:29 .. 20:15   six failures

Same flags, same checkpoint, same node. What moves is the budget — 0.47 to
1.70 GB across those failures — and it moves because it tracks what CUDA
reports free, which on unified memory is MemFree and not MemAvailable:
measured at 27.18 GB against 116 GB available, with 91 GB of page cache
between them. Reading a checkpoint off disk fills that cache, so a load can
take the room its own loader is about to ask for.

The variables are read — CONCURRENCY=1 halves the buffer — and none of them
bounds it. So the loader can be made to fit more often and not reliably, and
a curated recipe is a promise that a launch works.
"""

from __future__ import annotations

import pytest

from ainode.models.registry import CURATED_CLUSTER_MODELS, FALLBACK_CATALOG

#: Still the right value for anyone switching the loader on deliberately.
CAP = "67108864"


def _instanttensor_models():
    found = {}
    for source in (FALLBACK_CATALOG, CURATED_CLUSTER_MODELS):
        for key, info in source.items():
            args = list(info.extra_vllm_args or [])
            if "--load-format" in args and "instanttensor" in args:
                found[key] = info
    return found


class TestNoCuratedModelChoosesIt:
    def test_none_of_them_does(self):
        assert _instanttensor_models() == {}, (
            "a curated entry is a promise that the launch works, and this "
            "loader's success depends on a figure the engine will not explain")

    @pytest.mark.parametrize("key", sorted(
        k for k, v in CURATED_CLUSTER_MODELS.items()
        if (v.extra_env or {}).get("INSTANTTENSOR_BUFFER_SIZE")))
    def test_the_settings_stay_for_anyone_switching_it_on(self, key):
        """Dropping the flag and keeping the knobs is deliberate: an operator
        who adds --load-format instanttensor gets the configuration that
        makes it fit most often, rather than the bare default."""
        env = CURATED_CLUSTER_MODELS[key].extra_env
        assert env["INSTANTTENSOR_BUFFER_SIZE"] == CAP
        assert env["INSTANTTENSOR_BACKEND"] == "BUFFERED"


class TestTheReasonIsRecordedWhereItWillBeRead:
    def test_the_qwen_entry_carries_the_measurement(self):
        import inspect

        from ainode.models import registry

        source = inspect.getsource(registry)
        assert "coin flip" in source
        assert "15:14 ok" in source
