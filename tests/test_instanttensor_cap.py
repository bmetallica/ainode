"""Every model that uses the InstantTensor loader caps its staging buffer.

An engine rebuild (torch 2.11 -> 2.13) left Qwen3.8 unable to load on a node
with 28 GB genuinely free:

    RuntimeError: buffer_size (5086090240 B) exceeds device memory budget
    (825161728 B)

and 862404608 B on the next attempt. A budget derived from
gpu-memory-utilization would be constant; one that moves between runs comes
from a runtime query — and on GB10 that query answers oddly, the same way
nvidia-smi reports [N/A] for memory here. Unified memory is the common
thread, and it is not something a catalog entry can fix.

What a catalog entry CAN do is stop asking for 5 GB in one piece. 64 MiB is
what the GLM recipe uses, on this hardware, for a 175 GB checkpoint.

Four models ship the loader, so four needed the cap: the failure was reported
for one of them and was waiting in the other three.
"""

from __future__ import annotations

import pytest

from ainode.models.registry import CURATED_CLUSTER_MODELS, FALLBACK_CATALOG

CAP = "67108864"


def _instanttensor_models():
    found = {}
    for source in (FALLBACK_CATALOG, CURATED_CLUSTER_MODELS):
        for key, info in source.items():
            args = list(info.extra_vllm_args or [])
            if "--load-format" in args and "instanttensor" in args:
                found[key] = info
    return found


class TestEveryUserOfTheLoaderIsCapped:
    def test_there_are_models_using_it(self):
        assert _instanttensor_models(), "the test has lost its subject"

    @pytest.mark.parametrize("key", sorted(_instanttensor_models()))
    def test_the_buffer_is_capped(self, key):
        info = _instanttensor_models()[key]
        assert (info.extra_env or {}).get("INSTANTTENSOR_BUFFER_SIZE") == CAP

    def test_the_cap_matches_the_one_proven_on_this_hardware(self):
        """GLM's recipe has loaded a 175 GB checkpoint with it."""
        glm = CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"]
        assert glm.extra_env["INSTANTTENSOR_BUFFER_SIZE"] == CAP

    def test_the_measurement_is_recorded_next_to_the_cap(self):
        """A number without its evidence is a number the next person changes
        back."""
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
                  "registry.py").read_text()
        assert "825161728" in source and "862404608" in source
        assert "moves\n            # between runs" in source

    def test_the_way_out_is_named(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
                  "registry.py").read_text()
        assert "drop:--load-format" in source
