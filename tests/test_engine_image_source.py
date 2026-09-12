"""Where an engine image came from decides whether this backend uses it.

A catalog recipe pins `vllm/vllm-openai:v0.27.1` because the NVIDIA backend
defaults to a 0.17 build that rejects the model's flags. The eugr backend —
the default one — launches through eugr's launcher, whose default image
`vllm-node` is built here from vLLM main, satisfies the same requirement, and
is what eugr's own recipes for these very models name.

Using the pinned one there is not merely redundant. The launcher copies its
exec script into /workspace inside the container; `vllm-node` has that as its
working directory and `vllm/vllm-openai` does not have it at all. Observed on
hardware as:

    Copying launch script to head node...
    Error response from daemon: Could not find the file /workspace in container
    Error: docker cp to head node failed

— after a 20 GB pull of an image that was never going to work.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import eugr as E
from ainode.models.api_routes import apply_catalog_recipe, parse_load_overrides

QWEN = "unsloth/Qwen3.8-27B-NVFP4"


def _backend(**kwargs):
    backend = E.EugrBackend.__new__(E.EugrBackend)
    backend.config = NodeConfig(model=QWEN, **kwargs)
    return backend


class TestProvenanceIsRecorded:
    def test_a_recipe_image_is_marked_as_such(self):
        overrides, _ = apply_catalog_recipe(QWEN, {}, None)
        assert overrides["engine_image"]
        assert overrides["engine_image_source"] == "catalog"

    def test_a_caller_image_is_marked_as_such(self):
        overrides, err = parse_load_overrides({"engine_image": "mine:1"})
        assert err is None
        assert overrides["engine_image_source"] == "caller"

    def test_a_caller_image_survives_the_recipe(self):
        overrides, err = parse_load_overrides({"engine_image": "mine:1"})
        overrides, _ = apply_catalog_recipe(QWEN, overrides, None)
        assert overrides["engine_image"] == "mine:1"
        assert overrides["engine_image_source"] == "caller"

    def test_no_image_means_no_source(self):
        overrides, err = parse_load_overrides({})
        assert "engine_image_source" not in overrides


class TestEugrDeclinesACatalogImage:
    def test_the_launcher_is_not_told_about_it(self):
        backend = _backend(engine_image="vllm/vllm-openai:v0.27.1",
                           engine_image_source="catalog")
        assert backend._launcher_image_args() == []

    def test_and_it_is_not_pulled_or_shipped(self):
        # The same decision, or the head pulls 20 GB of an image the launcher
        # then does not use.
        backend = _backend(engine_image="vllm/vllm-openai:v0.27.1",
                           engine_image_source="catalog",
                           distributed_mode="head", peer_ips=["10.0.0.2"])
        with mock.patch.object(E, "ensure_local_image") as pull, \
             mock.patch.object(E, "ensure_peer_has_image") as ship:
            backend._distribute_engine_image_to_peers()
        pull.assert_not_called()
        ship.assert_not_called()

    def test_a_caller_image_is_honoured(self):
        backend = _backend(engine_image="mine:1", engine_image_source="caller")
        assert backend._launcher_image_args() == ["-t", "mine:1"]

    def test_an_unmarked_image_is_honoured(self):
        # Config written by an older build carries no source. Treating it as a
        # caller's choice keeps that install behaving as it did.
        backend = _backend(engine_image="mine:1")
        assert backend._launcher_image_args() == ["-t", "mine:1"]

    def test_no_image_means_the_launcher_default(self):
        assert _backend()._launcher_image_args() == []

    def test_an_unsafe_image_is_still_refused(self):
        backend = _backend(engine_image="mine:1; rm -rf /",
                           engine_image_source="caller")
        with pytest.raises(E.EugrBackendError):
            backend._launcher_image_args()


class TestTheNvidiaBackendIsUnaffected:
    def test_it_still_uses_the_catalog_image(self):
        # Its own default IS the 0.17 build the pin exists to escape.
        from ainode.engine.backends.nvidia import NvidiaBackend

        backend = NvidiaBackend.__new__(NvidiaBackend)
        backend.config = NodeConfig(model=QWEN, engine_image="vllm/vllm-openai:v0.27.1",
                                    engine_image_source="catalog")
        assert backend._engine_image() == "vllm/vllm-openai:v0.27.1"
