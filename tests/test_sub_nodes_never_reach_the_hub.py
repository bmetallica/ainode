"""A sub-node must get its weights from the head, or say it could not.

The operator's rule: only the head pulls from Hugging Face. A sub-node that
cannot get a checkpoint from the head raises an error there rather than
quietly downloading it.

That is the right way round, and the opposite of what the first version did.
Falling back to a download reads as robustness until you watch it happen: 1.1
MB/s unauthenticated, about five hours for a 19 GB checkpoint, with the card
reading "starting" the whole time. A failure visible in one second beats a
success indistinguishable from a hang for five hours.
"""

from __future__ import annotations

import inspect
from unittest import mock

from ainode.core.config import NodeConfig
from ainode.models.api_routes import _fetch_weights_from_a_peer

MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"


def _config(**kw):
    return NodeConfig(node_id="n3", ssh_user="admin", models_dir="/models", **kw)


class _Backend:
    _log_file = None

    def _progress(self, *a, **k):
        pass


def _call(config, outcome=None, raises=None):
    app = {"cluster_state": None}
    target = "ainode.engine.acquire.fetch_model_from_peer"
    patch = (mock.patch(target, side_effect=raises) if raises
             else mock.patch(target, return_value=outcome))
    with patch:
        return _fetch_weights_from_a_peer(app, _Backend(), MODEL, config)


class TestASubNode:
    """download_from_hub off: the head is the only source there is."""

    def test_a_successful_fetch_lets_the_launch_go_on(self):
        assert _call(_config(download_from_hub=False), outcome="fetched") is None

    def test_a_sync_does_too(self):
        assert _call(_config(download_from_hub=False), outcome="synced") is None

    def test_weights_already_here_are_enough(self):
        """A sync that found no peer to compare against is not a reason to
        refuse a model this node can already serve."""
        assert _call(_config(download_from_hub=False), outcome="present") is None

    def test_nobody_has_it_is_an_error(self):
        message = _call(_config(download_from_hub=False), outcome="absent")
        assert message and MODEL in message

    def test_the_error_says_what_to_do_about_it(self):
        message = _call(_config(download_from_hub=False), outcome="absent")
        assert "download" in message and "head" in message
        assert "/api/cluster/mirror-models" in message

    def test_a_transfer_failure_is_an_error_too(self):
        message = _call(_config(download_from_hub=False),
                        raises=RuntimeError("ssh: no such identity"))
        assert message and "no such identity" in message

    def test_a_missing_models_dir_is_an_error(self):
        config = _config(download_from_hub=False)
        config.models_dir = ""
        assert _call(config, outcome="absent")


class TestTheHead:
    """download_from_hub on, the default: downloading is slower, not wrong."""

    def test_nobody_has_it_is_not_an_error(self):
        assert _call(_config(), outcome="absent") is None

    def test_a_transfer_failure_is_not_an_error(self):
        assert _call(_config(), raises=RuntimeError("boom")) is None

    def test_the_default_allows_it(self):
        assert NodeConfig(node_id="n1").download_from_hub is True


class TestTheLaunchStops:
    def test_the_message_reaches_the_caller(self):
        from ainode.models import api_routes

        source = inspect.getsource(api_routes.append_solo_instance)
        assert "blocked = _fetch_weights_from_a_peer" in source
        assert '"ok": False, "error": blocked' in source
        # Before the engine is started, not after it has begun downloading.
        assert source.index("blocked = _fetch") < source.index("ok = backend.start()")

    def test_the_node_stops_advertising_the_model(self):
        """A refused launch must not leave the node claiming to serve it."""
        from ainode.models import api_routes

        source = inspect.getsource(api_routes.append_solo_instance)
        head = source[source.index("blocked = _fetch"):]
        assert head.index("_clear()") < head.index("ok = backend.start()")


class TestTheEngineIsOfflineToo:
    def test_a_sub_node_runs_the_engine_offline(self):
        from ainode.engine.backends.eugr import EugrBackend

        backend = EugrBackend(_config(download_from_hub=False))
        with mock.patch.object(EugrBackend, "_coord_interface", lambda s: None), \
             mock.patch.object(EugrBackend, "_nccl_ib_hca", lambda s: None):
            env = backend._build_env()
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"

    def test_the_head_is_not_forced_offline(self):
        from ainode.engine.backends.eugr import EugrBackend

        backend = EugrBackend(_config())
        with mock.patch.object(EugrBackend, "_coord_interface", lambda s: None), \
             mock.patch.object(EugrBackend, "_nccl_ib_hca", lambda s: None):
            env = backend._build_env()
        assert "HF_HUB_OFFLINE" not in env
