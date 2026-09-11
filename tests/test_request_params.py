"""Malformed request bodies must produce a 4xx, never a 500.

`(body.get("model") or "").strip()` assumes the body decoded to an object and
the field holds a string. A client controls both. `[1, 2]` and
`{"model": 123}` raised AttributeError straight out of the handler, which
aiohttp turned into a 500 with a traceback — a plainly bad request reading as
a server fault.

These cover the coercion helpers directly, and then the endpoints that take a
model id, so a regression shows up as a failing status code rather than as a
log full of tracebacks.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from ainode.api.params import as_object, int_field, str_field, str_list_field
from ainode.core.config import NodeConfig
from ainode.discovery.broadcast import NodeAnnouncement
from ainode.discovery.cluster import ClusterState
from ainode.engine.sharding_routes import (
    handle_sharding_launch,
    handle_sharding_relaunch,
)
import ainode.engine.backends as backends


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


class TestAsObject:
    @pytest.mark.parametrize("value", [[1, 2], "hi", None, 42, True])
    def test_non_objects_become_empty(self, value):
        assert as_object(value) == {}

    def test_an_object_passes_through(self):
        assert as_object({"a": 1}) == {"a": 1}


class TestStrField:
    def test_returns_a_stripped_string(self):
        assert str_field({"model": "  org/m  "}, "model") == "org/m"

    @pytest.mark.parametrize("value", [123, {"a": 1}, ["m"], None, True])
    def test_non_strings_read_as_absent(self, value):
        assert str_field({"model": value}, "model") == ""

    def test_whitespace_only_reads_as_absent(self):
        assert str_field({"model": "   "}, "model") == ""

    def test_falls_through_to_the_next_name(self):
        """Several endpoints accept hf_repo or model_id."""
        assert str_field({"model_id": "org/m"}, "hf_repo", "model_id") == "org/m"
        assert str_field({"hf_repo": "org/a", "model_id": "org/b"},
                         "hf_repo", "model_id") == "org/a"

    def test_default_is_returned_when_nothing_matches(self):
        assert str_field({}, "model", default="auto") == "auto"

    def test_survives_a_non_object_body(self):
        assert str_field([1, 2], "model") == ""


class TestStrListField:
    def test_returns_the_strings(self):
        assert str_list_field({"node_ids": ["a", " b "]}, "node_ids") == ["a", "b"]

    def test_a_bare_string_is_not_a_one_element_list(self):
        """"head" would otherwise iterate into ['h','e','a','d'] and the error
        would name four nodes nobody asked for."""
        assert str_list_field({"node_ids": "head"}, "node_ids") == []

    def test_non_string_entries_are_dropped(self):
        assert str_list_field({"node_ids": ["a", None, 7, "", "b"]}, "node_ids") == ["a", "b"]

    @pytest.mark.parametrize("value", [{"a": 1}, 5, None])
    def test_non_lists_become_empty(self, value):
        assert str_list_field({"node_ids": value}, "node_ids") == []


class TestIntField:
    def test_parses_an_int_and_a_numeric_string(self):
        assert int_field({"n": 3}, "n") == 3
        assert int_field({"n": "3"}, "n") == 3

    def test_clamps_to_the_bounds(self):
        assert int_field({"n": -5}, "n", minimum=1) == 1
        assert int_field({"n": 10**9}, "n", maximum=64) == 64

    def test_booleans_are_rejected(self):
        """True is an int in Python, but a caller who sent `true` meant a flag."""
        assert int_field({"n": True}, "n", default=7) == 7

    @pytest.mark.parametrize("value", [{"a": 1}, ["3"], "abc", None])
    def test_junk_falls_back_to_the_default(self, value):
        assert int_field({"n": value}, "n", default=1) == 1


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, config, on_ready=None, instance_id=""):
        self.config = config

    def is_running(self):
        return False

    def stop(self):
        pass

    def start(self):
        # The solo path (min_nodes <= 1) reaches handle_model_load, which
        # starts the engine directly rather than via start_distributed.
        return True

    def start_distributed(self):
        return True

    def wait_ready(self, *a, **k):
        return True

    @property
    def ready(self):
        return True


def _call(handler, body):
    config = NodeConfig(node_id="head")
    config.save = lambda *a, **k: None
    cluster = ClusterState(local_announcement=NodeAnnouncement(
        node_id="head", node_name="head", gpu_name="NVIDIA GB10",
        gpu_memory_gb=128.0, unified_memory=True, model="", status="starting",
        api_port=8000, web_port=3000, distributed_mode="head",
    ))
    app = {"cluster_state": cluster, "config": config, "engine": None,
           "instances": None}

    class _Req:
        def __init__(self):
            self.app = app

        async def json(self):
            return body

    with patch.object(backends, "get_backend", _FakeBackend):
        return asyncio.run(handler(_Req()))


MALFORMED = [
    ("body is a list", [1, 2]),
    ("body is a string", "hello"),
    ("body is null", None),
    ("body is a number", 42),
    ("model is a dict", {"model": {"x": 1}}),
    ("model is a number", {"model": 42}),
    ("model is a list", {"model": ["a"]}),
    ("model is empty", {"model": "   "}),
]


class TestMalformedBodiesAreRejectedCleanly:
    @pytest.mark.parametrize("label,body", MALFORMED, ids=[m[0] for m in MALFORMED])
    def test_launch(self, label, body):
        resp = _call(handle_sharding_launch, body)
        assert resp.status == 400
        assert json.loads(resp.body)["error"] == "model field required"

    @pytest.mark.parametrize("label,body", MALFORMED, ids=[m[0] for m in MALFORMED])
    def test_relaunch(self, label, body):
        resp = _call(handle_sharding_relaunch, body)
        assert resp.status == 400
        assert json.loads(resp.body)["error"] == "model field required"

    def test_node_ids_as_a_string_does_not_become_four_nodes(self):
        resp = _call(handle_sharding_launch, {"model": "m", "node_ids": "head"})
        # Falls through to the single-node path rather than inventing h/e/a/d.
        assert resp.status != 422 or "'h'" not in json.loads(resp.body).get("error", "")

    def test_junk_min_nodes_does_not_raise(self):
        for value in ({"a": 1}, ["3"], "abc", -5, True):
            resp = _call(handle_sharding_launch, {"model": "m", "min_nodes": value})
            assert resp.status < 500


class TestStrategyTypeIsStillRejected:
    """str_field would silently default a bogus strategy to "auto" — handing
    the caller a working launch on an axis they did not ask for. The raw value
    goes to Strategy.parse, which rejects it."""

    @pytest.mark.parametrize("value", [123, {"a": 1}, ["tensor"]])
    def test_non_string_strategy_is_a_400(self, value):
        resp = _call(handle_sharding_launch,
                     {"model": "m", "node_ids": ["head", "x"], "strategy": value})
        assert resp.status == 400
        assert "tensor, pipeline, data, auto" in json.loads(resp.body)["error"]

    @pytest.mark.parametrize("value", [None, ""])
    def test_absent_strategy_means_auto(self, value):
        resp = _call(handle_sharding_launch,
                     {"model": "m", "node_ids": ["head", "x"], "strategy": value})
        # Gets past strategy parsing and fails on the unknown node instead.
        assert resp.status == 422


# ---------------------------------------------------------------------------
# Device names reach a shell — validate at both ends
# ---------------------------------------------------------------------------


class TestDeviceNameValidation:
    """coord_interface / rdma_hcas are written into the launcher .env, whose
    CONTAINER_* values upstream re-quotes by interpolating them into a Python
    one-liner. A single quote in the value closes that literal, so a crafted
    device name becomes code running on the head. PATCH /api/config is
    unauthenticated unless the operator enabled auth, so these must never be
    stored, and never written even if config.json was edited by hand."""

    ACCEPTED = ["enP7s7", "rocep1s0f1", "roceP2p1s0f0", "mlx5_0", "eth0.100", "bond-0"]
    REJECTED = [
        "x'+__import__(\"os\").popen(\"id\").read()+'",   # the working payload
        "eth0;id",
        "eth0 && touch /tmp/x",
        "eth 0",
        "../../etc/passwd",
        "a" * 16,          # IFNAMSIZ-1 is 15
        "",
        None,
        123,
        ["eth0"],
    ]

    @pytest.mark.parametrize("name", ACCEPTED)
    def test_real_device_names_pass(self, name):
        from ainode.cluster.topology import is_safe_device_name

        assert is_safe_device_name(name)

    @pytest.mark.parametrize("name", REJECTED)
    def test_hostile_or_malformed_names_fail(self, name):
        from ainode.cluster.topology import is_safe_device_name

        assert not is_safe_device_name(name)

    def test_patch_config_rejects_a_crafted_interface(self, tmp_path, monkeypatch):
        import asyncio as _aio

        from ainode.api.server import handle_patch_config
        from ainode.core.config import NodeConfig

        monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
        monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
        config = NodeConfig(node_id="head")
        payload = {"coord_interface": "x'+__import__(\"os\").popen(\"id\").read()+'"}

        class _Req:
            app = {"config": config}

            async def json(self):
                return payload

        resp = _aio.run(handle_patch_config(_Req()))
        body = json.loads(resp.body)
        assert "coord_interface" in body["rejected"]
        assert "coord_interface" not in body["applied"]
        assert config.coord_interface == ""      # never stored

    def test_patch_config_accepts_a_real_interface(self, tmp_path, monkeypatch):
        import asyncio as _aio

        from ainode.api.server import handle_patch_config
        from ainode.core.config import NodeConfig

        monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
        monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
        config = NodeConfig(node_id="head")

        class _Req:
            app = {"config": config}

            async def json(self):
                return {"coord_interface": "enP7s7", "rdma_hcas": ["mlx5_0"]}

        resp = _aio.run(handle_patch_config(_Req()))
        body = json.loads(resp.body)
        assert body["rejected"] == []
        assert config.coord_interface == "enP7s7"
        assert config.rdma_hcas == ["mlx5_0"]

    def test_patch_config_rejects_a_crafted_hca_in_a_list(self, tmp_path, monkeypatch):
        import asyncio as _aio

        from ainode.api.server import handle_patch_config
        from ainode.core.config import NodeConfig

        monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
        monkeypatch.setattr("ainode.core.config.CONFIG_FILE", tmp_path / "config.json")
        config = NodeConfig(node_id="head")

        class _Req:
            app = {"config": config}

            async def json(self):
                return {"rdma_hcas": ["mlx5_0", "mlx5_1'; id; '"]}

        resp = _aio.run(handle_patch_config(_Req()))
        # One bad entry rejects the whole list — a partially applied device
        # list would silently change the NCCL ring.
        assert "rdma_hcas" in json.loads(resp.body)["rejected"]
        assert config.rdma_hcas == []

    def test_the_env_writer_refuses_a_hand_edited_config(self, tmp_path, monkeypatch):
        """config.json is also editable directly, so the sink checks too."""
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.eugr import EugrBackend, EugrBackendError

        monkeypatch.setattr(
            "ainode.engine.backends.eugr.EUGR_ENV_FILE", tmp_path / "eugr.env"
        )
        config = NodeConfig(
            node_id="head", distributed_mode="head", peer_ips=["10.0.0.12"],
            coord_interface="eth0'; id; '", rdma_hcas=["mlx5_0"],
        )
        with pytest.raises(EugrBackendError, match="Refusing to write device name"):
            EugrBackend(config)._write_eugr_env()
        assert not (tmp_path / "eugr.env").exists()

    def test_models_dir_with_injected_docker_flags_is_refused(self, tmp_path, monkeypatch):
        """VLLM_SPARK_EXTRA_DOCKER_ARGS is expanded unquoted into `docker run`,
        so whitespace in models_dir injects flags. `-v /:/host` there is a host
        compromise, and models_dir is PATCHable."""
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.eugr import EugrBackend, EugrBackendError

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr("ainode.engine.backends.eugr.EUGR_LAUNCHER", launcher)
        monkeypatch.setattr(
            "ainode.engine.backends.eugr.EUGR_ENV_FILE", tmp_path / "eugr.env"
        )

        config = NodeConfig(
            node_id="head", distributed_mode="head", peer_ips=["10.0.0.12"],
            models_dir="/models -v /:/host --privileged",
        )
        backend = EugrBackend(config)
        with patch.object(EugrBackend, "_write_eugr_env", lambda self: None), \
             patch.object(EugrBackend, "_write_distributed_launch_script",
                          lambda self: tmp_path / "s.sh"), \
             pytest.raises(EugrBackendError, match="Refusing to pass models_dir"):
            backend.start_distributed()

    @pytest.mark.parametrize(
        "path", ["/root/.ainode/models", "/mnt/shared-models", "/opt/a-b_c.1"]
    )
    def test_ordinary_model_paths_are_accepted(self, path):
        from ainode.engine.backends.eugr import _is_safe_path_arg

        assert _is_safe_path_arg(path)
