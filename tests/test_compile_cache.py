"""The engine's compile cache belongs to AINode, and can be cleared from the UI.

The launcher mounts $HOME/.cache into the engine container. Called from inside
this container, $HOME is /root — which the Docker daemon resolves on the HOST,
so the caches landed in root's home on the host: invisible to the operator,
outside every backup, and unreachable from here.

That matters because a stale entry is a real failure mode. The cache keys on
the model and its settings but not on the toolchain that built the kernels, so
changing the engine image can leave a kernel the GPU refuses to execute — an
illegal instruction deep inside a Triton launcher, with nothing pointing at the
cache. The remedy has to be a button, not a docker command.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


from ainode.core.config import NodeConfig


class TestTheCachesAreOurs:
    def test_they_live_under_ainode_home(self, monkeypatch, tmp_path):
        # Patched rather than reloaded: reloading the config module rebinds
        # AINODE_HOME for every other test in the session.
        import ainode.engine.backends.eugr as eugr

        monkeypatch.setattr(eugr, "ENGINE_CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(
            eugr, "host_path",
            lambda p: p.replace(str(tmp_path), "/home/admin/.ainode"))

        backend = eugr.EugrBackend.__new__(eugr.EugrBackend)
        backend.config = NodeConfig(model="org/model")
        mounts = " ".join(backend._cache_mounts())
        assert "/home/admin/.ainode/cache/vllm:/root/.cache/vllm" in mounts
        assert "/home/admin/.ainode/cache/flashinfer" in mounts
        assert "/home/admin/.ainode/cache/triton:/root/.triton" in mounts
        assert "/root/.cache/vllm:/root/.cache/vllm" not in mounts

    def test_the_launcher_is_told_not_to_add_its_own(self):
        # Without --no-cache-dirs it mounts $HOME/.cache on top of ours.
        src = (Path(__file__).resolve().parent.parent / "ainode" / "engine" /
               "backends" / "eugr.py").read_text()
        assert src.count('"--no-cache-dirs"') == 2      # solo and distributed


class _Request:
    def __init__(self, app=None):
        self.app = app or {}

    async def json(self):
        return {}


class TestClearing:
    def _clear(self, monkeypatch, root: Path):
        import ainode.core.config as config_module
        from ainode.models.api_routes import handle_clear_compile_cache

        monkeypatch.setattr(config_module, "ENGINE_CACHE_DIR", root)
        return asyncio.run(handle_clear_compile_cache(_Request()))

    def test_it_removes_the_contents_and_reports_what_it_freed(
            self, monkeypatch, tmp_path):
        cache = tmp_path / "vllm" / "torch_compile_cache" / "abc"
        cache.mkdir(parents=True)
        (cache / "kernel.cubin").write_bytes(b"x" * 2048)

        response = self._clear(monkeypatch, tmp_path)
        payload = json.loads(response.body)
        assert response.status == 200
        assert payload["ok"] is True
        assert "vllm" in payload["cleared"]
        assert payload["freed_mb"] >= 0
        assert not cache.exists()
        # The directory itself stays, so the next launch's mount is not a
        # root-owned directory the daemon creates behind us.
        assert (tmp_path / "vllm").is_dir()

    def test_it_warns_that_the_next_launch_is_slower(self, monkeypatch, tmp_path):
        (tmp_path / "vllm").mkdir()
        payload = json.loads(self._clear(monkeypatch, tmp_path).body)
        assert "recompiles" in payload["note"]

    def test_an_empty_cache_is_not_an_error(self, monkeypatch, tmp_path):
        assert json.loads(self._clear(monkeypatch, tmp_path).body)["ok"] is True

    def test_a_missing_cache_directory_is_not_an_error(self, monkeypatch, tmp_path):
        assert json.loads(
            self._clear(monkeypatch, tmp_path / "nope").body)["ok"] is True


class TestItIsReachable:
    def test_the_routes_exist(self):
        from ainode.api.server import create_app

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/engine/compile-cache" in paths
        # Node-targeted, because the cache that matters is on the node that
        # failed — which may not be this one.
        assert "/api/cluster/compile-cache" in paths


WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestTheButton:
    def test_a_failed_instance_offers_it(self):
        # It moved from the card into the details dialog, which is where the
        # error it repairs is now shown — but it is still one click from a
        # failure, and it is still the node-targeted endpoint.
        assert "instance-cache-clear" in APP_JS
        assert "/api/cluster/compile-cache" in APP_JS

    def test_the_note_still_singles_out_kernel_faults(self):
        # A model that died on a bad flag does not need its cache cleared, so
        # the failure note keeps its trigger — now as a sentence pointing at
        # the button rather than a second button of its own. The wording
        # widened: see tests/test_compile_cache_without_a_crash.py — "illegal
        # memory access" is a different CUDA error from "illegal instruction"
        # and was not matched, on the one fault this cluster produced.
        assert "illegal (instruction|memory access|address)|compiled kernel" in APP_JS
        assert "first thing to try" in APP_JS

    def test_it_says_what_it_costs(self):
        assert "recompiles" in APP_JS


class TestQuotedArguments:
    """The advanced field has to be able to express a JSON argument."""

    def test_the_ui_sends_a_command_line_not_a_split_array(self):
        # Splitting on whitespace in the browser could not express
        # --compilation-config '{"mode":0}': the quotes travelled with the
        # value and vLLM got a string that was not JSON.
        assert "advanced.extra_vllm_args = extraArgs.trim()" in APP_JS
        assert "freeForm.value.trim().split(/\\s+/)" not in APP_JS

    def test_the_server_shell_splits_it(self):
        from ainode.models.api_routes import parse_load_overrides

        overrides, err = parse_load_overrides(
            {"extra_vllm_args": '--compilation-config \'{"mode":0}\' --enforce-eager'})
        assert err is None
        assert overrides["extra_vllm_args"] == [
            "--compilation-config", '{"mode":0}', "--enforce-eager"]

    def test_an_unbalanced_quote_is_a_400(self):
        from ainode.models.api_routes import parse_load_overrides

        overrides, err = parse_load_overrides({"extra_vllm_args": "--flag 'unclosed"})
        assert overrides is None
        assert err.status == 400
        assert "quotes" in json.loads(err.body)["error"]


class TestNoDuplicateFlags:
    """Every built-in flag has to be suppressible from extra_vllm_args.

    Measured, on a launch where the operator typed --gpu-memory-utilization
    in the advanced field:

        WARNING [argparse_utils.py:441] Found duplicate keys
                --gpu-memory-utilization

    The `wanted()` check exists for exactly this, and --gpu-memory-utilization
    sat outside it, written into the script template. It is the obvious flag
    for an operator to type, since it is the knob that decides whether a model
    fits at all.
    """

    def _script(self, tmp_path, **config_kw) -> str:
        from unittest import mock

        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr
        from ainode.engine.parallelism import ParallelPlan

        backend = eugr.EugrBackend(NodeConfig(
            node_id="n", model="org/model", gpu_memory_utilization=0.6,
            **config_kw))
        with mock.patch.object(eugr, "detect_gpu", return_value=None), \
             mock.patch.object(eugr, "EUGR_LAUNCHER", tmp_path / "launch-cluster.sh"), \
             mock.patch.object(type(backend), "_serve_target_and_name",
                               lambda s: ("org/model", "")):
            return backend._write_launch_script(ParallelPlan(), solo=True).read_text()

    def test_the_built_in_value_is_used_by_default(self, tmp_path):
        assert "--gpu-memory-utilization 0.6" in self._script(tmp_path)

    def test_a_supplied_one_suppresses_it(self, tmp_path):
        script = self._script(
            tmp_path, extra_vllm_args=["--gpu-memory-utilization", "0.87"])
        assert script.count("--gpu-memory-utilization") == 1
        assert "0.87" in script and "0.6" not in script

    def test_the_equals_form_suppresses_it_too(self, tmp_path):
        script = self._script(tmp_path, extra_vllm_args=["--gpu-memory-utilization=0.9"])
        assert script.count("--gpu-memory-utilization") == 1
