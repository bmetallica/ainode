"""The image-generation engine: the server, and the backend that drives it.

The server runs inside the engine container and imports nothing from AINode —
it has to work under whatever Python that image has. So it is tested the way
it runs: by calling its functions with a stand-in pipeline.

The backend is tested for the contract that earns image generation everything
AINode does around instances: the same EngineBackend interface, the same load
phases, and a kill() the memory guard can use.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_server():
    """Import the server the way the container does — as a standalone file."""
    path = ROOT / "ainode" / "engine" / "diffusers_server.py"
    spec = importlib.util.spec_from_file_location("ainode_image_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ainode_image_server"] = module
    spec.loader.exec_module(module)
    return module


server = _load_server()


class _Image:
    def save(self, buffer, format="PNG"):
        buffer.write(b"\x89PNG-fake")


class _Result:
    def __init__(self, count=1):
        self.images = [_Image() for _ in range(count)]


class _Pipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _Result(kwargs.get("num_images_per_prompt", 1))


@pytest.fixture(autouse=True)
def _clean_state():
    server.STATE.update({"ready": True, "error": "", "model": "org/m",
                         "pipeline": _Pipeline(), "images": 0, "seconds": 0.0,
                         "steps": 0, "running": 0, "max_pixels": 1536 ** 2,
                         "default_steps": 20, "default_size": "1024x1024"})
    yield


class TestTheServerImportsNothingFromAINode:
    def test_it_stands_alone(self):
        # It runs inside the engine container, where ainode is not installed.
        source = (ROOT / "ainode" / "engine" / "diffusers_server.py").read_text()
        assert "from ainode" not in source
        assert "import ainode" not in source

    def test_it_is_mounted_rather_than_baked_into_the_image(self):
        # Changing how images are served must not mean rebuilding twenty
        # gigabytes.
        dockerfile = (ROOT / "scripts" / "Dockerfile.diffusers").read_text()
        assert "the backend supplies the server" in dockerfile.lower()


class TestGenerating:
    def test_a_prompt_comes_back_as_base64_png(self):
        out = server.generate({"prompt": "a red cube"})
        assert out["data"][0]["b64_json"]
        assert base64.b64decode(out["data"][0]["b64_json"]).startswith(b"\x89PNG")
        assert out["model"] == "org/m"

    def test_the_defaults_are_the_instance_defaults(self):
        server.generate({"prompt": "x"})
        call = server.STATE["pipeline"].calls[0]
        assert call["width"] == 1024 and call["height"] == 1024
        assert call["num_inference_steps"] == 20

    def test_the_diffusion_knobs_are_accepted(self):
        server.generate({"prompt": "x", "steps": 8, "guidance_scale": 3.5,
                         "negative_prompt": "blurry", "size": "512x768"})
        call = server.STATE["pipeline"].calls[0]
        assert call["num_inference_steps"] == 8
        assert call["guidance_scale"] == 3.5
        assert call["negative_prompt"] == "blurry"
        assert (call["width"], call["height"]) == (512, 768)

    def test_a_prompt_is_required(self):
        with pytest.raises(ValueError):
            server.generate({})

    def test_a_nonsense_size_is_refused_clearly(self):
        with pytest.raises(ValueError) as exc:
            server.generate({"prompt": "x", "size": "huge"})
        assert "1024x1024" in str(exc.value)

    def test_the_timing_comes_back_with_the_image(self):
        out = server.generate({"prompt": "x"})
        assert "seconds" in out["ainode"]
        assert out["ainode"]["size"] == "1024x1024"


class TestTheResolutionLimit:
    """The counterpart to --max-model-len: the one knob that stops a single
    request taking the node down. A diffusion peak is activations plus the VAE
    decode, and both grow with the square of the edge."""

    def test_an_oversized_request_is_refused(self):
        server.STATE["max_pixels"] = 1024 ** 2
        with pytest.raises(ValueError) as exc:
            server.generate({"prompt": "x", "size": "2048x2048"})
        assert "exceeds this instance's limit" in str(exc.value)

    def test_the_refusal_says_what_to_change(self):
        server.STATE["max_pixels"] = 1024 ** 2
        with pytest.raises(ValueError) as exc:
            server.generate({"prompt": "x", "size": "2048x2048"})
        assert "max_image_size" in str(exc.value)

    def test_it_is_a_pixel_budget_not_an_edge(self):
        # 2048x512 has the same area as 1024x1024 and the same peak.
        server.STATE["max_pixels"] = 1024 ** 2
        server.generate({"prompt": "x", "size": "2048x512"})

    def test_no_limit_means_no_limit(self):
        server.STATE["max_pixels"] = 0
        server.generate({"prompt": "x", "size": "4096x4096"})


class TestConcurrency:
    def test_runs_do_not_overlap_in_the_gpu(self):
        # Two concurrent diffusion runs on a unified-memory node do not halve
        # the time, they double the peak — and the peak is what kills a node.
        assert server.GPU_LOCK is not None
        source = (ROOT / "ainode" / "engine" / "diffusers_server.py").read_text()
        assert "with GPU_LOCK:" in source


class TestMetrics:
    def test_nothing_generated_yet_still_answers(self):
        assert "ainode:images_generated_total 0" in server.metrics()

    def test_it_counts_what_it_made(self):
        server.generate({"prompt": "x", "n": 2})
        text = server.metrics()
        assert "ainode:images_generated_total 2" in text
        assert "ainode:seconds_per_image" in text
        assert "ainode:steps_per_second" in text

    def test_it_is_prometheus_text(self):
        for line in server.metrics().splitlines():
            assert line.startswith("#") or len(line.split()) == 2


class TestReadiness:
    def test_v1_models_answers_503_until_the_weights_are_in(self):
        # AINode's readiness probe is this endpoint. Answering 200 early
        # would mark the instance ready while it is still loading.
        source = (ROOT / "ainode" / "engine" / "diffusers_server.py").read_text()
        block = source.split('if self.path.startswith("/v1/models")')[1][:400]
        assert "503" in block

    def test_the_load_speaks_the_phrases_the_tracker_knows(self):
        # So the card shows a load as a load, with no special casing.
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("Loading model weights from /models/org--m")
        assert tracker.phase == "loading_weights"
        assert tracker.observe("Application startup complete.")

    def test_a_failed_load_speaks_them_too(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe("Engine core initialization failed: OSError: no such file")
        tracker.fail("the image engine exited (code 1)")
        assert "OSError" in tracker.failure_reason()


class TestNoCpuOffload:
    def test_the_standard_advice_is_explicitly_not_taken(self):
        # On GB10 the CPU and the GPU share one physical pool, so offloading
        # moves nothing and pays for the copies. It is the right advice
        # everywhere else, which is exactly why it needs saying here.
        source = (ROOT / "ainode" / "engine" / "diffusers_server.py").read_text()
        assert "enable_model_cpu_offload" in source      # named…
        assert 'pipeline.to("cuda")' in source
        assert "one physical pool" in source             # …and refused


class _Config:
    node_id = "n1"
    model = "org/qwen-image"
    api_port = 8003
    models_dir = "/models-host"
    engine_image = ""
    extra_env = None
    max_image_size = 1536
    image_steps = 20
    image_size = "1024x1024"
    image_dtype = "bfloat16"


class TestTheBackend:
    def _backend(self, **kw):
        from ainode.engine.backends.diffusers import DiffusersBackend

        config = _Config()
        for key, value in kw.items():
            setattr(config, key, value)
        return DiffusersBackend(config, instance_id="8003")

    def test_it_is_reachable_through_get_backend(self):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends import DiffusersBackend, get_backend

        backend = get_backend(NodeConfig(node_id="n", model="org/m",
                                         engine_backend="diffusers"))
        assert isinstance(backend, DiffusersBackend)

    def test_an_unknown_backend_still_names_the_valid_ones(self):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends import get_backend

        with pytest.raises(ValueError) as exc:
            get_backend(NodeConfig(node_id="n", engine_backend="comfy"))
        assert "diffusers" in str(exc.value)

    def test_each_instance_gets_its_own_container_and_log(self):
        backend = self._backend()
        assert backend.container_name == "ainode_image-8003"
        assert backend.log_path.name == "images-8003.log"

    def test_a_missing_image_says_what_to_do(self):
        # It is built locally and is in no registry, so a docker pull was
        # never going to work. Failing at one would be the least useful thing
        # this could do.
        from ainode.engine.backends import diffusers as module

        backend = self._backend()
        with mock.patch.object(module, "_image_present", return_value=False):
            with pytest.raises(module.DiffusersBackendError) as exc:
                backend.start()
        assert "build-diffusers-image.sh" in str(exc.value)
        assert "update-cluster.sh --images" in str(exc.value)

    def test_a_missing_model_says_why_it_cannot_be_fetched_on_demand(self):
        from ainode.engine.backends import diffusers as module

        backend = self._backend()
        with mock.patch.object(module, "_image_present", return_value=True), \
             mock.patch.object(type(backend), "_model_path", lambda self: None):
            with pytest.raises(module.DiffusersBackendError) as exc:
                backend.start()
        assert "not on this node's disk" in str(exc.value)

    def test_the_model_path_is_the_containers_view(self, tmp_path):
        # models_dir does not exist inside the engine container; /models does.
        from ainode.engine.backends.diffusers import DiffusersBackend

        directory = tmp_path / "Qwen--Qwen-Image-2.1"
        directory.mkdir()
        (directory / "model_index.json").write_text("{}")

        config = _Config()
        config.models_dir = str(tmp_path)
        config.model = "Qwen/Qwen-Image-2.1"
        backend = DiffusersBackend(config)
        assert backend._model_path() == "/models/Qwen--Qwen-Image-2.1"

    def test_a_directory_without_a_pipeline_is_not_a_model(self, tmp_path):
        from ainode.engine.backends.diffusers import DiffusersBackend

        (tmp_path / "Qwen--Qwen-Image-2.1").mkdir()
        config = _Config()
        config.models_dir = str(tmp_path)
        config.model = "Qwen/Qwen-Image-2.1"
        assert DiffusersBackend(config)._model_path() is None

    def test_the_hub_cache_layout_is_followed(self, tmp_path):
        from ainode.engine.backends.diffusers import DiffusersBackend

        snapshot = tmp_path / "models--Qwen--Qwen-Image-2.1" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "model_index.json").write_text("{}")
        config = _Config()
        config.models_dir = str(tmp_path)
        config.model = "Qwen/Qwen-Image-2.1"
        path = DiffusersBackend(config)._model_path()
        assert path is not None and path.endswith("/snapshots/abc")

    def test_it_has_the_kill_the_memory_guard_needs(self):
        # The guard calls kill(), not stop(): a node seconds from death does
        # not have the minute a graceful teardown costs.
        backend = self._backend()
        assert callable(backend.kill)
        source = (ROOT / "ainode" / "engine" / "backends" /
                  "diffusers.py").read_text()
        kill = source.split("def kill(self)")[1].split("def ")[0]
        assert "graceful=False" in kill

    def test_it_reports_the_load_state_the_card_reads(self):
        backend = self._backend()
        for field in ("load_phase", "load_detail", "load_error",
                      "load_timeline", "load_seconds"):
            assert hasattr(backend, field), field

    def test_the_command_carries_the_resolution_limit(self):
        from ainode.engine.backends import diffusers as module

        backend = self._backend(max_image_size=1024)
        captured = {}

        class _Proc:
            stdout = None

            def poll(self):
                return None

        def _popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return _Proc()

        with mock.patch.object(module, "_image_present", return_value=True), \
             mock.patch.object(type(backend), "_model_path",
                               lambda self: "/models/x"), \
             mock.patch.object(type(backend), "_stage_server_script",
                               lambda self: Path("/tmp/s.py")), \
             mock.patch.object(type(backend), "_remove_container",
                               lambda self, graceful=False: None), \
             mock.patch.object(module.subprocess, "Popen", _popen):
            backend.start()
        cmd = captured["cmd"]
        assert "--max-image-size" in cmd
        assert cmd[cmd.index("--max-image-size") + 1] == "1024"
        assert "--network" in cmd and "host" in cmd
        assert "--gpus" in cmd
