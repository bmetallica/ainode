"""Four defects found by reading back over the image-generation work, and the
inconsistency the reading turned up in the older backend.

Each of them is the same shape: a field that exists at one end of a path and
is dropped somewhere along it, so the thing built on top of it silently never
works. They are the failures that tests of the individual pieces cannot see,
because every piece was right.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ainode.models.registry import ModelManager


class TestTheKindReachesTheBrowser:
    """The Images view asks "which loaded models make pictures". The answer
    travels record → announcement → /api/nodes → browser, and the projection
    in the middle listed its keys by hand and did not list this one. So the
    view was permanently empty, with every piece behind it correct."""

    def test_the_nodes_projection_carries_it(self):
        import inspect

        from ainode.api import server

        source = inspect.getsource(server.handle_nodes)
        assert '"kind": inst.get("kind")' in source

    def test_the_cluster_resources_projection_does_too(self):
        source = (Path(__file__).resolve().parent.parent / "ainode" / "api" /
                  "server.py").read_text()
        # Two projections, both hand-listed; a fix to one is not a fix.
        assert source.count('"kind": inst.get("kind")') == 2

    def test_the_card_builder_keeps_it(self):
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "kind: inst.kind || 'llm'," in app_js
        # And the port, which the details dialog shows and never received.
        assert "api_port: inst.api_port || 0," in app_js

    def test_the_view_reads_that_field(self):
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        block = app_js.split("imageInstances() {")[1].split("\n  },")[0]
        assert "inst.kind === 'image'" in block


class TestAModelNobodyCuratedStillGetsTheRightEngine:
    """A diffusion pipeline the operator downloaded themselves is in no
    catalog, so nothing said which engine to use — and the node default is
    vLLM, which cannot load one at all."""

    def _app(self, files):
        directory = Path(tempfile.mkdtemp())
        repo = directory / "someone--their-model"
        repo.mkdir()
        for name in files:
            (repo / name).write_text("{}")
        return {"model_manager": ModelManager(models_dir=str(directory))}

    def test_a_pipeline_on_disk_selects_the_image_engine(self):
        from ainode.models.api_routes import apply_detected_backend

        app = self._app(["model_index.json"])
        overrides = apply_detected_backend(app, "someone/their-model", {})
        assert overrides["engine_backend"] == "diffusers"

    def test_an_llm_on_disk_is_left_alone(self):
        from ainode.models.api_routes import apply_detected_backend

        app = self._app(["config.json"])
        assert apply_detected_backend(app, "someone/their-model", {}) == {}

    def test_an_explicit_choice_is_not_second_guessed(self):
        from ainode.models.api_routes import apply_detected_backend

        app = self._app(["model_index.json"])
        overrides = apply_detected_backend(
            app, "someone/their-model", {"engine_backend": "eugr"})
        assert overrides["engine_backend"] == "eugr"

    def test_the_load_path_calls_it_after_the_recipe(self):
        # The catalog wins where it has an opinion; this fills the silence.
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_model_load)
        assert "apply_detected_backend" in source
        assert source.index("apply_catalog_recipe") < \
            source.index("apply_detected_backend")

    def test_a_model_that_is_not_here_is_not_guessed_about(self):
        from ainode.models.api_routes import apply_detected_backend

        app = self._app([])
        assert apply_detected_backend(app, "nothing/at-all", {}) == {}


class TestAnImageModelsSpeedIsActuallyMeasured:
    """record_image_speed existed and nothing ever called it, so
    seconds_per_image was a field that could not be filled."""

    def test_latency_is_the_speed_when_there_are_no_tokens(self, tmp_path):
        from ainode.measure.recorder import Recorder
        from ainode.measure.store import MeasurementStore

        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("org/img", ok=True, kind="image", memory_gb=20.0)

        class _Collector:
            def model_stats(self):
                return {"org/img": {"requests": 20, "avg_latency_ms": 41300.0}}

        class _Record:
            model = "org/img"
            kind = "image"

        class _Instance:
            record = _Record()
            backend = type("B", (), {"load_phase": "ready", "config": None})()

        class _Manager:
            def instances(self):
                return [_Instance()]

        app = {"measurement_store": store, "metrics_collector": _Collector(),
               "instances": _Manager()}
        Recorder(app)._record_speeds()
        assert store.get("org/img").seconds_per_image == 41.3
        # And it is not reported as a token rate, which it is not.
        assert store.get("org/img").tokens_per_second == 0.0

    def test_an_llm_still_gets_tokens_per_second(self, tmp_path):
        from ainode.measure.recorder import Recorder
        from ainode.measure.store import MeasurementStore

        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("org/llm", ok=True, kind="llm", memory_gb=40.0)

        class _Collector:
            def model_stats(self):
                return {"org/llm": {"requests": 40, "avg_tokens_per_second": 98.5,
                                    "avg_latency_ms": 2000.0}}

        class _Record:
            model = "org/llm"
            kind = "llm"

        class _Instance:
            record = _Record()
            backend = type("B", (), {"load_phase": "ready", "config": None})()

        class _Manager:
            def instances(self):
                return [_Instance()]

        app = {"measurement_store": store, "metrics_collector": _Collector(),
               "instances": _Manager()}
        Recorder(app)._record_speeds()
        entry = store.get("org/llm")
        assert entry.tokens_per_second == 98.5
        assert entry.seconds_per_image == 0.0


class TestAProfileDoesNotDemoteAPeersImageModel:
    """Capturing a profile forced every entry gathered from a peer to
    KIND_LLM, so an image model on node 3 — which is where it is meant to
    live — came back as an LLM and restoring it started vLLM on a diffusers
    pipeline."""

    def test_the_peers_own_kind_survives(self):
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply.capture_profile)
        assert 'spec["kind"] = KIND_LLM' not in source
        assert 'spec.get("kind")' in source

    def test_a_peer_too_old_to_say_is_an_llm(self):
        # That is all there was before, so the default is not a guess.
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply.capture_profile)
        assert "or KIND_LLM" in source

    def test_the_local_half_already_did_this(self):
        from ainode.profiles.store import KIND_IMAGE, ProfileEntry

        entry = ProfileEntry(model="org/img", kind=KIND_IMAGE,
                             engine_backend="diffusers")
        assert entry.launch_body()["engine_backend"] == "diffusers"


class TestTheOlderBackendGotTheSameFix:
    def test_each_nvidia_instance_writes_its_own_log(self):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.nvidia import NvidiaBackend

        primary = NvidiaBackend(NodeConfig(node_id="n", model="a/b"))
        stacked = NvidiaBackend(NodeConfig(node_id="n", model="c/d"),
                                instance_id="8001")
        assert primary.log_path.name == "nvidia-vllm.log"
        assert stacked.log_path.name == "nvidia-vllm-8001.log"
        assert primary.log_path != stacked.log_path


class TestTheLaunchFormDoesNotLeakImageSettings:
    def test_they_only_travel_for_an_image_model(self):
        # The fields are hidden for a text model but keep whatever was last
        # typed into them, and max_image_size on an LLM would be persisted
        # onto its config as a meaningless value.
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "if (this.toggleImageFields && this.toggleImageFields(model)) {" \
            in app_js

    def test_the_fields_appear_for_a_model_already_selected(self):
        # A poll redraws the list with a selection already made; without this
        # the fields stayed hidden until it was picked again.
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "self.toggleImageFields(select.value);" in app_js


class TestTheGateStillWorksEndToEnd:
    def test_an_oversized_image_model_is_refused_with_its_own_reasoning(self):
        from ainode.models.registry import ModelManager
        from ainode.safety.admission import check_admission
        import ainode.planner.api_routes as planner_routes

        directory = Path(tempfile.mkdtemp())
        repo = directory / "someone--img"
        repo.mkdir()
        (repo / "model_index.json").write_text("{}")
        (repo / "w.safetensors").write_bytes(b"x" * 2048)

        class _Node:
            node_id = "n3"
            node_name = "S3"
            status = "online"
            gpu_memory_gb = 122.0
            gpu_memory_total_mb = 124928.0
            gpu_memory_used_mb = 124928.0 - 8000.0

        class _Cluster:
            def members(self):
                return [_Node()]

        class _Config:
            node_id = "n3"
            api_port = 8000
            max_image_size = 4096

        app = {"config": _Config(), "cluster_state": _Cluster(),
               "model_manager": ModelManager(models_dir=str(directory))}
        original = planner_routes._image_weights_gb
        planner_routes._image_weights_gb = lambda manager, model: 33.0
        try:
            refusal = check_admission(app, "someone/img")
        finally:
            planner_routes._image_weights_gb = original
        assert "4096x4096 image" in refusal
        assert "END of a picture" in refusal


class TestTheEngineContractHolds:
    def test_the_image_backend_answers_everything_that_is_called_on_one(self):
        # Every method AINode invokes on a backend, gathered from the source
        # rather than from memory — the interface grew twice this session.
        import re

        from ainode.engine.backends.diffusers import DiffusersBackend

        calls = set()
        for path in (Path(__file__).resolve().parent.parent / "ainode").rglob("*.py"):
            for match in re.finditer(r"\b(?:backend|inst\.backend)\.([a-z_]+)\(",
                                     path.read_text()):
                calls.add(match.group(1))
        missing = [c for c in calls
                   if not hasattr(DiffusersBackend, c) and c != "unload"]
        assert not missing, missing

    def test_the_guard_can_kill_it(self):
        from ainode.engine.backends.diffusers import DiffusersBackend

        assert callable(getattr(DiffusersBackend, "kill", None))


class TestTheAdmissionOrderIsRight:
    def test_the_guard_answers_before_any_arithmetic(self):
        # A host already short of memory needs no calculation to refuse.
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission.check_admission)
        assert source.index("memory_guard") < source.index("_planner_says")

    def test_the_image_branch_runs_before_the_llm_facts_are_read(self):
        # A diffusers pipeline has no config.json, so the LLM reader finds
        # nothing and would return "" — passing an unplanned launch.
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission._planner_says)
        assert source.index("_image_says") < source.index("local_facts(manager")


class TestThePagesSayWhichKindAModelIs:
    """The search and the Models page showed a picture model and a chat model
    as the same kind of thing, and the first hint that they are not would
    have been a failed launch."""

    APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
              "static" / "js" / "app.js").read_text()

    def test_the_models_page_badges_it(self):
        assert self.APP_JS.count("Image</span>") >= 2   # catalog and search

    def test_the_catalog_mapping_keeps_the_field(self):
        # It comes from /api/models and was dropped in the remap, so the
        # badge would never have had anything to read.
        assert "modality: m.modality || 'text'," in self.APP_JS

    def test_the_search_results_carry_it_from_the_server(self):
        from ainode.models.registry import ModelManager

        assert ModelManager.modality_for("text-to-image") == "image"
        source = (Path(__file__).resolve().parent.parent / "ainode" /
                  "models" / "registry.py").read_text()
        assert '"modality": modality,' in source


def _json(resp):
    return json.loads(resp.body)
