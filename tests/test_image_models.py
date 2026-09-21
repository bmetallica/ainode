"""Image models as first-class citizens: catalog, planner, gate, weights.

The rule this file defends is that image generation is not a special case
bolted on beside AINode but another kind of model inside it — found by the
same search, planned before it starts, refused by the same gate, and with its
weights distributed by the same rsync from the head that every other model
uses.
"""

from __future__ import annotations

import json


from ainode.models.registry import CURATED_CLUSTER_MODELS, ModelManager
from ainode.planner.compute import NodeBudget, plan_for_image


class TestTheSearchShowsThemAndSaysWhatTheyAre:
    def test_text_to_image_is_servable_now(self):
        # It was excluded while nothing could serve it, which was right. Now
        # that something can, hiding it would be the wrong half-truth.
        assert "text-to-image" in ModelManager.SERVABLE_PIPELINE_TAGS

    def test_the_modality_is_derived_from_the_tag(self):
        assert ModelManager.modality_for("text-to-image") == "image"
        assert ModelManager.modality_for("text-generation") == "text"
        assert ModelManager.modality_for("image-text-to-text") == "text"

    def test_a_vision_language_model_is_still_text(self):
        # It takes pictures; it does not make them. vLLM serves it.
        assert ModelManager.modality_for("image-text-to-text") == "text"

    def test_an_unknown_tag_defaults_to_text(self):
        assert ModelManager.modality_for("") == "text"
        assert ModelManager.modality_for("feature-extraction") == "text"

    def test_the_modality_picks_the_engine(self):
        assert ModelManager.backend_for_modality("image") == "diffusers"
        assert ModelManager.backend_for_modality("text") == ""


class TestTheCatalogEntry:
    def test_fp8_is_the_one_offered_first(self):
        entry = CURATED_CLUSTER_MODELS["qwen-image-2.1-fp8"]
        assert entry.modality == "image"
        assert entry.engine_backend == "diffusers"
        assert entry.size_gb == 18.0

    def test_the_full_precision_one_is_there_too(self):
        entry = CURATED_CLUSTER_MODELS["qwen-image-2.1"]
        assert entry.size_gb == 33.0
        assert entry.modality == "image"

    def test_neither_claims_to_be_verified(self):
        # Nothing here has run on the fleet yet, and the badge means it has.
        for key in ("qwen-image-2.1", "qwen-image-2.1-fp8"):
            assert CURATED_CLUSTER_MODELS[key].verified is False

    def test_the_description_warns_off_cpu_offload(self):
        # The standard advice everywhere else, and meaningless here.
        assert "offload" in CURATED_CLUSTER_MODELS["qwen-image-2.1-fp8"].description

    def test_the_recipe_carries_the_engine(self):
        from ainode.models.api_routes import catalog_recipe

        recipe = catalog_recipe("Rin247/Qwen-Image-2.1-FP8")
        assert recipe["engine_backend"] == "diffusers"

    def test_a_bare_load_gets_the_right_engine(self):
        # Clicking it in the dashboard sends {"model": ...} and nothing else.
        from ainode.models.api_routes import apply_catalog_recipe

        overrides, _ = apply_catalog_recipe("Rin247/Qwen-Image-2.1-FP8", {}, None)
        assert overrides["engine_backend"] == "diffusers"

    def test_an_explicit_choice_still_wins(self):
        from ainode.models.api_routes import apply_catalog_recipe

        overrides, _ = apply_catalog_recipe(
            "Rin247/Qwen-Image-2.1-FP8", {"engine_backend": "eugr"}, None)
        assert overrides["engine_backend"] == "eugr"


class TestTheLoadBodyAcceptsTheImageKnobs:
    def _parse(self, body):
        from ainode.models.api_routes import parse_load_overrides

        return parse_load_overrides(body)

    def test_the_resolution_limit_travels(self):
        overrides, error = self._parse({"max_image_size": 1024})
        assert error is None and overrides["max_image_size"] == 1024

    def test_an_absurd_limit_is_refused(self):
        _, error = self._parse({"max_image_size": 99999})
        assert error is not None and error.status == 400

    def test_an_unknown_engine_is_refused_by_name(self):
        _, error = self._parse({"engine_backend": "comfyui"})
        assert error is not None
        assert "diffusers" in json.loads(error.body)["error"]

    def test_the_image_fields_reset_between_loads(self):
        # inst_config is built from the shared, mutable app config. Without
        # these in the reset set, loading a text model after an image one
        # would inherit engine_backend="diffusers" and start the wrong engine.
        from ainode.models.api_routes import _OVERRIDE_KEYS

        for field in ("engine_backend", "max_image_size", "image_steps",
                      "image_size", "image_dtype"):
            assert field in _OVERRIDE_KEYS, field


def _nodes(free=60.0):
    return [NodeBudget(node_id="n3", name="SPARK3", total_gb=122.0, free_gb=free)]


class TestThePlannerUsesADifferentCalculation:
    """No KV cache, no context length, no parallel axis. What replaces the
    cache is the peak of a single run, and the knob is the resolution."""

    def test_a_model_that_fits_fits(self):
        plan = plan_for_image(18.0, _nodes(free=60.0), max_image_size=1024)
        assert plan.fits is True
        assert plan.node_ids == ["n3"]
        assert plan.strategy == "solo"

    def test_it_prints_no_kv_figures(self):
        # A planner that filled them with zeros would be lying quietly.
        plan = plan_for_image(18.0, _nodes()).to_dict()
        assert plan["kv_tokens"] == 0
        assert plan["max_model_len"] == 0
        assert plan["kv_bytes_per_token"] == 0

    def test_the_resolution_drives_the_headroom(self):
        small = plan_for_image(18.0, _nodes(free=30.0), max_image_size=512)
        large = plan_for_image(18.0, _nodes(free=30.0), max_image_size=4096)
        assert small.fits is True
        assert large.fits is False

    def test_the_refusal_names_the_way_out(self):
        plan = plan_for_image(33.0, _nodes(free=20.0))
        assert plan.fits is False
        assert "max_image_size" in plan.blocker

    def test_it_says_the_peak_comes_at_the_end(self):
        # The reason a node can survive the load and die on the first image.
        notes = " ".join(plan_for_image(18.0, _nodes()).notes)
        assert "END of a run" in notes

    def test_it_says_one_at_a_time(self):
        plan = plan_for_image(18.0, _nodes())
        assert plan.concurrent_requests == 1
        assert any("one image at a time" in w.lower() for w in plan.warnings)

    def test_a_model_not_on_disk_is_refused_not_estimated(self):
        plan = plan_for_image(0.0, _nodes(), model="org/img")
        assert plan.fits is False
        assert "directory of components" in plan.blocker

    def test_it_never_splits_across_nodes(self):
        # Two nodes do not make a diffusion pipeline fit; this engine runs one
        # process on one machine.
        nodes = [NodeBudget("a", "A", 122.0, 20.0), NodeBudget("b", "B", 122.0, 20.0)]
        assert plan_for_image(33.0, nodes).fits is False


class _Manager:
    def __init__(self, directory=None):
        self._dir = directory

    def model_dirs_for_repo(self, repo):
        return [self._dir] if self._dir else []

    def _catalog_lookup(self, model_id):
        return CURATED_CLUSTER_MODELS.get("qwen-image-2.1-fp8") \
            if "Qwen-Image" in model_id else None


class TestTheGateKnowsTheDifference:
    def test_an_image_model_is_recognised_from_the_catalog(self):
        from ainode.planner.api_routes import _is_image, _recipe

        app = {"model_manager": _Manager()}
        recipe = _recipe(app, "Rin247/Qwen-Image-2.1-FP8")
        assert _is_image(recipe, None, app, "Rin247/Qwen-Image-2.1-FP8") is True

    def test_and_from_the_disk_when_it_is_in_no_catalog(self, tmp_path):
        # An operator's own download is in no catalog, and model_index.json
        # where a config.json would be is what makes it a pipeline.
        from ainode.planner.api_routes import _is_image

        (tmp_path / "model_index.json").write_text("{}")
        app = {"model_manager": _Manager(tmp_path)}
        assert _is_image(None, None, app, "someone/their-own-model") is True

    def test_an_llm_is_not_mistaken_for_one(self, tmp_path):
        from ainode.planner.api_routes import _is_image

        (tmp_path / "config.json").write_text("{}")
        app = {"model_manager": _Manager(tmp_path)}
        assert _is_image(None, None, app, "org/llm") is False

    def test_the_gate_refuses_with_the_image_reasoning(self, tmp_path, monkeypatch):
        from ainode.safety import admission

        (tmp_path / "model_index.json").write_text("{}")
        (tmp_path / "big.safetensors").write_bytes(b"x" * 1024)

        class _Config:
            node_id = "n3"
            api_port = 8000
            max_image_size = 4096

        class _Node:
            node_id = "n3"
            node_name = "SPARK3"
            status = "online"
            gpu_memory_gb = 122.0
            gpu_memory_total_mb = 124928.0
            gpu_memory_used_mb = 124928.0 - 10000.0

        class _Cluster:
            def members(self):
                return [_Node()]

        app = {"config": _Config(), "cluster_state": _Cluster(),
               "model_manager": _Manager(tmp_path)}
        monkeypatch.setattr(
            "ainode.planner.api_routes._image_weights_gb",
            lambda manager, model: 33.0)
        refusal = admission.check_admission(app, "org/img")
        assert refusal
        assert "END of a picture" in refusal
        assert "force" in refusal


class TestTheWeightsTravelLikeEveryOtherModel:
    """rsync from the head after the initial download — the same path the LLMs
    use. It is directory-based, so a pipeline of five components travels no
    differently from one safetensors file."""

    def test_a_pipeline_directory_reads_as_present(self, tmp_path):
        from ainode.engine.acquire import model_is_local

        repo = tmp_path / "Rin247--Qwen-Image-2.1-FP8"
        (repo / "transformer").mkdir(parents=True)
        (repo / "model_index.json").write_text("{}")
        assert model_is_local("Rin247/Qwen-Image-2.1-FP8", str(tmp_path)) is True

    def test_an_aborted_download_does_not(self, tmp_path):
        from ainode.engine.acquire import model_is_local

        (tmp_path / "Rin247--Qwen-Image-2.1-FP8").mkdir()
        assert model_is_local("Rin247/Qwen-Image-2.1-FP8", str(tmp_path)) is False

    def test_the_solo_load_fetches_from_a_peer_before_starting(self):
        # The same call for every engine: an image model on a sub-node comes
        # from the head, not from Hugging Face.
        import inspect

        from ainode.models.api_routes import append_solo_instance

        source = inspect.getsource(append_solo_instance)
        assert "_fetch_weights_from_a_peer" in source
        assert source.index("_fetch_weights_from_a_peer") < source.index("backend.start()")

    def test_the_mirror_pushes_whatever_directory_it_is_given(self):
        import inspect

        from ainode.engine.mirror import mirror_model_to_peers

        source = inspect.getsource(mirror_model_to_peers)
        # No file-format assumptions anywhere in it.
        for token in ("safetensors", "config.json", ".bin"):
            assert token not in source, token


class TestTheInstanceSaysWhatItIs:
    def test_the_record_carries_the_kind(self):
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord(instance_id="i", model="m", head_node_id="n",
                                kind="image")
        assert InstanceRecord.from_dict(record.to_dict()).kind == "image"

    def test_an_older_peer_sends_none_and_that_reads_as_llm(self):
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord.from_dict(
            {"instance_id": "i", "model": "m", "head_node_id": "n"})
        assert record.kind == ""

    def test_a_profile_entry_restores_the_right_engine(self):
        from ainode.profiles.store import KIND_IMAGE, ProfileEntry

        entry = ProfileEntry(model="org/img", kind=KIND_IMAGE,
                             engine_backend="diffusers", max_image_size=1024)
        body = entry.launch_body()
        assert body["engine_backend"] == "diffusers"
        assert body["max_image_size"] == 1024

    def test_a_text_entry_carries_neither(self):
        from ainode.profiles.store import ProfileEntry

        body = ProfileEntry(model="org/llm").launch_body()
        assert "engine_backend" not in body
        assert "max_image_size" not in body

    def test_the_image_topic_is_routed_like_chat(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/v1/images/generations" in paths
