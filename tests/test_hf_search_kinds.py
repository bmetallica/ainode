"""The Hugging Face search: filtered by kind, honest about formats.

Three requirements behind this file. Any image model from the Hub must work,
not only the two in the catalog. Any quantisation must work, so long as this
deployment can actually load it. And the list must not offer what cannot run
here — which means knowing, per kind, what each engine can read.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


from ainode.models.registry import CURATED_CLUSTER_MODELS, ModelManager

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class TestTheKinds:
    def test_every_kind_this_deployment_serves_is_searchable(self):
        assert set(ModelManager.KIND_TAGS) == {"chat", "vision", "image",
                                               "embedding"}

    def test_each_kind_names_the_engine_it_needs(self):
        # The kind is not a label — it decides which engine runs the model,
        # and therefore which formats can work at all.
        assert ModelManager.KIND_BACKEND["image"] == "diffusers"
        assert ModelManager.KIND_BACKEND["embedding"] == "embeddings"
        assert ModelManager.KIND_BACKEND["chat"] == ""

    def test_the_tags_map_both_ways(self):
        for tag, kind in (("text-generation", "chat"),
                          ("image-text-to-text", "vision"),
                          ("text-to-image", "image"),
                          ("sentence-similarity", "embedding"),
                          ("feature-extraction", "embedding")):
            assert ModelManager.kind_for(tag) == kind

    def test_an_unknown_tag_is_treated_as_chat(self):
        assert ModelManager.kind_for("summarization") == "chat"
        assert ModelManager.kind_for("") == "chat"

    def test_a_vision_model_is_not_an_image_model(self):
        # It takes pictures; it does not make them. Different engine.
        assert ModelManager.modality_for("image-text-to-text") == "text"
        assert ModelManager.modality_for("text-to-image") == "image"


class TestWhatCanActuallyBeLoaded:
    """A search result that leads to a launch failure is worse than no search
    result, so the verdict is per kind — and the reason travels with it,
    because "GGUF" means nothing to someone who has not hit it before."""

    def test_gguf_is_refused_for_a_chat_model(self):
        ok, why = ModelManager.servable("chat", "org/model-GGUF")
        assert ok is False and "llama.cpp" in why

    def test_gguf_is_refused_for_an_image_model_for_its_own_reason(self):
        # Here the trap is different: the file holds the transformer alone,
        # and the text encoder is the larger half.
        ok, why = ModelManager.servable("image", "org/Qwen-Image-GGUF")
        assert ok is False
        assert "text encoder" in why

    def test_mlx_and_exllama_are_refused(self):
        assert ModelManager.servable("chat", "org/model-mlx")[0] is False
        assert ModelManager.servable("chat", "org/model-exl3")[0] is False

    def test_nunchaku_is_refused_with_the_real_obstacle(self):
        # Not "unsupported": no aarch64 build of its kernels.
        ok, why = ModelManager.servable("image", "org/Qwen-Image-W4A4-nunchaku")
        assert ok is False and "aarch64" in why

    def test_a_comfyui_single_file_is_refused(self):
        ok, why = ModelManager.servable(
            "image", "Comfy-Org/Qwen-Image", library="diffusion-single-file")
        assert ok is False and "pipeline directory" in why

    def test_every_quantisation_a_pipeline_can_carry_is_fine(self):
        # The requirement: arbitrary quantisation, as long as it loads.
        for repo in ("Rin247/Qwen-Image-2.1-FP8", "Rin247/Qwen-Image-2.1-INT4",
                     "someone/Qwen-Image-2.1-bf16"):
            assert ModelManager.servable("image", repo)[0] is True

    def test_the_usual_llm_quantisations_are_fine(self):
        for repo in ("org/m-AWQ", "org/m-GPTQ", "org/m-NVFP4", "org/m-FP8",
                     "org/m-W4A16", "org/m"):
            assert ModelManager.servable("chat", repo)[0] is True, repo

    def test_the_verdict_reads_tags_not_only_the_name(self):
        # A repo can be a single-file checkpoint without saying so in its
        # title.
        ok, _ = ModelManager.servable("image", "someone/pretty-pictures",
                                      tags=["diffusion-single-file"])
        assert ok is False

    def test_an_embedding_model_in_onnx_only_is_refused(self):
        ok, why = ModelManager.servable("embedding", "org/embed-onnx",
                                        tags=["onnx"])
        assert ok is False and "safetensors" in why


class _Manager(ModelManager):
    """Search without the network: the Hub call is the only thing faked."""

    fake = []

    def _fake_search(self, **kwargs):
        return list(self.fake)


class TestTheSearchRoute:
    def _request(self, app, **query):
        class _Req:
            def __init__(self):
                self.app = app
                self.query = {k: str(v) for k, v in query.items()}

        return _Req()

    def test_an_unknown_kind_is_refused_by_name(self):
        from ainode.models.api_routes import handle_search_models

        app = {"model_manager": ModelManager(models_dir="/tmp")}
        resp = asyncio.run(handle_search_models(
            self._request(app, q="qwen", kind="audio")))
        assert resp.status == 400
        body = json.loads(resp.body)
        assert "audio" in body["error"]
        assert "image" in body["known"]

    def test_the_kinds_asked_for_come_back_with_the_answer(self, monkeypatch):
        from ainode.models.api_routes import handle_search_models

        manager = ModelManager(models_dir="/tmp")
        monkeypatch.setattr(manager, "search_huggingface",
                            lambda q, limit, kinds=None: [])
        app = {"model_manager": manager}
        body = json.loads(asyncio.run(handle_search_models(
            self._request(app, q="qwen", kind="image,embedding"))).body)
        assert body["kinds"] == ["image", "embedding"]

    def test_no_kind_means_everything_servable(self, monkeypatch):
        from ainode.models.api_routes import handle_search_models

        manager = ModelManager(models_dir="/tmp")
        monkeypatch.setattr(manager, "search_huggingface",
                            lambda q, limit, kinds=None: [])
        body = json.loads(asyncio.run(handle_search_models(
            self._request({"model_manager": manager}, q="qwen"))).body)
        assert set(body["kinds"]) == set(ModelManager.KIND_TAGS)

    def test_the_kinds_narrow_which_tags_are_queried(self):
        # Not a filter applied afterwards: a filter would only narrow the
        # results of the wrong search.
        asked = []

        class _Api:
            def list_models(self, **kwargs):
                asked.append(kwargs.get("pipeline_tag"))
                return []

        ModelManager._search_every_servable_tag(
            _Api(), query="x", limit=5, kinds=["image"])
        assert asked == ["text-to-image"]

    def test_an_unknown_kind_cannot_widen_the_search(self):
        asked = []

        class _Api:
            def list_models(self, **kwargs):
                asked.append(kwargs.get("pipeline_tag"))
                return []

        ModelManager._search_every_servable_tag(
            _Api(), query="x", limit=5, kinds=["nonsense"])
        assert asked == []


class TestAnyImageModelNotJustTheCuratedOnes:
    def test_the_catalog_entries_are_suggestions_not_the_whole_list(self):
        # Two curated entries exist for convenience. Nothing in the load path
        # consults that list to decide whether a model may run.
        curated = [k for k, v in CURATED_CLUSTER_MODELS.items()
                   if getattr(v, "modality", "") == "image"]
        assert len(curated) == 2

    def test_a_repo_nobody_curated_is_detected_from_its_own_files(self):
        import tempfile

        from ainode.models.api_routes import apply_detected_backend

        directory = Path(tempfile.mkdtemp())
        repo = directory / "someone--anything-at-all"
        repo.mkdir()
        (repo / "model_index.json").write_text("{}")
        app = {"model_manager": ModelManager(models_dir=str(directory))}
        overrides = apply_detected_backend(app, "someone/anything-at-all", {})
        assert overrides["engine_backend"] == "diffusers"

    def test_the_search_marks_it_as_an_image_model_without_a_catalog_entry(self):
        # Straight from the Hub's own pipeline tag.
        assert ModelManager.kind_for("text-to-image") == "image"


class TestTheUI:
    def test_the_filter_is_offered(self):
        assert 'id="hf-kind-filter"' in APP_JS
        for kind in ("chat", "vision", "image", "embedding"):
            assert f"id: '{kind}'" in APP_JS, kind

    def test_it_re_runs_the_search_rather_than_filtering_locally(self):
        block = APP_JS.split("toggleHfKind(kind) {")[1].split("\n  },")[0]
        assert "self.searchHuggingFace" in block or \
            "this.searchHuggingFace" in block

    def test_the_kind_reaches_the_query(self):
        assert "'&kind=' + encodeURIComponent(kinds.join(','))" in APP_JS

    def test_the_card_says_why_something_cannot_run(self):
        # "Dimmed" tells someone that something is wrong and nothing about
        # what, and the answer is usually one sentence long.
        assert "hf-why-not" in APP_JS
        assert "m.not_servable_reason" in APP_JS

    def test_the_verdict_comes_from_the_server_with_a_fallback(self):
        block = APP_JS.split("hfRunnable(m) {")[1].split("\n  },")[0]
        assert "m.servable !== undefined" in block
        assert "m.vllm_ok !== false" in block      # older node

    def test_the_count_says_what_was_filtered(self):
        assert "this cluster cannot load" in APP_JS
