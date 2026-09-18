"""A multimodal model is a text generator that also takes pictures.

Reported: sparkarena/Minimax-M3-v0-NVFP4-REAP50 could not be found in the
model search. It exists, it fits two nodes, and vLLM can serve it — but the
Hub files it under

    pipeline_tag = image-text-to-text

while the search asked only for text-generation. Every multimodal model was
invisible, and an operator looking for one found nothing and reasonably
concluded it did not exist. MiniMax M3 and all of its quantisations are
tagged that way; Gemma 4 appears only because it is in the curated catalog.

This is the second time that one filter has hidden something the cluster can
run — embeddings were the first, and are excluded by construction for a
different reason.
"""

from __future__ import annotations


from ainode.models.registry import ModelManager


class _Model:
    def __init__(self, repo, downloads=0):
        self.id = repo
        self.downloads = downloads
        self.safetensors = None


class TestBothTagsAreSearched:
    def test_image_text_to_text_is_servable(self):
        assert "image-text-to-text" in ModelManager.SERVABLE_PIPELINE_TAGS
        assert "text-generation" in ModelManager.SERVABLE_PIPELINE_TAGS

    def test_every_tag_is_queried(self):
        calls = []

        class _Api:
            def list_models(self, **kw):
                calls.append(kw["pipeline_tag"])
                return []

        ModelManager._search_every_servable_tag(_Api(), query="x", limit=5)
        assert calls == list(ModelManager.SERVABLE_PIPELINE_TAGS)

    def test_the_results_are_merged(self):
        class _Api:
            def list_models(self, **kw):
                if kw["pipeline_tag"] == "text-generation":
                    return [_Model("org/text", 10)]
                return [_Model("org/vision", 20)]

        found = ModelManager._search_every_servable_tag(_Api(), query="x", limit=10)
        assert {m.id for m in found} == {"org/text", "org/vision"}

    def test_a_repo_in_both_appears_once(self):
        class _Api:
            def list_models(self, **kw):
                return [_Model("org/both", 5)]

        found = ModelManager._search_every_servable_tag(_Api(), query="x", limit=10)
        assert [m.id for m in found] == ["org/both"]

    def test_the_merged_list_is_download_sorted(self):
        """Two queries each sorted is not one sorted list."""
        class _Api:
            def list_models(self, **kw):
                if kw["pipeline_tag"] == "text-generation":
                    return [_Model("org/small", 1)]
                return [_Model("org/popular", 900)]

        found = ModelManager._search_every_servable_tag(_Api(), query="x", limit=10)
        assert [m.id for m in found] == ["org/popular", "org/small"]

    def test_the_limit_still_holds_across_both(self):
        class _Api:
            def list_models(self, **kw):
                return [_Model(f"{kw['pipeline_tag']}/{i}", i) for i in range(10)]

        found = ModelManager._search_every_servable_tag(_Api(), query="x", limit=4)
        assert len(found) == 4


class TestOneBadTagDoesNotCostTheRest:
    def test_a_failing_query_is_skipped(self):
        """The Hub has renamed pipeline tags before; one unknown name should
        not take the other tag's results with it."""
        class _Api:
            def list_models(self, **kw):
                if kw["pipeline_tag"] == "image-text-to-text":
                    raise ValueError("no such pipeline tag")
                return [_Model("org/text", 3)]

        found = ModelManager._search_every_servable_tag(_Api(), query="x", limit=5)
        assert [m.id for m in found] == ["org/text"]

    def test_all_failing_is_an_empty_list_not_an_exception(self):
        class _Api:
            def list_models(self, **kw):
                raise OSError("no route to host")

        assert ModelManager._search_every_servable_tag(_Api(), query="x", limit=5) == []


class TestTheSearchUsesIt:
    def test_search_huggingface_goes_through_the_merge(self):
        import inspect

        source = inspect.getsource(ModelManager.search_huggingface)
        assert "_search_every_servable_tag(" in source
        assert 'pipeline_tag="text-generation"' not in source

    def test_the_kwargs_still_reach_the_api(self):
        seen = {}

        class _Api:
            def list_models(self, **kw):
                seen.update(kw)
                return []

        ModelManager._search_every_servable_tag(
            _Api(), query="q", limit=7, expand=["safetensors"], sort="downloads")
        assert seen["expand"] == ["safetensors"]
        assert seen["sort"] == "downloads"
        assert seen["search"] == "q"
