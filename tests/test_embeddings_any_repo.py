"""Any sentence-transformers repo, not only the four in the catalog.

Asked: "warum finde ich nomic-ai/nomic-embed-text-v1 in AINode nicht?"

Two reasons, and neither was visible from the UI:

  * the model search filters on pipeline_tag="text-generation", so an
    embedding model (sentence-similarity / feature-extraction) cannot appear
    there by construction;
  * the embeddings tab rendered a fixed catalog — MiniLM, nomic v1.5, BGE —
    with no way to name anything else.

The backend was never the limit: the load route matches {model_id:.+} and
hands the string straight to SentenceTransformer, so an arbitrary repo has
always worked over curl. Only the UI said otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path

from ainode.embeddings.manager import KNOWN_EMBEDDING_MODELS

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static" /
          "js" / "app.js").read_text()


class TestTheFieldExists:
    def test_the_tab_offers_a_free_text_repo(self):
        assert 'id="embed-any-repo"' in APP_JS
        assert 'id="embed-any-load"' in APP_JS

    def test_it_posts_to_the_cluster_load_route(self):
        # Through the cluster route even for this node — an empty node_id
        # means "here". Two paths is how the placement got lost before.
        assert "'/api/cluster/embeddings/load'" in APP_JS
        assert "JSON.stringify({ model: id, node_id: nodeId })" in APP_JS

    def test_enter_submits(self):
        assert "if (e.key === 'Enter') loadTyped();" in APP_JS

    def test_a_bare_name_is_refused_before_the_request(self):
        # "nomic-embed-text-v1" without the owner reaches the Hub as a repo id
        # and fails minutes later, after a download attempt.
        assert "Use the full repo id, owner/name" in APP_JS

    def test_an_empty_catalog_still_shows_the_field(self):
        """The early return on an empty list left an operator with an empty
        tab and no next step."""
        assert "No embedding models in the catalog.</div>';\n        return;" not in APP_JS
        assert "name any repo above" in APP_JS

    def test_the_catalog_buttons_share_the_same_loader(self):
        # One code path, so a fix to either reaches both.
        assert APP_JS.count("var loadEmbedding = async function") == 1
        assert "loadEmbedding(btn.dataset.model, btn, 'Load');" in APP_JS


class TestTheBackendAlreadyAllowedIt:
    def test_the_route_takes_any_repo_id(self):
        source = (Path(__file__).resolve().parent.parent / "ainode" /
                  "embeddings" / "api_routes.py").read_text()
        assert "{model_id:.+}/load" in source

    def test_the_manager_does_not_gate_on_the_catalog(self):
        source = (Path(__file__).resolve().parent.parent / "ainode" /
                  "embeddings" / "manager.py").read_text()
        load = source[source.index("    def load(self, model_id"):]
        load = load[:load.index("\n    def ", 1)]
        assert "SentenceTransformer(model_id, cache_folder=self.models_dir)" in load
        # The catalog is consulted for metadata only, never as a whitelist.
        assert "KNOWN_EMBEDDING_MODELS.get(model_id, {})" in load


class TestTheSearchCannotFindThem:
    def test_the_model_search_is_text_generation_only(self):
        """Recorded so the next person does not go looking for a bug in the
        search: an embedding model is excluded by the query itself."""
        source = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
                  "registry.py").read_text()
        search = source[source.index("    def search_huggingface"):]
        assert re.search(r'pipeline_tag="text-generation"', search[:2000])

    def test_the_curated_list_is_still_there(self):
        assert "nomic-ai/nomic-embed-text-v1.5" in KNOWN_EMBEDDING_MODELS
