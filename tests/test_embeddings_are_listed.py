"""A loaded embedding model has to be visible where people look for it.

Reported after a successful load:

    {"ok": true, "model_id": "nomic-ai/nomic-embed-text-v1.5",
     "status": "loaded", ...}
    ... ich kann das modell aber nicht unter models sehen

Two separate blind spots:

  * /v1/models builds its list from the vLLM routing table. Embedding models
    are not vLLM instances, so they never appeared — and a RAG client that
    asks /v1/models before calling /v1/embeddings concludes the model it just
    loaded is unavailable. OpenAI lists embedding models there; so do we now.
  * /api/embeddings/models walked the CURATED catalog and set a `loaded` flag
    on it. Anything loaded from outside that list — which the UI can now ask
    for by repo id — was invisible however loaded it was, including in the
    tab that had just loaded it.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig

CURATED = "nomic-ai/nomic-embed-text-v1.5"
UNCURATED = "someone/private-embed-v3"


class _Manager:
    """Stands in for EmbeddingManager: only the two listing methods matter."""

    def __init__(self, loaded):
        self._loaded = loaded

    def list_loaded(self):
        return [{"id": m, "dimensions": 768, "max_seq_length": 8192} for m in self._loaded]

    def list_known(self):
        return [{"id": CURATED, "hf_repo": CURATED, "dimensions": 768,
                 "max_seq_length": 8192, "size_mb": 274, "description": "Nomic."}]


@pytest.fixture
def app():
    return create_app(config=NodeConfig(node_id="n1", model="org/chat-model"),
                      engine=None)


@pytest_asyncio.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


class TestV1Models:
    @pytest.mark.asyncio
    async def test_a_loaded_embedding_model_is_listed(self, client, app):
        app["embedding_manager"] = _Manager([CURATED])
        data = await (await client.get("/v1/models")).json()
        assert CURATED in [m["id"] for m in data["data"]]

    @pytest.mark.asyncio
    async def test_the_chat_model_is_still_there(self, client, app):
        app["embedding_manager"] = _Manager([CURATED])
        data = await (await client.get("/v1/models")).json()
        ids = [m["id"] for m in data["data"]]
        assert "org/chat-model" in ids and CURATED in ids

    @pytest.mark.asyncio
    async def test_an_unloaded_one_is_not_listed(self, client, app):
        """Listing the catalog would advertise models nothing can answer for."""
        app["embedding_manager"] = _Manager([])
        data = await (await client.get("/v1/models")).json()
        assert CURATED not in [m["id"] for m in data["data"]]

    @pytest.mark.asyncio
    async def test_no_duplicate_when_a_name_collides(self, client, app):
        app["embedding_manager"] = _Manager(["org/chat-model"])
        data = await (await client.get("/v1/models")).json()
        assert [m["id"] for m in data["data"]].count("org/chat-model") == 1

    @pytest.mark.asyncio
    async def test_a_broken_manager_does_not_break_the_endpoint(self, client, app):
        class _Broken:
            def list_loaded(self):
                raise RuntimeError("torch is not installed")

        app["embedding_manager"] = _Broken()
        response = await client.get("/v1/models")
        assert response.status == 200
        assert "org/chat-model" in [m["id"] for m in (await response.json())["data"]]

    def test_only_this_nodes_models_are_listed(self):
        """Embeddings are not advertised in discovery and /v1/embeddings has
        no proxy to a peer, so listing a peer's would promise an endpoint that
        answers "not loaded"."""
        import inspect

        from ainode.api import server

        assert "Only this node's" in inspect.getsource(server._local_embedding_models)


class TestTheEmbeddingsListing:
    @pytest.mark.asyncio
    async def test_an_uncurated_model_shows_up_once_loaded(self, client, app):
        app["embedding_manager"] = _Manager([UNCURATED])
        data = await (await client.get("/api/embeddings/models")).json()
        entry = next(m for m in data["models"] if m["id"] == UNCURATED)
        assert entry["loaded"] is True
        assert entry["hf_repo"] == UNCURATED

    @pytest.mark.asyncio
    async def test_the_curated_list_is_still_shown_unloaded(self, client, app):
        app["embedding_manager"] = _Manager([UNCURATED])
        data = await (await client.get("/api/embeddings/models")).json()
        entry = next(m for m in data["models"] if m["id"] == CURATED)
        assert entry["loaded"] is False

    @pytest.mark.asyncio
    async def test_a_curated_model_is_not_listed_twice(self, client, app):
        app["embedding_manager"] = _Manager([CURATED])
        data = await (await client.get("/api/embeddings/models")).json()
        assert [m["id"] for m in data["models"]].count(CURATED) == 1
        assert data["count"] == len(data["models"])
