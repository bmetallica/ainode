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

from unittest import mock

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

    @pytest.mark.asyncio
    async def test_a_peers_model_is_listed_too(self, client, app):
        """Once /v1/embeddings forwards, a peer's model is one this address
        can answer for — and that is the only promise the list makes."""
        from ainode.discovery.broadcast import NodeStatus
        from ainode.discovery.cluster import ClusterNode

        app["embedding_manager"] = _Manager([])
        app["cluster_state"].add_node(ClusterNode(
            node_id="n3", node_name="spark-3", gpu_name="GB10",
            gpu_memory_gb=128.0, unified_memory=True, model="",
            status=NodeStatus.ONLINE, api_port=8000, web_port=3000,
            last_seen=0.0, fabric_ip="10.0.0.3",
            embedding_models=[CURATED],
        ))
        data = await (await client.get("/v1/models")).json()
        assert CURATED in [m["id"] for m in data["data"]]


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


class TestPlacement:
    """Where an embedding model runs is a deployment decision, so it belongs
    in the dialog and in the profile — not in "whichever node's UI is open".
    """

    @pytest.mark.asyncio
    async def test_an_empty_node_id_loads_here(self, client, app):
        calls = []

        class _M(_Manager):
            def is_loaded(self, model_id):
                return False

            async def aload(self, model_id):
                calls.append(model_id)
                return {"id": model_id}

            def save_manifest(self):
                pass

        app["embedding_manager"] = _M([])
        response = await client.post("/api/cluster/embeddings/load",
                                     json={"model": CURATED})
        assert response.status == 200, await response.text()
        assert calls == [CURATED]

    @pytest.mark.asyncio
    async def test_a_missing_model_is_a_400(self, client, app):
        app["embedding_manager"] = _Manager([])
        response = await client.post("/api/cluster/embeddings/load",
                                     json={"node_id": "n3"})
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_an_unknown_node_is_a_404(self, client, app):
        app["embedding_manager"] = _Manager([])
        response = await client.post("/api/cluster/embeddings/load",
                                     json={"model": CURATED, "node_id": "nope"})
        assert response.status == 404
        assert "not found" in (await response.json())["error"]

    def test_the_routes_exist(self, app):
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/cluster/embeddings/load" in paths
        assert "/api/cluster/embeddings/unload" in paths


class TestTheProfileRemembersTheNode:
    def test_capture_records_this_nodes_placement(self):
        """An entry with no placement lands wherever the profile is applied —
        which is how a RAG model captured from node 3 came back on the head."""
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply.capture_profile)
        assert "node_ids=[own_id]" in source

    def test_capture_records_peers_too(self):
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply.capture_profile)
        assert "embedding_models" in source
        assert "node_ids=[node.node_id]" in source

    def test_apply_sends_an_entry_to_the_node_it_names(self):
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply._start_embedding_entry)
        assert "handle_cluster_embedding_load" in source
        # Placement is decided before the local manager is even consulted.
        assert source.index("_entry_target_node") < source.index("embedding_manager")


class TestTheWeightsLiveWhereTheMirrorLooks:
    """Reported: nomic could not be started on node 3.

    SentenceTransformer's default cache_folder is huggingface_hub's, which
    inside this container is /root/.cache/huggingface. Three consequences,
    all of them invisible until someone tried:

      * not mounted — the model was re-downloaded after every restart;
      * not under models_dir — list_downloaded() never saw it, so the mirror
        never carried it to the other nodes;
      * so a sub-node had to fetch it from Hugging Face itself, which is the
        one thing a head-only deployment forbids.
    """

    def test_the_cache_folder_is_models_dir(self, tmp_path):
        from ainode.embeddings.manager import EmbeddingManager

        assert EmbeddingManager(models_dir=str(tmp_path)).models_dir == str(tmp_path)

    def test_without_one_it_still_lands_somewhere_mounted(self, monkeypatch, tmp_path):
        """AINODE_HOME/models, not the HF default: a manager built without a
        config must not write to a directory that dies with the container."""
        import ainode.core.config as config_module
        from ainode.embeddings.manager import EmbeddingManager

        monkeypatch.setattr(config_module, "AINODE_HOME", tmp_path)
        assert EmbeddingManager().models_dir == str(tmp_path / "models")

    def test_the_load_passes_it_through(self):
        import inspect

        from ainode.embeddings.manager import EmbeddingManager

        source = inspect.getsource(EmbeddingManager.load)
        assert "cache_folder=self.models_dir" in source

    def test_the_server_gives_it_the_configured_directory(self):
        import inspect

        from ainode.api import server

        source = inspect.getsource(server.create_app)
        assert 'EmbeddingManager(\n        models_dir=getattr(config, "models_dir", "") or "")' \
            in source

    @pytest.mark.asyncio
    async def test_a_sub_node_refuses_rather_than_downloading(self, tmp_path):
        """The same rule as an LLM: from the head, or an error saying so."""
        from ainode.core.config import NodeConfig
        from ainode.embeddings.api_routes import _ensure_weights

        app = {"config": NodeConfig(node_id="n3", ssh_user="admin",
                                    models_dir=str(tmp_path),
                                    download_from_hub=False),
               "embedding_manager": _Manager([]),
               "cluster_state": None}
        app["embedding_manager"].models_dir = str(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="absent"):
            message = await _ensure_weights(app, CURATED)
        assert CURATED in message
        assert "mirror-models" in message

    @pytest.mark.asyncio
    async def test_a_peer_copy_is_enough(self, tmp_path):
        from ainode.core.config import NodeConfig
        from ainode.embeddings.api_routes import _ensure_weights

        app = {"config": NodeConfig(node_id="n3", ssh_user="admin",
                                    models_dir=str(tmp_path),
                                    download_from_hub=False),
               "embedding_manager": _Manager([]),
               "cluster_state": None}
        app["embedding_manager"].models_dir = str(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="fetched"):
            assert await _ensure_weights(app, CURATED) == ""

    @pytest.mark.asyncio
    async def test_the_head_may_still_download(self, tmp_path):
        from ainode.core.config import NodeConfig
        from ainode.embeddings.api_routes import _ensure_weights

        app = {"config": NodeConfig(node_id="head", ssh_user="admin",
                                    models_dir=str(tmp_path)),
               "embedding_manager": _Manager([]),
               "cluster_state": None}
        app["embedding_manager"].models_dir = str(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="absent"):
            assert await _ensure_weights(app, CURATED) == ""


class TestACompilerFailureExplainsItself:
    """A RAG tool reported:

        AI provider at http://.../v1 answered HTTP 503:
        Failed to find C compiler. Please specify via CC environment variable
        or set triton.knobs.build.impl.

    The model had loaded successfully minutes earlier — loading compiles
    nothing, the first embed does — so nothing before that point pointed at
    the image. The operator reasonably suspected the model, then vLLM's
    pooling mode, then the endpoint. It was none of those: this container
    gets --gpus all, nomic-bert's code reaches for Triton kernels, and Triton
    builds its launcher stubs at runtime with a C compiler that
    python:3.12-slim does not have.
    """

    def _raise(self, message: str):
        from ainode.embeddings.manager import _explain

        return _explain(RuntimeError(message))

    def test_the_compiler_error_gets_a_hint(self):
        raised = self._raise("Failed to find C compiler. Please specify via "
                             "CC environment variable.")
        assert "no C compiler" in str(raised)
        assert "update-cluster.sh" in str(raised)

    def test_the_triton_knobs_wording_matches_too(self):
        assert "packaging fault" in str(self._raise("set triton.knobs.build.impl"))

    def test_the_original_text_survives(self):
        """A hint never replaces the evidence — the same rule the launch-phase
        hints follow."""
        assert "Failed to find C compiler" in str(
            self._raise("Failed to find C compiler."))

    def test_an_unrelated_failure_is_passed_through_unchanged(self):
        original = RuntimeError("CUDA out of memory")
        from ainode.embeddings.manager import _explain

        assert _explain(original) is original

    def test_the_image_ships_a_compiler(self):
        from pathlib import Path

        dockerfile = (Path(__file__).resolve().parent.parent / "scripts" /
                      "Dockerfile.ainode").read_text()
        assert "gcc libc6-dev" in dockerfile
        # Only where torch is, since nothing else can invoke Triton.
        assert 'if [ -n "${AINODE_EXTRAS}" ]' in dockerfile

    def test_embed_routes_failures_through_the_explanation(self):
        import inspect

        from ainode.embeddings.manager import EmbeddingManager

        source = inspect.getsource(EmbeddingManager.embed)
        assert "raise _explain(exc) from exc" in source
