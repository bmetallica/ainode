"""Embedding model manager.

Wraps ``sentence-transformers`` so embedding models can be lazy-loaded,
cached, and served from the AINode API. Designed to be safe to import
without the ``sentence-transformers`` dependency present — a friendly
RuntimeError is raised only when an actual load is attempted.
"""

from __future__ import annotations

import asyncio
import logging
import json
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog of known embedding models
# ---------------------------------------------------------------------------

KNOWN_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "max_seq_length": 256,
        "size_mb": 91,
        "description": (
            "Compact, fast general-purpose embedding model. Great starter — "
            "small enough to run on CPU, high quality for its size."
        ),
        "default": True,
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "id": "nomic-ai/nomic-embed-text-v1.5",
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "max_seq_length": 8192,
        "size_mb": 274,
        "description": (
            "Long-context (8K) embedding model from Nomic. Excellent for "
            "RAG over long documents."
        ),
    },
    "BAAI/bge-large-en-v1.5": {
        "id": "BAAI/bge-large-en-v1.5",
        "hf_repo": "BAAI/bge-large-en-v1.5",
        "dimensions": 1024,
        "max_seq_length": 512,
        "size_mb": 1340,
        "description": (
            "High-quality English embedding model from BAAI. Top performer "
            "on the MTEB retrieval benchmark."
        ),
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "id": "mixedbread-ai/mxbai-embed-large-v1",
        "hf_repo": "mixedbread-ai/mxbai-embed-large-v1",
        "dimensions": 1024,
        "max_seq_length": 512,
        "size_mb": 670,
        "description": (
            "SOTA English embedding model from mixedbread. Balanced size "
            "and accuracy, strong on semantic search."
        ),
    },
}


_INSTALL_HINT = (
    "sentence-transformers is not installed. Install it with: "
    "pip install 'ainode[embeddings]'   or   pip install sentence-transformers"
)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EmbeddingManager:
    """Tracks loaded embedding models and serves embedding requests."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._models_dir = str(models_dir or "")

    @property
    def models_dir(self) -> str:
        """Where weights go — the same directory LLMs use.

        SentenceTransformer's default is huggingface_hub's cache, which inside
        this container is /root/.cache/huggingface: not mounted, so the model
        was re-downloaded after every restart; not under models_dir, so
        list_downloaded() never saw it and the mirror never carried it to the
        other nodes. A sub-node then had to fetch it from Hugging Face itself
        — the one thing a head-only deployment forbids.

        Falling back to AINODE_HOME/models rather than to the HF default, so
        a manager built without a config still writes somewhere that is
        mounted, backed up and mirrored.
        """
        if self._models_dir:
            return self._models_dir
        from ainode.core.config import AINODE_HOME

        return str(AINODE_HOME / "models")

    # -- persistence ----------------------------------------------------------
    #
    # Loaded models live in this process, so a restart loses them. An embedding
    # model backing a RAG pipeline is expected to be there the way a served LLM
    # is — and LLM instances are already replayed on boot. Without this, every
    # `systemctl restart ainode` silently breaks retrieval until somebody
    # notices and clicks Load again.

    @staticmethod
    def _manifest_path():
        from ainode.core.config import AINODE_HOME

        return AINODE_HOME / "embeddings.json"

    def save_manifest(self) -> None:
        """Record which models are loaded, for replay on the next boot."""
        try:
            path = self._manifest_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                loaded = sorted(self._models.keys())
            path.write_text(json.dumps({"models": loaded}))
        except Exception:
            logger.exception("Could not write the embedding manifest")

    def load_manifest(self) -> List[str]:
        """Model ids recorded by the last :meth:`save_manifest`."""
        try:
            path = self._manifest_path()
            if not path.exists():
                return []
            data = json.loads(path.read_text())
            return [str(m) for m in (data.get("models") or []) if str(m).strip()]
        except Exception:
            logger.exception("Could not read the embedding manifest")
            return []

    def replay(self) -> None:
        """Re-load every model from the manifest. Best effort, one at a time.

        A failure is logged and skipped rather than aborting the rest: one
        model that no longer resolves must not keep the others unloaded.
        """
        for model_id in self.load_manifest():
            if self.is_loaded(model_id):
                continue
            try:
                self.load(model_id)
                logger.info("Replayed embedding model %s", model_id)
            except Exception as exc:
                logger.warning("Could not replay embedding model %s: %s", model_id, exc)

    # -- catalog --------------------------------------------------------------

    def list_known(self) -> List[Dict[str, Any]]:
        """Return the static catalog of known embedding models."""
        return [dict(v) for v in KNOWN_EMBEDDING_MODELS.values()]

    def list_loaded(self) -> List[Dict[str, Any]]:
        """Return metadata for every model currently in-memory."""
        with self._lock:
            return [dict(meta) for meta in self._metadata.values()]

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._models

    def dimensions_of(self, model_id: str) -> Optional[int]:
        with self._lock:
            meta = self._metadata.get(model_id)
        if meta and meta.get("dimensions") is not None:
            return int(meta["dimensions"])
        catalog = KNOWN_EMBEDDING_MODELS.get(model_id)
        if catalog:
            return int(catalog.get("dimensions", 0)) or None
        return None

    # -- load / unload --------------------------------------------------------

    def _resolve_SentenceTransformer(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(_INSTALL_HINT) from exc
        return SentenceTransformer

    def load(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Load an embedding model into memory (blocking)."""
        with self._lock:
            if not force and model_id in self._models:
                return dict(self._metadata[model_id])

        SentenceTransformer = self._resolve_SentenceTransformer()

        logger.info("Loading embedding model %s", model_id)
        model = SentenceTransformer(model_id, cache_folder=self.models_dir)

        catalog = KNOWN_EMBEDDING_MODELS.get(model_id, {})
        dims: Optional[int] = None
        try:
            dims = int(model.get_sentence_embedding_dimension())
        except Exception:
            dims = catalog.get("dimensions")

        max_seq: Optional[int] = None
        try:
            max_seq = int(getattr(model, "max_seq_length", 0)) or catalog.get("max_seq_length")
        except Exception:
            max_seq = catalog.get("max_seq_length")

        meta = {
            "id": model_id,
            "hf_repo": catalog.get("hf_repo", model_id),
            "dimensions": dims,
            "max_seq_length": max_seq,
            "size_mb": catalog.get("size_mb"),
            "description": catalog.get("description"),
            "loaded_at": time.time(),
        }

        with self._lock:
            self._models[model_id] = model
            self._metadata[model_id] = meta
        logger.info("Loaded embedding model %s (%s dims)", model_id, dims)
        return dict(meta)

    def unload(self, model_id: str) -> bool:
        with self._lock:
            existed = model_id in self._models
            self._models.pop(model_id, None)
            self._metadata.pop(model_id, None)
        if existed:
            logger.info("Unloaded embedding model %s", model_id)
        return existed

    # -- inference ------------------------------------------------------------

    def embed(self, model_id: str, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts (blocking)."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str]")

        with self._lock:
            model = self._models.get(model_id)

        if model is None:
            self.load(model_id)
            with self._lock:
                model = self._models[model_id]

        vectors = model.encode(texts, convert_to_numpy=True)
        # Normalize to list[list[float]] regardless of numpy / torch / list return
        try:
            return [list(map(float, v)) for v in vectors.tolist()]
        except AttributeError:
            return [list(map(float, v)) for v in vectors]

    async def aembed(self, model_id: str, texts: List[str]) -> List[List[float]]:
        """Async wrapper — runs :meth:`embed` in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, model_id, texts)

    async def aload(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load, model_id, force)
