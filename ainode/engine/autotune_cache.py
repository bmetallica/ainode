"""No stored FlashInfer tuning results on any rank of a distributed launch.

vLLM keeps what FlashInfer's autotuner measured in its own cache directory
(``~/.ainode/cache/vllm/flashinfer_autotune_cache/<version>/<arch>/<hash>/
autotune_configs.json``), and on the next launch a rank that finds a result
there skips the measurement. Two facts make that deadlock a distributed
launch:

* the measurement is collective — tuning a kernel synchronises the ranks
  (``_sync_oom_across_tune_group`` does an ``all_reduce`` over the tune group);
* only rank 0 writes the file, and the key carries the rank. From the file on
  spark-1432, a key of Qwen3.8-Flash-Next's MoE:

      ('trtllm::fused_moe::gemm1', 'MoERunner', ((1024, 1280), (256, 1280, 160),
       …), (…, 10, 1, 0, 2, 0, 1, 0, …))
                  top_k, tp_size, tp_rank, ep_size, ep_rank, cluster size/rank

So from the second launch on, rank 0 finds its results ("Config cache hit")
and runs ahead; rank 1 looks for ep_rank=1, finds nothing, tunes, and waits in
the all_reduce; rank 0 meanwhile starts NCCL work and blocks loading its next
Triton kernel (cuModuleLoadData waits for the GPU). Gloo gives up after
thirty minutes:

    Worker_TP1_EP1 … flashinfer/autotuner/autotuner.py … all_reduce …
    RuntimeError: … Timed out waiting 1800000ms for send operation to complete

Copying rank 0's file to the others (the first attempt at this) changes
nothing — their keys are not in it. What works is what the first launch did:
every rank tunes, together. So before a distributed launch the directory is
emptied on every node the launch uses, the leading one included. It costs the
tuning (about a minute on Flash-Next) on every distributed launch, and keeps
the autotuner on. The files belong to root on a peer (the engine container
wrote them) and ssh reaches the peer as the install user, so the peers are
emptied through the AINode API.
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["autotune_dir", "snapshot", "replace", "push_to_peers",
           "clear_for_distributed_launch", "register_autotune_routes"]

#: Bounds for what is sent: tuning results are JSON of a few kilobytes each.
MAX_FILES = 500
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def autotune_dir() -> Path:
    from ainode.core.config import ENGINE_CACHE_DIR

    return Path(ENGINE_CACHE_DIR) / "vllm" / "flashinfer_autotune_cache"


def _safe(rel: str) -> Optional[PurePosixPath]:
    path = PurePosixPath(str(rel))
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        return None
    return path


def snapshot(root: Optional[Path] = None) -> Dict[str, str]:
    """{relative path: base64 content} of every file under ``root``."""
    root = Path(root or autotune_dir())
    out: Dict[str, str] = {}
    total = 0
    if not root.is_dir():
        return out
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file.relative_to(root).as_posix()
        if _safe(rel) is None:
            continue
        try:
            data = file.read_bytes()
        except OSError:
            continue
        if len(data) > MAX_FILE_BYTES or len(out) >= MAX_FILES \
                or total + len(data) > MAX_TOTAL_BYTES:
            logger.warning("autotune cache: leaving %s out (limits)", rel)
            continue
        total += len(data)
        out[rel] = base64.b64encode(data).decode("ascii")
    return out


def replace(files: Dict[str, str], root: Optional[Path] = None) -> int:
    """Make ``root`` hold exactly ``files``. Returns how many were written.
    Raises ValueError on a path that would leave the directory."""
    root = Path(root or autotune_dir())
    decoded = []
    total = 0
    if len(files) > MAX_FILES:
        raise ValueError("too many files")
    for rel, b64 in files.items():
        path = _safe(rel)
        if path is None:
            raise ValueError(f"unsafe path {rel!r}")
        data = base64.b64decode(str(b64), validate=True)
        total += len(data)
        if len(data) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("too large")
        decoded.append((path, data))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for path, data in decoded:
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return len(decoded)


def push_to_peers(peers: Iterable[str], port: int = 3000, *,
                  files: Optional[Dict[str, str]] = None,
                  timeout: float = 15.0) -> List[str]:
    """Send this node's tuning results to every peer. Returns the peers that
    could not be brought in line (the launch goes on; the log says so)."""
    from ainode.auth.cluster_key import cluster_headers

    body = json.dumps({"files": snapshot() if files is None else files}).encode()
    failed = []
    for host in peers:
        host = str(host or "").strip()
        if not host:
            continue
        request = urllib.request.Request(
            f"http://{host}:{int(port)}/api/engine/autotune-cache", data=body,
            method="PUT", headers={"Content-Type": "application/json",
                                   **cluster_headers()})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                if resp.status != 200:
                    failed.append(host)
        except Exception as exc:
            logger.warning("could not align the tuning cache on %s: %s", host, exc)
            failed.append(host)
    return failed


def clear_for_distributed_launch(peers: Iterable[str], port: int = 3000) -> List[str]:
    """Empty the tuning cache here and on every peer. Returns the peers that
    could not be emptied."""
    try:
        replace({})
    except Exception:
        logger.exception("could not empty the local FlashInfer tuning cache")
    return push_to_peers(peers, port, files={})


async def handle_put(request):
    """PUT /api/engine/autotune-cache {"files": {path: base64}} — replace this
    node's FlashInfer tuning results; the leading node of a distributed launch
    sends {} to empty them. Cluster key only."""
    import asyncio

    from aiohttp import web

    from ainode.auth.cluster_key import is_cluster_request

    if not is_cluster_request(request.headers):
        return web.json_response({"error": "cluster key required"}, status=403)
    try:
        files = (await request.json()).get("files") or {}
        if not isinstance(files, dict):
            raise ValueError("files must be an object")
        written = await asyncio.get_event_loop().run_in_executor(None, replace, files)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"ok": True, "files": written})


def register_autotune_routes(app) -> None:
    app.router.add_put("/api/engine/autotune-cache", handle_put)
