"""Get a model's weights onto this node before the engine asks for them.

A distributed launch pushes weights from the head to its peers. A solo launch
on a node that does not have them had no equivalent, so the engine fetched
from Hugging Face — measured on this cluster at 1.1 MB/s unauthenticated,
about five hours for a 19 GB checkpoint that was already sitting on the
machine next door. The same gap cost 17 GB of Qwen the day before.

The direction is the only difference: ask each peer what it has, and pull from
one that has it, over the direct RoCE link where there is a cable.

Deliberately best-effort. Every failure here falls through to the engine's own
download, which is what happened before this module existed — a peer that is
down, an SSH key that is not set up or a half-copied directory must not turn a
working (if slow) launch into a failed one.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ainode.engine.distribute import DistributionError, fetch_dir_from_peer, hf_cache_dir_name
from ainode.engine.serve_args import local_model_dir

logger = logging.getLogger(__name__)

__all__ = ["model_is_local", "peers_with_model", "fetch_model_from_peer"]

#: Long enough for a busy node to answer, short enough that three unreachable
#: peers do not add a minute to every launch.
PEER_QUERY_TIMEOUT = 5


def model_is_local(model: str, models_dir: str) -> bool:
    """True when the weights are already here, in any layout.

    Non-empty, in every layout — matching local_model_dir, which has always
    required it. An empty directory is what an aborted download leaves, and
    this cluster has produced several; treating one as "already here" would
    skip the copy and hand the engine a directory with nothing in it.

    Not a completeness check: a half-copied directory still reads as present.
    Detecting that needs the repo's file list, which means the network, in the
    path of every launch. rsync --partial resumes, and the engine downloads
    what is missing, so a partial copy costs time rather than correctness.
    """
    if not model or not models_dir:
        return False
    if local_model_dir(model, models_dir):
        return True
    for parent in (Path(models_dir), Path(models_dir) / "hub",
                   Path(models_dir) / "hf-cache" / "hub"):
        candidate = parent / hf_cache_dir_name(model)
        try:
            if candidate.is_dir() and any(candidate.iterdir()):
                return True
        except OSError:
            continue
    return False


def _peer_has(host: str, port: int, model: str) -> bool:
    """Ask a peer's /api/models/downloaded whether it has ``model``."""
    url = f"http://{host}:{port}/api/models/downloaded"
    try:
        with urllib.request.urlopen(url, timeout=PEER_QUERY_TIMEOUT) as response:
            payload = json.loads(response.read().decode())
    except Exception:
        logger.debug("could not ask %s what it has", host, exc_info=True)
        return False
    wanted = model.lower()
    for entry in (payload.get("models") or []):
        if str(entry.get("hf_repo") or "").lower() == wanted:
            return True
    return False


def peers_with_model(cluster, own_node_id: str, model: str) -> List[Tuple[str, str, list]]:
    """(node_id, coordination ip, ib_ips) for every peer that has ``model``.

    The head first. The operating rule is that models arrive on the head and
    are mirrored outward, so the head holds the version that defines the
    others; taking a copy from an arbitrary peer could propagate whatever that
    peer happened to end up with. The rest follow, because a head that is busy
    or down is not a reason to fall back to the internet.
    """
    if cluster is None or not model:
        return []
    try:
        nodes = cluster.get_nodes()
    except Exception:  # pragma: no cover - defensive
        return []
    found = []
    for node in nodes:
        if node.node_id == own_node_id:
            continue
        host = getattr(node, "fabric_ip", "") or ""
        if not host:
            continue
        if _peer_has(host, node.web_port, model):
            found.append((bool(getattr(node, "is_master", False)), node.node_id,
                          host, list(getattr(node, "ib_ips", []) or [])))
    found.sort(key=lambda item: (not item[0], item[1]))
    return [(node_id, host, ib) for _, node_id, host, ib in found]


def _transfer_ip(coord_ip: str, ib_ips: list) -> str:
    """The direct cable to that peer where there is one, else the shared LAN."""
    try:
        from ainode.cluster.topology import detect_cx7_links, transfer_address

        return transfer_address(ib_ips, coord_ip, detect_cx7_links()) or coord_ip
    except Exception:
        logger.debug("could not resolve a direct transfer address", exc_info=True)
        return coord_ip


def fetch_model_from_peer(
    *,
    cluster,
    own_node_id: str,
    ssh_user: str,
    model: str,
    models_dir: str,
    host_models_dir: Optional[str] = None,
    sync: bool = False,
    on_start: Optional[Callable[[str], None]] = None,
) -> str:
    """Try to copy ``model`` here from a peer that has it.

    Returns "present" (already here and not syncing), "fetched", "synced" or
    "absent" — never raises. A caller that gets anything else than a transfer
    simply launches, and the engine downloads as it always did.

    ``sync`` keeps going when the weights are already here, so the head's copy
    decides what this node serves. rsync compares sizes and timestamps, so an
    unchanged checkpoint costs a directory listing and sends nothing — which
    is what makes this affordable on every launch rather than a second full
    copy.
    """
    have_it = model_is_local(model, models_dir)
    if have_it and not sync:
        return "present"

    for node_id, coord_ip, ib_ips in peers_with_model(cluster, own_node_id, model):
        transfer_ip = _transfer_ip(coord_ip, ib_ips)
        label = "direct RoCE link" if transfer_ip != coord_ip else ""
        # Both layouts: the peer may hold a directory download or an HF cache
        # entry, and which one it is decides where this node must put it.
        parent = host_models_dir or models_dir
        attempts = (
            (parent, model.replace("/", "--"), models_dir),
            (str(Path(parent) / "hub"), hf_cache_dir_name(model),
             str(Path(models_dir) / "hub")),
        )
        for remote_parent, dir_name, target_parent in attempts:
            try:
                placed = fetch_dir_from_peer(
                    ssh_user=ssh_user,
                    transfer_ip=transfer_ip,
                    remote_parent=remote_parent,
                    dir_name=dir_name,
                    target_parent=target_parent,
                    label=label,
                    on_start=(lambda nid=node_id: on_start(nid)) if on_start else None,
                )
            except DistributionError:
                # Downloading is slower, not impossible. Say what happened and
                # let the launch proceed.
                logger.exception("fetching %s from %s failed", model, node_id)
                continue
            if placed:
                return "synced" if have_it else "fetched"
    return "present" if have_it else "absent"
