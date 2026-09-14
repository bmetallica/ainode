"""Keep the head's model directory on every node.

The operating rule this implements, stated by the operator: everything goes
through the head. Models are downloaded there, and nowhere else; the sub-nodes
receive copies and keep them, so a launch is a sync of what changed rather
than a fetch of 111 GB.

That is worth having for one measured reason. A node that had to fetch its own
weights did it from Hugging Face at 1.1 MB/s unauthenticated — about five
hours for a 19 GB checkpoint sitting on the machine next door. Copying at
download time moves that cost to a moment when nobody is waiting for a model
to come up, and makes it a link-speed transfer rather than an internet one.

rsync does the work, so a second pass over an unchanged checkpoint compares
sizes and timestamps and sends nothing. That is what makes "sync on every
launch" affordable rather than a second full copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ainode.engine.distribute import DistributionError, ensure_peer_has_dir, hf_cache_dir_name

logger = logging.getLogger(__name__)

__all__ = ["model_source_dir", "peer_targets", "mirror_model_to_peers"]


def model_source_dir(model: str, models_dir: str) -> Optional[Tuple[str, str]]:
    """(parent, dir_name) of this node's copy of ``model``, or None.

    Both layouts, newest first: the flat ``org--name`` our downloader writes,
    then the huggingface_hub cache an out-of-band pull leaves.
    """
    if not model or not models_dir:
        return None
    candidates = [
        (models_dir, model.replace("/", "--")),
        (models_dir, hf_cache_dir_name(model)),
        (str(Path(models_dir) / "hub"), hf_cache_dir_name(model)),
        (str(Path(models_dir) / "hf-cache" / "hub"), hf_cache_dir_name(model)),
    ]
    for parent, name in candidates:
        path = Path(parent) / name
        try:
            if path.is_dir() and any(path.iterdir()):
                return parent, name
        except OSError:
            continue
    return None


def peer_targets(cluster, own_node_id: str) -> List[Tuple[str, str, list]]:
    """(node_id, coordination ip, ib_ips) for every online peer."""
    if cluster is None:
        return []
    try:
        nodes = cluster.get_nodes()
    except Exception:  # pragma: no cover - defensive
        return []
    targets = []
    for node in nodes:
        if node.node_id == own_node_id:
            continue
        host = getattr(node, "fabric_ip", "") or ""
        if host:
            targets.append((node.node_id, host, list(getattr(node, "ib_ips", []) or [])))
    return targets


def _transfer_ip(coord_ip: str, ib_ips: list) -> str:
    try:
        from ainode.cluster.topology import detect_cx7_links, transfer_address

        return transfer_address(ib_ips, coord_ip, detect_cx7_links()) or coord_ip
    except Exception:
        logger.debug("could not resolve a direct transfer address", exc_info=True)
        return coord_ip


def mirror_model_to_peers(
    app,
    model: str,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """Copy ``model`` from this node to every peer. Never raises.

    Returns {node_id: "copied" | "present" | "failed: ..."} so the caller can
    put the outcome in front of the operator instead of logging it where
    nobody looks.
    """
    from ainode.core.config import host_path

    config = app.get("config")
    models_dir = str(getattr(config, "models_dir", "") or "")
    source = model_source_dir(model, models_dir)
    if source is None:
        logger.info("nothing to mirror: %s is not on this node", model)
        return {}
    parent, dir_name = source

    ssh_user = str(getattr(config, "ssh_user", "") or "")
    if not ssh_user:
        logger.warning("no ssh_user configured: cannot mirror %s", model)
        return {}

    own = str(getattr(config, "node_id", "") or "")
    results: dict = {}
    for node_id, coord_ip, ib_ips in peer_targets(app.get("cluster_state"), own):
        transfer_ip = _transfer_ip(coord_ip, ib_ips)
        if on_progress is not None:
            try:
                on_progress(node_id, "copying")
            except Exception:  # pragma: no cover
                logger.exception("mirror progress callback failed")
        try:
            placed = ensure_peer_has_dir(
                ssh_user=ssh_user,
                transfer_ip=transfer_ip,
                source_parent=parent,
                dir_name=dir_name,
                # The peer's own path as ITS host sees it. Our models_dir and
                # the peer's are the same path by construction — both are
                # AINODE_HOME/models — so the head's host view is the right
                # destination. See core.config.host_path.
                target_parent=host_path(parent),
                label="direct RoCE link" if transfer_ip != coord_ip else "",
                # Match, do not merely place: a peer holding a half-finished
                # copy from an earlier attempt passes a "does it exist" test
                # and is then never corrected. rsync sends only what differs.
                resync=True,
            )
        except DistributionError as exc:
            logger.exception("mirroring %s to %s failed", model, node_id)
            results[node_id] = f"failed: {exc}"
        else:
            results[node_id] = "copied" if placed else "failed: source vanished"
        if on_progress is not None:
            try:
                on_progress(node_id, results[node_id])
            except Exception:  # pragma: no cover
                logger.exception("mirror progress callback failed")
    return results
