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

from ainode.engine.acquire import model_is_local, required_repos
from ainode.engine.distribute import DistributionError, ensure_peer_has_dir, hf_cache_dir_name

logger = logging.getLogger(__name__)

__all__ = ["model_source_dir", "peer_targets", "mirror_model_to_peers",
           "ensure_dependencies"]


def ensure_dependencies(app, model: str) -> List[str]:
    """Download, here, the repos ``model``'s recipe pulls in of its own accord.

    A checkpoint is not always the whole of what a launch fetches. Gemma 4
    26B's recipe names a speculative drafter — google/gemma-4-26B-A4B-it-assistant,
    801 MB — and vLLM downloads it separately, at launch, on whichever node
    is launching. That produced exactly the situation this deployment exists
    to prevent: the drafter ended up on a sub-node, downloaded from Hugging
    Face, and the head did not have it at all.

    A dependency is part of the model. Downloading it here, with the model,
    means the mirror carries it and no sub-node ever reaches for it.

    Only on a node that may download, and never fatal: a drafter that cannot
    be fetched leaves a launch that would have failed anyway, and failing the
    *download* of a model that is otherwise complete helps nobody.

    Returns the dependency repos now on this node.
    """
    config = app.get("config")
    if not getattr(config, "download_from_hub", True):
        return []
    models_dir = str(getattr(config, "models_dir", "") or "")
    if not models_dir:
        return []

    manager = app.get("model_manager")
    info = None
    if manager is not None:
        try:
            info = manager._find_catalog_by_hf_repo(model)
        except Exception:
            logger.debug("no catalog entry for %s", model, exc_info=True)
    recipe_args = list(getattr(info, "extra_vllm_args", []) or [])
    wanted = [repo for repo in required_repos(model, recipe_args) if repo != model]

    present: List[str] = []
    for repo in wanted:
        if model_is_local(repo, models_dir):
            present.append(repo)
            continue
        # A peer before the internet. The drafter that started this was
        # downloaded by a sub-node at launch time, so the copy that exists is
        # on the fabric and the head is the one missing it — fetching it back
        # over the link beats fetching it again over the uplink.
        try:
            from ainode.core.config import host_path
            from ainode.engine.acquire import fetch_model_from_peer

            outcome = fetch_model_from_peer(
                cluster=app.get("cluster_state"),
                own_node_id=str(getattr(config, "node_id", "") or ""),
                ssh_user=str(getattr(config, "ssh_user", "") or ""),
                model=repo,
                models_dir=models_dir,
                host_models_dir=host_path(models_dir),
            )
        except Exception:
            logger.exception("could not ask the cluster for %s", repo)
            outcome = "absent"
        if outcome in ("fetched", "present"):
            present.append(repo)
            continue

        try:
            from huggingface_hub import snapshot_download

            logger.info("fetching %s, which %s's recipe requires", repo, model)
            snapshot_download(
                repo_id=repo,
                cache_dir=models_dir,
                token=str(getattr(config, "hf_token", "") or "") or None,
            )
            present.append(repo)
        except Exception:
            logger.exception("could not fetch %s for %s", repo, model)
    return present


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
