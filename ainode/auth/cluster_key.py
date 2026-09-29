"""A shared secret the nodes present to each other.

The web login (auth/web_login.py) puts every ``/api/*`` route behind a session.
Nodes call each other's ``/api/*`` all the time — unload fan-out, mirror
status, measurements, memory-guard settings, launch-config, telemetry — and
none of those callers is a browser with a cookie. Without this they would lock
each other out the moment the login is switched on.

The key is a random string in ``~/.ainode/cluster.key``. Every node needs the
SAME one. scripts/update-cluster.sh and scripts/install.sh copy the head's key
to the peers over the SSH they already use, so it arrives with the update that
needs it; ``ainode cluster-key`` prints or sets it by hand.

Why not trust the peers' IP addresses instead: discovery is an unauthenticated
UDP broadcast, so anything on the LAN can announce itself as a member, and
"a member's address" is therefore not evidence of anything.

The file is re-read when its mtime changes, so a key copied in by a script
takes effect without restarting the container that reads it.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["CLUSTER_HEADER", "cluster_key", "cluster_headers",
           "is_cluster_request", "key_path", "cluster_trace_config",
           "cluster_hosts"]

#: The header a node sends when it calls another node's API.
CLUSTER_HEADER = "X-AINode-Cluster-Key"

_CACHE: Dict[str, object] = {"mtime": None, "path": None, "key": ""}


def key_path() -> Path:
    from ainode.core.config import AINODE_HOME

    return Path(AINODE_HOME) / "cluster.key"


def cluster_key(create: bool = True) -> str:
    """This node's cluster key, created on first use when ``create``.

    Never raises. An unreadable key is the empty string, and an empty key
    matches nothing — a node that cannot read its key refuses peer calls
    rather than accepting all of them.
    """
    path = key_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        if not create:
            return ""
        return _create(path)
    except OSError:
        logger.debug("could not stat %s", path, exc_info=True)
        return ""
    if _CACHE["path"] == str(path) and _CACHE["mtime"] == stat.st_mtime:
        return str(_CACHE["key"])
    try:
        key = path.read_text().strip()
    except OSError:
        logger.warning("could not read the cluster key at %s", path)
        return ""
    _CACHE.update(path=str(path), mtime=stat.st_mtime, key=key)
    return key


def _create(path: Path) -> str:
    key = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL: two workers racing to create it must not end up with two
        # different keys, one of which is then on disk and the other in use.
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as sink:
            sink.write(key + "\n")
        logger.info("created a new cluster key at %s — copy it to the other "
                    "nodes with scripts/update-cluster.sh", path)
    except FileExistsError:
        return cluster_key(create=False)
    except OSError:
        logger.warning("could not create the cluster key at %s", path)
        return ""
    return cluster_key(create=False)


def cluster_headers() -> Dict[str, str]:
    """Headers for a call to another node. {} when there is no key."""
    key = cluster_key()
    return {CLUSTER_HEADER: key} if key else {}


def is_cluster_request(headers) -> bool:
    """Does this request carry this node's cluster key?"""
    offered: Optional[str] = None
    try:
        offered = headers.get(CLUSTER_HEADER)
    except Exception:
        return False
    if not offered:
        return False
    key = cluster_key(create=False)
    return bool(key) and hmac.compare_digest(str(offered), key)


def cluster_hosts(app) -> set:
    """Every address another node of this cluster is known by."""
    hosts = set()
    cluster = app.get("cluster_state") if app is not None else None
    if cluster is None:
        return hosts
    try:
        members = list(cluster.members())
    except Exception:
        return hosts
    own = str(getattr(app.get("config"), "node_id", "") or "")
    for node in members:
        if getattr(node, "node_id", "") == own:
            continue
        for attr in ("fabric_ip", "peer_ip", "node_name"):
            value = str(getattr(node, attr, "") or "").strip()
            if value:
                hosts.add(value)
        for value in (getattr(node, "ib_ips", None) or []):
            if value:
                hosts.add(str(value))
    return hosts


def cluster_trace_config(app):
    """An aiohttp TraceConfig that adds the key to calls bound for a peer.

    On the shared client session rather than at each call site, because there
    are fifteen of those and a sixteenth will be written. And added per request
    by destination rather than as a default header, because the same session
    also talks to engines on localhost and, through some routes, to the outside
    — the key must never leave the cluster.
    """
    import aiohttp

    async def _on_request_start(session, context, params) -> None:
        try:
            host = params.url.host or ""
        except Exception:
            return
        if host and host in cluster_hosts(app):
            key = cluster_key()
            if key and CLUSTER_HEADER not in params.headers:
                params.headers[CLUSTER_HEADER] = key

    trace = aiohttp.TraceConfig()
    trace.on_request_start.append(_on_request_start)
    return trace
