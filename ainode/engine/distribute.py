"""Push model weights to a peer over SSH.

Every rank has to be able to read the model from its own local disk, so a
distributed launch either finds the weights already there or puts them there.
Shared storage is the other way to solve this and AINode does not use it: the
only thing on ``/mnt/shared-models`` is a 3 KB NCCL init script. On a
switchless mesh shared storage would also be the wrong answer — no CX7 subnet
reaches every node, so an NFS export has to live on the shared 10G Ethernet,
where N ranks reading one multi-hundred-GB checkpoint at once each get 1/N of a
single link, on every load. A one-time copy to local disk over the direct
100G links costs disk space instead, which is the cheaper resource here.

The transfer address is chosen by the launch path (``peer_transfer_ips``): on a
mesh it is the far end of a direct cable, elsewhere it is the same address
coordination uses. See ``ainode.cluster.topology.transfer_address``.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["DistributionError", "ensure_peer_has_dir", "ensure_peer_has_image",
           "ensure_local_image", "hf_cache_dir_name", "local_image_id"]

# Long enough for a frontier MoE over a slow link; short enough that a hung
# transfer does not wedge a launch forever.
TRANSFER_TIMEOUT = 7200
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10"]


class DistributionError(RuntimeError):
    """Raised when weights could not be placed on a peer."""


def hf_cache_dir_name(model: str) -> str:
    """HF cache directory name for a repo id: ``org/name`` → ``models--org--name``."""
    return "models--" + model.replace("/", "--")


def ensure_peer_has_dir(
    *,
    ssh_user: str,
    transfer_ip: str,
    source_parent: str,
    dir_name: str,
    target_parent: str,
    label: str = "",
    on_start: Optional[Callable[[], None]] = None,
) -> bool:
    """Copy ``source_parent/dir_name`` to ``target_parent/dir_name`` on a peer.

    Returns True when the peer ends up with the directory — including the case
    where it already had it, which is the common one after the first launch.
    Returns False when the *source* is missing, since the engine will report
    that far more clearly than a transfer error would. Raises
    :class:`DistributionError` only when a transfer was attempted and failed.

    Uses rsync when available because it is resumable and incremental, so a
    re-launch after a dropped transfer does not re-send the whole checkpoint;
    falls back to tar-over-ssh, which every image has.
    """
    source = Path(source_parent) / dir_name
    if not source.is_dir():
        return False

    ssh_target = f"{ssh_user}@{transfer_ip}"
    remote_dir = target_parent.rstrip("/") + "/" + dir_name

    probe = subprocess.run(
        ["ssh", *SSH_OPTS, ssh_target,
         f"test -d {shlex.quote(remote_dir)} && echo present || echo missing"],
        capture_output=True, text=True, timeout=30,
    )
    if "present" in (probe.stdout or ""):
        return True

    if on_start is not None:
        try:
            on_start()
        except Exception:  # pragma: no cover — a status callback must not abort a copy
            logger.exception("on_start callback failed")

    logger.info("Distributing %s to %s%s", dir_name, transfer_ip,
                f" ({label})" if label else "")

    ssh_e = "ssh " + " ".join(SSH_OPTS)
    if shutil.which("rsync"):
        subprocess.run(
            ["ssh", *SSH_OPTS, ssh_target, f"mkdir -p {shlex.quote(target_parent)}"],
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["rsync", "-a", "--partial", "-e", ssh_e,
             f"{source}/", f"{ssh_target}:{remote_dir}/"],
            capture_output=True, text=True, timeout=TRANSFER_TIMEOUT,
        )
    else:
        tar = (
            f"tar -C {shlex.quote(source_parent)} -cf - {shlex.quote(dir_name)} | "
            f"{ssh_e} {shlex.quote(ssh_target)} "
            f"'mkdir -p {shlex.quote(target_parent)} && "
            f"tar -C {shlex.quote(target_parent)} -xf -'"
        )
        result = subprocess.run(["bash", "-lc", tar], capture_output=True,
                                text=True, timeout=TRANSFER_TIMEOUT)

    if result.returncode != 0:
        raise DistributionError(
            f"Failed to distribute {dir_name} to {transfer_ip} "
            f"(rc={result.returncode}): {result.stderr.strip()[:300]}"
        )
    logger.info("Distributed %s to %s", dir_name, transfer_ip)
    return True


# --- engine images -----------------------------------------------------------
#
# Every node runs the engine in a container, so every node needs that image.
# launch-cluster.sh checks and aborts when one is missing — which is correct and
# unhelpful: the operator is told to go and fix it on three machines by hand.
# A model pinning an engine_image (a catalog recipe does) makes this routine
# rather than exceptional, so the head places the image the way it places
# weights.
#
# Pull first, copy second. A registry image is far cheaper to pull on each peer
# in parallel — layers dedupe against what is already there — than to stream ~20
# GB through the head. The copy is the fallback for a locally built image, which
# has no registry to pull from.

IMAGE_PULL_TIMEOUT = 3600
IMAGE_COPY_TIMEOUT = 7200


def local_image_id(image: str) -> str:
    """Content-addressable id of a local image, or ""."""
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def ensure_local_image(image: str, on_pull: Optional[Callable[[], None]] = None) -> str:
    """Make ``image`` available on THIS node. Returns "present" or "pulled".

    The head needs it as much as the peers do: launch-cluster.sh inspects it
    here first and aborts before touching a worker. Pulling it as part of the
    launch beats telling the operator to go and pull it and come back.

    ``on_pull`` fires only when a pull actually starts — several GB of silence
    that the caller will want to put on screen.
    """
    image = (image or "").strip()
    if not image or local_image_id(image):
        return "present"
    if on_pull is not None:
        try:
            on_pull()
        except Exception:  # pragma: no cover — a status callback must not abort a pull
            logger.exception("on_pull callback failed")
    logger.info("Pulling %s locally", image)
    try:
        subprocess.run(["docker", "pull", image], capture_output=True,
                       text=True, timeout=IMAGE_PULL_TIMEOUT)
    except Exception:
        pass
    if not local_image_id(image):
        raise DistributionError(
            f"{image} is not on this node and could not be pulled. Check the "
            f"model's engine image, or pull or build it here first."
        )
    return "pulled"


def _peer_image_id(ssh_user: str, host: str, image: str) -> str:
    try:
        out = subprocess.run(
            ["ssh", *SSH_OPTS, f"{ssh_user}@{host}",
             f"docker image inspect --format '{{{{.Id}}}}' {shlex.quote(image)} 2>/dev/null || true"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return ""
    return (out.stdout or "").strip()


def ensure_peer_has_image(
    *,
    ssh_user: str,
    coord_ip: str,
    transfer_ip: str,
    image: str,
    on_start: Optional[Callable[[], None]] = None,
) -> str:
    """Make ``image`` available on a peer, identical to the head's copy.

    Returns what was done: "present", "pulled", "copied". Raises
    :class:`DistributionError` when the peer ends up without it.

    The launcher requires the image **id** to match across nodes, not just the
    tag — a tag that moved in the registry between two pulls would otherwise
    produce ranks running different builds, which fails later and far less
    clearly. So a pull that lands a different id falls through to the copy.

    Control traffic goes over ``coord_ip``; a copy streams through the head and
    goes over ``transfer_ip``, which on a mesh is a direct cable.
    """
    image = (image or "").strip()
    if not image:
        return "present"

    head_id = local_image_id(image)
    peer_id = _peer_image_id(ssh_user, coord_ip, image)
    if peer_id and (not head_id or peer_id == head_id):
        return "present"

    if on_start is not None:
        try:
            on_start()
        except Exception:  # pragma: no cover
            logger.exception("on_start callback failed")

    # 1. Pull on the peer.
    logger.info("Pulling %s on %s", image, coord_ip)
    try:
        subprocess.run(
            ["ssh", *SSH_OPTS, f"{ssh_user}@{coord_ip}",
             f"docker pull {shlex.quote(image)}"],
            capture_output=True, text=True, timeout=IMAGE_PULL_TIMEOUT,
        )
    except Exception:
        logger.info("Pull of %s on %s did not complete; will try copying",
                    image, coord_ip)
    peer_id = _peer_image_id(ssh_user, coord_ip, image)
    if peer_id and (not head_id or peer_id == head_id):
        return "pulled"

    # 2. Copy from the head. The only route for a locally built image, and the
    #    correction when a pull produced a different build of the same tag.
    if not head_id:
        raise DistributionError(
            f"{image} is on neither this node nor {coord_ip}, and could not be "
            f"pulled. Pull or build it on this node first, or correct the "
            f"model's engine image."
        )
    logger.info("Copying %s to %s%s", image, transfer_ip,
                " over a direct link" if transfer_ip != coord_ip else "")
    save = (
        f"docker save {shlex.quote(image)} | "
        f"ssh {' '.join(SSH_OPTS)} {shlex.quote(f'{ssh_user}@{transfer_ip}')} "
        f"docker load"
    )
    result = subprocess.run(["bash", "-lc", save], capture_output=True,
                            text=True, timeout=IMAGE_COPY_TIMEOUT)
    if result.returncode != 0:
        raise DistributionError(
            f"Could not place {image} on {transfer_ip} "
            f"(rc={result.returncode}): {result.stderr.strip()[:300]}"
        )
    return "copied"
