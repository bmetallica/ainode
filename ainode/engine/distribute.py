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

__all__ = ["DistributionError", "ensure_peer_has_dir", "hf_cache_dir_name"]

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
