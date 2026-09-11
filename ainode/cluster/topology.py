"""Cluster fabric topology detection — direct-attach vs. 3-node mesh.

``hca_discovery`` answers "which HCAs on this box are usable for NCCL?".
This module answers the question one level up: **how is this box wired into
the cluster**, and therefore which interface carries coordination traffic
(Ray, discovery, SSH) versus which devices carry the RDMA data path.

The distinction only matters on a switchless mesh. With two Sparks cabled
back-to-back (or several through a QSFP switch) there is exactly one CX7
subnet that every node shares, so a single interface can be both the
coordination path and the RDMA path — which is the assumption baked into
``NodeConfig.cluster_interface`` today. On a 3-node mesh each ConnectX-7
port is a private point-to-point link to a *different* neighbour, so no
CX7 subnet is shared by all three nodes: coordination has to move to the
common 10G Ethernet, and NCCL gets all four RoCE devices and routes over
the ring itself.

Detection heuristic — **count the active CX7 links**: two means
direct-attach, four means mesh. Adapted from ``detect_interfaces()`` in
eugr/spark-vllm-docker's ``autodiscover.sh`` (MIT); see ``docs/NETWORKING.md``
there for the cabling this encodes. Two deliberate deviations from the
original:

* it reads ``/sys/class/infiniband`` instead of shelling out to
  ``ibdev2netdev``, which is not installed in the AINode image (the same
  reason ``EugrBackend._detect_ib_hca`` reads sysfs), and
* in the mesh case it reports the RoCE devices it actually found rather
  than upstream's hardcoded ``rocep1s0f0,roceP2p1s0f0,...`` list, so a node
  using MOFED ``mlx5_*`` naming works too.

**Nothing here changes behaviour off the mesh path.** ``DIRECT`` and
``UNKNOWN`` deliberately report the caller's configured interface and an
empty HCA list, meaning "keep doing exactly what you did before" — the
existing 2- and 4-node tensor-parallel setups must stay untouched.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ainode.cluster import hca_discovery

logger = logging.getLogger(__name__)

# Number of active CX7 links that identifies each wiring. A DGX Spark exposes
# each physical QSFP port as two RoCE devices (two PCIe x4 links per port —
# see eugr's docs/NETWORKING.md), so one cable lights up two devices and two
# cables light up four.
DIRECT_LINK_COUNT = 2
MESH_LINK_COUNT = 4

# Coordination-interface candidates on a mesh, in preference order: the 10G
# RJ-45 port first, wireless only as a last resort. Same order and same names
# as autodiscover.sh's mesh branch.
MESH_COORD_CANDIDATES = ("enP7s7", "wlP9s9")

# NCCL settings that a mesh needs and a direct-attach fabric does not.
#
#   NCCL_NET_PLUGIN=none          — use NCCL's built-in IB transport; the
#                                   bundled plugin mis-handles the multi-subnet
#                                   ring.
#   NCCL_IB_SUBNET_AWARE_ROUTING=1 — pick the HCA whose subnet actually reaches
#                                   the peer instead of the first one that fits.
#   NCCL_IB_MERGE_NICS=0          — the four devices are four *distinct* links
#                                   to two different neighbours; merging them
#                                   into one logical NIC would route traffic
#                                   down a cable that does not reach the peer.
#
# Values verified upstream in eugr/spark-vllm-docker (autodiscover.sh mesh
# branch + the 3-node nccl-tests invocation in docs/NETWORKING.md).
MESH_NCCL_ENV: Dict[str, str] = {
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_MERGE_NICS": "0",
}

# ib_port_state enum (include/rdma/ib_verbs.h): 4 = ACTIVE, 5 = ACTIVE_DEFER.
# Both carry traffic; everything below is DOWN/INIT/ARMED. Parsed as a leading
# integer because the textual label varies by kernel ("4: ACTIVE", "4\n",
# "4 : ACTIVE").
_ACTIVE_PORT_STATES = frozenset({4, 5})


class Fabric(str, Enum):
    """How this node is wired into the cluster fabric."""

    #: Two active CX7 links — back-to-back pair or QSFP switch. One shared
    #: CX7 subnet, so coordination and RDMA can use the same interface.
    DIRECT = "direct"
    #: Four active CX7 links — switchless mesh. Every CX7 subnet is private
    #: to one neighbour pair, so coordination must move to shared Ethernet.
    MESH = "mesh"
    #: Any other count, including zero. Treated exactly like DIRECT: report
    #: the configured interface and change nothing.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CX7Link:
    """One active RoCE device and the netdev/address it is bound to."""

    hca: str
    netdev: str = ""
    ipv4: str = ""
    cidr: str = ""

    @property
    def has_ipv4(self) -> bool:
        return bool(self.ipv4)


@dataclass(frozen=True)
class TopologyInfo:
    """The wiring of this node plus what each traffic class should use."""

    fabric: Fabric
    links: List[CX7Link] = field(default_factory=list)
    #: Interface for Ray, UDP discovery and SSH. On a mesh this is the shared
    #: Ethernet; otherwise it is whatever the caller passed in.
    coord_interface: str = ""
    #: RoCE devices for ``NCCL_IB_HCA``. Populated on a mesh; **empty**
    #: elsewhere, which means "caller keeps its existing detection".
    rdma_hcas: List[str] = field(default_factory=list)

    @property
    def is_mesh(self) -> bool:
        return self.fabric is Fabric.MESH

    def nccl_env(self) -> Dict[str, str]:
        """Extra NCCL vars this topology needs. Empty off the mesh."""
        return dict(MESH_NCCL_ENV) if self.is_mesh else {}

    def to_dict(self) -> dict:
        return {
            "fabric": self.fabric.value,
            "coord_interface": self.coord_interface,
            "rdma_hcas": list(self.rdma_hcas),
            "links": [
                {"hca": link.hca, "netdev": link.netdev, "ipv4": link.ipv4,
                 "cidr": link.cidr}
                for link in self.links
            ],
        }


def detect_cx7_links() -> List[CX7Link]:
    """Return every RoCE device whose port is ACTIVE, with its netdev + IPv4.

    The sysfs equivalent of ``ibdev2netdev | grep 'Up)'``. Devices with a
    down port are skipped; a device with no associated netdev is still
    reported (with empty ``netdev``) because it counts towards the link
    count even when unaddressed — on a Spark, one twin of each port pair
    routinely has no IP.

    Returns an empty list on a host without RDMA (CI, a dev laptop), which
    lands the caller in :attr:`Fabric.UNKNOWN`.
    """
    base: Path = hca_discovery.SYS_INFINIBAND
    if not base.is_dir():
        return []

    links: List[CX7Link] = []
    for hca in hca_discovery.list_local_hcas():
        if not _port_is_active(base / hca):
            continue
        netdev = _netdev_for_hca(base / hca)
        ipv4, cidr = _netdev_ipv4(netdev) if netdev else ("", "")
        links.append(CX7Link(hca=hca, netdev=netdev, ipv4=ipv4, cidr=cidr))
    return links


def classify_fabric(links: Optional[List[CX7Link]] = None) -> Fabric:
    """Map an active-link count onto a :class:`Fabric`.

    Only the two counts upstream recognises are meaningful; anything else
    is UNKNOWN rather than an error, because an unrecognised count must
    degrade to today's behaviour instead of refusing to launch.
    """
    if links is None:
        links = detect_cx7_links()
    count = len(links)
    if count == MESH_LINK_COUNT:
        return Fabric.MESH
    if count == DIRECT_LINK_COUNT:
        return Fabric.DIRECT
    if count:
        logger.debug(
            "Unexpected active CX7 link count (%d); expected %d (direct) or "
            "%d (mesh). Treating fabric as unknown.",
            count, DIRECT_LINK_COUNT, MESH_LINK_COUNT,
        )
    return Fabric.UNKNOWN


def coordination_interface(
    fabric: Fabric,
    configured_interface: str = "",
    override: str = "",
) -> str:
    """Return the interface that should carry Ray / discovery / SSH.

    Precedence: explicit ``override`` (an operator who knows better than
    us always wins) → on a mesh, the first of
    :data:`MESH_COORD_CANDIDATES` that is up with an IPv4 address → the
    caller's ``configured_interface``.

    Falling back to ``configured_interface`` on a mesh with no shared
    Ethernet is deliberate: it will not form a cluster, but it fails the
    same way the current code does rather than in some new way, and the
    caller logs the reason.
    """
    if override:
        return override
    if fabric is not Fabric.MESH:
        return configured_interface

    for candidate in MESH_COORD_CANDIDATES:
        ipv4, _ = _netdev_ipv4(candidate)
        if ipv4:
            if candidate.startswith("wl"):
                logger.warning(
                    "Mesh coordination falling back to the wireless interface "
                    "%s (%s) — %s has no address. Cluster formation will work "
                    "but latency-sensitive coordination may suffer.",
                    candidate, ipv4, MESH_COORD_CANDIDATES[0],
                )
            return candidate

    logger.warning(
        "Mesh detected but none of %s has an IPv4 address; falling back to the "
        "configured cluster_interface %r. On a mesh that interface reaches only "
        "one neighbour, so cluster formation will likely fail — give this node "
        "an address on the shared Ethernet.",
        ", ".join(MESH_COORD_CANDIDATES), configured_interface,
    )
    return configured_interface


def transfer_address(
    peer_ib_ips: Optional[List[str]],
    peer_coord_ip: str,
    local_links: Optional[List[CX7Link]] = None,
) -> str:
    """Pick the address to push bulk data (model weights, images) to a peer.

    Prefers a peer RoCE address that sits on a subnet **this** node also has a
    link on — that is a direct cable, so the transfer runs at CX7 speed instead
    of over the shared 10G Ethernet. Falls back to ``peer_coord_ip`` when no
    such address exists.

    The fallback is not an edge case on a large mesh: only a *fully connected*
    mesh has a direct link between every pair. Three nodes in a ring happen to
    be fully connected, so every pair there gets a direct address; a fourth
    node would not, and those pairs correctly fall back to the Ethernet.

    Matching is a local subnet comparison, not a probe — no ping, no SSH, no
    timeout to wait out on a down peer. It answers "is there a cable between
    us", which is what routing needs; whether the peer is *up* is discovery's
    job and is already known by the time this is called.

    This is the split upstream's ``autodiscover.sh`` gets from scanning the
    direct-attach subnets for GB10 peers and writing ``COPY_HOSTS`` separately
    from ``CLUSTER_NODES`` (MIT). We reach the same separation from the
    announcement data we already have, without the SSH sweep.
    """
    if not peer_ib_ips:
        return peer_coord_ip
    if local_links is None:
        local_links = detect_cx7_links()

    local_networks = []
    for link in local_links:
        if not link.cidr:
            continue
        try:
            # strict=False: detect_cx7_links already normalises to a network
            # address, but a CX7Link built by hand may carry "10.0.0.11/24"
            # with host bits set, and silently dropping that link would look
            # like "no cable" rather than a malformed value.
            local_networks.append(ipaddress.ip_network(link.cidr, strict=False))
        except ValueError:
            continue
    if not local_networks:
        return peer_coord_ip

    for candidate in sorted(peer_ib_ips):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if any(address in network for network in local_networks):
            return candidate
    return peer_coord_ip


def local_ib_ips(links: Optional[List[CX7Link]] = None) -> List[str]:
    """This node's RoCE link addresses, for the discovery announcement.

    Only addressed links appear — an unaddressed twin is real for the link
    count but useless as a transfer target.
    """
    if links is None:
        links = detect_cx7_links()
    return [link.ipv4 for link in links if link.has_ipv4]


def topology_for_config(config) -> TopologyInfo:
    """:func:`detect_topology` driven by a :class:`~ainode.core.config.NodeConfig`.

    Duck-typed on purpose so this module stays free of a config import and
    remains callable with any object carrying the three attributes.
    """
    return detect_topology(
        configured_interface=getattr(config, "cluster_interface", "") or "",
        coord_override=getattr(config, "coord_interface", "") or "",
        hca_override=list(getattr(config, "rdma_hcas", None) or []),
    )


def detect_topology(
    configured_interface: str = "",
    coord_override: str = "",
    hca_override: Optional[List[str]] = None,
) -> TopologyInfo:
    """Detect the fabric and derive the per-traffic-class settings.

    ``configured_interface`` is ``NodeConfig.cluster_interface``.
    ``coord_override`` / ``hca_override`` are the operator escape hatches
    (``NodeConfig.coord_interface`` / ``rdma_hcas``); either one supplied
    wins over detection for that field alone.

    Off the mesh, ``rdma_hcas`` stays empty on purpose — the backends'
    existing subnet-filtered HCA detection is what the 2- and 4-node
    setups run today, and this must not quietly replace it.
    """
    links = detect_cx7_links()
    fabric = classify_fabric(links)

    if hca_override:
        rdma_hcas = list(hca_override)
    elif fabric is Fabric.MESH:
        # Every active device, no subnet filter: on a mesh the whole point is
        # that each device sits on a *different* subnet, so filtering to one
        # would leave NCCL a single cable to one neighbour.
        rdma_hcas = [link.hca for link in links]
    else:
        rdma_hcas = []

    info = TopologyInfo(
        fabric=fabric,
        links=links,
        coord_interface=coordination_interface(
            fabric, configured_interface, override=coord_override
        ),
        rdma_hcas=rdma_hcas,
    )
    logger.debug("Detected topology: %s", info.to_dict())
    return info


# ---------------------------------------------------------------------------
# sysfs / ip helpers
# ---------------------------------------------------------------------------


def _port_is_active(hca_dir: Path) -> bool:
    """True if port 1 of this HCA is ACTIVE or ACTIVE_DEFER."""
    try:
        state_text = (hca_dir / "ports" / "1" / "state").read_text().strip()
    except OSError:
        return False
    match = re.match(r"(\d+)", state_text)
    return bool(match) and int(match.group(1)) in _ACTIVE_PORT_STATES


def _netdev_for_hca(hca_dir: Path) -> str:
    """Name of the Ethernet netdev bound to this HCA, or "" if none.

    A pure-IB port with no Ethernet overlay has an empty ``device/net``
    directory; so does a device the kernel has not finished binding.
    """
    try:
        entries = sorted(p.name for p in (hca_dir / "device" / "net").iterdir())
    except OSError:
        return ""
    return entries[0] if entries else ""


def _netdev_ipv4(netdev: str) -> tuple[str, str]:
    """Return ``(address, cidr)`` for a netdev, or ``("", "")``.

    ``cidr`` is the normalised network the address sits in (e.g.
    ``192.168.177.0/24``), which is what a caller needs to decide whether a
    peer address is reachable over this link.
    """
    if not netdev:
        return "", ""
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "dev", netdev],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ip addr show %s failed: %s", netdev, exc)
        return "", ""
    if out.returncode != 0:
        return "", ""
    match = re.search(r"inet\s+(\S+)", out.stdout)
    if not match:
        return "", ""
    try:
        iface = ipaddress.ip_interface(match.group(1))
    except ValueError:
        return "", ""
    return str(iface.ip), str(iface.network)


__all__ = [
    "CX7Link",
    "DIRECT_LINK_COUNT",
    "Fabric",
    "MESH_COORD_CANDIDATES",
    "MESH_LINK_COUNT",
    "MESH_NCCL_ENV",
    "TopologyInfo",
    "classify_fabric",
    "coordination_interface",
    "detect_cx7_links",
    "detect_topology",
    "local_ib_ips",
    "topology_for_config",
    "transfer_address",
]
