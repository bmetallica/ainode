"""AINode doctor — cluster/node health report.

Full implementation arrives in v0.5.0. See
``ops/slices/nvidia-vllm-engine/runbooks/04-install-ux-spec.md`` § 3.1 for
the target UX — per-section green/yellow/red summary of hardware, network,
RoCE/RDMA, storage, credentials, peers, and the AINode service.

The **fabric** section is real: it reports how this node is cabled and which
interface/devices the engine will therefore use, which is the only way to
check the mesh wiring without launching a model. Everything else is still a
pointer to the spec.
"""

from __future__ import annotations

import os
from typing import List, Tuple

_TARGET_SPEC_DOC = (
    "ops/slices/nvidia-vllm-engine/runbooks/04-install-ux-spec.md § 3.1"
)


def fabric_report() -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """Return ``(verdict, rows, warnings)`` describing this node's fabric.

    ``verdict`` is "ok" | "warn" | "unknown". Kept separate from rendering so
    it stays testable without a terminal, and so ``--json`` can reuse it when
    the rest of doctor lands.
    """
    from ainode.cluster.hca_discovery import detect_fabric_ip
    from ainode.cluster.topology import Fabric, detect_topology
    from ainode.core.config import NodeConfig

    config = NodeConfig.load()
    topo = detect_topology(
        configured_interface=config.cluster_interface or "",
        coord_override=config.coord_interface or "",
        hca_override=list(config.rdma_hcas or []),
    )
    coord_ip = detect_fabric_ip(topo.coord_interface) or ""

    rows: List[Tuple[str, str]] = [
        ("Fabric", topo.fabric.value),
        ("Active CX7 links", str(len(topo.links))),
        ("Coordination iface", topo.coord_interface or "<unset>"),
        ("Coordination IP", coord_ip or "<none>"),
        ("NCCL_IB_HCA", ",".join(topo.rdma_hcas) or "<local autodetect>"),
    ]
    for key, value in sorted(topo.nccl_env().items()):
        rows.append((key, value))
    for link in topo.links:
        rows.append((
            f"  link {link.hca}",
            f"{link.netdev or '<no netdev>'} {link.cidr or '<no ipv4>'}",
        ))

    nccl = _nccl_version()
    rows.append(("NCCL version", nccl or "<unknown>"))

    warnings: List[str] = []
    verdict = "ok"

    if topo.fabric is Fabric.UNKNOWN:
        verdict = "unknown"
        warnings.append(
            f"{len(topo.links)} active CX7 link(s) — expected 2 (direct-attach) "
            "or 4 (mesh). Treating this node as direct-attach, i.e. no behaviour "
            "change. Check cabling and `ip link` if you expected a mesh."
        )
    if not coord_ip:
        verdict = "warn"
        warnings.append(
            f"No IPv4 on the coordination interface {topo.coord_interface!r}. "
            "Ray, discovery and SSH have no address to bind — the node cannot "
            "join a cluster."
        )
    if topo.is_mesh and topo.coord_interface.startswith("wl"):
        verdict = "warn"
        warnings.append(
            f"Mesh coordination is running over wireless ({topo.coord_interface}). "
            "Works, but cable the 10G port for predictable cluster formation."
        )
    if topo.is_mesh and not _nccl_supports_subnet_aware_routing(nccl):
        verdict = "warn"
        warnings.append(
            f"NCCL {nccl or 'version unknown'} predates v2.30.7, which is where "
            "NCCL_IB_SUBNET_AWARE_ROUTING was added. On a mesh that setting is "
            "what makes NCCL pick the HCA whose subnet reaches the peer; below "
            "2.30.7 it is silently ignored and the ring will not route. Rebuild "
            "the base image with AINODE_NCCL_TAG=v2.30.7-1 or newer."
        )
    if topo.is_mesh and len(topo.rdma_hcas) < 4:
        verdict = "warn"
        warnings.append(
            f"Mesh detected but only {len(topo.rdma_hcas)} RoCE device(s) will be "
            "given to NCCL. All four are needed to route the ring."
        )

    return verdict, rows, warnings


def _nccl_version() -> str:
    """Best-effort NCCL version of the engine image, or "".

    Read from the package the base image installs rather than from a running
    container, so it works before anything is launched.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "/bin/sh",
             os.environ.get("AINODE_ENGINE_IMAGE", "vllm-node"),
             "-c", "dpkg-query -W -f='${Version}' libnccl2 2>/dev/null || true"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def _nccl_supports_subnet_aware_routing(version: str) -> bool:
    """True unless the version is known to predate v2.30.7.

    Unknown reads as supported: doctor should not cry wolf on a node where the
    engine image is simply not pulled yet. The floor comes from NCCL's own
    source — NCCL_PARAM(IbSubnetAwareRouting, …) appears in v2.30.7-1 and not
    in 2.28.x, 2.29.x or 2.30.3.
    """
    import re as _re

    if not version:
        return True
    m = _re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not m:
        return True
    return tuple(int(g) for g in m.groups()) >= (2, 30, 7)


def cmd_doctor(args) -> None:
    """``ainode doctor`` — fabric section is live, the rest is still a stub."""
    verdict, rows, warnings = "unknown", [], []
    error = ""
    try:
        verdict, rows, warnings = fabric_report()
    except Exception as exc:  # pragma: no cover — defensive, never crash the CLI
        error = f"{type(exc).__name__}: {exc}"

    try:
        from rich.console import Console
        console = Console()
        console.print("[bold]ainode doctor[/bold] — fabric\n")
        if error:
            console.print(f"  [red]could not read fabric:[/red] {error}\n")
        else:
            colour = {"ok": "green", "warn": "yellow"}.get(verdict, "yellow")
            for label, value in rows:
                console.print(f"  {label:<22} [{colour}]{value}[/{colour}]")
            console.print()
            for warning in warnings:
                console.print(f"  [yellow]![/yellow] {warning}")
            if warnings:
                console.print()
        console.print(
            f"  Remaining sections (hardware / storage / credentials / peers /\n"
            f"  service) are still a stub — see {_TARGET_SPEC_DOC}.\n"
        )
    except Exception:  # pragma: no cover — rich should always be present
        print("ainode doctor — fabric")
        if error:
            print(f"  could not read fabric: {error}")
        for label, value in rows:
            print(f"  {label}: {value}")
        for warning in warnings:
            print(f"  ! {warning}")
        print(f"  Remaining sections are a stub. See {_TARGET_SPEC_DOC}.")

    raise SystemExit(0)
