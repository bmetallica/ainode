"""RoCE traffic, read where it actually is.

The dashboard reported zero bytes on the ring interfaces of a cluster whose
fabric was saturated, and the interfaces were not lying: RDMA writes straight
from one HCA to the other's memory, without the kernel's network stack ever
seeing a packet. ``/proc/net/dev`` — which is what psutil reads, and therefore
what ``system.network`` reports — counts what the stack handled, so on a node
doing nothing but RDMA it counts nothing.

The counters that do move are the HCA's own, under

    /sys/class/infiniband/<hca>/ports/<n>/counters/

with one trap that catches everyone who writes this by hand: ``port_xmit_data``
and ``port_rcv_data`` count **32-bit words, not bytes**. A figure that is a
quarter of the truth looks plausible enough to build a dashboard on, which is
the worst kind of wrong.

The error counters in the same directory matter more than the traffic ones on
a switchless ring: every link is a single cable to one neighbour, so a cable
starting to fail has no redundancy to hide behind. It shows up here as a
rising ``port_rcv_errors`` or ``link_downed`` long before NCCL gives up
mid-run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

__all__ = ["FabricSampler", "SYS_INFINIBAND", "WORD_BYTES"]

SYS_INFINIBAND = Path("/sys/class/infiniband")

#: port_xmit_data and port_rcv_data are in 32-bit words.
WORD_BYTES = 4

#: Counted and published as-is, because they only ever go up and what matters
#: is that they moved at all. On a switchless ring a single one of these
#: incrementing is a cable to look at.
_ERROR_COUNTERS = (
    "link_downed",
    "link_error_recovery",
    "local_link_integrity_errors",
    "port_rcv_errors",
    "port_xmit_discards",
    "symbol_error",
)


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _read_int(path: Path) -> int:
    raw = _read(path)
    try:
        return int(raw)
    except ValueError:
        return 0


def _state(raw: str) -> str:
    """``4: ACTIVE`` → ``ACTIVE``."""
    return raw.split(":")[-1].strip() if raw else ""


class FabricSampler:
    """Per-port RDMA counters, with rates between calls.

    Rates need two readings, so the first sample carries the raw counters and
    no rate — same contract as the system sampler, and the same reason: a rate
    invented from one reading would be a guess presented as a measurement.
    """

    def __init__(self, root: Path = SYS_INFINIBAND):
        self._root = Path(root)
        self._last: Dict[str, tuple] = {}
        self._last_at = 0.0

    def sample(self) -> Dict[str, Any]:
        if not self._root.is_dir():
            return {}
        now = time.time()
        elapsed = now - self._last_at if self._last_at else 0.0
        out: Dict[str, Any] = {}

        for hca in sorted(p for p in self._root.iterdir() if p.is_dir()):
            ports = hca / "ports"
            if not ports.is_dir():
                continue
            for port in sorted(p for p in ports.iterdir() if p.is_dir()):
                entry = self._port(hca.name, port, elapsed)
                if entry:
                    out[f"{hca.name}:{port.name}"] = entry

        self._last_at = now
        return out

    def _port(self, hca: str, port: Path, elapsed: float) -> Dict[str, Any]:
        counters = port / "counters"
        if not counters.is_dir():
            return {}
        key = f"{hca}:{port.name}"

        xmit_words = _read_int(counters / "port_xmit_data")
        rcv_words = _read_int(counters / "port_rcv_data")
        entry: Dict[str, Any] = {
            "hca": hca,
            "port": port.name,
            # Converted here, once, so nothing downstream has to remember the
            # unit. The raw word counts stay available for anyone who wants
            # to do the arithmetic themselves.
            "tx_bytes": xmit_words * WORD_BYTES,
            "rx_bytes": rcv_words * WORD_BYTES,
            "tx_packets": _read_int(counters / "port_xmit_packets"),
            "rx_packets": _read_int(counters / "port_rcv_packets"),
        }

        state = _state(_read(port / "state"))
        if state:
            entry["state"] = state
        phys = _state(_read(port / "phys_state"))
        if phys:
            entry["phys_state"] = phys
        rate = _read(port / "rate")
        if rate:
            # "400 Gb/sec (4X NDR)" — the headline number is what a dashboard
            # wants to divide by.
            entry["rate"] = rate
            head = rate.split()[0]
            try:
                entry["link_gbit"] = float(head)
            except ValueError:
                pass

        errors = {}
        for name in _ERROR_COUNTERS:
            path = counters / name
            if path.exists():
                errors[name] = _read_int(path)
        if errors:
            entry["errors"] = errors
            # One number to alarm on, so a dashboard does not have to know
            # which six counters exist on which firmware.
            entry["errors_total"] = sum(errors.values())

        previous = self._last.get(key)
        if previous is not None and elapsed > 0:
            tx_rate = max(0, entry["tx_bytes"] - previous[0]) / elapsed
            rx_rate = max(0, entry["rx_bytes"] - previous[1]) / elapsed
            entry["tx_mbit_s"] = round(tx_rate * 8 / 1e6, 2)
            entry["rx_mbit_s"] = round(rx_rate * 8 / 1e6, 2)
            link = entry.get("link_gbit")
            if link:
                # Per direction, not summed: a link can be saturated one way
                # and idle the other, and adding them hides exactly that.
                entry["tx_percent"] = round(
                    min(100.0, entry["tx_mbit_s"] / (link * 1000) * 100), 1)
                entry["rx_percent"] = round(
                    min(100.0, entry["rx_mbit_s"] / (link * 1000) * 100), 1)
            if previous[2] is not None and entry.get("errors_total") is not None:
                # Only the delta is actionable: a counter that has been at 3
                # since the machine was built is not a fault happening now.
                entry["errors_new"] = max(0, entry["errors_total"] - previous[2])

        self._last[key] = (entry["tx_bytes"], entry["rx_bytes"],
                           entry.get("errors_total"))
        return entry
