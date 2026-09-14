"""Host metrics: CPU, memory, disk, network, temperature.

The collector already knew about the GPU and about requests. Everything else a
person actually watches on a node — is the CPU pinned, is the disk filling up,
is that transfer using the 100G link or the 10G one — was not collected
anywhere, so it could be neither shown nor published.

Rates need two samples, which is why this is a class and not a function: the
first call establishes a baseline and reports no rate, every later call reports
the average over the interval since the previous one. That matches how it is
used — one sampler, polled on a timer.

Everything is best effort. A field that cannot be read is absent rather than
zero: a missing value and a genuine zero mean different things to whoever reads
the dashboard, and psutil's coverage varies by platform and container.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["SystemSampler", "interface_is_interesting"]

#: Interface name prefixes that are never worth reporting: the loopback, and
#: the churn of container networking. A Spark's real links (enP7s7, wlP9s9,
#: enP2p1s0f…) do not collide with these.
_SKIP_PREFIXES = ("lo", "veth", "docker", "br-", "virbr", "tun", "tap")


def interface_is_interesting(name: str) -> bool:
    """False for loopback and container plumbing."""
    return not any(name == p or name.startswith(p) for p in _SKIP_PREFIXES)


#: Where the kernel exposes RDMA port counters.
RDMA_ROOT = Path("/sys/class/infiniband")

#: port_xmit_data and port_rcv_data count 32-bit words, not bytes — an IB
#: convention that survives into RoCE. Reporting the raw value would understate
#: every link by exactly four.
RDMA_WORD_BYTES = 4


def rdma_ports() -> Dict[str, Path]:
    """``{netdev name or device name: counters directory}`` for every RoCE port.

    RDMA bypasses the kernel network stack — that is the point of it — so an
    all-reduce over RoCE moves nothing that /proc/net/dev can see. On this
    cluster that made the ring interfaces read 0 bytes per second while they
    were carrying every byte of a two-node tensor-parallel launch: the
    telemetry was at its least informative exactly when the fabric was
    busiest.

    Keyed by the netdev name where the mapping exists, so the RDMA figures
    land on the same interface entry as the kernel's — an operator comparing
    them should not have to know that rocep1s0f0 and enp1s0f0np0 are the same
    cable.
    """
    found: Dict[str, Path] = {}
    try:
        devices = sorted(RDMA_ROOT.iterdir())
    except OSError:
        return found
    for device in devices:
        try:
            ports = sorted((device / "ports").iterdir())
        except OSError:
            continue
        # The netdev this HCA port belongs to, when the kernel says so.
        name = device.name
        try:
            nets = sorted((device / "device" / "net").iterdir())
            if nets:
                name = nets[0].name
        except OSError:
            pass
        for port in ports:
            counters = port / "counters"
            if counters.is_dir():
                # One port per device is the shape on this hardware; a second
                # would overwrite, which is better than inventing a key that
                # matches no interface anywhere else in the payload.
                found.setdefault(name, counters)
    return found


def _read_counter(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _mb(value: float) -> float:
    return round(value / (1024 * 1024), 1)


def _gb(value: float) -> float:
    return round(value / (1024 * 1024 * 1024), 2)


class SystemSampler:
    """Point-in-time host metrics, with network and CPU rates between calls."""

    def __init__(self, disk_paths: Optional[list] = None):
        #: Where to report free space. The models directory is the one that
        #: actually fills up — a 400 GB checkpoint set is not unusual — and it
        #: is often not on the root filesystem.
        self._disk_paths = disk_paths or []
        self._last_net: Dict[str, Any] = {}
        self._last_net_at: float = 0.0
        self._last_rdma: Dict[str, Any] = {}
        self._primed = False

    # -- public ---------------------------------------------------------

    def sample(self) -> Dict[str, Any]:
        """Everything readable right now, as a JSON-shaped dict."""
        out: Dict[str, Any] = {"timestamp": round(time.time(), 3)}
        for name, fn in (
            ("cpu", self._cpu),
            ("memory", self._memory),
            ("disk", self._disk),
            ("network", self._network),
            ("temperature_c", self._temperatures),
            ("uptime_seconds", self._uptime),
        ):
            try:
                value = fn()
            except Exception:
                logger.debug("system metric %s unavailable", name, exc_info=True)
                continue
            if value not in (None, {}, []):
                out[name] = value
        self._primed = True
        return out

    # -- pieces ---------------------------------------------------------

    def _cpu(self) -> Dict[str, Any]:
        import psutil

        # interval=None compares against the previous call. The first call
        # after start therefore has nothing to compare with and returns 0.0 —
        # reported as absent rather than as an idle CPU.
        percent = psutil.cpu_percent(interval=None)
        cpu: Dict[str, Any] = {"cores": psutil.cpu_count(logical=True)}
        if self._primed:
            cpu["percent"] = round(percent, 1)
        try:
            one, five, fifteen = os.getloadavg()
            cpu["load_1m"] = round(one, 2)
            cpu["load_5m"] = round(five, 2)
            cpu["load_15m"] = round(fifteen, 2)
        except (OSError, AttributeError):
            pass
        return cpu

    def _memory(self) -> Dict[str, Any]:
        import psutil

        vm = psutil.virtual_memory()
        mem = {
            "total_mb": _mb(vm.total),
            "used_mb": _mb(vm.used),
            "available_mb": _mb(vm.available),
            "percent": round(vm.percent, 1),
        }
        try:
            sw = psutil.swap_memory()
            if sw.total:
                mem["swap_used_mb"] = _mb(sw.used)
                mem["swap_total_mb"] = _mb(sw.total)
        except Exception:
            pass
        return mem

    def _disk(self) -> Dict[str, Any]:
        import psutil

        out: Dict[str, Any] = {}
        seen = set()
        for path in ["/", *self._disk_paths]:
            try:
                real = str(Path(path).resolve())
                usage = psutil.disk_usage(real)
            except Exception:
                continue
            # Two paths on the same filesystem would otherwise be published
            # twice under different names, which reads as two disks.
            key = (usage.total, usage.free)
            if key in seen:
                continue
            seen.add(key)
            out[path] = {
                "total_gb": _gb(usage.total),
                "free_gb": _gb(usage.free),
                "percent": round(usage.percent, 1),
            }
        return out

    def _network(self) -> Dict[str, Any]:
        import psutil

        counters = psutil.net_io_counters(pernic=True)
        try:
            stats = psutil.net_if_stats()
        except Exception:
            stats = {}

        now = time.time()
        elapsed = now - self._last_net_at if self._last_net_at else 0.0
        out: Dict[str, Any] = {}

        for name, counter in counters.items():
            if not interface_is_interesting(name):
                continue
            nic_stat = stats.get(name)
            if nic_stat is not None and not nic_stat.isup:
                continue
            entry: Dict[str, Any] = {
                "bytes_sent": counter.bytes_sent,
                "bytes_recv": counter.bytes_recv,
            }
            speed = getattr(nic_stat, "speed", 0) or 0
            if speed:
                entry["link_mbit"] = speed

            previous = self._last_net.get(name)
            if previous is not None and elapsed > 0:
                sent_rate = max(0, counter.bytes_sent - previous[0]) / elapsed
                recv_rate = max(0, counter.bytes_recv - previous[1]) / elapsed
                entry["tx_mbit_s"] = round(sent_rate * 8 / 1e6, 2)
                entry["rx_mbit_s"] = round(recv_rate * 8 / 1e6, 2)
                if speed:
                    # Percent of the link, per direction. Full duplex, so the
                    # two are not added: a link can be saturated one way and
                    # idle the other, and summing them hides that.
                    entry["tx_percent"] = round(
                        min(100.0, entry["tx_mbit_s"] / speed * 100), 1)
                    entry["rx_percent"] = round(
                        min(100.0, entry["rx_mbit_s"] / speed * 100), 1)
            out[name] = entry

        self._last_net = {
            name: (c.bytes_sent, c.bytes_recv) for name, c in counters.items()
        }
        self._merge_rdma(out, elapsed, speeds=stats)
        self._last_net_at = now
        return out

    def _merge_rdma(self, out: Dict[str, Any], elapsed: float, speeds) -> None:
        """Add RDMA byte counters and rates to the interfaces that have them.

        Alongside the kernel's, not instead of them. Both numbers are true and
        they measure different things: an RDMA transfer moves nothing through
        /proc/net/dev, and a TCP transfer moves nothing through the HCA port
        counters. Showing only one made a busy 200 Gbit fabric read as idle.
        """
        readings: Dict[str, tuple] = {}
        for name, counters in rdma_ports().items():
            sent = _read_counter(counters / "port_xmit_data")
            recv = _read_counter(counters / "port_rcv_data")
            if sent is None or recv is None:
                continue
            sent *= RDMA_WORD_BYTES
            recv *= RDMA_WORD_BYTES
            readings[name] = (sent, recv)

            entry = out.setdefault(name, {})
            entry["rdma_bytes_sent"] = sent
            entry["rdma_bytes_recv"] = recv
            previous = self._last_rdma.get(name)
            if previous is None or elapsed <= 0:
                continue
            # Port counters are 64-bit here but can be reset by the driver;
            # max(0, ...) turns a wrap or reset into one missed sample rather
            # than into a negative rate or an implausible spike.
            tx = round(max(0, sent - previous[0]) / elapsed * 8 / 1e6, 2)
            rx = round(max(0, recv - previous[1]) / elapsed * 8 / 1e6, 2)
            entry["rdma_tx_mbit_s"] = tx
            entry["rdma_rx_mbit_s"] = rx
            speed = getattr(speeds.get(name), "speed", 0) or 0
            if speed:
                entry.setdefault("link_mbit", speed)
                entry["rdma_tx_percent"] = round(min(100.0, tx / speed * 100), 1)
                entry["rdma_rx_percent"] = round(min(100.0, rx / speed * 100), 1)
        self._last_rdma = readings

    def _temperatures(self) -> Dict[str, Any]:
        import psutil

        try:
            readings = psutil.sensors_temperatures()
        except (AttributeError, OSError):
            return {}
        out: Dict[str, Any] = {}
        for chip, entries in (readings or {}).items():
            for entry in entries:
                if entry.current is None:
                    continue
                label = (entry.label or chip).strip().replace(" ", "_").lower()
                out[label] = round(entry.current, 1)
        return out

    def _uptime(self) -> Optional[float]:
        import psutil

        return round(time.time() - psutil.boot_time(), 1)
