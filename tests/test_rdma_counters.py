"""The ring interfaces read 0 bytes/s while carrying every byte of a launch.

Reported: the mesh interfaces publish tx_mbit_s: 0 and rx_mbit_s: 0 to MQTT,
and never anything else. Measured at the kernel, with MiniMax running
tensor-parallel across two nodes and generating code:

    enp1s0f0np0  rx=1310025393 tx=400473758535   (30 s later: +736 bytes)

So the reading was truthful and the metric was useless. NCCL's own log says
why:

    NCCL INFO Using network IB
    NET/IB : Using [0]rocep1s0f0:1/RoCE ... speed=200000 [RO]

RDMA bypasses the kernel network stack — that is the point of it — so an
all-reduce over RoCE moves nothing /proc/net/dev can see, and psutil reads
/proc/net/dev. The telemetry was at its least informative exactly when the
fabric was busiest.

The RDMA counters live in /sys/class/infiniband/<hca>/ports/<n>/counters/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.metrics.system import (
    RDMA_WORD_BYTES,
    SystemSampler,
    rdma_ports,
)


def _hca(root: Path, device: str, netdev: str, xmit: int, rcv: int) -> Path:
    counters = root / device / "ports" / "1" / "counters"
    counters.mkdir(parents=True)
    (counters / "port_xmit_data").write_text(f"{xmit}\n")
    (counters / "port_rcv_data").write_text(f"{rcv}\n")
    net = root / device / "device" / "net" / netdev
    net.mkdir(parents=True)
    return counters


class TestFindingThePorts:
    def test_a_port_is_keyed_by_its_netdev(self, tmp_path, monkeypatch):
        """So the RDMA figures land on the same interface entry as the
        kernel's. An operator comparing them should not have to know that
        rocep1s0f0 and enp1s0f0np0 are the same cable."""
        _hca(tmp_path, "rocep1s0f0", "enp1s0f0np0", 1, 2)
        monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path)
        assert list(rdma_ports()) == ["enp1s0f0np0"]

    def test_a_device_without_a_netdev_keeps_its_own_name(self, tmp_path, monkeypatch):
        counters = tmp_path / "mlx5_0" / "ports" / "1" / "counters"
        counters.mkdir(parents=True)
        monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path)
        assert list(rdma_ports()) == ["mlx5_0"]

    def test_no_rdma_hardware_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path / "nope")
        assert rdma_ports() == {}


class _Counter:
    def __init__(self, sent=0, recv=0):
        self.bytes_sent = sent
        self.bytes_recv = recv


class _Stat:
    isup = True
    speed = 200000


@pytest.fixture
def fabric(tmp_path, monkeypatch):
    """One RoCE port whose kernel counters never move — the reported shape."""
    _hca(tmp_path, "rocep1s0f0", "enp1s0f0np0", 0, 0)
    monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path)
    monkeypatch.setattr("psutil.net_io_counters",
                        lambda pernic=True: {"enp1s0f0np0": _Counter(100, 100)})
    monkeypatch.setattr("psutil.net_if_stats", lambda: {"enp1s0f0np0": _Stat()})
    return tmp_path / "rocep1s0f0" / "ports" / "1" / "counters"


class TestTheRates:
    def test_words_are_converted_to_bytes(self, fabric):
        """port_xmit_data counts 32-bit words, an IB convention that survives
        into RoCE. Publishing the raw value understates every link by four."""
        (fabric / "port_xmit_data").write_text("1000")
        (fabric / "port_rcv_data").write_text("2000")
        entry = SystemSampler()._network()["enp1s0f0np0"]
        assert entry["rdma_bytes_sent"] == 1000 * RDMA_WORD_BYTES
        assert entry["rdma_bytes_recv"] == 2000 * RDMA_WORD_BYTES

    def test_the_first_sample_reports_no_rate(self, fabric):
        entry = SystemSampler()._network()["enp1s0f0np0"]
        assert "rdma_bytes_sent" in entry
        assert "rdma_tx_mbit_s" not in entry

    def test_traffic_invisible_to_the_kernel_shows_up(self, fabric):
        """The whole point: kernel counters flat, fabric busy."""
        sampler = SystemSampler()
        sampler._network()
        # 25 GB sent in the meantime, as an all-reduce would.
        (fabric / "port_xmit_data").write_text(str(25 * 1024 ** 3 // RDMA_WORD_BYTES))
        sampler._last_net_at -= 10.0          # pretend ten seconds passed
        entry = sampler._network()["enp1s0f0np0"]
        assert entry["tx_mbit_s"] == 0.0      # kernel still sees nothing
        assert entry["rdma_tx_mbit_s"] > 20000
        assert entry["rdma_tx_percent"] > 10

    def test_a_counter_reset_costs_one_sample_not_a_negative_rate(self, fabric):
        sampler = SystemSampler()
        (fabric / "port_xmit_data").write_text("1000000")
        sampler._network()
        (fabric / "port_xmit_data").write_text("5")     # driver reset
        sampler._last_net_at -= 10.0
        assert sampler._network()["enp1s0f0np0"]["rdma_tx_mbit_s"] == 0.0

    def test_an_unreadable_counter_is_skipped_not_zeroed(self, fabric):
        (fabric / "port_xmit_data").write_text("not a number")
        entry = SystemSampler()._network()["enp1s0f0np0"]
        assert "rdma_bytes_sent" not in entry
        # ...and the kernel figures are still reported.
        assert entry["bytes_sent"] == 100

    def test_the_kernel_counters_are_kept(self, fabric):
        """Both numbers are true and they measure different things: a TCP
        transfer moves nothing through the HCA port counters either."""
        entry = SystemSampler()._network()["enp1s0f0np0"]
        assert entry["bytes_sent"] == 100 and entry["bytes_recv"] == 100
        assert "rdma_bytes_sent" in entry


class TestWithoutRdma:
    def test_nothing_is_added(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path / "nope")
        monkeypatch.setattr("psutil.net_io_counters",
                            lambda pernic=True: {"eth0": _Counter(5, 5)})
        monkeypatch.setattr("psutil.net_if_stats", lambda: {"eth0": _Stat()})
        entry = SystemSampler()._network()["eth0"]
        assert not any(k.startswith("rdma_") for k in entry)

    def test_sample_still_works_end_to_end(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ainode.metrics.system.RDMA_ROOT", tmp_path / "nope")
        assert "network" in SystemSampler().sample()
