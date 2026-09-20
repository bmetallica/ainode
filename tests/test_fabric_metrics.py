"""RoCE traffic, read where it actually is.

The dashboard reported zero bytes on the ring interfaces of a cluster whose
fabric was saturated, and the interfaces were not lying: RDMA writes straight
from one HCA into the other's memory, so the kernel's network stack — which
is what /proc/net/dev counts and therefore what system.network reports —
never sees a packet.

The one number everybody gets wrong is in here too: port_xmit_data counts
32-bit WORDS. A figure a quarter of the truth looks plausible enough to build
a dashboard on, which is the worst kind of wrong.
"""

from __future__ import annotations

from ainode.metrics.fabric import WORD_BYTES, FabricSampler


def _hca(root, name="rocep1s0", port="1", **counters):
    directory = root / name / "ports" / port / "counters"
    directory.mkdir(parents=True, exist_ok=True)
    defaults = {"port_xmit_data": 0, "port_rcv_data": 0,
                "port_xmit_packets": 0, "port_rcv_packets": 0}
    defaults.update(counters)
    for key, value in defaults.items():
        (directory / key).write_text(f"{value}\n")
    return directory.parent


class TestReadingTheCounters:
    def test_words_are_converted_to_bytes(self, tmp_path):
        # The trap: port_xmit_data is in 32-bit words, not bytes.
        _hca(tmp_path, port_xmit_data=1000, port_rcv_data=500)
        entry = FabricSampler(tmp_path).sample()["rocep1s0:1"]
        assert entry["tx_bytes"] == 1000 * WORD_BYTES
        assert entry["rx_bytes"] == 500 * WORD_BYTES

    def test_one_entry_per_port(self, tmp_path):
        _hca(tmp_path, name="rocep1s0", port="1")
        _hca(tmp_path, name="rocep1s0", port="2")
        _hca(tmp_path, name="rocep2s0", port="1")
        assert sorted(FabricSampler(tmp_path).sample()) == [
            "rocep1s0:1", "rocep1s0:2", "rocep2s0:1"]

    def test_the_link_state_is_reported(self, tmp_path):
        port = _hca(tmp_path)
        (port / "state").write_text("4: ACTIVE\n")
        (port / "phys_state").write_text("5: LinkUp\n")
        (port / "rate").write_text("400 Gb/sec (4X NDR)\n")
        entry = FabricSampler(tmp_path).sample()["rocep1s0:1"]
        assert entry["state"] == "ACTIVE"
        assert entry["phys_state"] == "LinkUp"
        assert entry["link_gbit"] == 400.0

    def test_a_missing_sysfs_is_silence(self, tmp_path):
        assert FabricSampler(tmp_path / "nope").sample() == {}

    def test_an_unreadable_counter_reads_as_zero_not_a_crash(self, tmp_path):
        port = _hca(tmp_path)
        (port / "counters" / "port_xmit_data").write_text("not a number\n")
        assert FabricSampler(tmp_path).sample()["rocep1s0:1"]["tx_bytes"] == 0


class TestRates:
    def test_the_first_sample_carries_no_rate(self, tmp_path):
        # A rate needs two readings. One invented from a single reading would
        # be a guess presented as a measurement.
        _hca(tmp_path, port_xmit_data=1000)
        entry = FabricSampler(tmp_path).sample()["rocep1s0:1"]
        assert "tx_mbit_s" not in entry

    def test_the_second_sample_does(self, tmp_path, monkeypatch):
        import ainode.metrics.fabric as module

        clock = {"t": 1000.0}
        monkeypatch.setattr(module.time, "time", lambda: clock["t"])
        port = _hca(tmp_path, port_xmit_data=0)
        sampler = FabricSampler(tmp_path)
        sampler.sample()
        clock["t"] += 10
        # 125 MB in ten seconds is 100 Mbit/s.
        (port / "counters" / "port_xmit_data").write_text(
            str(125_000_000 // WORD_BYTES) + "\n")
        entry = sampler.sample()["rocep1s0:1"]
        assert entry["tx_mbit_s"] == 100.0

    def test_the_link_percentage_is_per_direction(self, tmp_path, monkeypatch):
        # A link can be saturated one way and idle the other; adding them
        # hides exactly that.
        import ainode.metrics.fabric as module

        clock = {"t": 1000.0}
        monkeypatch.setattr(module.time, "time", lambda: clock["t"])
        port = _hca(tmp_path)
        (port / "rate").write_text("400 Gb/sec (4X NDR)\n")
        sampler = FabricSampler(tmp_path)
        sampler.sample()
        clock["t"] += 1
        (port / "counters" / "port_xmit_data").write_text(
            str(25_000_000_000 // WORD_BYTES) + "\n")
        entry = sampler.sample()["rocep1s0:1"]
        assert entry["tx_percent"] == 50.0
        assert entry["rx_percent"] == 0.0

    def test_a_counter_that_wrapped_does_not_produce_a_negative_rate(self, tmp_path,
                                                                    monkeypatch):
        import ainode.metrics.fabric as module

        clock = {"t": 1000.0}
        monkeypatch.setattr(module.time, "time", lambda: clock["t"])
        port = _hca(tmp_path, port_xmit_data=1_000_000)
        sampler = FabricSampler(tmp_path)
        sampler.sample()
        clock["t"] += 10
        (port / "counters" / "port_xmit_data").write_text("5\n")
        assert sampler.sample()["rocep1s0:1"]["tx_mbit_s"] == 0.0


class TestErrors:
    """On a switchless ring every link is one cable to one neighbour. A cable
    starting to fail has no redundancy to hide behind, and shows up here long
    before NCCL gives up mid-run."""

    def test_they_are_summed_into_one_number_to_alarm_on(self, tmp_path):
        _hca(tmp_path, link_downed=2, port_rcv_errors=3, symbol_error=1)
        entry = FabricSampler(tmp_path).sample()["rocep1s0:1"]
        assert entry["errors"]["link_downed"] == 2
        assert entry["errors_total"] == 6

    def test_only_the_delta_is_actionable(self, tmp_path, monkeypatch):
        # A counter that has read 3 since the machine was built is not a
        # fault happening now.
        import ainode.metrics.fabric as module

        clock = {"t": 1000.0}
        monkeypatch.setattr(module.time, "time", lambda: clock["t"])
        port = _hca(tmp_path, port_rcv_errors=3)
        sampler = FabricSampler(tmp_path)
        sampler.sample()
        clock["t"] += 10
        assert sampler.sample()["rocep1s0:1"]["errors_new"] == 0
        clock["t"] += 10
        (port / "counters" / "port_rcv_errors").write_text("9\n")
        assert sampler.sample()["rocep1s0:1"]["errors_new"] == 6

    def test_a_firmware_without_a_counter_is_not_a_gap(self, tmp_path):
        _hca(tmp_path)          # none of the error counters exist
        entry = FabricSampler(tmp_path).sample()["rocep1s0:1"]
        assert "errors" not in entry


class TestItReachesTheBroker:
    def test_the_payload_carries_the_ports(self, tmp_path, monkeypatch):
        from ainode.core.config import NodeConfig
        from ainode.telemetry.payloads import build_payloads

        _hca(tmp_path, port_xmit_data=10)

        class _Sampler:
            def sample(self):
                return {}

        app = {"config": NodeConfig(node_id="n1"),
               "_fabric_sampler": FabricSampler(tmp_path)}
        payloads = build_payloads(app, _Sampler())
        assert "fabric" in payloads
        assert payloads["fabric"]["ports"]["rocep1s0:1"]["tx_bytes"] == 40
        assert payloads["fabric"]["node_id"] == "n1"

    def test_a_node_without_rdma_publishes_no_fabric_topic(self, tmp_path):
        # An empty topic every interval from a node with no HCA is noise.
        from ainode.core.config import NodeConfig
        from ainode.telemetry.payloads import build_payloads

        class _Sampler:
            def sample(self):
                return {}

        app = {"config": NodeConfig(node_id="n1"),
               "_fabric_sampler": FabricSampler(tmp_path / "nothing")}
        assert "fabric" not in build_payloads(app, _Sampler())
