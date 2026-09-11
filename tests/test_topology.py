"""Unit tests for ainode.cluster.topology.

No live /sys/class/infiniband and no ``ip`` binary is required: the sysfs
tree is a tmpdir we point ``hca_discovery.SYS_INFINIBAND`` at, and every
address lookup goes through a patched ``subprocess.run``.

The behavioural contract these lock down:
  * 4 active CX7 links  -> MESH, coordination moves to enP7s7/wlP9s9,
    all four RoCE devices go to NCCL, plus the three mesh NCCL vars.
  * 2 active CX7 links  -> DIRECT, and *nothing* changes: the configured
    interface is returned untouched and rdma_hcas stays empty so the
    backends keep their existing subnet-filtered detection.
  * anything else       -> UNKNOWN, treated exactly like DIRECT.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from ainode.cluster import hca_discovery, topology


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_hca(
    root: Path,
    name: str,
    port_state: str = "4: ACTIVE",
    netdev: str | None = None,
) -> None:
    """Create a fake /sys/class/infiniband/<name> tree under ``root``."""
    port = root / name / "ports" / "1"
    port.mkdir(parents=True, exist_ok=True)
    (port / "state").write_text(port_state + "\n")
    net_dir = root / name / "device" / "net"
    net_dir.mkdir(parents=True, exist_ok=True)
    if netdev:
        (net_dir / netdev).mkdir(exist_ok=True)


@pytest.fixture
def fake_sysfs(tmp_path, monkeypatch):
    """Re-point SYS_INFINIBAND at a tmpdir. topology reads it via the
    hca_discovery module attribute, so patching there covers both."""
    root = tmp_path / "sys_infiniband"
    root.mkdir()
    monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", root)
    return root


def _fake_ip(addresses: dict[str, str]):
    """Return a subprocess.run stand-in answering ``ip -o -4 addr show dev X``.

    ``addresses`` maps netdev name -> "addr/prefix". Unlisted devices come
    back with no ``inet`` line, i.e. up but unaddressed.
    """

    def _run(cmd, *args, **kwargs):
        dev = cmd[-1]
        cidr = addresses.get(dev)
        stdout = (
            f"1: {dev}    inet {cidr} scope global {dev}\\       valid_lft forever\n"
            if cidr else ""
        )
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    return _run


def _mesh_sysfs(root: Path) -> None:
    """A 3-node-mesh node: all four RoCE devices up (eugr NETWORKING.md)."""
    _make_hca(root, "rocep1s0f0", netdev="enp1s0f0np0")
    _make_hca(root, "rocep1s0f1", netdev="enp1s0f1np1")
    _make_hca(root, "roceP2p1s0f0", netdev="enP2p1s0f0np0")
    _make_hca(root, "roceP2p1s0f1", netdev="enP2p1s0f1np1")


def _direct_sysfs(root: Path) -> None:
    """A back-to-back pair: one cable, so one twin pair is up, one is down."""
    _make_hca(root, "rocep1s0f0", port_state="1: DOWN", netdev="enp1s0f0np0")
    _make_hca(root, "rocep1s0f1", netdev="enp1s0f1np1")
    _make_hca(root, "roceP2p1s0f0", port_state="1: DOWN", netdev="enP2p1s0f0np0")
    _make_hca(root, "roceP2p1s0f1", netdev="enP2p1s0f1np1")


MESH_ADDRS = {
    "enp1s0f0np0": "192.168.177.11/24",
    "enP2p1s0f0np0": "192.168.178.11/24",
    "enp1s0f1np1": "192.168.187.11/24",
    "enP2p1s0f1np1": "192.168.188.11/24",
    "enP7s7": "10.0.0.11/24",
}


# ---------------------------------------------------------------------------
# detect_cx7_links
# ---------------------------------------------------------------------------


class TestDetectCx7Links:
    def test_empty_without_rdma(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", tmp_path / "nope")
        assert topology.detect_cx7_links() == []

    def test_reports_active_links_with_addresses(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            links = topology.detect_cx7_links()
        assert [link.hca for link in links] == sorted(
            ["rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"]
        )
        by_hca = {link.hca: link for link in links}
        assert by_hca["rocep1s0f0"].netdev == "enp1s0f0np0"
        assert by_hca["rocep1s0f0"].ipv4 == "192.168.177.11"
        assert by_hca["rocep1s0f0"].cidr == "192.168.177.0/24"
        assert by_hca["rocep1s0f0"].has_ipv4

    def test_skips_down_ports(self, fake_sysfs):
        _direct_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip({})):
            links = topology.detect_cx7_links()
        assert [link.hca for link in links] == ["roceP2p1s0f1", "rocep1s0f1"]

    def test_active_defer_counts_as_up(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_0", port_state="5: ACTIVE_DEFER", netdev="eth0")
        with mock.patch("subprocess.run", side_effect=_fake_ip({})):
            assert [link.hca for link in topology.detect_cx7_links()] == ["mlx5_0"]

    def test_link_without_netdev_still_counts(self, fake_sysfs):
        """One twin of a port pair routinely has no IP — it must still be
        counted, or the mesh heuristic reads 2 instead of 4."""
        _make_hca(fake_sysfs, "mlx5_0", netdev=None)
        with mock.patch("subprocess.run", side_effect=_fake_ip({})) as run:
            links = topology.detect_cx7_links()
        assert links == [topology.CX7Link(hca="mlx5_0")]
        assert not links[0].has_ipv4
        run.assert_not_called()  # no netdev -> no address lookup attempted

    def test_unreadable_state_is_skipped(self, fake_sysfs):
        (fake_sysfs / "mlx5_0").mkdir()  # no ports/1/state at all
        _make_hca(fake_sysfs, "mlx5_1", netdev="eth1")
        with mock.patch("subprocess.run", side_effect=_fake_ip({})):
            assert [link.hca for link in topology.detect_cx7_links()] == ["mlx5_1"]

    def test_missing_ip_binary_degrades_to_no_address(self, fake_sysfs):
        _make_hca(fake_sysfs, "mlx5_0", netdev="eth0")
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            links = topology.detect_cx7_links()
        assert links[0].netdev == "eth0"
        assert links[0].ipv4 == ""


# ---------------------------------------------------------------------------
# classify_fabric
# ---------------------------------------------------------------------------


class TestClassifyFabric:
    @pytest.mark.parametrize(
        "count,expected",
        [
            (0, topology.Fabric.UNKNOWN),
            (1, topology.Fabric.UNKNOWN),
            (2, topology.Fabric.DIRECT),
            (3, topology.Fabric.UNKNOWN),
            (4, topology.Fabric.MESH),
            (5, topology.Fabric.UNKNOWN),
        ],
    )
    def test_link_count_maps_to_fabric(self, count, expected):
        links = [topology.CX7Link(hca=f"mlx5_{i}") for i in range(count)]
        assert topology.classify_fabric(links) is expected

    def test_detects_when_links_not_passed(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            assert topology.classify_fabric() is topology.Fabric.MESH


# ---------------------------------------------------------------------------
# coordination_interface
# ---------------------------------------------------------------------------


class TestCoordinationInterface:
    def test_direct_returns_configured_interface_untouched(self):
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)) as run:
            got = topology.coordination_interface(
                topology.Fabric.DIRECT, "enP2p1s0f1np1"
            )
        assert got == "enP2p1s0f1np1"
        run.assert_not_called()  # no probing off the mesh path

    def test_unknown_returns_configured_interface_untouched(self):
        got = topology.coordination_interface(topology.Fabric.UNKNOWN, "eno1")
        assert got == "eno1"

    def test_mesh_prefers_10g_ethernet(self):
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            got = topology.coordination_interface(
                topology.Fabric.MESH, "enP2p1s0f1np1"
            )
        assert got == "enP7s7"

    def test_mesh_falls_back_to_wireless(self):
        addrs = {"wlP9s9": "10.0.0.11/24"}  # enP7s7 unaddressed
        with mock.patch("subprocess.run", side_effect=_fake_ip(addrs)):
            got = topology.coordination_interface(
                topology.Fabric.MESH, "enP2p1s0f1np1"
            )
        assert got == "wlP9s9"

    def test_mesh_without_shared_ethernet_falls_back_to_configured(self):
        with mock.patch("subprocess.run", side_effect=_fake_ip({})):
            got = topology.coordination_interface(
                topology.Fabric.MESH, "enP2p1s0f1np1"
            )
        assert got == "enP2p1s0f1np1"

    def test_override_wins_everywhere(self):
        for fabric in topology.Fabric:
            with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
                got = topology.coordination_interface(
                    fabric, "eno1", override="br0"
                )
            assert got == "br0"


# ---------------------------------------------------------------------------
# detect_topology — the composed result
# ---------------------------------------------------------------------------


class TestDetectTopology:
    def test_mesh_moves_coordination_and_takes_all_hcas(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            info = topology.detect_topology("enP2p1s0f1np1")
        assert info.fabric is topology.Fabric.MESH
        assert info.is_mesh
        assert info.coord_interface == "enP7s7"
        assert info.rdma_hcas == sorted(
            ["rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"]
        )
        assert info.nccl_env() == {
            "NCCL_NET_PLUGIN": "none",
            "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
            "NCCL_IB_MERGE_NICS": "0",
        }

    def test_direct_changes_nothing(self, fake_sysfs):
        """The load-bearing regression guard: an existing 2-/4-node TP setup
        must come back with its configured interface and no HCA opinion."""
        _direct_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            info = topology.detect_topology("enP2p1s0f1np1")
        assert info.fabric is topology.Fabric.DIRECT
        assert not info.is_mesh
        assert info.coord_interface == "enP2p1s0f1np1"
        assert info.rdma_hcas == []
        assert info.nccl_env() == {}

    def test_no_rdma_host_is_unknown_and_inert(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", tmp_path / "nope")
        info = topology.detect_topology("eno1")
        assert info.fabric is topology.Fabric.UNKNOWN
        assert info.coord_interface == "eno1"
        assert info.rdma_hcas == []
        assert info.nccl_env() == {}

    def test_hca_override_wins_on_mesh(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            info = topology.detect_topology(
                "enP2p1s0f1np1", hca_override=["mlx5_1", "mlx5_3"]
            )
        assert info.rdma_hcas == ["mlx5_1", "mlx5_3"]
        assert info.is_mesh  # override the devices, not the classification

    def test_hca_override_applies_off_mesh_too(self, fake_sysfs):
        _direct_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            info = topology.detect_topology("eno1", hca_override=["mlx5_0"])
        assert info.fabric is topology.Fabric.DIRECT
        assert info.rdma_hcas == ["mlx5_0"]

    def test_coord_override_applies(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            info = topology.detect_topology("eno1", coord_override="bond0")
        assert info.coord_interface == "bond0"

    def test_to_dict_round_trips_for_logging(self, fake_sysfs):
        _mesh_sysfs(fake_sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip(MESH_ADDRS)):
            payload = topology.detect_topology("enP2p1s0f1np1").to_dict()
        assert payload["fabric"] == "mesh"
        assert payload["coord_interface"] == "enP7s7"
        assert len(payload["links"]) == 4
        assert payload["links"][0].keys() == {"hca", "netdev", "ipv4", "cidr"}
