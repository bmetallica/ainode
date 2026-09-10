"""How the engine backends behave on each fabric topology.

The existing backend tests run on a host with no ``/sys/class/infiniband``,
which lands them in :attr:`Fabric.UNKNOWN`. That proves the no-RDMA path is
unchanged but says nothing about a real 2-node Spark pair, so these tests
stand up a fake sysfs for each wiring and assert:

  * **DIRECT** (2 active CX7 links) — every value the backends emit is
    identical to the pre-mesh behaviour. This is the load-bearing guard for
    "existing 2- and 4-node TP setups must keep running unchanged".
  * **MESH** (4 active links) — coordination moves to the shared Ethernet,
    NCCL gets all four RoCE devices, and the three mesh NCCL vars appear.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from ainode.cluster import hca_discovery, topology
from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend, EugrBackendError
from ainode.engine.backends.nvidia import NvidiaBackend


CLUSTER_IF = "enP2p1s0f1np1"
COORD_IF = "enP7s7"

# Addresses as the mesh netplan in eugr's docs/NETWORKING.md lays them out for
# spark1, plus the shared 10G port.
ADDRS = {
    "enp1s0f0np0": "192.168.177.11/24",
    "enP2p1s0f0np0": "192.168.178.11/24",
    "enp1s0f1np1": "192.168.187.11/24",
    CLUSTER_IF: "192.168.188.11/24",
    COORD_IF: "10.0.0.11/24",
}

MESH_HCAS = ["roceP2p1s0f0", "roceP2p1s0f1", "rocep1s0f0", "rocep1s0f1"]


# ---------------------------------------------------------------------------
# Fake sysfs + ip
# ---------------------------------------------------------------------------


def _hca(root: Path, name: str, netdev: str, state: str = "4: ACTIVE") -> None:
    port = root / name / "ports" / "1"
    port.mkdir(parents=True, exist_ok=True)
    (port / "state").write_text(state + "\n")
    net = root / name / "device" / "net"
    net.mkdir(parents=True, exist_ok=True)
    (net / netdev).mkdir(exist_ok=True)


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    root = tmp_path / "sys_infiniband"
    root.mkdir()
    monkeypatch.setattr(hca_discovery, "SYS_INFINIBAND", root)
    return root


def _wire_mesh(root: Path) -> None:
    """All four links up — 3-node mesh, both QSFP ports cabled."""
    _hca(root, "rocep1s0f0", "enp1s0f0np0")
    _hca(root, "roceP2p1s0f0", "enP2p1s0f0np0")
    _hca(root, "rocep1s0f1", "enp1s0f1np1")
    _hca(root, "roceP2p1s0f1", CLUSTER_IF)


def _wire_direct(root: Path) -> None:
    """One cable — the twin pair on the outer port is up, the other is down."""
    _hca(root, "rocep1s0f0", "enp1s0f0np0", state="1: DOWN")
    _hca(root, "roceP2p1s0f0", "enP2p1s0f0np0", state="1: DOWN")
    _hca(root, "rocep1s0f1", "enp1s0f1np1")
    _hca(root, "roceP2p1s0f1", CLUSTER_IF)


def _fake_ip(cmd, *args, **kwargs):
    """Answer ``ip -o -4 addr show dev X`` from the ADDRS table."""
    dev = cmd[-1]
    cidr = ADDRS.get(dev)
    stdout = f"1: {dev}    inet {cidr} scope global {dev}\n" if cidr else ""
    return mock.Mock(returncode=0, stdout=stdout, stderr="")


def _config(**overrides) -> NodeConfig:
    defaults = dict(
        cluster_interface=CLUSTER_IF,
        model="meta-llama/Llama-3.2-3B-Instruct",
        distributed_mode="head",
        peer_ips=["10.0.0.12", "10.0.0.13"],
        ssh_user="ubuntu",
        models_dir="/tmp/ainode-models",
    )
    defaults.update(overrides)
    return NodeConfig(**defaults)


# ---------------------------------------------------------------------------
# NvidiaBackend
# ---------------------------------------------------------------------------


class TestNvidiaBackendOnFabric:
    def _env(self, config) -> dict:
        backend = NvidiaBackend(config)
        with mock.patch("subprocess.run", side_effect=_fake_ip), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="roceP2p1s0f1",
        ):
            return backend._build_nccl_env(is_head=True)

    def test_direct_keeps_cluster_interface_and_local_detection(self, sysfs):
        _wire_direct(sysfs)
        env = self._env(_config())
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                    "UCX_NET_DEVICES", "TP_SOCKET_IFNAME",
                    "OMPI_MCA_btl_tcp_if_include"):
            assert env[key] == CLUSTER_IF
        assert env["VLLM_HOST_IP"] == "192.168.188.11"
        assert env["MASTER_ADDR"] == "192.168.188.11"
        # Falls through to the GID-filtered local whitelist, as before.
        assert env["NCCL_IB_HCA"] == "roceP2p1s0f1"
        # No mesh vars leak into a direct-attach cluster.
        assert "NCCL_NET_PLUGIN" not in env
        assert "NCCL_IB_MERGE_NICS" not in env

    def test_mesh_moves_sockets_to_shared_ethernet(self, sysfs):
        _wire_mesh(sysfs)
        env = self._env(_config())
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                    "UCX_NET_DEVICES", "TP_SOCKET_IFNAME",
                    "OMPI_MCA_btl_tcp_if_include"):
            assert env[key] == COORD_IF
        # Ray + torch rendezvous on the address every node can reach.
        assert env["VLLM_HOST_IP"] == "10.0.0.11"
        assert env["MASTER_ADDR"] == "10.0.0.11"

    def test_mesh_gives_nccl_every_roce_device(self, sysfs):
        _wire_mesh(sysfs)
        env = self._env(_config())
        assert env["NCCL_IB_HCA"] == ",".join(MESH_HCAS)

    def test_mesh_sets_the_three_mesh_nccl_vars(self, sysfs):
        _wire_mesh(sysfs)
        env = self._env(_config())
        assert env["NCCL_NET_PLUGIN"] == "none"
        assert env["NCCL_IB_MERGE_NICS"] == "0"
        assert env["NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"
        assert env["NCCL_IB_DISABLE"] == "0"

    def test_head_fabric_ip_follows_coordination_interface(self, sysfs):
        _wire_mesh(sysfs)
        backend = NvidiaBackend(_config())
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            assert backend._head_fabric_ip() == "10.0.0.11"

    def test_direct_head_fabric_ip_unchanged(self, sysfs):
        _wire_direct(sysfs)
        backend = NvidiaBackend(_config())
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            assert backend._head_fabric_ip() == "192.168.188.11"

    def test_explicit_overrides_beat_detection(self, sysfs):
        _wire_mesh(sysfs)
        env = self._env(_config(coord_interface="enp1s0f1np1",
                                rdma_hcas=["mlx5_0"]))
        assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f1np1"
        assert env["NCCL_IB_HCA"] == "mlx5_0"

    def test_topology_detected_once_per_instance(self, sysfs):
        _wire_mesh(sysfs)
        backend = NvidiaBackend(_config())
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            first = backend._topology()
            second = backend._topology()
        assert first is second


# ---------------------------------------------------------------------------
# EugrBackend — the default backend, so this is the path most installs take
# ---------------------------------------------------------------------------


class TestEugrBackendOnFabric:
    def _env(self, config) -> dict:
        backend = EugrBackend(config)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            return backend._build_env()

    def test_direct_keeps_cluster_interface(self, sysfs):
        _wire_direct(sysfs)
        env = self._env(_config())
        assert env["NCCL_SOCKET_IFNAME"] == CLUSTER_IF
        assert env["GLOO_SOCKET_IFNAME"] == CLUSTER_IF
        assert env["UCX_NET_DEVICES"] == CLUSTER_IF
        assert "NCCL_NET_PLUGIN" not in env
        assert "NCCL_IB_MERGE_NICS" not in env

    def test_direct_hca_stays_subnet_filtered(self, sysfs):
        """Only the HCA on the cluster subnet — the dual-homed-node fix
        (bugs 1/2/4) must survive."""
        _wire_direct(sysfs)
        env = self._env(_config())
        assert env["NCCL_IB_HCA"] == "roceP2p1s0f1"

    def test_mesh_moves_sockets_and_takes_all_hcas(self, sysfs):
        _wire_mesh(sysfs)
        env = self._env(_config())
        assert env["NCCL_SOCKET_IFNAME"] == COORD_IF
        assert env["GLOO_SOCKET_IFNAME"] == COORD_IF
        assert env["UCX_NET_DEVICES"] == COORD_IF
        assert env["NCCL_IB_HCA"] == ",".join(MESH_HCAS)
        assert env["NCCL_NET_PLUGIN"] == "none"
        assert env["NCCL_IB_MERGE_NICS"] == "0"
        assert env["NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"


class TestEugrLauncherEnvFile:
    """``/opt/spark-vllm-docker/.env`` — what the launcher actually reads."""

    def _write(self, config, tmp_path, monkeypatch) -> dict[str, str]:
        env_file = tmp_path / "eugr.env"
        monkeypatch.setattr(
            "ainode.engine.backends.eugr.EUGR_ENV_FILE", env_file
        )
        backend = EugrBackend(config)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            backend._write_eugr_env()
        return dict(
            line.split("=", 1)
            for line in env_file.read_text().splitlines()
            if "=" in line
        )

    def test_direct_env_file_unchanged(self, sysfs, tmp_path, monkeypatch):
        _wire_direct(sysfs)
        env = self._write(_config(), tmp_path, monkeypatch)
        assert env["ETH_IF"] == CLUSTER_IF
        assert env["IB_IF"] == "roceP2p1s0f1"
        assert env["CONTAINER_NCCL_SOCKET_IFNAME"] == CLUSTER_IF
        assert env["CLUSTER_NODES"] == "192.168.188.11,10.0.0.12,10.0.0.13"
        # Subnet filter still handed to the per-node shim.
        assert env["CONTAINER_AINODE_CLUSTER_SUBNET"] == "192.168.188.0/24"
        assert "CONTAINER_NCCL_NET_PLUGIN" not in env

    def test_mesh_env_file_carries_mesh_settings(self, sysfs, tmp_path, monkeypatch):
        _wire_mesh(sysfs)
        env = self._write(_config(), tmp_path, monkeypatch)
        assert env["ETH_IF"] == COORD_IF
        assert env["IB_IF"] == ",".join(MESH_HCAS)
        assert env["CONTAINER_NCCL_SOCKET_IFNAME"] == COORD_IF
        assert env["CLUSTER_NODES"] == "10.0.0.11,10.0.0.12,10.0.0.13"
        assert env["CONTAINER_NCCL_NET_PLUGIN"] == "none"
        assert env["CONTAINER_NCCL_IB_MERGE_NICS"] == "0"
        assert env["CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING"] == "1"

    def test_mesh_disables_the_shim_subnet_filter(self, sysfs, tmp_path, monkeypatch):
        """Each mesh device is on its own subnet, so any filter strips the
        ring down to a single cable."""
        _wire_mesh(sysfs)
        env = self._write(_config(), tmp_path, monkeypatch)
        assert env["CONTAINER_AINODE_CLUSTER_SUBNET"] == ""

    def test_refuses_to_write_an_empty_ib_if(self, sysfs, tmp_path, monkeypatch):
        """An empty IB_IF drops launch-cluster.sh into its own autodiscovery,
        which needs ibdev2netdev — absent from the AINode image. Failing here
        with a readable message beats failing there with a confusing one."""
        _wire_direct(sysfs)
        # The real trap from the Phase-1 analysis: cluster_interface pointed at
        # the 10G port on a node that does not classify as a mesh. No RoCE HCA
        # sits on 10.0.0.0/24, so the subnet filter yields nothing.
        config = _config(cluster_interface=COORD_IF)
        with pytest.raises(EugrBackendError, match="ibdev2netdev"):
            self._write(config, tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Announcement — what peers learn about this node
# ---------------------------------------------------------------------------


class TestAnnouncedAddress:
    def _fabric_ip(self, config, sysfs_wired) -> str:
        from ainode.api.server import _build_announcement

        with mock.patch("subprocess.run", side_effect=_fake_ip), mock.patch(
            "ainode.api.server.detect_gpu", return_value=None
        ):
            return _build_announcement(config).fabric_ip

    def test_direct_announces_cluster_interface_address(self, sysfs):
        _wire_direct(sysfs)
        assert self._fabric_ip(_config(), sysfs) == "192.168.188.11"

    def test_mesh_announces_shared_ethernet_address(self, sysfs):
        """A mesh node's CX7 addresses each reach one neighbour only, so
        announcing one would leave the third node unable to reach us."""
        _wire_mesh(sysfs)
        assert self._fabric_ip(_config(), sysfs) == "10.0.0.11"


# ---------------------------------------------------------------------------
# The heuristic itself, end to end
# ---------------------------------------------------------------------------


class TestFabricClassification:
    def test_two_links_is_direct(self, sysfs):
        _wire_direct(sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            assert topology.detect_topology(CLUSTER_IF).fabric is topology.Fabric.DIRECT

    def test_four_links_is_mesh(self, sysfs):
        _wire_mesh(sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            assert topology.detect_topology(CLUSTER_IF).fabric is topology.Fabric.MESH
