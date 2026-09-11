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


# ---------------------------------------------------------------------------
# `ainode doctor` fabric section — the operator's check command
# ---------------------------------------------------------------------------


class TestDoctorFabricReport:
    def _report(self, sysfs_wired, config, monkeypatch):
        from ainode.cli import doctor

        monkeypatch.setattr(
            "ainode.core.config.NodeConfig.load", classmethod(lambda cls: config)
        )
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            return doctor.fabric_report()

    def test_mesh_reports_ok_with_the_derived_settings(self, sysfs, monkeypatch):
        _wire_mesh(sysfs)
        verdict, rows, warnings = self._report(sysfs, _config(), monkeypatch)
        fields = dict(rows)
        assert verdict == "ok"
        assert fields["Fabric"] == "mesh"
        assert fields["Active CX7 links"] == "4"
        assert fields["Coordination iface"] == COORD_IF
        assert fields["Coordination IP"] == "10.0.0.11"
        assert fields["NCCL_IB_HCA"] == ",".join(MESH_HCAS)
        assert fields["NCCL_NET_PLUGIN"] == "none"
        assert warnings == []

    def test_direct_reports_ok_and_defers_hca_detection(self, sysfs, monkeypatch):
        _wire_direct(sysfs)
        verdict, rows, warnings = self._report(sysfs, _config(), monkeypatch)
        fields = dict(rows)
        assert verdict == "ok"
        assert fields["Fabric"] == "direct"
        assert fields["Coordination iface"] == CLUSTER_IF
        assert fields["NCCL_IB_HCA"] == "<local autodetect>"
        assert warnings == []

    def test_odd_link_count_warns_but_stays_harmless(self, sysfs, monkeypatch):
        _hca(sysfs, "rocep1s0f1", "enp1s0f1np1")
        _hca(sysfs, "roceP2p1s0f1", CLUSTER_IF)
        _hca(sysfs, "rocep1s0f0", "enp1s0f0np0")
        verdict, rows, warnings = self._report(sysfs, _config(), monkeypatch)
        assert verdict == "unknown"
        assert dict(rows)["Fabric"] == "unknown"
        assert any("expected 2" in w for w in warnings)

    def test_missing_coordination_address_is_flagged(self, sysfs, monkeypatch):
        _wire_direct(sysfs)
        config = _config(cluster_interface="enX-unplugged")
        verdict, _rows, warnings = self._report(sysfs, config, monkeypatch)
        assert verdict == "warn"
        assert any("cannot join a cluster" in w for w in warnings)


# ---------------------------------------------------------------------------
# Part B — second address list for bulk transfer
# ---------------------------------------------------------------------------

# The mesh from eugr's docs/NETWORKING.md, seen from spark1 (.11):
#   spark1 <-> spark2 on 192.168.177/178
#   spark1 <-> spark3 on 192.168.187/188
#   spark2 <-> spark3 on 192.168.197/198  (no cable to us)
SPARK2_IB = ["192.168.177.12", "192.168.178.12", "192.168.197.12", "192.168.198.12"]
SPARK3_IB = ["192.168.187.13", "192.168.188.13", "192.168.197.13", "192.168.198.13"]
SPARK4_IB = ["192.168.207.14", "192.168.208.14"]  # a node we have no cable to


class TestTransferAddress:
    def _links(self, sysfs):
        _wire_mesh(sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            return topology.detect_cx7_links()

    def test_picks_the_peer_address_on_our_own_subnet(self, sysfs):
        links = self._links(sysfs)
        assert topology.transfer_address(SPARK2_IB, "10.0.0.12", links) == "192.168.177.12"
        assert topology.transfer_address(SPARK3_IB, "10.0.0.13", links) == "192.168.187.13"

    def test_falls_back_when_no_cable_to_the_peer(self, sysfs):
        """Only a fully connected mesh has a link between every pair. Three
        nodes in a ring do; a fourth would not."""
        links = self._links(sysfs)
        assert topology.transfer_address(SPARK4_IB, "10.0.0.14", links) == "10.0.0.14"

    def test_peer_without_roce_addresses_uses_coordination_path(self, sysfs):
        """An older peer announces no ib_ips — it must keep working."""
        links = self._links(sysfs)
        assert topology.transfer_address([], "10.0.0.12", links) == "10.0.0.12"
        assert topology.transfer_address(None, "10.0.0.12", links) == "10.0.0.12"

    def test_no_local_links_uses_coordination_path(self):
        assert topology.transfer_address(SPARK2_IB, "10.0.0.12", []) == "10.0.0.12"

    def test_malformed_addresses_are_skipped_not_fatal(self, sysfs):
        links = self._links(sysfs)
        got = topology.transfer_address(
            ["not-an-ip", "192.168.177.12"], "10.0.0.12", links
        )
        assert got == "192.168.177.12"

    def test_selection_is_deterministic(self, sysfs):
        """Two of our subnets can match (177 and 178 both reach spark2);
        the same one must win every time or a re-launch churns the cache."""
        links = self._links(sysfs)
        first = topology.transfer_address(SPARK2_IB, "10.0.0.12", links)
        assert all(
            topology.transfer_address(list(reversed(SPARK2_IB)), "10.0.0.12", links)
            == first
            for _ in range(3)
        )

    def test_direct_attach_peer_matches_on_the_shared_subnet(self, sysfs):
        """Off a mesh this is not a no-op: a 2-node pair shares the CX7 subnet,
        so the peer's RoCE address is already the direct one."""
        _wire_direct(sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            links = topology.detect_cx7_links()
        got = topology.transfer_address(["192.168.188.12"], "192.168.188.12", links)
        assert got == "192.168.188.12"


class TestLocalIbIps:
    def test_reports_addressed_links_only(self, sysfs):
        _wire_mesh(sysfs)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            got = topology.local_ib_ips()
        assert sorted(got) == [
            "192.168.177.11", "192.168.178.11", "192.168.187.11", "192.168.188.11",
        ]

    def test_unaddressed_link_is_not_a_transfer_target(self, sysfs):
        _hca(sysfs, "mlx5_0", "eth-no-addr")
        _hca(sysfs, "mlx5_1", CLUSTER_IF)
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            assert topology.local_ib_ips() == ["192.168.188.11"]


class TestAnnouncedIbIps:
    def _announce(self, config):
        from ainode.api.server import _build_announcement

        with mock.patch("subprocess.run", side_effect=_fake_ip), mock.patch(
            "ainode.api.server.detect_gpu", return_value=None
        ):
            return _build_announcement(config)

    def test_mesh_node_announces_all_four_link_addresses(self, sysfs):
        _wire_mesh(sysfs)
        ann = self._announce(_config())
        assert sorted(ann.ib_ips) == [
            "192.168.177.11", "192.168.178.11", "192.168.187.11", "192.168.188.11",
        ]
        # Coordination address stays separate and is NOT one of them.
        assert ann.fabric_ip == "10.0.0.11"
        assert ann.fabric_ip not in ann.ib_ips

    def test_survives_a_round_trip_through_the_wire(self, sysfs):
        from ainode.discovery.broadcast import NodeAnnouncement

        _wire_mesh(sysfs)
        ann = self._announce(_config())
        back = NodeAnnouncement.from_json(ann.to_json())
        assert back.ib_ips == ann.ib_ips

    def test_older_peer_without_the_field_still_parses(self):
        """from_json drops unknown keys and defaults missing ones — a node
        running the previous build must not break discovery."""
        import json

        from ainode.discovery.broadcast import NodeAnnouncement

        payload = json.loads(
            NodeAnnouncement(
                node_id="n1", node_name="spark1", gpu_name="GB10",
                gpu_memory_gb=128.0, unified_memory=True, model="", status="online",
                api_port=8000, web_port=3000,
            ).to_json()
        )
        payload.pop("ib_ips")
        back = NodeAnnouncement.from_json(json.dumps(payload))
        assert back.ib_ips == []

    def test_cluster_node_carries_it_through(self, sysfs):
        from ainode.discovery.broadcast import NodeStatus
        from ainode.discovery.cluster import ClusterNode

        _wire_mesh(sysfs)
        ann = self._announce(_config())
        node = ClusterNode.from_announcement(ann, NodeStatus.ONLINE)
        assert sorted(node.ib_ips) == sorted(ann.ib_ips)


class TestBackendTransferIp:
    def test_uses_the_direct_address_when_the_launch_found_one(self):
        config = _config(peer_transfer_ips={"10.0.0.12": "192.168.177.12"})
        backend = NvidiaBackend(config)
        assert backend._transfer_ip("10.0.0.12") == "192.168.177.12"

    def test_falls_back_to_the_coordination_address(self):
        backend = NvidiaBackend(_config())
        assert backend._transfer_ip("10.0.0.13") == "10.0.0.13"

    def test_peer_not_in_the_map_falls_back(self):
        config = _config(peer_transfer_ips={"10.0.0.12": "192.168.177.12"})
        backend = NvidiaBackend(config)
        assert backend._transfer_ip("10.0.0.13") == "10.0.0.13"

    def test_ray_and_ssh_still_use_the_coordination_address(self, sysfs):
        """Only bulk transfer moves. Coordination must not follow it, or the
        third mesh node becomes unreachable."""
        _wire_mesh(sysfs)
        config = _config(peer_transfer_ips={"10.0.0.12": "192.168.177.12"})
        backend = NvidiaBackend(config)
        with mock.patch("subprocess.run", side_effect=_fake_ip), mock.patch(
            "ainode.engine.backends.nvidia.build_nccl_ib_hca_whitelist",
            return_value="",
        ):
            cmd = backend._build_ray_docker_cmd(
                container_name="ainode-vllm-worker-x", role="worker",
                head_ip="10.0.0.11", node_ip="10.0.0.12", hf_cache_dir="/tmp/c",
            )
        joined = " ".join(cmd)
        assert "--address=10.0.0.11:6379" in joined
        assert "192.168.177.12" not in joined

    def test_link_cidr_with_host_bits_still_matches(self, sysfs):
        """A CX7Link built by hand may carry "10.0.0.11/24" rather than the
        normalised network; that must read as a cable, not as malformed."""
        link = topology.CX7Link(hca="rocep1s0f1", netdev="enp1s0f1np1",
                                ipv4="192.168.177.11", cidr="192.168.177.11/24")
        assert topology.transfer_address(SPARK2_IB, "10.0.0.12", [link]) == "192.168.177.12"


# ---------------------------------------------------------------------------
# Part C — the backends emit the flags for every axis
# ---------------------------------------------------------------------------


class TestNvidiaParallelFlags:
    def _args(self, plan):
        from ainode.engine.backends.nvidia import NvidiaBackend

        backend = NvidiaBackend(_config(engine_image="vllm/vllm-openai:v0.27.1"))
        return backend._build_vllm_serve_args(plan=plan)

    def test_pipeline_parallel_is_emitted(self):
        from ainode.engine.parallelism import ParallelPlan

        args = self._args(ParallelPlan(pipeline_parallel_size=3))
        assert "--pipeline-parallel-size" in args
        assert args[args.index("--pipeline-parallel-size") + 1] == "3"
        # Not split along an axis it is not using.
        assert "--tensor-parallel-size" not in args
        assert "--data-parallel-size" not in args
        # Ray places the ranks on the peers for any multi-node split.
        assert "--distributed-executor-backend" in args

    def test_data_parallel_is_emitted(self):
        from ainode.engine.parallelism import ParallelPlan

        args = self._args(ParallelPlan(data_parallel_size=3))
        assert args[args.index("--data-parallel-size") + 1] == "3"

    def test_tensor_parallel_is_unchanged(self):
        from ainode.engine.parallelism import ParallelPlan

        args = self._args(ParallelPlan(tensor_parallel_size=4))
        assert args[args.index("--tensor-parallel-size") + 1] == "4"
        assert "--pipeline-parallel-size" not in args

    def test_solo_emits_no_parallelism_flags_at_all(self):
        """A single-node serve must produce the same command line as before —
        vLLM defaults every axis to 1, and an explicit "1" is log noise."""
        from ainode.engine.parallelism import ParallelPlan

        args = self._args(ParallelPlan())
        assert "--tensor-parallel-size" not in args
        assert "--pipeline-parallel-size" not in args
        assert "--data-parallel-size" not in args
        assert "--distributed-executor-backend" not in args

    def test_combined_axes(self):
        from ainode.engine.parallelism import ParallelPlan

        args = self._args(ParallelPlan(tensor_parallel_size=2,
                                       pipeline_parallel_size=2))
        assert args[args.index("--tensor-parallel-size") + 1] == "2"
        assert args[args.index("--pipeline-parallel-size") + 1] == "2"

    def test_an_explicit_recipe_flag_still_wins(self):
        """extra_vllm_args suppresses the built-in — a published recipe that
        pins its own -pp must not end up with the flag twice."""
        from ainode.engine.backends.nvidia import NvidiaBackend
        from ainode.engine.parallelism import ParallelPlan

        cfg = _config(extra_vllm_args=["--pipeline-parallel-size", "2"])
        args = NvidiaBackend(cfg)._build_vllm_serve_args(
            plan=ParallelPlan(pipeline_parallel_size=3)
        )
        assert args.count("--pipeline-parallel-size") == 1
        assert args[args.index("--pipeline-parallel-size") + 1] == "2"


class TestNvidiaPlanFromConfig:
    def test_resolved_config_is_used(self):
        from ainode.engine.backends.nvidia import NvidiaBackend

        cfg = _config(peer_ips=["10.0.0.12", "10.0.0.13"],
                      parallel_strategy="pipeline", tensor_parallel_size=1,
                      pipeline_parallel_size=3, data_parallel_size=1)
        assert NvidiaBackend(cfg)._parallel_plan().pipeline_parallel_size == 3

    def test_unresolved_config_falls_back_to_tensor_across_all_nodes(self):
        """An older config.json carries no sizes; the backend must behave
        exactly as it did when it derived TP from peer_ips."""
        from ainode.engine.backends.nvidia import NvidiaBackend

        cfg = _config(peer_ips=["10.0.0.12", "10.0.0.13", "10.0.0.14"])
        plan = NvidiaBackend(cfg)._parallel_plan()
        assert plan.tensor_parallel_size == 4
        assert plan.pipeline_parallel_size == 1

    def test_a_plan_that_does_not_fill_the_nodes_is_ignored(self):
        """Stale sizes from a previous launch with a different node set must
        not silently under-fill; fall back to the node-count derivation."""
        from ainode.engine.backends.nvidia import NvidiaBackend

        cfg = _config(peer_ips=["10.0.0.12"], pipeline_parallel_size=3)
        assert NvidiaBackend(cfg)._parallel_plan().tensor_parallel_size == 2

    def test_tp_size_shim_still_answers(self):
        from ainode.engine.backends.nvidia import NvidiaBackend

        cfg = _config(peer_ips=["10.0.0.12"])
        assert NvidiaBackend(cfg)._tp_size() == 2


class TestEugrLaunchScript:
    def _script(self, config, tmp_path, monkeypatch) -> str:
        monkeypatch.setattr(
            "ainode.engine.backends.eugr.EUGR_LAUNCHER",
            tmp_path / "launch-cluster.sh",
        )
        monkeypatch.setattr(
            "ainode.engine.backends.eugr.detect_gpu", lambda: None
        )
        return EugrBackend(config)._write_distributed_launch_script().read_text()

    def test_pipeline_parallel_reaches_the_launcher(self, tmp_path, monkeypatch):
        """launch-cluster.sh parses -tp/-pp/-dp out of the script itself to
        size the node list, so the flags have to be in the text."""
        config = _config(peer_ips=["10.0.0.12", "10.0.0.13"],
                         parallel_strategy="pipeline", tensor_parallel_size=1,
                         pipeline_parallel_size=3, data_parallel_size=1)
        script = self._script(config, tmp_path, monkeypatch)
        assert "--tensor-parallel-size 1" in script
        assert "--pipeline-parallel-size 3" in script
        assert "--data-parallel-size" not in script

    def test_data_parallel_reaches_the_launcher(self, tmp_path, monkeypatch):
        config = _config(peer_ips=["10.0.0.12", "10.0.0.13"],
                         parallel_strategy="data", tensor_parallel_size=1,
                         pipeline_parallel_size=1, data_parallel_size=3)
        script = self._script(config, tmp_path, monkeypatch)
        assert "--data-parallel-size 3" in script

    def test_tensor_parallel_script_is_unchanged(self, tmp_path, monkeypatch):
        """The 2-node script must read exactly as it did: TP=2, PP=1."""
        config = _config(peer_ips=["10.0.0.12"])
        script = self._script(config, tmp_path, monkeypatch)
        assert "--tensor-parallel-size 2" in script
        assert "--pipeline-parallel-size 1" in script
        assert "--data-parallel-size" not in script


class TestAnnouncedParallelPlan:
    def _instances(self, config):
        from ainode.api.server import _head_instances

        return _head_instances(config)

    def test_pipeline_head_advertises_pp(self):
        got = self._instances(_config(
            peer_ips=["10.0.0.12", "10.0.0.13"], parallel_strategy="pipeline",
            tensor_parallel_size=1, pipeline_parallel_size=3, data_parallel_size=1,
        ))
        assert got[0]["pipeline_parallel_size"] == 3
        assert got[0]["tensor_parallel_size"] == 1

    def test_unresolved_head_advertises_tensor_as_before(self):
        got = self._instances(_config(peer_ips=["10.0.0.12", "10.0.0.13"]))
        assert got[0]["tensor_parallel_size"] == 3
        assert got[0]["pipeline_parallel_size"] == 1

    def test_solo_head_advertises_nothing(self):
        assert self._instances(_config(peer_ips=[])) == []

    def test_record_renders_a_badge_label(self):
        from ainode.discovery.instance import InstanceRecord

        record = InstanceRecord.from_dict(self._instances(_config(
            peer_ips=["10.0.0.12", "10.0.0.13"], parallel_strategy="pipeline",
            tensor_parallel_size=1, pipeline_parallel_size=3, data_parallel_size=1,
        ))[0])
        assert record.parallel_label() == "PP=3"
        assert record.world_size == 3


class TestNcclVersionFloor:
    """NCCL_IB_SUBNET_AWARE_ROUTING is defined from NCCL v2.30.7-1 onward
    (src/transport/net_ib/connect.cc) and is absent from 2.28.x, 2.29.x and
    2.30.3. Below the floor AINode sets the variable and NCCL ignores it, so a
    mesh silently fails to route — doctor has to say so."""

    @pytest.mark.parametrize("version", ["2.30.7-1", "2.31.2-1", "2.30.7", "3.0.0-1"])
    def test_supported_versions_pass(self, version):
        from ainode.cli.doctor import _nccl_supports_subnet_aware_routing

        assert _nccl_supports_subnet_aware_routing(version)

    @pytest.mark.parametrize(
        "version", ["2.28.3-1", "2.28.9-1", "2.29.7-1", "2.30.3-1", "2.27.5-1"]
    )
    def test_versions_below_the_floor_fail(self, version):
        from ainode.cli.doctor import _nccl_supports_subnet_aware_routing

        assert not _nccl_supports_subnet_aware_routing(version)

    @pytest.mark.parametrize("version", ["", "unknown", None])
    def test_unknown_reads_as_supported(self, version):
        """No engine image pulled yet is not a reason to cry wolf."""
        from ainode.cli.doctor import _nccl_supports_subnet_aware_routing

        assert _nccl_supports_subnet_aware_routing(version or "")

    def test_doctor_warns_on_a_mesh_with_old_nccl(self, sysfs, monkeypatch):
        from ainode.cli import doctor

        _wire_mesh(sysfs)
        monkeypatch.setattr(
            "ainode.core.config.NodeConfig.load", classmethod(lambda cls: _config())
        )
        monkeypatch.setattr(doctor, "_nccl_version", lambda: "2.28.3-1")
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            verdict, rows, warnings = doctor.fabric_report()
        assert verdict == "warn"
        assert any("2.30.7" in w for w in warnings)
        assert dict(rows)["NCCL version"] == "2.28.3-1"

    def test_doctor_is_quiet_on_a_mesh_with_new_nccl(self, sysfs, monkeypatch):
        from ainode.cli import doctor

        _wire_mesh(sysfs)
        monkeypatch.setattr(
            "ainode.core.config.NodeConfig.load", classmethod(lambda cls: _config())
        )
        monkeypatch.setattr(doctor, "_nccl_version", lambda: "2.30.7-1")
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            verdict, _rows, warnings = doctor.fabric_report()
        assert verdict == "ok"
        assert warnings == []

    def test_old_nccl_off_the_mesh_is_not_flagged(self, sysfs, monkeypatch):
        """A switched cluster never reads the parameter, so 2.28.3 is fine."""
        from ainode.cli import doctor

        _wire_direct(sysfs)
        monkeypatch.setattr(
            "ainode.core.config.NodeConfig.load", classmethod(lambda cls: _config())
        )
        monkeypatch.setattr(doctor, "_nccl_version", lambda: "2.28.3-1")
        with mock.patch("subprocess.run", side_effect=_fake_ip):
            verdict, _rows, warnings = doctor.fabric_report()
        assert verdict == "ok"
        assert warnings == []


class TestEugrEngineMountPaths:
    """The -v SOURCE goes to the host daemon and the serve script runs inside
    the engine container. Confusing the two views silently mounts an empty
    directory and points --download-dir at a path that does not exist there."""

    def _launch(self, sysfs_wired, tmp_path, monkeypatch, **cfg):
        from ainode.engine.backends.eugr import EugrBackend

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr("ainode.engine.backends.eugr.EUGR_LAUNCHER", launcher)
        monkeypatch.setattr("ainode.engine.backends.eugr.EUGR_ENV_FILE",
                            tmp_path / "eugr.env")
        backend = EugrBackend(_config(**cfg))
        captured = {}

        class _P:
            def __init__(self, *a, **k):
                captured["env"] = k.get("env", {})
                self.stdout = None

            def poll(self):
                return None

        with mock.patch("subprocess.Popen", _P), \
             mock.patch.object(EugrBackend, "_write_eugr_env", lambda self: None), \
             mock.patch.object(EugrBackend, "_write_distributed_launch_script",
                               lambda self: tmp_path / "s.sh"), \
             mock.patch.object(EugrBackend, "_distribute_model_to_peers",
                               lambda self: None), \
             mock.patch.object(EugrBackend, "_publish_nccl_init_script",
                               lambda self: None), \
             mock.patch("subprocess.run", side_effect=_fake_ip), \
             mock.patch("threading.Thread"):
            backend.start_distributed()
        return captured["env"].get("VLLM_SPARK_EXTRA_DOCKER_ARGS", "")

    def test_mount_source_is_the_host_path(self, sysfs, tmp_path, monkeypatch):
        _wire_mesh(sysfs)
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/admin/.ainode")
        monkeypatch.setattr("ainode.core.config.AINODE_HOME",
                            __import__("pathlib").Path("/root/.ainode"))
        args = self._launch(sysfs, tmp_path, monkeypatch,
                            models_dir="/root/.ainode/models")
        assert "/home/admin/.ainode/models:/models" in args
        assert "/root/.ainode/models:/models" not in args

    def test_mount_source_untouched_outside_a_container(self, sysfs, tmp_path, monkeypatch):
        _wire_mesh(sysfs)
        monkeypatch.delenv("AINODE_HOST_HOME", raising=False)
        args = self._launch(sysfs, tmp_path, monkeypatch,
                            models_dir="/srv/models")
        assert "/srv/models:/models" in args

    def test_serve_script_uses_the_container_mount_target(self, tmp_path, monkeypatch):
        from ainode.engine.backends.eugr import EugrBackend

        monkeypatch.setattr("ainode.engine.backends.eugr.EUGR_LAUNCHER",
                            tmp_path / "launch-cluster.sh")
        monkeypatch.setattr("ainode.engine.backends.eugr.detect_gpu", lambda: None)
        config = _config(peer_ips=["10.0.0.12"], models_dir="/root/.ainode/models")
        script = EugrBackend(config)._write_distributed_launch_script().read_text()
        # /models is where the launcher mounts it in every engine container;
        # AINode's own view of models_dir does not exist there.
        assert "--download-dir /models" in script
        assert "/root/.ainode/models" not in script
