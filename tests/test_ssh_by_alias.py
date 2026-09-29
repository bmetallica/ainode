"""`ssh <ip>` uses the settings of the operator's alias for that node.

Reported: "Passwordless SSH to 192.168.1.4 failed" — while `ssh admin@Spark3`
worked. The host's ~/.ssh/config names nodes by alias; the launcher connects
by address, which no block matched.
"""

from __future__ import annotations

from ainode.cluster.ssh_aliases import BEGIN, END, ensure_ip_aliases, ip_blocks, parse_ssh_config

HOST_CONFIG = """\
# Injected by AINode entrypoint for container→peer ssh
Host *
    User root-default
    StrictHostKeyChecking no

Host Spark2
    HostName 192.168.1.3
    User admin
    IdentityFile /root/.ssh/id_ed25519_nvsync_cluster_assistant
    IdentitiesOnly yes

Host Spark3
    HostName spark3.lan
    User admin
    IdentityFile /root/.ssh/id_ed25519_nvsync_cluster_assistant

Match host *.example
    User nobody
"""


def _resolve(name):
    return {"spark3.lan": ["192.168.1.4"]}.get(name, [])


class TestTheBlocks:
    def test_an_alias_whose_hostname_is_the_address(self):
        section = ip_blocks(parse_ssh_config(HOST_CONFIG), [["192.168.1.3"]], resolve=_resolve)
        assert "Host 192.168.1.3" in section
        assert "User admin" in section
        assert "IdentityFile /root/.ssh/id_ed25519_nvsync_cluster_assistant" in section
        assert "HostName" not in section

    def test_an_alias_whose_hostname_resolves_to_it(self):
        section = ip_blocks(parse_ssh_config(HOST_CONFIG), [["192.168.1.4"]], resolve=_resolve)
        assert "Host 192.168.1.4" in section and "# for Spark3" in section

    def test_every_address_of_the_node_gets_it(self):
        """The launcher may use the fabric address while the alias names the
        LAN one: the node's addresses are one group."""
        group = ["10.100.0.4", "192.168.1.4", "spark-1432", "10.200.0.4"]
        section = ip_blocks(parse_ssh_config(HOST_CONFIG), [group], resolve=_resolve)
        assert "Host 10.100.0.4 10.200.0.4 192.168.1.4" in section
        assert "spark-1432" not in section.split("\n")[1]

    def test_a_node_without_an_alias_gets_nothing(self):
        assert ip_blocks(parse_ssh_config(HOST_CONFIG), [["10.9.9.9"]], resolve=_resolve) == ""


class TestTheFile:
    def test_the_section_goes_first_and_is_replaced_not_stacked(self, tmp_path):
        path = tmp_path / "config"
        path.write_text(HOST_CONFIG)
        assert ensure_ip_aliases([["192.168.1.4"]], path=path, resolve=_resolve) == 1
        assert ensure_ip_aliases([["192.168.1.3"]], path=path, resolve=_resolve) == 1
        text = path.read_text()
        # First, before the entrypoint's `Host *`, whose User would win otherwise.
        assert text.startswith(BEGIN)
        assert text.count(BEGIN) == 1 and text.count(END) == 1
        assert "Host 192.168.1.3" in text and "Host 192.168.1.4" not in text
        # The operator's own blocks are untouched.
        assert "Host Spark3\n    HostName spark3.lan" in text

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert ensure_ip_aliases([["1.2.3.4"]], path=tmp_path / "none", resolve=_resolve) == 0


class TestTheLaunchUsesIt:
    def test_start_distributed_maps_the_peers_first(self, monkeypatch):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr
        import ainode.cluster.ssh_aliases as aliases

        seen = []
        monkeypatch.setattr(aliases, "ensure_ip_aliases", lambda groups: seen.append(groups))
        backend = eugr.EugrBackend(NodeConfig(model="m", distributed_mode="head",
                                              peer_ips=["192.168.1.4"]))
        backend.ssh_peer_addresses = [["192.168.1.4", "192.168.1.4", "spark-1432"]]
        backend._ensure_ssh_aliases()
        assert seen == [[["192.168.1.4", "192.168.1.4", "spark-1432"]]]

    def test_without_the_cluster_the_peer_addresses_alone(self, monkeypatch):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr
        import ainode.cluster.ssh_aliases as aliases

        seen = []
        monkeypatch.setattr(aliases, "ensure_ip_aliases", lambda groups: seen.append(groups))
        eugr.EugrBackend(NodeConfig(model="m", distributed_mode="head",
                                    peer_ips=["192.168.1.4"]))._ensure_ssh_aliases()
        assert seen == [[["192.168.1.4"]]]
