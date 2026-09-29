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


class TestWithoutAnAlias:
    """The alias may name a HostName only the host resolves (/etc/hosts), or
    there may be none: every key in ~/.ssh is offered for the address."""

    def test_every_key_is_tried(self, tmp_path):
        from ainode.cluster.ssh_aliases import private_keys

        for name in ("id_ed25519_nvsync_cluster_assistant", "id_rsa", "known_hosts",
                     "config", "cluster_key"):
            (tmp_path / name).write_text("x")
        (tmp_path / "cluster_key.pub").write_text("x")
        (tmp_path / "id_rsa.pub").write_text("x")
        keys = private_keys(tmp_path)
        assert [k.rsplit("/", 1)[1] for k in keys] == [
            "cluster_key", "id_ed25519_nvsync_cluster_assistant", "id_rsa"]

    def test_an_unmatched_address_gets_them(self):
        section = ip_blocks(parse_ssh_config(HOST_CONFIG), [["192.168.1.9"]],
                            resolve=_resolve, keys=["/root/.ssh/id_nv", "/root/.ssh/id_rsa"])
        assert "Host 192.168.1.9" in section
        assert "IdentityFile /root/.ssh/id_nv" in section

    def test_a_matched_one_keeps_its_own(self):
        section = ip_blocks(parse_ssh_config(HOST_CONFIG), [["192.168.1.3"], ["192.168.1.9"]],
                            resolve=_resolve, keys=["/root/.ssh/id_other"])
        first = section.split("# no alias found")[0]
        assert "Host 192.168.1.3" in first and "id_other" not in first


class TestTheCheck:
    def test_it_maps_then_tries_each_member(self, monkeypatch):
        import subprocess

        import ainode.cluster.ssh_aliases as aliases
        from ainode.core.config import NodeConfig

        mapped, ran = [], []
        monkeypatch.setattr(aliases, "ensure_ip_aliases", lambda groups: mapped.append(groups))

        class _Done:
            def __init__(self, code, out="", err=""):
                self.returncode, self.stdout, self.stderr = code, out, err

        def _run(cmd, **kw):
            ran.append(cmd)
            if cmd[1] == "-G":
                return _Done(0, "user admin\nidentityfile /root/.ssh/id_nv\n")
            return _Done(0 if cmd[-2] == "192.168.1.3" else 255,
                         err="admin@192.168.1.4: Permission denied (publickey)")

        monkeypatch.setattr(subprocess, "run", _run)
        nodes = [type("N", (), {"node_id": i, "node_name": n, "fabric_ip": ip,
                                "peer_ip": ip, "ib_ips": []})()
                 for i, n, ip in (("s1", "spark-13e1", "192.168.1.2"),
                                  ("s2", "spark-659b", "192.168.1.3"),
                                  ("s3", "spark-1432", "192.168.1.4"))]
        app = {"config": NodeConfig(node_id="s1"),
               "cluster_state": type("C", (), {"members": lambda self: nodes})()}
        results = aliases.check_ssh(app)
        assert [r["node_id"] for r in results] == ["s2", "s3"]
        assert results[0]["ok"] and not results[1]["ok"]
        assert "Permission denied" in results[1]["error"]
        assert results[0]["identity_files"] == ["/root/.ssh/id_nv"]
        assert len(mapped[0]) == 2
