"""The engine image has to be on every node, and the head puts it there.

launch-cluster.sh checks and aborts when a node lacks it, or when the ids
differ. That is correct and unhelpful: the operator is told to go and fix three
machines by hand. A catalog recipe pinning an engine_image makes that routine
rather than exceptional — so the head places the image the way it places
weights.

Pull first, copy second: a registry image is far cheaper to pull on each peer
in parallel, layers deduping against what is already there, than to stream ~20
GB through the head. The copy is the fallback for a locally built image.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ainode.engine.distribute import (
    DistributionError,
    ensure_local_image,
    ensure_peer_has_image,
)

IMAGE = "vllm/vllm-openai:v0.27.1"
HEAD_ID = "sha256:aaaa"
OTHER_ID = "sha256:bbbb"


def _runner(*, local="", peer="", after_pull=None, pull_rc=0, copy_rc=0):
    """Fake subprocess.run covering the commands this code issues."""
    state = {"peer": peer}

    def run(cmd, *a, **k):
        joined = " ".join(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return mock.Mock(returncode=0 if local else 1, stdout=local, stderr="")
        if "docker image inspect" in joined:          # over ssh
            return mock.Mock(returncode=0, stdout=state["peer"], stderr="")
        if cmd[:2] == ["docker", "pull"]:
            state["local"] = local
            return mock.Mock(returncode=pull_rc, stdout="", stderr="")
        if "docker pull" in joined:                    # over ssh
            if after_pull is not None:
                state["peer"] = after_pull
            return mock.Mock(returncode=pull_rc, stdout="", stderr="")
        if cmd[:2] == ["bash", "-lc"] and "docker save" in cmd[2]:
            return mock.Mock(returncode=copy_rc, stdout="",
                             stderr="no space left on device" if copy_rc else "")
        return mock.Mock(returncode=0, stdout="", stderr="")

    return run


def _place(**kw):
    return ensure_peer_has_image(
        ssh_user="admin", coord_ip="192.168.1.3", transfer_ip="10.100.36.2",
        image=IMAGE, **kw,
    )


class TestPeerAlreadyHasIt:
    def test_matching_id_is_left_alone(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer=HEAD_ID)) as run:
            assert _place() == "present"
        assert not any("docker pull" in " ".join(c[0][0]) for c in run.call_args_list)

    def test_a_differing_id_is_not_accepted(self):
        """The launcher requires identical ids across nodes: a tag that moved
        in the registry would otherwise leave ranks on different builds, which
        fails later and far less clearly."""
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer=OTHER_ID)):
            assert _place() == "copied"


class TestPullFirst:
    def test_a_registry_image_is_pulled_on_the_peer(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer="",
                                            after_pull=HEAD_ID)) as run:
            assert _place() == "pulled"
        cmds = [" ".join(c[0][0]) for c in run.call_args_list]
        assert any("docker pull" in c for c in cmds)
        assert not any("docker save" in c for c in cmds)

    def test_the_head_does_not_stream_when_a_pull_works(self):
        """~20 GB through one link, avoided."""
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer="",
                                            after_pull=HEAD_ID)) as run:
            _place()
        assert not any("docker save" in " ".join(c[0][0])
                       for c in run.call_args_list)


class TestCopyFallback:
    def test_a_locally_built_image_is_copied(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer="",
                                            after_pull="")) as run:
            assert _place() == "copied"
        assert any("docker save" in " ".join(c[0][0]) for c in run.call_args_list)

    def test_the_copy_uses_the_direct_link(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer="")) as run:
            _place()
        save = next(" ".join(c[0][0]) for c in run.call_args_list
                    if "docker save" in " ".join(c[0][0]))
        assert "10.100.36.2" in save        # transfer address, not coordination
        assert "192.168.1.3" not in save

    def test_a_failed_copy_raises_with_the_reason(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID, peer="", copy_rc=1)), \
             pytest.raises(DistributionError, match="no space left"):
            _place()

    def test_nowhere_to_copy_from_says_so(self):
        """Neither node has it and it cannot be pulled — the operator needs to
        know that pulling or building it here is the way out."""
        with mock.patch("subprocess.run",
                        side_effect=_runner(local="", peer="")), \
             pytest.raises(DistributionError, match="neither this node nor"):
            _place()


class TestLocalImage:
    def test_present_is_a_no_op(self):
        with mock.patch("subprocess.run",
                        side_effect=_runner(local=HEAD_ID)) as run:
            assert ensure_local_image(IMAGE) == "present"
        assert not any(c[0][0][:2] == ["docker", "pull"] for c in run.call_args_list)

    def test_missing_is_pulled(self):
        calls = []

        def run(cmd, *a, **k):
            calls.append(cmd)
            if cmd[:3] == ["docker", "image", "inspect"]:
                found = any(c[:2] == ["docker", "pull"] for c in calls)
                return mock.Mock(returncode=0 if found else 1,
                                 stdout=HEAD_ID if found else "", stderr="")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=run):
            assert ensure_local_image(IMAGE) == "pulled"

    def test_unpullable_raises(self):
        with mock.patch("subprocess.run", side_effect=_runner(local="")), \
             pytest.raises(DistributionError, match="could not be pulled"):
            ensure_local_image(IMAGE)

    def test_no_image_is_a_no_op(self):
        assert ensure_local_image("") == "present"


class TestBothBackendsPlaceIt:
    def test_eugr_places_it_before_launching(self, tmp_path, monkeypatch):
        """Ordering matters: the launcher aborts on a missing image, so the
        placement has to happen before it is spawned."""
        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr as E

        launcher = tmp_path / "launch-cluster.sh"
        launcher.write_text("#!/bin/bash\n")
        monkeypatch.setattr(E, "EUGR_LAUNCHER", launcher)
        order = []

        class _P:
            def __init__(self, *a, **k):
                order.append("launcher")
                self.stdout = None

            def poll(self):
                return None

        backend = E.EugrBackend(NodeConfig(
            node_id="n", model="m", distributed_mode="head",
            peer_ips=["10.0.0.2"], engine_image=IMAGE,
        ))
        with mock.patch.object(E, "subprocess") as sp, \
             mock.patch.object(type(backend), "_write_eugr_env", lambda s: None), \
             mock.patch.object(type(backend), "_write_launch_script",
                               lambda s, plan, solo=False: tmp_path / "s.sh"), \
             mock.patch.object(type(backend), "_publish_nccl_init_script",
                               lambda s: None), \
             mock.patch.object(type(backend), "_launcher_env", lambda s, **k: {}), \
             mock.patch.object(type(backend), "_distribute_model_to_peers",
                               lambda s: None), \
             mock.patch.object(E, "ensure_local_image",
                               lambda i, **k: order.append("image") or "present"), \
             mock.patch.object(E, "ensure_peer_has_image",
                               lambda **k: "present"), \
             mock.patch("threading.Thread"):
            sp.Popen = _P
            backend.start_distributed()
        assert order == ["image", "launcher"]

    def test_nvidia_places_it_before_starting_workers(self):
        from pathlib import Path

        from ainode.engine.backends import nvidia

        src = Path(nvidia.__file__).read_text()
        assert src.index("self._distribute_engine_image_to_peers()") < src.index(
            "self._ssh_launch_worker(\n                peer_ip=peer_ip,"
        )

    def test_eugr_skips_it_without_a_pinned_image(self):
        """The launcher default is built locally on every node."""
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.eugr import EugrBackend

        backend = EugrBackend(NodeConfig(node_id="n", peer_ips=["10.0.0.2"]))
        with mock.patch("ainode.engine.backends.eugr.ensure_peer_has_image") as p:
            backend._distribute_engine_image_to_peers()
        p.assert_not_called()
