"""A solo launch should take weights from a neighbour, not from the internet.

Measured on the cluster while loading Gemma 4 26B on a third node:

    237,474,984,041 -> 237,542,066,339 bytes in 60 s  =  1.1 MB/s

about five hours for a 19 GB checkpoint. The engine had already started —
"Multi-modal warmup completed" — and was simply waiting on Hugging Face,
unauthenticated and rate-limited, for weights that were on the machine next
door. The same gap cost 17 GB of Qwen the day before.

A distributed launch pushes weights head→peer (ensure_peer_has_dir). A solo
launch on a node that lacks them had no equivalent. The only difference is
direction.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from ainode.engine import acquire
from ainode.engine.distribute import DistributionError

MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
FLAT = "nvidia--Gemma-4-26B-A4B-NVFP4"
CACHED = "models--nvidia--Gemma-4-26B-A4B-NVFP4"


class _Node:
    def __init__(self, node_id, fabric_ip, ib_ips=(), web_port=3000):
        self.node_id = node_id
        self.fabric_ip = fabric_ip
        self.ib_ips = list(ib_ips)
        self.web_port = web_port


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self, include_offline=False):
        return self._nodes


def _place(root: Path, relative: str) -> Path:
    target = root / relative
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"\x00" * 16)
    return target


class TestIsItAlreadyHere:
    def test_a_directory_download_counts(self, tmp_path):
        _place(tmp_path, FLAT)
        assert acquire.model_is_local(MODEL, str(tmp_path))

    @pytest.mark.parametrize("layout", ["", "hub", "hf-cache/hub"])
    def test_every_cache_layout_counts(self, tmp_path, layout):
        _place(tmp_path, str(Path(layout) / CACHED))
        assert acquire.model_is_local(MODEL, str(tmp_path))

    @pytest.mark.parametrize("layout", [FLAT, f"hub/{CACHED}"])
    def test_an_empty_directory_does_not(self, tmp_path, layout):
        """What an aborted download leaves. Counting it as present would skip
        the copy and hand the engine an empty directory."""
        (tmp_path / layout).mkdir(parents=True)
        assert not acquire.model_is_local(MODEL, str(tmp_path))

    def test_nothing_is_nothing(self, tmp_path):
        assert not acquire.model_is_local(MODEL, str(tmp_path))

    def test_no_models_dir_is_not_a_crash(self):
        assert not acquire.model_is_local(MODEL, "")


def _answer(has: bool):
    payload = json.dumps({"models": [{"hf_repo": MODEL}] if has else []}).encode()

    class _Response:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Response()


class TestWhichPeerHasIt:
    def test_a_peer_that_has_it_is_returned(self):
        cluster = _Cluster([_Node("n2", "10.0.0.2")])
        with mock.patch("urllib.request.urlopen", return_value=_answer(True)):
            assert acquire.peers_with_model(cluster, "n1", MODEL) == [
                ("n2", "10.0.0.2", [])]

    def test_a_peer_without_it_is_not(self):
        cluster = _Cluster([_Node("n2", "10.0.0.2")])
        with mock.patch("urllib.request.urlopen", return_value=_answer(False)):
            assert acquire.peers_with_model(cluster, "n1", MODEL) == []

    def test_this_node_is_never_asked(self):
        cluster = _Cluster([_Node("n1", "10.0.0.1")])
        with mock.patch("urllib.request.urlopen") as urlopen:
            assert acquire.peers_with_model(cluster, "n1", MODEL) == []
        urlopen.assert_not_called()

    def test_an_unreachable_peer_is_skipped_not_raised(self):
        """One node down must not stop a launch on a healthy one."""
        cluster = _Cluster([_Node("n2", "10.0.0.2")])
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no route")):
            assert acquire.peers_with_model(cluster, "n1", MODEL) == []

    def test_no_cluster_is_not_a_crash(self):
        assert acquire.peers_with_model(None, "n1", MODEL) == []


class TestFetching:
    def _fetch(self, tmp_path, **kw):
        return acquire.fetch_model_from_peer(
            cluster=_Cluster([_Node("n2", "10.0.0.2", ["10.100.36.2"])]),
            own_node_id="n1", ssh_user="admin", model=MODEL,
            models_dir=str(tmp_path), **kw)

    def test_present_weights_are_not_fetched_again(self, tmp_path):
        _place(tmp_path, FLAT)
        with mock.patch.object(acquire, "fetch_dir_from_peer") as fetch:
            assert self._fetch(tmp_path) == "present"
        fetch.assert_not_called()

    def test_a_peer_copy_is_pulled(self, tmp_path):
        with mock.patch("urllib.request.urlopen", return_value=_answer(True)), \
             mock.patch.object(acquire, "fetch_dir_from_peer",
                               return_value=True) as fetch:
            assert self._fetch(tmp_path) == "fetched"
        assert fetch.call_args.kwargs["dir_name"] == FLAT
        assert fetch.call_args.kwargs["ssh_user"] == "admin"

    def test_the_cache_layout_is_the_second_try(self, tmp_path):
        """The peer may hold either, and which it is decides where this node
        has to put it."""
        calls = []

        def fetch(**kw):
            calls.append(kw["dir_name"])
            return kw["dir_name"] == CACHED

        with mock.patch("urllib.request.urlopen", return_value=_answer(True)), \
             mock.patch.object(acquire, "fetch_dir_from_peer", side_effect=fetch):
            assert self._fetch(tmp_path) == "fetched"
        assert calls == [FLAT, CACHED]

    def test_the_host_path_is_what_crosses_the_ssh(self, tmp_path):
        # This process runs in a container; the ssh lands on the peer's HOST.
        with mock.patch("urllib.request.urlopen", return_value=_answer(True)), \
             mock.patch.object(acquire, "fetch_dir_from_peer",
                               return_value=True) as fetch:
            self._fetch(tmp_path, host_models_dir="/home/admin/.ainode/models")
        assert fetch.call_args.kwargs["remote_parent"] == "/home/admin/.ainode/models"
        # ...while the destination is this container's own view.
        assert fetch.call_args.kwargs["target_parent"] == str(tmp_path)

    def test_a_failed_transfer_falls_through_to_downloading(self, tmp_path):
        """Downloading is slower, not impossible. A half-copied directory or a
        missing ssh key must not turn a working launch into a failed one."""
        with mock.patch("urllib.request.urlopen", return_value=_answer(True)), \
             mock.patch.object(acquire, "fetch_dir_from_peer",
                               side_effect=DistributionError("rc=255")):
            assert self._fetch(tmp_path) == "absent"

    def test_nobody_has_it(self, tmp_path):
        with mock.patch("urllib.request.urlopen", return_value=_answer(False)):
            assert self._fetch(tmp_path) == "absent"


class TestTheLaunchPathUsesIt:
    def test_it_runs_before_the_engine_starts(self):
        source = Path("ainode/models/api_routes.py").read_text()
        body = source[source.index("def append_solo_instance"):]
        assert body.index("_fetch_weights_from_a_peer(app, backend, model, config)") \
            < body.index("ok = backend.start()")

    def test_it_cannot_fail_a_launch_on_a_node_that_may_download(self):
        """Where downloading is allowed, every path here is an optimisation
        over it: a peer that is down must not turn a working, if slower,
        launch into a failed one.

        Where it is NOT allowed — a sub-node in a head-only deployment — the
        opposite holds and the launch must stop. See
        tests/test_sub_nodes_never_reach_the_hub.py.
        """
        source = Path("ainode/models/api_routes.py").read_text()
        helper = source[source.index("def _fetch_weights_from_a_peer"):]
        helper = helper[:helper.index("\ndef ", 1)]
        assert "except Exception as exc:" in helper
        assert "if may_download:" in helper
        assert "return None" in helper

    def test_it_tells_the_card_what_it_is_doing(self):
        # A multi-gigabyte copy writes no launcher output, and an unexplained
        # ten-minute pause at "starting" is how this was reported in the first
        # place.
        source = Path("ainode/models/api_routes.py").read_text()
        assert 'f"copying the weights from {node_id}"' in source


class TestTheCardIsToldWhatIsHappening:
    """The callback existed and was never passed.

    _fetch_weights_from_a_peer defined an `announce` that would have advanced
    the phase, and then called fetch_model_from_peer without it. Nothing else
    writes a line during this step — the launcher has not started, so the
    phase tracker has nothing to read — and the card sat at the "idle" 8%
    through a sync that can take minutes. It was reported as a hang, twice.

    Python does not warn about an unused local function and neither does ruff,
    so the only thing that catches this is a test that follows the wire.
    """

    def _progress_calls(self, outcome: str) -> list:
        from unittest import mock

        from ainode.core.config import NodeConfig
        from ainode.models.api_routes import _fetch_weights_from_a_peer

        seen = []

        class _Backend:
            _log_file = None

            def _progress(self, phase, detail, log_file):
                seen.append((phase, detail))

        def fake(**kwargs):
            starter = kwargs.get("on_start")
            assert starter is not None, "on_start was not passed"
            starter("spark-13e1")
            return outcome

        config = NodeConfig(node_id="n3", ssh_user="admin", models_dir="/models")
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer", side_effect=fake):
            _fetch_weights_from_a_peer({"cluster_state": None}, _Backend(),
                                       "org/model", config)
        return seen

    def test_the_search_itself_is_announced(self):
        # An HTTP round trip per peer, five seconds each on a node that is
        # down, with nothing on screen.
        details = [detail for _, detail in self._progress_calls("fetched")]
        assert details[0] == "looking for the weights on the other nodes"

    def test_the_copy_names_the_node_it_comes_from(self):
        details = [detail for _, detail in self._progress_calls("fetched")]
        assert "copying the weights from spark-13e1" in details

    def test_the_phase_moves_off_idle(self):
        # 8% is PHASE_INFO's "idle" — the fallback that reads as a hang.
        phases = {phase for phase, _ in self._progress_calls("fetched")}
        assert phases == {"distributing"}
