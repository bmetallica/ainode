"""The head holds the models; the sub-nodes hold copies of the head's.

The operator's rule, stated plainly: nothing is done on a sub-node's UI, no
model arrives on one by any other route, everything goes through the head.
Sub-nodes keep their copies, so a launch is a sync of what changed rather
than a fetch.

What that needs, and what this covers:

  * a download on the head is pushed outward immediately — at a moment when
    nobody is waiting for a model to come up;
  * models already on the head before any of this existed can be pushed once,
    explicitly, because that can be several hundred gigabytes;
  * a copy that MATCHES the head, not merely exists: a peer left with a
    half-finished directory passed the old "does it exist" test and was never
    corrected;
  * the head preferred as the source when a node does pull;
  * and the browser UI switched off where nobody opens it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig
from ainode.engine import mirror

MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
FLAT = "nvidia--Gemma-4-26B-A4B-NVFP4"


class _Node:
    def __init__(self, node_id, fabric_ip, is_master=False, web_port=3000):
        self.node_id = node_id
        self.fabric_ip = fabric_ip
        self.ib_ips = []
        self.web_port = web_port
        self.is_master = is_master


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes(self, include_offline=False):
        return self._nodes


def _app(tmp_path, nodes=(), node_id="head"):
    return {
        "config": NodeConfig(node_id=node_id, node_name="head",
                             ssh_user="admin", models_dir=str(tmp_path)),
        "cluster_state": _Cluster(list(nodes)),
    }


class TestFindingTheCopy:
    def test_the_flat_layout(self, tmp_path):
        target = tmp_path / FLAT
        target.mkdir()
        (target / "w.safetensors").write_bytes(b"\x00")
        assert mirror.model_source_dir(MODEL, str(tmp_path)) == (str(tmp_path), FLAT)

    def test_the_hub_layout(self, tmp_path):
        target = tmp_path / "hub" / f"models--{FLAT}"
        target.mkdir(parents=True)
        (target / "w.safetensors").write_bytes(b"\x00")
        parent, name = mirror.model_source_dir(MODEL, str(tmp_path))
        assert Path(parent).name == "hub" and name == f"models--{FLAT}"

    def test_an_empty_directory_is_not_a_copy(self, tmp_path):
        """What an aborted download leaves. Mirroring it would propagate the
        abort to every node."""
        (tmp_path / FLAT).mkdir()
        assert mirror.model_source_dir(MODEL, str(tmp_path)) is None

    def test_nothing_here(self, tmp_path):
        assert mirror.model_source_dir(MODEL, str(tmp_path)) is None


class TestMirroring:
    def _ready(self, tmp_path):
        target = tmp_path / FLAT
        target.mkdir()
        (target / "w.safetensors").write_bytes(b"\x00" * 32)

    def test_every_peer_gets_it(self, tmp_path):
        self._ready(tmp_path)
        app = _app(tmp_path, [_Node("n2", "10.0.0.2"), _Node("n3", "10.0.0.3")])
        with mock.patch.object(mirror, "ensure_peer_has_dir", return_value=True):
            assert mirror.mirror_model_to_peers(app, MODEL) == {
                "n2": "copied", "n3": "copied"}

    def test_it_asks_for_a_match_not_a_placement(self, tmp_path):
        """A peer holding a half-finished copy passes a "does it exist" test
        and is then never corrected."""
        self._ready(tmp_path)
        app = _app(tmp_path, [_Node("n2", "10.0.0.2")])
        with mock.patch.object(mirror, "ensure_peer_has_dir",
                               return_value=True) as push:
            mirror.mirror_model_to_peers(app, MODEL)
        assert push.call_args.kwargs["resync"] is True

    def test_a_failing_peer_does_not_stop_the_others(self, tmp_path):
        from ainode.engine.distribute import DistributionError

        self._ready(tmp_path)
        app = _app(tmp_path, [_Node("n2", "10.0.0.2"), _Node("n3", "10.0.0.3")])

        def push(**kw):
            if kw["transfer_ip"] == "10.0.0.2":
                raise DistributionError("no space left on device")
            return True

        with mock.patch.object(mirror, "ensure_peer_has_dir", side_effect=push):
            results = mirror.mirror_model_to_peers(app, MODEL)
        assert "no space left" in results["n2"]
        assert results["n3"] == "copied"

    def test_nothing_to_mirror_is_not_an_error(self, tmp_path):
        app = _app(tmp_path, [_Node("n2", "10.0.0.2")])
        with mock.patch.object(mirror, "ensure_peer_has_dir") as push:
            assert mirror.mirror_model_to_peers(app, MODEL) == {}
        push.assert_not_called()

    def test_without_an_ssh_user_it_refuses_rather_than_guesses(self, tmp_path):
        self._ready(tmp_path)
        app = _app(tmp_path, [_Node("n2", "10.0.0.2")])
        app["config"].ssh_user = ""
        with mock.patch.object(mirror, "ensure_peer_has_dir") as push:
            assert mirror.mirror_model_to_peers(app, MODEL) == {}
        push.assert_not_called()

    def test_progress_is_reported_per_node(self, tmp_path):
        self._ready(tmp_path)
        app = _app(tmp_path, [_Node("n2", "10.0.0.2")])
        seen = []
        with mock.patch.object(mirror, "ensure_peer_has_dir", return_value=True):
            mirror.mirror_model_to_peers(app, MODEL, lambda n, s: seen.append((n, s)))
        assert seen == [("n2", "copying"), ("n2", "copied")]


class TestTheDownloadPushesOutward:
    def test_the_catalog_download_mirrors(self):
        import inspect

        from ainode.models.api_routes import _run_download

        assert "_mirror_after_download" in inspect.getsource(_run_download)

    def test_the_repo_download_mirrors_only_on_success(self):
        source = Path("ainode/models/api_routes.py").read_text()
        assert ('if terminal.get("status") == "completed":\n'
                "        await _mirror_after_download(app, hf_repo, jobs[job_id])") in source

    def test_a_mirror_failure_does_not_fail_the_download(self):
        source = Path("ainode/models/api_routes.py").read_text()
        helper = source[source.index("async def _mirror_after_download"):]
        helper = helper[:helper.index("\n# --", 1)]
        assert "except Exception" in helper


class TestTheInitialSweep:
    """Models already on the head predate all of this and would otherwise sit
    there until someone launched them somewhere.

    Both halves of the sweep are patched, not just the copy: ensure_dependencies
    reaches the Hub, and leaving it live made teardown wait out a real
    snapshot_download — 47 seconds in a suite that otherwise runs in
    seventeen. A test that touches the network is a test that fails on a
    train.
    """

    @pytest_asyncio.fixture
    async def client(self, tmp_path):
        app = create_app(config=NodeConfig(node_id="head", ssh_user="admin",
                                           models_dir=str(tmp_path)), engine=None)
        async with TestClient(TestServer(app)) as c:
            yield c

    @pytest.mark.asyncio
    async def test_it_reports_what_it_will_send(self, client):
        with mock.patch.object(type(client.app["model_manager"]), "list_downloaded",
                               lambda self: [{"hf_repo": MODEL}]), \
             mock.patch("ainode.engine.mirror.mirror_model_to_peers", return_value={}), \
             mock.patch("ainode.engine.mirror.ensure_dependencies", return_value=[]):
            response = await client.post("/api/cluster/mirror-models", json={})
            assert response.status == 202
            assert (await response.json())["models"] == [MODEL]

    @pytest.mark.asyncio
    async def test_nothing_downloaded_is_a_404_not_a_silent_success(self, client):
        with mock.patch.object(type(client.app["model_manager"]), "list_downloaded",
                               lambda self: []):
            response = await client.post("/api/cluster/mirror-models", json={})
        assert response.status == 404

    @pytest.mark.asyncio
    async def test_a_named_subset_is_honoured(self, client):
        with mock.patch.object(type(client.app["model_manager"]), "list_downloaded",
                               lambda self: [{"hf_repo": MODEL}, {"hf_repo": "a/b"}]), \
             mock.patch("ainode.engine.mirror.mirror_model_to_peers", return_value={}), \
             mock.patch("ainode.engine.mirror.ensure_dependencies", return_value=[]):
            response = await client.post("/api/cluster/mirror-models",
                                         json={"models": [MODEL]})
        assert (await response.json())["models"] == [MODEL]

    @pytest.mark.asyncio
    async def test_status_is_readable(self, client):
        response = await client.get("/api/cluster/mirror-status")
        assert response.status == 200
        assert (await response.json())["running"] is False


class TestTheHeadIsPreferredAsSource:
    def test_the_master_comes_first(self):
        from ainode.engine import acquire

        cluster = _Cluster([_Node("n2", "10.0.0.2"), _Node("n1", "10.0.0.1", True)])
        payload = json.dumps({"models": [{"hf_repo": MODEL}]}).encode()

        class _R:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("urllib.request.urlopen", return_value=_R()):
            order = [node_id for node_id, _, _ in
                     acquire.peers_with_model(cluster, "me", MODEL)]
        assert order == ["n1", "n2"]

    def test_a_launch_syncs_even_when_the_weights_are_here(self):
        source = Path("ainode/models/api_routes.py").read_text()
        helper = source[source.index("def _fetch_weights_from_a_peer"):]
        helper = helper[:helper.index("\ndef ", 1)]
        assert "sync=True" in helper


class TestTheUiCanBeSwitchedOff:
    @pytest.mark.asyncio
    async def test_off_says_where_to_go_instead_of_404ing_blankly(self):
        app = create_app(config=NodeConfig(node_id="n3", node_name="spark-3",
                                           web_ui_enabled=False), engine=None)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/")
            assert response.status == 404
            text = await response.text()
        assert "operated from the head" in text
        assert "spark-3" in text

    @pytest.mark.asyncio
    async def test_the_api_is_untouched(self):
        """Discovery, the cluster dispatch that places models here and the
        federated proxy all run over it — this is not a way to close a node
        off."""
        app = create_app(config=NodeConfig(node_id="n3", web_ui_enabled=False),
                         engine=None)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/api/health")).status == 200
            assert (await client.get("/v1/models")).status == 200

    @pytest.mark.asyncio
    async def test_on_is_the_default(self):
        app = create_app(config=NodeConfig(node_id="n1"), engine=None)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/")).status == 200


class TestADependencyIsPartOfTheModel:
    """The drafter that made this necessary.

    Gemma 4 26B's recipe names google/gemma-4-26B-A4B-it-assistant, and vLLM
    downloads it at LAUNCH time, on whichever node is launching. Measured: 801
    MB, 113 seconds from Hugging Face, on a sub-node — which is precisely the
    thing a head-only deployment forbids, and the head did not have it at all.

    Telling the operator to rsync it back by hand was the wrong answer: "das
    muss doch alles automatisch gemacht werden".
    """

    SPEC = ('{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant",'
            '"num_speculative_tokens":4}')
    DRAFTER = "google/gemma-4-26B-A4B-it-assistant"

    class _Info:
        def __init__(self, args):
            self.extra_vllm_args = args

    def _app(self, tmp_path, **config_kw):
        class _Manager:
            def __init__(self, info):
                self._info = info

            def _find_catalog_by_hf_repo(self, repo):
                return self._info

        return {
            "config": NodeConfig(node_id="head", ssh_user="admin",
                                 models_dir=str(tmp_path), **config_kw),
            "cluster_state": _Cluster([]),
            "model_manager": _Manager(self._Info(["--speculative-config", self.SPEC])),
        }

    def test_a_peer_is_asked_before_the_internet(self, tmp_path):
        """The copy that exists is on the fabric — the sub-node downloaded it.
        Fetching it back over the link beats fetching it again over the
        uplink."""
        app = self._app(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="fetched") as peer, \
             mock.patch("huggingface_hub.snapshot_download") as hub:
            assert mirror.ensure_dependencies(app, MODEL) == [self.DRAFTER]
        assert peer.call_args.kwargs["model"] == self.DRAFTER
        hub.assert_not_called()

    def test_the_hub_is_the_fallback(self, tmp_path):
        app = self._app(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="absent"), \
             mock.patch("huggingface_hub.snapshot_download") as hub:
            assert mirror.ensure_dependencies(app, MODEL) == [self.DRAFTER]
        assert hub.call_args.kwargs["repo_id"] == self.DRAFTER

    def test_an_already_present_dependency_is_not_fetched(self, tmp_path):
        target = tmp_path / "models--google--gemma-4-26B-A4B-it-assistant"
        target.mkdir()
        (target / "w.safetensors").write_bytes(b"\x00")
        app = self._app(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer") as peer, \
             mock.patch("huggingface_hub.snapshot_download") as hub:
            assert mirror.ensure_dependencies(app, MODEL) == [self.DRAFTER]
        peer.assert_not_called()
        hub.assert_not_called()

    def test_a_sub_node_fetches_nothing(self, tmp_path):
        """It may not download, and the head is supposed to have sent this."""
        app = self._app(tmp_path, download_from_hub=False)
        with mock.patch("huggingface_hub.snapshot_download") as hub:
            assert mirror.ensure_dependencies(app, MODEL) == []
        hub.assert_not_called()

    def test_a_failure_does_not_fail_the_download(self, tmp_path):
        """Refusing to finish a model that is otherwise complete helps nobody;
        the launch would have failed either way, and says so more clearly."""
        app = self._app(tmp_path)
        with mock.patch("ainode.engine.acquire.fetch_model_from_peer",
                        return_value="absent"), \
             mock.patch("huggingface_hub.snapshot_download",
                        side_effect=RuntimeError("404")):
            assert mirror.ensure_dependencies(app, MODEL) == []

    def test_a_model_with_no_recipe_wants_nothing(self, tmp_path):
        app = self._app(tmp_path)
        app["model_manager"] = None
        assert mirror.ensure_dependencies(app, MODEL) == []

    def test_the_sweep_mirrors_them_too(self):
        import inspect

        from ainode.api.server import handle_cluster_mirror_models

        source = inspect.getsource(handle_cluster_mirror_models)
        assert "ensure_dependencies" in source
        assert "for repo in [model, *extras]:" in source

    def test_a_download_mirrors_them_too(self):
        import inspect

        from ainode.models.api_routes import _mirror_after_download

        source = inspect.getsource(_mirror_after_download)
        assert "ensure_dependencies" in source
        assert "for repo in [model, *extras]:" in source
