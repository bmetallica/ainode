"""The peers must get the directory the launch will serve from.

These two disagreed. The launch serves the flat `org--name` directory our
downloader writes; the distribution copied the Hugging Face cache layout,
`models_dir/hub/models--org--name`. On a cluster where the operator downloaded
through the UI that source does not exist, so nothing was distributed —
silently, because a missing source is also the normal "the peer already has
it" case.

The head then served a path the peer did not have, and transformers, given a
path it cannot find, falls back to the Hub:

    HFValidationError: Repo id must be in the form 'repo_name' or
    'namespace/repo_name': '/models/local-inference-lab--GLM-5.3-…'

raised inside ray::RayWorkerProc.initialize_worker() on the peer, while the
head — which had the weights — showed nothing wrong.
"""

from __future__ import annotations

from unittest import mock

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import eugr as E

MODEL = "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
SLUG = "local-inference-lab--GLM-5.3-Flash-NVFP4-Spark"


def _backend(models_dir, peers=("10.0.0.2",)):
    backend = E.EugrBackend.__new__(E.EugrBackend)
    backend.config = NodeConfig(model=MODEL, models_dir=str(models_dir),
                                distributed_mode="head", peer_ips=list(peers),
                                ssh_user="admin")
    backend._distributed_log = models_dir / "distributed.log"
    backend._phase = E.LoadPhaseTracker()
    backend._phase.reset()
    return backend


def _downloaded_flat(root):
    directory = root / SLUG
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    return directory


class TestItCopiesWhatTheLaunchServes:
    def test_the_flat_directory_is_what_travels(self, tmp_path):
        _downloaded_flat(tmp_path)
        backend = _backend(tmp_path)
        with mock.patch.object(E, "ensure_peer_has_dir",
                               return_value=True) as copy:
            backend._distribute_model_to_peers()
        kwargs = copy.call_args.kwargs
        assert kwargs["dir_name"] == SLUG
        assert kwargs["source_parent"] == str(tmp_path)
        assert kwargs["target_parent"] == str(tmp_path)

    def test_it_matches_what_the_launch_script_serves(self, tmp_path):
        # The two decisions have to agree, so compare them directly.
        _downloaded_flat(tmp_path)
        backend = _backend(tmp_path)
        serve_target, _ = backend._serve_target_and_name()
        with mock.patch.object(E, "ensure_peer_has_dir",
                               return_value=True) as copy:
            backend._distribute_model_to_peers()
        assert serve_target.endswith(copy.call_args.kwargs["dir_name"])

    def test_without_a_local_copy_it_falls_back_to_the_cache_layout(self, tmp_path):
        # Serving by repo id: vLLM resolves through the HF cache, so that is
        # the layout the peers need.
        backend = _backend(tmp_path)
        with mock.patch.object(E, "ensure_peer_has_dir",
                               return_value=True) as copy:
            backend._distribute_model_to_peers()
        kwargs = copy.call_args.kwargs
        assert kwargs["dir_name"] == "models--" + MODEL.replace("/", "--")
        assert kwargs["source_parent"].endswith("/hub")

    def test_every_peer_gets_it(self, tmp_path):
        _downloaded_flat(tmp_path)
        backend = _backend(tmp_path, peers=("10.0.0.2", "10.0.0.3"))
        with mock.patch.object(E, "ensure_peer_has_dir",
                               return_value=True) as copy:
            backend._distribute_model_to_peers()
        assert copy.call_count == 2


class TestWhenThereIsNothingToSend:
    def test_it_says_so_rather_than_going_quiet(self, tmp_path):
        # Silence here is what made the peer's failure look like the head's
        # problem. The message reaches the log and the instance card.
        backend = _backend(tmp_path)
        with mock.patch.object(E, "ensure_peer_has_dir", return_value=False):
            backend._distribute_model_to_peers()
        assert "fetch it" in backend.load_detail
        # And in the file, which is where someone who came back later looks.
        assert "not on this node's disk" in (tmp_path / "distributed.log").read_text()

    def test_a_failed_transfer_still_aborts(self, tmp_path):
        # A half-copied checkpoint is worse than none.
        _downloaded_flat(tmp_path)
        backend = _backend(tmp_path)
        with mock.patch.object(E, "ensure_peer_has_dir",
                               side_effect=E.DistributionError("disk full")):
            with pytest.raises(E.DistributionError):
                backend._distribute_model_to_peers()
