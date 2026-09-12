"""A launch must never be silent for minutes.

Before the launcher writes its first line, a distributed start can pull a ~20
GB engine image and copy tens of GB of weights to each peer. Neither step logs
anything the operator sees: the card said "STARTING · 12%", vllm.log and
distributed.log were empty, and no engine container existed yet — which is
exactly what a wedged process looks like. Observed on hardware 2026-09-12,
five minutes in, with no way to tell whether anything was happening at all.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from ainode.core.config import NodeConfig
from ainode.engine.backends import eugr as E
from ainode.engine.load_phase import LoadPhaseTracker

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()
SERVER = (Path(__file__).resolve().parent.parent / "ainode" / "api" /
          "server.py").read_text()
IMAGE = "vllm/vllm-openai:v0.27.1"


class TestPhaseDetail:
    def test_a_note_is_reported(self):
        t = LoadPhaseTracker()
        t.reset()
        t.note("pulling the engine image")
        assert t.detail == "pulling the engine image"

    def test_moving_on_clears_it(self):
        # A stale "copying weights to spark-2" under "loading weights" is
        # worse than no detail at all.
        t = LoadPhaseTracker()
        t.reset()
        t.note("copying weights to 10.0.0.2")
        t.advance("loading_weights")
        assert t.detail == ""

    def test_a_new_launch_starts_clean(self):
        t = LoadPhaseTracker()
        t.note("stale")
        t.reset()
        assert t.detail == ""


def _backend(tmp_path, monkeypatch, **config_kwargs):
    launcher = tmp_path / "launch-cluster.sh"
    launcher.write_text("#!/bin/bash\n")
    monkeypatch.setattr(E, "EUGR_LAUNCHER", launcher)
    backend = E.EugrBackend(NodeConfig(
        node_id="n", model="org/model", distributed_mode="head",
        peer_ips=["10.0.0.2"], engine_image=IMAGE, **config_kwargs))
    backend._distributed_log = tmp_path / "distributed.log"
    return backend


class TestPreLaunchIsNarrated:
    def test_the_image_pull_reaches_the_card_and_the_log(self, tmp_path, monkeypatch):
        backend = _backend(tmp_path, monkeypatch)
        seen = []

        def _pull(image, on_pull=None):
            if on_pull is not None:
                on_pull()
            seen.append(backend.load_detail)
            return "pulled"

        with mock.patch.object(E, "ensure_local_image", _pull), \
             mock.patch.object(E, "ensure_peer_has_image", lambda **k: "present"):
            backend._distribute_engine_image_to_peers()

        assert any(IMAGE in d and "pulling" in d for d in seen), seen
        written = (tmp_path / "distributed.log").read_text()
        assert IMAGE in written and "pulling" in written

    def test_a_present_image_says_so_without_claiming_a_pull(self, tmp_path, monkeypatch):
        backend = _backend(tmp_path, monkeypatch)
        with mock.patch.object(E, "ensure_local_image", lambda i, **k: "present"), \
             mock.patch.object(E, "ensure_peer_has_image", lambda **k: "present"):
            backend._distribute_engine_image_to_peers()
        written = (tmp_path / "distributed.log").read_text()
        assert "pulling" not in written
        assert "present" in written

    def test_placing_the_image_on_a_peer_names_the_peer(self, tmp_path, monkeypatch):
        backend = _backend(tmp_path, monkeypatch)

        def _peer(**kwargs):
            on_start = kwargs.get("on_start")
            if on_start is not None:
                on_start()
            return "copied"

        with mock.patch.object(E, "ensure_local_image", lambda i, **k: "present"), \
             mock.patch.object(E, "ensure_peer_has_image", _peer):
            backend._distribute_engine_image_to_peers()
        assert "10.0.0.2" in (tmp_path / "distributed.log").read_text()

    def test_the_weight_copy_names_the_model_and_the_address(self, tmp_path, monkeypatch):
        backend = _backend(tmp_path, monkeypatch)

        def _dir(**kwargs):
            on_start = kwargs.get("on_start")
            if on_start is not None:
                on_start()
            return True

        with mock.patch.object(E, "ensure_peer_has_dir", _dir):
            backend._distribute_model_to_peers()
        written = (tmp_path / "distributed.log").read_text()
        assert "org/model" in written and "10.0.0.2" in written

    def test_the_phase_advances_past_starting(self, tmp_path, monkeypatch):
        # 12% with an empty log is the shape of the hang this fixes.
        backend = _backend(tmp_path, monkeypatch)
        backend._phase.reset()
        with mock.patch.object(E, "ensure_local_image", lambda i, **k: "present"), \
             mock.patch.object(E, "ensure_peer_has_image", lambda **k: "present"):
            backend._distribute_engine_image_to_peers()
        assert backend.load_phase == "distributing"


class TestItReachesTheOperator:
    def test_status_carries_the_detail(self):
        assert '"load_detail"' in SERVER

    def test_the_card_renders_it(self):
        assert "load_detail" in APP_JS
        assert "loadDetail" in APP_JS


class TestFailuresNameTheEvidence:
    """A hint must not replace the line it is explaining.

    "the engine rejected a command-line flag" is true of every exit-2 and
    names none of them. The operator needs the flag.
    """

    def _failed(self, *lines):
        tracker = LoadPhaseTracker()
        tracker.reset()
        for line in lines:
            tracker.observe(line)
        tracker.fail("the launcher exited (code 2)")
        return tracker.failure_reason()

    def test_the_rejected_flag_is_named(self):
        reason = self._failed(
            "INFO starting",
            "vllm serve: error: unrecognized arguments: --speculative_config {}",
            "Stopping cluster...")
        assert "--speculative_config" in reason

    def test_and_the_explanation_still_comes_with_it(self):
        reason = self._failed(
            "vllm serve: error: unrecognized arguments: --nope")
        assert "--nope" in reason
        assert "catalog recipe" in reason

    def test_a_drafter_failure_names_the_architecture_line(self):
        reason = self._failed(
            "INFO Resolved architecture: Qwen3DSparkModel",
            "AttributeError: 'NoneType' object has no attribute 'draft_model_config'")
        assert "Qwen3DSparkModel" in reason
        assert "DRAFT model" in reason

    def test_teardown_noise_is_still_not_the_explanation(self):
        reason = self._failed("Stopping cluster...", "Cluster stopped.")
        assert "Stopping cluster" not in reason or "exited" in reason


class TestThePhaseFollowsThisVllm:
    """The markers have to match the phrasing the engine actually prints.

    "Loading model from scratch..." is what the current build says, and it
    matched nothing — so a launch reading 23 GB off disk showed "starting ·
    12%" for its whole duration, which is what a hang looks like. Reported as
    one, twice.
    """

    def _phase_after(self, *lines):
        tracker = LoadPhaseTracker()
        tracker.reset()
        for line in lines:
            tracker.observe(line)
        return tracker

    def test_loading_model_from_scratch_advances_the_phase(self):
        t = self._phase_after("INFO [model_runner.py:396] Loading model from scratch...")
        assert t.current() == "loading_weights"

    def test_init_engine_counts_as_profiling(self):
        t = self._phase_after(
            "INFO [core.py:372] init engine (profile, create kv cache, warmup model) took 150.72 s")
        assert t.current() == "profiling"

    def test_a_download_is_called_a_download(self):
        # Reading 23 GB off disk and pulling it over the internet are both
        # "loading weights", and the operator wants to know which one.
        t = self._phase_after(
            "INFO Loading model from scratch...",
            "Warning: You are sending unauthenticated requests to the HF Hub.")
        assert "Hugging Face" in t.detail

    def test_reading_from_disk_says_so(self):
        t = self._phase_after("INFO Loading safetensors checkpoint shards: 10%")
        assert "from disk" in t.detail

    def test_compilation_is_named_as_a_one_off(self):
        t = self._phase_after("INFO torch.compile takes 38.49 s in total")
        assert "first launch" in t.detail

    def test_readiness_still_wins(self):
        t = self._phase_after("INFO Loading model from scratch...",
                              "INFO:     Application startup complete.")
        assert t.current() == "ready"
