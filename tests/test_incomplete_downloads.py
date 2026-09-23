"""A download that was interrupted must not look like one that finished.

Reported from the cluster:

    wenn ainode (z.b. wegen einem update) neustartet während ein modell von
    hf heruntergeladen wird, dieses nach dem neustart als vollständig
    heruntergeladen gelistet wird … ich muss es erst komplett löschen und
    erneut herunterladen

The listing answered "is there a directory for this repo". A half-downloaded
model satisfies that, so it showed as On disk, refused to launch minutes in
with something about safetensors, and the only way out was to delete it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ainode.models.completeness import download_state, is_complete


def _repo(tmp_path, shards=("a", "b"), present=("a", "b"), partial=()):
    directory = tmp_path / "org--model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    if shards:
        names = [f"model-0000{i}-of-0000{len(shards)}.safetensors"
                 for i in range(1, len(shards) + 1)]
        weight_map = {f"layer.{i}": names[i] for i in range(len(shards))}
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
        for index, shard in enumerate(shards):
            if shard in present:
                (directory / names[index]).write_bytes(b"x")
    for name in partial:
        cache = directory / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{name}.incomplete").write_bytes(b"x")
    return directory


class TestFinishedDownloads:
    def test_all_the_shards_are_there(self, tmp_path):
        assert download_state(_repo(tmp_path)) == (True, "")

    def test_a_repo_with_no_index_is_given_the_benefit_of_the_doubt(self, tmp_path):
        # Refusing to launch a model that is fine is worse than the failure
        # this catches, so anything it cannot reason about passes.
        directory = tmp_path / "org--small"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"x")
        assert is_complete(directory)

    def test_a_directory_that_is_not_there_is_not_incomplete(self, tmp_path):
        assert is_complete(tmp_path / "nothing")


class TestInterruptedMidFile:
    def test_a_partial_file_is_caught(self, tmp_path):
        complete, reason = download_state(
            _repo(tmp_path, partial=("model-00002-of-00002.safetensors",)))
        assert complete is False
        assert "interrupted" in reason

    def test_it_names_the_file(self, tmp_path):
        _, reason = download_state(_repo(tmp_path, partial=("shard-7.safetensors",)))
        assert "shard-7.safetensors" in reason
        assert ".incomplete" not in reason

    def test_many_partials_are_summarised(self, tmp_path):
        directory = _repo(tmp_path, partial=tuple(f"s{i}.bin" for i in range(6)))
        _, reason = download_state(directory)
        assert "6 file(s)" in reason and "and 3 more" in reason


class TestInterruptedBetweenFiles:
    """The case the restart produced: a shard never started leaves no trace
    at all except its absence from an index that lists it."""

    def test_a_missing_shard_is_caught(self, tmp_path):
        complete, reason = download_state(
            _repo(tmp_path, shards=("a", "b", "c"), present=("a", "b")))
        assert complete is False
        assert "missing 1 of the shards" in reason

    def test_it_names_them(self, tmp_path):
        _, reason = download_state(
            _repo(tmp_path, shards=("a", "b"), present=("a",)))
        assert "model-00002-of-00002.safetensors" in reason

    def test_it_says_what_to_do(self, tmp_path):
        _, reason = download_state(
            _repo(tmp_path, shards=("a", "b"), present=("a",)))
        assert "Resume the download" in reason


class TestTheListingSaysSo:
    def test_the_entry_carries_it(self, tmp_path):
        from ainode.models.registry import ModelManager

        _repo(tmp_path, shards=("a", "b"), present=("a",))
        entries = ModelManager(models_dir=str(tmp_path)).list_downloaded()
        entry = next(e for e in entries if e["hf_repo"] == "org/model")
        assert entry["complete"] is False
        assert "missing" in entry["incomplete_reason"]

    def test_a_finished_one_says_complete(self, tmp_path):
        from ainode.models.registry import ModelManager

        _repo(tmp_path)
        entries = ModelManager(models_dir=str(tmp_path)).list_downloaded()
        assert next(e for e in entries
                    if e["hf_repo"] == "org/model")["complete"] is True

    def test_the_card_shows_it(self):
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "Incomplete" in app_js
        assert "incompleteReason" in app_js


class TestTheLaunchRefusesIt:
    def test_the_gate_says_which_shards(self, tmp_path):
        from ainode.safety.admission import check_admission

        directory = _repo(tmp_path, shards=("a", "b"), present=("a",))

        class _Manager:
            def model_dirs_for_repo(self, repo):
                return [directory]

        refusal = check_admission({"model_manager": _Manager()}, "org/model")
        assert "missing" in refusal and "org/model" in refusal

    def test_it_is_asked_first(self):
        # Cheapest and most certain: a missing shard is not a memory question.
        import inspect

        from ainode.safety import admission

        source = inspect.getsource(admission.check_admission)
        assert source.index("_completeness_says") < source.index("_quantization_says")


class TestPauseAndResume:
    def test_both_routes_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="n1"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/models/download-pause" in paths
        assert "/api/models/download-resume" in paths

    def test_pausing_keeps_the_partial_tree(self):
        # The whole difference between pause and cancel.
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes._run_download_repo)
        assert "_keep_partial" in source
        assert "discard_partial = not paused" in source

    def test_the_pause_sets_that_flag(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_pause_download)
        assert '"_keep_partial"' in source
        assert '"_cancel"' in source

    def test_a_cancel_still_deletes(self):
        import inspect

        from ainode.models import api_routes

        assert "_keep_partial" not in inspect.getsource(
            api_routes.handle_cancel_download)

    def test_resume_refuses_to_race_a_running_job(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_resume_download)
        assert "already downloading" in source

    def test_the_ui_has_both_buttons(self):
        app_js = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert "data-pause-job" in app_js
        assert "data-resume-repo" in app_js
        assert "pauseDownload(" in app_js and "resumeDownload(" in app_js
