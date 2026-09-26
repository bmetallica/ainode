"""A dropped packet is not a failed download, and a failed one must say so.

Reported from the cluster, on a link that comes and goes:

    der download (unvollständig) wird übrigens immernoch in der modelliste als
    fertig (ohne hinweiß das er nicht vollständig ist) gelistet ... kannst du
    noch irgendwas an der downloadstabilität machen damit er nicht bei kleinen
    netzaussetzern komplett abbricht sondern einfach weiter macht?

Two failures with one cause. The pull fetches through a thread pool and
propagates the FIRST error, cancelling everything not yet started — so one
timeout on one shard of twenty-four ends the transfer. And what it leaves
behind can satisfy all three completeness heuristics at once: no shard index
yet, so nothing to check the shards against; the tokenizer files that did
arrive pass the tokenizer check; the in-flight files finish renaming before
anybody looks. Three inferences, all agreeing on the wrong answer.
"""

from __future__ import annotations

import errno
import json

import pytest

from ainode.models.completeness import (DOWNLOAD_MARK, clear_download_mark,
                                        clear_partials, download_state,
                                        mark_download_started)
from ainode.models.download_retry import MAX_ATTEMPTS, retry_delay, should_retry


class _Http(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class _Cancelled(Exception):
    pass


class TestWhatIsWorthAnotherGo:
    @pytest.mark.parametrize("exc", [
        TimeoutError("read timed out"),
        ConnectionResetError("peer reset"),
        OSError(errno.ECONNRESET, "reset"),
        _Http(502), _Http(503), _Http(429), _Http(500),
        Exception("IncompleteRead(1024 bytes read)"),
    ])
    def test_a_transient_failure_is_retried(self, exc):
        assert should_retry(exc) is True

    @pytest.mark.parametrize("exc", [
        _Http(401), _Http(403), _Http(404),
        OSError(errno.ENOSPC, "no space left on device"),
        OSError(errno.EACCES, "permission denied"),
    ])
    def test_an_answer_that_will_not_change_is_not_retried(self, exc):
        """Retrying a 404 turns a clear error into a slow one, and retrying a
        full disk makes it fuller."""
        assert should_retry(exc) is False

    def test_a_cancellation_is_never_retried(self):
        # That is the operator's decision; repeating it would override them.
        assert should_retry(_Cancelled()) is False

    def test_the_backoff_grows_and_is_capped(self):
        delays = [retry_delay(a) for a in range(2, MAX_ATTEMPTS + 1)]
        assert delays == sorted(delays)
        assert max(delays) <= 30.0

    def test_the_whole_budget_is_about_a_minute(self):
        # Long enough for a radio to reacquire; short enough that a repo does
        # not stall on one file while the link is actually down.
        total = sum(retry_delay(a) for a in range(2, MAX_ATTEMPTS + 1))
        assert 20 <= total <= 90


class TestTheRecordBeatsTheHeuristics:
    def _tree(self, tmp_path):
        """A directory that every inference reads as finished: config, a
        tokenizer, a shard, no index, no partials."""
        directory = tmp_path / "org--model"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer.json").write_text("{}")
        (directory / "model-00004-of-00024.safetensors").write_bytes(b"x")
        return directory

    def test_the_heuristics_do_read_it_as_finished(self, tmp_path):
        # Which is the bug, stated as a test rather than as a complaint.
        assert download_state(self._tree(tmp_path)) == (True, "")

    def test_the_mark_overrides_them(self, tmp_path):
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model")
        complete, reason = download_state(directory)
        assert complete is False
        assert "started and did not finish" in reason

    def test_it_says_the_record_is_not_an_inference(self, tmp_path):
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model")
        assert "not inferred from the files" in download_state(directory)[1]

    def test_the_last_error_is_carried(self, tmp_path):
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model", attempts=5,
                             last_error="ConnectionResetError: peer reset")
        reason = download_state(directory)[1]
        assert "ConnectionResetError" in reason

    def test_finishing_clears_it(self, tmp_path):
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model")
        assert clear_download_mark(directory) is True
        assert download_state(directory) == (True, "")

    def test_clearing_twice_is_not_an_error(self, tmp_path):
        directory = self._tree(tmp_path)
        assert clear_download_mark(directory) is False

    def test_the_mark_is_json_and_names_the_repo(self, tmp_path):
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model")
        blob = json.loads((directory / DOWNLOAD_MARK).read_text())
        assert blob["repo"] == "org/model"
        assert blob["started_at"] > 0

    def test_an_import_clears_it(self, tmp_path):
        """Completing a download by hand is a completion. Otherwise a model
        carried in over a dead pull would stay marked forever."""
        directory = self._tree(tmp_path)
        mark_download_started(directory, "org/model")
        clear_partials(directory)
        assert download_state(directory) == (True, "")

    def test_a_missing_shard_still_wins_when_there_is_no_mark(self, tmp_path):
        # The original check must keep working on its own.
        directory = self._tree(tmp_path)
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00004-of-00024.safetensors",
                            "b": "model-00005-of-00024.safetensors"}}))
        complete, reason = download_state(directory)
        assert complete is False and "missing 1 of the shards" in reason


class TestBothDownloadPathsRecordIt:
    def test_the_repo_pull_marks_and_retries(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes._run_download_repo)
        assert "mark_download_started" in source
        assert "clear_download_mark" in source
        assert "should_retry" in source

    def test_the_catalog_pull_marks_and_retries(self):
        import inspect

        from ainode.models.registry import ModelManager

        source = inspect.getsource(ModelManager.download_model)
        assert "mark_download_started" in source
        assert "clear_download_mark" in source
        assert "should_retry" in source

    def test_a_failed_repo_pull_keeps_the_mark(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes._run_download_repo)
        # The mark is re-written with the error in the failure branch, not
        # cleared — that is the whole point of it.
        failure = source.split('terminal = {"status": "failed"')[1]
        assert "mark_download_started" in failure


class TestResumeChecksBeforeItFetches:
    """Asked alongside the stability question:

        die hf modelle bestehen ja meist aus vielen einzeldateien. es wäre also
        schon gut wenn man einen button "Download fortsetzen" hätte welcher
        dann die vorhandenen dateien nach vollständigkeit prüft, evtl
        unvollständiges löscht und dann alles fehlende weiter herunterlädt

    Two of the three steps already existed: hf_hub_download skips a finished
    file and resumes an unfinished one, so a second pull IS "fetch what is
    missing". What it cannot do is notice a file that is SHORT but whose
    metadata says otherwise — killed mid-write, or a lost tail. That file is
    skipped and the engine finds out later.
    """

    EXPECTED = {"config.json": 20, "model-00001-of-00002.safetensors": 1000,
                "model-00002-of-00002.safetensors": 1000,
                "tokenizer.json": 50}

    def _tree(self, tmp_path, good=(), short=(), staged=()):
        from ainode.models import resume

        directory = tmp_path / "org--model"
        directory.mkdir(parents=True)
        for name in good:
            (directory / name).write_bytes(b"x" * self.EXPECTED[name])
        for name in short:
            (directory / name).write_bytes(b"x" * (self.EXPECTED[name] // 2))
        cache = directory / ".cache" / "huggingface" / "download"
        for name in staged:
            cache.mkdir(parents=True, exist_ok=True)
            (cache / (name + ".incomplete")).write_bytes(b"x" * 10)
        return directory, resume

    def test_a_short_file_is_found(self, tmp_path):
        directory, resume = self._tree(
            tmp_path, good=["config.json"],
            short=["model-00001-of-00002.safetensors"])
        report = resume.inspect_local(directory, self.EXPECTED)
        assert [e["path"] for e in report["wrong_size"]] == \
            ["model-00001-of-00002.safetensors"]
        assert report["present"] == ["config.json"]

    def test_a_missing_file_is_found(self, tmp_path):
        directory, resume = self._tree(tmp_path, good=["config.json"])
        report = resume.inspect_local(directory, self.EXPECTED)
        assert "tokenizer.json" in report["missing"]

    def test_preparing_removes_the_short_one_and_the_staging(
            self, tmp_path, monkeypatch):
        directory, resume = self._tree(
            tmp_path, good=["config.json", "tokenizer.json"],
            short=["model-00001-of-00002.safetensors"],
            staged=["model-00002-of-00002.safetensors"])
        monkeypatch.setattr(resume, "hub_sizes",
                            lambda repo, token=None: self.EXPECTED)
        report, freed = resume.prepare_resume(directory, "org/model")
        assert not (directory / "model-00001-of-00002.safetensors").exists()
        assert not list(directory.rglob("*.incomplete"))
        assert len(report["removed"]) == 2 and freed > 0

    def test_preparing_leaves_the_good_files_alone(self, tmp_path, monkeypatch):
        directory, resume = self._tree(
            tmp_path, good=["config.json", "tokenizer.json"],
            short=["model-00001-of-00002.safetensors"])
        monkeypatch.setattr(resume, "hub_sizes",
                            lambda repo, token=None: self.EXPECTED)
        report, freed = resume.prepare_resume(directory, "org/model")
        assert (directory / "config.json").is_file()
        assert not (directory / "model-00001-of-00002.safetensors").exists()
        assert report["removed"] == ["model-00001-of-00002.safetensors"]
        assert freed == self.EXPECTED["model-00001-of-00002.safetensors"] // 2

    def test_what_was_removed_is_what_will_be_fetched(self, tmp_path,
                                                     monkeypatch):
        directory, resume = self._tree(
            tmp_path, good=["config.json"],
            short=["model-00001-of-00002.safetensors"])
        monkeypatch.setattr(resume, "hub_sizes",
                            lambda repo, token=None: self.EXPECTED)
        report, _ = resume.prepare_resume(directory, "org/model")
        assert "model-00001-of-00002.safetensors" in report["will_fetch"]
        assert "tokenizer.json" in report["will_fetch"]
        assert "config.json" not in report["will_fetch"]

    def test_our_own_files_are_never_judged(self, tmp_path, monkeypatch):
        from ainode.models.completeness import DOWNLOAD_MARK

        directory, resume = self._tree(tmp_path, good=["config.json"])
        (directory / DOWNLOAD_MARK).write_text("{}")
        monkeypatch.setattr(resume, "hub_sizes",
                            lambda repo, token=None: self.EXPECTED)
        resume.prepare_resume(directory, "org/model")
        # The mark belongs to the download, not to the repo. Clearing it is the
        # successful pull's job, not the preparation's.
        assert (directory / DOWNLOAD_MARK).is_file()

    def test_no_hub_means_no_size_check_and_it_says_so(self, tmp_path,
                                                      monkeypatch):
        directory, resume = self._tree(
            tmp_path, good=["config.json"],
            short=["model-00001-of-00002.safetensors"],
            staged=["model-00002-of-00002.safetensors"])
        monkeypatch.setattr(resume, "hub_sizes", lambda repo, token=None: {})
        report, _ = resume.prepare_resume(directory, "org/model")
        assert report["verified"] is False
        assert "not size-checked" in report["note"]
        # Still cleans the staging files, and does NOT delete a file it could
        # not judge.
        assert (directory / "model-00001-of-00002.safetensors").is_file()
        assert not list(directory.rglob("*.incomplete"))


class TestTheButtonIsOnTheCardThatNeedsIt:
    SOURCE = None

    @classmethod
    def setup_class(cls):
        from pathlib import Path

        cls.SOURCE = (Path(__file__).resolve().parent.parent / "ainode" / "web"
                      / "static" / "js" / "app.js").read_text()

    def test_an_incomplete_card_offers_a_resume(self):
        """It used to live only on a paused job in the Downloads queue, which
        is gone after a restart — so a model interrupted by a dropped link had
        no button at all, and the only offered route was deleting it."""
        assert "data-resume-download" in self.SOURCE

    def test_the_button_is_wired(self):
        assert "querySelectorAll('[data-resume-download]')" in self.SOURCE

    def test_it_reports_what_the_check_found(self):
        # A silent check is indistinguishable from no check.
        assert "resumeCheckNote" in self.SOURCE
        assert "unusable file(s)" in self.SOURCE

    def test_the_route_prepares_before_it_pulls(self):
        import inspect

        from ainode.models import api_routes

        source = inspect.getsource(api_routes.handle_resume_download)
        assert "prepare_resume" in source
        assert source.index("prepare_resume") < source.index("_run_download_repo")
