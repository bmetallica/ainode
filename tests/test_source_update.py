"""Updating from our own fork's source, from the UI.

The update path that already exists compares this node's VERSION against the
latest published image tag. That is right for a deployment that pulls images
and wrong for this one, which builds on the head from a checkout: the code
moves far more often than the version number, so a node can be forty commits
behind and report itself current.

The question asked here is the other one — which commit was this image built
from, and how far is the fork's branch ahead of it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


from ainode.update.runner import CONTAINER_SOURCE_DIR, UpdateRunner
from ainode.update.source import SourceState, built_from, check_for_updates

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()
INDEX = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
         "templates" / "index.html").read_text()


class TestKnowingWhatWeAreRunning:
    def test_the_commit_is_baked_into_the_image(self):
        dockerfile = (Path(__file__).resolve().parent.parent / "scripts" /
                      "Dockerfile.ainode").read_text()
        assert "ARG AINODE_GIT_SHA" in dockerfile
        assert "ENV AINODE_GIT_SHA" in dockerfile

    def test_the_build_script_passes_the_real_one(self):
        build = (Path(__file__).resolve().parent.parent / "scripts" /
                 "build-ainode-image.sh").read_text()
        assert "git rev-parse HEAD" in build

    def test_an_image_without_one_reports_nothing_rather_than_guessing(self,
                                                                      monkeypatch):
        monkeypatch.setenv("AINODE_GIT_SHA", "unknown")
        assert built_from() == ""
        monkeypatch.delenv("AINODE_GIT_SHA")
        assert built_from() == ""


class TestTheCheck:
    def _github(self, monkeypatch, head="b" * 40, ahead=3):
        import ainode.update.source as module

        def _get(url):
            if "/commits/" in url:
                return {"sha": head}
            if "/compare/" in url:
                return {"ahead_by": ahead, "commits": [
                    {"sha": f"{i}" * 40,
                     "commit": {"message": f"subject {i}\n\nbody"}}
                    for i in range(ahead)]}
            raise AssertionError(url)

        monkeypatch.setattr(module, "_get", _get)

    def test_being_behind_is_counted_and_listed(self, monkeypatch):
        self._github(monkeypatch, ahead=3)
        state = check_for_updates("bmetallica/ainode", "main", current="a" * 40)
        assert state.update_available is True
        assert state.behind == 3
        assert state.commits[0]["subject"] == "subject 2"      # newest first
        assert "\n" not in state.commits[0]["subject"]         # subject only

    def test_being_current_is_not_an_update(self, monkeypatch):
        self._github(monkeypatch, head="a" * 40)
        state = check_for_updates("bmetallica/ainode", "main", current="a" * 40)
        assert state.update_available is False
        assert state.error == ""

    def test_it_asks_about_the_fork_not_upstream(self, monkeypatch):
        asked = []
        import ainode.update.source as module

        monkeypatch.setattr(module, "_get",
                            lambda url: asked.append(url) or {"sha": "a" * 40})
        check_for_updates("bmetallica/ainode", "main", current="a" * 40)
        assert "bmetallica/ainode" in asked[0]
        assert "getainode" not in asked[0]

    def test_a_branch_other_than_main_is_honoured(self, monkeypatch):
        asked = []
        import ainode.update.source as module

        monkeypatch.setattr(module, "_get",
                            lambda url: asked.append(url) or {"sha": "a" * 40})
        check_for_updates("me/mine", "release", current="a" * 40)
        assert asked[0].endswith("/commits/release")

    def test_github_being_unreachable_is_reported_not_raised(self, monkeypatch):
        import ainode.update.source as module

        def _boom(url):
            raise OSError("no route to host")

        monkeypatch.setattr(module, "_get", _boom)
        state = check_for_updates("me/mine", "main", current="a" * 40)
        assert state.update_available is False
        assert "no route to host" in state.error

    def test_an_image_that_records_no_commit_says_so(self, monkeypatch):
        # Neither "up to date" nor "behind" would be true, so it says neither.
        self._github(monkeypatch)
        state = check_for_updates("me/mine", "main", current="")
        assert state.update_available is False
        assert "does not record the commit" in state.error

    def test_a_nonsense_repository_is_refused_before_the_network(self):
        state = check_for_updates("not-a-repo", "main", current="a" * 40)
        assert "owner/name" in state.error

    def test_a_known_difference_survives_a_failed_comparison(self, monkeypatch):
        # The head is known and differs; that alone is worth reporting even
        # if the commit list cannot be fetched.
        import ainode.update.source as module

        def _get(url):
            if "/commits/" in url:
                return {"sha": "b" * 40}
            raise OSError("rate limited")

        monkeypatch.setattr(module, "_get", _get)
        state = check_for_updates("me/mine", "main", current="a" * 40)
        assert state.update_available is True
        assert "rate limited" in state.error


class _Config:
    node_id = "head"
    source_repo = "bmetallica/ainode"
    source_branch = "main"
    source_dir = ""
    cluster_ssh_nodes = ["Spark2", "Spark3"]

    def save(self):
        type(self).saved = True


class TestFindingTheCheckout:
    def _checkout(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "update-cluster.sh").write_text("#!/bin/bash\n")
        return tmp_path

    def test_a_configured_checkout_is_found(self, tmp_path):
        config = _Config()
        config.source_dir = str(self._checkout(tmp_path))
        runner = UpdateRunner({"config": config})
        assert runner.source_dir() == tmp_path
        assert runner.why_not() == ""

    def test_a_directory_without_the_script_is_not_a_checkout(self, tmp_path):
        (tmp_path / ".git").mkdir()
        config = _Config()
        config.source_dir = str(tmp_path)
        assert UpdateRunner({"config": config}).source_dir() is None

    def test_in_a_container_the_missing_mount_is_named_with_the_fix(
            self, tmp_path, monkeypatch):
        # Failing at `git` would be true and useless. The operator needs the
        # two commands that make the button work.
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        config = _Config()
        config.source_dir = str(tmp_path / "nowhere")
        why = UpdateRunner({"config": config}).why_not()
        assert "ainode service install" in why
        assert CONTAINER_SOURCE_DIR in why
        assert "scripts/update-cluster.sh on the head" in why

    def test_outside_a_container_it_says_something_simpler(self, tmp_path,
                                                           monkeypatch):
        monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)
        config = _Config()
        config.source_dir = str(tmp_path / "nowhere")
        why = UpdateRunner({"config": config}).why_not()
        assert "No checkout at" in why
        assert "ainode service install" not in why


class TestTheUnitMountsIt:
    def test_the_mount_appears_when_a_checkout_is_there(self, tmp_path,
                                                        monkeypatch):
        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        monkeypatch.setenv("AINODE_SOURCE_DIR", str(tmp_path))
        from ainode.service.systemd import generate_unit_file

        assert f"{tmp_path}:{CONTAINER_SOURCE_DIR}" in generate_unit_file()

    def test_a_node_without_one_gets_no_mount(self, tmp_path, monkeypatch):
        # A mount that does not exist fails the container start, so a node
        # installed from an image alone must not be given one.
        monkeypatch.setenv("AINODE_SOURCE_DIR", str(tmp_path / "nothing"))
        monkeypatch.setattr("ainode.update.runner.DEFAULT_SOURCE_DIR",
                            str(tmp_path / "also-nothing"))
        from ainode.service.systemd import generate_unit_file

        assert CONTAINER_SOURCE_DIR not in generate_unit_file()


class _Req:
    def __init__(self, app, body=None, query=None):
        self.app = app
        self._body = body
        self.query = query or {}
        self.can_read_body = body is not None

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class TestTheSettings:
    def _app(self):
        return {"config": _Config()}

    def test_the_ssh_targets_are_saved(self):
        from ainode.update.api_routes import handle_put_settings

        app = self._app()
        body = json.loads(asyncio.run(handle_put_settings(
            _Req(app, {"cluster_ssh_nodes": ["Spark2", "Spark3"]}))).body)
        assert body["settings"]["cluster_ssh_nodes"] == ["Spark2", "Spark3"]

    def test_a_shell_metacharacter_in_an_ssh_target_is_refused(self):
        # It is pasted into an ssh command line by the update script, so this
        # would be a command on three machines rather than a typo.
        from ainode.update.api_routes import handle_put_settings

        resp = asyncio.run(handle_put_settings(
            _Req(self._app(), {"cluster_ssh_nodes": ["Spark2; rm -rf /"]})))
        assert resp.status == 400

    def test_duplicates_collapse_and_order_is_kept(self):
        from ainode.update.api_routes import handle_put_settings

        app = self._app()
        body = json.loads(asyncio.run(handle_put_settings(
            _Req(app, {"cluster_ssh_nodes": ["b", "a", "b"]}))).body)
        assert body["settings"]["cluster_ssh_nodes"] == ["b", "a"]

    def test_a_repository_that_is_not_owner_slash_name_is_refused(self):
        from ainode.update.api_routes import handle_put_settings

        resp = asyncio.run(handle_put_settings(
            _Req(self._app(), {"source_repo": "bmetallica"})))
        assert resp.status == 400

    def test_changing_the_repo_drops_the_cached_answer(self):
        # Otherwise the status line describes the previous repository.
        from ainode.update.api_routes import handle_put_settings

        app = self._app()
        app["_update_state"] = SourceState(repo="old/repo")
        asyncio.run(handle_put_settings(_Req(app, {"source_repo": "new/repo"})))
        assert "_update_state" not in app

    def test_the_default_repo_is_the_fork(self):
        from ainode.core.config import NodeConfig

        assert NodeConfig(node_id="n").source_repo == "bmetallica/ainode"


class TestRunningIt:
    def test_it_refuses_without_a_checkout(self):
        from ainode.update.api_routes import handle_run

        resp = asyncio.run(handle_run(_Req({"config": _Config()}, {})))
        assert resp.status == 409
        assert "checkout" in json.loads(resp.body)["error"].lower()

    def test_an_unusable_ssh_target_is_refused_before_anything_runs(self):
        from ainode.update.api_routes import handle_run

        resp = asyncio.run(handle_run(
            _Req({"config": _Config()}, {"nodes": ["a b"]})))
        assert resp.status == 400

    def test_it_pulls_before_it_builds(self, tmp_path, monkeypatch):
        ran = []

        class _Runner(UpdateRunner):
            def _step(self, command, cwd):
                ran.append(command)

        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "update-cluster.sh").write_text("#!/bin/bash\n")
        config = _Config()
        config.source_dir = str(tmp_path)
        runner = _Runner({"config": config})
        runner._run(["Spark2"], False, False)
        assert ran[0][:2] == ["git", "pull"]
        assert ran[1][0] == "scripts/update-cluster.sh"
        assert "--nodes" in ran[1] and "Spark2" in ran[1]

    def test_a_failure_is_recorded_rather_than_raised(self, tmp_path):
        class _Runner(UpdateRunner):
            def _step(self, command, cwd):
                raise RuntimeError("git exited with 1")

        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "update-cluster.sh").write_text("")
        config = _Config()
        config.source_dir = str(tmp_path)
        runner = _Runner({"config": config})
        runner._run([], False, False)
        assert runner.job["status"] == "failed"
        assert "git exited" in runner.job["error"]
        assert runner.job["running"] is False

    def test_two_at_once_is_refused(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "update-cluster.sh").write_text("")
        config = _Config()
        config.source_dir = str(tmp_path)
        runner = UpdateRunner({"config": config})
        runner.job["running"] = True
        assert runner.start(nodes=[])["status"] == 409

    def test_the_output_is_bounded(self, tmp_path):
        config = _Config()
        runner = UpdateRunner({"config": config})
        for i in range(1000):
            runner._say(f"line {i}")
        assert len(runner.job["lines"]) <= 400
        assert runner.job["lines"][-1] == "line 999"


class TestTheRoutes:
    def test_they_exist(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert {"/api/update/check", "/api/update/run", "/api/update/status",
                "/api/update/settings"} <= paths

    def test_a_manual_check_is_not_answered_from_the_cache(self):
        # "Check now" that returns a cached answer is not a check.
        import inspect

        from ainode.update import api_routes

        source = inspect.getsource(api_routes.handle_check)
        assert "not force" in source


class TestTheUI:
    def test_there_is_a_settings_section(self):
        assert 'data-section="updates"' in INDEX
        assert "renderConfigUpdates" in APP_JS

    def test_the_ssh_targets_are_editable(self):
        assert 'id="upd-nodes"' in APP_JS
        assert "Spark2, Spark3" in APP_JS

    def test_there_is_a_manual_check_button(self):
        assert 'id="upd-check"' in APP_JS
        assert "checkForSourceUpdate(true)" in APP_JS

    def test_the_hourly_check_is_wired(self):
        assert "60 * 60 * 1000" in APP_JS
        assert "sourceUpdateInterval" in APP_JS

    def test_the_dashboard_says_when_one_is_available(self):
        assert 'id="update-banner"' in INDEX
        assert "renderUpdateBanner" in APP_JS

    def test_the_banner_is_hidden_when_there_is_nothing_to_say(self):
        block = APP_JS.split("renderUpdateBanner() {")[1].split("\n  },")[0]
        assert "mount.style.display = 'none'" in block

    def test_the_confirmation_says_what_it_costs(self):
        # Every node restarts; loaded models do not survive it.
        block = APP_JS.split("async runSourceUpdate() {")[1][:1200]
        assert "loaded models are unloaded" in block
        assert "restarts LAST" in block

    def test_the_output_is_shown_while_it_runs(self):
        assert 'id="upd-log"' in APP_JS
        assert "pollUpdateJob" in APP_JS

    def test_the_poll_expects_its_own_disappearance(self):
        # The head restarts last, which from inside means this container
        # stops itself.
        block = APP_JS.split("async pollUpdateJob() {")[1]
        assert "this container stops" in block
