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
        # command that makes the button work — and it is the installer, not
        # `ainode service install`: on the host that name is a wrapper into
        # this very container, which is what it says when you try.
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        config = _Config()
        config.source_dir = str(tmp_path / "nowhere")
        why = UpdateRunner({"config": config}).why_not()
        assert "scripts/install.sh | bash" in why
        assert CONTAINER_SOURCE_DIR in why
        assert "scripts/update-cluster.sh" in why

    def test_it_does_not_send_anyone_to_a_command_that_refuses(self, tmp_path,
                                                              monkeypatch):
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        config = _Config()
        config.source_dir = str(tmp_path / "nowhere")
        why = UpdateRunner({"config": config}).why_not()
        # It may MENTION it, to say why it is not the answer — but never as
        # the instruction.
        instruction = why.split("\n\n")[1] if "\n\n" in why else why
        assert "ainode service install" not in instruction

    def test_outside_a_container_it_says_something_simpler(self, tmp_path,
                                                           monkeypatch):
        monkeypatch.delenv("AINODE_IN_CONTAINER", raising=False)
        config = _Config()
        config.source_dir = str(tmp_path / "nowhere")
        why = UpdateRunner({"config": config}).why_not()
        assert "No checkout at" in why
        assert "ainode service install" not in why


class TestBothRenderersAgree:
    """There are two: ainode/service/systemd.py and the heredoc in
    scripts/install.sh. This deployment uses the installer — the CLI refuses
    to render a unit from inside the container, where there is no systemd bus
    — so a mount added to only one of them is a mount that does not exist."""

    INSTALL = (Path(__file__).resolve().parent.parent / "scripts" /
               "install.sh").read_text()

    def test_the_installer_mounts_the_checkout(self):
        assert f":{CONTAINER_SOURCE_DIR}" in self.INSTALL
        assert "SOURCE_MOUNT" in self.INSTALL

    def test_it_only_does_so_when_there_is_one(self):
        # A node installed from the image alone has no repository, and a
        # mount pointing at nothing fails the container start.
        block = self.INSTALL.split('SOURCE_MOUNT=""')[1][:500]
        assert "-d \"$candidate/.git\"" in block

    def test_it_honours_a_checkout_somewhere_else(self):
        assert "AINODE_SOURCE_DIR" in self.INSTALL

    def test_the_mount_reaches_the_exec_start(self):
        exec_start = self.INSTALL.split("EXEC_START=")[1].split('"\n\n')[0]
        assert "${SOURCE_MOUNT}" in exec_start


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
            def _step(self, command, cwd, **kw):
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
        # And not a second time, as root, inside the script.
        assert "--skip-pull" in ran[1]

    def test_a_failure_is_recorded_rather_than_raised(self, tmp_path):
        class _Runner(UpdateRunner):
            def _step(self, command, cwd, **kw):
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


class TestTheToolsTheUpdateNeeds:
    """The orchestrator image is a slim Python container. It had no `git`,
    so the button failed with

        [ainode] FAILED: [Errno 2] No such file or directory: 'git'

    which says nothing about what to do — and what to do is unusual, because
    an image without git cannot run the update that would give it one."""

    def _runner(self, tmp_path, monkeypatch, *, missing=("git",)):
        (tmp_path / ".git").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "update-cluster.sh").write_text("#!/bin/sh\n")
        config = _Config()
        config.source_dir = str(tmp_path)
        monkeypatch.setattr("ainode.update.runner.shutil.which",
                            lambda tool: None if tool in missing else f"/usr/bin/{tool}")
        monkeypatch.setattr("ainode.update.runner._last_run_path",
                            lambda app: tmp_path / "update-last.json")
        return UpdateRunner({"config": config})

    def test_the_image_is_checked_before_anything_runs(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path, monkeypatch)
        result = runner.start(nodes=["Spark2"])
        assert result["ok"] is False
        assert "git" in result["error"]

    def test_it_says_what_the_tool_was_for(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path, monkeypatch, missing=("ssh",))
        assert "reach the other nodes" in runner.tool_message(["ssh"])

    def test_in_a_container_it_names_the_one_time_bootstrap(self, tmp_path,
                                                            monkeypatch):
        monkeypatch.setenv("AINODE_IN_CONTAINER", "1")
        runner = self._runner(tmp_path, monkeypatch)
        message = runner.tool_message(["git"])
        assert "update-cluster.sh" in message
        assert "cannot run the update that would replace it" in message

    def test_ssh_is_only_required_when_there_are_peers(self, tmp_path,
                                                       monkeypatch):
        runner = self._runner(tmp_path, monkeypatch, missing=("ssh",))
        assert runner.missing_tools(nodes=[]) == []
        assert runner.missing_tools(nodes=["Spark2"]) == ["ssh"]

    def test_a_complete_image_is_not_blocked(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path, monkeypatch, missing=())
        assert runner.missing_tools(nodes=["Spark2"]) == []

    def test_the_panel_asks_the_same_question_as_the_button(self, tmp_path,
                                                            monkeypatch):
        # They disagreed: the panel asked only about the mount, so it offered
        # an update the run then refused.
        from ainode.update.api_routes import _blocked

        runner = self._runner(tmp_path, monkeypatch)
        assert "git" in _blocked(runner, _Config())


class TestTheResultSurvivesTheRestart:
    """The last thing a successful update does is stop this container. The
    process that reports the outcome is never the process that ran it."""

    def _runner(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.update.runner._last_run_path",
                            lambda app: tmp_path / "update-last.json")
        return UpdateRunner({"config": _Config()})

    def test_a_finished_run_is_written_to_disk(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path, monkeypatch)
        runner.job = {"running": True, "status": "running", "lines": ["x"],
                      "error": "", "finished_at": 0.0, "nodes": ["Spark2"]}
        runner._run([], False, False)          # no source dir → fails, persists
        saved = json.loads((tmp_path / "update-last.json").read_text())
        assert saved["status"] == "failed"

    def test_the_next_process_reads_it_back(self, tmp_path, monkeypatch):
        (tmp_path / "update-last.json").write_text(json.dumps(
            {"status": "done", "running": True, "lines": [], "nodes": ["Spark2"],
             "finished_at": 12.0, "error": ""}))
        runner = self._runner(tmp_path, monkeypatch)
        assert runner.job["status"] == "done"
        # Whatever it said when it was killed, it is not running now.
        assert runner.job["running"] is False
        assert runner.job["restored"] is True

    def test_nothing_on_disk_is_simply_idle(self, tmp_path, monkeypatch):
        runner = self._runner(tmp_path, monkeypatch)
        assert runner.job["status"] == "idle"
        assert runner.last_summary() is None

    def test_the_panel_says_what_happened(self):
        assert "last_run" in APP_JS
        assert "Last update finished" in APP_JS


class TestTheScriptInsideTheContainer:
    """scripts/update-cluster.sh was written to be run from the head's shell:
    it primes sudo and ends with `sudo systemctl restart ainode`. Neither
    exists in the container the UI runs it from."""

    SCRIPT = (Path(__file__).resolve().parent.parent / "scripts" /
              "update-cluster.sh").read_text()

    def test_it_knows_where_it_is(self):
        assert "AINODE_IN_CONTAINER" in self.SCRIPT
        assert "/.dockerenv" in self.SCRIPT

    def test_it_does_not_prime_a_sudo_that_is_not_there(self):
        primer = self.SCRIPT.split("Ask for the local password once")[0]
        guard = primer.rstrip().splitlines()[-2]
        assert "IN_CONTAINER -eq 0" in guard

    def test_the_head_restarts_through_docker(self):
        tail = self.SCRIPT.split("--- 8.")[1]
        assert "docker stop -t 30 ainode" in tail

    def test_it_refuses_to_self_stop_under_an_unswappable_unit(self):
        # There, a stop drops the head instead of swapping its image.
        tail = self.SCRIPT.split("--- 8.")[1]
        assert "AINODE_UNIT_SWAPPABLE" in tail
        assert tail.index("AINODE_UNIT_SWAPPABLE") < tail.index("docker stop")

    def test_the_self_stop_is_the_last_thing_it_does(self):
        # Everything above it — distributing, restarting peers, verifying —
        # needs this process alive.
        assert self.SCRIPT.rstrip().endswith("fi")
        assert self.SCRIPT.index("docker stop -t 30 ainode") > \
            self.SCRIPT.index("step \"Verifying\"")

    def test_it_pins_the_image_it_just_built(self):
        # The unit starts whatever image.env names. Without this the build is
        # discarded and an update that changed nothing looks like one that
        # worked.
        tail = self.SCRIPT.split("--- 8.")[1]
        assert "image.env" in tail

    def test_it_does_not_verify_a_node_it_has_not_restarted_yet(self):
        verify = self.SCRIPT.split('step "Verifying"')[1]
        head_check = verify.split('check_node "this node" ""')[0]
        assert "IN_CONTAINER -eq 0" in head_check


class TestTheImageCarriesGit:
    DOCKERFILE = (Path(__file__).resolve().parent.parent / "scripts" /
                  "Dockerfile.ainode").read_text()

    def test_git_is_installed(self):
        apt = self.DOCKERFILE.split("apt-get install")[1][:400]
        assert "git \\" in apt


class TestGitAndTheCheckoutsOwner:
    """The checkout belongs to the operator; the container runs as root.

        fatal: detected dubious ownership in repository at '/ainode-src'

    Git refusing to act on a repository owned by someone else. The obvious
    fix — declare it safe and pull as root — works and then leaves objects
    under .git owned by root, so the next pull the operator runs in their own
    shell fails on permissions, in a repository that was theirs until we
    touched it. So git runs as the owner instead.
    """

    def _env(self, tmp_path, uid=1000, gid=1000, euid=0):
        from ainode.update import runner as runner_module

        class _Stat:
            st_uid, st_gid = uid, gid

        return runner_module, _Stat

    def test_it_runs_git_as_the_owner(self, tmp_path, monkeypatch):
        module, stat = self._env(tmp_path)
        monkeypatch.setattr(module.os, "stat", lambda p: stat)
        monkeypatch.setattr(module.os, "geteuid", lambda: 0)
        kwargs, env = module._owner_of(tmp_path, {})
        assert kwargs == {"user": 1000, "group": 1000}

    def test_it_moves_home_out_of_roots(self, tmp_path, monkeypatch):
        # git reads its config from HOME before it does anything else, and
        # that uid cannot read /root.
        module, stat = self._env(tmp_path)
        monkeypatch.setattr(module.os, "stat", lambda p: stat)
        monkeypatch.setattr(module.os, "geteuid", lambda: 0)
        _, env = module._owner_of(tmp_path, {"HOME": "/root"})
        assert env["HOME"] != "/root"

    def test_a_checkout_we_own_is_left_alone(self, tmp_path, monkeypatch):
        module, stat = self._env(tmp_path, uid=0, gid=0)
        monkeypatch.setattr(module.os, "stat", lambda p: stat)
        monkeypatch.setattr(module.os, "geteuid", lambda: 0)
        kwargs, _ = module._owner_of(tmp_path, {})
        assert kwargs == {}

    def test_not_being_root_means_no_switching(self, tmp_path, monkeypatch):
        module, stat = self._env(tmp_path)
        monkeypatch.setattr(module.os, "stat", lambda p: stat)
        monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
        kwargs, _ = module._owner_of(tmp_path, {})
        assert kwargs == {}

    def test_safe_directory_is_set_either_way(self, tmp_path):
        from ainode.update.runner import _owner_of

        _, env = _owner_of(tmp_path, {})
        assert env["GIT_CONFIG_COUNT"] == "1"
        assert env["GIT_CONFIG_KEY_0"] == "safe.directory"
        assert env["GIT_CONFIG_VALUE_0"] == str(tmp_path)

    def test_it_does_not_write_a_global_config(self):
        # `git config --global` would outlive this update and apply to every
        # repository this container ever touches.
        import inspect

        from ainode.update import runner as runner_module

        source = inspect.getsource(runner_module)
        # The comment saying why is allowed to name it; a call is not.
        assert "git\", \"config" not in source
        assert "'git', 'config'" not in source
        assert '"config", "--global"' not in source

    def test_an_existing_git_config_env_is_not_clobbered(self, tmp_path):
        from ainode.update.runner import _owner_of

        _, env = _owner_of(tmp_path, {
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.name",
            "GIT_CONFIG_VALUE_0": "someone"})
        assert env["GIT_CONFIG_COUNT"] == "2"
        assert env["GIT_CONFIG_KEY_0"] == "user.name"
        assert env["GIT_CONFIG_KEY_1"] == "safe.directory"

    def test_the_pull_asks_for_it_and_the_build_does_not(self):
        # The build runs docker, which needs the socket, which is root's.
        import inspect

        from ainode.update.runner import UpdateRunner

        source = inspect.getsource(UpdateRunner._run)
        assert "as_owner=True" in source
        assert source.index("as_owner=True") < source.index("update-cluster.sh")


class TestTheCommitSurvivesTheBuild:
    """After an update that otherwise succeeded, the node said:

        "this image does not record the commit it was built from, so it
         cannot be compared — the next update fixes that"

    `git rev-parse HEAD` in build-ainode-image.sh runs as root against a
    checkout owned by the operator. Git's ownership check refused it, the
    `|| echo unknown` swallowed the failure, and the image was built without
    the one field the update check needs — so it could never report itself
    out of date again.
    """

    BUILD = (Path(__file__).resolve().parent.parent / "scripts" /
             "build-ainode-image.sh").read_text()

    def test_every_step_gets_the_safe_directory_exception(self, tmp_path):
        # Not only the step that calls git directly: the build script calls
        # it too, for the SHA it stamps into the image.
        import inspect

        from ainode.update.runner import UpdateRunner

        source = inspect.getsource(UpdateRunner._step)
        assert "_git_safe_env" in source
        # Before the as_owner branch, so it applies to both kinds of step.
        assert source.index("_git_safe_env") < source.index("if as_owner")

    def test_the_exception_names_the_checkout(self, tmp_path):
        from ainode.update.runner import _git_safe_env

        env = _git_safe_env(tmp_path, {})
        assert env["GIT_CONFIG_KEY_0"] == "safe.directory"
        assert env["GIT_CONFIG_VALUE_0"] == str(tmp_path)

    def test_it_appends_rather_than_clobbers(self, tmp_path):
        from ainode.update.runner import _git_safe_env

        env = _git_safe_env(tmp_path, {
            "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.name",
            "GIT_CONFIG_VALUE_0": "someone"})
        assert env["GIT_CONFIG_COUNT"] == "2"
        assert env["GIT_CONFIG_KEY_1"] == "safe.directory"

    def test_the_build_script_says_when_it_loses_the_sha(self):
        # Silently baking "unknown" is how this went unnoticed for a release.
        assert "could not read the commit for this build" in self.BUILD
        assert "safe.directory" in self.BUILD

    def test_it_still_builds_without_one(self):
        # A missing SHA costs the update check, not the image.
        block = self.BUILD.split('GIT_SHA="$(git rev-parse')[1][:600]
        assert 'GIT_SHA="unknown"' in block
        assert "exit 1" not in block

    def test_the_sha_reaches_the_build_arg(self):
        assert '--build-arg "AINODE_GIT_SHA=${GIT_SHA}"' in self.BUILD

    def test_the_build_prints_what_it_stamped(self):
        assert "source at ${GIT_SHA:0:7}" in self.BUILD
