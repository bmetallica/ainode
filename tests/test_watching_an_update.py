"""An update in flight has to survive the browser being reloaded.

Reported from the cluster:

    ich musste während dem update prozess übers ui die seite neu laden und
    jetzt sehe ich nichtmehr wie weit er ist

The panel resumed from `this._updateJobPolling`, a variable in the tab that
started the run — so a reload left it looking idle while a twenty-minute
build went on without it. The server knew all along; nobody asked it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


@pytest_asyncio.fixture
async def client():
    app = create_app(config=NodeConfig(node_id="n1"), engine=None)
    async with TestClient(TestServer(app)) as c:
        yield c


class TestTheNodeSaysSo:
    @pytest.mark.asyncio
    async def test_status_carries_it(self, client):
        # On /api/status because every page polls that one; a dedicated
        # endpoint would be asked only by the page that already knows.
        data = await (await client.get("/api/status")).json()
        assert data["update_running"] is False

    @pytest.mark.asyncio
    async def test_it_is_true_while_one_runs(self, client):
        from ainode.update.api_routes import get_update_runner

        runner = get_update_runner(client.app)
        runner.job = {"running": True, "status": "running", "lines": ["x"]}
        data = await (await client.get("/api/status")).json()
        assert data["update_running"] is True

    @pytest.mark.asyncio
    async def test_a_broken_runner_is_not_a_broken_status(self, client):
        class _Broken:
            @property
            def job(self):
                raise RuntimeError("no")

        client.app["update_runner"] = _Broken()
        response = await client.get("/api/status")
        assert response.status == 200
        assert (await response.json())["update_running"] is False


class TestThePanelAsksTheServer:
    def test_it_resumes_without_a_local_flag(self):
        assert "resumeUpdateJob" in APP_JS
        # Unconditionally: the flag it used to check is the thing a reload
        # throws away.
        assert "if (this._updateJobPolling) this.pollUpdateJob();" not in APP_JS

    def test_resuming_shows_the_output_it_missed(self):
        body = APP_JS.split("async resumeUpdateJob()")[1][:900]
        assert "/api/update/status" in body
        assert "upd-log" in body

    def test_it_does_not_start_a_second_poller(self):
        body = APP_JS.split("async resumeUpdateJob()")[1][:900]
        assert "!this._updateJobPolling" in body

    def test_an_idle_updater_shows_nothing(self):
        body = APP_JS.split("async resumeUpdateJob()")[1][:900]
        assert "status === 'idle'" in body


class TestTheBannerSaysItIsRunning:
    def test_the_node_state_reaches_it(self):
        assert "update_running" in APP_JS
        assert "this.state.updateRunning" in APP_JS

    def test_running_outranks_the_offer_of_an_update(self):
        # "3 commits behind" is not the thing to say while a build is running.
        banner = APP_JS.split("renderUpdateBanner() {")[1][:1600]
        assert banner.index("updateRunning") < banner.index("update_available")

    def test_there_is_still_only_one_banner_function(self):
        # A second key of the same name in an object literal silently wins,
        # and the loser is whichever one was written first.
        assert APP_JS.count("  renderUpdateBanner() {") == 1


class TestItIsAlsoInTheLog:
    """So it can be followed without a browser at all."""

    def test_every_line_reaches_the_node_log(self):
        import inspect

        from ainode.update.runner import UpdateRunner

        assert 'logger.info("update | %s", line)' in inspect.getsource(
            UpdateRunner._say)

    def test_the_readme_says_how_to_watch_it(self):
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        assert "docker logs -f ainode" in readme
        assert "/api/update/status" in readme


class TestTellingOneRunFromTheNextOne:
    """A stored last-run read from a shell says "failed" with no context, and
    a failure that has since been fixed looks exactly like a current one.
    Reported after exactly that confusion on the cluster."""

    def _summary(self, **job):
        from ainode.update.api_routes import _summary

        return _summary(job)

    def test_nothing_has_run(self):
        assert "no update has run" in self._summary(status="idle")

    def test_a_finished_run_carries_its_time(self):
        line = self._summary(status="done", finished_at=1790000000.0,
                             to_sha="9e06f61")
        assert line.startswith("done 20")
        assert "9e06f61" in line

    def test_a_failure_carries_its_reason(self):
        line = self._summary(status="failed", finished_at=1790000000.0,
                             error="scripts/update-cluster.sh exited with 1")
        assert "failed" in line and "exited with 1" in line

    def test_a_running_one_says_how_long(self):
        import time as _time

        line = self._summary(status="running", running=True,
                             started_at=_time.time() - 600)
        assert "running for 10 min" in line

    def test_the_endpoint_includes_it(self):
        import inspect

        from ainode.update import api_routes

        assert '"summary"' in inspect.getsource(api_routes.handle_status)

    def test_the_readme_recipe_uses_it(self):
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        assert '["summary"]' in readme
