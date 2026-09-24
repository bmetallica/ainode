"""Bringing a model in by hand, when the link cannot carry it.

    kann ich die modelle eigentlich auch irgendwie auf einem anderen pc
    downloaden und dann nach ainode bringen (also z.b. mit einem usbstick)?
    ich vermute das die LTE leitung hier am abbrechenden download schuld ist.

An 86 GB checkpoint over LTE is a bet on the connection staying up for hours,
and this deployment lost it twice — once losing the tokenizer files, once the
shards. A download that has to be perfect to be useful is the wrong shape for
a link that is not.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.core.config import NodeConfig

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


@pytest_asyncio.fixture
async def client(tmp_path):
    app = create_app(config=NodeConfig(node_id="n1", models_dir=str(tmp_path)),
                     engine=None)
    async with TestClient(TestServer(app)) as c:
        yield c


def _partial(tmp_path, repo="org/model"):
    """A repo as an interrupted download leaves it: weights, half a tokenizer."""
    directory = tmp_path / repo.replace("/", "--")
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "merges.txt").write_text("a b\n")
    (directory / "added_tokens.json").write_text("{}")
    (directory / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"x")
    return directory


class TestThePlan:
    @pytest.mark.asyncio
    async def test_it_names_what_is_missing(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("ainode.models.import_routes._hub_files",
                            lambda repo: [])
        _partial(tmp_path)
        data = await (await client.get(
            "/api/models/import/plan?hf_repo=org/model")).json()
        missing = [f["path"] for f in data["missing"]]
        assert "model-00002-of-00002.safetensors" in missing
        assert "tokenizer.json" in missing or "vocab.json" in missing

    @pytest.mark.asyncio
    async def test_it_does_not_list_what_is_there(self, client, tmp_path,
                                                  monkeypatch):
        monkeypatch.setattr("ainode.models.import_routes._hub_files",
                            lambda repo: [])
        _partial(tmp_path)
        data = await (await client.get(
            "/api/models/import/plan?hf_repo=org/model")).json()
        assert "model-00001-of-00002.safetensors" not in \
            [f["path"] for f in data["missing"]]

    @pytest.mark.asyncio
    async def test_every_file_carries_a_url_to_fetch_it_from(self, client,
                                                             tmp_path, monkeypatch):
        # The whole point: a browser on another machine can be pointed at it.
        monkeypatch.setattr("ainode.models.import_routes._hub_files",
                            lambda repo: [])
        _partial(tmp_path)
        data = await (await client.get(
            "/api/models/import/plan?hf_repo=org/model")).json()
        for entry in data["files"]:
            assert entry["url"].startswith(
                "https://huggingface.co/org/model/resolve/main/")

    @pytest.mark.asyncio
    async def test_it_says_which_list_this_is(self, client, tmp_path, monkeypatch):
        # A local list cannot know about files the index does not mention,
        # and saying so is the difference between a plan and a guess.
        monkeypatch.setattr("ainode.models.import_routes._hub_files",
                            lambda repo: [])
        _partial(tmp_path)
        data = await (await client.get(
            "/api/models/import/plan?hf_repo=org/model")).json()
        assert data["source"] == "local"

    @pytest.mark.asyncio
    async def test_the_hub_wins_when_it_answers(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ainode.models.import_routes._hub_files",
            lambda repo: [{"path": "something-only-the-hub-knows.bin",
                           "size": 1234}])
        _partial(tmp_path)
        data = await (await client.get(
            "/api/models/import/plan?hf_repo=org/model")).json()
        assert data["source"] == "hub"
        assert data["missing"][0]["path"] == "something-only-the-hub-knows.bin"
        assert data["missing_bytes"] == 1234

    @pytest.mark.asyncio
    async def test_a_repo_id_is_required(self, client):
        assert (await client.get("/api/models/import/plan?hf_repo=nope")).status == 400


class TestTheUpload:
    async def _upload(self, client, repo, path, payload):
        import aiohttp

        form = aiohttp.FormData()
        form.add_field("hf_repo", repo)
        form.add_field("path", path)
        form.add_field("file", payload, filename=path.split("/")[-1])
        return await client.post("/api/models/import/upload", data=form)

    @pytest.mark.asyncio
    async def test_a_file_lands_where_the_downloader_would_put_it(
            self, client, tmp_path):
        resp = await self._upload(client, "org/model", "tokenizer.json", b"{}")
        assert resp.status == 200
        assert (tmp_path / "org--model" / "tokenizer.json").read_bytes() == b"{}"

    @pytest.mark.asyncio
    async def test_a_nested_path_is_kept(self, client, tmp_path):
        await self._upload(client, "org/model", "transformer/config.json", b"{}")
        assert (tmp_path / "org--model" / "transformer" / "config.json").is_file()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["../escape.json", "/etc/passwd",
                                      "a/../../b.json"])
    async def test_a_path_that_leaves_the_directory_is_refused(
            self, client, tmp_path, path):
        resp = await self._upload(client, "org/model", path, b"x")
        assert resp.status == 400
        assert not (tmp_path.parent / "escape.json").exists()

    @pytest.mark.asyncio
    async def test_nothing_partial_is_left_behind(self, client, tmp_path):
        # Written to a temporary name and renamed, so an interrupted upload
        # never leaves a file that looks whole.
        import inspect

        from ainode.models import import_routes

        source = inspect.getsource(import_routes.handle_upload)
        assert ".ainode-upload" in source
        assert "tmp.replace(target)" in source


class TestFinishing:
    @pytest.mark.asyncio
    async def test_an_incomplete_import_is_not_spread(self, client, tmp_path):
        _partial(tmp_path)
        data = await (await client.post("/api/models/import/finish",
                                        json={"hf_repo": "org/model"})).json()
        assert data["complete"] is False
        assert data["mirroring"] is False

    @pytest.mark.asyncio
    async def test_a_complete_one_is(self, client, tmp_path, monkeypatch):
        directory = _partial(tmp_path)
        (directory / "model-00002-of-00002.safetensors").write_bytes(b"x")
        (directory / "tokenizer.json").write_text("{}")
        sent = {}

        async def _mirror(app, model, job):
            sent["model"] = model
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/finish",
                                        json={"hf_repo": "org/model"})).json()
        assert data["complete"] is True
        # A job, not a result: the transfer outlives the request now.
        assert data["mirroring"] is True and data["job_id"]
        await asyncio.sleep(0)
        assert sent["model"] == "org/model"

    @pytest.mark.asyncio
    async def test_nothing_imported_is_a_404(self, client):
        assert (await client.post("/api/models/import/finish",
                                  json={"hf_repo": "org/nothing"})).status == 404


class TestTheUI:
    def test_the_button_is_in_the_models_view(self):
        assert "import-model-btn" in APP_JS
        assert "Import from files" in APP_JS

    def test_it_lists_the_missing_files_as_links(self):
        assert "renderImportPanel" in APP_JS
        assert "/api/models/import/plan" in APP_JS

    def test_it_uploads_one_file_at_a_time(self):
        body = APP_JS.split("async uploadImportFiles")[1][:1800]
        assert "/api/models/import/upload" in body
        assert "FormData" in body

    def test_it_maps_a_picked_file_back_to_its_path_in_the_repo(self):
        # A file chosen from a subdirectory has to land in that subdirectory.
        body = APP_JS.split("async uploadImportFiles")[1][:1800]
        assert "byName[file.name]" in body

    def test_it_says_when_the_list_came_from_disk(self):
        assert "the Hub could not be reached" in APP_JS or \
            "The Hub could not be reached" in APP_JS

    def test_finishing_distributes(self):
        assert "/api/models/import/finish" in APP_JS


class TestTheDropDirectory:
    """A browser upload is right for twenty small files and wrong for 43 GB
    of shards. A directory and a file manager are right for that, and every
    operating system already ships both.

        baue es bitte so das ich das heruntergeladene modell auf dem head
        auch einfach unter /model-import ablegen kann und er es von dort
        zieht
    """

    @pytest_asyncio.fixture
    async def dropped(self, tmp_path, monkeypatch):
        drop = tmp_path / "drop"
        (drop / "org--model").mkdir(parents=True)
        (drop / "org--model" / "config.json").write_text("{}")
        (drop / "org--model" / "tokenizer.json").write_text("{}")
        (drop / "org--model" / "model.safetensors").write_bytes(b"x" * 10)
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(drop))
        return drop

    @pytest.mark.asyncio
    async def test_it_lists_what_is_lying_there(self, client, dropped):
        data = await (await client.get("/api/models/import/dropbox")).json()
        assert data["exists"] is True
        entry = data["entries"][0]
        assert entry["repo"] == "org/model"
        assert entry["files"] == 3

    @pytest.mark.asyncio
    async def test_a_nested_org_directory_is_understood_too(
            self, client, tmp_path, monkeypatch):
        drop = tmp_path / "drop2"
        (drop / "org" / "model").mkdir(parents=True)
        (drop / "org" / "model" / "config.json").write_text("{}")
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(drop))
        data = await (await client.get("/api/models/import/dropbox")).json()
        assert data["entries"][0]["repo"] == "org/model"

    @pytest.mark.asyncio
    async def test_a_missing_directory_says_how_to_get_one(
            self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(tmp_path / "nowhere"))
        data = await (await client.get("/api/models/import/dropbox")).json()
        assert data["exists"] is False
        assert "re-run the installer" in data["hint"]

    @pytest.mark.asyncio
    async def test_taking_it_in_moves_the_files(self, client, tmp_path, dropped,
                                                monkeypatch):
        async def _mirror(app, model, job):
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/dropbox",
                                        json={"name": "org--model"})).json()
        assert data["moved"] == 3
        assert (tmp_path / "org--model" / "tokenizer.json").is_file()
        # Moved, not copied: 129 GB should not exist twice.
        assert not (dropped / "org--model" / "tokenizer.json").exists()

    @pytest.mark.asyncio
    async def test_a_complete_one_is_distributed(self, client, dropped, monkeypatch):
        sent = {}

        async def _mirror(app, model, job):
            sent["model"] = model
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/dropbox",
                                        json={"name": "org--model"})).json()
        assert data["complete"] is True and data["mirroring"] is True
        await asyncio.sleep(0)
        assert sent["model"] == "org/model"

    @pytest.mark.asyncio
    async def test_an_incomplete_one_is_not(self, client, tmp_path, monkeypatch):
        drop = tmp_path / "drop3"
        (drop / "org--half").mkdir(parents=True)
        (drop / "org--half" / "config.json").write_text("{}")
        (drop / "org--half" / "merges.txt").write_text("a b\n")
        (drop / "org--half" / "model.safetensors").write_bytes(b"x")
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(drop))
        data = await (await client.post("/api/models/import/dropbox",
                                        json={"name": "org--half"})).json()
        assert data["complete"] is False
        assert data["mirroring"] is False

    @pytest.mark.asyncio
    async def test_a_name_that_says_nothing_about_the_repo_is_refused(
            self, client, tmp_path, monkeypatch):
        drop = tmp_path / "drop4"
        (drop / "justafolder").mkdir(parents=True)
        (drop / "justafolder" / "config.json").write_text("{}")
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(drop))
        resp = await client.post("/api/models/import/dropbox",
                                 json={"name": "justafolder"})
        assert resp.status == 400
        assert "org--name" in (await resp.json())["error"]

    @pytest.mark.asyncio
    async def test_a_name_that_leaves_the_directory_is_refused(
            self, client, dropped):
        resp = await client.post("/api/models/import/dropbox",
                                 json={"name": "../etc", "hf_repo": "a/b"})
        assert resp.status == 400


class TestTheImportClearsWhatItReplaced:
    """Reported from the cluster, after the model was carried in by hand:

        sparkarena/Minimax-M3-v0-NVFP4-REAP50: a download of this model was
        interrupted: 2 file(s) are still partial … das modell wurde über den
        /model-import weg importiert

    A Xet transfer stages under the file's content id, so those stubs survive
    an import that makes them meaningless — and they are not small. The
    import removes them, because it is the operator declaring what the
    directory holds.
    """

    @pytest_asyncio.fixture
    async def dropped_over_a_dead_transfer(self, tmp_path, monkeypatch):
        drop = tmp_path / "drop-x"
        (drop / "org--model").mkdir(parents=True)
        for name in ("config.json", "tokenizer.json"):
            (drop / "org--model" / name).write_text("{}")
        (drop / "org--model" / "model.safetensors").write_bytes(b"x" * 10)
        monkeypatch.setenv("AINODE_IMPORT_DIR", str(drop))
        # What the interrupted download left in the target directory.
        stubs = tmp_path / "org--model" / ".cache" / "huggingface" / "download"
        stubs.mkdir(parents=True)
        (stubs / "0k4AjklGyGCyIWbHx36RsIjxBNg=.3565b5.f12932e7.incomplete"
         ).write_bytes(b"x" * 1000)
        (stubs / "zAulchGaJLrQXYGB2u_NDxralws=.ecf58a.e1abaf54.incomplete"
         ).write_bytes(b"x" * 2000)
        return drop

    @pytest.mark.asyncio
    async def test_the_stubs_are_gone(self, client, tmp_path, monkeypatch,
                                      dropped_over_a_dead_transfer):
        async def _mirror(app, model, job):
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/dropbox",
                                        json={"name": "org--model"})).json()
        assert data["cleared_partials"] == 2
        assert data["reclaimed_bytes"] == 3000
        assert not list((tmp_path / "org--model").rglob("*.incomplete"))

    @pytest.mark.asyncio
    async def test_the_model_is_then_complete_and_distributed(
            self, client, tmp_path, monkeypatch,
            dropped_over_a_dead_transfer):
        async def _mirror(app, model, job):
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/dropbox",
                                        json={"name": "org--model"})).json()
        assert data["complete"] is True and data["mirroring"] is True

    @pytest.mark.asyncio
    async def test_finishing_again_clears_them_too(
            self, client, tmp_path, monkeypatch):
        # The way out for a model that was already imported before this
        # existed: press Finish again rather than delete and re-fetch.
        target = tmp_path / "org--already"
        target.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json"):
            (target / name).write_text("{}")
        (target / "model.safetensors").write_bytes(b"x")
        stubs = target / ".cache" / "huggingface" / "download"
        stubs.mkdir(parents=True)
        (stubs / "abc=.def.ghi.incomplete").write_bytes(b"x" * 64)

        async def _mirror(app, model, job):
            job["mirror"] = {}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/finish",
                                        json={"hf_repo": "org/already"})).json()
        assert data["cleared_partials"] == 1
        assert data["complete"] is True


class TestTheUISaysWhatWasReclaimed:
    SOURCE = (Path(__file__).resolve().parent.parent
              / "ainode" / "web" / "static" / "js" / "app.js").read_text()

    def test_it_has_a_line_for_it(self):
        assert "clearedNote" in self.SOURCE
        assert "cleared_partials" in self.SOURCE

    def test_it_names_the_space(self):
        assert "reclaimed_bytes" in self.SOURCE


class TestTheMirrorOutlivesTheRequest:
    """Reported from the cluster, on a 129 GB import:

        curl -s localhost:3000/api/models/import/finish … ^C

    The handler awaited the whole push to the peers before answering, so the
    request hung for as long as the transfer took, and aiohttp cancels a
    handler whose client has gone — pressing Ctrl-C took the transfer with it.
    The download path has had a job for this all along. So does this one now.
    """

    @pytest_asyncio.fixture
    async def ready(self, tmp_path, monkeypatch):
        directory = tmp_path / "org--model"
        directory.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json"):
            (directory / name).write_text("{}")
        (directory / "model.safetensors").write_bytes(b"x")
        return directory

    @pytest.mark.asyncio
    async def test_the_answer_comes_back_before_the_transfer_does(
            self, client, ready, monkeypatch):
        started = asyncio.Event()
        release = asyncio.Event()

        async def _mirror(app, model, job):
            started.set()
            await release.wait()
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        resp = await client.post("/api/models/import/finish",
                                 json={"hf_repo": "org/model"})
        data = await resp.json()
        # Answered while the transfer is still in flight — the whole point.
        assert resp.status == 202
        assert data["mirroring"] is True and data["job_id"]
        await asyncio.wait_for(started.wait(), timeout=2)
        release.set()

    @pytest.mark.asyncio
    async def test_the_job_is_visible_while_it_runs(
            self, client, ready, monkeypatch):
        release = asyncio.Event()

        async def _mirror(app, model, job):
            job["mirror"] = {"n2": "sending"}
            await release.wait()
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/finish",
                                        json={"hf_repo": "org/model"})).json()
        await asyncio.sleep(0)
        active = await (await client.get("/api/models/downloads/active")).json()
        mine = [j for j in active["jobs"] if j["job_id"] == data["job_id"]]
        assert mine and mine[0]["status"] == "mirroring"
        assert mine[0]["model_id"] == "org/model"
        release.set()
        await asyncio.sleep(0)
        after = await (await client.get("/api/models/downloads/active")).json()
        mine = [j for j in after["jobs"] if j["job_id"] == data["job_id"]]
        assert mine[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_the_job_is_json(self, client, ready, monkeypatch):
        # A Task on the job dict would make this route 500 — the reference
        # that keeps the task alive belongs somewhere else.
        async def _mirror(app, model, job):
            job["mirror"] = {"n2": "ok"}

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        await client.post("/api/models/import/finish",
                          json={"hf_repo": "org/model"})
        assert (await client.get("/api/models/downloads/active")).status == 200

    @pytest.mark.asyncio
    async def test_a_failure_lands_on_the_job_not_on_the_import(
            self, client, ready, monkeypatch):
        async def _mirror(app, model, job):
            raise RuntimeError("node 2 is down")

        monkeypatch.setattr("ainode.models.api_routes._mirror_after_download",
                            _mirror)
        data = await (await client.post("/api/models/import/finish",
                                        json={"hf_repo": "org/model"})).json()
        assert data["ok"] is True          # the import itself succeeded
        await asyncio.sleep(0)
        active = await (await client.get("/api/models/downloads/active")).json()
        mine = [j for j in active["jobs"] if j["job_id"] == data["job_id"]][0]
        assert mine["status"] == "failed"
        assert "node 2 is down" in mine["error"]

    def test_the_ui_follows_the_job(self):
        source = (Path(__file__).resolve().parent.parent / "ainode" / "web"
                  / "static" / "js" / "app.js").read_text()
        assert "watchImportMirror" in source
        assert "downloads/active" in source


class TestTheMountExists:
    """The drop directory is on the host and AINode is in a container."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_the_installer_mounts_it(self):
        install = (self.ROOT / "scripts" / "install.sh").read_text()
        assert "IMPORT_MOUNT" in install
        assert "${IMPORT_DIR}:${IMPORT_DIR}" in install

    def test_it_mounts_it_at_the_same_path(self):
        # So the name an operator types on the host is the name the product
        # uses. A different path inside would need explaining forever.
        from ainode.models.import_routes import DROP_DIR

        assert DROP_DIR == "/model-import"

    def test_only_when_it_exists(self):
        # A bind mount whose source is missing fails the container start.
        install = (self.ROOT / "scripts" / "install.sh").read_text()
        block = install.split("IMPORT_MOUNT=\"\"")[1][:400]
        assert 'if [ -d "$IMPORT_DIR" ]' in block

    def test_the_other_renderer_agrees(self):
        import inspect

        from ainode.service import systemd

        source = inspect.getsource(systemd)
        assert "_import_mount" in source
        assert "{import_mount}" in source

    def test_the_container_is_told_where_it_is(self):
        install = (self.ROOT / "scripts" / "install.sh").read_text()
        assert "AINODE_IMPORT_DIR=${IMPORT_DIR}" in install
