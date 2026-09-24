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
        assert data["mirrored"] is False

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
        assert data["mirrored"] is True
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
