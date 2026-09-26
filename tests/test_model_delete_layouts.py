"""Deleting a model has to know every layout the scan knows.

A download that was aborted part-way could be listed in the UI but not
removed through it:

    Delete failed: Model not downloaded: nvidia/Gemma-4-26B-A4B-NVFP4

list_downloaded() scans four directory layouts; the two delete paths knew
only one of them. Nothing was wrong with the model or the disk — the two
halves simply disagreed about where weights live.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from aiohttp import web

from ainode.models.api_routes import handle_delete_repo
from ainode.models.registry import CatalogAggregator, ModelManager

REPO = "nvidia/Gemma-4-26B-A4B-NVFP4"
SLUG = "nvidia--Gemma-4-26B-A4B-NVFP4"
HF_SLUG = f"models--{SLUG}"

LAYOUTS = {
    "direct download": Path(SLUG),
    "hf cache, flat": Path(HF_SLUG),
    "hf cache, hub": Path("hub") / HF_SLUG,
    "hf cache, HF_HOME": Path("hf-cache") / "hub" / HF_SLUG,
}


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    monkeypatch.setattr(CatalogAggregator, "CACHE_FILE", tmp_path / "catalog-cache.json")
    monkeypatch.setattr(CatalogAggregator, "fetch", lambda self, force_refresh=False: [])


@pytest.fixture
def manager(tmp_path) -> ModelManager:
    models = tmp_path / "models"
    models.mkdir()
    return ModelManager(models_dir=models)


#: Decimal, because _dir_size_gb reports decimal GB — the same unit the Hub
#: and the planner use. The name stays GB-shaped on purpose.
GB = 10 ** 9
GIB = GB          # historical alias; the tests below read either


def _place(manager: ModelManager, layout: Path, size: int = GIB) -> Path:
    """A copy of the model, weighing `size` bytes.

    Sparse: the sizes have to be GB-scale for the two-decimal rounding in the
    response to carry any information, and nobody needs a test writing a real
    gigabyte of zeros.
    """
    target = manager.models_dir / layout
    target.mkdir(parents=True)
    weights = target / "model.safetensors"
    weights.touch()
    os.truncate(weights, size)
    return target


class TestEveryLayoutIsFound:
    @pytest.mark.parametrize("name", sorted(LAYOUTS))
    def test_a_copy_in_any_layout_is_locatable(self, manager, name):
        placed = _place(manager, LAYOUTS[name])
        assert manager.model_dirs_for_repo(REPO) == [placed]

    def test_nothing_on_disk_finds_nothing(self, manager):
        assert manager.model_dirs_for_repo(REPO) == []

    def test_copies_in_several_layouts_are_all_found(self, manager):
        """Downloading the same repo twice by different routes leaves two
        copies, and freeing the disk means removing both."""
        placed = [_place(manager, LAYOUTS[n]) for n in sorted(LAYOUTS)]
        assert sorted(manager.model_dirs_for_repo(REPO)) == sorted(placed)

    def test_a_file_of_that_name_is_not_a_model(self, manager):
        (manager.models_dir / SLUG).write_text("not a directory")
        assert manager.model_dirs_for_repo(REPO) == []

    def test_it_agrees_with_the_scan(self, manager):
        """The invariant the bug broke: anything list_downloaded() reports is
        something the delete path can find."""
        for layout in LAYOUTS.values():
            _place(manager, layout)
        listed = {entry["hf_repo"] for entry in manager.list_downloaded()}
        for hf_repo in listed:
            assert manager.model_dirs_for_repo(hf_repo), hf_repo


class TestTheEndpoint:
    async def _post(self, manager, body) -> tuple[int, dict]:
        app = web.Application()
        app["model_manager"] = manager
        app.router.add_post("/api/models/delete-repo", handle_delete_repo)
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/models/delete-repo", json=body)
            return response.status, await response.json()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", sorted(LAYOUTS))
    async def test_it_deletes_whatever_layout_is_on_disk(self, manager, name):
        placed = _place(manager, LAYOUTS[name])
        status, payload = await self._post(manager, {"hf_repo": REPO})
        assert status == 200, payload
        assert not placed.exists()
        assert payload["freed_gb"] == 1.0

    @pytest.mark.asyncio
    async def test_an_aborted_download_is_deletable(self, manager):
        """The reported case: a cancelled pull left a cache directory holding
        only a partial blob. It has no config.json and is not loadable, which
        is exactly why the operator wants it gone."""
        target = manager.models_dir / "hub" / HF_SLUG / "blobs"
        target.mkdir(parents=True)
        (target / "abc123.incomplete").write_bytes(b"\x00" * 1024)
        status, payload = await self._post(manager, {"hf_repo": REPO})
        assert status == 200, payload
        assert not (manager.models_dir / "hub" / HF_SLUG).exists()

    @pytest.mark.asyncio
    async def test_every_copy_goes_and_the_freed_size_is_the_total(self, manager):
        for index, name in enumerate(sorted(LAYOUTS)):
            _place(manager, LAYOUTS[name], size=GIB * (index + 1))
        status, payload = await self._post(manager, {"hf_repo": REPO})
        assert status == 200
        assert manager.model_dirs_for_repo(REPO) == []
        assert payload["freed_gb"] == 10.0       # 1 + 2 + 3 + 4
        assert len(payload["removed"]) == len(LAYOUTS)

    @pytest.mark.asyncio
    async def test_genuinely_absent_still_says_so(self, manager):
        status, payload = await self._post(manager, {"hf_repo": REPO})
        assert status == 404
        assert "not downloaded" in payload["error"]

    @pytest.mark.asyncio
    async def test_a_repo_id_without_an_org_is_refused(self, manager):
        status, _ = await self._post(manager, {"hf_repo": "Gemma-4"})
        assert status == 400


class TestCatalogDelete:
    def test_delete_model_finds_the_cache_layouts_too(self, manager):
        """The other delete path, DELETE /api/models/{id}, had the same gap."""
        model_id = next(iter(manager.get_catalog_map()))
        hf_repo = manager.get_catalog_map()[model_id].hf_repo
        slug = f"models--{ModelManager._repo_to_dirname(hf_repo)}"
        (manager.models_dir / "hub" / slug).mkdir(parents=True)
        assert manager.delete_model(model_id) is True
        assert not (manager.models_dir / "hub" / slug).exists()

    def test_it_still_reports_false_when_there_is_nothing(self, manager):
        model_id = next(iter(manager.get_catalog_map()))
        assert manager.delete_model(model_id) is False


class TestARenamedRepo:
    """The weights are there, under a name the UI never shows.

    demon-zombie/MiniMax-M2.7-AWQ-4bit now 307-redirects to
    et0dev/MiniMax-M2.7-AWQ-4bit. huggingface_hub follows the redirect, so an
    aborted download lands in models--et0dev--... while the catalog entry (and
    the delete dialog the operator is looking at) still says demon-zombie.
    The reply was

        Delete failed: Model not downloaded: demon-zombie/MiniMax-M2.7-AWQ-4bit

    about ~50 GB of partial download sitting on the disk.
    """

    OLD = "demon-zombie/MiniMax-M2.7-AWQ-4bit"
    NEW = "et0dev/MiniMax-M2.7-AWQ-4bit"

    def _place_new(self, manager):
        target = manager.models_dir / "hub" / "models--et0dev--MiniMax-M2.7-AWQ-4bit"
        target.mkdir(parents=True)
        (target / "model.safetensors").write_bytes(b"\x00" * 4096)
        return target

    def test_the_other_owner_is_found(self, manager):
        self._place_new(manager)
        assert manager.other_owners_on_disk(self.OLD) == [self.NEW]

    def test_the_repo_itself_is_not_reported_as_an_alias(self, manager):
        self._place_new(manager)
        assert manager.other_owners_on_disk(self.NEW) == []

    def test_an_unrelated_model_is_not_dragged_in(self, manager):
        self._place_new(manager)
        assert manager.other_owners_on_disk("someone/Qwen3.8-27B-NVFP4") == []

    def test_nothing_on_disk_finds_nothing(self, manager):
        assert manager.other_owners_on_disk(self.OLD) == []

    @pytest.mark.asyncio
    async def test_the_404_names_it(self, manager):
        self._place_new(manager)
        app = web.Application()
        app["model_manager"] = manager
        app.router.add_post("/api/models/delete-repo", handle_delete_repo)
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/models/delete-repo",
                                         json={"hf_repo": self.OLD})
            assert response.status == 404
            payload = await response.json()
        assert self.NEW in payload["error"]
        assert "renamed upstream" in payload["error"]
        assert payload["also_on_disk"] == [self.NEW]

    @pytest.mark.asyncio
    async def test_deleting_the_new_name_works(self, manager):
        placed = self._place_new(manager)
        app = web.Application()
        app["model_manager"] = manager
        app.router.add_post("/api/models/delete-repo", handle_delete_repo)
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/models/delete-repo",
                                         json={"hf_repo": self.NEW})
            assert response.status == 200, await response.json()
        assert not placed.exists()
