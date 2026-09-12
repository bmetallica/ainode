"""Profiles: the stored description of what a node should be serving.

Covers the store (validation, default, persistence) and the API surface. The
convergence behaviour of apply() has its own file.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ainode.profiles.api_routes import (
    handle_apply_profile,
    handle_create_profile,
    handle_delete_profile,
    handle_get_profile,
    handle_list_profiles,
    handle_put_profile,
    handle_set_default,
)
from ainode.profiles.store import Profile, ProfileEntry, ProfileError, ProfileStore


@pytest.fixture()
def store(tmp_path):
    return ProfileStore(tmp_path / "profiles.json")


class _Req:
    """The parts of a request the profile handlers use."""

    def __init__(self, app, body=None, **match):
        self.app = app
        self._body = body
        self.match_info = match

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _call(handler, app, body=None, **match):
    return asyncio.run(handler(_Req(app, body, **match)))


def _json(resp):
    return json.loads(resp.body)


class TestProfileEntry:
    def test_an_entry_needs_a_model(self):
        with pytest.raises(ProfileError):
            ProfileEntry(model="  ")

    def test_launch_body_omits_what_was_never_set(self):
        # An unset field must stay out of the body so the catalog recipe and
        # the config defaults still apply at launch time.
        body = ProfileEntry(model="a/b").launch_body()
        assert body == {"model": "a/b"}

    def test_launch_body_carries_what_was_set(self):
        body = ProfileEntry(
            model="a/b", node_ids=["head", "n2"], gpu_memory_utilization=0.4,
            max_model_len=8192, kv_cache_dtype="fp8",
            extra_vllm_args=["--enable-prefix-caching"],
        ).launch_body()
        assert body["node_ids"] == ["head", "n2"]
        assert body["gpu_memory_utilization"] == 0.4
        assert body["max_model_len"] == 8192
        assert body["kv_cache_dtype"] == "fp8"
        assert body["extra_vllm_args"] == ["--enable-prefix-caching"]

    def test_out_of_range_memory_fraction_is_rejected(self):
        with pytest.raises(ProfileError):
            ProfileEntry(model="a/b", gpu_memory_utilization=1.5)

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ProfileError):
            ProfileEntry(model="a/b", kind="magic")

    def test_single_node_entry_is_not_distributed(self):
        assert ProfileEntry(model="a/b", node_ids=["head"]).is_distributed is False
        assert ProfileEntry(model="a/b", node_ids=["head", "n2"]).is_distributed


class TestProfile:
    def test_a_model_cannot_appear_twice(self):
        with pytest.raises(ProfileError):
            Profile(name="p", entries=[{"model": "a/b"}, {"model": "a/b"}])

    def test_name_must_be_boring(self):
        for bad in ("", "../etc", "a/b", "x" * 65):
            with pytest.raises(ProfileError):
                Profile(name=bad)
        Profile(name="Endausbau 1.0")

    def test_round_trips_through_a_dict(self):
        p = Profile(name="p", description="d",
                    entries=[{"model": "a/b", "node_ids": ["head", "n2"]}])
        again = Profile.from_dict(p.to_dict())
        assert again.entries[0].node_ids == ["head", "n2"]
        assert again.created_at == p.created_at


class TestProfileStore:
    def test_survives_a_restart(self, tmp_path):
        path = tmp_path / "profiles.json"
        ProfileStore(path).put(Profile(name="p", entries=[{"model": "a/b"}]))
        assert ProfileStore(path).get("p").entries[0].model == "a/b"

    def test_a_corrupt_file_does_not_break_boot(self, tmp_path):
        # A hand-edited profiles.json must not stop the node from serving.
        path = tmp_path / "profiles.json"
        path.write_text("{ not json")
        assert ProfileStore(path).names() == []

    def test_one_unusable_profile_does_not_hide_the_others(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"profiles": [
            {"name": "good", "entries": [{"model": "a/b"}]},
            {"name": "", "entries": []},
        ]}))
        assert ProfileStore(path).names() == ["good"]

    def test_default_must_name_an_existing_profile(self, store):
        with pytest.raises(ProfileError):
            store.set_default("ghost")

    def test_deleting_the_default_clears_it(self, store):
        store.put(Profile(name="p"))
        store.set_default("p")
        store.delete("p")
        assert store.default_name == ""
        assert store.default_profile() is None

    def test_default_survives_a_restart(self, tmp_path):
        path = tmp_path / "profiles.json"
        s = ProfileStore(path)
        s.put(Profile(name="p"))
        s.set_default("p")
        assert ProfileStore(path).default_profile().name == "p"

    def test_replacing_keeps_the_creation_time(self, store):
        first = store.put(Profile(name="p"))
        second = store.put(Profile(name="p", description="changed"))
        assert second.created_at == first.created_at


class TestProfileRoutes:
    @pytest.fixture()
    def app(self, store):
        return {"profiles": store, "config": None}

    def test_create_list_get_delete(self, app):
        resp = _call(handle_create_profile, app,
                     {"name": "p", "entries": [{"model": "a/b"}]})
        assert resp.status == 200

        listed = _json(_call(handle_list_profiles, app))
        assert [p["name"] for p in listed["profiles"]] == ["p"]
        assert listed["default"] == ""

        got = _json(_call(handle_get_profile, app, name="p"))
        assert got["profile"]["entries"][0]["model"] == "a/b"

        assert _call(handle_delete_profile, app, name="p").status == 200
        assert _call(handle_get_profile, app, name="p").status == 404

    def test_creating_twice_conflicts(self, app):
        _call(handle_create_profile, app, {"name": "p"})
        assert _call(handle_create_profile, app, {"name": "p"}).status == 409

    def test_a_bad_entry_is_a_400_not_a_500(self, app):
        resp = _call(handle_create_profile, app,
                     {"name": "p", "entries": [{"model": ""}]})
        assert resp.status == 400
        assert "model" in _json(resp)["error"]

    def test_put_replaces(self, app):
        _call(handle_create_profile, app, {"name": "p", "entries": [{"model": "a/b"}]})
        resp = _call(handle_put_profile, app,
                     {"name": "p", "entries": [{"model": "c/d"}]}, name="p")
        assert resp.status == 200
        assert _json(resp)["profile"]["entries"][0]["model"] == "c/d"

    def test_put_refuses_to_rename(self, app):
        _call(handle_create_profile, app, {"name": "p"})
        resp = _call(handle_put_profile, app, {"name": "other"}, name="p")
        assert resp.status == 400

    def test_setting_and_clearing_the_default(self, app):
        _call(handle_create_profile, app, {"name": "p"})
        assert _json(_call(handle_set_default, app, {}, name="p"))["default"] == "p"
        assert _json(_call(handle_set_default, app, {"default": False},
                           name="p"))["default"] == ""

    def test_default_for_an_unknown_profile_is_404(self, app):
        assert _call(handle_set_default, app, {}, name="ghost").status == 404

    def test_applying_an_unknown_profile_is_404(self, app):
        assert _call(handle_apply_profile, app, {}, name="ghost").status == 404

    def test_handlers_survive_a_missing_body(self, app):
        # The UI sends no body for "set default"; a handler must not 500 on it.
        _call(handle_create_profile, app, {"name": "p"})
        assert _call(handle_set_default, app, None, name="p").status == 200
