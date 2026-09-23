"""A model placed on node 3 runs on node 3, and nowhere else.

Reported from the cluster:

    das embedding modell ist im profil für node3 eingestellt. es wird aber
    immerwieder auch auf node1 geladen ohne das ich das manuell auslöse, es
    läuft also dann immer auf node1 und node3.

Two independent ways for that to happen, and both were open:

  * applying the profile asked "is this model in the profile" before deciding
    what to stop, rather than "should this model run HERE". A copy on the head
    was therefore never unloaded — and because it stayed in the head's
    manifest, the boot replay brought it back for good.
  * an embeddings request that arrived at the head for a model nobody had
    loaded fell through to the local manager, which loads whatever it is
    asked for. One RAG call placed a model the operator had placed elsewhere.
"""

from __future__ import annotations

import asyncio

from ainode.profiles.apply import _entry_runs_here, apply_profile
from ainode.profiles.store import KIND_EMBEDDING, KIND_LLM, Profile, ProfileEntry


class _Config:
    node_id = "node1"
    node_name = "spark-1"


class _App(dict):
    pass


def _entry(model, kind, nodes):
    return ProfileEntry(model=model, kind=kind, node_ids=list(nodes))


class TestWhoseEntryIsIt:
    def test_an_unplaced_entry_belongs_to_whoever_applies_it(self):
        app = _App(config=_Config())
        assert _entry_runs_here(app, _entry("m", KIND_LLM, [])) is True

    def test_an_entry_naming_this_node_is_ours(self):
        app = _App(config=_Config())
        assert _entry_runs_here(app, _entry("m", KIND_LLM, ["node1"])) is True

    def test_an_entry_naming_another_node_is_not(self):
        app = _App(config=_Config())
        assert _entry_runs_here(app, _entry("m", KIND_LLM, ["node3"])) is False

    def test_a_distributed_entry_is_ours_when_we_are_in_it(self):
        # TP=2 across node1 and node3: both run it, neither may stop it.
        app = _App(config=_Config())
        assert _entry_runs_here(app, _entry("m", KIND_LLM, ["node3", "node1"])) is True


class _Embeddings:
    def __init__(self, loaded):
        self.loaded = list(loaded)
        self.saved = 0

    def list_loaded(self):
        return [{"id": m} for m in self.loaded]

    def is_loaded(self, model_id):
        return model_id in self.loaded

    def unload(self, model_id):
        self.loaded.remove(model_id)
        return True

    def save_manifest(self):
        self.saved += 1


class TestApplyingTheProfile:
    def _apply(self, app, profile):
        return asyncio.run(apply_profile(app, profile, wait=False))

    def test_a_local_copy_of_a_remote_entry_is_unloaded(self, monkeypatch):
        # The head has it loaded; the profile says node 3. Before this, the
        # head kept it AND node 3 was told to load it.
        embeddings = _Embeddings(["nomic-ai/nomic-embed-text-v1.5"])
        app = _App(config=_Config(), embedding_manager=embeddings,
                   instances=None)
        sent = []

        async def _fake(app_, entry):
            sent.append(entry.model)
            from ainode.profiles.apply import ApplyResult
            return ApplyResult(entry.model, "launched", True, node_id="node3")

        monkeypatch.setattr("ainode.profiles.apply._start_embedding_entry", _fake)
        monkeypatch.setattr("ainode.models.api_routes.save_instance_manifest",
                            lambda app_: None)
        profile = Profile(name="p", entries=[
            _entry("nomic-ai/nomic-embed-text-v1.5", KIND_EMBEDDING, ["node3"])])

        report = self._apply(app, profile)

        assert embeddings.loaded == []
        assert "nomic-ai/nomic-embed-text-v1.5" in report["stopped"]
        # And the unload was persisted, or the boot replay undoes it.
        assert embeddings.saved >= 1
        assert sent == ["nomic-ai/nomic-embed-text-v1.5"]

    def test_a_local_entry_is_left_alone(self, monkeypatch):
        embeddings = _Embeddings(["nomic-ai/nomic-embed-text-v1.5"])
        app = _App(config=_Config(), embedding_manager=embeddings, instances=None)

        async def _fake(app_, entry):
            from ainode.profiles.apply import ApplyResult
            return ApplyResult(entry.model, "already_running", True)

        monkeypatch.setattr("ainode.profiles.apply._start_embedding_entry", _fake)
        monkeypatch.setattr("ainode.models.api_routes.save_instance_manifest",
                            lambda app_: None)
        profile = Profile(name="p", entries=[
            _entry("nomic-ai/nomic-embed-text-v1.5", KIND_EMBEDDING, ["node1"])])

        report = self._apply(app, profile)
        assert embeddings.loaded == ["nomic-ai/nomic-embed-text-v1.5"]
        assert report["stopped"] == []


class TestAnLLMEntryGoesToItsNode:
    def test_a_solo_entry_for_a_peer_is_dispatched(self):
        import inspect

        from ainode.profiles import apply

        source = inspect.getsource(apply._start_llm_entry)
        # Not append_solo_instance, which starts it on whichever node applies
        # the profile — the head, for a captured cluster profile.
        assert "handle_cluster_load" in source
        assert source.index("_entry_target_node") < source.index(
            "from ainode.models.api_routes import append_solo_instance")


class _Manager:
    def __init__(self):
        self.loaded = []

    def is_loaded(self, model_id):
        return model_id in self.loaded

    def list_loaded(self):
        return [{"id": m} for m in self.loaded]


class _Store:
    def __init__(self, profile):
        self._profile = profile

    def default_profile(self):
        return self._profile


class TestAnImplicitLoadIsNotAPlacement:
    """`embed()` loads whatever it is asked for — which is right on a node
    that owns the model, and wrong on one the operator placed it away from."""

    MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def _app(self, nodes, cluster=None):
        from ainode.embeddings.api_routes import assigned_node

        profile = Profile(name="p", entries=[
            _entry(self.MODEL, KIND_EMBEDDING, nodes)])
        app = _App(config=_Config(), profiles=_Store(profile),
                   cluster_state=cluster)
        return assigned_node(app, self.MODEL)

    def test_an_entry_for_this_node_does_not_redirect(self):
        assert self._app(["node1"]) is None

    def test_an_unplaced_entry_does_not_redirect(self):
        assert self._app([]) is None

    def test_a_model_no_profile_mentions_does_not_redirect(self):
        from ainode.embeddings.api_routes import assigned_node

        app = _App(config=_Config(), profiles=_Store(Profile(name="p", entries=[])),
                   cluster_state=None)
        assert assigned_node(app, self.MODEL) is None

    def test_a_remote_entry_names_the_node(self):
        placed = self._app(["node3"])
        assert placed is not None and placed[0] == "node3"

    def test_an_absent_node_still_names_itself(self):
        # So the refusal can say which node, rather than loading here.
        node_id, host, _ = self._app(["node3"])
        assert (node_id, host) == ("node3", "")

    def test_the_request_path_consults_it(self):
        import inspect

        from ainode.embeddings import api_routes

        source = inspect.getsource(api_routes.handle_v1_embeddings)
        assert "assigned_node" in source
        # Only after asking whether anyone already has it loaded.
        assert source.index("peer_with_model") < source.index("assigned_node")
        assert "aembed" in source
        assert source.index("assigned_node") < source.index("aembed")
