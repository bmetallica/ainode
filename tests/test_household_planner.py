"""The profile wizard's planner: several models sharing nodes.

wizzard.md §2–§3. Asked for:

    auf node1 laufen modell a und b, ich erhöhe den kvcache von modell a dann
    wird er im plan bei b weniger damit noch alles auf den node passt
"""

from __future__ import annotations

import math

import pytest

from ainode.planner.household import HouseholdNode, Item, solve


def _nodes(*budgets):
    return [HouseholdNode(node_id=f"n{i + 1}", name=f"Spark{i + 1}",
                          total_gb=128, budget_gb=b) for i, b in enumerate(budgets)]


class TestTheVerteiler:
    def test_more_for_a_is_less_for_b(self):
        """The user's example: A pinned larger, B automatic, same node."""
        def plan(a_cache):
            a = Item("a", "A", node_ids=["n1"], fixed_gb=30, mode="size",
                     pinned_gb=a_cache)
            b = Item("b", "B", node_ids=["n1"], fixed_gb=20, mode="auto", min_gb=2)
            conflicts = solve([a, b], _nodes(100))
            return a, b, conflicts

        a, b, conflicts = plan(10)
        assert not conflicts
        assert b.cache_gb == pytest.approx(100 - 30 - 20 - 10)
        a, b2, _ = plan(30)
        assert b2.cache_gb == pytest.approx(b.cache_gb - 20)
        assert a.fixed_gb + a.cache_gb + b2.fixed_gb + b2.cache_gb == pytest.approx(100)

    def test_equal_priority_shares_equally(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=10, mode="auto")
        b = Item("b", "B", node_ids=["n1"], fixed_gb=10, mode="auto")
        solve([a, b], _nodes(60))
        assert a.cache_gb == pytest.approx(20) and b.cache_gb == pytest.approx(20)

    def test_priority_weights_the_share(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=10, mode="auto", priority=3)
        b = Item("b", "B", node_ids=["n1"], fixed_gb=10, mode="auto", priority=1)
        solve([a, b], _nodes(50))
        assert a.cache_gb == pytest.approx(22.5) and b.cache_gb == pytest.approx(7.5)

    def test_a_model_across_two_nodes_couples_them(self):
        """C spans n1+n2, A only n1, B only n2. A pinned large on n1 limits C,
        and what C cannot take on n2 goes to B."""
        def plan(a_cache):
            a = Item("a", "A", node_ids=["n1"], fixed_gb=20, mode="size",
                     pinned_gb=a_cache)
            b = Item("b", "B", node_ids=["n2"], fixed_gb=20, mode="auto")
            c = Item("c", "C", node_ids=["n1", "n2"], fixed_gb=40, mode="auto")
            solve([a, b, c], _nodes(100, 100))
            return b, c

        b_small, c_small = plan(10)
        b_large, c_large = plan(30)
        assert c_large.cache_gb < c_small.cache_gb
        assert b_large.cache_gb > b_small.cache_gb
        # n1 is full, n2 is full.
        assert 20 + 30 + 40 + c_large.cache_gb == pytest.approx(100)
        assert 20 + 40 + c_large.cache_gb + b_large.cache_gb == pytest.approx(100)

    def test_a_ceiling_passes_the_rest_on(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=10, mode="auto", cap_gb=5)
        b = Item("b", "B", node_ids=["n1"], fixed_gb=10, mode="auto")
        solve([a, b], _nodes(60))
        assert a.cache_gb == pytest.approx(5) and b.cache_gb == pytest.approx(35)

    def test_image_and_embedding_are_fixed_blocks(self):
        img = Item("i", "flux", kind="image", node_ids=["n1"], fixed_gb=25)
        emb = Item("e", "bge", kind="embedding", node_ids=["n1"], fixed_gb=3)
        llm = Item("l", "L", node_ids=["n1"], fixed_gb=30, mode="auto")
        solve([img, emb, llm], _nodes(100))
        assert img.cache_gb == 0 and emb.cache_gb == 0
        assert llm.cache_gb == pytest.approx(42)

    def test_too_much_pinned_is_a_named_conflict(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=50, mode="size", pinned_gb=40)
        b = Item("b", "B", node_ids=["n1"], fixed_gb=30, mode="usage", pinned_gb=10)
        conflicts = solve([a, b], _nodes(100))
        assert len(conflicts) == 1
        assert "Spark1" in conflicts[0] and "30.0 GB too much" in conflicts[0]
        assert "(fixed)" in conflicts[0]

    def test_an_item_with_errors_takes_nothing(self):
        bad = Item("x", "X", node_ids=["n1"], fixed_gb=500, errors=["kaputt"])
        ok = Item("a", "A", node_ids=["n1"], fixed_gb=10, mode="auto")
        assert solve([bad, ok], _nodes(50)) == []
        assert ok.cache_gb == pytest.approx(40)

    def test_nothing_automatic_leaves_the_rest_free(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=10, mode="size", pinned_gb=5)
        solve([a], _nodes(50))
        assert a.cache_gb == 5
        assert math.isinf(a.cap_gb)


# -- a draft, planned against an app ------------------------------------------

from ainode.core.config import NodeConfig  # noqa: E402
from ainode.measure.store import MeasurementStore  # noqa: E402
from ainode.planner.facts import ModelFacts  # noqa: E402


def _facts(repo, weight_bytes=60e9, heads=32, kvh=8, layers=48):
    return ModelFacts(repo=repo, num_layers=layers, attention_layers=layers,
                      num_kv_heads=kvh, head_dim=128, num_attention_heads=heads,
                      torch_dtype="bfloat16", weight_bytes=int(weight_bytes),
                      max_position_embeddings=262144)


class _Node:
    def __init__(self, node_id, used_mib=0, baseline_mb=8000.0):
        self.node_id = node_id
        self.node_name = node_id.upper()
        self.status = "online"
        self.gpu_memory_total_mb = 122070        # 128 GB
        self.gpu_memory_used_mb = used_mib
        self.gpu_memory_gb = 128
        self.memory_limit_gb = 0.0
        self.baseline_used_mb = baseline_mb


@pytest.fixture
def app(tmp_path, monkeypatch):
    import ainode.planner.facts as facts_module
    from ainode.planner import api_routes

    catalog = {"org/a": _facts("org/a"), "org/b": _facts("org/b", 30e9),
               "org/big": _facts("org/big", 150e9)}
    monkeypatch.setattr(facts_module, "local_facts",
                        lambda manager, model, dtype="": catalog.get(model) or _facts(model, 0))
    monkeypatch.setattr(api_routes, "_recipe", lambda app, model: None)
    monkeypatch.setattr(api_routes, "_image_weights_gb",
                        lambda manager, model, dtype="": 20.0 if model == "org/flux" else 0.0)
    nodes = [_Node("s1"), _Node("s2"), _Node("s3")]
    cluster = type("C", (), {"members": lambda self: list(nodes)})()
    manager = type("M", (), {"model_dirs_for_repo": lambda self, m: [],
                             "_dir_size_gb": lambda self, d: 0.0})()
    return {"config": NodeConfig(node_id="s1", kv_cache_dtype="fp8"),
            "cluster_state": cluster, "model_manager": manager,
            "measurement_store": MeasurementStore(tmp_path / "m.json")}


def _plan(app, *models, limits=None):
    from ainode.planner.household import plan_household

    return plan_household(app, {"models": list(models), "limits": limits or {}})


class TestPlanningADraft:
    def test_budgets_start_from_the_total(self, app):
        result = _plan(app)
        node = result["nodes"][0]
        assert node["total_gb"] == pytest.approx(128, abs=0.1)
        assert node["baseline_gb"] == pytest.approx(8.4, abs=0.1)
        assert 90 < node["budget_gb"] < 128

    def test_a_limit_in_the_draft_is_planned_with(self, app):
        result = _plan(app, limits={"s2": 80})
        s2 = next(n for n in result["nodes"] if n["node_id"] == "s2")
        assert s2["limit_gb"] == 80
        assert s2["budget_gb"] == pytest.approx(80 - 8.4, abs=0.2)

    def test_two_models_fill_one_node(self, app):
        result = _plan(app,
                       {"id": "a", "model": "org/a", "node_ids": ["s1"], "mode": "auto",
                        "max_model_len": 32768},
                       {"id": "b", "model": "org/b", "node_ids": ["s1"], "mode": "auto",
                        "max_model_len": 32768})
        assert result["ok"], result
        s1 = result["nodes"][0]
        assert s1["used_gb"] == pytest.approx(s1["budget_gb"], abs=0.2)
        a, b = result["models"]
        assert a["cache_per_node_gb"] == pytest.approx(b["cache_per_node_gb"], abs=0.1)
        assert a["sessions"] >= 1 and a["max_num_seqs"] == a["sessions"]
        # The reservation fits the node: the memory fractions add up under it.
        assert (a["gpu_memory_utilization"] + b["gpu_memory_utilization"]) * 128 \
            <= s1["budget_gb"] + 0.5

    def test_usage_mode_takes_what_the_sessions_need(self, app):
        result = _plan(app, {"id": "a", "model": "org/a", "node_ids": ["s1"],
                             "mode": "usage", "max_model_len": 65536, "sessions": 2})
        a = result["models"][0]
        assert a["sessions"] == 2
        assert a["kv_tokens"] >= 65536 * 2

    def test_a_model_that_does_not_fit_is_a_conflict(self, app):
        result = _plan(app, {"id": "x", "model": "org/big", "node_ids": ["s2"]})
        assert not result["ok"]
        assert result["conflicts"] and "S2" in result["conflicts"][0]

    def test_split_over_two_nodes_fits(self, app):
        result = _plan(app, {"id": "x", "model": "org/big", "node_ids": ["s1", "s2"],
                             "strategy": "tensor"})
        assert result["ok"], result
        assert result["models"][0]["strategy"] == "tensor"
        assert result["entries"][0]["node_ids"] == ["s1", "s2"]

    def test_a_split_without_the_head_is_led_by_its_first_node(self, app):
        """It used to be refused: 'das muss doch auch ohne den head gehen'."""
        result = _plan(app, {"id": "x", "model": "org/big", "node_ids": ["s2", "s3"]})
        model = result["models"][0]
        assert result["ok"] and not model["errors"]
        assert model["launched_by"] == "s2"
        assert any("SSH" in w for w in model["warnings"])

    def test_a_replica_on_another_node_is_fine(self, app):
        result = _plan(app, {"id": "a1", "model": "org/a", "node_ids": ["s1"]},
                       {"id": "a2", "model": "org/a", "node_ids": ["s2"]})
        assert result["ok"], result
        assert len(result["entries"]) == 2

    def test_the_same_model_twice_on_one_node_is_not(self, app):
        result = _plan(app, {"id": "a1", "model": "org/a", "node_ids": ["s1"]},
                       {"id": "a2", "model": "org/a", "node_ids": ["s1"]})
        assert result["models"][1]["errors"]

    def test_a_model_not_on_disk_is_named(self, app):
        result = _plan(app, {"id": "q", "model": "org/nothere", "node_ids": ["s1"]})
        assert "not downloaded" in result["models"][0]["errors"][0]

    def test_image_and_embedding_take_their_blocks(self, app):
        result = _plan(app,
                       {"id": "i", "model": "org/flux", "kind": "image",
                        "node_ids": ["s3"], "max_image_size": 1024},
                       {"id": "e", "model": "BAAI/bge-large-en-v1.5",
                        "kind": "embedding", "node_ids": ["s3"]},
                       {"id": "a", "model": "org/a", "node_ids": ["s3"]})
        assert result["ok"], result
        s3 = next(n for n in result["nodes"] if n["node_id"] == "s3")
        kinds = {seg["kind"]: seg for seg in s3["segments"]}
        assert kinds["image"]["fixed_gb"] == pytest.approx(26.5, abs=0.1)
        assert kinds["embedding"]["fixed_gb"] == pytest.approx(1.34 * 1.3 + 1, abs=0.1)
        entries = {e["kind"]: e for e in result["entries"]}
        assert entries["image"]["engine_backend"] == "diffusers"
        assert entries["embedding"]["node_ids"] == ["s3"]

    def test_the_entries_carry_the_plan(self, app):
        result = _plan(app, {"id": "a", "model": "org/a", "node_ids": ["s1"],
                             "mode": "usage", "max_model_len": 32768, "sessions": 3,
                             "extra_vllm_args": ["--max-num-seqs", "99", "--foo"]})
        entry = result["entries"][0]
        assert entry["max_model_len"] == 32768
        assert entry["extra_vllm_args"] == ["--foo", "--max-num-seqs", "3"]
        assert 0 < entry["gpu_memory_utilization"] <= 0.95
        # The node default dtype is not written as an instruction.
        assert "kv_cache_dtype" not in entry


@pytest.mark.asyncio
async def test_the_route_plans_and_launches_nothing(app):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from ainode.planner.api_routes import register_planner_routes

    web_app = web.Application()
    for key, value in app.items():
        web_app[key] = value
    register_planner_routes(web_app)
    async with TestClient(TestServer(web_app)) as client:
        resp = await client.post("/api/planner/household", json={
            "models": [{"id": "a", "model": "org/a", "node_ids": ["s1"]}]})
        assert resp.status == 200
        data = await resp.json()
        assert data["ok"] and data["entries"][0]["model"] == "org/a"
        assert (await client.post("/api/planner/household", data="x")).status == 400


class TestWhatTheWizardOffers:
    def test_kinds_are_read_off_the_disk(self, tmp_path):
        from ainode.planner.household import kind_on_disk

        image = tmp_path / "flux"
        image.mkdir()
        (image / "model_index.json").write_text("{}")
        emb = tmp_path / "bge"
        emb.mkdir()
        (emb / "modules.json").write_text("[]")
        llm = tmp_path / "qwen"
        llm.mkdir()
        (llm / "config.json").write_text("{}")
        assert kind_on_disk(image) == "image"
        assert kind_on_disk(emb) == "embedding"
        assert kind_on_disk(llm) == "llm"

    def test_the_listing_groups_by_kind(self, tmp_path):
        from ainode.planner.household import wizard_models

        flux = tmp_path / "flux"
        flux.mkdir()
        (flux / "model_index.json").write_text("{}")

        class _M:
            def list_downloaded(self):
                return [{"hf_repo": "org/flux", "size_gb": 20},
                        {"hf_repo": "org/qwen", "size_gb": 60, "complete": False,
                         "incomplete_reason": "2 shards missing"},
                        {"hf_repo": "BAAI/bge-large-en-v1.5", "size_gb": 1.3}]

            def model_dirs_for_repo(self, repo):
                return [flux] if repo == "org/flux" else [tmp_path / "none"]

        models = wizard_models({"model_manager": _M()})
        assert [(m["model"], m["kind"]) for m in models] == [
            ("org/qwen", "llm"), ("BAAI/bge-large-en-v1.5", "embedding"),
            ("org/flux", "image")]
        assert models[0]["complete"] is False


class TestHowFarASliderMayGo:
    def test_room_is_what_the_automatic_neighbours_can_give(self):
        a = Item("a", "A", node_ids=["n1"], fixed_gb=30, mode="size", pinned_gb=10)
        b = Item("b", "B", node_ids=["n1"], fixed_gb=20, mode="auto", min_gb=5)
        solve([a, b], _nodes(100))
        # 100 - 30 - 20 - B's minimum 5 = 45 for A.
        assert a.room_gb == pytest.approx(45)
        assert b.cache_gb == pytest.approx(40)

    def test_a_spanning_model_is_bound_by_its_tightest_node(self):
        c = Item("c", "C", node_ids=["n1", "n2"], fixed_gb=40, mode="size", pinned_gb=5)
        solve([c], _nodes(100, 60))
        assert c.room_gb == pytest.approx(20)


class TestATooSmallPinStaysInTheArithmetic:
    def test_it_is_an_error_but_still_counted(self, app):
        from ainode.planner.household import plan_household

        result = plan_household(app, {"models": [
            {"id": "a", "model": "org/a", "node_ids": ["s1"], "mode": "size",
             "cache_gb": 0.5, "max_model_len": 131072},
            {"id": "b", "model": "org/b", "node_ids": ["s1"], "mode": "auto",
             "max_model_len": 32768}]})
        a, b = result["models"]
        assert any("less than one" in e for e in a["errors"])
        s1 = result["nodes"][0]
        # A still holds its weights and its half gigabyte: B does not get them.
        assert {seg["id"] for seg in s1["segments"]} == {"a", "b"}
        assert s1["used_gb"] == pytest.approx(s1["budget_gb"], abs=0.2)


class TestTheContextGrowsWithTheRoom:
    """Reported: shrinking the image model on the same node did not change
    the small Qwen's context — an automatic cache grew its session count."""

    def _plan(self, app, image_size, **llm):
        from ainode.planner.household import plan_household

        return plan_household(app, {"models": [
            {"id": "i", "model": "org/flux", "kind": "image", "node_ids": ["s1"],
             "max_image_size": image_size},
            {"id": "q", "model": "org/b", "node_ids": ["s1"], **llm}]})["models"][1]

    def test_by_default_the_context_grows(self, app):
        big = self._plan(app, 2048, sessions=8)
        small = self._plan(app, 768, sessions=8)
        assert small["max_model_len"] > big["max_model_len"]
        assert small["sessions"] == big["sessions"] == 8

    def test_up_to_the_checkpoint_then_more_sessions(self, app):
        from ainode.planner.household import plan_household

        out = plan_household(app, {"models": [
            {"id": "q", "model": "org/b", "node_ids": ["s2"]}]})["models"][0]
        assert out["max_model_len"] == 262144
        assert out["sessions"] > 1

    def test_or_the_sessions_grow_at_a_fixed_context(self, app):
        big = self._plan(app, 2048, grow="sessions", max_model_len=32768)
        small = self._plan(app, 768, grow="sessions", max_model_len=32768)
        assert small["max_model_len"] == big["max_model_len"] == 32768
        assert small["sessions"] > big["sessions"]

    def test_several_sessions_share_the_growth(self, app):
        one = self._plan(app, 1024, sessions=1)
        two = self._plan(app, 1024, sessions=2)
        assert two["sessions"] == 2
        assert two["max_model_len"] < one["max_model_len"]


class TestEverySettingReachesTheEntry:
    def test_llm_settings(self, app):
        from ainode.planner.household import plan_household

        result = plan_household(app, {"models": [{
            "id": "a", "model": "org/a", "node_ids": ["s1"], "mode": "usage",
            "max_model_len": 32768, "sessions": 2, "tool_calling": "off",
            "trust_remote_code": True, "served_model_name": "coder, qwen",
            "extra_vllm_args": ('--enable-prefix-caching --speculative-config '
                                '\'{"method":"mtp"}\''),
            "extra_env": {"VLLM_X": "1"}}]})
        entry = result["entries"][0]
        assert entry["tool_calling"] == "off" and entry["trust_remote_code"] is True
        assert entry["served_model_name"] == ["coder", "qwen"]
        assert entry["extra_vllm_args"][:3] == ["--enable-prefix-caching",
                                                "--speculative-config", '{"method":"mtp"}']
        assert entry["extra_env"] == {"VLLM_X": "1"}

    def test_image_settings(self, app):
        from ainode.planner.household import plan_household

        result = plan_household(app, {"models": [{
            "id": "i", "model": "org/flux", "kind": "image", "node_ids": ["s1"],
            "max_image_size": 1024, "image_steps": 28, "image_size": "768x768",
            "image_dtype": "float16", "image_guidance": 4.5}]})
        entry = result["entries"][0]
        assert (entry["image_steps"], entry["image_size"], entry["image_dtype"],
                entry["image_guidance"]) == (28, "768x768", "float16", 4.5)

    def test_the_profile_entry_carries_them_to_the_load(self):
        from ainode.profiles.store import ProfileEntry

        body = ProfileEntry(model="m", kind="image", image_dtype="float16",
                            image_guidance=4.5, tool_calling="off").launch_body()
        assert body["image_dtype"] == "float16" and body["image_guidance"] == 4.5
        assert body["tool_calling"] == "off"


class TestAnImageModelOnTheBooks:
    """'Refusing stacked load: this node already reserves 0.50 … the requested
    0.48 would total 0.98' — the image model held ~0.24 in reality."""

    def test_it_counts_what_it_holds(self, tmp_path, monkeypatch):
        from ainode.measure.store import Measurement, MeasurementStore
        from ainode.models import api_routes

        store = MeasurementStore(tmp_path / "m.json")
        store._write({"org/flux": Measurement(model="org/flux", launches=1,
                                              last_ok=1.0, memory_gb=31.7)})
        collector = type("C", (), {"get_gpu_metrics":
                                   lambda self: {"memory_total_mb": 124650}})()
        app = {"measurement_store": store, "metrics_collector": collector}
        inst = type("I", (), {})()
        inst.record = type("R", (), {"model": "org/flux"})()
        inst.backend = type("B", (), {"config": NodeConfig(
            engine_backend="diffusers", gpu_memory_utilization=0.5)})()
        assert api_routes._reserved_share(app, inst) == pytest.approx(31.7 / 130.7, abs=0.01)

    def test_a_vllm_instance_still_counts_its_fraction(self):
        from ainode.models import api_routes

        inst = type("I", (), {})()
        inst.record = type("R", (), {"model": "m"})()
        inst.backend = type("B", (), {"config": NodeConfig(gpu_memory_utilization=0.48)})()
        assert api_routes._reserved_share({}, inst) == 0.48
