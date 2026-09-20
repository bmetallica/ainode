"""The deterministic planner: does it fit, how to split it, what it holds.

The numbers in this file are not invented. The MiniMax M2.7 case reproduces
the arithmetic that was done by hand for the catalog entry — 62 layers, 8 KV
heads, head_dim 128, fp8 KV, two nodes — and the planner has to land on the
same 124 KiB per token and the same order of cache, or it is not replacing
that hand calculation, it is competing with it.
"""

from __future__ import annotations

import asyncio
import json

from ainode.planner.compute import (
    ENGINE_OVERHEAD_GB,
    NodeBudget,
    SYSTEM_RESERVE_GB,
    kv_bytes_per_token,
    plan_for,
)
from ainode.planner.facts import (
    ModelFacts,
    facts_from_config,
    local_facts,
    read_config,
    snapshot_dir,
    weight_bytes_on_disk,
)

MINIMAX = {
    "architectures": ["MiniMaxM2ForCausalLM"],
    "num_hidden_layers": 62, "num_attention_heads": 48,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 6144,
    "max_position_embeddings": 196608, "torch_dtype": "bfloat16",
    "num_local_experts": 256, "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "awq"},
}

DENSE_70B = {
    "num_hidden_layers": 80, "num_attention_heads": 64,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 8192,
    "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
}


def _spark(node_id, free=118.0, total=122.0):
    return NodeBudget(node_id=node_id, name=node_id.upper(),
                      total_gb=total, free_gb=free)


def _three_sparks(free=118.0):
    return [_spark("n1", free), _spark("n2", free), _spark("n3", free)]


class TestReadingTheCheckpoint:
    def test_the_shape_comes_out_of_config_json(self):
        facts = facts_from_config(MINIMAX, "org/m", weight_bytes=int(130e9))
        assert facts.num_layers == 62
        assert facts.num_kv_heads == 8
        assert facts.head_dim == 128
        assert facts.max_position_embeddings == 196608
        assert facts.is_moe and facts.num_experts == 256
        assert facts.quantization == "awq"

    def test_head_dim_is_derived_when_absent(self):
        facts = facts_from_config(
            {"num_hidden_layers": 32, "num_attention_heads": 32,
             "hidden_size": 4096}, "org/m")
        assert facts.head_dim == 128

    def test_kv_heads_default_to_attention_heads(self):
        # An old multi-head checkpoint states no KV head count at all.
        facts = facts_from_config(
            {"num_hidden_layers": 32, "num_attention_heads": 32,
             "hidden_size": 4096}, "org/m")
        assert facts.num_kv_heads == 32

    def test_a_multimodal_config_is_read_through_its_text_half(self):
        # The layer counts sit under text_config; the top level has none, and
        # a model with zero layers has an unplannable cache.
        facts = facts_from_config({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
            "text_config": {"num_hidden_layers": 48, "num_attention_heads": 32,
                            "num_key_value_heads": 8, "head_dim": 128,
                            "max_position_embeddings": 131072},
        }, "org/vlm")
        assert facts.num_layers == 48 and facts.num_kv_heads == 8

    def test_a_hybrid_stack_charges_only_its_attention_layers(self):
        # Charging all 62 layers of a mostly-recurrent model overstates the
        # cache by an order of magnitude.
        facts = facts_from_config({
            "num_hidden_layers": 4, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128,
            "layer_types": ["linear_attention", "linear_attention",
                            "full_attention", "linear_attention"],
        }, "org/hybrid")
        assert facts.attention_layers == 1
        assert facts.is_hybrid is True

    def test_an_interval_style_hybrid_is_recognised(self):
        facts = facts_from_config({
            "num_hidden_layers": 48, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128,
            "full_attention_interval": 4}, "org/hybrid")
        assert facts.attention_layers == 12 and facts.is_hybrid

    def test_an_ordinary_stack_is_not_hybrid(self):
        assert facts_from_config(MINIMAX, "org/m").is_hybrid is False
        assert facts_from_config(MINIMAX, "org/m").attention_layers == 62

    def test_missing_facts_are_named_not_guessed(self):
        facts = facts_from_config({}, "org/m")
        assert facts.unknown == ["config.json"]
        assert facts.usable is False


class TestOnDisk:
    def test_a_flat_download_is_read_directly(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(MINIMAX))
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1024)
        assert snapshot_dir(tmp_path) == tmp_path
        assert read_config(tmp_path)["num_hidden_layers"] == 62
        assert weight_bytes_on_disk(tmp_path) == 1024

    def test_a_hub_cache_layout_is_followed_into_its_snapshot(self, tmp_path):
        snap = tmp_path / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(DENSE_70B))
        (snap / "model-00001.safetensors").write_bytes(b"x" * 2048)
        assert read_config(tmp_path)["num_hidden_layers"] == 80
        assert weight_bytes_on_disk(tmp_path) == 2048

    def test_only_weight_files_count(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "README.md").write_bytes(b"x" * 9999)
        (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
        assert weight_bytes_on_disk(tmp_path) == 10

    def test_a_model_that_is_not_on_disk_says_so(self):
        class _Manager:
            def model_dirs_for_repo(self, repo):
                return []

        facts = local_facts(_Manager(), "org/missing")
        assert facts.unknown == ["not on local disk"]


class TestTheKVFormula:
    def test_it_reproduces_the_hand_calculation(self):
        # 2 x 62 layers x 8 KV heads x 128 head dim x 1 byte = 124 KiB.
        facts = facts_from_config(MINIMAX, "org/m")
        assert kv_bytes_per_token(facts, "fp8") == 126976
        assert kv_bytes_per_token(facts, "fp8") / 1024 == 124.0

    def test_fp8_halves_a_bf16_cache(self):
        facts = facts_from_config(MINIMAX, "org/m")
        assert kv_bytes_per_token(facts, "auto") == \
            2 * kv_bytes_per_token(facts, "fp8")

    def test_latent_attention_uses_its_own_formula(self):
        # MLA caches one compressed vector per layer, not a K and a V per
        # head — which is why such a model holds a million tokens where a
        # conventional one of the same size holds a tenth of that.
        facts = facts_from_config({
            "num_hidden_layers": 61, "num_attention_heads": 128,
            "num_key_value_heads": 128, "hidden_size": 7168,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64,
            "torch_dtype": "bfloat16"}, "org/mla")
        assert kv_bytes_per_token(facts, "fp8") == 61 * 576

    def test_an_unreadable_shape_yields_no_number_rather_than_a_wrong_one(self):
        assert kv_bytes_per_token(ModelFacts(repo="x")) == 0


class TestThePlan:
    def test_the_minimax_case_end_to_end(self):
        # The catalog entry says: two nodes at 0.87 leave about 85 GB of
        # cache, near 700k tokens, roughly 11 sessions at 64K. The planner
        # has to agree with that, having been told none of it.
        facts = facts_from_config(MINIMAX, "MiniMaxAI/M2.7", int(130e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        max_model_len=65536, recipe_context=196608,
                        recommended_gmu=0.87)
        assert plan.fits
        assert plan.tensor_parallel_size == 2
        assert plan.node_ids == ["n1", "n2"]
        assert 80 <= plan.kv_gb <= 90
        assert 650_000 <= plan.kv_tokens <= 700_000
        assert plan.concurrent_requests == 10
        assert plan.gpu_memory_utilization == 0.87

    def test_a_model_that_fits_one_node_gets_one_node(self):
        # Multi-node tensor parallelism does not make a single stream faster;
        # it adds two all-reduces per layer over the fabric.
        facts = facts_from_config(DENSE_70B, "meta/70b-nvfp4", int(40e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        max_model_len=32768)
        assert plan.node_ids == ["n1"]
        assert plan.strategy == "solo"
        assert plan.tensor_parallel_size == 1

    def test_three_nodes_never_produce_tp_3(self):
        facts = facts_from_config(DENSE_70B, "org/big", int(250e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        max_model_len=32768)
        assert plan.fits
        assert plan.strategy == "pipeline"
        assert plan.pipeline_parallel_size == 3
        assert plan.tensor_parallel_size == 1

    def test_pipeline_is_not_offered_to_an_architecture_that_cannot_do_it(self):
        facts = facts_from_config(DENSE_70B, "org/big", int(250e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        supports_pipeline=False)
        assert plan.fits is False
        assert "pipeline" in " ".join(plan.notes)

    def test_a_model_the_cluster_cannot_hold_says_how_far_off_it_is(self):
        facts = facts_from_config(DENSE_70B, "zai/glm", int(433e9))
        plan = plan_for(facts, _three_sparks())
        assert plan.fits is False
        assert "433" in plan.blocker and "GB free" in plan.blocker

    def test_a_busy_node_is_planned_around(self):
        # Free memory, not total: a node already serving a model has most of
        # its memory inside that engine's pool.
        facts = facts_from_config(MINIMAX, "org/m", int(130e9))
        nodes = [_spark("n1", free=20.0), _spark("n2"), _spark("n3")]
        plan = plan_for(facts, nodes, kv_cache_dtype="fp8", max_model_len=65536)
        assert plan.node_ids == ["n2", "n3"]

    def test_the_tightest_node_decides(self):
        # A rank cannot borrow memory from its peers.
        facts = facts_from_config(MINIMAX, "org/m", int(130e9))
        wide = plan_for(facts, [_spark("n1"), _spark("n2")],
                        kv_cache_dtype="fp8", max_model_len=65536)
        tight = plan_for(facts, [_spark("n1"), _spark("n2", free=80.0)],
                         kv_cache_dtype="fp8", max_model_len=65536)
        assert tight.kv_gb < wide.kv_gb

    def test_a_context_the_cache_cannot_back_comes_down_instead_of_refusing(self):
        facts = facts_from_config(MINIMAX, "org/m", int(130e9))
        plan = plan_for(facts, [_spark("n1"), _spark("n2")],
                        kv_cache_dtype="fp8", max_model_len=196608,
                        concurrency=8)
        assert plan.fits
        assert plan.max_model_len < 196608
        assert plan.concurrent_requests >= 8
        assert any("more cache than" in w for w in plan.warnings)

    def test_a_context_that_exceeds_the_checkpoint_is_capped(self):
        facts = facts_from_config(DENSE_70B, "org/m", int(40e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        max_model_len=1_000_000)
        assert plan.max_model_len <= 131072
        assert any("trained" in w for w in plan.warnings)

    def test_weights_that_fit_with_no_room_for_a_cache_are_not_a_plan(self):
        # The trap this exists to catch: the engine starts, spends minutes
        # loading, and refuses at the last step.
        facts = facts_from_config(DENSE_70B, "org/m", int(40e9))
        plan = plan_for(facts, [_spark("n1", free=46.7, total=122.0)],
                        kv_cache_dtype="fp8")
        assert plan.fits is False
        assert "KV cache" in plan.blocker

    def test_the_system_keeps_its_reserve(self):
        facts = facts_from_config(DENSE_70B, "org/m", int(40e9))
        node = _spark("n1", free=50.0)
        plan = plan_for(facts, [node], kv_cache_dtype="fp8", max_model_len=8192)
        assert plan.kv_gb <= 50.0 - SYSTEM_RESERVE_GB - 40.0 - ENGINE_OVERHEAD_GB + 0.1

    def test_an_explicit_axis_is_obeyed(self):
        facts = facts_from_config(DENSE_70B, "org/m", int(40e9))
        plan = plan_for(facts, _three_sparks(), strategy="pipeline",
                        kv_cache_dtype="fp8", max_model_len=8192)
        assert plan.strategy in ("pipeline", "")
        assert plan.tensor_parallel_size == 1

    def test_a_model_not_on_disk_is_refused_rather_than_estimated(self):
        # A plan from the parameter count in the name is off by between two
        # and eight times on a quantised checkpoint, and it would be acted on.
        facts = facts_from_config(MINIMAX, "org/m", weight_bytes=0)
        plan = plan_for(facts, _three_sparks())
        assert plan.fits is False
        assert "not on this node's disk" in plan.blocker

    def test_a_hybrid_plan_says_its_number_is_a_floor(self):
        facts = facts_from_config({
            "num_hidden_layers": 48, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128,
            "max_position_embeddings": 131072,
            "full_attention_interval": 4}, "org/hybrid", int(60e9))
        plan = plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                        max_model_len=32768)
        assert any("hybrid" in w for w in plan.warnings)

    def test_the_arithmetic_is_shown(self):
        # A number an operator cannot check is a number they cannot overrule.
        facts = facts_from_config(MINIMAX, "org/m", int(130e9))
        notes = " ".join(plan_for(facts, _three_sparks(), kv_cache_dtype="fp8",
                                  max_model_len=65536).notes)
        assert "Weights" in notes and "KV per token" in notes
        assert "124.0 KiB" in notes


class _Req:
    def __init__(self, app, query=None):
        self.app = app
        self.query = query or {}


class _Node:
    def __init__(self, node_id, free_mb=120000.0):
        self.node_id = node_id
        self.node_name = node_id.upper()
        self.status = "online"
        self.gpu_memory_gb = 122.0
        self.gpu_memory_total_mb = 124928.0
        self.gpu_memory_used_mb = 124928.0 - free_mb


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)


class _Manager:
    def __init__(self, directory):
        self._dir = directory

    def model_dirs_for_repo(self, repo):
        return [self._dir]

    def _catalog_lookup(self, model_id):
        return None


class TestTheRoute:
    def _app(self, tmp_path, config=MINIMAX, weight_gb=130):
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1024)
        manager = _Manager(tmp_path)
        # Real bytes on disk would mean writing 130 GB, so the size is
        # injected; everything else is read for real.
        manager.model_dirs_for_repo = lambda repo: [tmp_path]
        app = {"cluster_state": _Cluster([_Node("n1"), _Node("n2"), _Node("n3")]),
               "model_manager": manager}
        return app

    def test_a_plan_comes_back_with_its_working(self, tmp_path, monkeypatch):
        from ainode.planner import api_routes

        monkeypatch.setattr(api_routes, "local_facts",
                            lambda m, repo: facts_from_config(
                                MINIMAX, repo, int(130e9)))
        app = self._app(tmp_path)
        resp = asyncio.run(api_routes.handle_plan(
            _Req(app, {"model": "org/m", "kv_cache_dtype": "fp8",
                       "max_model_len": "65536"})))
        body = json.loads(resp.body)
        assert body["fits"] is True
        assert body["tensor_parallel_size"] == 2
        assert body["facts"]["num_kv_heads"] == 8
        assert body["notes"]

    def test_the_model_is_required(self, tmp_path):
        assert asyncio.run(
            __import__("ainode.planner.api_routes", fromlist=["x"]).handle_plan(
                _Req(self._app(tmp_path)))).status == 400

    def test_the_nodes_it_planned_on_are_reported(self, tmp_path, monkeypatch):
        from ainode.planner import api_routes

        monkeypatch.setattr(api_routes, "local_facts",
                            lambda m, repo: facts_from_config(
                                MINIMAX, repo, int(130e9)))
        body = json.loads(asyncio.run(api_routes.handle_plan(
            _Req(self._app(tmp_path), {"model": "org/m", "nodes": "n1,n2"}))).body)
        assert [n["node_id"] for n in body["nodes"]] == ["n1", "n2"]

    def test_free_memory_is_what_is_planned_against(self, tmp_path, monkeypatch):
        from ainode.planner import api_routes

        monkeypatch.setattr(api_routes, "local_facts",
                            lambda m, repo: facts_from_config(
                                MINIMAX, repo, int(130e9)))
        app = self._app(tmp_path)
        app["cluster_state"] = _Cluster([_Node("n1", free_mb=10000.0),
                                         _Node("n2"), _Node("n3")])
        body = json.loads(asyncio.run(api_routes.handle_plan(
            _Req(app, {"model": "org/m", "kv_cache_dtype": "fp8"}))).body)
        assert "n1" not in body["node_ids"]

    def test_the_route_is_registered(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/planner" in paths


WEB = __import__("pathlib").Path(__file__).resolve().parent.parent / "ainode" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestTheForm:
    def test_the_hint_prefers_the_plan_over_the_local_estimate(self):
        assert "if (self.renderPlanHint()) return;" in APP_JS

    def test_the_plan_is_only_shown_for_the_selection_it_describes(self):
        # Otherwise a poll repaints last selection's numbers over this one's.
        assert "plan.key !== this.launchPlanKey().key" in APP_JS

    def test_asking_is_debounced(self):
        assert "clearTimeout(this._planTimer)" in APP_JS

    def test_the_plan_can_be_applied(self):
        assert "id=\"plan-apply\"" in APP_JS
        assert "applyPlan()" in APP_JS

    def test_a_node_set_without_the_head_is_expressible(self):
        # The head used to be forced on, which made a solo load on another
        # node — and any pin or plan that leaves this node out — impossible to
        # select.
        assert "if (d.dataset.head) { d.classList.add('active'); return; }" \
            not in APP_JS.split("_selectNodeIds")[1][:600]
