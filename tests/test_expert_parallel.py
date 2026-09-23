"""A mixture-of-experts that is not expert-parallel is replicated, not split.

Reported from the cluster, after three kills by the memory guard:

    ich bin mit dem max context mittlerweile bis auf 3072 runter gegangen,
    aber keinerlei Veränderung … das Modell müsste doch eigentlich auf node1
    und node2 (tp=2) passen.

The context being irrelevant is the whole clue: the KV cache is not what is
filling the node. Tensor parallelism splits attention heads and dense layers
across ranks and replicates the experts, so a 129 GB checkpoint needs 129 GB
*per node* until --enable-expert-parallel is passed.

The curated catalog has known this since MiniMax M2.7 — "without this every
rank holds every expert and the weights do not fit" — which is exactly right
and no help at all to a checkpoint the operator downloaded themselves. The
architecture is in config.json either way.
"""

from __future__ import annotations

from ainode.models.architecture import EXPERT_PARALLEL, architecture_args


class _Facts:
    def __init__(self, moe=True, experts=256, weights=128.9):
        self.is_moe = moe
        self.num_experts = experts
        self.weights_gb = weights
        self.weight_bytes = int(weights * 1e9)


class _Manager:
    def __init__(self, facts):
        self.facts = facts


def _app(facts, monkeypatch):
    monkeypatch.setattr("ainode.planner.facts.local_facts",
                        lambda manager, repo: facts)
    return {"model_manager": _Manager(facts)}


class TestWhatTheArchitectureNeeds:
    def test_a_multi_node_moe_gets_expert_parallelism(self, monkeypatch):
        app = _app(_Facts(), monkeypatch)
        assert architecture_args(app, "x/moe", 2) == [EXPERT_PARALLEL]

    def test_one_node_needs_nothing(self, monkeypatch):
        # Nothing to shard across, and the flag costs a little on one rank.
        app = _app(_Facts(), monkeypatch)
        assert architecture_args(app, "x/moe", 1) == []

    def test_a_dense_model_needs_nothing(self, monkeypatch):
        app = _app(_Facts(moe=False, experts=0), monkeypatch)
        assert architecture_args(app, "x/dense", 2) == []

    def test_an_unreadable_checkpoint_is_silence(self):
        assert architecture_args({}, "x/unknown", 2) == []

    def test_a_broken_manager_does_not_break_a_launch(self, monkeypatch):
        def _boom(manager, repo):
            raise RuntimeError("no disk")

        monkeypatch.setattr("ainode.planner.facts.local_facts", _boom)
        assert architecture_args({"model_manager": object()}, "x/m", 2) == []


class TestTheLaunchCarriesIt:
    def test_the_distributed_path_merges_it(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "architecture_args" in source
        # Merged like a recipe, so a caller's own flag and drop: still win.
        assert "merge_vllm_args" in source

    def test_the_caller_can_still_drop_it(self):
        from ainode.engine.serve_args import merge_vllm_args

        merged = merge_vllm_args([EXPERT_PARALLEL],
                                 ["drop:" + EXPERT_PARALLEL])
        assert EXPERT_PARALLEL not in merged

    def test_a_caller_who_asked_for_it_gets_it_once(self):
        from ainode.engine.serve_args import merge_vllm_args

        merged = merge_vllm_args([EXPERT_PARALLEL], [EXPERT_PARALLEL])
        assert merged.count(EXPERT_PARALLEL) == 1


class TestThePlanSaysWhatItAssumed:
    def _plan(self, nodes=2):
        from ainode.planner.compute import NodeBudget, plan_for
        from ainode.planner.facts import facts_from_config

        facts = facts_from_config(
            {"num_hidden_layers": 62, "num_attention_heads": 64,
             "num_key_value_heads": 8, "hidden_size": 6144,
             "num_experts": 256, "max_position_embeddings": 196608,
             "torch_dtype": "bfloat16"},
            "x/moe", int(128.9e9))
        budgets = [NodeBudget(node_id=f"n{i}", name=f"n{i}", total_gb=125.0,
                              free_gb=104.0) for i in range(1, nodes + 1)]
        return plan_for(facts, budgets, kv_cache_dtype="fp8")

    def test_a_multi_node_moe_plan_names_the_assumption(self):
        plan = self._plan(2)
        assert any("--enable-expert-parallel" in w for w in plan.warnings)

    def test_it_says_what_dropping_it_costs(self):
        warning = next(w for w in self._plan(2).warnings
                       if "expert-parallel" in w)
        assert "129 GB per node" in warning

    def test_a_single_node_plan_does_not_mention_it(self):
        assert not any("expert-parallel" in w for w in self._plan(1).warnings)


class TestTheGuardsMemoryKnowsItIsADifferentLaunch:
    """Otherwise the fix for an out-of-memory is refused on the strength of
    the out-of-memory it fixes."""

    MODEL = "sparkarena/Minimax-M3-v0-NVFP4-REAP50"

    def test_a_launch_that_adds_the_flag_is_let_through(self, tmp_path, monkeypatch):
        from ainode.measure.store import MeasurementStore
        from ainode.safety.admission import check_admission

        store = MeasurementStore(tmp_path / "m.json")
        store.record_guard_stop(self.MODEL, gpu_memory_utilization=0.75,
                                max_model_len=130048, nodes=2, extra_args=[])
        app = _app(_Facts(), monkeypatch)
        app["measurement_store"] = store
        assert check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                               gpu_memory_utilization=0.75,
                               max_model_len=130048) == ""

    def test_the_same_launch_is_still_refused(self, tmp_path, monkeypatch):
        from ainode.measure.store import MeasurementStore
        from ainode.safety.admission import check_admission

        store = MeasurementStore(tmp_path / "m.json")
        store.record_guard_stop(self.MODEL, gpu_memory_utilization=0.75,
                                max_model_len=130048, nodes=2,
                                extra_args=[EXPERT_PARALLEL])
        app = _app(_Facts(), monkeypatch)
        app["measurement_store"] = store
        refusal = check_admission(app, self.MODEL, node_ids=["n1", "n2"],
                                  gpu_memory_utilization=0.75,
                                  max_model_len=130048)
        assert "already had to stop" in refusal

    def test_the_killed_flags_are_recorded(self, tmp_path):
        from ainode.measure.store import MeasurementStore

        store = MeasurementStore(tmp_path / "m.json")
        store.record_guard_stop(self.MODEL, extra_args=["--kv-cache-dtype", "fp8"])
        assert store.get(self.MODEL).guard_stop_args == ["--kv-cache-dtype", "fp8"]


class TestAQuantizationVLLMCannotFind:
    """The checkpoint says what it is and not how to read it.

    sparkarena/Minimax-M3-v0-NVFP4-REAP50 carries

        "quant_algo": "NVFP4", "group_size": 16, "exclude_modules": [...]

    and no ``quant_method`` — the key vLLM selects its quantization backend
    on. Its answer to "I do not recognise this" is to treat the layers as
    unquantized, and it said so in its own log:

        Using FlashInfer CUTLASS Unquantized MoE backend

    A 129 GB checkpoint read as unquantized is not a 129 GB checkpoint any
    more, which is why the node filled up at max-model-len 3072 and why
    --enable-expert-parallel on its own did not save it.

    The obvious fix does not work. Reported from the cluster:

        Value error, Quantization method specified in the model config (None)
        does not match the quantization method specified in the
        `quantization` argument (modelopt_fp4).

    vLLM derives a method from the config first and rejects an argument that
    disagrees with it, and a config without the key derives None. So the key
    is the fix — and vLLM's own ModelOpt reader is written for exactly this
    layout, gated on the name it does not find.
    """

    NVFP4 = {"quantization_config": {"quant_algo": "NVFP4", "group_size": 16,
                                     "exclude_modules": ["lm_head"]}}

    def test_it_names_the_method_to_write(self):
        from ainode.models.quantization import missing_quant_method

        assert missing_quant_method(self.NVFP4) == "modelopt"

    def test_a_config_that_names_its_own_method_is_left_alone(self):
        from ainode.models.quantization import missing_quant_method

        assert missing_quant_method({"quantization_config": {
            "quant_method": "compressed-tensors", "quant_algo": "NVFP4"}}) == ""

    def test_an_unquantized_model_is_left_alone(self):
        from ainode.models.quantization import missing_quant_method

        assert missing_quant_method({"model_type": "qwen3"}) == ""

    def test_an_algorithm_we_cannot_map_is_left_alone(self):
        # Guessing a backend is worse than letting the engine decide.
        from ainode.models.quantization import missing_quant_method

        assert missing_quant_method(
            {"quantization_config": {"quant_algo": "something-new"}}) == ""

    def test_no_flag_is_passed_for_it(self):
        # The flag is refused by the engine; only the key works.
        import inspect

        from ainode.models import architecture

        assert "--quantization" not in inspect.getsource(
            architecture.architecture_args)


class TestRepairingTheConfig:
    NVFP4 = TestAQuantizationVLLMCannotFind.NVFP4

    def _checkpoint(self, tmp_path, config=None):
        import json

        (tmp_path / "config.json").write_text(
            json.dumps(config if config is not None else self.NVFP4))
        return tmp_path

    def test_it_writes_the_key(self, tmp_path):
        import json

        from ainode.models.quantization import repair_quant_method

        changed, message = repair_quant_method(self._checkpoint(tmp_path))
        assert changed is True
        written = json.loads((tmp_path / "config.json").read_text())
        assert written["quantization_config"]["quant_method"] == "modelopt"
        assert "quant_method" in message

    def test_the_rest_of_the_config_survives(self, tmp_path):
        import json

        from ainode.models.quantization import repair_quant_method

        directory = self._checkpoint(tmp_path, {
            **self.NVFP4, "model_type": "minimax_m3_vl", "num_hidden_layers": 60})
        repair_quant_method(directory)
        written = json.loads((directory / "config.json").read_text())
        assert written["model_type"] == "minimax_m3_vl"
        assert written["quantization_config"]["group_size"] == 16
        assert written["quantization_config"]["exclude_modules"] == ["lm_head"]

    def test_the_original_is_kept(self, tmp_path):
        import json

        from ainode.models.quantization import repair_quant_method

        repair_quant_method(self._checkpoint(tmp_path))
        backup = tmp_path / "config.json.ainode-backup"
        assert backup.is_file()
        assert "quant_method" not in json.loads(
            backup.read_text())["quantization_config"]

    def test_repairing_twice_keeps_the_first_backup(self, tmp_path):
        import json

        from ainode.models.quantization import repair_quant_method

        directory = self._checkpoint(tmp_path)
        repair_quant_method(directory)
        first = (tmp_path / "config.json.ainode-backup").read_text()
        changed, _ = repair_quant_method(directory)
        assert changed is False
        assert (tmp_path / "config.json.ainode-backup").read_text() == first
        assert "quant_method" not in json.loads(first)["quantization_config"]

    def test_a_checkpoint_that_needs_nothing_is_untouched(self, tmp_path):
        from ainode.models.quantization import repair_quant_method

        directory = self._checkpoint(tmp_path, {"model_type": "qwen3"})
        changed, message = repair_quant_method(directory)
        assert changed is False
        assert "already names" in message
        assert not (tmp_path / "config.json.ainode-backup").exists()

    def test_a_directory_with_no_config_says_so(self, tmp_path):
        from ainode.models.quantization import repair_quant_method

        assert repair_quant_method(tmp_path)[0] is False

    def test_the_route_exists(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="n1"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/models/repair-quantization" in paths


class TestTheGateSaysSoBeforeTheLaunch:
    def _app(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text(json.dumps(
            TestAQuantizationVLLMCannotFind.NVFP4))

        class _Manager:
            def model_dirs_for_repo(self, repo):
                return [tmp_path]

        return {"model_manager": _Manager()}

    def test_it_refuses_with_the_reason(self, tmp_path):
        from ainode.safety.admission import check_admission

        refusal = check_admission(self._app(tmp_path), "x/nvfp4")
        assert "quant_method" in refusal
        assert "as if it were not quantized" in refusal

    def test_it_says_a_flag_will_not_do(self, tmp_path):
        # Because that is the first thing anyone tries, and the engine
        # rejects it in a way that reads like our mistake.
        from ainode.safety.admission import check_admission

        assert "--quantization that disagrees" in check_admission(
            self._app(tmp_path), "x/nvfp4")

    def test_the_refusal_is_marked_repairable(self, tmp_path):
        from ainode.safety.admission import check_admission

        refusal = check_admission(self._app(tmp_path), "x/nvfp4")
        assert getattr(refusal, "repairable_model", "") == "x/nvfp4"

    def test_a_proper_config_passes(self, tmp_path):
        import json

        from ainode.safety.admission import check_admission

        (tmp_path / "config.json").write_text(json.dumps({
            "quantization_config": {"quant_method": "modelopt",
                                    "quant_algo": "NVFP4"}}))

        class _Manager:
            def model_dirs_for_repo(self, repo):
                return [tmp_path]

        assert check_admission({"model_manager": _Manager()}, "x/ok") == ""

    def test_both_routes_carry_the_marker(self):
        import inspect

        from ainode.engine import sharding_routes
        from ainode.models import api_routes

        for source in (inspect.getsource(sharding_routes.handle_sharding_launch),
                       inspect.getsource(api_routes.handle_model_load)):
            assert '"repairable"' in source


class TestTheUIOffersTheRepair:
    APP_JS = (__import__("pathlib").Path(__file__).resolve().parent.parent /
              "ainode" / "web" / "static" / "js" / "app.js").read_text()

    def test_the_launch_offers_it(self):
        assert "offerToRepairTheConfig" in self.APP_JS
        assert "/api/models/repair-quantization" in self.APP_JS

    def test_it_says_what_is_written_and_what_is_kept(self):
        body = self.APP_JS.split("async offerToRepairTheConfig")[1][:900]
        assert "One key is added" in body
        assert "config.json.ainode-backup" in body

    def test_it_retries_the_launch_only_when_something_changed(self):
        body = self.APP_JS.split("async offerToRepairTheConfig")[1][:1400]
        assert "return !!out.changed;" in body
