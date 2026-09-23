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
