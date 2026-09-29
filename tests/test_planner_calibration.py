"""The planner learns from what the engine reported.

P1–P5 and P7 in upgrade-fixes.md. The planner's two deciding numbers — what
the weights take per node and what one token of cache costs — were arithmetic
on the checkpoint: TP_REPLICATION for the weights, a formula for the cache
that knows nothing of indexer caches (DeepSeek V4, Qwen3.8), the recurrent
state of a hybrid, or ds_mla's scale block. vLLM prints the real figures at
every launch, and #219 started keeping them. This uses them:

* the cost per token from the engine's own cache, at the dtype and rank count
  it was measured at;
* for a mixture-of-experts nobody has launched yet, the median loaded/on-disk
  ratio of the ones that were, once there are enough of them;
* and the plan says which of its numbers are measured, so a refusal that rests
  on an estimate reads as one.
"""

from __future__ import annotations

import pytest

from ainode.measure.engine_report import parse_engine_report
from ainode.measure.store import Measurement, MeasurementStore
from ainode.planner import api_routes
from ainode.planner.compute import NodeBudget, kv_bytes_per_token, plan_for
from ainode.planner.facts import ModelFacts

NODES = [NodeBudget(node_id="a", name="a", total_gb=128.0, free_gb=120.0),
         NodeBudget(node_id="b", name="b", total_gb=128.0, free_gb=120.0)]


def _facts(moe=False, weight_bytes=100_000_000_000):
    facts = ModelFacts(repo="org/model", num_layers=48, attention_layers=12,
                       num_kv_heads=4, head_dim=128, num_attention_heads=32,
                       torch_dtype="bfloat16", weight_bytes=weight_bytes,
                       max_position_embeddings=262144,
                       num_experts=64 if moe else 0)
    facts.is_moe = moe
    return facts


class TestTheCostPerTokenFromTheEngine:
    def test_a_measurement_at_this_split_replaces_the_formula(self):
        formula = kv_bytes_per_token(_facts(), "fp8")
        plan = plan_for(_facts(), NODES[:1], kv_cache_dtype="fp8",
                        measured_weights_per_node=100.0, measured_rank_count=1,
                        measured_bytes_per_token=formula * 2)
        assert plan.kv_source == "measured"
        assert plan.kv_bytes_per_token == formula * 2

    def test_at_another_split_it_is_not_used(self):
        formula = kv_bytes_per_token(_facts(), "fp8")
        plan = plan_for(_facts(), NODES[:1], kv_cache_dtype="fp8",
                        measured_weights_per_node=50.0, measured_rank_count=2,
                        measured_bytes_per_token=formula * 2)
        assert plan.kv_source == "estimated"
        assert plan.kv_bytes_per_token == formula

    def test_it_is_said_in_the_notes(self):
        plan = plan_for(_facts(), NODES[:1], kv_cache_dtype="fp8",
                        measured_weights_per_node=100.0, measured_rank_count=1,
                        measured_bytes_per_token=50_000)
        assert any("measured: the engine's own cache" in n for n in plan.notes)


class TestTheMoEFactor:
    def test_it_scales_an_unmeasured_moe(self):
        # Too big for one node, so both plans are the two-node split.
        big = 150_000_000_000
        plain = plan_for(_facts(moe=True, weight_bytes=big), NODES,
                         strategy="tensor", kv_cache_dtype="fp8")
        scaled = plan_for(_facts(moe=True, weight_bytes=big), NODES,
                          strategy="tensor", kv_cache_dtype="fp8",
                          moe_weight_factor=0.9)
        assert scaled.weights_source == "calibrated"
        assert scaled.weights_per_node_gb == pytest.approx(150 * 0.9 / 2, abs=0.1)
        assert scaled.weights_per_node_gb < plain.weights_per_node_gb

    def test_it_leaves_a_dense_model_alone(self):
        plan = plan_for(_facts(), NODES, strategy="tensor",
                        kv_cache_dtype="fp8", moe_weight_factor=0.9)
        assert plan.weights_source == "estimated"

    def test_a_measurement_of_the_model_itself_wins(self):
        plan = plan_for(_facts(moe=True, weight_bytes=150_000_000_000), NODES,
                        strategy="tensor", kv_cache_dtype="fp8",
                        moe_weight_factor=0.9,
                        measured_weights_per_node=61.0, measured_rank_count=2)
        assert plan.weights_source == "measured"
        assert plan.weights_per_node_gb == pytest.approx(61.0)

    def test_a_refusal_on_an_estimate_says_so(self):
        big = _facts(moe=True, weight_bytes=400_000_000_000)
        plan = plan_for(big, NODES, kv_cache_dtype="fp8")
        assert not plan.fits
        assert any("ESTIMATE" in w for w in plan.warnings)

    def test_the_sources_reach_the_browser(self):
        payload = plan_for(_facts(), NODES[:1], kv_cache_dtype="fp8").to_dict()
        assert payload["weights_source"] == "estimated"
        assert payload["kv_source"] == "estimated"


def _store(tmp_path, *entries):
    store = MeasurementStore(tmp_path / "m.json")
    store._write({e.model: e for e in entries})
    return {"measurement_store": store}


def _moe(name, loaded, ranks, disk):
    return Measurement(model=name, is_moe=True, weights_gb=loaded,
                       rank_count=ranks, disk_weights_gb=disk,
                       launches=1, last_ok=1.0)


class TestTheFactorIsTakenFromTheStore:
    def test_fewer_than_three_is_no_factor(self, tmp_path):
        app = _store(tmp_path, _moe("a", 72.7, 2, 159.4), _moe("b", 40, 1, 44))
        assert api_routes._moe_weight_factor(app, "x") == (0.0, 2)

    def test_three_give_the_median(self, tmp_path):
        app = _store(tmp_path, _moe("a", 45, 2, 100), _moe("b", 80, 1, 100),
                     _moe("c", 95, 1, 100))
        factor, samples = api_routes._moe_weight_factor(app, "x")
        assert samples == 3 and factor == pytest.approx(0.9)

    def test_the_model_asked_about_is_not_its_own_sample(self, tmp_path):
        app = _store(tmp_path, _moe("a", 45, 2, 100), _moe("b", 80, 1, 100),
                     _moe("c", 95, 1, 100))
        assert api_routes._moe_weight_factor(app, "a") == (0.0, 2)

    def test_a_nonsense_ratio_is_not_used(self, tmp_path):
        app = _store(tmp_path, _moe("a", 300, 1, 100), _moe("b", 300, 1, 100),
                     _moe("c", 300, 1, 100))
        assert api_routes._moe_weight_factor(app, "x")[0] == 0.0


class TestTheMeasuredCostPerToken:
    def _app(self, tmp_path, dtype="fp8"):
        entry = Measurement(model="m", kv_tokens=1_000_000, kv_cache_gb=15.0,
                            rank_count=2, kv_cache_dtype=dtype, weights_gb=70,
                            launches=1, last_ok=1.0)
        return _store(tmp_path, entry)

    def test_it_is_the_whole_cache_over_its_tokens(self, tmp_path):
        assert api_routes._measured_bytes_per_token(
            self._app(tmp_path), "m", "fp8") == 30_000

    def test_only_at_the_same_dtype(self, tmp_path):
        assert api_routes._measured_bytes_per_token(
            self._app(tmp_path), "m", "auto") == 0

    def test_not_from_a_measurement_without_a_dtype(self, tmp_path):
        assert api_routes._measured_bytes_per_token(
            self._app(tmp_path, dtype=""), "m", "fp8") == 0


class TestTheEngineVersion:
    def test_it_is_read_from_the_log(self):
        report = parse_engine_report(
            "INFO 09-29 api_server.py:1: vLLM API server version "
            "0.11.1rc2.dev104+g1a2b3c\nModel loading took 10.0 GiB memory")
        assert report["engine_version"] == "0.11.1rc2.dev104+g1a2b3c"

    def test_a_measurement_from_another_build_is_marked(self, tmp_path):
        old = Measurement(model="m", engine_version="0.10.0", memory_gb=80,
                          weights_gb=67.7, kv_cache_gb=13.7, launches=1,
                          last_ok=1.0)
        new = Measurement(model="n", engine_version="0.11.1", launches=1,
                          last_ok=2.0)
        app = _store(tmp_path, old, new)
        payload = {"needed_per_node_gb": 80}
        api_routes._attach_measurement(app, "m", payload)
        assert payload["measured"]["other_build"] == "0.11.1"

    def test_the_engine_overhead_is_measured_beside_the_plan(self, tmp_path):
        entry = Measurement(model="m", memory_gb=85.0, weights_gb=67.7,
                            kv_cache_gb=13.7, launches=1, last_ok=1.0)
        payload = {}
        api_routes._attach_measurement(_store(tmp_path, entry), "m", payload)
        assert payload["measured"]["overhead_gb"] == pytest.approx(3.6)


class TestTheRecorderKeepsTheContext:
    def test_the_cache_figures_carry_their_dtype(self):
        from ainode.core.config import NodeConfig
        from ainode.measure.recorder import _engine_report

        class _Backend:
            config = NodeConfig(kv_cache_dtype="fp8_ds_mla")

            def logs(self, n):
                return "GPU KV cache size: 1,109,643 tokens"

        report = _engine_report(_Backend())
        assert report["kv_cache_dtype"] == "fp8_ds_mla"


class TestTheCheckpointIsNotWalkedPerKeystroke:
    """O1: every field change re-plans, and every plan walked and stat()ed all
    of the checkpoint's shards."""

    def _tree(self, tmp_path):
        from ainode.planner import facts as facts_module

        facts_module.forget_weight_bytes()
        model = tmp_path / "org--m"
        model.mkdir()
        (model / "a.safetensors").write_bytes(b"x" * 100)
        return model, facts_module

    def test_the_second_plan_asks_the_cache(self, tmp_path, monkeypatch):
        model, facts_module = self._tree(tmp_path)
        walks = []
        real = facts_module._weight_bytes
        monkeypatch.setattr(facts_module, "_weight_bytes",
                            lambda p: walks.append(p) or real(p))
        assert facts_module.weight_bytes_on_disk(model) == 100
        assert facts_module.weight_bytes_on_disk(model) == 100
        assert len(walks) == 1

    def test_a_forgotten_size_is_measured_again(self, tmp_path):
        model, facts_module = self._tree(tmp_path)
        assert facts_module.weight_bytes_on_disk(model) == 100
        (model / "a.safetensors").write_bytes(b"x" * 300)
        from ainode.models.registry import ModelManager

        ModelManager.forget_size(model)     # what a download does when done
        assert facts_module.weight_bytes_on_disk(model) == 300
