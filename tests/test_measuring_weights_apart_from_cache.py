"""One number was two, and it was being compared with a third thing.

From the cluster, Smaug-Flash across two nodes:

    ▣ Measured here: 83.1 GB ... (-0.6 GB vs the plan)
      Weights: 159.4 GB on disk, split 2 ways = 83.7 GB per node

Agreeing to within a gigabyte read as a planner that was right. It was two
errors cancelling. The engine's own log for that launch:

    Model loading took 67.7 GiB memory and 421.882 seconds
    Available KV cache memory: 13.72 GiB
    GPU KV cache size: 1,109,643 tokens, Maximum concurrency for 131,072 ...

67.7 of weights plus 13.7 of cache plus the engine is the 83.1 that was
measured. The plan's 83.7 was its estimate of the 67.7 — nineteen percent high,
in the direction that refuses launches which fit.

So: read the engine's own figures instead of inferring a split from one
host-level number, keep weights and cache apart, compare a footprint against a
predicted footprint, and let a measurement displace the estimate.
"""

from __future__ import annotations

import pytest

from ainode.measure.engine_report import parse_engine_report

REAL_LOG = """
(Worker_TP0 pid=244) INFO 09-27 08:00:01 [gpu_model_runner.py:5355] Model loading took 67.7038 GiB memory and 421.882142 seconds
(Worker_TP0 pid=244) INFO 09-27 08:07:10 [gpu_worker.py:693] Available KV cache memory: 13.72 GiB
(EngineCore pid=172) INFO 09-27 08:07:12 [kv_cache_utils.py:2410] GPU KV cache size: 1,109,643 tokens, Maximum concurrency for 131,072 tokens per request: 8.47x
"""


class TestReadingTheEnginesOwnFigures:
    def test_the_weights(self):
        # 67.7038 GiB is 72.7 decimal GB — the unit everything the planner
        # compares is in.
        assert parse_engine_report(REAL_LOG)["weights_gb"] == 72.7

    def test_the_cache_in_bytes(self):
        assert parse_engine_report(REAL_LOG)["kv_cache_gb"] == 14.7

    def test_the_cache_in_tokens(self):
        assert parse_engine_report(REAL_LOG)["kv_tokens"] == 1_109_643

    def test_the_window_that_figure_belongs_to(self):
        assert parse_engine_report(REAL_LOG)["kv_at_max_model_len"] == 131_072

    def test_the_concurrency_the_engine_computed(self):
        assert parse_engine_report(REAL_LOG)["kv_concurrency"] == 8.47

    def test_the_last_launch_wins(self):
        """A log is appended to across relaunches; the newest figures are the
        ones describing what is running."""
        twice = REAL_LOG + REAL_LOG.replace("67.7038", "40.0000")
        assert parse_engine_report(twice)["weights_gb"] == 42.9

    @pytest.mark.parametrize("text", ["", "nothing about memory here", None])
    def test_a_log_that_says_nothing_says_nothing(self, text):
        """{} and not zeroes: the caller keeps the host-level measurement it
        already had, and a partial answer must not overwrite a whole one."""
        assert parse_engine_report(text) == {}

    def test_a_partial_log_yields_only_what_it_has(self):
        report = parse_engine_report(
            "INFO Available KV cache memory: 13.72 GiB\n")
        assert set(report) == {"kv_cache_gb"}


class TestTheStoreKeepsThemApart:
    @pytest.fixture
    def store(self, tmp_path):
        from ainode.measure.store import MeasurementStore

        return MeasurementStore(path=tmp_path / "m.json")

    def test_the_split_is_written_down(self, store):
        store.record_launch("org/m", ok=True, memory_gb=83.1,
                            engine_report={"weights_gb": 72.7,
                                           "kv_cache_gb": 14.7,
                                           "kv_tokens": 1_109_643,
                                           "rank_count": 2})
        entry = store.load()["org/m"]
        assert entry.memory_gb == 83.1
        assert entry.weights_gb == 72.7
        assert entry.kv_cache_gb == 14.7
        assert entry.rank_count == 2

    def test_the_footprint_still_stands_alone(self, store):
        # An engine that reported nothing leaves the host measurement intact.
        store.record_launch("org/m", ok=True, memory_gb=83.1, engine_report={})
        entry = store.load()["org/m"]
        assert entry.memory_gb == 83.1 and entry.weights_gb == 0.0

    def test_a_failed_launch_records_no_memory(self, store):
        store.record_launch("org/m", ok=False,
                            engine_report={"weights_gb": 72.7})
        assert store.load()["org/m"].weights_gb == 0.0

    def test_the_rank_count_is_kept_with_the_figure(self, store):
        """Weights per node depend on the split, so a figure without one cannot
        be compared to anything."""
        store.record_launch("org/m", ok=True,
                            engine_report={"weights_gb": 72.7,
                                           "rank_count": 2})
        assert store.load()["org/m"].rank_count == 2


class TestThePlannerPrefersTheMeasurement:
    def _facts(self):
        from ainode.planner.facts import ModelFacts

        return ModelFacts(
            repo="abacusai/Smaug-Flash", num_layers=43, attention_layers=43,
            kv_lora_rank=512, qk_rope_head_dim=64, num_attention_heads=64,
            num_kv_heads=1, head_dim=512, torch_dtype="bfloat16",
            weight_bytes=166_900_000_000, max_position_embeddings=1048576,
            num_experts=256)

    def _nodes(self):
        from ainode.planner.compute import NodeBudget

        return [NodeBudget(node_id="a", name="a", total_gb=128.0, free_gb=126.0),
                NodeBudget(node_id="b", name="b", total_gb=128.0, free_gb=126.0)]

    def _plan(self, **kw):
        from ainode.planner.compute import plan_for

        return plan_for(self._facts(), self._nodes(), kv_cache_dtype="fp8",
                        concurrency=2, **kw)

    def test_the_estimate_is_the_fallback(self):
        # 166.9 / 2 x 1.05
        assert round(self._plan().weights_per_node_gb, 1) == 87.6

    def test_a_measurement_at_this_rank_count_replaces_it(self):
        plan = self._plan(measured_weights_per_node=72.7, measured_rank_count=2)
        assert plan.weights_per_node_gb == 72.7

    def test_and_buys_back_the_cache_the_estimate_had_spent(self):
        assert self._plan(measured_weights_per_node=72.7,
                          measured_rank_count=2).kv_gb > self._plan().kv_gb

    def test_a_measurement_from_a_different_split_is_ignored(self):
        """Weights per node depend on the split. A one-node figure applied to a
        two-node plan would be wrong by the rank count."""
        plan = self._plan(measured_weights_per_node=145.0,
                          measured_rank_count=1)
        assert round(plan.weights_per_node_gb, 1) == 87.6

    def test_no_measurement_changes_nothing(self):
        assert self._plan(measured_weights_per_node=0.0,
                          measured_rank_count=2).weights_per_node_gb == \
            self._plan().weights_per_node_gb


class TestTheComparisonIsLikeForLike:
    def test_the_footprint_is_judged_against_the_predicted_footprint(self):
        from ainode.planner.api_routes import _attach_measurement

        payload = {"weights_per_node_gb": 87.6, "needed_per_node_gb": 104.1}

        import ainode.measure.recorder as recorder

        original = recorder.measured_for
        recorder.measured_for = lambda a, m: {
            "memory_gb": 83.1, "weights_gb": 72.7, "launches": 1,
            "failures": 0, "max_model_len": 131072}
        try:
            _attach_measurement({}, "org/m", payload)
        finally:
            recorder.measured_for = original
        assert payload["measured"]["vs_plan_gb"] == pytest.approx(-21.0, abs=0.05)
        assert payload["measured"]["vs_plan_basis"] == "footprint per node"
        # And the weights against the weights, separately.
        assert payload["measured"]["weights_vs_plan_gb"] == pytest.approx(
            -14.9, abs=0.05)

    def test_a_measurement_from_another_window_says_so(self):
        from ainode.planner.api_routes import _attach_measurement

        payload = {"needed_per_node_gb": 104.1, "max_model_len": 262144}

        import ainode.measure.recorder as recorder

        original = recorder.measured_for
        recorder.measured_for = lambda a, m: {
            "memory_gb": 83.1, "launches": 1, "failures": 0,
            "max_model_len": 131072}
        try:
            _attach_measurement({}, "org/m", payload)
        finally:
            recorder.measured_for = original
        assert payload["measured"]["different_window"] is True


class TestTheRecorderIsStillOneClass:
    def test_its_methods_are_on_it(self):
        """A module-level def between two methods ends the class body. It
        happened while this was being written, ruff accepted it, and the whole
        suite passed — because nothing asserted the shape."""
        from ainode.measure.recorder import Recorder

        for name in ("_record", "_record_speeds"):
            assert callable(getattr(Recorder, name, None)), name
