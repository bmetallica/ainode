"""Regression cases from the cluster's own export (2026-09-29).

Each one is a figure the planner or the measurements got wrong, taken from
ainode-measurements-20260929-1851.json.
"""

from __future__ import annotations

import pytest

from ainode.core.config import NodeConfig
from ainode.measure.store import Measurement, MeasurementStore
from ainode.planner import api_routes


def _app(tmp_path, *measurements):
    store = MeasurementStore(tmp_path / "m.json")
    store._write({m.model: m for m in measurements})
    node = type("N", (), {"node_id": "db0a7a50", "gpu_memory_total_mb": 124650,
                          "gpu_memory_gb": 130.7})()
    cluster = type("C", (), {"members": lambda self: [node]})()
    return {"measurement_store": store, "cluster_state": cluster,
            "config": NodeConfig(node_id="db0a7a50")}


class TestWeightsTheEngineDoesNotCount:
    """nvidia/Qwen3.8-Flash-Next-NVFP4 at TP=2: 'Model loading took' 40.5 GB
    per rank, cache 29.2 GB, pool 0.83 x 130.7 = 108.5 GB. The checkpoint's
    51.2 GB of FP8 per-layer embedding tables are not in the 40.5."""

    def test_the_planner_plans_with_what_the_pool_held(self, tmp_path):
        app = _app(tmp_path, Measurement(
            model="nvidia/Qwen3.8-Flash-Next-NVFP4", node_id="db0a7a50",
            launches=1, last_ok=1.0, weights_gb=40.5, rank_count=2,
            kv_cache_gb=29.2, gpu_memory_utilization=0.83, memory_gb=107.5))
        weights, ranks = api_routes._measured_weights(app, "nvidia/Qwen3.8-Flash-Next-NVFP4")
        assert ranks == 2
        # 108.5 - 29.2 - (2.5 + 0.5) = 76.3, not 40.5.
        assert weights == pytest.approx(0.83 * 130.7 - 29.2 - 3.0, abs=0.2)

    def test_a_model_the_engine_counts_fully_keeps_its_figure(self, tmp_path):
        """unsloth/Qwen3.8-27B-NVFP4: 23.6 weights, 41.7 cache at 0.55 —
        the pool leaves 4 GB beyond weights and overhead (its MTP drafter,
        activations), and the larger of the two is used."""
        app = _app(tmp_path, Measurement(
            model="unsloth/Qwen3.8-27B-NVFP4", node_id="db0a7a50", launches=1,
            last_ok=1.0, weights_gb=23.6, rank_count=1, kv_cache_gb=41.7,
            gpu_memory_utilization=0.55))
        weights, _ = api_routes._measured_weights(app, "unsloth/Qwen3.8-27B-NVFP4")
        assert weights == pytest.approx(0.55 * 130.7 - 41.7 - 2.5, abs=0.2)
        assert weights > 23.6

    def test_without_the_pool_figures_the_weights_stand(self, tmp_path):
        app = _app(tmp_path, Measurement(
            model="m", node_id="db0a7a50", launches=1, last_ok=1.0,
            weights_gb=40.5, rank_count=2))
        assert api_routes._measured_weights(app, "m") == (40.5, 2)

    def test_the_node_total_is_recorded_for_next_time(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("m", ok=True, memory_gb=60, node_total_gb=130.7)
        assert store.get("m").node_total_gb == 130.7


class TestWhyALaunchFailed:
    """Qwen3.8-27B on spark-659b: 14 launches, 23 failures, and not one reason."""

    def test_the_reason_is_kept(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("m", ok=False, error="ValueError: unrecognized arguments: --x")
        assert "unrecognized" in store.get("m").history[-1]["error"]

    def test_a_success_carries_none(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("m", ok=True, error="stale")
        assert "error" not in store.get("m").history[-1]


class TestAnImageModelOnAPeer:
    """Qwen/Qwen-Image-2.1 on spark-659b: seconds_per_image 0.0 after eight
    launches. Requests arrive at the head, which did not know the model was an
    image model, since it runs elsewhere."""

    def test_its_kind_is_known_from_the_cluster(self):
        from ainode.measure.recorder import Recorder

        peer = type("N", (), {"instances": [{"model": "Qwen/Qwen-Image-2.1",
                                             "kind": "image"}]})()
        app = {"cluster_state": type("C", (), {"members": lambda self: [peer]})(),
               "instances": None}
        assert Recorder(app)._kinds()["Qwen/Qwen-Image-2.1"] == "image"


class TestTheExportDoesNotInventASplit:
    def test_an_unknown_split_is_said(self):
        from ainode.measure.export import _prediction

        out = _prediction({"model_manager": object()}, "abacusai/Smaug-Flash",
                          {"memory_gb": 103.1, "rank_count": 0, "launch": {}})
        assert "unknown" in out["split"]
