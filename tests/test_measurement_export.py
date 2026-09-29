"""Exporting what the cluster measured, with how each run was started.

Asked for: a script to hand the measurements on "direkt mit deren
Startparametern", so other models can be judged against real runs.
"""

from __future__ import annotations

import asyncio

import pytest

from ainode.core.config import NodeConfig
from ainode.measure.recorder import _launch_of
from ainode.measure.store import Measurement, MeasurementStore
from ainode.planner.facts import ModelFacts


class _Record:
    tensor_parallel_size = 2
    pipeline_parallel_size = 1
    data_parallel_size = 1
    peer_ips = ["10.0.0.2"]


class TestTheLaunchIsWrittenDown:
    def test_the_split_window_and_flags(self):
        config = NodeConfig(max_model_len=131072, gpu_memory_utilization=0.87,
                            parallel_strategy="tensor",
                            extra_vllm_args=["--kv-cache-dtype", "fp8", "--max-num-seqs", "4",
                                             "--enable-expert-parallel"])
        launch = _launch_of(_Record(), config)
        assert launch["tensor_parallel_size"] == 2 and launch["nodes"] == 2
        assert launch["strategy"] == "tensor"
        assert launch["max_model_len"] == 131072
        assert launch["kv_cache_dtype"] == "fp8" and launch["max_num_seqs"] == 4
        assert "--enable-expert-parallel" in launch["extra_vllm_args"]

    def test_it_is_kept_with_the_measurement_and_in_the_history(self, tmp_path):
        store = MeasurementStore(tmp_path / "m.json")
        store.record_launch("m", ok=True, memory_gb=80, launch={"max_model_len": 65536})
        store.record_launch("m", ok=False, launch={"max_model_len": 262144})
        entry = store.get("m")
        # The measurement's launch is the one its figures belong to …
        assert entry.launch == {"max_model_len": 65536}
        # … and the one that failed is on record too.
        assert [h["launch"]["max_model_len"] for h in entry.history] == [65536, 262144]
        assert entry.history[-1]["ok"] is False


@pytest.fixture
def app(tmp_path, monkeypatch):
    import ainode.core.config as config_module
    import ainode.planner.facts as facts_module

    monkeypatch.setattr(config_module, "AINODE_HOME", tmp_path)
    (tmp_path / "engine-build.env").write_text("ENGINE_VLLM_VERSION=0.12.0\nENGINE_IMAGE_ID=sha256:1\n")
    facts = ModelFacts(repo="org/m", num_layers=48, attention_layers=48, num_kv_heads=8,
                       head_dim=128, num_attention_heads=32, torch_dtype="bfloat16",
                       weight_bytes=int(160e9), max_position_embeddings=262144)
    monkeypatch.setattr(facts_module, "local_facts", lambda manager, model: facts)
    store = MeasurementStore(tmp_path / "m.json")
    store._write({"org/m": Measurement(
        model="org/m", launches=2, last_ok=1.0, memory_gb=85.0, weights_gb=72.7,
        kv_cache_gb=10.0, kv_tokens=100000, rank_count=2, kv_cache_dtype="fp8",
        launch={"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                "kv_cache_dtype": "fp8", "max_model_len": 131072})})
    node = type("N", (), {"node_id": "s1", "node_name": "spark1", "status": "online",
                          "gpu_name": "GB10", "unified_memory": True,
                          "gpu_memory_total_mb": 122070, "gpu_memory_used_mb": 9000,
                          "gpu_memory_gb": 128, "memory_limit_gb": 0,
                          "baseline_used_mb": 9000, "version": "0.6.0"})()
    cluster = type("C", (), {"members": lambda self: [node], "get_nodes": lambda self: []})()
    from ainode.profiles.store import ProfileStore

    return {"config": NodeConfig(node_id="s1"), "cluster_state": cluster,
            "measurement_store": store, "model_manager": object(),
            "profiles": ProfileStore(tmp_path / "p.json"), "instances": None}


class TestTheExport:
    def test_it_carries_measurements_launch_facts_and_prediction(self, app):
        from ainode.measure.export import build_export

        data = asyncio.run(build_export(app))
        assert data["engine_build"]["ENGINE_VLLM_VERSION"] == "0.12.0"
        assert data["nodes"][0]["idle_use_gb"] == pytest.approx(9.4, abs=0.1)
        (model,) = data["models"]
        assert model["facts"]["num_kv_heads"] == 8
        row = model["measurements"][0]
        assert row["launch"]["tensor_parallel_size"] == 2
        planner = row["planner"]
        # 160 GB / 2 ranks x 1.05 = 84 estimated, against 72.7 measured.
        assert planner["weights_per_node_gb_estimated"] == pytest.approx(84.0, abs=0.1)
        assert planner["weights_error_percent"] == pytest.approx(15.5, abs=0.2)
        assert planner["kv_bytes_per_token_measured"] == 200000
        assert planner["footprint_beyond_weights_and_cache_gb"] == pytest.approx(2.3, abs=0.01)
        assert planner["strategy"] == "tensor"
        assert "ENGINE_OVERHEAD_GB" in data["planner_constants"]

    def test_nothing_secret_goes_in(self, app):
        import json

        from ainode.measure.export import build_export

        text = json.dumps(asyncio.run(build_export(app))).lower()
        for word in ("password", "api_key", "cluster.key", "hf_token", "secret"):
            assert word not in text, word

    def test_the_script_calls_the_route(self):
        from pathlib import Path

        script = (Path(__file__).resolve().parent.parent / "scripts"
                  / "export-measurements.sh").read_text()
        assert "/api/measurements/export" in script
