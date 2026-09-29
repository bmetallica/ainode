"""--kv-cache-dtype auto on a checkpoint that asks for an fp8 cache is fp8.

The wizard offered unsloth/Qwen3.8-27B-NVFP4 "46.5 GB cache per node · 708.8K
tokens" — 65.6 KB a token, the bf16 formula. Its config.json carries an 8-bit
float kv_cache_scheme, vLLM cached in fp8 under "auto", and the engine said
1,037,863 tokens in 39.1 GB: 37.7 KB.
"""

from __future__ import annotations

from ainode.planner.compute import effective_kv_dtype, kv_bytes_per_token
from ainode.planner.facts import facts_from_config

_QWEN27 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "vision_config": {},
    "quantization_config": {
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": {"num_bits": 8, "type": "float", "strategy": "tensor"},
    },
    "text_config": {"num_hidden_layers": 64, "full_attention_interval": 4,
                    "num_attention_heads": 24, "num_key_value_heads": 4,
                    "head_dim": 256, "dtype": "bfloat16"},
    "dtype": "bfloat16",
}


class TestTheCheckpointSaysSo:
    def test_compressed_tensors_kv_cache_scheme(self):
        assert facts_from_config(_QWEN27).kv_cache_quant == "fp8"

    def test_modelopt_kv_cache_quant_algo(self):
        facts = facts_from_config({"architectures": ["X"], "quantization_config": {
            "quant_method": "modelopt", "kv_cache_quant_algo": "FP8"}})
        assert facts.kv_cache_quant == "fp8"

    def test_none_named_is_none(self):
        # nvidia/Qwen3.8-Flash-Next-NVFP4: kv_cache_quant_algo null — bf16 cache.
        facts = facts_from_config({"architectures": ["X"], "quantization_config": {
            "quant_method": "modelopt", "kv_cache_quant_algo": None}})
        assert facts.kv_cache_quant == ""


class TestTheCostPerToken:
    def test_auto_costs_fp8(self):
        facts = facts_from_config(_QWEN27)
        assert effective_kv_dtype(facts, "auto") == "fp8"
        assert kv_bytes_per_token(facts, "auto") == kv_bytes_per_token(facts, "fp8") == 32768

    def test_an_explicit_dtype_is_left_alone(self):
        facts = facts_from_config(_QWEN27)
        assert effective_kv_dtype(facts, "fp8") == "fp8"
        assert effective_kv_dtype(facts, "bfloat16") == "bfloat16"

    def test_without_a_scheme_auto_is_the_model_dtype(self):
        config = dict(_QWEN27, quantization_config={"quant_method": "compressed-tensors"})
        assert kv_bytes_per_token(facts_from_config(config), "auto") == 65536


class TestTheMeasurementIsFoundEitherWay:
    def test_an_fp8_label_stands_for_auto(self, tmp_path, monkeypatch):
        from ainode.measure.store import Measurement, MeasurementStore
        from ainode.planner import api_routes

        store = MeasurementStore(tmp_path / "m.json")
        store._write({"m": Measurement(model="m", launches=1, last_ok=1.0, memory_gb=69.2,
                                       kv_cache_gb=39.1, kv_tokens=1037863, rank_count=1,
                                       kv_cache_dtype="fp8")})
        monkeypatch.setattr(api_routes, "local_facts",
                            lambda manager, model: facts_from_config(_QWEN27))
        app = {"measurement_store": store}
        assert api_routes._measured_bytes_per_token(app, "m", "auto") == 37673
        assert api_routes._measured_bytes_per_token(app, "m", "fp8") == 37673

    def test_a_bf16_measurement_does_not(self, tmp_path, monkeypatch):
        from ainode.measure.store import Measurement, MeasurementStore
        from ainode.planner import api_routes

        store = MeasurementStore(tmp_path / "m.json")
        store._write({"m": Measurement(model="m", launches=1, last_ok=1.0, memory_gb=69.2,
                                       kv_cache_gb=39.1, kv_tokens=1037863, rank_count=1,
                                       kv_cache_dtype="auto")})
        config = dict(_QWEN27, quantization_config={"quant_method": "compressed-tensors"})
        monkeypatch.setattr(api_routes, "local_facts",
                            lambda manager, model: facts_from_config(config))
        app = {"measurement_store": store}
        assert api_routes._measured_bytes_per_token(app, "m", "fp8") == 0
