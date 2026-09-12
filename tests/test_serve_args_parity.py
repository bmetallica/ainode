"""The default backend must honour the same launch knobs as the other one.

Every per-load field in the UI writes to NodeConfig, and NodeConfig is read by
whichever backend is configured. The eugr backend — the default — read only
some of them, so an API alias, a KV dtype and trust_remote_code were accepted
by the UI, stored in the config, and then silently dropped on the way to vLLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.parallelism import ParallelPlan
from ainode.engine.serve_args import (
    effective_kv_cache_dtype,
    is_multimodal_model,
    local_model_dir,
    supplied_flags,
)


class TestSharedHelpers:
    def test_supplied_flags_sees_both_spellings(self):
        assert supplied_flags(["--kv-cache-dtype", "fp8", "--max-model-len=4096"]) == {
            "--kv-cache-dtype", "--max-model-len"}

    def test_supplied_flags_tolerates_nothing(self):
        assert supplied_flags(None) == set()

    def test_local_model_dir_finds_the_flat_layout(self, tmp_path):
        d = tmp_path / "org--name"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert local_model_dir("org/name", str(tmp_path)) == str(d)

    def test_local_model_dir_is_none_when_not_downloaded(self, tmp_path):
        assert local_model_dir("org/name", str(tmp_path)) is None

    def test_an_empty_directory_does_not_count_as_downloaded(self, tmp_path):
        (tmp_path / "org--name").mkdir()
        assert local_model_dir("org/name", str(tmp_path)) is None

    @pytest.mark.parametrize("config,expected", [
        ({"vision_config": {}}, True),
        ({"architectures": ["Qwen2VLForConditionalGeneration"]}, True),
        ({"architectures": "SomeVisionModel"}, True),
        ({"architectures": ["LlamaForCausalLM"]}, False),
        ({}, False),
    ])
    def test_multimodal_detection(self, tmp_path, config, expected):
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_multimodal_model(str(tmp_path)) is expected

    def test_unreadable_config_is_not_multimodal(self, tmp_path):
        assert is_multimodal_model(str(tmp_path)) is False
        assert is_multimodal_model(None) is False

    def test_fp8_is_downgraded_for_a_vision_model(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"vision_config": {}}))
        config = NodeConfig(kv_cache_dtype="fp8")
        assert effective_kv_cache_dtype(config, str(tmp_path)) == "auto"

    def test_an_explicit_fp8_survives_on_a_vision_model(self, tmp_path):
        # The operator's way back in for a model they know handles fp8 KV.
        (tmp_path / "config.json").write_text(json.dumps({"vision_config": {}}))
        config = NodeConfig(kv_cache_dtype="fp8", kv_cache_dtype_explicit=True)
        assert effective_kv_cache_dtype(config, str(tmp_path)) == "fp8"

    def test_text_models_keep_fp8(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(
            {"architectures": ["LlamaForCausalLM"]}))
        assert effective_kv_cache_dtype(NodeConfig(kv_cache_dtype="fp8"),
                                        str(tmp_path)) == "fp8"


def _flat(script: str) -> str:
    """The script as the shell sees it: continuations joined, spaces collapsed.

    The generator puts one argument per continued line, so ``--kv-cache-dtype``
    and its value are on separate lines in the file and adjacent on the command
    line. Asserting on the file text would test the layout, not the command.
    """
    import re
    return re.sub(r"\s+", " ", script.replace("\\\n", " "))


def _script(**config_kwargs) -> str:
    """Render the eugr launch script for a config, without launching anything."""
    from ainode.engine.backends.eugr import EugrBackend

    config = NodeConfig(model="org/name", **config_kwargs)
    backend = EugrBackend.__new__(EugrBackend)
    backend.config = config
    path = EugrBackend._write_launch_script(backend, ParallelPlan(), solo=True)
    return Path(path).read_text()


class TestEugrLaunchScript:
    def test_served_model_name_reaches_vllm(self):
        assert "--served-model-name chat" in _flat(_script(served_model_name=["chat"]))

    def test_several_aliases_are_all_emitted(self):
        script = _flat(_script(served_model_name=["chat", "gpt-4"]))
        assert "--served-model-name chat gpt-4" in script

    def test_kv_cache_dtype_reaches_vllm(self):
        assert "--kv-cache-dtype fp8" in _flat(_script(kv_cache_dtype="fp8"))

    def test_trust_remote_code_reaches_vllm(self):
        assert "--trust-remote-code" in _script(trust_remote_code=True)

    def test_a_recipe_flag_suppresses_the_built_in_one(self):
        # Qwen3.8's recipe pins --kv-cache-dtype auto; emitting the config's
        # fp8 as well would hand vLLM the flag twice.
        script = _flat(_script(kv_cache_dtype="fp8",
                               extra_vllm_args=["--kv-cache-dtype", "auto"]))
        assert script.count("--kv-cache-dtype") == 1
        assert "--kv-cache-dtype auto" in script

    def test_a_recipe_context_length_wins_over_the_config(self):
        script = _script(max_model_len=4096,
                         extra_vllm_args=["--max-model-len=262144"])
        assert script.count("--max-model-len") == 1

    def test_nothing_set_emits_no_stray_flags(self):
        script = _script(kv_cache_dtype="")
        for flag in ("--served-model-name", "--trust-remote-code", "--kv-cache-dtype"):
            assert flag not in script
