"""The generated launch script is bash, so its arguments go through bash.

Observed on hardware:

    vllm serve: error: argument --speculative-config/-sc:
        Value method:qwen3_5_mtp cannot be converted to <function loads ...>

The recipe passed a perfectly good JSON object. What reached vLLM was
`method:qwen3_5_mtp` — because the writer interpolated it unquoted into a
shell script, and

    {"method":"qwen3_5_mtp","num_speculative_tokens":2}

is brace expansion: bash splits it at the comma and strips the quotes. The
launch looked like a bad recipe and was a bad quoting decision.

These tests run the generated script through a real bash with a stub `vllm`
on PATH, and compare the arguments vLLM would actually receive. Asserting on
the text of the script would have passed throughout the bug.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.parallelism import ParallelPlan

SPEC_JSON = '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'


def _argv(tmp_path, **config_kwargs) -> list:
    """The argv a real `vllm` would be started with by the generated script."""
    backend = EugrBackend.__new__(EugrBackend)
    backend.config = NodeConfig(model="org/model", **config_kwargs)
    script = Path(EugrBackend._write_launch_script(backend, ParallelPlan(), solo=True))

    # A stub that records exactly what it was handed.
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    out = tmp_path / "argv.json"
    stub = bindir / "vllm"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(out)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)

    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    result = subprocess.run(["bash", str(script)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert out.exists(), f"the script never reached vllm: {result.stderr[:400]}"
    return json.loads(out.read_text())


class TestArgumentsSurviveTheShell:
    def test_a_json_speculative_config_arrives_intact(self, tmp_path):
        argv = _argv(tmp_path, extra_vllm_args=["--speculative_config", SPEC_JSON])
        assert SPEC_JSON in argv
        # The failure mode, named so a regression is unmistakable.
        assert "method:qwen3_5_mtp" not in argv

    def test_the_json_still_parses_as_json(self, tmp_path):
        argv = _argv(tmp_path, extra_vllm_args=["--speculative_config", SPEC_JSON])
        value = argv[argv.index("--speculative_config") + 1]
        assert json.loads(value)["method"] == "qwen3_5_mtp"

    def test_a_dotted_speculative_model_arrives_intact(self, tmp_path):
        # Nemotron's recipe shape: a repo id as the value of a dotted flag.
        argv = _argv(tmp_path, extra_vllm_args=[
            "--speculative_config.model", "nvidia/Some-Model-DSpark"])
        assert "nvidia/Some-Model-DSpark" in argv

    @pytest.mark.parametrize("value", [
        '{"a":1,"b":2}',            # brace expansion
        "two words",                # word splitting
        "it's fine",                # a quote
        "star*glob",                # pathname expansion
        "$HOME",                    # parameter expansion
        "`whoami`",                 # command substitution
        "a;b",                      # command separator
    ])
    def test_hostile_values_reach_vllm_unchanged(self, tmp_path, value):
        argv = _argv(tmp_path, extra_vllm_args=["--flag", value])
        assert argv[argv.index("--flag") + 1] == value

    def test_an_api_alias_with_a_space_survives(self, tmp_path):
        argv = _argv(tmp_path, served_model_name=["My Model"])
        assert "My Model" in argv

    def test_the_model_id_survives(self, tmp_path):
        argv = _argv(tmp_path)
        assert "org/model" in argv

    def test_a_plain_launch_is_unchanged(self, tmp_path):
        argv = _argv(tmp_path, max_model_len=8192)
        assert "--max-model-len" in argv
        assert argv[argv.index("--max-model-len") + 1] == "8192"
        assert "--host" in argv and "0.0.0.0" in argv
