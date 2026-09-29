"""The engine image is asked what it accepts, before a launch relies on it.

F1 and F2 in upgrade-fixes.md: a reasoning parser is added for a family that
thinks only when the image registers it, and a recipe or Advanced flag the
image does not have is refused before anything slow starts.
"""

from __future__ import annotations

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends import eugr
from ainode.engine.serve_flags import flag_problems, normalise
from ainode.models.reasoning_parsers import reasoning_args, reasoning_parser_for

#: A parser listing the size of vLLM's (the check refuses to judge by less).
OPTIONS = {f"--opt-{i}": None for i in range(150)}
OPTIONS.update({
    "--max-model-len": None, "--kv-cache-dtype": ["auto", "fp8", "fp8_ds_mla"],
    "--quantization": ["fp8", "modelopt_fp4", "awq"],
    "--reasoning-parser": ["deepseek_v4", "qwen3", "glm45"],
    "--reasoning-parser-plugin": None,
    "--enable-expert-parallel": None, "--no-enable-expert-parallel": None,
    "--compilation-config": None, "--speculative-config": None,
    "--trust-remote-code": None, "--tool-call-parser": None,
    "--enable-auto-tool-choice": None,
})


class TestTheFamily:
    @pytest.mark.parametrize("model, parser", [
        ("deepseek-ai/DeepSeek-V4-Flash-DSpark", "deepseek_v4"),
        ("Qwen/Qwen3.8-Flash-Next", "qwen3"),
        ("Qwen/Qwen3.5-122B-A10B-FP8", "qwen3"),
        ("zai-org/GLM-4.7-Flash", "glm45"),
        ("MiniMaxAI/MiniMax-M2.7", "minimax_m2"),
        ("openai/gpt-oss-120b", "openai_gptoss"),
        ("nvidia/Nemotron-3-Super-NVFP4", "nemotron_v3"),
    ])
    def test_a_family_that_thinks(self, model, parser):
        assert reasoning_parser_for(model) == parser

    @pytest.mark.parametrize("model", [
        "Qwen/Qwen3-Coder-Next", "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-235B-A22B-Instruct-2507", "nvidia/Nemotron-3-Nano-NVFP4",
        "meta-llama/Llama-3.3-70B-Instruct",
    ])
    def test_one_that_does_not_gets_nothing(self, model):
        assert reasoning_parser_for(model) == ""

    def test_a_finetune_is_known_by_its_architecture(self):
        # abacusai/Smaug-Flash: DeepSeek-V4-Flash under another name.
        assert reasoning_parser_for("abacusai/Smaug-Flash",
                                    "DeepseekV4ForCausalLM") == "deepseek_v4"


class TestOnlyWhatTheImageHas:
    def test_added_when_registered(self):
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next", [], ["qwen3"]) == \
            ["--reasoning-parser", "qwen3"]

    def test_not_when_the_image_lacks_it(self):
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next", [], ["deepseek_v4"]) == []

    def test_not_when_the_image_could_not_be_asked(self):
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next", [], None) == []
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next", [], []) == []

    def test_a_named_parser_is_not_second_guessed(self):
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next",
                              ["--reasoning-parser", "deepseek_r1"], ["qwen3"]) == []
        assert reasoning_args("Qwen/Qwen3.8-Flash-Next",
                              ["--reasoning_parser=x"], ["qwen3"]) == []


class TestTheFlagsAreChecked:
    def test_a_good_launch_passes(self):
        assert flag_problems(["--kv-cache-dtype", "fp8", "--max_model_len",
                              "131072", "--enable-expert-parallel",
                              "--compilation-config.mode", "3"], OPTIONS) == []

    def test_an_unknown_flag_is_named(self):
        problems = flag_problems(["--enable-magic", "--kv-cache-dtype", "fp8"],
                                 OPTIONS)
        assert len(problems) == 1 and "--enable-magic" in problems[0]

    def test_a_value_outside_the_choices_is_named(self):
        problems = flag_problems(["--quantization", "modelopt_mixed"], OPTIONS)
        assert problems and "modelopt_fp4" in problems[0]

    def test_the_equals_form_too(self):
        assert flag_problems(["--kv-cache-dtype=nvfp4_ds_mla"], OPTIONS)

    def test_a_plugin_widens_the_parser_list(self):
        assert flag_problems(["--reasoning-parser-plugin", "x.py",
                              "--reasoning-parser", "nano_v3"], OPTIONS) == []

    def test_a_thin_listing_judges_nothing(self):
        assert flag_problems(["--anything"], {"--max-model-len": None}) == []
        assert flag_problems(["--anything"], None) == []

    def test_spellings_are_one_flag(self):
        assert normalise("--max_model_len=4") == "--max-model-len"
        assert normalise("--compilation-config.mode") == "--compilation-config"


class TestTheLaunchScript:
    def _backend(self, monkeypatch, found, model="Qwen/Qwen3.8-Flash-Next",
                 args=()):
        cfg = NodeConfig(model=model, api_port=8000,
                         extra_vllm_args=list(args))
        backend = eugr.EugrBackend(cfg)
        monkeypatch.setattr(eugr.EugrBackend, "_engine_probe",
                            lambda self: found)
        monkeypatch.setattr(eugr, "detect_gpu", lambda: None)
        return backend

    def _script(self, backend, tmp_path, monkeypatch):
        monkeypatch.setattr(eugr, "EUGR_LAUNCHER", tmp_path / "launch-cluster.sh")
        from ainode.engine.parallelism import ParallelPlan

        path = backend._write_launch_script(ParallelPlan(), solo=True)
        return path.read_text()

    def test_the_parser_is_added_and_recorded(self, monkeypatch, tmp_path):
        backend = self._backend(monkeypatch, {"reasoning_parsers": ["qwen3"]})
        script = self._script(backend, tmp_path, monkeypatch)
        assert "--reasoning-parser qwen3" in script
        # So the opencode config reads "reasoning": true for it.
        assert "--reasoning-parser" in backend.config.extra_vllm_args

    def test_switched_off_it_is_not(self, monkeypatch, tmp_path):
        backend = self._backend(monkeypatch, {"reasoning_parsers": ["qwen3"]})
        backend.config.auto_reasoning_parser = False
        assert "--reasoning-parser" not in self._script(backend, tmp_path,
                                                        monkeypatch)

    def test_a_bad_flag_stops_the_launch_before_it_starts(self, monkeypatch,
                                                           tmp_path):
        backend = self._backend(monkeypatch, {"options": OPTIONS,
                                              "vllm_version": "0.12.0"},
                                args=["--quantization", "modelopt_mixed"])
        with pytest.raises(eugr.EugrBackendError) as caught:
            self._script(backend, tmp_path, monkeypatch)
        assert "0.12.0" in str(caught.value)
        assert "Launch anyway" in str(caught.value)

    def test_launch_anyway_goes_past_it(self, monkeypatch, tmp_path):
        backend = self._backend(monkeypatch, {"options": OPTIONS},
                                args=["--quantization", "modelopt_mixed"])
        backend.skip_flag_check = True
        assert "modelopt_mixed" in self._script(backend, tmp_path, monkeypatch)


class TestTheProbe:
    def test_an_image_docker_does_not_know_is_not_asked(self, monkeypatch):
        from ainode.engine import image_probe

        monkeypatch.setattr(image_probe, "image_id", lambda image: "")
        monkeypatch.setattr(image_probe, "_run",
                            lambda *a: pytest.fail("ran a container"))
        assert image_probe.probe("vllm-node:latest") == {}

    def test_an_answer_is_kept_by_image_id(self, monkeypatch, tmp_path):
        import ainode.core.config as config
        from ainode.engine import image_probe

        monkeypatch.setattr(config, "AINODE_HOME", tmp_path)
        monkeypatch.setattr(image_probe, "_MEMO", {})
        monkeypatch.setattr(image_probe, "image_id", lambda image: "sha256:abc")
        runs = []
        monkeypatch.setattr(image_probe, "_run", lambda image, timeout: runs.append(1)
                            or {"reasoning_parsers": ["qwen3"]})
        assert image_probe.probe("vllm-node:latest")["reasoning_parsers"] == ["qwen3"]
        image_probe._MEMO.clear()                # a restart
        assert image_probe.probe("vllm-node:latest")["reasoning_parsers"] == ["qwen3"]
        assert len(runs) == 1

    def test_the_answer_line_is_found_among_the_import_noise(self, monkeypatch):
        from ainode.engine import image_probe

        class _Done:
            returncode = 0
            stderr = ""
            stdout = ("INFO platform detected\n"
                      'AINODE_PROBE {"reasoning_parsers": ["qwen3"]}\n')

        monkeypatch.setattr(image_probe.subprocess, "run", lambda *a, **k: _Done())
        assert image_probe._run("x", 1) == {"reasoning_parsers": ["qwen3"]}
