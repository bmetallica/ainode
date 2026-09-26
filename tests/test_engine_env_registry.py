"""A knob the engine does not read is a knob that did nothing.

From the cluster's own launch log:

    WARNING [interface.py:1274] Unknown vLLM environment variable detected:
    VLLM_BASE_DIR

That is the whole feedback, once, mid-launch, between a Triton notice and a
NCCL banner. AINode passed extra_env through blind — a catalog recipe and an
Advanced field both land in `docker -e` and neither was checked against the
image they were going to.

vLLM keeps the answer itself, at vllm.envs.environment_variables, so this asks
the image rather than maintaining a list here — which on a rolling engine
build would be wrong within the week.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ainode.engine import engine_env

KNOWN = {"VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_LOGGING_LEVEL"}


@pytest.fixture(autouse=True)
def _no_memo():
    engine_env._MEMO.clear()
    yield
    engine_env._MEMO.clear()


class TestReadingTheRegistryOutOfTheImage:
    def _run(self, stdout, code=0):
        class _Done:
            returncode = code
            stderr = ""

        _Done.stdout = stdout
        return _Done()

    def test_it_parses_the_json_the_probe_prints(self):
        with patch("subprocess.run",
                   return_value=self._run(json.dumps(sorted(KNOWN)))):
            assert engine_env.probe_image_env("img") == KNOWN

    def test_import_noise_before_the_json_is_ignored(self):
        # The image prints Triton and platform notices on import; the answer
        # is the last line that parses, not the whole of stdout.
        noisy = ("INFO [importing.py:74] Triton is installed but 0 active "
                 "driver(s) found.\n"
                 "INFO [importing.py:98] Triton not installed.\n"
                 + json.dumps(sorted(KNOWN)) + "\n")
        with patch("subprocess.run", return_value=self._run(noisy)):
            assert engine_env.probe_image_env("img") == KNOWN

    def test_a_failed_probe_is_empty_not_wrong(self):
        with patch("subprocess.run", return_value=self._run("boom", code=1)):
            assert engine_env.probe_image_env("img") == set()

    def test_docker_being_unavailable_is_not_an_exception(self):
        with patch("subprocess.run", side_effect=OSError("no docker")):
            assert engine_env.probe_image_env("img") == set()

    def test_no_image_is_no_probe(self):
        with patch("subprocess.run", side_effect=AssertionError("asked")):
            assert engine_env.probe_image_env("") == set()


class TestTheVerdict:
    def test_a_name_outside_the_registry_is_named(self):
        assert engine_env.unregistered_names(
            {"VLLM_BASE_DIR": "/x", "VLLM_USE_V1": "1"}, KNOWN) \
            == ["VLLM_BASE_DIR"]

    def test_everything_registered_says_nothing(self):
        assert engine_env.unregistered_names({"VLLM_USE_V1": "1"}, KNOWN) == []

    @pytest.mark.parametrize("name", ["NCCL_IB_HCA", "HF_HUB_OFFLINE",
                                      "INSTANTTENSOR_BUFFER_SIZE",
                                      "TRITON_CACHE_DIR", "TORCH_CUDA_ARCH_LIST"])
    def test_only_vllm_names_are_judged(self, name):
        """Everything else is read by a library that keeps no registry.
        Calling those unknown would be a false alarm on every launch that does
        anything interesting — INSTANTTENSOR_BUFFER_SIZE among them."""
        assert engine_env.unregistered_names({name: "1"}, KNOWN) == []

    def test_an_unprobed_image_gives_no_verdict(self):
        # Empty means "could not ask", never "reads nothing".
        assert engine_env.unregistered_names({"VLLM_BASE_DIR": "/x"}, set()) == []

    def test_a_list_works_as_well_as_a_mapping(self):
        assert engine_env.unregistered_names(["VLLM_BASE_DIR"], KNOWN) \
            == ["VLLM_BASE_DIR"]


class TestTheSentenceTheOperatorSees:
    def _warn(self, env):
        with patch.object(engine_env, "known_env_names", lambda *a, **k: KNOWN):
            return engine_env.env_warning(env, "img")

    def test_it_names_the_variable(self):
        assert "VLLM_BASE_DIR" in self._warn({"VLLM_BASE_DIR": "/x"})

    def test_it_says_the_launch_still_happens(self):
        # Not a refusal: the setting is inert, the model still serves.
        assert "do\nnothing" in self._warn({"VLLM_BASE_DIR": "/x"}).replace(" ", "\n")

    def test_it_agrees_with_itself_about_number(self):
        one = self._warn({"VLLM_BASE_DIR": "/x"})
        two = self._warn({"VLLM_BASE_DIR": "/x", "VLLM_NOPE": "1"})
        assert " is not read" in one and " are not read" in two

    def test_silence_when_there_is_nothing_to_say(self):
        assert self._warn({"VLLM_USE_V1": "1"}) == ""
        assert self._warn({}) == ""


class TestTheCache:
    def test_a_probe_is_not_repeated_within_the_process(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setattr(engine_env, "_cache_path",
                            lambda image: tmp_path / "x.json")
        calls = []

        def _probe(image, timeout=120):
            calls.append(image)
            return KNOWN

        monkeypatch.setattr(engine_env, "probe_image_env", _probe)
        engine_env.known_env_names("img")
        engine_env.known_env_names("img")
        assert calls == ["img"]

    def test_a_stale_file_is_re_probed(self, tmp_path, monkeypatch):
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"at": 0, "names": ["VLLM_OLD"]}))
        monkeypatch.setattr(engine_env, "_cache_path", lambda image: path)
        monkeypatch.setattr(engine_env, "probe_image_env",
                            lambda image, timeout=120: KNOWN)
        assert engine_env.known_env_names("img") == KNOWN

    def test_a_failed_probe_is_not_cached_as_an_answer(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(engine_env, "_cache_path",
                            lambda image: tmp_path / "x.json")
        monkeypatch.setattr(engine_env, "probe_image_env",
                            lambda image, timeout=120: set())
        assert engine_env.known_env_names("img") == set()
        assert not (tmp_path / "x.json").exists()


class TestTheLaunchBannerSaysIt:
    def test_the_backend_checks_the_recipe_environment(self):
        import inspect

        from ainode.engine.backends import eugr

        source = inspect.getsource(eugr)
        assert "env_warning" in source
        assert "ignored by this image" in source

    def test_it_asks_about_the_image_that_actually_serves(self):
        # This backend serves on the launcher's vllm-node whatever a catalog
        # names, so asking config.engine_image would probe the wrong thing.
        import inspect

        from ainode.engine.backends import eugr

        source = inspect.getsource(eugr.EugrBackend._engine_image_for_env)
        assert "vllm-node" in source
