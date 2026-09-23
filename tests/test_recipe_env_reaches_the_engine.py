"""A recipe's environment has to reach the engine that runs the model.

The catalog entry for unsloth/Qwen3.8-27B-NVFP4 carries two things about the
InstantTensor loader:

    "--load-format", "instanttensor",          # extra_vllm_args
    "INSTANTTENSOR_BUFFER_SIZE": "67108864",   # extra_env

and the comment beside the second describes precisely the failure it
prevents — the same error text, the same budget that moves between runs, the
same hardware. On the cluster only the first arrived. The model failed on
every launch with a 4.7 GB staging buffer it had been configured not to ask
for, and the way through was to drop the loader the recipe had chosen
deliberately.

NvidiaBackend applies extra_env. DiffusersBackend applies extra_env. The
eugr backend — the one this deployment runs — built its docker arguments
without ever reading it.
"""

from __future__ import annotations

import inspect

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend, EugrBackendError


def _backend(**env):
    config = NodeConfig(node_id="n1", model="org/m", models_dir="/models",
                        extra_env=dict(env))
    return EugrBackend(config)


class TestTheEnvironmentTravels:
    def test_each_entry_becomes_a_docker_flag(self):
        args = _backend(INSTANTTENSOR_BUFFER_SIZE="67108864")._recipe_env_args()
        assert args == ["-e", "INSTANTTENSOR_BUFFER_SIZE=67108864"]

    def test_several_entries_all_travel(self):
        args = _backend(A="1", B="2")._recipe_env_args()
        assert args.count("-e") == 2
        assert "A=1" in args and "B=2" in args

    def test_no_recipe_environment_adds_nothing(self):
        assert _backend()._recipe_env_args() == []

    def test_it_reaches_the_launcher_arguments(self):
        env = _backend(INSTANTTENSOR_BUFFER_SIZE="67108864")._launcher_env()
        assert "-e INSTANTTENSOR_BUFFER_SIZE=67108864" in \
            env["VLLM_SPARK_EXTRA_DOCKER_ARGS"]

    def test_the_mounts_are_still_there(self):
        # Added alongside, not instead of.
        env = _backend(A="1")._launcher_env()
        assert "/models:" in env["VLLM_SPARK_EXTRA_DOCKER_ARGS"]


class TestItIsValidatedLikeEveryOtherUnquotedArgument:
    """VLLM_SPARK_EXTRA_DOCKER_ARGS is expanded unquoted by the launcher."""

    @pytest.mark.parametrize("key,value", [
        ("A", "1 --privileged"),
        ("A", "$(id)"),
        ("A;B", "1"),
        ("A", "1;rm -rf /"),
    ])
    def test_a_metacharacter_is_refused(self, key, value):
        with pytest.raises(EugrBackendError) as caught:
            _backend(**{key: value})._recipe_env_args()
        assert "unquoted" in str(caught.value)

    def test_an_ordinary_value_passes(self):
        assert _backend(VLLM_USE_V1="1")._recipe_env_args() == \
            ["-e", "VLLM_USE_V1=1"]


class TestEveryBackendAppliesIt:
    """Two of the three did. The one this cluster runs did not."""

    def test_the_eugr_backend_does(self):
        assert "extra_env" in inspect.getsource(EugrBackend._recipe_env_args)

    def test_the_nvidia_backend_does(self):
        from ainode.engine.backends.nvidia import NvidiaBackend

        assert "extra_env" in inspect.getsource(NvidiaBackend)

    def test_the_diffusers_backend_does(self):
        from ainode.engine.backends.diffusers import DiffusersBackend

        assert "extra_env" in inspect.getsource(DiffusersBackend)


class TestTheRecipeStillCarriesBoth:
    def test_qwen_has_the_loader_and_its_cap(self):
        from ainode.models.registry import CURATED_CLUSTER_MODELS

        entry = CURATED_CLUSTER_MODELS["qwen3.8-27b-nvfp4"]
        assert "instanttensor" in entry.extra_vllm_args
        assert entry.extra_env["INSTANTTENSOR_BUFFER_SIZE"] == "67108864"

    def test_and_so_do_the_others_that_use_it(self):
        from ainode.models.registry import CURATED_CLUSTER_MODELS

        for entry in CURATED_CLUSTER_MODELS.values():
            if "instanttensor" in (entry.extra_vllm_args or []):
                assert (entry.extra_env or {}).get("INSTANTTENSOR_BUFFER_SIZE"), \
                    f"{entry.id} chooses the loader without capping its buffer"


class TestTheLaunchSaysWhatItPassed:
    """The environment does not appear in the serve command — it travels as
    `docker -e` — so a launch that lost it looks identical to one that
    carried it. Which is exactly how INSTANTTENSOR_BUFFER_SIZE went missing
    for five days without anyone being able to see it."""

    def test_the_banner_names_the_environment(self, tmp_path):
        script = tmp_path / "launch.sh"
        script.write_text("vllm serve /models/org--m\n")
        log = tmp_path / "vllm.log"
        _backend(INSTANTTENSOR_BUFFER_SIZE="67108864")._log_serve_command(
            script, log)
        assert "recipe environment: INSTANTTENSOR_BUFFER_SIZE=67108864" in \
            log.read_text()

    def test_no_environment_says_so_rather_than_nothing(self, tmp_path):
        # A blank line reads as "not implemented"; "(none)" reads as an answer.
        script = tmp_path / "launch.sh"
        script.write_text("vllm serve /models/org--m\n")
        log = tmp_path / "vllm.log"
        _backend()._log_serve_command(script, log)
        assert "recipe environment: (none)" in log.read_text()

    def test_it_is_sorted_so_two_launches_can_be_compared(self, tmp_path):
        script = tmp_path / "launch.sh"
        script.write_text("vllm serve /models/org--m\n")
        log = tmp_path / "vllm.log"
        _backend(B="2", A="1")._log_serve_command(script, log)
        assert "recipe environment: A=1, B=2" in log.read_text()
