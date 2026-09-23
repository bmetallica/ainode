"""Nothing hands `vllm serve` an empty model name.

Reported while restoring a profile:

    HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'.
    The name cannot start or end with '-' or '.' and the maximum length is
    96: ''.

An empty repo id, from an instance whose config had no model. The engine
turned it into a message about a name that is not a name, which says nothing
about the instance that has nothing to serve — and the launch had already
spent a container start getting there.
"""

from __future__ import annotations

import inspect

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend, EugrBackendError


class TestTheLaunchRefusesIt:
    def test_an_empty_model_is_refused_before_the_engine(self, tmp_path):
        from ainode.engine.parallelism import ParallelPlan

        backend = EugrBackend(NodeConfig(node_id="n1", model="",
                                         models_dir=str(tmp_path)))
        with pytest.raises(EugrBackendError) as caught:
            backend._write_launch_script(ParallelPlan(), solo=True)
        assert "no model to serve" in str(caught.value)

    def test_a_model_that_is_there_is_not(self, tmp_path):
        from ainode.engine.parallelism import ParallelPlan

        backend = EugrBackend(NodeConfig(node_id="n1", model="org/m",
                                         models_dir=str(tmp_path)))
        script = backend._write_launch_script(ParallelPlan(), solo=True)
        assert "vllm serve" in script.read_text()

    def test_the_guard_is_where_the_command_is_built(self):
        source = inspect.getsource(EugrBackend._write_launch_script)
        assert "has no model to serve" in source
        # Before anything is written: a refusal must not leave a half-built
        # script behind.
        assert source.index("serve_target, implied_name") < source.index(
            "has no model to serve")

    def test_it_says_where_to_look(self):
        source = inspect.getsource(EugrBackend._write_launch_script)
        assert "if this came from a profile" in source


class TestTheCommandIsRecorded:
    def test_the_serve_line_is_logged(self):
        source = inspect.getsource(EugrBackend._log_serve_command)
        assert "serve command" in source

    def test_the_solo_launch_calls_it(self):
        assert "_log_serve_command" in inspect.getsource(EugrBackend.start_solo)

    def test_it_joins_a_continued_command(self, tmp_path, caplog):
        script = tmp_path / "launch.sh"
        script.write_text(
            "#!/bin/bash\nset -e\n"
            "vllm serve /models/org--m \\\n"
            "    --port 8000 \\\n"
            "    --served-model-name org/m\n")
        backend = EugrBackend(NodeConfig(node_id="n1", model="org/m"))
        with caplog.at_level("INFO", logger="ainode.engine.backends.eugr"):
            backend._log_serve_command(script)
        assert "vllm serve /models/org--m --port 8000 --served-model-name org/m" \
            in caplog.text

    def test_a_missing_script_is_not_an_error(self, tmp_path):
        backend = EugrBackend(NodeConfig(node_id="n1", model="org/m"))
        backend._log_serve_command(tmp_path / "nope.sh")   # must not raise


class TestCaptureSkipsAnInstanceWithNoModel:
    def test_the_local_path_guards_like_every_peer_path(self):
        from ainode.profiles.apply import local_launch_specs

        source = inspect.getsource(local_launch_specs)
        assert 'getattr(record, "model", "")' in source
        assert "continue" in source

    def test_a_profile_entry_still_requires_one(self):
        from ainode.profiles.store import ProfileEntry, ProfileError

        with pytest.raises(ProfileError):
            ProfileEntry(model="")
