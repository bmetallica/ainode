"""A model already on disk must not be downloaded again.

AINode's downloader writes a flat `org--name` directory. Given a repo id,
vLLM resolves it through the Hugging Face cache, which uses `models--org--name`
— so the weights sitting right there in the mounted directory were invisible,
and the engine pulled all 23 GB again on every launch. Observed as "loading
takes extremely long", with the model directory already populated.

The NVIDIA backend has served from the path for exactly this reason since the
launch that "nuked the WAN on a TP=2 launch". The eugr backend, which is the
default one, never did.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import ENGINE_MODELS_DIR, EugrBackend
from ainode.engine.parallelism import ParallelPlan

MODEL = "unsloth/Qwen3.8-27B-NVFP4"
SLUG = "unsloth--Qwen3.8-27B-NVFP4"


def _downloaded(root: Path) -> Path:
    directory = root / SLUG
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    return directory


def _script(models_dir: Path, **kwargs) -> str:
    backend = EugrBackend.__new__(EugrBackend)
    backend.config = NodeConfig(model=MODEL, models_dir=str(models_dir), **kwargs)
    path = EugrBackend._write_launch_script(backend, ParallelPlan(), solo=True)
    return re.sub(r"\s+", " ", Path(path).read_text().replace("\\\n", " "))


class TestServingFromDisk:
    def test_a_downloaded_model_is_served_from_its_directory(self, tmp_path):
        _downloaded(tmp_path)
        script = _script(tmp_path)
        assert f"vllm serve {ENGINE_MODELS_DIR}/{SLUG}" in script

    def test_and_keeps_its_api_id(self, tmp_path):
        # Serving a path would otherwise publish the path as the model id, and
        # every client addressing the repo id would get a 404.
        _downloaded(tmp_path)
        assert f"--served-model-name {MODEL}" in _script(tmp_path)

    def test_an_absent_model_is_still_served_by_repo_id(self, tmp_path):
        script = _script(tmp_path)
        assert f"vllm serve {MODEL}" in script
        assert ENGINE_MODELS_DIR + "/" + SLUG not in script

    def test_an_empty_directory_does_not_count_as_downloaded(self, tmp_path):
        (tmp_path / SLUG).mkdir()
        assert f"vllm serve {MODEL}" in _script(tmp_path)

    def test_an_explicit_alias_wins_over_the_repo_id(self, tmp_path):
        _downloaded(tmp_path)
        script = _script(tmp_path, served_model_name=["chat"])
        assert "--served-model-name chat" in script
        assert f"--served-model-name {MODEL}" not in script


def _env_of(config) -> dict:
    """_build_env for a bare backend — it also consults the fabric topology,
    which has no business being detected in a unit test."""
    backend = EugrBackend.__new__(EugrBackend)
    backend.config = config
    from ainode.cluster.topology import Fabric, TopologyInfo

    backend._topology_cache = TopologyInfo(
        fabric=Fabric.UNKNOWN, links=[], coord_interface="eth0")
    return EugrBackend._build_env(backend)


class TestHfCacheMount:
    def test_hf_home_is_a_host_path(self, monkeypatch, tmp_path):
        # The launcher turns HF_HOME into a bind-mount source and the Docker
        # daemon resolves those on the host, not inside this container. Our own
        # view of the path mounted the host's /root/.ainode/models — an empty
        # directory belonging to root.
        # The unit sets this to the host directory it mounted at
        # AINODE_HOME — the .ainode directory itself, not the home above it.
        monkeypatch.setenv("AINODE_HOST_HOME", "/home/admin/.ainode")
        env = _env_of(NodeConfig(model=MODEL, models_dir="/root/.ainode/models"))
        assert env["HF_HOME"] == "/home/admin/.ainode/models"

    def test_without_a_host_home_it_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("AINODE_HOST_HOME", raising=False)
        env = _env_of(NodeConfig(model=MODEL, models_dir="/root/.ainode/models"))
        assert env["HF_HOME"] == "/root/.ainode/models"


class TestItStillRunsThroughBash:
    def test_the_path_form_survives_the_shell(self, tmp_path):
        _downloaded(tmp_path)
        backend = EugrBackend.__new__(EugrBackend)
        backend.config = NodeConfig(model=MODEL, models_dir=str(tmp_path))
        script = Path(EugrBackend._write_launch_script(backend, ParallelPlan(), solo=True))

        bindir = tmp_path / "bin"
        bindir.mkdir()
        out = tmp_path / "argv.json"
        stub = bindir / "vllm"
        stub.write_text("#!/usr/bin/env python3\nimport json,sys\n"
                        f"open({str(out)!r},'w').write(json.dumps(sys.argv[1:]))\n")
        stub.chmod(0o755)
        subprocess.run(["bash", str(script)],
                       env=dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}"),
                       capture_output=True, timeout=60)
        argv = json.loads(out.read_text())
        assert argv[0] == "serve"
        assert argv[1] == f"{ENGINE_MODELS_DIR}/{SLUG}"
        assert MODEL in argv
