"""Image generation, as an engine AINode drives like any other.

The whole point of implementing ``EngineBackend`` rather than running a
container beside AINode: everything the platform does around instances hangs
off this interface and the InstanceManager, and whatever satisfies it inherits
the lot — the card with its status word, the memory guard that will kill it
before it takes the node down, the admission gate, the load phases with their
timings, the error assistant, placement, profiles, log forwarding, telemetry.

On a unified-memory node that is not a convenience. A thirty-gigabyte process
outside AINode's view shares the same physical pool as the models AINode is
watching, and the guard would then kill the wrong instance.

The container is started directly with ``docker run`` and not through eugr's
launcher: there is no Ray cluster, no rank, no NCCL — one process on one node.
The serving script is mounted rather than baked in, so changing how images are
served does not mean rebuilding a twenty-gigabyte image.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ainode.core.config import LOGS_DIR, NodeConfig, host_path
from ainode.engine.backends.base import EngineBackend
from ainode.engine.load_phase import LoadPhaseTracker

logger = logging.getLogger(__name__)

__all__ = ["DiffusersBackend", "DiffusersBackendError", "DEFAULT_IMAGE"]

DEFAULT_IMAGE = "ainode-diffusers:latest"

#: Where models_dir is mounted in the engine container — the same path the
#: vLLM engine uses, so a model directory has one name across both engines.
ENGINE_MODELS_DIR = "/models"

#: Where the server script is mounted.
SERVER_CONTAINER_PATH = "/workspace/ainode_image_server.py"

SERVER_SOURCE = Path(__file__).resolve().parent.parent / "diffusers_server.py"


class DiffusersBackendError(RuntimeError):
    """The image engine could not be started."""


class DiffusersBackend(EngineBackend):
    """Runs one diffusers pipeline in a container, serving images."""

    CONTAINER_BASENAME = "ainode_image"

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None,
                 instance_id: str = ""):
        self.config = config
        self.on_ready = on_ready
        self._instance_id = str(instance_id or "").strip()
        self._process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._ready = False
        self._phase = LoadPhaseTracker()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        token = self._instance_id
        self._log_file: Path = LOGS_DIR / (f"images-{token}.log" if token
                                           else "images.log")

    # -- identity ----------------------------------------------------------

    @property
    def container_name(self) -> str:
        token = self._instance_id
        return f"{self.CONTAINER_BASENAME}-{token}" if token \
            else self.CONTAINER_BASENAME

    @property
    def engine_image(self) -> str:
        return (getattr(self.config, "engine_image", "") or "").strip() \
            or DEFAULT_IMAGE

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"

    @property
    def log_path(self) -> Path:
        return self._log_file

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @process.setter
    def process(self, value) -> None:
        self._process = value

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        if self.is_running():
            return True
        image = self.engine_image
        if not _image_present(image):
            # It is built locally by scripts/build-diffusers-image.sh and is
            # in no registry, so a pull was never going to work. Say what to
            # do instead of failing at one.
            raise DiffusersBackendError(
                f"The engine image {image} is not on this node, and it cannot "
                f"be pulled — it is built locally. On the head, run "
                f"scripts/build-diffusers-image.sh and then "
                f"scripts/update-cluster.sh --images --nodes <peers>.")

        model_path = self._model_path()
        if model_path is None:
            raise DiffusersBackendError(
                f"{self.config.model} is not on this node's disk. Download it "
                f"first — an image pipeline is a directory of several "
                f"components, not one file the engine can fetch on demand.")

        script = self._stage_server_script()
        self._remove_container()

        cmd = [
            "docker", "run", "--rm", "--name", self.container_name,
            "--network", "host", "--gpus", "all", "--ipc", "host",
            "-v", f"{host_path(str(self.config.models_dir))}:{ENGINE_MODELS_DIR}",
            "-v", f"{host_path(str(script))}:{SERVER_CONTAINER_PATH}:ro",
        ]
        for key, value in (getattr(self.config, "extra_env", None) or {}).items():
            cmd += ["-e", f"{key}={value}"]
        # python3: the engine image has python3/python3-pip and no
        # unversioned alias, so `python` is not on PATH in it.
        cmd += [image, "python3", SERVER_CONTAINER_PATH,
                "--model-path", model_path,
                "--served-model-name", self.config.model or "",
                "--port", str(self.config.api_port),
                "--dtype", _dtype(self.config),
                "--max-image-size", str(_max_image_size(self.config)),
                "--steps", str(_int_option(self.config, "image_steps", 20)),
                "--size", _str_option(self.config, "image_size", "1024x1024")]

        logger.info("Starting the image engine: %s", " ".join(cmd))
        self._phase.reset()
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True)
        self._log_thread = threading.Thread(
            target=self._stream_logs, args=(self._process, self._log_file),
            daemon=True)
        self._log_thread.start()
        return self._process.poll() is None

    def stop(self) -> None:
        self._remove_container(graceful=True)
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=15)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    logger.debug("could not kill the image engine",
                                 exc_info=True)
        self._process = None
        self._ready = False
        self._phase.reset()

    def kill(self) -> None:
        """Immediately. Called by the host memory guard, which has seconds."""
        self._remove_container(graceful=False)
        if self._process and self._process.poll() is None:
            try:
                self._process.kill()
            except Exception:
                logger.debug("could not kill the image engine", exc_info=True)
        self._process = None
        self._ready = False

    def wait_ready(self, timeout: float = 600.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            if self._probe():
                self._ready = True
                self._phase.mark_ready()
                if self.on_ready:
                    try:
                        self.on_ready()
                    except Exception:  # pragma: no cover
                        logger.exception("on_ready callback failed")
                return True
            time.sleep(2)
        return False

    def is_running(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        # The container outlives the docker-run process in some teardowns, and
        # an instance whose container is up is running whatever this object
        # thinks.
        return _container_running(self.container_name)

    def health_check(self) -> dict:
        return {
            "process_alive": self.is_running(),
            "api_responding": self._probe(),
            "models_loaded": [self.config.model] if self._ready else [],
            "container": self.container_name,
            "image": self.engine_image,
        }

    def logs(self, n: int = 100) -> str:
        if not self._log_file.exists():
            return ""
        try:
            return "\n".join(self._log_file.read_text().splitlines()[-n:])
        except OSError:
            return ""

    # -- load state, read by the card, the assistant and telemetry ----------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_phase(self) -> str:
        return self._phase.current(ready_latch=self._ready)

    @property
    def load_detail(self) -> str:
        return self._phase.detail

    @property
    def load_error(self) -> str:
        return self._phase.failure_reason()

    @property
    def load_timeline(self) -> list:
        return self._phase.timeline()

    @property
    def load_seconds(self) -> float:
        return self._phase.elapsed

    # -- internals ---------------------------------------------------------

    def _model_path(self) -> Optional[str]:
        """The model directory as the CONTAINER sees it, or None."""
        from ainode.models.registry import ModelManager

        repo = self.config.model or ""
        try:
            directories = ModelManager(
                models_dir=str(self.config.models_dir)).model_dirs_for_repo(repo)
        except Exception:
            logger.debug("could not locate %s", repo, exc_info=True)
            directories = []
        for directory in directories:
            # A diffusers pipeline is a directory with model_index.json at its
            # root; the hub-cache layout nests it under snapshots/<hash>.
            for candidate in (directory, *sorted(
                    (directory / "snapshots").glob("*")
                    if (directory / "snapshots").is_dir() else [])):
                if (candidate / "model_index.json").is_file():
                    relative = candidate.relative_to(Path(self.config.models_dir))
                    return f"{ENGINE_MODELS_DIR}/{relative}"
        return None

    def _stage_server_script(self) -> Path:
        """Put the server where the container can mount it."""
        target = LOGS_DIR.parent / "image-server" / "ainode_image_server.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SERVER_SOURCE, target)
        return target

    def _probe(self) -> bool:
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, OSError):
            return False

    def _remove_container(self, graceful: bool = False) -> None:
        name = self.container_name
        commands = ([["docker", "stop", "-t", "20", name]] if graceful else
                    [["docker", "kill", name]])
        commands.append(["docker", "rm", "-f", name])
        for command in commands:
            try:
                subprocess.run(command, capture_output=True, text=True,
                               timeout=60 if graceful else 20)
            except Exception:
                logger.debug("%s failed", " ".join(command), exc_info=True)

    def _stream_logs(self, process: subprocess.Popen, target: Path) -> None:
        if not process.stdout:
            return
        with open(target, "a") as sink:
            for line in process.stdout:
                sink.write(line)
                sink.flush()
                if self._phase.observe(line) and not self._ready:
                    self._ready = True
                    if self.on_ready:
                        try:
                            self.on_ready()
                        except Exception:  # pragma: no cover
                            logger.exception("on_ready callback failed")
        if not self._ready:
            try:
                code = process.wait(timeout=10)
            except Exception:
                code = None
            self._phase.fail(f"the image engine exited (code {code})"
                             if code is not None
                             else "the image engine stopped producing output")


# --- helpers -----------------------------------------------------------------

def _image_present(image: str) -> bool:
    try:
        result = subprocess.run(["docker", "image", "inspect", image],
                                capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        logger.debug("could not inspect %s", image, exc_info=True)
        return False


def _container_running(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=20)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def _dtype(config) -> str:
    value = (getattr(config, "image_dtype", "") or "").strip()
    return value or "bfloat16"


def _max_image_size(config) -> int:
    return _int_option(config, "max_image_size", 1536)


def _int_option(config, name: str, default: int) -> int:
    try:
        value = int(getattr(config, name, None) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _str_option(config, name: str, default: str) -> str:
    return str(getattr(config, name, "") or default)

