"""EugrBackend — drive vLLM inside the AINode unified container image
via eugr/spark-vllm-docker's launch-cluster.sh.

Running *inside* the container (which is how the image ships — systemd on the
host runs ``docker run ... ainode`` once), this backend handles two modes:

* ``start_solo()`` — one engine container on this host, via the launcher's
  ``--solo`` mode. Used when ``config.distributed_mode == "solo"``.
* ``start_distributed()`` — shells out to ``/opt/spark-vllm-docker/launch-
  cluster.sh`` (baked into the image) to SSH-orchestrate peer workers and
  form a Ray cluster across nodes. Used when ``config.distributed_mode ==
  "head"``. Refuses to run in any other mode.

Public surface mirrors :class:`ainode.engine.vllm_engine.VLLMEngine` —
``start``, ``stop``, ``wait_ready``, ``is_running``, ``health_check``,
``api_url``, ``log_path``, ``process`` — so ``cmd_start``/``cmd_status``
can dispatch polymorphically via a factory.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from ainode.cluster import hca_discovery
from ainode.cluster.topology import (
    TopologyInfo,
    is_safe_device_name,
    topology_for_config,
)
from ainode.core.config import LOGS_DIR, NodeConfig, host_path
from ainode.core.gpu import detect_gpu
from ainode.engine.backends.base import EngineBackend
from ainode.engine.load_phase import LoadPhaseTracker
from ainode.engine.distribute import (
    DistributionError,
    ensure_peer_has_dir,
    ensure_local_image,
    ensure_peer_has_image,
    hf_cache_dir_name,
)
from ainode.engine.parallelism import ParallelPlan, Strategy

logger = logging.getLogger(__name__)

# Path to eugr's launcher inside the unified image. See scripts/Dockerfile.ainode.
# Where the host's models_dir is mounted inside every engine container. The
# launcher passes one VLLM_SPARK_EXTRA_DOCKER_ARGS to every node, so this path
# is the same everywhere and is the only one the serve script can reference —
# AINode's own view of models_dir does not exist in that container.
ENGINE_MODELS_DIR = "/models"

EUGR_LAUNCHER = Path("/opt/spark-vllm-docker/launch-cluster.sh")
EUGR_ENV_FILE = Path("/opt/spark-vllm-docker/.env")

# Per-node NCCL init shim. Baked into the ainode image via Dockerfile COPY;
# AINode copies it onto shared storage at launch so every peer's vllm_node
# container can mount it via ``-v`` and run it via ``--entrypoint``. The shim
# detects each node's local HCAs (mlx5_* or rocep*/roceP*), filters by Up
# state and cluster subnet, and exports NCCL_IB_HCA before exec-ing CMD.
#
# TODO(v0.4.10): bake the shim into ainode-base so every vllm_node has it at
# /usr/local/bin/nccl-env-init.sh without needing NFS distribution. Then the
# NCCL_INIT_SHARED_* paths and ``_publish_nccl_init_script`` become optional.
NCCL_INIT_IMAGE_PATH = Path("/usr/local/bin/nccl-env-init.sh")
NCCL_INIT_SHARED_DIR = Path("/mnt/shared-models/.ainode")
NCCL_INIT_SHARED_PATH = NCCL_INIT_SHARED_DIR / "nccl-env-init.sh"
# In-container path used by ``--entrypoint`` AND by the head's exec-script source hook.
NCCL_INIT_CONTAINER_PATH = "/mnt/shared-models/.ainode/nccl-env-init.sh"


# Characters a path may contain before it is handed to launch-cluster.sh.
# ``VLLM_SPARK_EXTRA_DOCKER_ARGS`` is appended to ``DOCKER_ARGS`` and expanded
# UNQUOTED into the ``docker run`` line, so whitespace in a path does not just
# break the mount — it injects additional docker flags. ``models_dir`` is
# settable over ``PATCH /api/config``, which is unauthenticated unless the
# operator enabled auth, and a flag like ``-v /:/host`` or ``--privileged``
# there is a host compromise. Backslash and quotes are excluded too: the value
# passes through a shell, and a plausible model path needs none of them.
_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./@+:-]+$")


def _is_safe_path_arg(path: str) -> bool:
    """True if ``path`` is safe to interpolate into the launcher's docker args."""
    return bool(isinstance(path, str) and path and _SAFE_PATH_RE.match(path))


class EugrBackendError(RuntimeError):
    """Raised when the backend cannot be driven (missing binary, bad config)."""


# Back-compat alias — old code imports ``DockerEngineError`` from the
# ``docker_engine`` shim, which in turn re-exports this name.
DockerEngineError = EugrBackendError


class EugrBackend(EngineBackend):
    """Single backend that handles solo + head invocation of vLLM via eugr.

    Stateful only on the current process instance — the actual engine state
    (GPU memory, Ray cluster, peer containers) lives in the OS.
    """

    def __init__(self, config: NodeConfig, on_ready: Optional[Callable] = None):
        self.config = config
        self.on_ready = on_ready
        self._process: Optional[subprocess.Popen] = None
        self._ready = False
        self._log_thread: Optional[threading.Thread] = None
        # Fabric wiring, resolved lazily on first use — see _topology().
        self._topology_cache: Optional[TopologyInfo] = None
        self._phase = LoadPhaseTracker()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file: Path = LOGS_DIR / "vllm.log"
        self._distributed_log: Path = LOGS_DIR / "distributed.log"

    # ------------------------------------------------------------------
    # Public API (lifecycle)
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the engine in the mode dictated by ``config.distributed_mode``.

        Returns True if the launch sequence was kicked off successfully;
        False if validation failed or subprocess couldn't spawn. Use
        ``wait_ready()`` to block until the API serves.
        """
        mode = (self.config.distributed_mode or "solo").lower()
        if mode == "solo":
            return self.start_solo()
        if mode == "head":
            return self.start_distributed()
        raise EugrBackendError(
            f"Unknown distributed_mode={mode!r}; expected 'solo' or 'head'. "
            "Workers are launched by the head via eugr's launcher — they don't "
            "run a full ainode process directly."
        )

    def start_solo(self) -> bool:
        """Run a single-node ``vllm serve`` in an engine container.

        Goes through the launcher in ``--solo`` mode rather than spawning
        ``vllm`` directly. The direct spawn is a leftover from the unified
        image: AINode's own container is now ``python:3.12-slim`` with no CUDA,
        no vLLM and no NCCL, so ``Popen(["vllm", ...])`` there fails with
        ``[Errno 2] No such file or directory: 'vllm'``. The engine lives in
        its own container — the same one the distributed path uses.

        ``--solo`` skips peer discovery and Ray; the container runs with
        ``--network host``, so the engine binds ``api_port`` on the host
        directly and no port mapping is needed.
        """
        if self.is_running():
            return True
        if not EUGR_LAUNCHER.exists():
            raise EugrBackendError(
                f"eugr launcher missing at {EUGR_LAUNCHER}. Is this running "
                "inside the ainode image?"
            )

        launch_script = self._write_launch_script(ParallelPlan(), solo=True)
        cmd = [str(EUGR_LAUNCHER), "--solo", *self._launcher_image_args(),
               "--launch-script", str(launch_script)]
        env = self._launcher_env()

        logger.info("Starting solo vLLM via the launcher: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=str(EUGR_LAUNCHER.parent),
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs, args=(self._process, self._log_file), daemon=True
        )
        self._log_thread.start()
        return self._process.poll() is None

    def start_distributed(self) -> bool:
        """Invoke eugr's launcher for cross-node TP/PP.

        Requires ``config.distributed_mode == "head"``, a non-empty
        ``peer_ips`` list, and passwordless SSH from this node to every peer.
        """
        if self.config.distributed_mode != "head":
            raise EugrBackendError(
                "start_distributed() only runs when distributed_mode='head'. "
                f"Current mode: {self.config.distributed_mode!r}."
            )
        if not self.config.peer_ips:
            raise EugrBackendError(
                "peer_ips is empty; cannot launch distributed cluster without peers."
            )
        if not EUGR_LAUNCHER.exists():
            raise EugrBackendError(
                f"eugr launcher missing at {EUGR_LAUNCHER}. Is this running inside the ainode image?"
            )

        self._write_eugr_env()
        launch_script = self._write_distributed_launch_script()
        self._distribute_engine_image_to_peers()
        self._distribute_model_to_peers()

        # Bug 3 fix: publish per-node shim to shared storage so every peer's
        # vllm_node can mount + exec it as --entrypoint. Returns None if the
        # shim/shared-storage isn't available; falls back cleanly to head-only
        # detection (bugs 1/2/4 still fixed).
        shim_container_path = self._publish_nccl_init_script()

        cmd = [str(EUGR_LAUNCHER), *self._launcher_image_args(),
               "--launch-script", str(launch_script)]
        env = self._launcher_env(shim_container_path=shim_container_path)

        logger.info(
            "Starting distributed vLLM: %s across %d peers via eugr launcher",
            self._parallel_plan().label(),
            len(self.config.peer_ips),
        )
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=str(EUGR_LAUNCHER.parent),
        )
        self._log_thread = threading.Thread(
            target=self._stream_logs,
            args=(self._process, self._distributed_log),
            daemon=True,
        )
        self._log_thread.start()
        return self._process.poll() is None

    def launch_distributed(self, sharding_config=None) -> bool:
        """Launch distributed inference — compatibility shim for the /api/models/load path.

        Accepts an optional sharding_config (from VLLMEngine's interface) but
        delegates to start_distributed() which reads peer_ips and TP size from
        self.config. If the config isn't already in head mode, we flip it.
        """
        if sharding_config is not None:
            # Apply sharding config fields to our config
            if hasattr(sharding_config, "model") and sharding_config.model:
                self.config.model = sharding_config.model
            if hasattr(sharding_config, "peer_ips") and sharding_config.peer_ips:
                self.config.peer_ips = sharding_config.peer_ips
            if hasattr(sharding_config, "strategy"):
                pass  # TP vs PP handled by eugr launcher via env vars

        if self.config.distributed_mode != "head":
            self.config.distributed_mode = "head"
            try:
                self.config.save()
            except Exception:
                pass

        return self.start_distributed()

    def stop(self) -> None:
        """Graceful shutdown. For distributed, also invokes eugr's ``stop``."""
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            self._process = None

        if self.config.distributed_mode == "head" and EUGR_LAUNCHER.exists():
            try:
                subprocess.run(
                    [str(EUGR_LAUNCHER), "stop"],
                    cwd=str(EUGR_LAUNCHER.parent),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception:  # pragma: no cover - best-effort teardown
                logger.exception("eugr launch-cluster.sh stop failed")

        self._ready = False

    def wait_ready(self, timeout: float = 300.0) -> bool:
        """Poll ``/v1/models`` on the API port until 2xx or timeout."""
        url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                # Subprocess died before ever serving.
                return False
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        self._ready = True
                        if self.on_ready:
                            try:
                                self.on_ready()
                            except Exception:  # pragma: no cover
                                logger.exception("on_ready callback failed")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
                pass
            time.sleep(2)
        return False

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def health_check(self) -> dict:
        """Used by ``ainode status``. Mirrors VLLMEngine.health_check."""
        result = {
            "process_alive": self.is_running(),
            "api_responding": False,
            "models_loaded": [],
        }
        try:
            url = f"http://127.0.0.1:{self.config.api_port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                result["api_responding"] = True
                result["models_loaded"] = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return result

    def logs(self, n: int = 100) -> str:
        log = self._distributed_log if self.config.distributed_mode == "head" else self._log_file
        if not log.exists():
            return ""
        try:
            lines = log.read_text().splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # ------------------------------------------------------------------
    # Compatibility properties (VLLMEngine parity)
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.config.api_port}/v1"

    @property
    def log_path(self) -> Path:
        return (
            self._distributed_log
            if self.config.distributed_mode == "head"
            else self._log_file
        )

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    @process.setter
    def process(self, value: Optional[subprocess.Popen]) -> None:
        # Preserve the mutability of the original public ``process`` attr
        # — existing tests set it directly to mock instances.
        self._process = value

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_solo_cmd(self) -> List[str]:
        """Assemble ``vllm serve`` args mirroring VLLMEngine.build_cmd."""
        gpu = detect_gpu()
        cmd: List[str] = [
            "vllm",
            "serve",
            self.config.model,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.config.api_port),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.config.max_model_len:
            cmd.extend(["--max-model-len", str(self.config.max_model_len)])
        if self.config.quantization:
            quant = self.config.quantization
        elif self.config.model and "awq" in self.config.model.lower():
            # Model name contains AWQ — explicitly pin to awq (not awq_marlin).
            # vLLM auto-upgrades AWQ → awq_marlin which requires Marlin CUDA
            # kernels not compiled for GB10 (sm_12.1) in the eugr base image.
            quant = "awq"
        else:
            quant = None
        if quant:
            cmd.extend(["--quantization", quant])
        if self.config.models_dir:
            cmd.extend(["--download-dir", self.config.models_dir])
        if gpu and gpu.unified_memory:
            cmd.extend(["--dtype", "bfloat16"])
        return cmd

    def _build_env(self) -> dict:
        """Base environment for subprocess calls, including NCCL/RoCE tuning."""
        env = os.environ.copy()
        env.setdefault("HF_HOME", self.config.models_dir or "/root/.ainode/models")

        # Propagate HF token if configured — needed for gated repos (Llama etc.)
        if self.config.hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = self.config.hf_token
            env["HF_TOKEN"] = self.config.hf_token

        # Socket interface = the COORDINATION path. On a mesh that is the
        # shared Ethernet, because this node's cluster_interface address
        # reaches only one of its two neighbours. Off the mesh the topology
        # hands back cluster_interface and this is unchanged.
        iface = self._coord_interface()
        if iface:
            env["NCCL_SOCKET_IFNAME"] = iface
            env["GLOO_SOCKET_IFNAME"] = iface
            env["UCX_NET_DEVICES"] = iface

        ib_hca = self._nccl_ib_hca()
        if ib_hca:
            env.setdefault("NCCL_IB_HCA", ib_hca)
        env.setdefault("NCCL_IB_DISABLE", "0")
        env.setdefault("NCCL_P2P_DISABLE", "0")
        env.setdefault("NCCL_NET_GDR_LEVEL", "5")
        env.setdefault("NCCL_IGNORE_CPU_AFFINITY", "1")
        for key, value in self._topology().nccl_env().items():
            env.setdefault(key, value)

        return env

    # ------------------------------------------------------------------
    # Fabric topology
    # ------------------------------------------------------------------

    def _topology(self) -> TopologyInfo:
        """This node's fabric wiring, detected once per backend instance."""
        if self._topology_cache is None:
            self._topology_cache = topology_for_config(self.config)
            logger.info(
                "Fabric topology: %s (coordination on %s, NCCL_IB_HCA=%s)",
                self._topology_cache.fabric.value,
                self._topology_cache.coord_interface or "<unset>",
                ",".join(self._topology_cache.rdma_hcas) or "<autodetect>",
            )
        return self._topology_cache

    def _coord_interface(self) -> str:
        """Interface carrying Ray, the launcher's SSH and the socket env."""
        return self._topology().coord_interface or ""

    def _nccl_ib_hca(self) -> Optional[str]:
        """Comma-joined RoCE devices for ``NCCL_IB_HCA``, or None.

        Two paths:

        * **Mesh** — every active device, straight from topology detection.
          A subnet filter would be actively wrong here: each device sits on a
          *different* subnet by design, so filtering to the coordination
          subnet would leave NCCL nothing (the 10G port is not RoCE) and
          filtering to any one CX7 subnet would leave it a single cable to a
          single neighbour.
        * **Everything else** — the existing subnet-filtered detection,
          unchanged. Bugs 1/2/4 fix: accept both MOFED (mlx5_*) and stock
          rdma-core (rocep*/roceP*) naming, filter by (Up) state, filter by
          cluster subnet so direct-connect HCAs on dual-homed nodes are
          excluded. If detection returns nothing, leave NCCL_IB_HCA unset —
          better for NCCL to auto-detect locally than to pin to a hardcoded
          name that may not exist on this host.
        """
        topo = self._topology()
        if topo.rdma_hcas:
            return ",".join(topo.rdma_hcas)
        return self._detect_ib_hca(subnet_cidr=self._cluster_subnet())

    def _cluster_subnet(self) -> Optional[str]:
        """Return the CIDR of the ``cluster_interface`` netdev (e.g. '192.168.0.0/24').

        Needed to filter out vestigial direct-connect HCAs (10.0.0.x on
        dual-homed Sparks 1 & 2 in our reference cluster) vs the switched
        fabric HCAs (192.168.0.x). Without this, ``_detect_ib_hca`` would
        return direct-connect devices that don't exist on all cluster nodes,
        hanging NCCL ring init.
        """
        iface = self.config.cluster_interface
        if not iface:
            return None
        try:
            out = subprocess.run(
                ["ip", "-o", "-4", "addr", "show", "dev", iface],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0:
                return None
            m = re.search(r"inet\s+(\S+)", out.stdout)
            if not m:
                return None
            return str(ipaddress.ip_network(m.group(1), strict=False))
        except Exception:
            return None

    @staticmethod
    def _detect_ib_hca(subnet_cidr: Optional[str] = None) -> Optional[str]:
        """Return a comma-joined list of Up, on-subnet HCAs, or None.

        Reads ``/sys/class/infiniband/`` directly. Prior implementation
        shelled out to ``ibdev2netdev``, but that tool ships in MOFED /
        infiniband-diags on the host and is NOT in our containers (neither
        the eugr base nor the ainode layer). sysfs is kernel-provided and
        always available.

        Accepts any HCA name the kernel exposes. Udev rules set the name:
        MOFED gives ``mlx5_*``, stock Ubuntu rdma-core gives ``rocep*`` /
        ``roceP*``. Regex limits to those patterns so the detector doesn't
        pick up unrelated entries (virtual HCAs, etc.).

        Port state is parsed as the leading integer and matched against the
        ``ib_port_state`` kernel enum (``include/rdma/ib_verbs.h``):
        0 = NOP, 1 = DOWN, 2 = INIT, 3 = ARMED, 4 = ACTIVE, 5 = ACTIVE_DEFER.
        Accept {4, 5}, reject everything else. Leading-integer parse is
        format-stable where substring matching on the textual label is not
        (kernels emit ``"4: ACTIVE"``, ``"4\\n"``, ``"4 : ACTIVE"``, etc.).

        Subnet filter: if ``subnet_cidr`` is given, exclude HCAs whose
        netdev does not have an IP inside that subnet. On nodes 1 & 2 in
        our reference cluster, this filters out the vestigial direct-
        connect fabric (10.0.0.x) so it never enters the NCCL ring.

        Returns None if no HCA passes the filters — caller should leave
        ``NCCL_IB_HCA`` unset rather than fall back to a guess.

        TODO(v0.4.10): eugr's ``autodiscover.sh`` also depends on
        ``ibdev2netdev``. When this function returns None, the launcher
        writes ``IB_IF=`` (empty) → eugr runs its own autodiscover →
        fails with the same "ibdev2netdev not found" error. Upstream a
        /sys-based autodiscover to eugr, or wrap it.
        """
        # Same sysfs root as hca_discovery / topology, referenced through the
        # module so all three walkers see one tree (and one test fixture can
        # re-point them together) instead of three hardcoded copies.
        ib_base = hca_discovery.SYS_INFINIBAND
        if not ib_base.is_dir():
            return None

        name_re = re.compile(r"^(mlx5_\d+|rocep\w+|roceP\w+)$")

        subnet_obj = None
        if subnet_cidr:
            try:
                subnet_obj = ipaddress.ip_network(subnet_cidr, strict=False)
            except Exception:
                subnet_obj = None

        try:
            hca_dirs = sorted(ib_base.iterdir())
        except Exception:
            return None

        devs: List[str] = []
        for hca_dir in hca_dirs:
            hca = hca_dir.name
            if not name_re.match(hca):
                continue

            # Port state — parse leading integer and match against the
            # ib_port_state enum (include/rdma/ib_verbs.h):
            #   0 = NOP, 1 = DOWN, 2 = INIT, 3 = ARMED,
            #   4 = ACTIVE (normal up),
            #   5 = ACTIVE_DEFER (also functional for traffic).
            # Accept {4, 5}, reject everything else. File content on recent
            # kernels is "4: ACTIVE\n" but can be "4\n" or "4 : ACTIVE" on
            # others — leading-integer parse is format-stable where a
            # substring match on the textual label is not.
            try:
                state_text = (hca_dir / "ports" / "1" / "state").read_text().strip()
            except Exception:
                continue
            state_match = re.match(r"(\d+)", state_text)
            if not state_match or int(state_match.group(1)) not in (4, 5):
                continue

            # Netdev via /sys/class/infiniband/<hca>/device/net/<netdev>.
            # iterdir() returns empty if no netdev is associated (e.g.
            # a pure-IB port with no Ethernet overlay); skip those.
            try:
                netdev_dir = hca_dir / "device" / "net"
                netdevs = list(netdev_dir.iterdir())
            except Exception:
                continue
            if not netdevs:
                continue
            netdev = netdevs[0].name

            if subnet_obj is not None:
                try:
                    ip_out = subprocess.run(
                        ["ip", "-o", "-4", "addr", "show", "dev", netdev],
                        capture_output=True, text=True, timeout=3,
                    )
                    addr_m = re.search(r"inet\s+(\S+)", ip_out.stdout)
                    if not addr_m:
                        continue
                    netdev_ip = ipaddress.ip_interface(addr_m.group(1)).ip
                    if netdev_ip not in subnet_obj:
                        continue
                except Exception:
                    continue  # can't verify subnet → exclude, don't guess

            devs.append(hca)

        return ",".join(devs) or None

    # ------------------------------------------------------------------
    # Distributed (eugr) wiring
    # ------------------------------------------------------------------

    def _launcher_image_args(self) -> List[str]:
        """``-t <image>`` when the instance pins an engine image, else nothing.

        The launcher defaults to IMAGE_NAME="vllm-node", and this backend used
        to pass nothing — so a catalog recipe's ``engine_image`` was silently
        ignored and its flags ran against whatever vLLM the local base image
        happens to contain. A recipe proven on vllm/vllm-openai:v0.27.1 then
        fails with exit code 2, argparse's "unrecognized arguments", which says
        nothing about the engine being the wrong one.

        The image must be present on every participating node; the launcher
        checks that itself and reports a mismatch clearly.
        """
        image = (getattr(self.config, "engine_image", "") or "").strip()
        if not image:
            return []
        if not _is_safe_path_arg(image):
            raise EugrBackendError(
                f"Refusing to pass engine_image {image!r} to the launcher: an "
                f"image reference is [A-Za-z0-9_./@+:-], and this value is "
                f"expanded unquoted into a docker command line."
            )
        return ["-t", image]

    def _launcher_env(self, shim_container_path: Optional[str] = None) -> dict:
        """Environment for a ``launch-cluster.sh`` invocation.

        Shared by the solo and distributed paths: both spawn the same engine
        container and both need the model directory mounted at the same place.
        ``VLLM_SPARK_EXTRA_DOCKER_ARGS`` is appended to the launcher's
        ``DOCKER_ARGS`` and expanded unquoted, hence the path check.
        """
        models_dir = self.config.models_dir or "/root/.ainode/models"
        # The SOURCE of a -v goes to the host daemon, which reads it literally.
        # Passing our own container view mounted whatever sat at
        # /root/.ainode/models on the host — root's home, not the installing
        # user's, and usually empty. See core.config.host_path.
        models_dir_host = host_path(models_dir)
        if not _is_safe_path_arg(models_dir_host):
            raise EugrBackendError(
                f"Refusing to pass models_dir {models_dir_host!r} to the launcher: it "
                f"is expanded unquoted into the docker run arguments, so "
                f"whitespace or a shell metacharacter there injects docker "
                f"flags rather than naming a directory."
            )
        extra_docker_args = ["-v", f"{models_dir_host}:{ENGINE_MODELS_DIR}"]
        if shim_container_path is not None:
            # Mount the shared dir read-only and replace the vllm_node
            # container's default entrypoint with the shim. The shim detects
            # local HCAs, exports NCCL_IB_HCA, and execs the original CMD
            # (typically ``sleep infinity`` from eugr's launcher).
            extra_docker_args.extend([
                "-v", "/mnt/shared-models:/mnt/shared-models:ro",
                "--entrypoint", shim_container_path,
            ])
        env = self._build_env()
        env["VLLM_SPARK_EXTRA_DOCKER_ARGS"] = " ".join(extra_docker_args)
        return env

    def _transfer_ip(self, peer_ip: str) -> str:
        """Address to push bulk data to ``peer_ip`` over.

        ``peer_ip`` is the peer's coordination address. Where the launch path
        found a direct RoCE cable it recorded the far side in
        ``peer_transfer_ips``; absent, the coordination address is used, which
        is what every node did before.
        """
        return (getattr(self.config, "peer_transfer_ips", None) or {}).get(
            peer_ip, peer_ip
        )

    def _distribute_engine_image_to_peers(self) -> None:
        """Place the engine image on every peer before the launcher looks.

        launch-cluster.sh aborts when a node lacks it, or when the ids differ —
        correct, and no help: the operator is told to fix three machines by
        hand. A catalog recipe pinning an engine_image makes that routine
        rather than exceptional.

        Unlike the weights, this is NOT best effort. The launcher will refuse
        anyway, so failing here with "could not place <image> on <node>" is
        strictly more useful than failing there with "image missing".
        """
        image = (getattr(self.config, "engine_image", "") or "").strip()
        if not image:
            return  # the launcher default is built locally on every node
        # The head first: the launcher inspects it here and aborts before it
        # ever looks at a worker.
        logger.info("Engine image %s locally: %s", image, ensure_local_image(image))
        for peer_ip in self.config.peer_ips:
            action = ensure_peer_has_image(
                ssh_user=self.config.ssh_user,
                coord_ip=peer_ip,
                transfer_ip=self._transfer_ip(peer_ip),
                image=image,
            )
            logger.info("Engine image %s on %s: %s", image, peer_ip, action)

    def _distribute_model_to_peers(self) -> None:
        """Make sure every peer can read the model from its own local disk.

        The launcher mounts each node's own ``models_dir`` into that node's
        engine container, so a rank whose host lacks the weights would download
        them from Hugging Face independently — N copies pulled over the WAN
        instead of one copy moved over the fabric, and N chances to be rate
        limited mid-launch.

        Best-effort: a peer that cannot be reached is left alone and the launch
        proceeds, because the head cannot know whether that peer already has
        the weights through some other route. A transfer that starts and fails
        does abort — a half-copied checkpoint is worse than none.
        """
        model = (self.config.model or "").strip()
        if not model:
            return
        models_dir = self.config.models_dir or "/root/.ainode/models"
        # HF_HOME is set to models_dir (see _build_env), so the cache lands in
        # models_dir/hub/models--org--name — the layout registry.py scans.
        hub = str(Path(models_dir) / "hub")
        dir_name = hf_cache_dir_name(model)

        for peer_ip in self.config.peer_ips:
            transfer_ip = self._transfer_ip(peer_ip)
            try:
                placed = ensure_peer_has_dir(
                    ssh_user=self.config.ssh_user,
                    transfer_ip=transfer_ip,
                    source_parent=hub,
                    dir_name=dir_name,
                    target_parent=hub,
                    label="direct RoCE link" if transfer_ip != peer_ip else "",
                )
            except DistributionError:
                raise
            except Exception:
                logger.exception(
                    "Could not check or copy %s to %s; continuing — the peer may "
                    "already have it by another route", dir_name, transfer_ip,
                )
                continue
            if not placed:
                logger.info(
                    "%s is not in %s on this node; leaving each rank to fetch it",
                    dir_name, hub,
                )
                return

    def _parallel_plan(self) -> ParallelPlan:
        """How this instance splits across head + peers, one GPU per node.

        Reads the axis sizes the launch path resolved into the config
        snapshot; an unresolved config falls back to tensor parallelism across
        every node, which is what this returned before the plan existed.
        """
        node_count = 1 + len(self.config.peer_ips)
        plan = ParallelPlan.from_dict({
            "tensor_parallel_size": getattr(self.config, "tensor_parallel_size", 0) or 0,
            "pipeline_parallel_size": getattr(self.config, "pipeline_parallel_size", 0) or 0,
            "data_parallel_size": getattr(self.config, "data_parallel_size", 0) or 0,
            "strategy": getattr(self.config, "parallel_strategy", "") or "",
        })
        if plan.world_size == node_count and plan.is_distributed:
            return plan
        return ParallelPlan(tensor_parallel_size=node_count, strategy=Strategy.TENSOR)

    def _tp_size(self) -> int:
        """Back-compat shim — the TP factor of :meth:`_parallel_plan`."""
        return self._parallel_plan().tensor_parallel_size

    def _publish_nccl_init_script(self) -> Optional[str]:
        """Publish the per-node shim to shared storage so every peer sees it.

        Bug 3 fix: per-node NCCL detection can't be broadcast from a single
        ``.env`` (eugr's launcher propagates one cluster-wide value), so we
        mount the shim into each peer's vllm_node via ``--entrypoint``. The
        shim itself is baked into the ainode image by Dockerfile COPY; here
        we stage it on the shared NFS path that every peer's ``docker run
        -v`` can bind-mount from.

        Returns the in-container path to pass to ``--entrypoint``, or None
        if publishing failed (shim missing from image, or shared storage
        unavailable). On None, ``start_distributed`` falls back to head-only
        detection — bugs 1, 2, 4 still fixed; bug 3 degrades to partial.

        TODO(v0.4.10): bake the shim into ainode-base so every vllm_node
        has it at ``/usr/local/bin/nccl-env-init.sh`` without NFS. Then this
        publish step and the ``/mnt/shared-models`` dependency go away.
        """
        if not NCCL_INIT_IMAGE_PATH.exists():
            logger.warning(
                "NCCL init shim %s missing from image; per-node NCCL env "
                "disabled (bugs 1/2/4 still fixed).",
                NCCL_INIT_IMAGE_PATH,
            )
            return None
        try:
            NCCL_INIT_SHARED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(NCCL_INIT_IMAGE_PATH, NCCL_INIT_SHARED_PATH)
            NCCL_INIT_SHARED_PATH.chmod(0o755)
            logger.info("Published NCCL shim to %s", NCCL_INIT_SHARED_PATH)
            return NCCL_INIT_CONTAINER_PATH
        except Exception as exc:
            logger.warning(
                "Could not publish NCCL shim to %s: %s. Falling back to "
                "head-only detection (bugs 1/2/4 still fixed).",
                NCCL_INIT_SHARED_PATH, exc,
            )
            return None

    def _write_eugr_env(self) -> None:
        """Populate ``/opt/spark-vllm-docker/.env`` for the launcher.

        ``ETH_IF`` is the coordination interface, matching what upstream's
        ``autodiscover.sh`` puts there: the launcher hands it to every node as
        NCCL_SOCKET_IFNAME/GLOO/UCX and derives the Ray addresses from it.

        Writing both ``ETH_IF`` and ``IB_IF`` makes the launcher's own
        ``detect_interfaces()`` return early, so its mesh branch never runs and
        cannot set the mesh NCCL vars for us — we emit them ourselves as
        ``CONTAINER_*`` below. Leaving either one *empty* is worse than wrong:
        it drops the launcher into autodiscovery, which needs ``ibdev2netdev``,
        which the AINode image does not ship — a hard failure with a confusing
        message. So both are asserted non-empty before the file is written.
        """
        topo = self._topology()
        iface = self._coord_interface()
        head_ip = _local_ip_for_interface(iface)
        cluster_nodes = ",".join([head_ip] + self.config.peer_ips)
        subnet = self._cluster_subnet()

        # Bug 4 fix: IB_IF carries the HCA device list (eugr maps it to
        # NCCL_IB_HCA at launch-cluster.sh:810). Prior code wrote the netdev
        # name here — ``IB_IF=enP2p1s0f1np1`` — which produced a garbage
        # ``NCCL_IB_HCA=enP2p1s0f1np1`` (a netdev is not an HCA device).
        ib_hca = self._nccl_ib_hca() or ""

        # Every value below lands in a file whose CONTAINER_* entries
        # launch-cluster.sh re-quotes by interpolating them into a Python
        # one-liner — a single quote in a device name becomes code that runs on
        # the head. These fields are settable over PATCH /api/config, which is
        # unauthenticated unless the operator enabled auth, so they are checked
        # here at the sink as well as at that entry point: config.json can also
        # be edited by hand. See topology.is_safe_device_name.
        unsafe = [n for n in ([iface] + ib_hca.split(",") if ib_hca else [iface])
                  if n and not is_safe_device_name(n)]
        if unsafe:
            raise EugrBackendError(
                f"Refusing to write device name(s) {unsafe!r} into the launcher "
                f".env: a network-interface or RDMA device name is at most 15 "
                f"characters of [A-Za-z0-9_.-]. Check cluster_interface / "
                f"coord_interface / rdma_hcas in config.json."
            )

        if not iface or not ib_hca:
            raise EugrBackendError(
                "Refusing to write an incomplete launcher .env "
                f"(ETH_IF={iface!r}, IB_IF={ib_hca!r}). An empty value makes "
                "launch-cluster.sh fall back to its own autodiscovery, which "
                "requires ibdev2netdev — not present in the AINode image. "
                f"Detected fabric: {topo.fabric.value}. Set cluster_interface / "
                "coord_interface / rdma_hcas explicitly for this node."
            )

        lines = [
            f"CLUSTER_NODES={cluster_nodes}",
            f"ETH_IF={iface}",
            f"IB_IF={ib_hca}",
            "MASTER_PORT=29501",
            f"SSH_USER={self.config.ssh_user}",
            # CONTAINER_* vars are injected into every per-node container.
            "CONTAINER_NCCL_DEBUG=INFO",
            "CONTAINER_NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH",
            f"CONTAINER_NCCL_SOCKET_IFNAME={iface}",
            # Bug 3 fix: CONTAINER_NCCL_IB_HCA deliberately omitted. Each
            # vllm_node's per-node shim (mounted via --entrypoint) exports
            # its own node-local NCCL_IB_HCA at startup. A cluster-wide
            # value would be wrong for heterogeneous HCA naming.
            "CONTAINER_NCCL_IB_DISABLE=0",
            "CONTAINER_NCCL_IGNORE_CPU_AFFINITY=1",
            "CONTAINER_NCCL_NET_GDR_LEVEL=5",
            f"CONTAINER_UCX_NET_DEVICES={iface}",
            # Consumed by the per-node shim to filter HCAs by subnet. Empty on
            # a mesh: there the shim must NOT filter, because each device is on
            # its own subnet and any filter would strip the ring down to one
            # cable. _nccl_ib_hca() explains the same reasoning.
            f"CONTAINER_AINODE_CLUSTER_SUBNET={'' if topo.is_mesh else (subnet or '')}",
        ]

        # Mesh NCCL settings. Upstream's autodiscover.sh exports these from its
        # own mesh branch; because we hand the launcher a complete ETH_IF/IB_IF
        # that branch never runs, so we emit the same three values directly.
        for key, value in topo.nccl_env().items():
            lines.append(f"CONTAINER_{key}={value}")

        EUGR_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        EUGR_ENV_FILE.write_text("\n".join(lines) + "\n")

    def _write_distributed_launch_script(self) -> Path:
        """Back-compat alias — the distributed script for the current plan."""
        return self._write_launch_script(self._parallel_plan())

    def _write_launch_script(self, plan: ParallelPlan, solo: bool = False) -> Path:
        """Emit a ``vllm serve`` script for eugr to execute inside the container.

        One writer for both paths. ``solo`` drops the Ray executor and the
        parallelism flags: with a single rank there is nothing to place and
        nothing to split, and the launcher would otherwise read the flags and
        size a node list from them.
        """
        gpu = detect_gpu()
        dtype_line = ""
        if gpu and gpu.unified_memory:
            dtype_line = "    --dtype bfloat16 \\\n"

        extra = ""
        if self.config.max_model_len:
            extra += f"    --max-model-len {self.config.max_model_len} \\\n"
        if self.config.quantization:
            extra += f"    --quantization {self.config.quantization} \\\n"

        # Note: --enforce-eager intentionally omitted. CUDA graphs add ~60s
        # to initial warmup but give 2-3x steady-state throughput, which
        # matters far more for the user than first-token latency. If the
        # graphs capture fails on a particular model, we can re-add eager
        # per-invocation via config.

        # Bug 3 belt-and-suspenders: ``docker exec`` on the head does not
        # inherit PID-1 runtime env from ``--entrypoint``. Source the shim
        # in --export-only mode so NCCL_IB_HCA lands in THIS shell before
        # vllm starts, and thereby in every child (driver + Ray workers).
        # ``|| true`` and the presence check keep this safe if the shim
        # wasn't staged (operator without shared storage).
        env_init_hook = (
            f"if [ -x {NCCL_INIT_CONTAINER_PATH} ]; then\n"
            f'    eval "$({NCCL_INIT_CONTAINER_PATH} --export-only 2>/dev/null || true)"\n'
            f"fi\n"
        )

        # Parallelism. The launcher parses these same flags out of the script
        # to decide how many nodes to use (parse_parallelism_from_text in
        # launch-cluster.sh), so they have to appear here verbatim and not only
        # in the env. Previously this hardcoded "--pipeline-parallel-size 1",
        # which is why the UI's Pipeline pill could never do anything.
        parallel_lines = ""
        executor_line = ""
        if not solo:
            parallel_lines = f"    --tensor-parallel-size {plan.tensor_parallel_size} \\\n"
            parallel_lines += f"    --pipeline-parallel-size {plan.pipeline_parallel_size} \\\n"
            if plan.data_parallel_size > 1:
                parallel_lines += f"    --data-parallel-size {plan.data_parallel_size} \\\n"
            executor_line = "    --distributed-executor-backend ray \\\n"

        # Recipe passthrough, same as the solo/nvidia paths: a model whose
        # published recipe needs flags AINode does not model must be able to
        # carry them here too, or the UI's Advanced fields are inert on this
        # backend.
        for arg in (getattr(self.config, "extra_vllm_args", None) or []):
            extra += f"    {arg} \\\n"

        script = f"""#!/bin/bash
# Auto-generated by ainode EugrBackend. Do not edit.
{env_init_hook}
vllm serve {self.config.model} \\
    --host 0.0.0.0 --port {self.config.api_port} \\
{executor_line}{parallel_lines}    --gpu-memory-utilization {self.config.gpu_memory_utilization} \\
{dtype_line}{extra}    --download-dir {ENGINE_MODELS_DIR}
"""
        name = "ainode-solo.sh" if solo else "ainode-distributed.sh"
        target = EUGR_LAUNCHER.parent / "examples" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script)
        target.chmod(0o755)
        return target

    # ------------------------------------------------------------------
    # Log streaming
    # ------------------------------------------------------------------

    @property
    def load_phase(self) -> str:
        """Coarse load phase for the UI's launching card.

        This backend reported none, so every launch showed a flat 8% — the
        UI's fallback for "unknown" — for the whole of a load that can take
        minutes, which is indistinguishable from a hang.
        """
        return self._phase.current(ready_latch=self._ready)

    @property
    def load_error(self) -> str:
        """Why the last launch died, or "". Quotes the engine's own last lines."""
        return self._phase.failure_reason()

    def _stream_logs(self, process: subprocess.Popen, target: Path) -> None:
        """Tee subprocess stdout to a log file, tracking readiness and phase."""
        if not process.stdout:
            return
        self._phase.reset()
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

        # The stream ended. If the engine never reported itself ready, the
        # launch died — say so. Without this the card sits at "starting"
        # indefinitely and a dead launcher is indistinguishable from a model
        # that simply takes minutes to load.
        if not self._ready:
            try:
                rc = process.wait(timeout=10)
            except Exception:
                rc = None
            self._phase.fail(
                f"the launcher exited (code {rc})" if rc is not None
                else "the launcher stopped producing output"
            )
            logger.error("Launch failed: %s", self._phase.failure_reason())


# Back-compat alias — old code imports ``DockerEngine`` from the
# ``docker_engine`` shim, which in turn re-exports this name.
DockerEngine = EugrBackend


def _local_ip_for_interface(iface: Optional[str]) -> str:
    """Return the IPv4 address on ``iface`` — or fall back to hostname."""
    if not iface:
        return "127.0.0.1"
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", iface],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                parts = line.split()
                if "inet" in parts:
                    idx = parts.index("inet")
                    return parts[idx + 1].split("/")[0]
    except Exception:
        pass
    import socket as _socket
    try:
        return _socket.gethostbyname(_socket.gethostname())
    except Exception:
        return "127.0.0.1"
