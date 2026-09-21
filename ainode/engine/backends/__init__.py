"""Pluggable engine backends. Each backend drives a specific inference-engine
stack (vLLM via eugr's launch-cluster.sh, vLLM via NVIDIA's run_cluster.sh,
or future options). All implement the EngineBackend ABC."""

from ainode.engine.backends.base import EngineBackend
from ainode.engine.backends.diffusers import DiffusersBackend
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.backends.nvidia import NvidiaBackend


def get_backend(config, on_ready=None, instance_id="") -> EngineBackend:
    """Return the configured engine backend instance.

    Dispatches on ``config.engine_backend`` (new field). Defaults to ``"eugr"``
    for backward compatibility with existing installs. ``instance_id`` (P2-2)
    disambiguates per-instance container names for the nvidia backend.
    """
    backend = (getattr(config, "engine_backend", None) or "eugr").lower()
    if backend == "eugr":
        # instance_id was dropped here, so every eugr instance shared one
        # container name and one generated script.
        return EugrBackend(config, on_ready=on_ready, instance_id=instance_id)
    if backend == "nvidia":
        return NvidiaBackend(config, on_ready=on_ready, instance_id=instance_id)
    if backend == "diffusers":
        # Image generation. Not a vLLM at all: one process, one node, no Ray —
        # but the same interface, so everything AINode does around instances
        # applies to it unchanged.
        return DiffusersBackend(config, on_ready=on_ready,
                                instance_id=instance_id)
    raise ValueError(
        f"Unknown engine_backend={backend!r}. "
        f"Valid options: 'eugr', 'nvidia', 'diffusers'."
    )


__all__ = ["EngineBackend", "DiffusersBackend", "EugrBackend", "NvidiaBackend",
           "get_backend"]
