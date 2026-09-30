"""The startup replay never starts a distributed model solo.

After an update restarted AINode on spark-1432, Qwen3.8-Flash-Next — marked
``distributed: True`` in the manifest, and not running at that moment — was
replayed through append_solo_instance: world_size=1, 132 GB on one node,
OOM-killed after 75 s, then retried once and killed again.
"""

from __future__ import annotations

import asyncio

from ainode.core.config import NodeConfig
from ainode.models import api_routes


def test_only_the_solo_entry_is_replayed(monkeypatch):
    started = []
    monkeypatch.setattr(api_routes, "load_instance_manifest", lambda: [
        {"model": "nvidia/Qwen3.8-Flash-Next-NVFP4", "api_port": 8000, "distributed": True,
         "peer_ips": ["192.168.1.4"], "tensor_parallel_size": 2},
        {"model": "org/small", "api_port": 8001, "gpu_memory_utilization": 0.2},
    ])

    async def _no_sleep(_):
        return None

    async def _ready(port, timeout=0):
        return True

    monkeypatch.setattr(api_routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(api_routes, "_wait_port_ready", _ready)
    monkeypatch.setattr(api_routes, "append_solo_instance",
                        lambda app, model, gmu, overrides=None, persist=False:
                        started.append(model) or {"ok": False})
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("R", (), {"stdout": ""})())

    class _Manager:
        def instances(self):
            return []

        def by_model(self, model):
            return None

    app = {"config": NodeConfig(model=""), "instances": _Manager()}
    asyncio.run(api_routes.replay_instances_on_startup(app))
    assert started == ["org/small"]
