"""API routes for model sharding — plan, launch, and monitor distributed inference."""

from __future__ import annotations

import json
import logging
from typing import Optional

from aiohttp import web

from ainode.discovery.cluster import ClusterState
from ainode.engine.parallelism import ParallelPlanError, Strategy, plan_for
from ainode.engine.sharding import ShardingPlanner, ShardingStrategy, ShardingConfig
from ainode.engine.ray_setup import get_ray_status

logger = logging.getLogger(__name__)

# Module-level state for active sharding session
_active_sharding: Optional[ShardingConfig] = None

# The memory planner (sharding.py) models tensor and pipeline splits only — a
# data-parallel replica set has different memory behaviour (a full copy per
# node) that it does not describe, so the preview declines it rather than
# returning a number that would be wrong.
_SHARDING_STRATEGY_BY_AXIS = {
    Strategy.AUTO: ShardingStrategy.AUTO,
    Strategy.TENSOR: ShardingStrategy.TENSOR_PARALLEL,
    Strategy.PIPELINE: ShardingStrategy.PIPELINE_PARALLEL,
}


def register_sharding_routes(app: web.Application) -> None:
    """Register sharding API endpoints on the aiohttp app."""
    app.router.add_get("/api/sharding/plan", handle_sharding_plan)
    app.router.add_post("/api/sharding/launch", handle_sharding_launch)
    app.router.add_get("/api/sharding/status", handle_sharding_status)


async def handle_sharding_plan(request: web.Request) -> web.Response:
    """GET /api/sharding/plan?model=X — preview sharding plan for a model.

    Query params:
        model (required): HuggingFace model ID
        strategy (optional): tensor_parallel, pipeline_parallel, auto (default: auto)
    """
    model = request.query.get("model")
    if not model:
        return web.json_response({"error": "model parameter required"}, status=400)

    strategy_str = request.query.get("strategy", "auto")
    # Normalise first, so the preview accepts the same spellings the launch
    # route does ("tensor" from the UI, "tensor_parallel" from the docs).
    try:
        strategy = _SHARDING_STRATEGY_BY_AXIS[Strategy.parse(strategy_str)]
    except ParallelPlanError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except KeyError:
        return web.json_response(
            {"error": f"The planner cannot preview {strategy_str!r} yet; "
                      f"use auto, tensor or pipeline."},
            status=400,
        )

    cluster: ClusterState = request.app["cluster_state"]
    planner = ShardingPlanner()

    try:
        config = planner.plan_sharding(model, cluster, strategy)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=422)

    return web.json_response({
        "plan": config.to_dict(),
        "can_fit": planner.can_fit_model(model, cluster),
        "cluster_nodes": len(cluster.get_nodes(include_offline=False)),
    })


async def handle_sharding_launch(request: web.Request) -> web.Response:
    """POST /api/sharding/launch — launch a model distributed across the cluster.

    JSON body:
        model (required): HuggingFace model ID
        strategy (optional): auto | tensor_parallel | pipeline_parallel
        min_nodes (optional, default 1): nodes to span

    When min_nodes > 1, this endpoint flips the local engine into head mode:
    it discovers member nodes from the cluster state, takes their peer IPs
    from UDP recvfrom, writes them into config, stops the current (solo)
    engine, and starts the distributed engine via the configured backend
    (eugr's launch-cluster.sh, or NvidiaBackend's run_cluster path).
    When min_nodes == 1, it falls through to the existing single-node load
    path (/api/models/load).
    """
    from ainode.core.config import NodeConfig
    from ainode.engine.backends import get_backend

    global _active_sharding

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = body.get("model")
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    try:
        min_nodes = int(body.get("min_nodes", 1) or 1)
    except (TypeError, ValueError):
        min_nodes = 1

    # Explicit node selection (preferred): the exact nodes to span, head = this
    # node + the rest as peers. `tp_size` is the legacy count form. Either sets
    # the effective node count so the min_nodes<=1 solo path still triggers.
    node_ids = body.get("node_ids") or None
    if node_ids:
        min_nodes = len(node_ids)
    elif body.get("tp_size"):
        try:
            min_nodes = int(body.get("tp_size"))
        except (TypeError, ValueError):
            pass

    # Parallelism axis. Previously read and ignored ("any min_nodes > 1
    # triggers TP"), which is why the UI's Pipeline pill did nothing and why a
    # 3-node launch would have built the unsupported TP=3. The plan itself is
    # resolved further down, once the participating nodes are known.
    strategy_str = body.get("strategy") or "auto"
    try:
        strategy = Strategy.parse(strategy_str)
    except ParallelPlanError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    cluster: ClusterState = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]

    if min_nodes <= 1:
        # Delegate to the single-node load path so behaviour stays
        # consistent with what the UI called before.
        request._rewritten_body = {"model": model}  # for observability
        from ainode.models.api_routes import handle_model_load  # lazy import
        # Re-inject body so handle_model_load can read it
        class _ReqShim:
            def __init__(self, orig, body):
                self._o = orig
                self._b = body
            def __getattr__(self, k): return getattr(self._o, k)
            async def json(self): return self._b
        shim = _ReqShim(request, {"model": model})
        return await handle_model_load(shim)

    # Distributed path. Resolve the participating peers to their FABRIC IPs
    # (BUG D: never the mgmt-LAN UDP peer_ip, which lands a Ray worker on a
    # non-GPU address). Two selection modes:
    #   node_ids  — explicit set chosen in the UI; head = this node, peers = rest
    #   min_nodes — legacy count: take the first (N-1) discovered members
    members = [
        n for n in cluster.members()
        if getattr(n, "distributed_mode", "solo") == "member"
        and (n.status.value if hasattr(n.status, "value") else str(n.status)) in ("online", "member-ready", "serving")
    ]
    def fabric_of(n):
        return (getattr(n, "fabric_ip", "") or "").strip()
    members_dump = [
        {"node_id": n.node_id, "node_name": n.node_name, "fabric_ip": fabric_of(n),
         "status": n.status.value if hasattr(n.status, "value") else str(n.status)}
        for n in members
    ]

    if node_ids:
        # Head is always this node; peers are the other selected nodes.
        wanted = [nid for nid in node_ids if nid != config.node_id]
        by_id = {n.node_id: n for n in members}
        missing = [nid for nid in wanted if nid not in by_id]
        if missing:
            return web.json_response({
                "error": f"Selected node(s) not available as members: {missing}",
                "discovered_members": members_dump,
            }, status=422)
        chosen = [by_id[nid] for nid in wanted]
    else:
        want_peers = max(0, min_nodes - 1)
        if len(members) < want_peers:
            return web.json_response({
                "error": (
                    f"Requested {want_peers + 1} node(s) but only {len(members) + 1} "
                    f"available (1 head + {len(members)} member(s))."
                ),
                "discovered_members": members_dump,
            }, status=422)
        chosen = members[:want_peers]

    # Refuse to launch on a peer with no known fabric IP — that's exactly the
    # BUG-D failure mode (would fall back to a mgmt address).
    no_fabric = [n.node_id for n in chosen if not fabric_of(n)]
    if no_fabric:
        return web.json_response({
            "error": f"No fabric IP known for node(s) {no_fabric}; cannot launch over the fabric.",
            "hint": "Those nodes must broadcast a fabric_ip (cluster_interface configured).",
            "discovered_members": members_dump,
        }, status=422)

    chosen_peers = [fabric_of(n) for n in chosen]

    # Resolve the split now that the node set is final. This is where a 3-node
    # tensor-parallel request is refused — with the alternatives named, before
    # anything is launched — instead of failing inside vLLM's engine startup.
    try:
        plan = plan_for(strategy, 1 + len(chosen_peers))
    except ParallelPlanError as exc:
        return web.json_response({
            "error": str(exc),
            "strategy": strategy.value,
            "node_count": 1 + len(chosen_peers),
        }, status=422)

    # Second address list: where head and peer share a direct RoCE cable, bulk
    # transfer (model weights) goes over it instead of the coordination
    # Ethernet. Coordination itself stays on chosen_peers — on a mesh no single
    # RoCE address reaches every node. Peers that announce no RoCE address, or
    # that we have no cable to, are simply absent and transfer as before.
    peer_transfer_ips: dict = {}
    try:
        from ainode.cluster.topology import detect_cx7_links, transfer_address

        local_links = detect_cx7_links()
        for node, coord_ip in zip(chosen, chosen_peers):
            direct = transfer_address(
                list(getattr(node, "ib_ips", []) or []), coord_ip, local_links
            )
            if direct and direct != coord_ip:
                peer_transfer_ips[coord_ip] = direct
    except Exception:
        logger.exception("Could not resolve direct transfer addresses; "
                         "falling back to the coordination path")

    # P2-2: APPEND a new instance — do NOT tear down existing ones. Each instance
    # gets its own port (8000, 8001, …), container-name token, and config SNAPSHOT
    # (never the shared app["config"], which would cross-wire instances).
    from dataclasses import replace

    from ainode.discovery.instance import InstanceRecord
    from ainode.engine.instance_manager import InstanceManager

    manager = request.app.get("instances")
    if manager is None:
        manager = InstanceManager(base_port=config.api_port)
        request.app["instances"] = manager

    # Re-launching a model that's already up replaces THAT instance (stop it first).
    existing = manager.by_model(model)
    if existing is not None:
        try:
            existing.backend.stop()
        except Exception:
            logger.exception("stop() failed replacing instance for %s", model)
        manager.remove(existing.record.instance_id)

    is_primary = manager.is_empty()
    port = manager.allocate_port()
    name_token = "" if port == config.api_port else str(port)  # primary keeps legacy names
    instance_id = f"{config.node_id or 'head'}:{model}"

    # Honour a per-launch GPU memory fraction if the UI supplied one (the same
    # #launch-gmu box the solo path uses). Without this the distributed backend
    # always renders the shared NodeConfig default (0.5), silently dropping the
    # value the user typed for a TP>1 launch. Ignore junk / out-of-range input.
    overrides: dict = {}
    gmu_raw = body.get("gpu_memory_utilization")
    if gmu_raw is not None:
        try:
            gmu = float(gmu_raw)
        except (TypeError, ValueError):
            gmu = None
        if gmu is not None and 0.0 < gmu <= 1.0:
            overrides["gpu_memory_utilization"] = gmu

    inst_config = replace(config, model=model, distributed_mode="head",
                          peer_ips=chosen_peers, peer_transfer_ips=peer_transfer_ips,
                          parallel_strategy=plan.strategy.value,
                          tensor_parallel_size=plan.tensor_parallel_size,
                          pipeline_parallel_size=plan.pipeline_parallel_size,
                          data_parallel_size=plan.data_parallel_size,
                          api_port=port, **overrides)
    backend = get_backend(inst_config, instance_id=name_token)
    try:
        started = backend.start_distributed()
    except Exception as exc:
        logger.exception("start_distributed raised")
        return web.json_response({"error": f"Distributed launch failed: {exc}"}, status=500)
    if not started:
        return web.json_response({"error": "Distributed launch returned False"}, status=500)

    manager.add(InstanceRecord(
        instance_id=instance_id, model=model, head_node_id=config.node_id or "head",
        peer_ips=chosen_peers, api_port=port,
        tensor_parallel_size=plan.tensor_parallel_size,
        pipeline_parallel_size=plan.pipeline_parallel_size,
        data_parallel_size=plan.data_parallel_size,
        status="starting"), backend)

    if is_primary:
        # Back-compat: the proxy/status path reads app["config"] + app["engine"].
        config.model = model
        config.distributed_mode = "head"
        config.peer_ips = chosen_peers
        config.peer_transfer_ips = peer_transfer_ips
        config.parallel_strategy = plan.strategy.value
        config.tensor_parallel_size = plan.tensor_parallel_size
        config.pipeline_parallel_size = plan.pipeline_parallel_size
        config.data_parallel_size = plan.data_parallel_size
        try:
            config.save()
        except Exception:
            logger.exception("Failed to persist config.json before distributed launch")
        request.app["engine"] = backend

    return web.json_response({
        "status": "launching",
        "instance_id": instance_id,
        "model": model,
        "distributed_mode": "head",
        "peer_ips": chosen_peers,
        "api_port": port,
        # Flat sizes stay at the top level for callers that read them today.
        "tensor_parallel_size": plan.tensor_parallel_size,
        "pipeline_parallel_size": plan.pipeline_parallel_size,
        "data_parallel_size": plan.data_parallel_size,
        # The resolved axis, not what was asked for: "auto" on three nodes
        # comes back as "pipeline".
        "strategy": plan.strategy.value,
        "parallel_plan": plan.to_dict(),
    })


async def handle_sharding_status(request: web.Request) -> web.Response:
    """GET /api/sharding/status — current sharding state and Ray cluster health."""
    engine = request.app.get("engine")

    engine_running = False
    engine_ready = False
    if engine is not None:
        try:
            engine_running = engine.is_running()
        except Exception:
            engine_running = False
        engine_ready = getattr(engine, "ready", False)

    # When a distributed head engine is up, derive ray health from the engine
    # itself — the orchestrator container has no ray binary to probe.
    engine_config = getattr(engine, "config", None) if engine is not None else None
    distributed_mode = getattr(engine_config, "distributed_mode", "solo")
    peer_ips = getattr(engine_config, "peer_ips", None) or []

    if distributed_mode == "head" and peer_ips and engine_running:
        probe = get_ray_status()
        ray = {
            "running": True,
            "is_head": True,
            "num_nodes": 1 + len(peer_ips),
            "total_cpus": getattr(probe, "total_cpus", 0) or 0,
            "total_gpus": getattr(probe, "total_gpus", 0) or 0,
            "error": None,
            "source": "engine",
        }
    else:
        ray = get_ray_status().to_dict()
        ray["source"] = "ray_probe"

    result = {
        "active_sharding": _active_sharding.to_dict() if _active_sharding else None,
        "engine_running": engine_running,
        "engine_ready": engine_ready,
        "ray": ray,
    }

    return web.json_response(result)
