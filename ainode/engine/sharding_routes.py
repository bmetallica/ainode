"""API routes for model sharding — plan, launch, and monitor distributed inference."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from aiohttp import web

from ainode.api.params import as_object, int_field, str_field, str_list_field
from ainode.discovery.cluster import ClusterState
from ainode.engine.parallelism import (
    ParallelPlanError,
    Strategy,
    plan_for_model,
)
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
    app.router.add_post("/api/sharding/relaunch", handle_sharding_relaunch)
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


def _refuse_distributed_image(app, model: str) -> str:
    """"" unless this is an image model, which cannot be split at all."""
    try:
        from ainode.planner.api_routes import _is_image, _recipe

        if not _is_image(_recipe(app, model), None, app, model):
            return ""
    except Exception:
        logger.debug("could not tell whether %s is an image model", model,
                     exc_info=True)
        return ""
    return (
        f"{model} is an image model, and the image engine runs one process on "
        f"one node — there is no tensor or pipeline axis to split it along. "
        f"Select a single node and launch it there. Which node is free "
        f"choice: any of them can serve it, provided it has the memory."
    )


def _remembered_placement(app, model: str):
    """Where this model was last told to run, or None.

    Never raises: a placement is a convenience, and a broken placement file or
    a missing store must cost the operator a click, not a launch.
    """
    try:
        from ainode.placement.api_routes import get_placement_store

        placement = get_placement_store(app).get(model)
    except Exception:
        logger.debug("could not read the placement for %s", model, exc_info=True)
        return None
    if placement is not None:
        logger.info("placement for %s: nodes=%s strategy=%s",
                    model, placement.node_ids, placement.strategy or "auto")
    return placement


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

    body = as_object(body)
    model = str_field(body, "model")
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    from ainode.models.api_routes import drafter_base_model

    base = drafter_base_model(model)
    if base:
        return web.json_response({
            "error": (
                f"{model} is a speculative-decoding draft model, not a servable "
                f"one. Load {base} instead — its catalog recipe pairs this "
                f"drafter automatically."
            ),
            "load_instead": base,
        }, status=422)

    # An image model has no parallel axis at all: the diffusers engine runs
    # one process on one node. Selecting two and pressing launch would form a
    # Ray cluster for vLLM and fail minutes later with a message about a
    # checkpoint vLLM cannot read — true, and no help in working out that the
    # mistake was the node count.
    image_refusal = _refuse_distributed_image(request.app, model)
    if image_refusal:
        return web.json_response({"error": image_refusal}, status=422)

    min_nodes = int_field(body, "min_nodes", default=1, minimum=1) or 1

    # Explicit node selection (preferred): the exact nodes to span, head = this
    # node + the rest as peers. `tp_size` is the legacy count form. Either sets
    # the effective node count so the min_nodes<=1 solo path still triggers.
    node_ids = str_list_field(body, "node_ids") or None
    remembered = None
    if not node_ids:
        # Nothing chosen for this launch: use where this model was last told
        # to run. A cluster settles into an arrangement, and re-picking the
        # nodes on every relaunch, every restart and every retry after a
        # failed load is the difference between running a cluster and
        # re-configuring one. An explicit choice always wins.
        remembered = _remembered_placement(request.app, model)
        if remembered is not None and remembered.node_ids:
            node_ids = list(remembered.node_ids)
    if node_ids:
        min_nodes = len(node_ids)
    else:
        tp_size = int_field(body, "tp_size", minimum=1)
        if tp_size:
            min_nodes = tp_size

    # Parallelism axis. Previously read and ignored ("any min_nodes > 1
    # triggers TP"), which is why the UI's Pipeline pill did nothing and why a
    # 3-node launch would have built the unsupported TP=3. The plan itself is
    # resolved further down, once the participating nodes are known.
    # Deliberately NOT coerced with str_field: Strategy.parse already rejects a
    # non-string with a 400 naming the valid axes, and silently defaulting a
    # bogus `strategy` to "auto" would hand the caller a working launch on an
    # axis they did not ask for — on a cluster, for minutes.
    strategy_str = body.get("strategy")
    if (strategy_str is None or strategy_str == "") and remembered is not None:
        # Same rule as the nodes: remembered only when the caller said
        # nothing. Stored empty means "let the planner decide", which is what
        # an absent value already does, so it changes nothing.
        strategy_str = remembered.strategy or strategy_str
    try:
        strategy = Strategy.parse(strategy_str)
    except ParallelPlanError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    cluster: ClusterState = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]

    # The same gate the solo path uses. This path had none at all, and it is
    # the one that took two nodes down: it launches on machines the head
    # cannot see the memory of, so nothing discovered the overcommit until the
    # engines had loaded their weights and started sizing their caches.
    from ainode.safety.admission import check_admission

    refusal = check_admission(
        request.app, model, node_ids=node_ids,
        strategy=str(strategy_str or "auto"),
        max_model_len=int_field(body, "max_model_len", minimum=0) or 0,
        force=bool(body.get("force")))
    if refusal:
        return web.json_response({"error": refusal, "refused_by": "admission"},
                                 status=507)

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
    def _status_of(n):
        return n.status.value if hasattr(n.status, "value") else str(n.status)

    def fabric_of(n):
        return (getattr(n, "fabric_ip", "") or "").strip()

    # Eligibility is being reachable, not being idle.
    #
    # This used to require distributed_mode == "member". That field records
    # what a node LAST DID, not what it can do: loading one solo model sets it
    # to "solo" and it never goes back, so a node that had ever served
    # anything could never join a distributed launch again. On a cluster where
    # every node had served a model — which is every cluster that has been
    # used — every selection was refused with "not available as members",
    # whichever nodes were picked.
    #
    # Whether a node has the MEMORY is a different question, answered below
    # and by the engine itself. A node that is busy is still a node.
    members = [
        n for n in cluster.members()
        if n.node_id != config.node_id
        and _status_of(n) in ("online", "member-ready", "serving")
    ]
    everyone = list(cluster.members())
    members_dump = [
        {"node_id": n.node_id, "node_name": n.node_name, "fabric_ip": fabric_of(n),
         "status": _status_of(n),
         "distributed_mode": getattr(n, "distributed_mode", ""),
         "model": getattr(n, "model", "") or ""}
        for n in everyone
    ]

    if node_ids:
        # Head is always this node; peers are the other selected nodes.
        wanted = [nid for nid in node_ids if nid != config.node_id]
        by_id = {n.node_id: n for n in members}
        missing = [nid for nid in wanted if nid not in by_id]
        if missing:
            # Say which node and why, not just that something is wrong. The
            # three reasons need three different actions from the operator.
            known = {n.node_id: n for n in everyone}
            reasons = []
            for nid in missing:
                node = known.get(nid)
                if node is None:
                    reasons.append(f"{nid}: not discovered — check the cluster_id "
                                   f"and that AINode is running there")
                else:
                    reasons.append(f"{nid} ({getattr(node, 'node_name', '')}): "
                                   f"status {_status_of(node)}")
            return web.json_response({
                "error": "Selected node(s) cannot take part: " + "; ".join(reasons),
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
    from ainode.models.api_routes import catalog_proven_tp, catalog_supports_pipeline

    try:
        plan, plan_note = plan_for_model(
            strategy, 1 + len(chosen_peers), catalog_proven_tp(model),
            supports_pipeline=catalog_supports_pipeline(model),
        )
    except ParallelPlanError as exc:
        return web.json_response({
            "error": str(exc),
            "strategy": strategy.value,
            "node_count": 1 + len(chosen_peers),
        }, status=422)
    if plan_note:
        logger.info("%s: %s", model, plan_note)

    # A peer that is already serving has its memory committed to that model.
    # Not a refusal — the operator may be about to unload it, and the engine
    # reports an out-of-memory far more precisely than a guess here could —
    # but worth saying before twenty minutes of loading.
    busy = [f"{n.node_name or n.node_id} ({n.model})"
            for n in chosen if getattr(n, "model", "")]
    if busy:
        busy_note = ("Already serving, so their memory is committed: "
                     + ", ".join(busy) + ". Unload first if this launch needs it.")
        plan_note = f"{plan_note} {busy_note}".strip() if plan_note else busy_note
        logger.info("%s: %s", model, busy_note)

    # Second address list: where head and peer share a direct RoCE cable, bulk
    # transfer (model weights) goes over it instead of the coordination
    # Ethernet. Coordination itself stays on chosen_peers — on a mesh no single
    # RoCE address reaches every node. Peers that announce no RoCE address, or
    # that we have no cable to, are simply absent and transfer as before.
    peer_transfer_ips: dict = {}
    try:
        from ainode.cluster.topology import detect_cx7_links, transfer_address

        # Off the event loop: detection shells out to `ip` once per interface,
        # each with a 5 s timeout. On a healthy node that is milliseconds, but a
        # NIC in a bad state would freeze every other request — including the
        # chat proxy — for as long as it takes to time out.
        local_links = await asyncio.get_event_loop().run_in_executor(
            None, detect_cx7_links
        )
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
    # The same per-load knobs the solo path accepts — context length, KV dtype,
    # quantisation, extra vLLM flags, engine image. A distributed launch used to
    # drop all of them and honour only gpu_memory_utilization, so the launch
    # that most needs a batching flag or a context limit was the one that could
    # not carry them.
    from ainode.models.api_routes import (
        apply_catalog_recipe,
        apply_tool_calling,
        parse_load_overrides,
    )

    overrides, err = parse_load_overrides(body)
    if err is not None:
        return err

    gmu_raw = body.get("gpu_memory_utilization")
    if gmu_raw is not None:
        try:
            gmu = float(gmu_raw)
        except (TypeError, ValueError):
            gmu = None
        if gmu is not None and 0.0 < gmu <= 1.0:
            overrides["gpu_memory_utilization"] = gmu

    # The model's proven recipe fills whatever the caller left out — the same
    # way the solo path does it, so the same model is configured identically
    # however many nodes it spans.
    overrides, recipe_gmu = apply_catalog_recipe(
        model, overrides, overrides.get("gpu_memory_utilization")
    )
    if recipe_gmu is not None:
        overrides["gpu_memory_utilization"] = recipe_gmu
    overrides = apply_tool_calling(model, overrides, str_field(body, "tool_calling"))

    inst_config = replace(config, model=model, distributed_mode="head",
                          peer_ips=chosen_peers, peer_transfer_ips=peer_transfer_ips,
                          parallel_strategy=plan.strategy.value,
                          tensor_parallel_size=plan.tensor_parallel_size,
                          pipeline_parallel_size=plan.pipeline_parallel_size,
                          data_parallel_size=plan.data_parallel_size,
                          api_port=port, **overrides)
    backend = get_backend(inst_config, instance_id=name_token)
    try:
        # In a worker thread: start_distributed SSHes to every peer, compares
        # engine images, mirrors the weights and forms the Ray cluster — all
        # synchronous, all minutes. Run inline it froze the head's own event
        # loop, which is where the UDP announcement and the whole API live.
        started = await asyncio.get_event_loop().run_in_executor(
            None, backend.start_distributed)
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
        # When the split is not the one that was asked for, say so where the
        # operator is looking. A log line explaining that a model proven at
        # TP=2 is being pipelined across three nodes helps nobody who is
        # watching the dashboard.
        "note": plan_note,
    })


async def handle_sharding_relaunch(request: web.Request) -> web.Response:
    """POST /api/sharding/relaunch — re-run a degraded instance on the nodes
    that are still online.

    JSON body:
        model (required): the model of the instance to relaunch
        strategy (optional): axis to use; default auto (re-planned for the
            smaller node set, so a TP=4 instance losing a node comes back as
            PP=3 rather than an impossible TP=3)

    An instance keeps running on the head after a member node disappears, but
    with ranks that Ray placed on the lost node — it cannot serve. Relaunching
    is deliberately an explicit action rather than something the head does by
    itself: a model spread across three nodes usually does not fit on two, and
    guessing would trade a visible outage for an OOM. This endpoint therefore
    checks the fit first and explains a refusal instead of trying.
    """
    from ainode.core.config import NodeConfig

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    body = as_object(body)
    model = str_field(body, "model")
    if not model:
        return web.json_response({"error": "model field required"}, status=400)

    cluster: ClusterState = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]

    manager = request.app.get("instances")
    instance = manager.by_model(model) if manager is not None else None
    if instance is None:
        return web.json_response(
            {"error": f"No instance is running for {model!r} on this node."},
            status=404,
        )

    launched_peers = list(getattr(instance.record, "peer_ips", []) or [])
    if not launched_peers:
        return web.json_response(
            {"error": f"{model!r} is a single-node instance; there is nothing "
                      f"to relaunch across."},
            status=409,
        )

    # Which of the nodes it launched across are still online?
    online = {
        (getattr(n, "fabric_ip", "") or ""): n
        for n in cluster.members()
        if (n.status.value if hasattr(n.status, "value") else str(n.status))
        in ("online", "member-ready", "serving")
    }
    surviving = [online[ip] for ip in launched_peers if ip in online]
    missing = [ip for ip in launched_peers if ip not in online]

    if not missing:
        return web.json_response({
            "error": f"{model!r} is not degraded — all "
                     f"{len(launched_peers)} peer(s) are online.",
            "hint": "Use /api/sharding/launch to change its placement.",
        }, status=409)

    node_ids = [config.node_id or "head"] + [n.node_id for n in surviving]
    node_count = len(node_ids)

    # Two ways a relaunch can be impossible, and the caller deserves to know
    # which. First: no split of this axis fills the surviving nodes (a TP=4
    # instance down to 3 nodes has no valid tensor split).
    strategy = body.get("strategy")   # see the note in handle_sharding_launch
    try:
        from ainode.models.api_routes import (
            catalog_proven_tp,
            catalog_supports_pipeline,
        )

        plan, _ = plan_for_model(
            strategy, node_count, catalog_proven_tp(model),
            supports_pipeline=catalog_supports_pipeline(model))
    except ParallelPlanError as exc:
        return web.json_response({
            "error": str(exc), "model": model,
            "surviving_node_ids": node_ids, "missing_peer_ips": missing,
        }, status=422)

    # Second: the weights no longer fit. This is the common case and the whole
    # reason the head does not do this by itself.
    fit_error = _fit_check(model, surviving, cluster, config, plan)
    if fit_error:
        return web.json_response({
            "error": fit_error, "model": model,
            "surviving_node_ids": node_ids, "missing_peer_ips": missing,
            "parallel_plan": plan.to_dict(),
        }, status=422)

    logger.info(
        "Relaunching %s on %d surviving node(s) as %s (lost %s)",
        model, node_count, plan.label(), ", ".join(missing),
    )

    # Delegate to the launch path so placement, transfer-address resolution and
    # instance bookkeeping have exactly one implementation. It stops the
    # existing instance for this model before starting the replacement.
    class _ReqShim:
        def __init__(self, orig, payload):
            self._o = orig
            self._b = payload

        def __getattr__(self, k):
            return getattr(self._o, k)

        async def json(self):
            return self._b

    payload = {"model": model, "node_ids": node_ids,
               "strategy": plan.strategy.value}
    if body.get("gpu_memory_utilization") is not None:
        payload["gpu_memory_utilization"] = body["gpu_memory_utilization"]

    resp = await handle_sharding_launch(_ReqShim(request, payload))
    if resp.status == 200:
        merged = json.loads(resp.body)
        merged["relaunched_from"] = {
            "node_count": 1 + len(launched_peers),
            "missing_peer_ips": missing,
        }
        return web.json_response(merged)
    return resp


def _fit_check(model, surviving_nodes, cluster, config, plan) -> str:
    """Return a human-readable reason the model will not fit, or "".

    Deliberately conservative and deliberately approximate: it uses the same
    size heuristic the planner shows in the UI preview, so a refusal here
    matches the number the operator already saw. It exists to catch the
    obvious "three nodes' worth of weights onto two" case before a launch
    burns minutes and ends in an OOM — not to be authoritative about memory.
    """
    from ainode.engine.sharding import MEMORY_OVERHEAD_FACTOR, estimate_model_size

    local = cluster.get_node(config.node_id) if config.node_id else None
    nodes = ([local] if local is not None else []) + list(surviving_nodes)
    if not nodes:
        return ""

    required = estimate_model_size(model) * MEMORY_OVERHEAD_FACTOR
    # Data parallelism keeps a full replica per node; the others divide the
    # weights, so the per-node share is what has to fit.
    per_node = required if plan.data_parallel_size > 1 else required / len(nodes)

    def free_gb(n):
        total = float(getattr(n, "gpu_memory_gb", 0) or 0)
        used = float(getattr(n, "gpu_memory_used_mb", 0) or 0) / 1024.0
        return max(0.0, total - used)

    short = [n for n in nodes if free_gb(n) < per_node]
    if not short:
        return ""

    names = ", ".join(
        f"{getattr(n, 'node_name', n.node_id)} (~{free_gb(n):.0f} GB free)"
        for n in short
    )
    return (
        f"{model} needs ~{per_node:.0f} GB per node as {plan.label()} across "
        f"{len(nodes)} node(s), but {names} cannot hold that. Free memory on "
        f"those nodes, bring the missing node back, or load a smaller model."
    )


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
