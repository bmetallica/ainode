"""AINode API proxy server — aiohttp app that serves the web UI and proxies to vLLM."""

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import aiohttp
from urllib.parse import quote
from aiohttp import web

from ainode.api.params import str_field
from ainode.core.config import NodeConfig
from ainode.core.gpu import detect_gpu, GPUInfo
from ainode.web.serve import get_index_html, get_onboarding_html, get_static_path
from ainode.models.api_routes import register_model_routes
from ainode.onboarding.api_routes import register_onboarding_routes
from ainode.auth.middleware import AuthConfig, auth_middleware
from ainode.auth.api_routes import register_auth_routes
from ainode.metrics.collector import MetricsCollector
from ainode.metrics.api_routes import register_metrics_routes
from ainode.training.engine import TrainingManager
from ainode.training.api_routes import setup_training_routes
from ainode.datasets.manager import DatasetManager
from ainode.datasets.api_routes import setup_dataset_routes
from ainode.discovery.broadcast import (
    BroadcastSender,
    BroadcastListener,
    NodeAnnouncement,
)
from ainode.discovery.cluster import ClusterState
from ainode.engine.sharding_routes import register_sharding_routes
from ainode.engine.ray_autostart import (
    RayAutostartState,
    autostart_loop as _ray_autostart_loop,
)
from ainode.secrets import SecretsManager
from ainode.secrets.api_routes import register_secrets_routes
from ainode.embeddings.manager import EmbeddingManager
from ainode.bench.api_routes import register_bench_routes
from ainode.embeddings.api_routes import register_embedding_routes
from ainode.placement.api_routes import register_placement_routes
from ainode.assist.api_routes import register_assist_routes
from ainode.planner.api_routes import register_planner_routes
from ainode.measure.api_routes import register_measurement_routes
from ainode.safety.api_routes import register_safety_routes
from ainode.profiles.api_routes import register_profile_routes
from ainode.profiles.store import ProfileStore
from ainode.telemetry.api_routes import register_telemetry_routes
from ainode.telemetry.mqtt import MqttPublisher
from ainode.api.server_routes import (
    register_server_routes,
    request_log_middleware,
    init_server_state,
)

from ainode import __version__

logger = logging.getLogger(__name__)

def _client_max_bytes(config) -> int:
    """Inbound request-body ceiling for the API server, in bytes.

    NOT optional: aiohttp defaults to 1 MB, which rejects long-context prompts
    at the proxy. A model advertising 262k context can only be fed ~190k tokens
    through our own endpoint before the caller gets an opaque 413 that says
    nothing about which hop refused it (found 2026-08-25 benchmarking decode
    against context depth). A bad value falls back to the default rather than
    producing a server that rejects every body.
    """
    try:
        mb = int(getattr(config, "max_request_mb", 64))
    except (TypeError, ValueError):
        mb = 64
    return max(1, mb) * 1024 * 1024


def create_app(
    config: Optional[NodeConfig] = None,
    engine=None,
) -> web.Application:
    """Create and return the aiohttp application.

    Parameters
    ----------
    config : NodeConfig
        Node configuration (defaults created if None).
    engine : EngineBackend | VLLMEngine | None
        Optional engine instance for health/status queries. May be any of
        the concrete backends in :mod:`ainode.engine.backends` (eugr,
        nvidia) or the legacy :class:`ainode.engine.vllm_engine.VLLMEngine`
        pip-venv engine.
    """
    if config is None:
        config = NodeConfig()

    auth_config = AuthConfig.load()

    app = web.Application(
        middlewares=[cors_middleware, request_log_middleware, auth_middleware],
        client_max_size=_client_max_bytes(config),
    )
    init_server_state(app)
    # Instantiate shared services
    collector = MetricsCollector()
    dataset_manager = DatasetManager()
    manager = TrainingManager(dataset_manager=dataset_manager)

    # Build local node announcement for discovery
    announcement = _build_announcement(config, engine)
    cluster = ClusterState(local_announcement=announcement)

    app["config"] = config
    app["auth_config"] = auth_config
    app["engine"] = engine
    # Seed the InstanceManager with the boot engine as the PRIMARY instance, so
    # a later solo load APPENDS on the next port (8001…) instead of colliding
    # with the boot container on the legacy name/port. The boot engine is owned
    # by `ainode start` and binds `ainode-vllm-node-solo` + api_port; without
    # this seed the manager would hand the same name/port to a 2nd backend and
    # the two fight (repeated docker-name Conflict, neither serving).
    if engine is not None and getattr(config, "model", None) \
            and (getattr(config, "distributed_mode", "solo") or "solo") == "solo":
        from ainode.discovery.instance import InstanceRecord
        from ainode.engine.instance_manager import InstanceManager
        _seed = InstanceManager(base_port=config.api_port)
        _seed.add(InstanceRecord(
            instance_id=f"{config.node_id or 'head'}:{config.model}",
            model=config.model, head_node_id=config.node_id or "head",
            peer_ips=[], api_port=config.api_port, tensor_parallel_size=1,
            status="starting"), engine)
        app["instances"] = _seed
    app["start_time"] = time.time()
    app["client_session"] = None  # lazy-init in startup
    app["metrics_collector"] = collector
    app["training_manager"] = manager
    app["dataset_manager"] = dataset_manager
    app["cluster_state"] = cluster
    app["announcement"] = announcement
    app["broadcast_sender"] = None
    app["broadcast_listener"] = None
    app["secrets_manager"] = SecretsManager()
    # models_dir, so embedding weights land where LLM weights land: mounted,
    # visible to list_downloaded(), and carried by the mirror to every node.
    app["embedding_manager"] = EmbeddingManager(
        models_dir=getattr(config, "models_dir", "") or "")
    # Profiles are read at construction, before the app starts serving, so the
    # startup restore and the routes share one store.
    app["profiles"] = ProfileStore()
    app["mqtt_publisher"] = MqttPublisher(app)
    # Ray autostart is only meaningful for the legacy eugr backend. The NVIDIA
    # backend manages its own Ray lifecycle via run_cluster.sh at model-load time;
    # running `ray start` here fights with that (session-name mismatch on peer
    # nodes, port conflicts on port 6379). Disable the autostart loop in that case.
    _engine_backend_for_ray = (getattr(config, "engine_backend", None) or "eugr").lower()
    app["ray_autostart_state"] = RayAutostartState(enabled=(_engine_backend_for_ray == "eugr"))

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # The browser UI, unless this node is not operated from a browser. The
    # API below is registered either way: discovery, the cluster dispatch that
    # places models here, the federated proxy and /v1/* all go through it.
    if getattr(config, "web_ui_enabled", True):
        app.router.add_get("/", handle_index)
        app.router.add_get("/onboarding", handle_onboarding)
    else:
        app.router.add_get("/", _web_ui_disabled)
        app.router.add_get("/onboarding", _web_ui_disabled)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/nodes", handle_nodes)
    app.router.add_get("/api/cluster/info", handle_cluster_info)
    app.router.add_get("/api/cluster/resources", handle_cluster_resources)
    app.router.add_post("/api/cluster/role", handle_cluster_set_role)
    app.router.add_post("/api/cluster/id", handle_cluster_set_id)
    app.router.add_post("/api/cluster/load", handle_cluster_load)
    # Clearing the compile cache: local, and node-targeted through the same
    # dispatch the load routes use, because the cache that matters is on the
    # node whose launch failed.
    app.router.add_post("/api/engine/compile-cache", _clear_compile_cache)
    app.router.add_post("/api/cluster/compile-cache", handle_cluster_compile_cache)
    app.router.add_get("/api/instances/launch-config", handle_launch_config)
    app.router.add_get("/api/cluster/models", handle_cluster_models)
    app.router.add_post("/api/cluster/delete-repo", handle_cluster_delete_repo)
    app.router.add_get("/api/cluster/safety/memory", handle_cluster_memory_get)
    app.router.add_put("/api/cluster/safety/memory", handle_cluster_memory_put)
    app.router.add_get("/api/clients/opencode", handle_opencode_config)
    app.router.add_post("/api/cluster/mirror-models", handle_cluster_mirror_models)
    app.router.add_get("/api/cluster/mirror-status", handle_cluster_mirror_status)
    app.router.add_post("/api/cluster/embeddings/load", handle_cluster_embedding_load)
    app.router.add_post("/api/cluster/embeddings/unload", handle_cluster_embedding_unload)
    app.router.add_post("/api/cluster/unload", handle_cluster_unload)
    app.router.add_post("/api/cluster/update-all", handle_cluster_update_all)
    app.router.add_get("/api/cluster/update-status", handle_cluster_update_status)
    app.router.add_get("/api/config", handle_get_config)
    app.router.add_post("/api/engine/set-model", handle_set_model)
    app.router.add_get("/api/version/check", handle_version_check)
    app.router.add_post("/api/engine/update", handle_engine_update)
    app.router.add_patch("/api/config", handle_patch_config)

    app.router.add_get("/v1/models", handle_v1_models)
    app.router.add_post("/v1/chat/completions", proxy_to_vllm)
    app.router.add_post("/v1/completions", proxy_to_vllm)
    # Images, through the same proxy. It routes on the model name and forwards
    # the path verbatim, so an image instance on node 3 is reachable from the
    # head exactly like a chat model — no second routing table, no special
    # case. The generation itself can take minutes; the proxy already leaves
    # the total timeout uncapped for a slow cold start.
    app.router.add_post("/v1/images/generations", proxy_to_vllm)

    register_model_routes(app)

    register_onboarding_routes(app)

    register_auth_routes(app)

    # --- Metrics routes ------------------------------------------------------
    register_metrics_routes(app, collector)

    # --- Training routes -----------------------------------------------------
    setup_training_routes(app, manager)

    # --- Dataset routes ------------------------------------------------------
    setup_dataset_routes(app, dataset_manager)

    # --- Sharding routes ----------------------------------------------------
    register_sharding_routes(app)

    # --- Secrets routes ------------------------------------------------------
    register_secrets_routes(app)

    # --- Embedding routes ----------------------------------------------------
    register_embedding_routes(app)

    # --- Profile routes ------------------------------------------------------
    register_profile_routes(app)

    register_placement_routes(app)

    # The error assistant. Registered on every node: the head asks a peer for
    # its engine log, and a peer's own UI has to be able to diagnose too.
    register_assist_routes(app)

    # The launch planner. Read-only: it computes what a launch would do, and
    # the launch form fills itself in from it.
    register_planner_routes(app)

    # What each model actually cost here, so the next plan can prefer a
    # measurement to an estimate.
    register_measurement_routes(app)

    # The host memory guard. Registered everywhere: the nodes are what crashed.
    register_safety_routes(app)

    # --- Telemetry routes ----------------------------------------------------
    register_telemetry_routes(app)

    # --- Server view routes --------------------------------------------------
    register_server_routes(app)

    register_bench_routes(app)

    if getattr(config, "web_ui_enabled", True):
        app.router.add_static("/static", get_static_path(), name="static")

    return app


async def _web_ui_disabled(request: web.Request) -> web.Response:
    """Say why, rather than 404. A blank page on the wrong port is a support
    ticket; "operate this cluster from the head" is an answer."""
    config: NodeConfig = request.app["config"]
    head = ""
    cluster = request.app.get("cluster_state")
    if cluster is not None:
        try:
            for node in cluster.get_nodes():
                if getattr(node, "is_master", False) and getattr(node, "fabric_ip", ""):
                    head = f"http://{node.fabric_ip}:{node.web_port}"
                    break
        except Exception:
            logger.debug("could not name the head", exc_info=True)
    return web.Response(
        status=404,
        content_type="text/plain",
        text=(f"The web UI is switched off on {config.node_name or config.node_id}.\n"
              f"This node is operated from the head"
              + (f": {head}\n" if head else ".\n")
              + "Its API is unaffected — /api/health, /v1/models and the "
                "cluster routes all answer as usual.\n"),
    )


def _head_instances(config) -> list:
    """Phase 2: the instances list this node HEADS, as wire dicts.

    Today there's at most one (derived from config), so this returns a 0- or
    1-element list. When the engine layer owns multiple instances (P2-2) it will
    return all of them. Mirrors the legacy distributed_instance_id/peers.
    """
    from ainode.discovery.instance import InstanceRecord

    peer_ips = list(getattr(config, "peer_ips", []) or [])
    if not peer_ips:
        return []
    iid = f"{config.node_id or 'head'}:{config.model}"

    # The split this head is actually running. A config that carries no
    # resolved sizes (an older config.json, or a systemd-path launch that
    # never went through /api/sharding/launch) reads as tensor-parallel across
    # every node — what this advertised before the other axes existed.
    from ainode.engine.parallelism import ParallelPlan

    plan = ParallelPlan.from_dict({
        "tensor_parallel_size": getattr(config, "tensor_parallel_size", 0) or 0,
        "pipeline_parallel_size": getattr(config, "pipeline_parallel_size", 0) or 0,
        "data_parallel_size": getattr(config, "data_parallel_size", 0) or 0,
    })
    if plan.world_size != 1 + len(peer_ips) or not plan.is_distributed:
        plan = ParallelPlan(tensor_parallel_size=1 + len(peer_ips))

    return [InstanceRecord(
        instance_id=iid,
        model=config.model or "",
        head_node_id=config.node_id or "unknown",
        peer_ips=peer_ips,
        api_port=config.api_port,
        tensor_parallel_size=plan.tensor_parallel_size,
        pipeline_parallel_size=plan.pipeline_parallel_size,
        data_parallel_size=plan.data_parallel_size,
        status="serving",
    ).to_dict()]


def _build_announcement(config: NodeConfig, engine=None) -> NodeAnnouncement:
    """Create a NodeAnnouncement from current node state."""
    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_name = gpu.name if gpu else "CPU"
    gpu_memory_gb = round(gpu.memory_total_mb / 1024, 1) if gpu else 0.0
    unified_memory = gpu.unified_memory if gpu else False

    # This node's address for a head to reach us on: SSH, Ray, model transfer
    # (BUG D fix — not the mgmt-LAN UDP source address).
    #
    # Taken from the COORDINATION interface, which off a mesh is
    # cluster_interface, i.e. unchanged. On a switchless mesh it has to be the
    # shared Ethernet: a node's CX7 addresses each reach exactly one neighbour,
    # so announcing one would leave the third node unable to reach us at all.
    # Part B adds the per-link IB addresses alongside this, for bulk transfer.
    fabric_ip = ""
    ib_ips: list = []
    try:
        from ainode.cluster.hca_discovery import detect_fabric_ip
        from ainode.cluster.topology import local_ib_ips, topology_for_config

        # One detection, reused: topology_for_config already walked sysfs and
        # resolved every link's address, and each walk costs an `ip` subprocess
        # per interface. Calling detect_cx7_links() again here doubled that.
        topo = topology_for_config(config)
        fabric_ip = detect_fabric_ip(topo.coord_interface) or ""
        # The RoCE link addresses, so a head with a cable to us can push model
        # weights over it instead of the shared Ethernet (see transfer_address).
        ib_ips = local_ib_ips(topo.links)
    except Exception:
        pass

    engine_ready = False
    if engine is not None:
        engine_ready = getattr(engine, "ready", False)

    distributed_mode = getattr(config, "distributed_mode", "solo") or "solo"
    # Member nodes report "member-ready" so the UI can distinguish them
    # from solo nodes that just haven't loaded a model yet.
    if distributed_mode == "member":
        status = "member-ready"
    elif engine_ready:
        status = "serving"
    else:
        status = "starting"

    # Head nodes with an active distributed instance advertise the instance id
    # and the peer node ids it spans so the UI can paint "DISTRIBUTED TP=N".
    # We use peer_ips as the peer key here — the UI resolves them against
    # discovered members.
    distributed_instance_id = None
    distributed_peers: list = []
    if distributed_mode == "head" and engine_ready:
        peer_ips = list(getattr(config, "peer_ips", []) or [])
        if peer_ips:
            distributed_instance_id = f"{config.node_id or 'head'}:{config.model}"
            distributed_peers = peer_ips

    return NodeAnnouncement(
        node_id=config.node_id or "unknown",
        node_name=config.node_name or socket.gethostname() or "unknown",
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        unified_memory=unified_memory,
        model=config.model or "" if distributed_mode != "member" else "",
        status=status,
        api_port=config.api_port,
        web_port=config.web_port,
        cluster_id=getattr(config, "cluster_id", "default"),
        role=getattr(config, "cluster_role", "auto"),
        is_master=False,  # runtime flag, updated by sync loop
        distributed_mode=distributed_mode,
        distributed_instance_id=distributed_instance_id,
        distributed_peers=distributed_peers,
        fabric_ip=fabric_ip,
        ib_ips=ib_ips,
        instances=(_head_instances(config) if (distributed_mode == "head" and engine_ready) else []),
        version=__version__,
    )


async def _on_startup(app: web.Application) -> None:
    app["client_session"] = aiohttp.ClientSession()

    # Before anything else can be launched. Its own thread, deliberately: the
    # launch path blocks this event loop for minutes, and a guard living in
    # the loop would be deaf during exactly the window it exists to cover.
    from ainode.safety.memory_guard import MemoryGuard

    guard = MemoryGuard(
        app,
        warn_gb=float(getattr(app["config"], "host_memory_warn_gb", 8.0) or 8.0),
        critical_gb=float(
            getattr(app["config"], "host_memory_critical_gb", 4.0) or 4.0),
        enabled=bool(getattr(app["config"], "host_memory_guard", True)),
    )
    def _announce_stop(action: dict) -> None:
        from ainode.telemetry.mqtt import publish_event

        publish_event(app, "safety")

    guard.on_action = _announce_stop
    app["memory_guard"] = guard
    try:
        guard.start()
    except Exception:
        logger.exception("could not start the host memory guard")

    # Telemetry, if it is configured. Started before the engine work below so
    # a node that fails to bring a model up still reports why it is unhappy.
    publisher = app.get("mqtt_publisher")
    if publisher is not None:
        try:
            if publisher.start():
                logger.info("MQTT telemetry publishing every %ss",
                            app["config"].mqtt_interval)
        except Exception:
            logger.exception("could not start MQTT telemetry")

    config: NodeConfig = app["config"]
    announcement: NodeAnnouncement = app["announcement"]
    cluster: ClusterState = app["cluster_state"]

    # Always-on: re-load the persisted solo instance set so a `systemctl restart
    # ainode` brings every previously-loaded model back with no manual step.
    # Skipped when the operator requested a clean boot (closet #310, set in the
    # CLI start path) so a restart can actually free the node.
    if getattr(config, "_skip_replay", False):
        logger.info("start-clean: skipping persisted-instance replay")
    else:
        try:
            # Embedding models are replayed too — they are in-process, so a
            # restart drops them, and a RAG pipeline breaks silently until
            # someone notices and clicks Load again.
            try:
                emb = app.get("embedding_manager")
                if emb is not None:
                    asyncio.get_event_loop().run_in_executor(None, emb.replay)
            except Exception:
                logger.exception("embedding replay could not be scheduled")

            # A default profile describes the whole deployment, including
            # distributed instances the manifest cannot record. When one is
            # set it replaces the replay rather than running alongside it —
            # two sources for "what should be running" contradict each other.
            from ainode.models.api_routes import replay_instances_on_startup
            from ainode.profiles.apply import startup_restore

            async def _restore() -> None:
                applied = False
                try:
                    applied = await startup_restore(app)
                except Exception:
                    logger.exception("default profile restore failed")
                if not applied:
                    await replay_instances_on_startup(app)

            app["_instance_replay_task"] = asyncio.get_event_loop().create_task(
                _restore()
            )
        except Exception:
            logger.exception("Failed to schedule instance replay")

    if config.cluster_enabled:
        # Start broadcast sender
        _collector = app.get("metrics_collector")
        sender = BroadcastSender(
            announcement=announcement,
            discovery_port=config.discovery_port,
            # Stamp live GPU telemetry onto every broadcast so the head can
            # render real per-peer VRAM/util (metrics fan-out).
            metrics_provider=(_collector.get_gpu_metrics if _collector else None),
        )
        await sender.start()
        app["broadcast_sender"] = sender
        logger.info("Discovery sender started on port %d", config.discovery_port)

        # Start broadcast listener
        def on_node_found(ann: NodeAnnouncement):
            logger.info("Discovered node %s (%s)", ann.node_id, ann.node_name)

        def on_node_lost(node_id: str):
            logger.info("Lost node %s", node_id)
            cluster.remove_node(node_id)

        listener = BroadcastListener(
            local_node_id=announcement.node_id,
            discovery_port=config.discovery_port,
            on_node_found=on_node_found,
            on_node_lost=on_node_lost,
        )
        await listener.start()
        app["broadcast_listener"] = listener
        logger.info("Discovery listener started on port %d", config.discovery_port)

        # Start a periodic task to sync listener registry into ClusterState
        app["_cluster_sync_task"] = asyncio.get_event_loop().create_task(
            _cluster_sync_loop(app)
        )

        # Kick off Ray autostart — master starts head, workers join once a
        # master is discovered. Gracefully no-ops if ray is not installed.
        def _get_master_address() -> Optional[str]:
            master = cluster.get_master()
            if master is None:
                return None
            # Same node → no remote address needed
            if master.node_id == announcement.node_id:
                return None
            # node_name can be None/"unknown" (no hostname resolution yet); the
            # Ray join would hang for 60s on a bogus address and block the
            # asyncio event loop. Skip until we have real peer IP plumbing.
            name = master.node_name
            if not name or name in ("unknown", "localhost", "None"):
                return None
            return f"{name}:6379"

        if app["ray_autostart_state"].enabled:
            app["_ray_autostart_task"] = asyncio.get_event_loop().create_task(
                _ray_autostart_loop(
                    cluster_state=cluster,
                    get_master_address=_get_master_address,
                    state=app["ray_autostart_state"],
                )
            )


async def _engine_serving(backend, loop) -> bool:
    """True iff the engine's OpenAI API actually answers right now.

    The latched `ready` flag never flips False when an engine crashes or is
    killed out-of-band, so a dead engine reads READY forever (phantom-READY →
    ghost routing → 502s). An active localhost probe is the truthful signal —
    and it correctly reads False while a model is still loading (api not up
    yet), so we never advertise a not-yet-serving OR already-dead engine.

    health_check() uses a blocking 5s-timeout urlopen, so run it in the default
    executor to keep the event loop free (a hung engine must not stall sync).
    """
    if backend is None:
        return False
    try:
        hc = await loop.run_in_executor(None, backend.health_check)
        return bool(hc.get("api_responding"))
    except Exception:
        return False


def _stamp_load_state(inst) -> None:
    """Copy the backend's load state onto the record that gets advertised.

    The record is what crosses the wire to the head; the backend does not.
    Without this the dashboard had one error and one phase per NODE and drew
    them on every card it had.
    """
    backend = getattr(inst, "backend", None)
    record = getattr(inst, "record", None)
    if backend is None or record is None:
        return
    for field in ("load_error", "load_phase", "load_detail"):
        try:
            setattr(record, field, str(getattr(backend, field, "") or ""))
        except Exception:
            logger.debug("could not stamp %s", field, exc_info=True)
    # Not a string: a list of phases and their durations.
    try:
        record.load_timeline = list(getattr(backend, "load_timeline", None) or [])
        record.load_seconds = float(getattr(backend, "load_seconds", 0.0) or 0.0)
    except Exception:
        logger.debug("could not stamp the load timeline", exc_info=True)


def _instance_is_starting(inst) -> bool:
    """True while an instance's engine process is alive but not yet answering.

    Two ways to ask, because backends differ: ``is_running`` where it exists,
    and the ``process_alive`` field of a health check otherwise.
    """
    backend = getattr(inst, "backend", None)
    if backend is None:
        return False
    if str(getattr(backend, "load_phase", "") or "") == "failed":
        return False
    probe = getattr(backend, "is_running", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    try:
        return bool((backend.health_check() or {}).get("process_alive"))
    except Exception:
        return False


async def _live_instance_records(manager, loop) -> list:
    """Probe every managed instance; return the records whose engine answers.

    Also flips each live record's status to ``serving`` (F3). The record is
    stamped ``starting`` at load time and was never updated once the engine
    came up, so the announcement advertised a phantom ``starting`` forever and
    the dashboard kept painting a "STARTING · 8%" progress bar for an instance
    that was actually serving traffic. The localhost /v1/models probe (via
    ``_engine_serving``) is the truthful liveness signal, so a passing probe is
    exactly when the status should read ``serving``.
    """
    live = []
    for inst in manager.instances():
        _stamp_load_state(inst)
        if await _engine_serving(inst.backend, loop):
            if inst.record.status != "serving":
                inst.record.status = "serving"
            live.append(inst.record)
        elif _instance_is_starting(inst):
            # Still coming up: advertise it as such. Dropping it made a model
            # that takes twenty minutes to load invisible from the head for
            # those twenty minutes — on a node that was working exactly as
            # asked, which reads as "nothing happened when I clicked load".
            # Only an instance whose process is gone stops being advertised.
            if inst.record.status not in ("starting", "failed"):
                inst.record.status = "starting"
            live.append(inst.record)
        else:
            # Not answering and not coming up: it crashed, was killed out of
            # band, or its launch died. Say so — and keep advertising it.
            #
            # Dropping it was worse than the phantom READY it was written to
            # prevent: an instance the operator had loaded simply VANISHED from
            # the dashboard, with no card, no error and no way to unload it.
            # Seen on the cluster when a model died during CUDA-graph capture.
            # A truthful `failed` gives the UI something to render and a button
            # to press. If the engine recovers, the next cycle flips it back.
            if inst.record.status != "failed":
                inst.record.status = "failed"
            live.append(inst.record)
    return live


async def _cluster_sync_loop(app: web.Application) -> None:
    """Periodically sync the listener registry into ClusterState."""
    try:
        while True:
            await asyncio.sleep(5)
            # What each launch actually cost, written down by the node that
            # ran it. Here rather than in the telemetry loop: a measurement
            # that only existed when MQTT was configured would be missing
            # from exactly the deployments that most need it.
            recorder = app.get("measurement_recorder")
            if recorder is None:
                from ainode.measure.recorder import Recorder

                recorder = Recorder(app)
                app["measurement_recorder"] = recorder
            recorder.poll()

            listener: Optional[BroadcastListener] = app.get("broadcast_listener")
            cluster: ClusterState = app["cluster_state"]
            if listener:
                cluster.update_from_discovered(listener.registry)
                # Update sender announcement with current engine status + master flag
                sender: Optional[BroadcastSender] = app.get("broadcast_sender")
                engine = app.get("engine")
                config: NodeConfig = app["config"]
                is_master = cluster.is_master_of_cluster()
                updates: dict = {
                    "is_master": is_master,
                    "cluster_id": getattr(config, "cluster_id", "default"),
                    "role": getattr(config, "cluster_role", "auto"),
                    "distributed_mode": getattr(config, "distributed_mode", "solo") or "solo",
                }
                dmode = updates["distributed_mode"]
                loop = asyncio.get_event_loop()
                # Liveness: the latched `ready` flag never flips False when an engine
                # crashes or is killed out-of-band, so a dead engine reads READY forever
                # (phantom-READY → ghost routing → 502s, BUG A FIX 2). Probe the engine's
                # own API instead (see _engine_serving) — also reads False while loading,
                # so we never advertise a not-yet-serving OR already-dead engine.
                # ponytail: one localhost probe per instance per 5s cycle; a transient
                # blip drops the model for one cycle and self-heals on the next probe.
                engine_serving = await _engine_serving(engine, loop)
                engine_proc_alive = bool(engine is not None and engine.is_running())
                # Re-broadcast the live primary model every cycle, gated on real
                # liveness — fixes both the stale `model` field (BUG A) and the
                # phantom-READY-after-crash case (FIX 2). Members serve via the head's
                # sharded engine, not their own model.
                # A member used to be blanked unconditionally — "members serve
                # via the head's sharded engine, not their own model". That is
                # true of a member participating in a distributed launch and
                # false of the deployment people actually build: one model per
                # node, each serving on its own. The liveness probe already
                # covers the case the blanking was for, since a member that
                # runs no engine of its own does not answer.
                updates["model"] = "" if not engine_serving else (config.model or "")
                if dmode == "member":
                    updates["status"] = "serving" if engine_serving else "member-ready"
                elif engine is not None:
                    updates["status"] = (
                        "serving" if engine_serving
                        else ("starting" if engine_proc_alive else "stopped")
                    )

                # Advertise distributed instance metadata once the head's
                # sharded engine is serving — the UI uses this to render
                # "DISTRIBUTED TP=N across X nodes".
                if dmode == "head" and engine_serving:
                    peer_ips = list(getattr(config, "peer_ips", []) or [])
                    if peer_ips:
                        updates["distributed_instance_id"] = f"{config.node_id or 'head'}:{config.model}"
                        updates["distributed_peers"] = peer_ips
                    else:
                        updates["distributed_instance_id"] = None
                        updates["distributed_peers"] = []
                elif dmode != "head":
                    updates["distributed_instance_id"] = None
                    updates["distributed_peers"] = []
                manager = app.get("instances")
                if manager is not None and not manager.is_empty():
                    # Only advertise instances whose engine actually answers — a dead
                    # stacked instance drops out of the broadcast within one cycle —
                    # and flip each live record's status to `serving` so the UI stops
                    # showing a phantom `starting` progress bar (F3).
                    live_records = await _live_instance_records(manager, loop)
                    updates["instances"] = [r.to_dict() for r in live_records]
                else:
                    updates["instances"] = (
                        _head_instances(config) if (dmode == "head" and engine_serving) else []
                    )
                # Embedding models are in-process and appear in no instance
                # record, so without this the head cannot see that the RAG
                # model is running on another node — cannot list it, cannot
                # route to it, and cannot capture it into a profile.
                updates["embedding_models"] = _local_embedding_models(app)
                if sender:
                    sender.update_announcement(**updates)
                    # Keep the app-level announcement in sync so /api/status sees fresh values
                    for k, v in updates.items():
                        if hasattr(sender.announcement, k):
                            setattr(sender.announcement, k, v)
    except asyncio.CancelledError:
        pass


async def _on_cleanup(app: web.Application) -> None:
    guard = app.get("memory_guard")
    if guard is not None:
        try:
            guard.stop()
        except Exception:
            logger.debug("could not stop the memory guard", exc_info=True)

    publisher = app.get("mqtt_publisher")
    if publisher is not None:
        try:
            await publisher.stop()
        except Exception:
            logger.exception("could not stop MQTT telemetry")

    # Stop the instance-replay task if still running
    replay_task = app.get("_instance_replay_task")
    if replay_task:
        replay_task.cancel()
        try:
            await replay_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # Stop cluster sync task
    sync_task = app.get("_cluster_sync_task")
    if sync_task:
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass

    # Stop Ray autostart task
    ray_task = app.get("_ray_autostart_task")
    if ray_task:
        ray_task.cancel()
        try:
            await ray_task
        except asyncio.CancelledError:
            pass

    # Stop discovery sender and listener
    sender: Optional[BroadcastSender] = app.get("broadcast_sender")
    if sender:
        await sender.stop()
        logger.info("Discovery sender stopped")

    listener: Optional[BroadcastListener] = app.get("broadcast_listener")
    if listener:
        await listener.stop()
        logger.info("Discovery listener stopped")

    session: Optional[aiohttp.ClientSession] = app.get("client_session")
    if session and not session.closed:
        await session.close()

@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers to every response so the dashboard can fetch freely."""
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as exc:
            resp = exc

    origin = request.headers.get("Origin", "")
    allowed = origin if origin.startswith(("http://localhost", "http://127.0.0.1")) else ""
    resp.headers["Access-Control-Allow-Origin"] = allowed
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

async def handle_index(request: web.Request) -> web.Response:
    """Serve the dashboard, or redirect to onboarding if not set up."""
    config: NodeConfig = request.app["config"]
    if not config.onboarded:
        raise web.HTTPFound("/onboarding")
    html = get_index_html()
    return web.Response(text=html, content_type="text/html")

async def handle_onboarding(request: web.Request) -> web.Response:
    """Serve the onboarding wizard. Redirect to dashboard if already onboarded."""
    config: NodeConfig = request.app["config"]
    if config.onboarded:
        raise web.HTTPFound("/")
    html = get_onboarding_html()
    return web.Response(text=html, content_type="text/html")

async def handle_health(_request: web.Request) -> web.Response:
    """Simple liveness probe."""
    return web.json_response({"status": "ok"})

async def handle_status(request: web.Request) -> web.Response:
    """Return rich node status."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    start_time: float = request.app["start_time"]
    session: Optional[aiohttp.ClientSession] = request.app.get("client_session")

    gpu: Optional[GPUInfo] = detect_gpu()
    gpu_info = asdict(gpu) if gpu else None
    # detect_gpu() caches and reports free==total on GB10 unified memory, so the
    # dashboard showed 0% used. Overlay the collector's live psutil reading — the
    # same source /api/nodes uses — so the two endpoints agree.
    if gpu_info is not None:
        collector = request.app.get("metrics_collector")
        if collector is not None:
            try:
                m = collector.get_gpu_metrics() or {}
                if not m.get("error"):
                    total_mb = m.get("memory_total_mb") or gpu_info.get("memory_total_mb")
                    used_mb = m.get("memory_used_mb")
                    if total_mb:
                        gpu_info["memory_total_mb"] = round(total_mb)
                        if used_mb is not None:
                            gpu_info["memory_free_mb"] = max(0, round(total_mb - used_mb))
            except Exception:
                pass

    engine_ready = False
    models_loaded: list[str] = []

    # Live-probe wins: a vLLM that answers /v1/models with >=1 model right now
    # is the single source of liveness. The latched engine.ready is no longer
    # trusted for status (wait_ready still uses it internally).
    if session is not None:
        try:
            vllm_url = f"http://localhost:{config.api_port}/v1/models"
            async with session.get(vllm_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models_loaded = [m.get("id", "") for m in data.get("data", [])]
                    engine_ready = len(models_loaded) > 0
        except Exception:
            engine_ready = False
    elif engine is not None:
        # No HTTP session — fall back to the managed engine's health check.
        try:
            hc = engine.health_check()
            models_loaded = hc.get("models_loaded", [])
            engine_ready = bool(hc.get("api_responding")) and len(models_loaded) > 0
        except Exception:
            engine_ready = False

    cluster: ClusterState = request.app["cluster_state"]
    master = cluster.get_master()
    effective_role = cluster.get_cluster_role_for(config.node_id) if config.node_id else "worker"

    return web.json_response({
        "node_id": config.node_id,
        "node_name": config.node_name,
        "model": config.model,
        "gpu": gpu_info,
        "engine_ready": engine_ready,
        # Coarse engine load phase for the UI launching card (3c):
        # idle | starting | loading_weights | distributed_init | profiling | ready
        # Truthful phase: the live /v1/models probe above is the reliable
        # readiness signal — the engine's _ready latch can miss the vLLM startup
        # log marker and stay False on a model that is actually serving. Report
        # 'ready' when the probe says serving; else the engine's coarse phase.
        "load_phase": ("ready" if engine_ready else (getattr(engine, "load_phase", "idle") if engine is not None else "idle")),
        # Kept after readiness too: "why did that take five minutes" is asked
        # once the model is up, not while waiting.
        "load_timeline": (list(getattr(engine, "load_timeline", None) or [])
                          if engine is not None else []),
        "load_seconds": (float(getattr(engine, "load_seconds", 0.0) or 0.0)
                         if engine is not None else 0.0),
        # Why a launch died, quoting the engine's own last lines. Empty unless
        # the phase is "failed" — without it the UI can say a launch failed but
        # not why, and the operator is sent to hunt through a log file.
        # What a silent pre-launch step is doing (pulling an engine image,
        # copying weights). Empty once the launcher speaks for itself.
        "load_detail": ("" if engine_ready else (getattr(engine, "load_detail", "") if engine is not None else "")),
        "load_error": ("" if engine_ready else (getattr(engine, "load_error", "") if engine is not None else "")),
        "uptime": round(time.time() - start_time, 1),
        "version": __version__,
        "powered_by": "argentos.ai",
        "models_loaded": models_loaded,
        "api_port": config.api_port,
        "cluster_role": effective_role,
        "cluster_id": getattr(config, "cluster_id", "default"),
        "master_node_id": master.node_id if master else None,
    })

async def handle_nodes(request: web.Request) -> web.Response:
    """Return the list of known cluster nodes."""
    config: NodeConfig = request.app["config"]
    engine = request.app["engine"]
    cluster: ClusterState = request.app["cluster_state"]

    cluster_nodes = cluster.get_nodes(include_offline=False)
    collector = request.app.get("metrics_collector")
    local_id = config.node_id
    if cluster_nodes:
        nodes_list = []
        for n in cluster_nodes:
            status_str = n.status.value if hasattr(n.status, "value") else str(n.status)
            dmode = getattr(n, "distributed_mode", "solo") or "solo"
            # Members are "ready for work" once discovered, even though they
            # don't run a local vLLM.
            ready = status_str in ("online", "serving") or (
                dmode == "member" and status_str in ("online", "member-ready")
            )
            # Live GPU telemetry: peers come from their broadcast; the local
            # node's ClusterNode is built once at startup, so read it fresh
            # from our own collector here. (metrics fan-out)
            used_mb = float(getattr(n, "gpu_memory_used_mb", 0.0) or 0.0)
            total_mb = float(getattr(n, "gpu_memory_total_mb", 0.0) or 0.0)
            util = float(getattr(n, "gpu_utilization", 0.0) or 0.0)
            temp = float(getattr(n, "gpu_temp", 0.0) or 0.0)
            if n.node_id == local_id and collector is not None:
                try:
                    m = collector.get_gpu_metrics() or {}
                    if not m.get("error"):
                        used_mb = float(m.get("memory_used_mb", used_mb) or used_mb)
                        total_mb = float(m.get("memory_total_mb", total_mb) or total_mb)
                        util = float(m.get("utilization_percent", util) or util)
                        temp = float(m.get("temperature_c", temp) or temp)
                except Exception:
                    pass
            if not total_mb and n.gpu_memory_gb:
                total_mb = n.gpu_memory_gb * 1024
            used_pct = round(used_mb / total_mb * 100) if total_mb else 0
            nodes_list.append({
                "node_id": n.node_id,
                "node_name": n.node_name,
                "fabric_ip": getattr(n, "fabric_ip", "") or "",
                "host": "localhost",
                "api_port": n.api_port,
                "web_port": n.web_port,
                "model": n.model,
                "gpu_name": n.gpu_name,
                "gpu_memory_gb": n.gpu_memory_gb,
                "unified_memory": n.unified_memory,
                "gpu_memory_used_pct": used_pct,
                "gpu_utilization": round(util),
                "gpu_temp": round(temp),
                "status": status_str,
                "engine_ready": ready,
                "distributed_mode": dmode,
                "distributed_instance_id": getattr(n, "distributed_instance_id", None),
                "distributed_peers": list(getattr(n, "distributed_peers", []) or []),
                # Per-node instance list (primary + any stacked models on ports
                # 8001+). The node card renders a sub-row per stacked instance so
                # they're no longer invisible in the dashboard. Same source the
                # proxy's _routing_candidates uses, so views and routing agree.
                # Carry the per-instance load state too. The member computes
                # it correctly (LoadPhaseTracker reads the engine's own
                # output), serialises it onto the record, and broadcasts it —
                # and this projection dropped it, keeping only three keys. So
                # a node genuinely sitting in `loading_weights` reached the
                # dashboard with no phase at all, and the card fell back to
                # the status word: "STARTING · 12%" for the whole of a load,
                # which is exactly the hang-or-working question the phase was
                # added to answer.
                "instances": [
                    {"model": inst.get("model"),
                     "api_port": inst.get("api_port"),
                     "status": inst.get("status"),
                     "load_phase": inst.get("load_phase") or "",
                     "load_detail": inst.get("load_detail") or "",
                     "load_error": inst.get("load_error") or "",
                     "load_timeline": inst.get("load_timeline") or [],
                     "load_seconds": inst.get("load_seconds") or 0}
                    for inst in (getattr(n, "instances", []) or [])
                    if isinstance(inst, dict) and inst.get("model")
                ],
                # In-process, so they are in no instance record — and the
                # cluster graphic reads this endpoint, so without them a node
                # serving embeddings looked idle on hover.
                "embedding_models": [
                    str(e) for e in (getattr(n, "embedding_models", []) or []) if e
                ],
            })
    else:
        # Fallback: return this node
        engine_ready = False
        if engine is not None:
            engine_ready = getattr(engine, "ready", False)
        dmode = getattr(config, "distributed_mode", "solo") or "solo"
        nodes_list = [{
            "node_id": config.node_id,
            "node_name": config.node_name,
            "host": config.host,
            "api_port": config.api_port,
            "web_port": config.web_port,
            "model": config.model,
            "engine_ready": engine_ready or dmode == "member",
            "distributed_mode": dmode,
        }]
    return web.json_response({"nodes": nodes_list})

def _placed_node(app, model: str) -> str:
    """The single node this model is pinned to, or "". Never raises: a pin is
    a convenience, and a broken placement file must not stop a load."""
    if not model:
        return ""
    try:
        from ainode.placement.api_routes import get_placement_store

        placement = get_placement_store(app).get(model)
    except Exception:
        logger.debug("could not read the placement for %s", model, exc_info=True)
        return ""
    if placement is not None and len(placement.node_ids) == 1:
        return placement.node_ids[0]
    return ""


async def _cluster_dispatch(request: web.Request, path: str):
    """F2: forward a load/unload to a node's local /api/models endpoint.

    node_id == this node → call the local handler directly (back-compat). Remote →
    POST over the fabric to http://<fabric_ip>:<web_port><path>. Reuses each node's
    existing /api/models/load|unload; the master never SSHes.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    node_id = str_field(body, "node_id")
    if not node_id and path.endswith("/load"):
        # No node named: fall back to where this model was pinned. Only a
        # single-node placement applies here — this route loads one model on
        # one node, and honouring a two-node pin by taking its first node
        # would load a sharded model whole and OOM the node. Multi-node pins
        # are read by /api/sharding/launch instead.
        placed = _placed_node(request.app, str_field(body, "model"))
        if placed:
            node_id = placed
            body = {**body, "node_id": placed}
    if not node_id or node_id == config.node_id:
        # Local: hand the body to the local model handler unchanged.
        from ainode.models.api_routes import handle_model_load, handle_model_unload

        class _Shim:
            def __init__(self, orig, b):
                self._o, self._b = orig, b
            def __getattr__(self, k):
                return getattr(self._o, k)
            async def json(self):
                return self._b
        if path.endswith("/compile-cache"):
            handler = _clear_compile_cache
        elif path.endswith("/load"):
            handler = handle_model_load
        elif path.endswith("/delete-repo"):
            from ainode.models.api_routes import handle_delete_repo

            handler = handle_delete_repo
        else:
            handler = handle_model_unload
        return await handler(_Shim(request, body))

    node = cluster.get_node(node_id) if cluster is not None else None
    host = (getattr(node, "fabric_ip", "") or "") if node else ""
    if not host:
        return web.json_response(
            {"error": f"node '{node_id}' not found or has no fabric IP"}, status=404)
    url = f"http://{host}:{node.web_port}{path}"
    session: aiohttp.ClientSession = request.app["client_session"]
    fwd = {k: v for k, v in body.items() if k != "node_id"}
    try:
        async with session.post(url, json=fwd, timeout=aiohttp.ClientTimeout(total=60)) as up:
            data = await up.read()
            # web.Response rejects a content_type carrying a charset; the node's
            # json_response sends "application/json; charset=utf-8" — strip it.
            ctype = up.headers.get("Content-Type", "application/json").split(";")[0].strip()
            return web.Response(status=up.status, body=data, content_type=ctype)
    except aiohttp.ClientError as exc:
        return web.json_response(
            {"error": f"failed to reach node '{node_id}' at {url}: {exc}"}, status=502)


async def handle_cluster_load(request: web.Request) -> web.Response:
    """POST /api/cluster/load {node_id, model} — load a model on any node (F2)."""
    return await _cluster_dispatch(request, "/api/models/load")


class _MatchInfoShim:
    """A request whose match_info names one model — the embedding routes read
    the model from the URL, and a cluster dispatch carries it in the body."""

    def __init__(self, original, model_id: str):
        self._original = original
        self.match_info = {"model_id": model_id}

    def __getattr__(self, name):
        return getattr(self._original, name)


async def _embedding_dispatch(request: web.Request, action: str) -> web.Response:
    """POST /api/cluster/embeddings/{load,unload} {node_id, model}.

    Embedding models are in-process, so they load on whichever node's API is
    asked — which meant the only way to place one on node 3 was to open node
    3's own UI. The placement is a deployment decision; it belongs in the same
    dialog as everything else.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    model = str_field(body, "model", "model_id")
    if not model:
        return web.json_response({"error": "model required"}, status=400)
    node_id = str_field(body, "node_id")

    if not node_id or node_id == config.node_id:
        from ainode.embeddings.api_routes import (
            handle_load_embedding_model,
            handle_unload_embedding_model,
        )

        handler = (handle_load_embedding_model if action == "load"
                   else handle_unload_embedding_model)
        return await handler(_MatchInfoShim(request, model))

    node = cluster.get_node(node_id) if cluster is not None else None
    host = (getattr(node, "fabric_ip", "") or "") if node else ""
    if not host:
        return web.json_response(
            {"error": f"node '{node_id}' not found or has no fabric IP"}, status=404)
    url = (f"http://{host}:{node.web_port}/api/embeddings/models/"
           f"{quote(model, safe='')}/{action}")
    session: aiohttp.ClientSession = request.app["client_session"]
    try:
        async with session.post(url, timeout=aiohttp.ClientTimeout(total=600)) as upstream:
            data = await upstream.read()
            ctype = upstream.headers.get(
                "Content-Type", "application/json").split(";")[0].strip()
            return web.Response(status=upstream.status, body=data, content_type=ctype)
    except aiohttp.ClientError as exc:
        return web.json_response(
            {"error": f"failed to reach node '{node_id}' at {url}: {exc}"}, status=502)


async def handle_opencode_config(request: web.Request) -> web.Response:
    """GET /api/clients/opencode — a ready-to-paste OpenCode provider config.

    Assembled from what is RUNNING, because the three settings that matter are
    all per-instance: whether the model reasons, whether it takes images, and
    the context window it was launched with. Getting any of them from the
    model's own advertised figures produces a config that works until it
    quietly does not.
    """
    from ainode.clients.opencode import build_opencode_config

    config: NodeConfig = request.app["config"]
    base = str_field(await _json_body(request), "base_url")
    if not base:
        host = getattr(config, "fabric_ip", "") or "127.0.0.1"
        base = f"http://{host}:{getattr(config, 'web_port', 3000)}"
    try:
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None, build_opencode_config, request.app, base)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("could not build the opencode config")
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(payload)


async def _json_body(request: web.Request) -> dict:
    if request.method != "POST":
        return {"base_url": request.query.get("base_url", "")}
    try:
        return await request.json()
    except Exception:
        return {}


async def handle_launch_config(request: web.Request) -> web.Response:
    """GET /api/instances/launch-config — how this node's models were started.

    A profile has to carry the launch parameters, or applying it restores the
    models without the flags they need and the operator finds out at the first
    request. Those parameters live in each node's own InstanceManager and
    appear in no announcement — a broadcast carrying every model's argument
    list would be a different thing entirely.

    So the head asks. One request per peer, on a deliberate action a person
    performs rarely.
    """
    from ainode.profiles.apply import local_launch_specs

    try:
        specs = local_launch_specs(request.app)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("could not read the local launch config")
        return web.json_response({"error": str(exc), "instances": []}, status=500)
    return web.json_response({
        "node_id": getattr(request.app["config"], "node_id", "") or "",
        "instances": specs,
    })


async def handle_cluster_memory_get(request: web.Request) -> web.Response:
    """GET /api/cluster/safety/memory — every node's reserve and reading.

    The whole fleet in one answer, because the reserve is a per-node setting
    and the nodes that ran out are not the one whose UI is open. A settings
    page that could only show this node's figure would be the least useful
    place to look after a node has just been lost.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    session: aiohttp.ClientSession = request.app.get("client_session")

    async def _local() -> dict:
        from ainode.safety.api_routes import handle_get

        resp = await handle_get(request)
        row = json.loads(resp.body)
        row.update({"node_id": config.node_id or "local",
                    "node_name": getattr(config, "node_name", "") or "",
                    "reachable": True})
        return row

    async def _peer(node) -> dict:
        row = {"node_id": node.node_id,
               "node_name": getattr(node, "node_name", "") or node.node_id,
               "reachable": False, "available": False}
        host = (getattr(node, "fabric_ip", "") or "").strip()
        if not host or session is None:
            return row
        url = f"http://{host}:{getattr(node, 'web_port', 3000)}/api/safety/memory"
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    row.update(await resp.json(content_type=None))
                    row["reachable"] = True
        except Exception:
            logger.debug("could not ask %s about its memory reserve",
                         node.node_id, exc_info=True)
        return row

    tasks = [_local()]
    for node in (cluster.members() if cluster is not None else []):
        if node.node_id == config.node_id:
            continue
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        tasks.append(_peer(node))

    rows = [r for r in await asyncio.gather(*tasks, return_exceptions=True)
            if not isinstance(r, BaseException)]
    return web.json_response({"nodes": rows})


async def handle_cluster_memory_put(request: web.Request) -> web.Response:
    """PUT /api/cluster/safety/memory {node_id|all, warn_gb, critical_gb, …}.

    ``all: true`` sets every reachable node at once — one reserve policy for
    the fleet is what an operator actually wants, and setting three nodes by
    opening three UIs is how the setting ends up inconsistent.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    session: aiohttp.ClientSession = request.app.get("client_session")
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    settings = {k: v for k, v in body.items() if k not in ("node_id", "all")}
    targets = []
    if body.get("all"):
        targets = [n for n in (cluster.members() if cluster is not None else [])
                   if n.node_id != config.node_id]
    node_id = str_field(body, "node_id")
    local = bool(body.get("all")) or not node_id or node_id == config.node_id
    if node_id and not body.get("all"):
        targets = [n for n in (cluster.members() if cluster is not None else [])
                   if n.node_id == node_id]

    results = []
    if local:
        from ainode.safety.api_routes import handle_put

        class _Shim:
            def __init__(self, orig, payload):
                self._o, self._b = orig, payload

            def __getattr__(self, name):
                return getattr(self._o, name)

            async def json(self):
                return self._b

        resp = await handle_put(_Shim(request, settings))
        results.append({"node_id": config.node_id or "local",
                        "ok": resp.status == 200,
                        "result": json.loads(resp.body)})

    async def _peer(node):
        host = (getattr(node, "fabric_ip", "") or "").strip()
        if not host or session is None:
            return {"node_id": node.node_id, "ok": False,
                    "error": "no fabric IP"}
        url = f"http://{host}:{getattr(node, 'web_port', 3000)}/api/safety/memory"
        try:
            async with session.put(
                    url, json=settings,
                    timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return {"node_id": node.node_id, "ok": resp.status == 200,
                        "result": await resp.json(content_type=None)}
        except Exception as exc:
            return {"node_id": node.node_id, "ok": False, "error": str(exc)}

    if targets:
        results.extend([r for r in await asyncio.gather(
            *[_peer(n) for n in targets], return_exceptions=True)
            if not isinstance(r, BaseException)])
    return web.json_response({"results": results})


async def handle_cluster_delete_repo(request: web.Request) -> web.Response:
    """POST /api/cluster/delete-repo {hf_repo, node_id?} — delete on any node.

    The Models page now lists what the whole cluster holds, so its Delete
    button has to be able to reach the disk the weights are actually on.
    Without a node it deletes here, which is what it always did.
    """
    return await _cluster_dispatch(request, "/api/models/delete-repo")


async def handle_cluster_models(request: web.Request) -> web.Response:
    """GET /api/cluster/models — which models are on which node's disk.

    The Models page showed a scan of THIS node's disk, while the instance
    panel beside it showed what the whole cluster was serving. A model
    downloaded to node 3 and running there appeared in one and not the other,
    which reads as the page having lost it.

    One request per peer, in parallel, each with a short timeout: a node that
    does not answer costs its own row, not the page.
    """
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    session: aiohttp.ClientSession = request.app.get("client_session")

    async def _local() -> tuple:
        manager = request.app.get("model_manager")
        if manager is None:
            return config.node_id or "local", []
        loop = asyncio.get_event_loop()
        try:
            models = await loop.run_in_executor(None, manager.list_downloaded)
        except Exception:
            logger.exception("could not list this node's models")
            models = []
        return config.node_id or "local", models

    async def _peer(node) -> tuple:
        host = (getattr(node, "fabric_ip", "") or "").strip()
        if not host or session is None:
            return node.node_id, []
        url = f"http://{host}:{getattr(node, 'web_port', 3000)}/api/models/downloaded"
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return node.node_id, []
                data = await resp.json(content_type=None)
        except Exception:
            logger.debug("could not ask %s for its models", node.node_id,
                         exc_info=True)
            return node.node_id, []
        return node.node_id, (data or {}).get("models") or []

    tasks = [_local()]
    names = {config.node_id or "local": getattr(config, "node_name", "") or ""}
    for node in (cluster.members() if cluster is not None else []):
        if node.node_id == config.node_id:
            continue
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        names[node.node_id] = getattr(node, "node_name", "") or node.node_id
        tasks.append(_peer(node))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    by_repo: dict = {}
    for result in results:
        if isinstance(result, BaseException):
            continue
        node_id, models = result
        for entry in models:
            repo = str(entry.get("hf_repo") or entry.get("id") or "")
            if not repo:
                continue
            row = by_repo.setdefault(repo, {
                "hf_repo": repo,
                "name": entry.get("name") or repo,
                "size_gb": entry.get("size_gb") or entry.get("local_size_gb") or 0,
                "nodes": [],
            })
            if node_id not in row["nodes"]:
                row["nodes"].append(node_id)
            # The largest copy wins: a partial mirror on one node should not
            # make the model look smaller than it is.
            size = entry.get("local_size_gb") or entry.get("size_gb") or 0
            if size and size > (row["size_gb"] or 0):
                row["size_gb"] = size

    return web.json_response({
        "models": sorted(by_repo.values(), key=lambda r: r["hf_repo"].lower()),
        "node_names": names,
    })


async def handle_cluster_mirror_models(request: web.Request) -> web.Response:
    """POST /api/cluster/mirror-models {models?} — push this node's models out.

    Downloads mirror themselves from now on, but everything already on the
    head predates that and would otherwise stay put until someone launched it
    somewhere — which is the slow path this exists to remove.

    Explicit, never automatic: this can be several hundred gigabytes, and an
    unannounced transfer of that size on a node someone is using is not a
    favour. Runs in the background and reports per model and per node, because
    a caller cannot hold a connection open for an hour.
    """
    jobs: dict = request.app.setdefault("mirror_jobs", {})
    if jobs.get("running"):
        return web.json_response({"error": "a mirror run is already in progress",
                                  "status": jobs}, status=409)
    try:
        body = await request.json()
    except Exception:
        body = {}
    wanted = [str(m) for m in (body.get("models") or []) if m]

    manager = request.app.get("model_manager")
    if manager is None:
        return web.json_response({"error": "no model manager"}, status=503)
    try:
        downloaded = [str(entry.get("hf_repo") or "")
                      for entry in manager.list_downloaded()]
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    models = [m for m in downloaded if m and (not wanted or m in wanted)]
    if not models:
        return web.json_response({"error": "nothing downloaded here to mirror"},
                                 status=404)

    jobs.clear()
    jobs.update({"running": True, "started_at": time.time(), "models": {},
                 "total": len(models), "done": 0})

    async def _run() -> None:
        from ainode.engine.mirror import ensure_dependencies, mirror_model_to_peers

        loop = asyncio.get_event_loop()
        for model in models:
            jobs["models"][model] = {"state": "copying"}
            try:
                # What the recipe pulls in counts as part of the model. Fetch
                # it here first if it is missing, so the sweep leaves every
                # node able to launch rather than merely holding a checkpoint.
                extras = await loop.run_in_executor(
                    None, lambda m=model: ensure_dependencies(request.app, m))
                nodes: dict = {}
                for repo in [model, *extras]:
                    nodes.update(await loop.run_in_executor(
                        None, lambda r=repo: mirror_model_to_peers(request.app, r)))
                entry = {"state": "done", "nodes": nodes}
                if extras:
                    entry["requires"] = extras
                jobs["models"][model] = entry
            except Exception as exc:  # pragma: no cover - mirror never raises
                jobs["models"][model] = {"state": "failed", "error": str(exc)}
            jobs["done"] += 1
        jobs["running"] = False
        jobs["finished_at"] = time.time()

    asyncio.get_event_loop().create_task(_run())
    return web.json_response({"started": True, "models": models}, status=202)


async def handle_cluster_mirror_status(request: web.Request) -> web.Response:
    """GET /api/cluster/mirror-status — how the mirror run is going."""
    return web.json_response(request.app.get("mirror_jobs") or {"running": False})


async def handle_cluster_embedding_load(request: web.Request) -> web.Response:
    return await _embedding_dispatch(request, "load")


async def handle_cluster_embedding_unload(request: web.Request) -> web.Response:
    return await _embedding_dispatch(request, "unload")


async def _clear_compile_cache(request: web.Request) -> web.Response:
    from ainode.models.api_routes import handle_clear_compile_cache

    return await handle_clear_compile_cache(request)


async def handle_cluster_compile_cache(request: web.Request) -> web.Response:
    """POST /api/cluster/compile-cache {node_id} — clear it on any node."""
    return await _cluster_dispatch(request, "/api/engine/compile-cache")


async def handle_cluster_unload(request: web.Request) -> web.Response:
    """POST /api/cluster/unload {node_id, model} — unload on any node (F2)."""
    return await _cluster_dispatch(request, "/api/models/unload")


def _instance_can_answer(inst: dict) -> bool:
    """True for an instance a request can actually be sent to.

    The dashboard wants every instance with its true state, including the ones
    still loading — that is what makes a twenty-minute launch visible. A client
    reading /v1/models wants the opposite: a model listed there is one it may
    call, and listing a loading model put it in OpenWebUI's dropdown minutes
    before it could answer. Same data, two audiences.

    A record from an older build carries no status; treating that as serving
    keeps those nodes routable, which is how they behaved before.
    """
    status = str(inst.get("status") or "serving")
    return status in ("serving", "member-ready")


def _routing_candidates(cluster, model: str, local_node_id: str, local_port: int) -> list:
    """All (host, port) currently serving `model` (routing-truth).

    Returns a LIST so proxy_to_vllm can fail over when the first target is a
    stale/ghost claim (a node that crashed but still advertises the model). A
    crashed node is indistinguishable from a live one in cluster state, so
    failover — not ordering — is what makes routing robust. Local node first
    (cheapest hop), then remote peers.
    """
    local, remote = [], []
    for n in (cluster.members() if cluster is not None else []):
        status = n.status.value if hasattr(n.status, "value") else str(n.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        is_local = n.node_id == local_node_id
        host = "localhost" if is_local else (getattr(n, "fabric_ip", "") or "")
        if not host:
            continue
        node_port = local_port if is_local else n.api_port
        bucket = local if is_local else remote
        seen = set()
        # The node's primary/solo model is served on its main api_port.
        if getattr(n, "model", "") == model and node_port not in seen:
            bucket.append((host, node_port))
            seen.add(node_port)
        # Each stacked instance is served on its OWN port — a co-resident 2nd
        # model on this node lives at :8001, not the node's main :8000.
        for inst in (getattr(n, "instances", []) or []):
            if inst.get("model") != model or not _instance_can_answer(inst):
                continue
            iport = inst.get("api_port") or node_port
            if iport not in seen:
                bucket.append((host, iport))
                seen.add(iport)
    return local + remote


def _routing_table(cluster, local_node_id: str, local_port: int) -> dict:
    """model name → (host, port) for every model served across the fleet (F1).

    Built from cluster broadcast state: each node advertises its solo model and
    any instances it heads. The local node routes to localhost; remote nodes to
    their fabric IP (reachable from the master over the cluster fabric).
    """
    table: dict = {}
    for n in cluster.members():
        status = n.status.value if hasattr(n.status, "value") else str(n.status)
        if status not in ("online", "serving", "member-ready"):
            continue
        is_local = n.node_id == local_node_id
        host = "localhost" if is_local else (getattr(n, "fabric_ip", "") or "")
        if not host:
            continue
        port = local_port if is_local else n.api_port
        if getattr(n, "model", ""):
            table.setdefault(n.model, (host, port))
        for inst in (getattr(n, "instances", []) or []):
            m = inst.get("model")
            if m and _instance_can_answer(inst):
                table.setdefault(m, (host, inst.get("api_port") or port))
    return table


def _local_embedding_models(app) -> list[str]:
    """Embedding models loaded on THIS node."""
    manager = app.get("embedding_manager")
    if manager is None:
        return []
    try:
        return sorted(
            str(entry.get("id")) for entry in manager.list_loaded()
            if entry.get("id")
        )
    except Exception:
        logger.debug("could not list loaded embedding models", exc_info=True)
        return []


async def handle_v1_models(request: web.Request) -> web.Response:
    """Federated /v1/models — the UNION of models served across the fleet (F1)."""
    config: NodeConfig = request.app["config"]
    cluster = request.app.get("cluster_state")
    table = _routing_table(cluster, config.node_id, config.api_port) if cluster is not None else {}
    if not table and config.model:
        table = {config.model: ("localhost", config.api_port)}
    data = [{"id": m, "object": "model", "owned_by": "ainode"} for m in sorted(table)]
    # Embedding models too. They are not vLLM instances, so they never entered
    # the routing table, and a RAG client that asks /v1/models before calling
    # /v1/embeddings concluded the model it had just loaded was unavailable.
    # OpenAI lists its embedding models here; so do we.
    #
    # The whole fleet's, not only this node's: peers advertise what they have
    # loaded and /v1/embeddings forwards to the node that has it, so a model
    # listed here is one this address can actually answer for — which is the
    # only promise the list makes.
    embeddings = set(_local_embedding_models(request.app))
    if cluster is not None:
        try:
            for node in cluster.get_nodes():
                embeddings.update(getattr(node, "embedding_models", []) or [])
        except Exception:
            logger.debug("could not read peers' embedding models", exc_info=True)
    for model_id in sorted(embeddings):
        if model_id not in table:
            data.append({"id": model_id, "object": "model", "owned_by": "ainode"})
    return web.json_response({"object": "list", "data": data})


def _completion_tokens(body: bytes) -> int:
    """Tokens generated, from an OpenAI-shaped response body. 0 if absent.

    The body has already been read here, so this is a parse of bytes in hand
    rather than extra work on the wire. Anything unexpected — an error
    response, an endpoint with no usage block — counts as nothing rather than
    raising inside the proxy.
    """
    import json as _json

    try:
        usage = _json.loads(body).get("usage") or {}
        return max(0, int(usage.get("completion_tokens") or 0))
    except Exception:
        return 0


async def proxy_to_vllm(request: web.Request) -> web.StreamResponse:
    """Forward the request to the node serving the requested model (F1 federation)."""
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]
    collector: MetricsCollector = request.app["metrics_collector"]
    # Extract the model name first — it drives BOTH routing and metrics.
    model = config.model or "unknown"
    body_bytes = None
    if request.method == "POST":
        body_bytes = await request.read()
        try:
            import json as _json
            model = _json.loads(body_bytes).get("model", model)
        except Exception:
            pass
    # Tag the request so the server-view log middleware can capture the model
    try:
        request["_log_model"] = model
    except Exception:
        pass

    # Federated routing with failover (F1 + routing-truth): try every node that
    # serves this model (ready ones first), so a stale/ghost claim from a crashed
    # node doesn't 502 a request another node can serve. Built from cluster state.
    cluster = request.app.get("cluster_state")
    candidates = _routing_candidates(cluster, model, config.node_id, config.api_port)
    if not candidates:
        if model and model != "unknown" and cluster is not None and cluster.members():
            return web.json_response(
                {"error": {"type": "model_not_found",
                           "message": f"Model '{model}' is not loaded on any node",
                           "code": "model_not_found"}},
                status=404)
        candidates = [("localhost", config.api_port)]  # back-compat: empty fleet → local

    # Build upstream request kwargs. Strip content-length: aiohttp recomputes it
    # from `data`, and forwarding the original alongside makes the upstream wait
    # for a body that never arrives (the proxy hangs). Strip host/transfer-encoding
    # for the usual reverse-proxy reasons.
    kwargs: dict = {
        "headers": {k: v for k, v in request.headers.items()
                    if k.lower() not in ("host", "transfer-encoding", "content-length")},
        # Fast failover: a dead/ghost node must fail the CONNECT quickly so the
        # loop moves on to the next candidate — but leave total uncapped so a live
        # node's slow cold-start generation (35s+) can still stream to completion.
        "timeout": aiohttp.ClientTimeout(total=None, sock_connect=5),
    }
    if body_bytes is not None:
        kwargs["data"] = body_bytes

    start_time = time.time()
    last_err = None
    for host, port in candidates:
        vllm_url = f"http://{host}:{port}{request.path}"
        try:
            async with session.request(request.method, vllm_url, **kwargs) as upstream:
                is_sse = "text/event-stream" in upstream.headers.get("Content-Type", "")
                if is_sse:
                    resp = web.StreamResponse(
                        status=upstream.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                    await resp.prepare(request)
                    streamed_tokens = 0
                    async for chunk in upstream.content.iter_any():
                        # vLLM emits one SSE event per token, so counting the
                        # event markers counts tokens without parsing JSON in
                        # the streaming hot path. A chunk boundary landing
                        # inside a marker can miscount by one; this feeds an
                        # average speed gauge, not a bill.
                        streamed_tokens += chunk.count(b"data: ")
                        await resp.write(chunk)
                    await resp.write_eof()
                    collector.record_request(
                        model, (time.time() - start_time) * 1000,
                        tokens_generated=streamed_tokens, error=False)
                    return resp
                body = await upstream.read()
                collector.record_request(
                    model, (time.time() - start_time) * 1000,
                    tokens_generated=_completion_tokens(body), error=False)
                return web.Response(
                    status=upstream.status, body=body,
                    content_type=upstream.headers.get("Content-Type", "application/json").split(";")[0].strip(),
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_err = exc  # unreachable / connect-timeout (likely a ghost) — try the next
            continue
    # Every candidate failed.
    collector.record_request(model, (time.time() - start_time) * 1000, error=True)
    return web.json_response(
        {"error": {"message": f"no reachable node is serving '{model}' ({last_err})",
                   "type": "server_error"}},
        status=502,
    )

async def handle_cluster_info(request: web.Request) -> web.Response:
    """Return the current cluster topology from this node's perspective."""
    config: NodeConfig = request.app["config"]
    cluster: ClusterState = request.app["cluster_state"]

    master = cluster.get_master()
    members = cluster.members()
    master_address: Optional[str] = getattr(config, "master_address", None)
    if master and not master_address and master.node_id != config.node_id:
        master_address = f"{master.node_name}:{master.web_port}"

    return web.json_response({
        "my_role": cluster.get_cluster_role_for(config.node_id) if config.node_id else "worker",
        "my_node_id": config.node_id,
        "cluster_id": getattr(config, "cluster_id", "default"),
        "configured_role": getattr(config, "cluster_role", "auto"),
        "master_node_id": master.node_id if master else None,
        "master_address": master_address,
        "members": [
            {
                "node_id": m.node_id,
                "node_name": m.node_name,
                "api_port": m.api_port,
                "web_port": m.web_port,
                "role": m.role,
                "effective_role": "master" if (master and master.node_id == m.node_id) else "worker",
                "status": m.status.value if hasattr(m.status, "value") else str(m.status),
                "last_seen": m.last_seen,
                "gpu_name": m.gpu_name,
                "gpu_memory_gb": m.gpu_memory_gb,
            }
            for m in members
        ],
    })


async def handle_cluster_resources(request: web.Request) -> web.Response:
    """Return aggregated cluster resources (VRAM, GPUs) across ready nodes."""
    cluster: ClusterState = request.app["cluster_state"]
    ray_state: RayAutostartState = request.app.get("ray_autostart_state") or RayAutostartState()

    members = cluster.members()
    # Include member-ready too: those nodes have reserved their GPU for
    # Ray workers launched by the head, so they contribute to total VRAM
    # even though they don't run their own local vLLM.
    ready = [
        n for n in members
        if (n.status.value if hasattr(n.status, "value") else str(n.status))
        in ("online", "serving", "starting", "member-ready")
    ]
    total_vram = sum(float(n.gpu_memory_gb or 0) for n in ready)
    total_gpus = len(ready)  # one GPU per node today; future: per-node GPU count

    # Phase 2: the LIST of distributed instances across all nodes — each head
    # advertises the instances it heads. Resolve each instance's peer FABRIC IPs
    # (BUG D) back to member node ids/names. `distributed_instance` (singular)
    # stays = the first one, for one release of back-compat.
    from ainode.discovery.instance import InstanceRecord

    by_fabric = {
        (getattr(m, "fabric_ip", "") or ""): m
        for m in cluster.members() if getattr(m, "fabric_ip", "")
    }

    def _resolve_instance(head, inst):
        iid = inst.get("instance_id", "") or ""
        peers = list(inst.get("peer_ips", []) or [])
        peer_node_ids, member_names = [], [head.node_name]
        # A peer_ip with no online member behind it is a node that has gone
        # away since this instance launched. The instance is still *running* on
        # the head, but its ranks are incomplete — vLLM loses the ranks Ray
        # placed there — so it is reported degraded rather than healthy.
        missing_peer_ips = []
        for ip in peers:
            m = by_fabric.get(ip)
            if m is None:
                missing_peer_ips.append(ip)
            peer_node_ids.append(m.node_id if m else ip)
            member_names.append(m.node_name if m else ip)
        # model can be stale ("") if the head started idle; it's authoritative in
        # instance_id ("<node_id>:<model>").
        model = inst.get("model") or (iid.split(":", 1)[1] if ":" in iid else "") or head.model
        return {
            "instance_id": iid,
            "head_node_id": head.node_id,
            "head_node_name": head.node_name,
            "peer_ips": peers,
            "peer_node_ids": peer_node_ids,
            "member_names": member_names,
            "tensor_parallel_size": inst.get("tensor_parallel_size") or (1 + len(peers)),
            # PP/DP default to 1, so an instance advertised by a node running an
            # older build reads as the tensor-parallel split it actually is.
            "pipeline_parallel_size": inst.get("pipeline_parallel_size") or 1,
            "data_parallel_size": inst.get("data_parallel_size") or 1,
            # Ready-made badge text ("PP=3", "TP=2 · PP=2") so the UI does not
            # re-derive the label from three sizes in three places.
            "parallel_label": InstanceRecord.from_dict(inst).parallel_label(),
            # Peers this instance was launched across that are no longer online,
            # and the node_ids that ARE — what a relaunch would run on.
            "missing_peer_ips": missing_peer_ips,
            "degraded": bool(missing_peer_ips),
            "surviving_node_ids": [head.node_id] + [
                by_fabric[ip].node_id for ip in peers if ip in by_fabric
            ],
            "model": model,
            "status": inst.get("status", "serving"),
            # This instance's own load state. Without it the card had nothing
            # to draw and fell back to the state of whichever node the browser
            # was pointed at — so a model serving happily on a sub-node read
            # "STARTING · 8%" because the HEAD's engine was idle.
            "load_phase": inst.get("load_phase") or "",
            "load_detail": inst.get("load_detail") or "",
            "load_error": inst.get("load_error") or "",
            "load_timeline": inst.get("load_timeline") or [],
            "load_seconds": inst.get("load_seconds") or 0,
        }

    distributed_instances = []
    for n in ready:
        node_instances = list(getattr(n, "instances", []) or [])
        if not node_instances:
            # Back-compat: an older node advertises only the singular fields.
            iid = getattr(n, "distributed_instance_id", None)
            if iid:
                node_instances = [{
                    "instance_id": iid,
                    "model": (iid.split(":", 1)[1] if ":" in iid else getattr(n, "model", "")),
                    "peer_ips": list(getattr(n, "distributed_peers", []) or []),
                }]
        for inst in node_instances:
            distributed_instances.append(_resolve_instance(n, inst))

    distributed_instance = distributed_instances[0] if distributed_instances else None

    nodes_payload = []
    for n in ready:
        nodes_payload.append({
            "node_id": n.node_id,
            "hostname": n.node_name,
            "fabric_ip": getattr(n, "fabric_ip", "") or "",
            # RoCE link addresses, so an operator can see which peers the head
            # has a direct cable to (bulk transfer path — see topology.py).
            "ib_ips": list(getattr(n, "ib_ips", []) or []),
            "vram_gb": round(float(n.gpu_memory_gb or 0), 1),
            "gpus": 1,
            "gpu_name": n.gpu_name,
            "unified_memory": n.unified_memory,
            "status": n.status.value if hasattr(n.status, "value") else str(n.status),
            "distributed_mode": getattr(n, "distributed_mode", "solo") or "solo",
            "ray_status": (
                "head" if (ray_state.is_head and cluster.is_master_of_cluster() and n.node_id == (cluster._local_announcement.node_id if cluster._local_announcement else ""))
                else ("joined" if ray_state.joined_as_worker and cluster._local_announcement and n.node_id == cluster._local_announcement.node_id
                      else "unknown")
            ),
        })

    return web.json_response({
        "total_vram_gb": round(total_vram, 1),
        "available_vram_gb": round(total_vram, 1),  # best-effort; same as total until live utilization is wired
        "total_gpus": total_gpus,
        "total_nodes": len(ready),
        "nodes": nodes_payload,
        "distributed_instance": distributed_instance,
        "distributed_instances": distributed_instances,
        "ray": ray_state.to_dict(),
    })


def _rebuild_announcement(app: web.Application) -> None:
    """Apply the current config to the live broadcast announcement.

    Lets role/cluster-id changes take effect without a server restart.
    """
    config: NodeConfig = app["config"]
    sender: Optional[BroadcastSender] = app.get("broadcast_sender")
    if sender is not None:
        sender.update_announcement(
            cluster_id=getattr(config, "cluster_id", "default"),
            role=getattr(config, "cluster_role", "auto"),
        )
    # Also refresh the stored announcement
    announcement: NodeAnnouncement = app.get("announcement")
    if announcement is not None:
        announcement.cluster_id = getattr(config, "cluster_id", "default")
        announcement.role = getattr(config, "cluster_role", "auto")


# In-memory store for cluster update job state
_cluster_update_state: dict = {}


async def handle_cluster_update_all(request: web.Request) -> web.Response:
    """POST /api/cluster/update-all — pull latest image and restart on all nodes.

    Workers are updated via HTTP (POST /api/engine/update on each worker's
    API port) — no SSH required. The master updates itself last.
    Poll GET /api/cluster/update-status for live per-node progress.
    """
    import asyncio
    import subprocess as _sp
    cluster: "ClusterState" = request.app["cluster_state"]
    config: NodeConfig = request.app["config"]
    session: aiohttp.ClientSession = request.app["client_session"]

    # Optional target version, threaded to self and every peer so the whole
    # cluster converges on the same image.
    requested = None
    try:
        if request.can_read_body:
            body = await request.json()
            if isinstance(body, dict):
                requested = body.get("version")
    except Exception:
        requested = None

    nodes = cluster.members()
    all_nodes = [{"node_id": config.node_id, "node_name": config.node_id, "host": "localhost", "port": config.web_port or 3000, "is_self": True}]
    for n in nodes:
        if n.node_id != config.node_id:
            peer_ip = getattr(n, "peer_ip", None) or n.host
            port = getattr(n, "web_port", 3000) or 3000
            all_nodes.append({
                "node_id": n.node_id,
                "node_name": getattr(n, "node_name", n.node_id),
                "host": peer_ip,
                "port": port,
                "is_self": False,
            })

    if len(all_nodes) == 0:
        return web.json_response({"error": "No nodes in cluster"}, status=400)

    update_id = f"update-{int(asyncio.get_event_loop().time())}"
    _cluster_update_state[update_id] = {
        "id": update_id,
        "status": "running",
        "nodes": {n["node_id"]: {"node_name": n["node_name"], "status": "pending", "log": ""} for n in all_nodes},
        "started_at": asyncio.get_event_loop().time(),
    }

    # Resolve the target tag once for the whole cluster.
    target = requested
    if not target:
        try:
            target = await asyncio.get_event_loop().run_in_executor(
                None, _fetch_latest_ghcr_tag
            )
        except Exception:
            target = None
    if not target:
        # Mark the just-created job failed instead of leaving it stuck at
        # "running" forever in the in-memory dict (a poll would never resolve).
        job = _cluster_update_state.get(update_id)
        if job is not None:
            job["status"] = "failed"
            for n in job["nodes"].values():
                n["status"] = "failed"
                n["log"] = "Could not resolve a target version from GHCR"
        return web.json_response(
            {"error": "Could not resolve a target version from GHCR"}, status=502
        )
    image = f"{AINODE_GHCR_REPO}:{target}"

    async def _update_node(node: dict) -> None:
        nid = node["node_id"]
        state = _cluster_update_state[update_id]["nodes"][nid]
        state["status"] = "updating"

        if node["is_self"]:
            # Update self: docker pull → write image.env → self-stop (systemd
            # Restart=always relaunches on the new image). systemctl does not
            # work from inside the container.
            try:
                loop = asyncio.get_event_loop()
                try:
                    pull = await loop.run_in_executor(
                        None, lambda: _sp.run(
                            ["docker", "pull", image],
                            capture_output=True, text=True, timeout=600
                        )
                    )
                except _sp.TimeoutExpired:
                    state["status"] = "failed"
                    state["log"] = "docker pull timed out after 600s"
                    return
                if pull.returncode != 0:
                    state["status"] = "failed"
                    state["log"] = pull.stderr[:500]
                    return
                try:
                    _write_image_env(image)
                except Exception as exc:
                    state["status"] = "failed"
                    state["log"] = f"image.env write failed: {exc}"[:200]
                    return
                # Same swappable-unit gate as handle_engine_update: never self-stop
                # a node whose unit predates the swappable image, or we'd drop it /
                # reboot the old image and still report "done". Report honestly.
                if not _unit_is_swappable():
                    state["status"] = "needs-migration"
                    state["log"] = (
                        "Image pulled + pinned, but this node's systemd unit "
                        "predates the swappable unit — not restarted. Re-run the "
                        "installer on the host to migrate."
                    )
                    return
                state["status"] = "done"
                state["log"] = "Updated — restarting on new image"

                async def _restart_self():
                    await asyncio.sleep(2)
                    await loop.run_in_executor(
                        None, lambda: _sp.run(
                            ["docker", "stop", "ainode"],
                            capture_output=True, text=True, timeout=60
                        )
                    )

                asyncio.get_event_loop().create_task(_restart_self())
            except Exception as exc:
                state["status"] = "failed"
                state["log"] = str(exc)[:200]
        else:
            # Update remote worker via HTTP — no SSH needed.
            # Each worker runs the same AINode container with /api/engine/update.
            url = f"http://{node['host']}:{node['port']}/api/engine/update"
            try:
                async with session.post(url, json={"version": target}, timeout=aiohttp.ClientTimeout(total=700)) as resp:
                    data = await resp.json()
                    if resp.status < 300:
                        state["status"] = "done"
                        state["log"] = data.get("message", "Updated and restarting")
                    else:
                        state["status"] = "failed"
                        state["log"] = data.get("error", f"HTTP {resp.status}")[:300]
            except asyncio.TimeoutError:
                state["status"] = "failed"
                state["log"] = "Timeout — worker may still be pulling the image"
            except Exception as exc:
                state["status"] = "failed"
                state["log"] = str(exc)[:200]

    async def _run_all():
        workers = [n for n in all_nodes if not n["is_self"]]
        self_node = next((n for n in all_nodes if n["is_self"]), None)

        await asyncio.gather(*[_update_node(n) for n in workers])

        if self_node:
            await _update_node(self_node)

        _cluster_update_state[update_id]["status"] = "complete"

    asyncio.get_event_loop().create_task(_run_all())

    return web.json_response({
        "update_id": update_id,
        "nodes": list(_cluster_update_state[update_id]["nodes"].keys()),
        "message": f"Updating {len(all_nodes)} node(s) via HTTP. Poll /api/cluster/update-status?id={update_id} for progress.",
    }, status=202)


async def handle_cluster_update_status(request: web.Request) -> web.Response:
    """GET /api/cluster/update-status?id=... — poll update progress."""
    update_id = request.query.get("id", "")
    if not update_id or update_id not in _cluster_update_state:
        # Return most recent if no ID given
        if _cluster_update_state:
            update_id = max(_cluster_update_state.keys())
        else:
            return web.json_response({"error": "No update in progress"}, status=404)
    return web.json_response(_cluster_update_state[update_id])


async def handle_cluster_set_role(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    role = body.get("role", "")
    if role not in ("auto", "master", "worker"):
        return web.json_response(
            {"error": {"message": "role must be one of auto|master|worker", "type": "invalid_request"}},
            status=400,
        )
    config: NodeConfig = request.app["config"]
    config.cluster_role = role
    config.save()
    _rebuild_announcement(request.app)
    return web.json_response({"ok": True, "cluster_role": role})


async def handle_cluster_set_id(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    cluster_id = str(body.get("cluster_id", "")).strip() or "default"
    if len(cluster_id) > 64 or not all(c.isalnum() or c in "-_." for c in cluster_id):
        return web.json_response(
            {"error": {"message": "cluster_id must be alphanumeric (plus -_.), 1-64 chars",
                       "type": "invalid_request"}},
            status=400,
        )
    config: NodeConfig = request.app["config"]
    config.cluster_id = cluster_id
    config.save()
    _rebuild_announcement(request.app)
    return web.json_response({"ok": True, "cluster_id": cluster_id})


# Fields that can be updated via PATCH /api/config. Keep this list tight --
# never expose auth or secret-related fields here.
PATCHABLE_CONFIG_FIELDS = {
    # Deployment policy, settable over the API so a sub-node can be switched
    # over from the head instead of by hand-editing JSON on each machine —
    # which is the whole point of a head-only deployment.
    "web_ui_enabled",
    "download_from_hub",
    "node_name",
    "email",
    "host",
    "api_port",
    "web_port",
    "discovery_port",
    "model",
    "models_dir",
    "max_model_len",
    "gpu_memory_utilization",
    "quantization",
    "trust_remote_code",
    "cluster_enabled",
    "cluster_role",
    "cluster_id",
    "master_address",
    "cluster_interface",
    "coord_interface",
    "rdma_hcas",
    "datasets_dir",
    "training_dir",
    "hf_cache_dir",
    "cors_origins",
    "telemetry",
    "training_default_method",
    "training_default_epochs",
    "training_default_batch_size",
    "training_default_learning_rate",
}


def _safe_config_dict(config: NodeConfig) -> dict:
    """Return a safely serializable view of the config (no secrets)."""
    data = asdict(config)
    # Scrub anything that might carry a credential.
    data.pop("cluster_secret", None)
    return data


async def handle_get_config(request: web.Request) -> web.Response:
    config: NodeConfig = request.app["config"]
    return web.json_response(_safe_config_dict(config))


async def handle_set_model(request: web.Request) -> web.Response:
    """POST /api/engine/set-model — switch to a different model and restart engine."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    model = str_field(body, "model")
    if not model or "/" not in model:
        return web.json_response({"error": "model must be a HF repo ID (org/name)"}, status=400)

    config: NodeConfig = request.app["config"]
    engine = request.app.get("engine")

    # Persist the new model choice
    config.model = model
    config.save()

    # Stop current engine and start fresh with new model
    if engine is not None:
        try:
            engine.stop()
        except Exception:
            pass
        try:
            engine.config.model = model
            engine.start()
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    return web.json_response({"status": "restarting", "model": model})


# Re-exported from core.config so the update check, the systemd unit and the
# installer cannot drift apart. Importers of this name keep working.
from ainode.core.config import AINODE_GHCR_REPO  # noqa: E402


def _fetch_latest_ghcr_tag() -> Optional[str]:
    """Return the highest numeric GHCR version tag, or None.

    Blocking (uses urllib); call via ``run_in_executor``. Resolves anonymously
    against the public image — no auth required.
    """
    import urllib.request
    import json as _json

    repo_path = AINODE_GHCR_REPO.split("/", 1)[-1] if "/" in AINODE_GHCR_REPO else AINODE_GHCR_REPO
    token_url = (
        f"https://ghcr.io/token?service=ghcr.io&scope=repository:{repo_path}:pull"
    )
    with urllib.request.urlopen(token_url, timeout=5) as r:
        token = _json.loads(r.read())["token"]
    tags_url = f"https://ghcr.io/v2/{repo_path}/tags/list"
    req = urllib.request.Request(tags_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = _json.loads(r.read())
    # Highest numeric version tag (ignore 'latest' and non-numeric tags).
    versions = sorted(
        [t for t in data.get("tags", []) if t and t != "latest" and t[0].isdigit()],
        key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit()),
        reverse=True,
    )
    return versions[0] if versions else None


def _ainode_home_path() -> Path:
    """Host-mounted AINode home (``~/.ainode`` bind-mounted to /root/.ainode)."""
    return Path(os.environ.get("AINODE_HOME", str(Path.home() / ".ainode")))


def _write_image_env(image: str) -> None:
    """Persist the target image for the host systemd unit's EnvironmentFile.

    Inside the container this writes ``$AINODE_HOME/image.env`` which is the
    same file the host unit reads via the ``~/.ainode`` bind mount.

    Written atomically (temp file + rename) so the host unit never reads a
    half-written line if it happens to (re)load while we're mid-write.
    """
    home = _ainode_home_path()
    home.mkdir(parents=True, exist_ok=True)
    tmp = home / "image.env.tmp"
    tmp.write_text(f"AINODE_IMAGE={image}\n")
    tmp.replace(home / "image.env")


def _unit_is_swappable() -> bool:
    """True iff this container was launched by the swappable-image systemd unit.

    The swappable unit (the one this deploy path relies on) stamps
    ``AINODE_UNIT_SWAPPABLE=1`` into the container env. Its ABSENCE means the
    host is still on a pre-swappable unit (pinned ExecStart, no EnvironmentFile),
    where a self-``docker stop`` is destructive rather than an image swap: the
    old unit either won't relaunch at all (Restart=on-failure + clean exit) or
    relaunches the SAME pinned image (it never reads image.env). In that case we
    must NOT self-stop — pull + pin image.env, and tell the operator to migrate
    the unit on the host first.
    """
    return os.environ.get("AINODE_UNIT_SWAPPABLE") == "1"


async def handle_version_check(request: web.Request) -> web.Response:
    """GET /api/version/check — compare local version against latest GHCR tag."""
    current = __version__
    try:
        loop = asyncio.get_event_loop()
        latest = await loop.run_in_executor(None, _fetch_latest_ghcr_tag)
    except Exception:
        latest = None

    update_available = False
    if latest and latest != current:
        try:
            cur_parts = tuple(int(x) for x in current.split(".") if x.isdigit())
            lat_parts = tuple(int(x) for x in latest.split(".") if x.isdigit())
            update_available = lat_parts > cur_parts
        except Exception:
            update_available = latest != current

    return web.json_response({
        "current": current,
        "latest": latest,
        "update_available": update_available,
    })


async def handle_engine_update(request: web.Request) -> web.Response:
    """POST /api/engine/update — pull a target image and swap to it.

    Body (optional): ``{"version": "X.Y.Z"}``. When omitted, targets the highest
    numeric GHCR tag. On a successful ``docker pull`` we write ``image.env``
    (which the host systemd unit reads via EnvironmentFile) and then, after a
    short delay, ``docker stop ainode`` on the mounted socket — systemd's
    ``Restart=always`` relaunches the container on the new image. ``systemctl``
    is deliberately NOT used: it does not work from inside the container.

    On pull failure we do NOT write image.env and do NOT restart — the node
    keeps running the current image.
    """
    import subprocess as _sp
    loop = asyncio.get_event_loop()

    # Optional requested version from the body.
    requested = None
    try:
        if request.can_read_body:
            body = await request.json()
            if isinstance(body, dict):
                requested = body.get("version")
    except Exception:
        requested = None

    target = requested
    if not target:
        try:
            target = await loop.run_in_executor(None, _fetch_latest_ghcr_tag)
        except Exception:
            target = None
    if not target:
        return web.json_response(
            {"error": "Could not resolve a target version from GHCR"}, status=502
        )

    image = f"{AINODE_GHCR_REPO}:{target}"

    try:
        pull = await loop.run_in_executor(
            None,
            lambda: _sp.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=600
            )
        )
    except _sp.TimeoutExpired:
        # Same clean no-env-write failure path as a nonzero-return pull — the
        # node keeps running the current image; nothing was pinned or restarted.
        return web.json_response(
            {"error": "docker pull failed", "detail": "docker pull timed out after 600s"},
            status=502,
        )
    if pull.returncode != 0:
        return web.json_response(
            {"error": "docker pull failed", "detail": (pull.stderr or "")[-500:]},
            status=502,
        )

    # Pull succeeded — pin the new image so the swappable unit boots it.
    try:
        _write_image_env(image)
    except Exception as exc:
        return web.json_response(
            {"error": f"failed to write image.env: {exc}"}, status=500
        )

    # Only self-stop when THIS container was launched by the swappable unit. On a
    # node still running a pre-swappable unit, `docker stop` is destructive (see
    # _unit_is_swappable): the pull + pin above are harmless and ready the node,
    # but restarting would either drop the node or reboot the SAME old image, so
    # we refuse and point the operator at the host-side migration.
    if not _unit_is_swappable():
        return web.json_response({
            "status": "pulled",
            "restarted": False,
            "target": target,
            "image": image,
            "message": (
                "Image pulled and pinned, but this node's systemd unit predates "
                "the swappable-image unit and will not pick it up. Migrate it on "
                "the host (re-run the installer: curl -fsSL "
                "https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh | bash) to boot the new image."
            ),
        })

    async def _self_restart():
        await asyncio.sleep(2)
        await loop.run_in_executor(
            None,
            lambda: _sp.run(
                ["docker", "stop", "ainode"],
                capture_output=True, text=True, timeout=60
            )
        )

    asyncio.get_event_loop().create_task(_self_restart())
    return web.json_response(
        {"status": "updating", "target": target, "image": image, "restarted": True}
    )


async def handle_patch_config(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request"}},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": {"message": "Body must be an object", "type": "invalid_request"}},
            status=400,
        )

    config: NodeConfig = request.app["config"]
    applied: dict = {}
    rejected: list = []

    for key, value in body.items():
        if key not in PATCHABLE_CONFIG_FIELDS:
            rejected.append(key)
            continue
        if not hasattr(config, key):
            rejected.append(key)
            continue
        # Basic validation on role / cluster_id
        if key == "cluster_role" and value not in ("auto", "master", "worker"):
            rejected.append(key)
            continue
        # Device names reach a file that a shell script parses, so a crafted
        # value is code execution rather than a bad config (see
        # topology.is_safe_device_name). Reject before it is ever stored.
        if key in ("cluster_interface", "coord_interface"):
            from ainode.cluster.topology import is_safe_device_name
            if value not in ("", None) and not is_safe_device_name(value):
                rejected.append(key)
                continue
        if key == "rdma_hcas":
            from ainode.cluster.topology import is_safe_device_name
            if not isinstance(value, list) or not all(
                is_safe_device_name(v) for v in value
            ):
                rejected.append(key)
                continue
        setattr(config, key, value)
        applied[key] = value

    config.save()
    if any(k in applied for k in ("cluster_id", "cluster_role")):
        _rebuild_announcement(request.app)

    return web.json_response({
        "ok": True,
        "applied": applied,
        "rejected": rejected,
        "config": _safe_config_dict(config),
    })


def run_server(config: Optional[NodeConfig] = None, engine=None) -> None:
    """Start the API server (blocking)."""
    if config is None:
        config = NodeConfig()
    app = create_app(config=config, engine=engine)
    web.run_app(app, host=config.host, port=config.web_port, print=None)
