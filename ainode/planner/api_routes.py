"""GET /api/planner — the launch this model should get, and the arithmetic.

One endpoint, deliberately read-only. It computes; it does not launch. The
launch form uses it to fill itself in, the fit hint uses it to say what will
actually happen, and an operator can call it directly to check a plan before
committing a node for ten minutes.
"""

from __future__ import annotations

import logging

from aiohttp import web

from ainode.planner.compute import NodeBudget, plan_for
from ainode.core.units import gb_from_gib, gb_from_mib
from ainode.planner.facts import local_facts

logger = logging.getLogger(__name__)

__all__ = ["register_planner_routes", "node_budgets"]


def register_planner_routes(app: web.Application) -> None:
    app.router.add_get("/api/planner", handle_plan)
    app.router.add_post("/api/planner/household", handle_household)


async def handle_household(request: web.Request) -> web.Response:
    """POST /api/planner/household — plan a profile wizard draft
    (planner/household.py). Read-only: computes, launches nothing."""
    import asyncio

    from ainode.planner.household import plan_household

    try:
        draft = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    if not isinstance(draft, dict):
        return web.json_response({"error": "a draft is an object"}, status=400)
    # In an executor: resolving reads every checkpoint's config and sizes.
    result = await asyncio.get_event_loop().run_in_executor(
        None, plan_household, request.app, draft)
    return web.json_response(result)


def node_budgets(app, node_ids=None) -> list:
    """Every node's memory as the planner needs it: total and genuinely free.

    Free, not total. A node already serving a model has most of its memory
    inside that engine's pool, and planning against the total is how a second
    load gets recommended onto a node that has nothing left to give.
    """
    cluster = app.get("cluster_state")
    config = app.get("config")
    own_id = str(getattr(config, "node_id", "") or "")
    live = _own_memory(app)
    wanted = set(node_ids or [])
    out = []
    for node in (cluster.members() if cluster is not None else []):
        node_id = str(getattr(node, "node_id", "") or "")
        if wanted and node_id not in wanted:
            continue
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        if not wanted and status not in ("online", "serving", "member-ready"):
            continue
        total_mb = float(getattr(node, "gpu_memory_total_mb", 0) or 0)
        used_mb = float(getattr(node, "gpu_memory_used_mb", 0) or 0)
        # This node's own entry in cluster state is built once at startup and
        # refreshed by the broadcast it sends, not by the one it receives —
        # so for itself it can be minutes old, or the figure it had before it
        # loaded anything. /api/nodes already reads it fresh from the
        # collector for exactly this reason; planning must too, or a node
        # plans a launch into memory it is already using.
        if live and node_id == own_id:
            total_mb, used_mb = live
        # Decimal GB, the same unit facts.weights_gb is in. This divided MiB
        # by 1024 and called the result _gb, so every plan subtracted decimal
        # gigabytes of weights from binary gigabytes of memory — 7.4% in the
        # direction that makes the model look bigger than the node, all of it
        # landing on the KV cache. See ainode/core/units.py.
        # The fallback is the rendering as sent, which is decimal since #212;
        # converting it as GiB overstated an updated node by 7%. Only a node
        # that sends no raw figure gets here (R4).
        total_gb = gb_from_mib(total_mb) if total_mb else \
            float(getattr(node, "gpu_memory_gb", 0) or 0)
        free_gb = gb_from_mib(total_mb - used_mb) if total_mb else total_gb
        # The operator's limit for this node (wizzard.md §5, E2): total use
        # may not pass it, so what is free is at most the limit minus what is
        # in use. Its own node reads the config, which is fresher than the
        # announcement it is about to send.
        limit = float((getattr(config, "memory_limit_gb", 0) if node_id == own_id
                       else getattr(node, "memory_limit_gb", 0)) or 0)
        if limit > 0 and total_mb:
            free_gb = min(free_gb, limit - gb_from_mib(used_mb))
        out.append(NodeBudget(
            node_id=node_id,
            name=str(getattr(node, "node_name", "") or node_id),
            total_gb=round(total_gb, 1),
            free_gb=round(max(0.0, free_gb), 1),
            limit_gb=round(limit, 1) if limit > 0 else 0.0,
        ))
    return out


def budgets_with_guard_reserve(app, node_ids=None) -> list:
    """Node budgets with the memory guard's line held back, not just the
    planner's own reserve.

    These were two numbers. The planner held back SYSTEM_RESERVE_GB and the
    guard refused launches below its warning line, and when the operator
    raised the guard's reserve — which the UI invites — the planner went on
    planning into memory the guard would not allow anyone to touch. The
    dialog said a model fitted, the gate refused it, and on the launch that
    slipped between them the node died.

    One number now: what the guard will not let go below is what the planner
    will not plan into.
    """
    budgets = node_budgets(app, node_ids)
    for budget in budgets:
        budget.free_gb = max(0.0, budget.free_gb - held_back_gb(app, budget.total_gb))
    return budgets


def held_back_gb(app, total_gb: float) -> float:
    """What a plan keeps clear on a node of ``total_gb``, above the planner's
    own SYSTEM_RESERVE_GB: the memory guard's line, and room to stand back
    from it. Shared by the launch planner and the profile wizard.
    """
    from ainode.planner.compute import SYSTEM_RESERVE_GB, plan_headroom_gb

    guard = app.get("memory_guard")
    # The reading's warn line, not the configured one: it carries the cap
    # against the machine's total memory, and the planner has to hold back
    # what the guard will actually enforce.
    try:
        warn_gb = float(guard.read().warn_mb) / 1024 if guard is not None else 0.0
    except Exception:
        warn_gb = float(getattr(guard, "warn_mb", 0.0) or 0.0) / 1024
    # warn_gb comes from the guard, which works in GiB off /proc/meminfo.
    # SYSTEM_RESERVE_GB and everything else here is decimal.
    extra = max(0.0, gb_from_gib(warn_gb) - SYSTEM_RESERVE_GB)
    # The guard's line, and then room to stand back from it. Planning up to
    # the line puts a launch that went exactly to plan one page-cache
    # fluctuation away from being killed — which is how two idle nodes filled
    # up on a model that fitted on paper.
    return extra + plan_headroom_gb(total_gb)


def _own_memory(app):
    """(total_mb, used_mb) measured here, now. None when unreadable."""
    collector = app.get("metrics_collector")
    if collector is None:
        return None
    try:
        metrics = collector.get_gpu_metrics() or {}
    except Exception:
        logger.debug("could not read local GPU metrics", exc_info=True)
        return None
    if metrics.get("error"):
        return None
    total = float(metrics.get("memory_total_mb") or 0)
    used = float(metrics.get("memory_used_mb") or 0)
    return (total, used) if total else None


def _recipe(app, model: str):
    """The catalog entry for this model, if there is one.

    ``_catalog_lookup`` rather than the public ``get_model_info``: the latter
    returns a serialised dict for the UI, and the planner wants the recipe
    object — ``supports_pipeline`` and ``recommended_gmu`` are not in that
    dict, and they are two of the three things the recipe contributes.
    """
    manager = app.get("model_manager")
    lookup = getattr(manager, "_catalog_lookup", None)
    if not callable(lookup):
        return None
    try:
        return lookup(model)
    except Exception:
        logger.debug("could not look %s up in the catalog", model, exc_info=True)
        return None


def _is_image(recipe, facts, app, model: str) -> bool:
    """Is this a diffusion pipeline rather than an LLM?

    Three signals, in order of authority: the catalog says so, the engine
    backend says so, or the checkpoint on disk has a model_index.json where a
    config.json would be. The last one matters because a model an operator
    downloaded themselves is in no catalog.
    """
    if recipe is not None and str(getattr(recipe, "modality", "")) == "image":
        return True
    if str(getattr(recipe, "engine_backend", "")) == "diffusers":
        return True
    manager = app.get("model_manager")
    try:
        for directory in (manager.model_dirs_for_repo(model)
                          if manager is not None else []):
            if (directory / "model_index.json").is_file():
                return True
            snapshots = directory / "snapshots"
            if snapshots.is_dir() and any(
                    (s / "model_index.json").is_file()
                    for s in snapshots.iterdir() if s.is_dir()):
                return True
    except Exception:
        logger.debug("could not inspect %s on disk", model, exc_info=True)
    return False


def _image_weights_gb(manager, model: str) -> float:
    """Bytes on disk for a pipeline directory, which has no config.json and
    so never reaches the LLM fact reader."""
    from ainode.planner.facts import weight_bytes_on_disk

    total = 0
    try:
        for directory in manager.model_dirs_for_repo(model):
            total = max(total, weight_bytes_on_disk(directory))
    except Exception:
        logger.debug("could not size %s", model, exc_info=True)
    return total / 1e9


def _measured_weights(app, model: str):
    """(weights per node, rank count) the engine actually reported, or (0, 0).

    The rank count matters: weights per node depend on how many ways they were
    split, so a figure from a two-node launch says nothing about one node. The
    store keeps the split the measurement was taken at, and a measurement
    without one is not used rather than assumed to be this launch's shape.
    """
    try:
        from ainode.measure.recorder import measured_for

        measurement = measured_for(app, model) or {}
    except Exception:
        logger.debug("could not read a measurement for %s", model, exc_info=True)
        return 0.0, 0
    weights = float(measurement.get("weights_gb") or 0)
    ranks = int(measurement.get("rank_count") or 0)
    return (weights, ranks) if weights > 0 and ranks > 0 else (0.0, 0)


#: MoE launches with a measured weights figure needed before their median
#: ratio replaces TP_REPLICATION. One was the Smaug case — and one point says
#: nothing about the next model.
MOE_CALIBRATION_MIN = 3


def _all_measurements(app) -> list:
    try:
        from ainode.measure.store import MeasurementStore

        store = app.get("measurement_store") or MeasurementStore()
        return list(store.load().values())
    except Exception:
        logger.debug("could not read the measurement store", exc_info=True)
        return []


def _moe_weight_factor(app, model: str):
    """(median loaded/on-disk ratio of measured MoE launches, how many), or
    (0.0, n) when there are fewer than MOE_CALIBRATION_MIN of them."""
    ratios = []
    for entry in _all_measurements(app):
        if entry.model == model or not entry.is_moe:
            continue
        if entry.weights_gb > 0 and entry.rank_count > 0 \
                and entry.disk_weights_gb > 0:
            ratios.append(entry.weights_gb * entry.rank_count
                          / entry.disk_weights_gb)
    if len(ratios) < MOE_CALIBRATION_MIN:
        return 0.0, len(ratios)
    ratios.sort()
    middle = len(ratios) // 2
    median = ratios[middle] if len(ratios) % 2 else \
        (ratios[middle - 1] + ratios[middle]) / 2
    # A ratio above 1.2 or below 0.5 is a measurement of something else — an
    # adapter, a draft model, a log from another launch — not a MoE factor.
    return (round(median, 3), len(ratios)) if 0.5 <= median <= 1.2 \
        else (0.0, len(ratios))


def _measured_bytes_per_token(app, model: str, kv_dtype: str) -> int:
    """What one token of this model's cache really cost, from vLLM's own
    figures (P2, P3, P5): the per-rank cache memory across all ranks, over
    the tokens the engine said it holds. 0 unless the measurement was taken at
    this KV dtype, since fp8 and bf16 differ by two."""
    try:
        from ainode.measure.recorder import measured_for

        measurement = measured_for(app, model) or {}
    except Exception:
        return 0
    tokens = int(measurement.get("kv_tokens") or 0)
    per_rank = float(measurement.get("kv_cache_gb") or 0)
    ranks = int(measurement.get("rank_count") or 0)
    measured_dtype = str(measurement.get("kv_cache_dtype") or "").lower()
    if not (tokens and per_rank and ranks and measured_dtype):
        return 0
    if measured_dtype != str(kv_dtype or "auto").lower():
        return 0
    return int(per_rank * ranks * 1e9 / tokens)


def _current_engine_version(app) -> str:
    """The vLLM this node runs now: what build-base-image.sh recorded when it
    built the engine (~/.ainode/engine-build.env), or failing that the vLLM of
    the newest measurement."""
    try:
        from ainode.core.config import AINODE_HOME

        for line in (AINODE_HOME / "engine-build.env").read_text().splitlines():
            if line.startswith("ENGINE_VLLM_VERSION="):
                version = line.split("=", 1)[1].strip()
                if version:
                    return version
    except (OSError, ImportError):
        pass
    newest = None
    for entry in _all_measurements(app):
        if entry.engine_version and (newest is None
                                     or entry.last_ok > newest.last_ok):
            newest = entry
    return newest.engine_version if newest is not None else ""


def _attach_measurement(app, model: str, payload: dict) -> None:
    """Put what this cluster has actually measured beside what was estimated.

    Not instead of: the plan's own arithmetic stays visible, because a
    measurement taken at 64k context says nothing about the same model at
    256k, and quietly replacing one number with the other would hide that.
    Side by side, the difference is the interesting part — a plan that
    predicted 68 GB for something that cost 74 is a planner worth correcting.
    """
    from ainode.measure.recorder import measured_for

    measurement = measured_for(app, model)
    if measurement is None:
        return
    payload["measured"] = {
        "memory_gb": measurement.get("memory_gb"),
        "load_seconds": measurement.get("load_seconds"),
        "launches": measurement.get("launches"),
        "failures": measurement.get("failures"),
        "last_ok": measurement.get("last_ok"),
        "node_id": measurement.get("node_id"),
        "max_model_len": measurement.get("max_model_len"),
        "tokens_per_second": measurement.get("tokens_per_second"),
        "seconds_per_image": measurement.get("seconds_per_image"),
        # The engine's own split of memory_gb, where it said so.
        "weights_gb": measurement.get("weights_gb"),
        "kv_cache_gb": measurement.get("kv_cache_gb"),
        "kv_tokens": measurement.get("kv_tokens"),
        "kv_at_max_model_len": measurement.get("kv_at_max_model_len"),
        "kv_cache_dtype": measurement.get("kv_cache_dtype"),
        "engine_version": measurement.get("engine_version"),
        "memory_by_node": measurement.get("memory_by_node") or {},
    }
    # P7: the base image follows a rolling upstream build. A figure from
    # another vLLM is still the best there is, but it is said.
    version = measurement.get("engine_version") or ""
    current = _current_engine_version(app)
    if version and current and version != current:
        payload["measured"]["other_build"] = current
    # P4: what the engine cost beyond weights and cache — the planner assumes
    # ENGINE_OVERHEAD_GB and nobody had ever checked it. The host drop minus
    # the two figures the engine reports is that check, per node.
    footprint = float(measurement.get("memory_gb") or 0)
    weights = float(measurement.get("weights_gb") or 0)
    cache = float(measurement.get("kv_cache_gb") or 0)
    if footprint and weights and cache:
        payload["measured"]["overhead_gb"] = round(
            footprint - weights - cache, 1)
    # Like against like, which took two goes to get right.
    #
    # It first compared a measurement taken on ONE node against the weights
    # across ALL of them, so a two-node plan reported itself 76 GB out. Fixing
    # that to per-node made the figure read -0.6 GB, which looked like a
    # planner accurate to within a gigabyte. It was two errors cancelling:
    # memory_gb is the drop in MemAvailable on the host — the WHOLE footprint,
    # weights and engine and the cache the engine sized to fill its pool —
    # against an estimate of the weights alone. From the cluster, Smaug-Flash
    # at TP=2, where the engine's own log said:
    #
    #   Model loading took 67.7 GiB            (the weights, per rank)
    #   GPU KV cache size: 1,109,643 tokens    (13.7 GB per rank)
    #
    # 67.7 + 13.7 + the engine is the 83.1 that was measured, and the plan's
    # 83.7 was its guess at the 67.7. Nineteen percent over, hidden behind a
    # coincidence.
    #
    # So the footprint is compared against the footprint the plan predicts, and
    # the weights — where the engine reported them — against the weights the
    # plan estimated. Two comparisons, each of two things that are the same
    # kind of thing.
    actual = measurement.get("memory_gb") or 0
    predicted = payload.get("needed_per_node_gb") or 0
    if actual and predicted:
        payload["measured"]["vs_plan_gb"] = round(actual - predicted, 1)
        payload["measured"]["vs_plan_basis"] = "footprint per node"
    measured_weights = measurement.get("weights_gb") or 0
    estimated_weights = payload.get("weights_per_node_gb") or 0
    if measured_weights and estimated_weights:
        payload["measured"]["weights_vs_plan_gb"] = round(
            measured_weights - estimated_weights, 1)
    # A measurement taken at a different window is not a measurement of this
    # plan. Said rather than silently compared.
    at = measurement.get("max_model_len") or 0
    if at and payload.get("max_model_len") and at != payload["max_model_len"]:
        payload["measured"]["different_window"] = True


def _int(request, name, default=0):
    try:
        return int(request.query.get(name) or default)
    except (TypeError, ValueError):
        return default


async def handle_plan(request: web.Request) -> web.Response:
    """GET /api/planner?model=&nodes=&strategy=&max_model_len=&kv_cache_dtype=
    &concurrency="""
    model = request.query.get("model") or ""
    if not model:
        return web.json_response({"error": "model parameter required"}, status=400)

    node_ids = [n for n in (request.query.get("nodes") or "").split(",") if n]
    # The same budgets the admission gate uses. A dialog that is more
    # optimistic than the gate offers launches that are then refused — or
    # worse, is more optimistic than the hardware.
    nodes = budgets_with_guard_reserve(request.app, node_ids)
    manager = request.app.get("model_manager")
    if manager is None:
        return web.json_response({"error": "no model manager"}, status=503)

    facts = local_facts(manager, model)
    recipe = _recipe(request.app, model)

    # An image model is a different calculation, not the same one with other
    # constants: no KV cache, no context length, no parallel axis. Answering
    # it with the LLM planner would print KV figures for a model that has
    # none, which is worse than printing nothing.
    if _is_image(recipe, facts, request.app, model):
        from ainode.planner.compute import plan_for_image

        weights = facts.weights_gb
        if not weights:
            weights = _image_weights_gb(manager, model)
        plan = plan_for_image(
            weights, nodes, model=model,
            max_image_size=_int(request, "max_image_size", 1536) or 1536)
        payload = plan.to_dict()
        _attach_measurement(request.app, model, payload)
        payload["modality"] = "image"
        payload["nodes"] = [{"node_id": n.node_id, "name": n.name,
                             "total_gb": n.total_gb, "free_gb": n.free_gb}
                            for n in nodes]
        payload["from_catalog"] = recipe is not None
        return web.json_response(payload)

    kv_dtype = planning_kv_dtype(request.app, recipe,
                                 request.query.get("kv_cache_dtype") or "")
    measured_weights, measured_ranks = _measured_weights(request.app, model)
    moe_factor, moe_samples = _moe_weight_factor(request.app, model)

    plan = plan_for(
        facts, nodes,
        strategy=(request.query.get("strategy") or "auto").lower(),
        max_model_len=_int(request, "max_model_len"),
        kv_cache_dtype=kv_dtype or "auto",
        concurrency=_int(request, "concurrency", 1),
        supports_pipeline=bool(getattr(recipe, "supports_pipeline", True)),
        recipe_context=int(getattr(recipe, "context_length", 0) or 0),
        recommended_gmu=float(getattr(recipe, "recommended_gmu", 0.0) or 0.0),
        measured_weights_per_node=measured_weights,
        measured_rank_count=measured_ranks,
        # A window without a concurrency is the launch form asking "how many
        # sessions at this length": the answer is the concurrency it launches.
        concurrency_derived=(not request.query.get("concurrency")
                             and bool(_int(request, "max_model_len"))),
        moe_weight_factor=moe_factor,
        moe_weight_samples=moe_samples,
        measured_bytes_per_token=_measured_bytes_per_token(
            request.app, model, kv_dtype or "auto"),
    )
    return _plan_response(request.app, model, recipe, facts, nodes, plan, kv_dtype)


def planning_kv_dtype(app, recipe, requested: str = "") -> str:
    """The KV-cache dtype a launch of this model would use, for planning.

    Shared with the profile wizard's household planner (planner/household.py).
    """
    kv_dtype = str(requested or "")
    if not kv_dtype:
        # What the LAUNCH would use if nobody said otherwise — NodeConfig's
        # default is fp8, and the launch form says so in words ("Default (fp8
        # — required for long context on GB10)"). This defaulted to "auto"
        # instead, which is the model's own dtype, so every plan for a model
        # with no recipe was computed at twice the real cost per token. On
        # Qwen3-Coder-Next that is 24.0 KiB against 12.0, and a panel
        # reporting 165,774 tokens where the launch would hold 331,548.
        #
        # The recipe branch below was added for exactly this reason and only
        # covered curated models. The default is the other half of it.
        kv_dtype = str(getattr(app.get("config"), "kv_cache_dtype",
                               "") or "")
    if not kv_dtype and recipe is not None:
        # The recipe's own flags are part of the plan: a model whose proven
        # configuration is fp8 should be planned with an fp8-sized cache, or
        # the planner and the launch disagree by a factor of two.
        args = list(getattr(recipe, "extra_vllm_args", None) or [])
        if "--kv-cache-dtype" in args:
            index = args.index("--kv-cache-dtype")
            if index + 1 < len(args):
                kv_dtype = args[index + 1]
    return kv_dtype


def _plan_response(app, model, recipe, facts, nodes, plan, kv_dtype):
    payload = plan.to_dict()
    _attach_measurement(app, model, payload)
    payload["kv_cache_dtype"] = kv_dtype or "auto"
    payload["nodes"] = [{"node_id": n.node_id, "name": n.name,
                         "total_gb": n.total_gb, "free_gb": n.free_gb,
                         "limit_gb": n.limit_gb}
                        for n in nodes]
    payload["facts"] = {
        "architecture": facts.architecture,
        "num_layers": facts.num_layers,
        "attention_layers": facts.attention_layers,
        "num_kv_heads": facts.num_kv_heads,
        "head_dim": facts.head_dim,
        "max_position_embeddings": facts.max_position_embeddings,
        "torch_dtype": facts.torch_dtype,
        "quantization": facts.quantization,
        "is_moe": facts.is_moe,
        "num_experts": facts.num_experts,
        "is_hybrid": facts.is_hybrid,
        "weights_gb": round(facts.weights_gb, 1),
        "unknown": list(facts.unknown),
    }
    payload["from_catalog"] = recipe is not None
    return web.json_response(payload)
