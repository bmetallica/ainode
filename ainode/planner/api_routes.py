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
from ainode.planner.facts import local_facts

logger = logging.getLogger(__name__)

__all__ = ["register_planner_routes", "node_budgets"]


def register_planner_routes(app: web.Application) -> None:
    app.router.add_get("/api/planner", handle_plan)


def node_budgets(app, node_ids=None) -> list:
    """Every node's memory as the planner needs it: total and genuinely free.

    Free, not total. A node already serving a model has most of its memory
    inside that engine's pool, and planning against the total is how a second
    load gets recommended onto a node that has nothing left to give.
    """
    cluster = app.get("cluster_state")
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
        total_gb = float(getattr(node, "gpu_memory_gb", 0) or 0) or total_mb / 1024
        free_gb = (total_mb - used_mb) / 1024 if total_mb else total_gb
        out.append(NodeBudget(
            node_id=node_id,
            name=str(getattr(node, "node_name", "") or node_id),
            total_gb=round(total_gb, 1),
            free_gb=round(max(0.0, free_gb), 1),
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
    from ainode.planner.compute import SYSTEM_RESERVE_GB, plan_headroom_gb

    budgets = node_budgets(app, node_ids)
    guard = app.get("memory_guard")
    # The reading's warn line, not the configured one: it carries the cap
    # against the machine's total memory, and the planner has to hold back
    # what the guard will actually enforce.
    try:
        warn_gb = float(guard.read().warn_mb) / 1024 if guard is not None else 0.0
    except Exception:
        warn_gb = float(getattr(guard, "warn_mb", 0.0) or 0.0) / 1024
    extra = max(0.0, warn_gb - SYSTEM_RESERVE_GB)
    for budget in budgets:
        # The guard's line, and then room to stand back from it. Planning up
        # to the line puts a launch that went exactly to plan one page-cache
        # fluctuation away from being killed — which is how two idle nodes
        # filled up on a model that fitted on paper.
        held_back = extra + plan_headroom_gb(budget.total_gb)
        budget.free_gb = max(0.0, budget.free_gb - held_back)
    return budgets


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
    }
    estimated = payload.get("weights_gb") or 0
    actual = measurement.get("memory_gb") or 0
    if estimated and actual:
        payload["measured"]["vs_plan_gb"] = round(actual - estimated, 1)


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

    kv_dtype = request.query.get("kv_cache_dtype") or ""
    if not kv_dtype and recipe is not None:
        # The recipe's own flags are part of the plan: a model whose proven
        # configuration is fp8 should be planned with an fp8-sized cache, or
        # the planner and the launch disagree by a factor of two.
        args = list(getattr(recipe, "extra_vllm_args", None) or [])
        if "--kv-cache-dtype" in args:
            index = args.index("--kv-cache-dtype")
            if index + 1 < len(args):
                kv_dtype = args[index + 1]

    plan = plan_for(
        facts, nodes,
        strategy=(request.query.get("strategy") or "auto").lower(),
        max_model_len=_int(request, "max_model_len"),
        kv_cache_dtype=kv_dtype or "auto",
        concurrency=_int(request, "concurrency", 1),
        supports_pipeline=bool(getattr(recipe, "supports_pipeline", True)),
        recipe_context=int(getattr(recipe, "context_length", 0) or 0),
        recommended_gmu=float(getattr(recipe, "recommended_gmu", 0.0) or 0.0),
    )

    payload = plan.to_dict()
    _attach_measurement(request.app, model, payload)
    payload["kv_cache_dtype"] = kv_dtype or "auto"
    payload["nodes"] = [{"node_id": n.node_id, "name": n.name,
                         "total_gb": n.total_gb, "free_gb": n.free_gb}
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
