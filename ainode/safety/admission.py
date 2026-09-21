"""May this launch start here?

One answer, asked by both launch paths. Before this there were two: a ratio
check that applied only to a stacked load and knew nothing about the host, and
— on the distributed path — nothing at all. The distributed path is the one
that took two nodes down.

Two questions, in order:

1. Is there host memory to spare *right now*? The guard knows.
2. Will what this launch is about to claim still leave that much? The planner
   knows, because it reads the checkpoint and every node's free memory.

Both are refusals a person can act on, and both can be overridden deliberately
— the planner is conservative on purpose, and an operator who knows what a
launch needs is allowed to be right.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["check_admission", "AdmissionRefusal"]


class AdmissionRefusal(str):
    """A refusal message. Falsy when there is nothing to refuse."""


def check_admission(app, model: str, *, node_ids=None, strategy: str = "auto",
                    max_model_len: int = 0, gpu_memory_utilization=None,
                    force: bool = False) -> str:
    """"" when the launch may proceed, else one sentence saying why not."""
    if force:
        return ""

    guard = app.get("memory_guard")
    if guard is not None:
        try:
            blocked = guard.accepting_loads()
        except Exception:
            logger.debug("the memory guard could not answer", exc_info=True)
            blocked = ""
        if blocked:
            return blocked

    return _planner_says(app, model, node_ids=node_ids, strategy=strategy,
                         max_model_len=max_model_len)


def _planner_says(app, model: str, *, node_ids=None, strategy: str = "auto",
                  max_model_len: int = 0) -> str:
    """The planner's verdict, or "" when it cannot form one.

    Silence is deliberate on anything it cannot compute. A planner that
    refuses what it does not understand would block every checkpoint whose
    config.json it cannot read — and the operator would then have no way to
    launch a model that works.
    """
    try:
        from ainode.planner.compute import plan_for
        from ainode.planner.facts import local_facts
    except Exception:  # pragma: no cover - defensive
        return ""

    manager = app.get("model_manager")
    if manager is None:
        return ""

    image_refusal = _image_says(app, manager, model)
    if image_refusal is not None:
        return image_refusal

    try:
        facts = local_facts(manager, model)
    except Exception:
        logger.debug("could not read the facts for %s", model, exc_info=True)
        return ""
    if not facts.weight_bytes:
        # Not on this node's disk: nothing to compute from, and the launch may
        # well be a distributed one whose weights live on a peer.
        return ""

    budgets = _budgets_with_reserve(app, node_ids)
    if not budgets:
        return ""

    recipe = None
    lookup = getattr(manager, "_catalog_lookup", None)
    if callable(lookup):
        try:
            recipe = lookup(model)
        except Exception:
            recipe = None

    try:
        plan = plan_for(
            facts, budgets, strategy=strategy, max_model_len=max_model_len,
            kv_cache_dtype=_kv_dtype(recipe),
            supports_pipeline=bool(getattr(recipe, "supports_pipeline", True)),
            recipe_context=int(getattr(recipe, "context_length", 0) or 0),
            recommended_gmu=float(getattr(recipe, "recommended_gmu", 0.0) or 0.0),
        )
    except Exception:
        logger.debug("the planner could not plan %s", model, exc_info=True)
        return ""
    if plan.fits:
        return ""
    return (
        f"{plan.blocker} — refusing the launch rather than letting the engine "
        f"find out after it has loaded the weights. On this hardware that "
        f"discovery takes the node down. Launch it anyway with \"force\": true "
        f"if you know better than this estimate."
    )


def _image_says(app, manager, model: str):
    """The image planner's verdict, or None when this is not an image model.

    None and "" mean different things here: "" is "an image model, and it
    fits", None is "not an image model, ask the LLM planner".
    """
    try:
        from ainode.planner.api_routes import _image_weights_gb, _is_image, _recipe
        from ainode.planner.compute import plan_for_image
        from ainode.planner.facts import local_facts
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        recipe = _recipe(app, model)
        if not _is_image(recipe, local_facts(manager, model), app, model):
            return None
        weights = _image_weights_gb(manager, model)
        if not weights:
            # Not downloaded yet. The backend refuses with a better message
            # than anything that could be said from here.
            return ""
        config = app.get("config")
        plan = plan_for_image(
            weights, _budgets_with_reserve(app, None), model=model,
            max_image_size=int(getattr(config, "max_image_size", 1536) or 1536))
    except Exception:
        logger.debug("the image planner could not plan %s", model, exc_info=True)
        return None
    if plan.fits:
        return ""
    return (
        f"{plan.blocker} — refusing the launch rather than letting the engine "
        f"find out. A diffusion run's peak lands at the END of a picture, so "
        f"the node would survive the load and die on the first image. Launch "
        f"it anyway with \"force\": true if you know better than this estimate."
    )


def _budgets_with_reserve(app, node_ids=None):
    """Node budgets, minus the host reserve this node keeps.

    The planner already holds back SYSTEM_RESERVE_GB per node. This tops that
    up to whatever the operator set as the warning line — never double-counts
    it — so the two numbers say the same thing: what the guard refuses to fall
    below is what the planner refuses to plan into.
    """
    from ainode.planner.api_routes import node_budgets
    from ainode.planner.compute import SYSTEM_RESERVE_GB

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
    if extra:
        for budget in budgets:
            budget.free_gb = max(0.0, budget.free_gb - extra)
    return budgets


def _kv_dtype(recipe) -> str:
    args = list(getattr(recipe, "extra_vllm_args", None) or [])
    if "--kv-cache-dtype" in args:
        index = args.index("--kv-cache-dtype")
        if index + 1 < len(args):
            return str(args[index + 1])
    return "auto"
