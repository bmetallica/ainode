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
import time

logger = logging.getLogger(__name__)

__all__ = ["check_admission", "AdmissionRefusal"]


class AdmissionRefusal(str):
    """A refusal message. Falsy when there is nothing to refuse.

    A string, because every caller treats it as one. ``clearable`` is set
    when the refusal rests on a record the operator can drop — the guard's
    memory of a kill — so a route can offer that instead of leaving them to
    find the endpoint in a paragraph of prose.
    """

    clearable: str = ""

    @classmethod
    def clearing(cls, text: str, model: str) -> "AdmissionRefusal":
        refusal = cls(text)
        refusal.clearable = model
        return refusal


def check_admission(app, model: str, *, node_ids=None, strategy: str = "auto",
                    max_model_len: int = 0, gpu_memory_utilization=None,
                    force: bool = False) -> str:
    """"" when the launch may proceed, else one sentence saying why not."""
    if force:
        return ""

    partial = _completeness_says(app, model)
    if partial:
        return partial

    unreadable = _quantization_says(app, model)
    if unreadable:
        return unreadable

    killed = _guard_history_says(app, model, node_ids=node_ids,
                                 gpu_memory_utilization=gpu_memory_utilization,
                                 max_model_len=max_model_len)
    if killed:
        return killed

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


def _guard_history_says(app, model: str, *, node_ids=None,
                        gpu_memory_utilization=None,
                        max_model_len: int = 0) -> str:
    """Refuse a launch the guard has already had to stop, unless it asks for
    less than the one that was stopped.

    The hardest fact this node holds about a model. An estimate can be wrong
    in either direction; a kill is proof that this node, configured that way,
    could not hold it. Asking for less — fewer tokens, a smaller share of
    memory, more nodes to spread it over — is a different question, and it is
    allowed through to be answered by the arithmetic below.
    """
    from ainode.measure.recorder import guard_stopped_for

    entry = guard_stopped_for(app, model)
    if entry is None:
        return ""

    # A launch that now carries a flag the killed one did not is a different
    # launch. Expert parallelism is the case this exists for: without it a
    # mixture-of-experts is replicated onto every rank, and the fix for that
    # would otherwise be refused on the strength of the failure it fixes.
    from ainode.models.architecture import architecture_args

    recorded_args = set(entry.get("guard_stop_args") or [])
    wanted_nodes_now = len(list(node_ids or [])) or 1
    added = [flag for flag in architecture_args(app, model, wanted_nodes_now)
             if flag not in recorded_args]
    if added:
        logger.info("%s: the guard stopped it without %s; this launch has it",
                    model, " ".join(added))
        return ""

    killed_gmu = float(entry.get("guard_stop_gmu") or 0)
    killed_len = int(entry.get("guard_stop_max_model_len") or 0)
    killed_nodes = int(entry.get("guard_stop_nodes") or 0)
    wanted_nodes = len(list(node_ids or [])) or 1
    if killed_nodes and wanted_nodes > killed_nodes:
        return ""
    try:
        wanted_gmu = float(gpu_memory_utilization) if gpu_memory_utilization else 0.0
    except (TypeError, ValueError):
        wanted_gmu = 0.0
    if killed_gmu and wanted_gmu and wanted_gmu < killed_gmu:
        return ""
    if killed_len and max_model_len and max_model_len < killed_len:
        return ""

    when = time.strftime("%Y-%m-%d %H:%M",
                         time.localtime(float(entry.get("last_guard_stop") or 0)))
    asked = []
    if killed_gmu:
        asked.append(f"gpu-memory-utilization {killed_gmu:.2f}")
    if killed_len:
        asked.append(f"max-model-len {killed_len:,}")
    if killed_nodes:
        asked.append(f"{killed_nodes} node(s)")
    detail = ", ".join(asked) or "these settings"
    times = int(entry.get("guard_stops") or 1)
    return AdmissionRefusal.clearing((
        f"The host memory guard has already had to stop {model} on this node "
        f"{'once' if times == 1 else f'{times} times'}, most recently on "
        f"{when}, launched with {detail}. That is not an estimate — the node "
        f"ran out of memory and an engine had to be killed to save it. "
        f"Launch it with fewer tokens, a lower gpu-memory-utilization or more "
        f"nodes and this refusal lifts by itself; \"force\": true overrides "
        f"it outright. If what caused it has been fixed since — a flag added, "
        f"an engine image rebuilt — clear the record and start again:\n\n"
        f"    curl -X POST localhost:3000/api/measurements/forget-stops \\\n"
        f"      -H 'Content-Type: application/json' -d '{{\"model\": \"{model}\"}}'"
    ), model)


def _completeness_says(app, model: str) -> str:
    """Refuse a checkpoint that is not all there.

    First of all the checks, because it is the cheapest and the most certain:
    a missing shard is not a question of memory, and the launch it produces
    fails minutes in with something about safetensors rather than about the
    download that was interrupted.
    """
    manager = app.get("model_manager")
    if manager is None:
        return ""
    try:
        from ainode.models.completeness import download_state

        directories = manager.model_dirs_for_repo(model)
    except Exception:
        logger.debug("could not locate %s on disk", model, exc_info=True)
        return ""
    for directory in directories:
        complete, reason = download_state(directory)
        if not complete:
            return f"{model}: {reason}"
    return ""


def _quantization_says(app, model: str) -> str:
    """Refuse a checkpoint this engine provably cannot parse.

    Before the memory questions, because it is not one: there is no amount of
    free memory that makes an unreadable quantization_config readable, and the
    failure it prevents costs two nodes and several minutes before vLLM gets
    as far as parsing it.

    This is the one refusal here that `force` still overrides — the operator
    may have put the plugin in the engine image since, and this file cannot
    see inside it.
    """
    manager = app.get("model_manager")
    if manager is None:
        return ""
    try:
        from ainode.models.quantization import quantization_verdict
        from ainode.planner.facts import read_config

        directories = manager.model_dirs_for_repo(model)
    except Exception:
        logger.debug("could not locate %s on disk", model, exc_info=True)
        return ""

    for directory in directories:
        config = read_config(directory)
        if not config:
            continue
        servable, reason = quantization_verdict(config)
        if not servable:
            return reason
        return ""
    return ""


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

    image_refusal = _image_says(app, manager, model, node_ids)
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

    measured = _measured_cost(app, model)
    if measured:
        # This model has run here. What it actually cost beats any arithmetic
        # about what it ought to cost — but only when it was launched the same
        # way: a memory figure from a 64k context says nothing about 256k.
        refusal = _measured_says(app, model, measured, max_model_len, node_ids)
        if refusal is not None:
            return refusal

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


def _measured_cost(app, model: str):
    """The last measured host cost of this model, or None."""
    from ainode.measure.recorder import measured_for

    measurement = measured_for(app, model)
    if measurement is None or not measurement.get("memory_gb"):
        return None
    return measurement


def _measured_says(app, model: str, measured: dict, max_model_len: int,
                   node_ids=None):
    """Verdict from what the model actually cost, or None to fall through.

    Only when this launch matches the measured one. A model measured at 64k
    and launched at 256k needs a different amount, and pretending otherwise
    would be the same overconfidence the estimate is criticised for.
    """
    wanted = int(max_model_len or 0)
    measured_len = int(measured.get("max_model_len") or 0)
    if wanted and measured_len and wanted != measured_len:
        return None

    budgets = _budgets_with_reserve(app, node_ids)
    if not budgets:
        return None
    need = float(measured["memory_gb"])
    roomiest = max(b.usable_gb for b in budgets)
    if roomiest >= need:
        return ""
    return (
        f"{model} cost {need:.0f} GB the last time it ran here, and the "
        f"roomiest node has {roomiest:.0f} GB free. That is a measurement, "
        f"not an estimate. Free memory, or launch anyway with "
        f"\"force\": true."
    )


def _image_says(app, manager, model: str, node_ids=None):
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
            weights, _budgets_with_reserve(app, node_ids), model=model,
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

    One implementation, shared with the launch dialog — they used to be two,
    and the dialog's was the more optimistic of them.
    """
    from ainode.planner.api_routes import budgets_with_guard_reserve

    return budgets_with_guard_reserve(app, node_ids)


def _kv_dtype(recipe) -> str:
    args = list(getattr(recipe, "extra_vllm_args", None) or [])
    if "--kv-cache-dtype" in args:
        index = args.index("--kv-cache-dtype")
        if index + 1 < len(args):
            return str(args[index + 1])
    return "auto"
