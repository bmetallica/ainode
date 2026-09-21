"""Bring the node to the state a profile describes.

Applying a profile *converges*: what the profile asks for and is not running is
started, what is running and the profile does not mention is stopped, and what
already matches is left alone. The alternative — "start everything in the
profile, touch nothing else" — makes the result depend on what happened to be
running before, so applying the same profile twice could leave two different
machines, and the models the profile replaced would go on holding memory.

Starts are serialised, and each model has to be answering before the next one
begins. Two vLLM engines loading at once on a unified-memory node compete for
the same pool while neither has finished reserving, and the loser is OOM-killed
minutes in. The startup replay already works this way; this follows it.

A failing entry is reported, not fatal. One model that cannot load must not
stop the other two from coming up — the operator gets a per-entry result and
can fix the one that broke.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import List, Optional

from ainode.profiles.store import (
    KIND_EMBEDDING,
    Profile,
    ProfileEntry,
    ProfileError,
)

logger = logging.getLogger(__name__)

__all__ = ["apply_profile", "capture_profile", "ApplyResult"]

#: How long one model gets to start answering before the next one is launched.
#: Generous: a frontier MoE reading tens of GB off disk on first load is slow,
#: and giving up early would start the next model into a node that is still
#: reserving memory — exactly the race the serialisation exists to avoid.
ENTRY_READY_TIMEOUT = 900.0


class ApplyResult(dict):
    """Per-entry outcome, JSON-shaped: model, action, ok, error."""

    def __init__(self, model: str, action: str, ok: bool, error: str = "", **extra):
        super().__init__(model=model, action=action, ok=ok, error=error, **extra)


class _AppRequest:
    """The parts of an aiohttp request the launch routes actually use.

    The distributed launch is an HTTP handler, and reimplementing it here would
    give the cluster two launch paths that drift apart — the exact problem the
    catalog recipe just had. So a profile calls the real handler with a stand-in
    request instead. The relaunch route does the same thing with its own shim.
    """

    def __init__(self, app, body: dict):
        self.app = app
        self._body = body

    async def json(self) -> dict:
        return self._body


def _instance_config(instance) -> object:
    return getattr(getattr(instance, "backend", None), "config", None)


def _node_set_of(instance, config) -> set:
    """Node ids an instance currently spans, as far as this node can tell.

    Instances record peer *addresses*, not ids, so the comparison is made on
    node count plus this node's own id. That is enough to notice "the profile
    wants three nodes and this is running on one"; it will not notice a swap of
    one peer for another, which re-planning would fix anyway.
    """
    peers = list(getattr(getattr(instance, "record", None), "peer_ips", []) or [])
    own = str(getattr(config, "node_id", "") or "head")
    return {own} | {f"peer:{ip}" for ip in peers}


def _entry_matches(entry: ProfileEntry, instance, config) -> bool:
    """True when the running instance already is what the entry describes.

    Only fields the entry actually states are compared. An entry that says
    nothing about ``kv_cache_dtype`` is happy with whatever the recipe chose,
    so a matching instance must not be restarted over it.
    """
    inst_config = _instance_config(instance)
    if inst_config is None:
        return False

    wanted_nodes = max(1, len(entry.node_ids))
    running_nodes = len(_node_set_of(instance, config))
    if wanted_nodes != running_nodes:
        return False

    simple = (
        "gpu_memory_utilization", "max_model_len", "kv_cache_dtype",
        "quantization", "engine_image",
    )
    for name in simple:
        wanted = getattr(entry, name, None)
        if wanted in (None, "", []):
            continue
        current = getattr(inst_config, name, None)
        if name == "gpu_memory_utilization":
            if current is None or abs(float(current) - float(wanted)) > 1e-6:
                return False
        elif str(current or "") != str(wanted):
            return False

    for name in ("served_model_name", "extra_vllm_args"):
        wanted_list = getattr(entry, name, None) or []
        if not wanted_list:
            continue
        if list(getattr(inst_config, name, []) or []) != list(wanted_list):
            return False

    if entry.extra_env and dict(getattr(inst_config, "extra_env", {}) or {}) != entry.extra_env:
        return False
    if entry.trust_remote_code is not None and bool(
        getattr(inst_config, "trust_remote_code", False)
    ) != bool(entry.trust_remote_code):
        return False
    return True


async def _start_llm_entry(app, entry: ProfileEntry) -> ApplyResult:
    """Launch one entry through the route that already knows how.

    Solo and distributed are genuinely different paths — one starts a container
    here, the other forms a Ray cluster across nodes — so the entry picks by
    node count and then delegates. Neither is reimplemented.
    """
    body = entry.launch_body()

    if entry.is_distributed:
        from ainode.engine.sharding_routes import handle_sharding_launch

        response = await handle_sharding_launch(_AppRequest(app, body))
        if response.status == 200:
            import json as _json
            payload = _json.loads(response.body)
            return ApplyResult(entry.model, "launched", True,
                               api_port=payload.get("api_port"),
                               plan=payload.get("parallel_plan"))
        return ApplyResult(entry.model, "launch_failed", False,
                           _response_error(response))

    from ainode.models.api_routes import append_solo_instance, parse_load_overrides

    overrides, err = parse_load_overrides(body)
    if err is not None:
        return ApplyResult(entry.model, "rejected", False,
                           _response_error(err))
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: append_solo_instance(
            app, entry.model, entry.gpu_memory_utilization,
            overrides=overrides, persist=True,
        ),
    )
    if not isinstance(result, dict) or not result.get("ok"):
        error = (result or {}).get("error", "launch failed")
        return ApplyResult(entry.model, "launch_failed", False, str(error))
    return ApplyResult(entry.model, "launched", True,
                       api_port=result.get("api_port"))


def _response_error(response) -> str:
    import json as _json
    try:
        return str(_json.loads(response.body).get("error") or response.body)
    except Exception:
        return f"HTTP {getattr(response, 'status', '?')}"


def _entry_target_node(app, entry: ProfileEntry) -> str:
    """The node id an entry names, or "" for this one."""
    wanted = [n for n in (entry.node_ids or []) if n]
    if not wanted:
        return ""
    own = str(getattr(app.get("config"), "node_id", "") or "")
    return "" if wanted[0] == own else wanted[0]


async def _start_embedding_entry(app, entry: ProfileEntry) -> ApplyResult:
    # Placement first: an embedding entry captured on the head can name
    # another node, and loading it here instead would put the RAG model on
    # the wrong machine — silently, since both nodes answer.
    target = _entry_target_node(app, entry)
    if target:
        from ainode.api.server import handle_cluster_embedding_load

        response = await handle_cluster_embedding_load(
            _AppRequest(app, {"node_id": target, "model": entry.model}))
        if response.status == 200:
            return ApplyResult(entry.model, "launched", True, node_id=target)
        return ApplyResult(entry.model, "launch_failed", False,
                           _response_error(response), node_id=target)

    manager = app.get("embedding_manager")
    if manager is None:
        return ApplyResult(entry.model, "unavailable", False,
                           "This build has no embedding manager.")
    if manager.is_loaded(entry.model):
        return ApplyResult(entry.model, "already_running", True)
    try:
        await manager.aload(entry.model)
    except Exception as exc:
        logger.exception("profile: embedding load failed for %s", entry.model)
        return ApplyResult(entry.model, "launch_failed", False, str(exc))
    try:
        manager.save_manifest()
    except Exception:
        logger.exception("profile: could not persist embedding manifest")
    return ApplyResult(entry.model, "launched", True)


def _stop_llm(app, instance) -> None:
    manager = app.get("instances")
    try:
        instance.backend.stop()
    except Exception:
        logger.exception("profile: stop() failed for %s", instance.record.model)
    if manager is not None:
        manager.remove(instance.record.instance_id)
    if app.get("engine") is getattr(instance, "backend", None):
        app["engine"] = None


async def apply_profile(app, profile: Profile, *, wait: bool = True) -> dict:
    """Converge this node onto ``profile``. Never raises; reports per entry."""
    config = app.get("config")
    manager = app.get("instances")
    embeddings = app.get("embedding_manager")

    llm_entries = [e for e in profile.entries if e.kind != KIND_EMBEDDING]
    emb_entries = [e for e in profile.entries if e.kind == KIND_EMBEDDING]
    wanted_llm = {e.model for e in llm_entries}
    wanted_emb = {e.model for e in emb_entries}

    results: List[ApplyResult] = []
    stopped: List[str] = []

    # 1. Stop what the profile does not ask for. First, so the memory it held
    #    is free before anything new reserves.
    running = {i.record.model: i for i in manager.instances()} if manager else {}
    for model, instance in list(running.items()):
        if model not in wanted_llm:
            _stop_llm(app, instance)
            stopped.append(model)
            running.pop(model, None)

    if embeddings is not None:
        for meta in list(embeddings.list_loaded()):
            model_id = meta.get("id") or ""
            if model_id and model_id not in wanted_emb:
                try:
                    embeddings.unload(model_id)
                    embeddings.save_manifest()
                    stopped.append(model_id)
                except Exception:
                    logger.exception("profile: unload failed for %s", model_id)

    # 2. Start what is missing, one at a time, waiting for each to answer.
    from ainode.models.api_routes import _wait_port_ready, save_instance_manifest

    for entry in llm_entries:
        instance = running.get(entry.model)
        if instance is not None and _entry_matches(entry, instance, config):
            results.append(ApplyResult(entry.model, "already_running", True,
                                       api_port=instance.record.api_port))
            continue
        if instance is not None:
            logger.info("profile %s: %s is running with a different "
                        "configuration — relaunching", profile.name, entry.model)
        result = await _start_llm_entry(app, entry)
        results.append(result)
        if result.get("ok") and wait and result.get("api_port"):
            ready = await _wait_port_ready(int(result["api_port"]),
                                           timeout=ENTRY_READY_TIMEOUT)
            result["serving"] = ready
            if not ready:
                # Not a failure of the launch — the engine may still be reading
                # weights. Say so plainly instead of calling it either way.
                result["error"] = (
                    f"{entry.model} did not answer on port {result['api_port']} "
                    f"within {int(ENTRY_READY_TIMEOUT)}s; it may still be "
                    f"loading. The next model was started anyway."
                )

    # Embedding models load in-process and do not contend for the engine's
    # memory reservation the way a vLLM instance does, so they go last and
    # without the readiness wait.
    for entry in emb_entries:
        results.append(await _start_embedding_entry(app, entry))

    try:
        save_instance_manifest(app)
    except Exception:
        logger.exception("profile: could not persist the instance manifest")

    ok = all(r.get("ok") for r in results)
    logger.info("Applied profile %s: %d entr(y/ies), %d stopped, %s",
                profile.name, len(results), len(stopped),
                "all ok" if ok else "with errors")
    return {
        "profile": profile.name,
        "ok": ok,
        "results": list(results),
        "stopped": stopped,
    }


def local_launch_specs(app) -> List[dict]:
    """Every LLM instance on THIS node, as profile-entry keyword arguments.

    One extraction, used by two callers that must not disagree: the local half
    of :func:`capture_profile`, and the /api/instances/launch-config endpoint a
    head calls to learn what a peer is running. A profile whose peer entries
    carried different fields from its local ones would restore two nodes
    differently from one cluster.
    """
    from ainode.profiles.store import KIND_IMAGE, KIND_LLM

    config = app.get("config")
    manager = app.get("instances")
    own_id = str(getattr(config, "node_id", "") or "head")

    specs: List[dict] = []
    for instance in (manager.instances() if manager is not None else []):
        inst_config = _instance_config(instance)
        record = instance.record
        peers = list(getattr(record, "peer_ips", []) or [])
        node_ids = [own_id] + [_peer_node_id(app, ip) or ip for ip in peers]
        specs.append({
            "model": record.model,
            # An image instance restored as an LLM would start vLLM on a
            # diffusers pipeline and fail; the profile has to carry which.
            "kind": (KIND_IMAGE if str(getattr(record, "kind", "")) == "image"
                     else KIND_LLM),
            "node_ids": node_ids if peers else [],
            "strategy": str(getattr(inst_config, "parallel_strategy", "") or ""),
            "gpu_memory_utilization": getattr(
                inst_config, "gpu_memory_utilization", None),
            "max_model_len": getattr(inst_config, "max_model_len", None),
            "kv_cache_dtype": str(getattr(inst_config, "kv_cache_dtype", "") or ""),
            "quantization": str(getattr(inst_config, "quantization", "") or ""),
            "served_model_name": list(
                getattr(inst_config, "served_model_name", None) or []),
            "trust_remote_code": bool(
                getattr(inst_config, "trust_remote_code", False)) or None,
            "extra_vllm_args": list(getattr(inst_config, "extra_vllm_args", None) or []),
            "extra_env": dict(getattr(inst_config, "extra_env", None) or {}),
            "engine_image": str(getattr(inst_config, "engine_image", "") or ""),
            "engine_backend": str(getattr(inst_config, "engine_backend", "") or ""),
            "max_image_size": getattr(inst_config, "max_image_size", None),
            "image_steps": getattr(inst_config, "image_steps", None),
            "image_size": str(getattr(inst_config, "image_size", "") or ""),
        })
    return specs


def capture_profile(app, name: str, description: str = "") -> Profile:
    """Turn what is running right now into a profile.

    This is how an operator realistically gets a profile: tune the deployment
    by hand until it is right, then keep it — rather than filling in a dozen
    fields per model in a form and hoping it matches what worked.
    """
    from ainode.profiles.store import KIND_LLM

    config = app.get("config")
    embeddings = app.get("embedding_manager")
    own_id = str(getattr(config, "node_id", "") or "head")

    entries: List[ProfileEntry] = []
    for spec in local_launch_specs(app):
        entries.append(ProfileEntry(**spec))

    # Embeddings, with the node each one is on. Recorded even for this node:
    # a profile is applied on whichever node has it, and an entry with no
    # placement lands wherever it is applied — which is how a RAG model
    # captured from node 3 would come back on the head.
    seen_embeddings = set()
    if embeddings is not None:
        for meta in embeddings.list_loaded():
            model_id = meta.get("id") or ""
            if model_id and model_id not in seen_embeddings:
                seen_embeddings.add(model_id)
                entries.append(ProfileEntry(model=model_id, kind=KIND_EMBEDDING,
                                            node_ids=[own_id]))

    # And the peers'. They advertise what they have loaded, so a profile saved
    # on the head describes the whole cluster rather than one machine of it.
    cluster = app.get("cluster_state")
    nodes = []
    if cluster is not None:
        try:
            nodes = [n for n in cluster.get_nodes() if n.node_id != own_id]
        except Exception:
            logger.debug("could not read the cluster for capture", exc_info=True)

    for node in nodes:
        for model_id in (getattr(node, "embedding_models", []) or []):
            if model_id and model_id not in seen_embeddings:
                seen_embeddings.add(model_id)
                entries.append(ProfileEntry(model=model_id,
                                            kind=KIND_EMBEDDING,
                                            node_ids=[node.node_id]))

    # The peers' LLM instances. Only this node's InstanceManager was read
    # above, so a model launched on another node — a solo Qwen on node 3, say
    # — was left out of the profile entirely, and applying it would bring the
    # cluster back one model short. The same blind spot the embeddings had:
    # the head describing itself and calling it the cluster.
    #
    # Asked for, not inferred: each peer hands over the exact launch
    # parameters of every model it runs. A profile without them restores the
    # models and loses the flags they need, which is a failure nobody sees
    # until the first request.
    served = {e.model for e in entries}
    for node in nodes:
        for spec in _peer_launch_specs(node):
            model_id = str(spec.get("model") or "")
            if not model_id or model_id in served:
                continue
            served.add(model_id)
            spec = dict(spec)
            # What the peer said it is, not what we assume. Forcing KIND_LLM
            # here captured an image model running on node 3 as an LLM, and
            # restoring that profile started vLLM on a diffusers pipeline —
            # the exact failure the kind exists to prevent. A peer too old to
            # send one is an LLM, because that is all there was.
            spec["kind"] = str(spec.get("kind") or "").strip() or KIND_LLM
            placement = [n for n in (spec.get("node_ids") or []) if n]
            spec["node_ids"] = placement or [node.node_id]
            try:
                entries.append(ProfileEntry(**spec))
            except (ProfileError, TypeError):
                logger.exception("peer %s sent an unusable entry for %s",
                                 node.node_id, model_id)
        # Whatever the peer could not tell us about, from its announcement.
        # An older build, or one that did not answer: the model still belongs
        # in the profile, with the catalog recipe filling in the rest.
        for record in (getattr(node, "instances", []) or []):
            if not isinstance(record, dict):
                continue
            model_id = str(record.get("model") or "")
            if not model_id or model_id in served:
                continue
            if str(record.get("status") or "serving") == "failed":
                # A profile is what should be running, and a launch that died
                # is not that. Capturing it would restore the failure.
                continue
            served.add(model_id)
            peers = list(record.get("peer_ips") or [])
            node_ids = [node.node_id] + [_peer_node_id(app, ip) or ip for ip in peers]
            entries.append(ProfileEntry(
                model=model_id,
                kind=KIND_LLM,
                node_ids=node_ids if peers else [node.node_id],
                # The axis it was launched on. An entry carries the strategy
                # and the node set; the sizes follow from those, which is how
                # a locally captured entry works too.
                #
                # The per-load overrides — memory fraction, context length,
                # extra flags — live in that node's own config and do not
                # cross the wire, so they stay empty and the launch falls back
                # to the catalog recipe and the node's defaults. Recording a
                # guess would be worse than recording nothing: it would look
                # exact.
                strategy=_strategy_of(record),
            ))

    return Profile(name=name, description=description, entries=entries)


#: One request per peer, on an action a person performs deliberately and
#: rarely. Short, because a node that has gone away must not hold up a save.
_LAUNCH_CONFIG_TIMEOUT = 8


def _peer_launch_specs(node) -> List[dict]:
    """Ask a peer how it started its models. [] when it cannot say.

    The launch parameters live in that node's InstanceManager and appear in no
    announcement. A profile that omits them looks complete and restores a
    model without the flags it needs.
    """
    host = getattr(node, "fabric_ip", "") or ""
    if not host:
        return []
    url = f"http://{host}:{getattr(node, 'web_port', 3000) or 3000}/api/instances/launch-config"
    try:
        with urllib.request.urlopen(url, timeout=_LAUNCH_CONFIG_TIMEOUT) as response:
            payload = json.loads(response.read().decode())
    except Exception:
        # An older build has no such route, and a node that is down answers
        # nothing. Both fall back to the announcement.
        logger.debug("no launch config from %s", getattr(node, "node_id", "?"),
                     exc_info=True)
        return []
    specs = payload.get("instances")
    return [s for s in specs if isinstance(s, dict)] if isinstance(specs, list) else []


def _strategy_of(record: dict) -> str:
    """The parallel axis an advertised instance was launched on, or "" if solo."""
    for size, strategy in (("tensor_parallel_size", "tensor"),
                           ("pipeline_parallel_size", "pipeline"),
                           ("data_parallel_size", "data")):
        try:
            if int(record.get(size) or 1) > 1:
                return strategy
        except (TypeError, ValueError):
            continue
    return ""


def _peer_node_id(app, fabric_ip: str) -> Optional[str]:
    """Map a peer's fabric address back to its node id, or None.

    A profile stores ids so it survives an address change; an instance records
    addresses because that is what it launched with. Where the cluster still
    knows the node, capture writes the id.
    """
    cluster = app.get("cluster_state")
    if cluster is None:
        return None
    try:
        for node in cluster.members():
            if (getattr(node, "fabric_ip", "") or "").strip() == fabric_ip:
                return node.node_id
    except Exception:
        logger.exception("profile: could not resolve %s to a node id", fabric_ip)
    return None


#: How long to let discovery find the peers before applying a profile that
#: spans them. A distributed entry launched into a half-discovered cluster
#: would be rejected for naming a node that is not a member yet.
DISCOVERY_GRACE = 45.0
SOLO_GRACE = 10.0


async def startup_restore(app) -> bool:
    """Apply the default profile at boot, if one is set.

    Returns True when a profile was applied — the caller then skips the
    instance-manifest replay. Two mechanisms deciding what should be running
    would contradict each other: the manifest records solo instances only, so
    after a restart it would quietly re-add a model the profile had replaced.
    A default profile is the operator saying which description wins.

    With no default profile set, nothing here happens and the manifest replay
    stays exactly as it was.
    """
    store = app.get("profiles")
    if store is None:
        # Read-only fallback: the app is already running by now, and assigning
        # into a started aiohttp application is deprecated. create_app puts the
        # store in place, so this only fires for an app built by hand.
        from ainode.profiles.store import ProfileStore
        store = ProfileStore()

    profile = store.default_profile()
    if profile is None:
        return False

    distributed = any(e.is_distributed for e in profile.entries)
    grace = DISCOVERY_GRACE if distributed else SOLO_GRACE
    logger.info("Default profile %r will be applied in %ds (%d entries%s)",
                profile.name, int(grace), len(profile.entries),
                ", waiting for peer discovery" if distributed else "")
    await asyncio.sleep(grace)

    try:
        report = await apply_profile(app, profile)
    except Exception:
        logger.exception("Applying default profile %r failed", profile.name)
        return True

    for result in report.get("results", []):
        if not result.get("ok"):
            logger.error("Default profile %r: %s — %s", profile.name,
                         result.get("model"), result.get("error"))
    return True
