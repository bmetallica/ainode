"""A whole profile's memory, planned per node — the profile wizard's planner.

wizzard.md §2–§3. The launch planner (compute.plan_for) answers for one model
against what is free now. A profile is several models sharing nodes, planned
for the node as it will be once the profile is applied: every node has a
budget, every instance a fixed part (weights, engine, an image model's peak,
an embedding model) and — for an LLM — a KV cache that can be distributed.

    budget(n) = min(limit(n), total(n) − guard line and headroom) − baseline(n)
    Σ over instances on n of (fixed + cache) ≤ budget(n)

The cache of an LLM is set one of three ways (``mode``):

* ``usage`` — context × sessions, so the cache follows from them;
* ``size``  — GB of cache per node, so the sessions follow from the context;
* ``auto``  — a share of whatever is left, weighted by ``priority`` (E3).

``auto`` is solved by progressive filling: every automatic model grows at the
rate of its priority until one of its nodes is full or it reaches its ceiling,
the rest keep growing. That is what makes "more cache for A on node 1" mean
"less for B on node 1" — and, through a model that spans node 1 and node 2,
more room for something on node 2.

``solve`` is arithmetic on resolved items and has no app; ``plan_household``
resolves a draft from the wizard against the app (checkpoints, measurements,
recipes, node budgets) and then solves it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["HouseholdNode", "Item", "solve", "plan_household", "MODES"]

MODES = ("usage", "size", "auto")

#: vLLM hands out the cache in blocks and keeps a little back; a cache sized to
#: exactly context × sessions falls a few blocks short.
BLOCK_MARGIN = 1.02

#: An embedding model runs inside the AINode container: its weights, the
#: activations of a batch, and a CUDA context of its own.
EMBEDDING_FACTOR = 1.3
EMBEDDING_CONTEXT_GB = 1.0

#: The ceiling on any LLM's memory fraction when its recipe names none.
MAX_GMU = 0.95

_EPS = 1e-9


@dataclass
class HouseholdNode:
    node_id: str
    name: str = ""
    total_gb: float = 0.0
    limit_gb: float = 0.0
    baseline_gb: float = 0.0
    baseline_estimated: bool = False
    #: What profiles may occupy on this node, all deductions made.
    budget_gb: float = 0.0
    online: bool = True


@dataclass
class Item:
    """One instance, resolved: what it costs per node and how its cache is set."""
    id: str
    model: str
    kind: str = "llm"
    node_ids: List[str] = field(default_factory=list)
    #: GB on each of its nodes that do not move: weights, engine, rounding pad,
    #: an image model's peak, an embedding model.
    fixed_gb: float = 0.0
    mode: str = "auto"
    #: For usage/size: the cache per node it is pinned to.
    pinned_gb: float = 0.0
    #: For auto: the least it may have (one request at its context) and the
    #: most (its memory fraction's ceiling), and its weight.
    min_gb: float = 0.0
    cap_gb: float = math.inf
    priority: float = 1.0
    #: Filled by solve().
    cache_gb: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def has_cache(self) -> bool:
        return self.kind == "llm" and not self.errors


def solve(items: List[Item], nodes: List[HouseholdNode]) -> List[str]:
    """Set every item's ``cache_gb``. Returns the conflicts, one per node that
    cannot hold what is pinned to it."""
    by_id = {n.node_id: n for n in nodes}
    remaining: Dict[str, float] = {n.node_id: n.budget_gb for n in nodes}
    conflicts: List[str] = []

    placed = [i for i in items if not i.errors]
    for item in placed:
        for node_id in item.node_ids:
            if node_id in remaining:
                remaining[node_id] -= item.fixed_gb
    for item in placed:
        if not item.has_cache:
            continue
        if item.mode in ("usage", "size"):
            item.cache_gb = item.pinned_gb
        else:
            item.cache_gb = item.min_gb
        for node_id in item.node_ids:
            if node_id in remaining:
                remaining[node_id] -= item.cache_gb

    for node_id, left in remaining.items():
        if left < -0.05:
            node = by_id[node_id]
            on_node = [i for i in placed if node_id in i.node_ids]
            parts = ", ".join(
                f"{i.model} {i.fixed_gb + (i.cache_gb if i.has_cache else 0):.1f}"
                + (" (fest)" if i.mode in ("usage", "size") and i.has_cache else "")
                for i in on_node)
            conflicts.append(
                f"{node.name or node_id}: {-left:.1f} GB zu viel — "
                f"{parts} GB gegen ein Budget von {node.budget_gb:.1f} GB")

    # Progressive filling of the automatic ones over whatever is left.
    active = [i for i in placed if i.has_cache and i.mode == "auto"
              and i.cache_gb < i.cap_gb - _EPS
              and all(remaining.get(n, 0) > _EPS for n in i.node_ids)]
    for _ in range(4 * (len(items) + len(nodes)) + 4):
        if not active:
            break
        rate: Dict[str, float] = {}
        for item in active:
            for node_id in item.node_ids:
                rate[node_id] = rate.get(node_id, 0.0) + item.priority
        steps = [remaining[n] / r for n, r in rate.items() if r > 0]
        steps += [(i.cap_gb - i.cache_gb) / i.priority for i in active
                  if i.cap_gb < math.inf]
        step = max(0.0, min(steps)) if steps else 0.0
        for item in active:
            item.cache_gb += item.priority * step
        for node_id, r in rate.items():
            remaining[node_id] -= r * step
        active = [i for i in active
                  if i.cache_gb < i.cap_gb - 1e-6
                  and all(remaining.get(n, 0) > 1e-6 for n in i.node_ids)]
    return conflicts


# -- resolving a draft ------------------------------------------------------


def household_nodes(app, limits: Optional[Dict[str, float]] = None) -> List[HouseholdNode]:
    """Every cluster node's budget for a profile, planned against its total.

    ``limits`` overrides the node's own limit — the wizard shows the effect of
    a limit before it is saved.
    """
    from ainode.core.units import gb_from_mib, node_total_gb
    from ainode.planner.api_routes import held_back_gb
    from ainode.planner.compute import SYSTEM_RESERVE_GB
    from ainode.safety.baseline import DEFAULT_BASELINE_GB

    cluster = app.get("cluster_state")
    config = app.get("config")
    own_id = str(getattr(config, "node_id", "") or "")
    own_baseline = app.get("idle_baseline")
    out: List[HouseholdNode] = []
    for node in (cluster.members() if cluster is not None else []):
        node_id = str(getattr(node, "node_id", "") or "")
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        total = node_total_gb(node)
        if node_id == own_id:
            limit = float(getattr(config, "memory_limit_gb", 0) or 0)
            base_mb = float(own_baseline.used_mb() if own_baseline is not None
                            else getattr(node, "baseline_used_mb", 0) or 0)
        else:
            limit = float(getattr(node, "memory_limit_gb", 0) or 0)
            base_mb = float(getattr(node, "baseline_used_mb", 0) or 0)
        if limits and node_id in limits:
            limit = float(limits[node_id] or 0)
        baseline = gb_from_mib(base_mb) if base_mb else DEFAULT_BASELINE_GB
        ceiling = total - held_back_gb(app, total) - SYSTEM_RESERVE_GB
        cap = min(limit, ceiling) if limit > 0 else ceiling
        out.append(HouseholdNode(
            node_id=node_id, name=str(getattr(node, "node_name", "") or node_id),
            total_gb=round(total, 1), limit_gb=round(limit, 1),
            baseline_gb=round(baseline, 1), baseline_estimated=not base_mb,
            budget_gb=round(max(0.0, cap - baseline), 1),
            online=status in ("online", "serving", "member-ready")))
    return out


def _flag(args, name) -> Optional[str]:
    args = [str(a) for a in (args or [])]
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _without_flag(args, name) -> List[str]:
    args = [str(a) for a in (args or [])]
    out, skip = [], False
    for index, arg in enumerate(args):
        if skip:
            skip = False
            continue
        if arg == name:
            skip = index + 1 < len(args) and not args[index + 1].startswith("--")
            continue
        if arg.startswith(name + "="):
            continue
        out.append(arg)
    return out


def _resolve_llm(app, spec: dict, item: Item, nodes: Dict[str, HouseholdNode],
                 head_id: str) -> dict:
    """Fixed cost, cache limits and the arithmetic for one LLM. Returns what
    the result reports about it beyond the Item."""
    from ainode.planner import api_routes as routes
    from ainode.planner.compute import (COMM_OVERHEAD_GB, ENGINE_OVERHEAD_GB,
                                        LEN_GRANULARITY, _tensor_ok,
                                        kv_bytes_per_token, weights_per_node)
    from ainode.planner.facts import local_facts

    info: dict = {"warnings": []}
    manager = app.get("model_manager")
    facts = local_facts(manager, item.model) if manager is not None else None
    if facts is None or not facts.weight_bytes:
        item.errors.append(f"{item.model} ist auf diesem Node nicht heruntergeladen.")
        return info
    recipe = routes._recipe(app, item.model)
    count = len(item.node_ids)
    strategy = str(spec.get("strategy") or "").lower()
    if count == 1:
        strategy = "solo"
    elif strategy not in ("tensor", "pipeline"):
        strategy = "tensor" if not _tensor_ok(facts, count) else "pipeline"
    info["strategy"] = strategy
    if count > 1 and head_id and head_id not in item.node_ids:
        item.errors.append(
            "Ein Modell über mehrere Nodes muss den Head einschließen — die "
            "verteilte Engine wird vom Head aus gestartet.")
    if strategy == "tensor":
        why = _tensor_ok(facts, count)
        if why:
            item.errors.append(f"Tensor über {count} Nodes: {why}.")
    elif strategy == "pipeline":
        if not bool(getattr(recipe, "supports_pipeline", True)):
            item.errors.append("Dieses Modell unterstützt keine Pipeline-Aufteilung.")
        elif facts.num_layers and facts.num_layers < count:
            item.errors.append(f"Pipeline über {count} Nodes: nur {facts.num_layers} Layer.")

    requested_dtype = str(spec.get("kv_cache_dtype") or "")
    dtype = routes.planning_kv_dtype(app, recipe, requested_dtype) or "auto"
    info["kv_cache_dtype"] = dtype
    measured_w, measured_ranks = routes._measured_weights(app, item.model)
    moe_factor, _ = routes._moe_weight_factor(app, item.model)
    per_node, weights_source = weights_per_node(
        facts, count, strategy, measured_w, measured_ranks, moe_factor)
    bpt = routes._measured_bytes_per_token(app, item.model, dtype)
    kv_source = "measured" if bpt and measured_ranks == count else "estimated"
    if kv_source != "measured":
        bpt = kv_bytes_per_token(facts, dtype)
    overhead = ENGINE_OVERHEAD_GB + (COMM_OVERHEAD_GB if count > 1 else 0.0)
    totals = [nodes[n].total_gb for n in item.node_ids if n in nodes and nodes[n].total_gb]
    tightest = min(totals) if totals else 0.0
    # The memory fraction is written with two decimals; this keeps room for
    # rounding it up so the reservation never lands past the budget.
    pad = 0.01 * (max(totals) if totals else 0.0)
    item.fixed_gb = per_node + overhead + pad
    info.update(weights_per_node_gb=round(per_node, 1), weights_source=weights_source,
                overhead_gb=round(overhead, 1), bytes_per_token=int(bpt),
                kv_source=kv_source, count=count, tightest_total_gb=tightest,
                weights_gb=round(facts.weights_gb, 1))
    if not bpt:
        item.errors.append("Der Checkpoint nennt keine Layer-/Head-Zahlen; "
                           "der KV-Cache lässt sich nicht berechnen.")
        return info

    ceiling = int(facts.max_position_embeddings or
                  getattr(recipe, "context_length", 0) or 0)
    recipe_len = _flag(getattr(recipe, "extra_vllm_args", None), "--max-model-len")
    window = int(spec.get("max_model_len") or 0) or int(recipe_len or 0) \
        or min(ceiling or 131072, 131072)
    if ceiling and window > ceiling:
        info["warnings"].append(f"{window:,} überschreitet die {ceiling:,}, für "
                                f"die der Checkpoint trainiert ist; geplant mit {ceiling:,}.")
        window = ceiling
    window = max(LEN_GRANULARITY, window)
    info["max_model_len"] = window
    per_token_node_gb = bpt / count / 1e9
    gmu_cap = min(MAX_GMU, float(getattr(recipe, "recommended_gmu", 0) or 0) or MAX_GMU)
    if tightest:
        item.cap_gb = max(0.0, gmu_cap * tightest - item.fixed_gb + pad)
    info["gmu_cap"] = gmu_cap

    sessions = max(1, int(spec.get("sessions") or 1))
    one_request = window * per_token_node_gb * BLOCK_MARGIN
    if item.mode == "usage":
        item.pinned_gb = window * sessions * per_token_node_gb * BLOCK_MARGIN
        info["sessions"] = sessions
    elif item.mode == "size":
        item.pinned_gb = max(0.0, float(spec.get("cache_gb") or 0))
        if item.pinned_gb < one_request:
            item.errors.append(
                f"{item.pinned_gb:.1f} GB Cache pro Node halten keine einzige "
                f"Anfrage mit {window:,} Tokens (mindestens {one_request:.1f} GB).")
    else:
        item.min_gb = one_request
    if item.mode in ("usage", "size") and item.pinned_gb > item.cap_gb + 0.05:
        info["warnings"].append(
            f"Dafür wären mehr als {gmu_cap:.2f} des Speichers nötig "
            f"(Rezept-/Sicherheitsgrenze); der Cache wird dort nicht ganz ankommen.")
    info["per_token_node_gb"] = per_token_node_gb
    return info


def _resolve_image(app, spec: dict, item: Item) -> dict:
    from ainode.planner import api_routes as routes
    from ainode.planner.compute import (ENGINE_OVERHEAD_GB, IMAGE_OVERHEAD_GB,
                                        IMAGE_REFERENCE_PIXELS)

    manager = app.get("model_manager")
    weights = routes._image_weights_gb(manager, item.model) if manager is not None else 0.0
    if not weights:
        item.errors.append(f"{item.model} ist auf diesem Node nicht heruntergeladen.")
    if len(item.node_ids) != 1:
        item.errors.append("Ein Bildmodell läuft auf genau einem Node.")
    size = int(spec.get("max_image_size") or 1536)
    overhead = ENGINE_OVERHEAD_GB + IMAGE_OVERHEAD_GB * (size * size / IMAGE_REFERENCE_PIXELS)
    item.fixed_gb = weights + overhead
    return {"weights_gb": round(weights, 1), "max_image_size": size,
            "overhead_gb": round(overhead, 1), "warnings": []}


def _resolve_embedding(app, spec: dict, item: Item) -> dict:
    size_gb = 0.0
    try:
        from ainode.embeddings.manager import KNOWN_EMBEDDING_MODELS

        known = KNOWN_EMBEDDING_MODELS.get(item.model) or {}
        size_gb = float(known.get("size_mb") or 0) / 1000.0
    except Exception:
        pass
    if not size_gb:
        manager = app.get("model_manager")
        try:
            for directory in manager.model_dirs_for_repo(item.model):
                size_gb = max(size_gb, manager._dir_size_gb(directory))
        except Exception:
            pass
    if not size_gb:
        size_gb = 1.0
    if len(item.node_ids) != 1:
        item.errors.append("Ein Embedding-Modell läuft auf genau einem Node.")
    item.fixed_gb = size_gb * EMBEDDING_FACTOR + EMBEDDING_CONTEXT_GB
    return {"weights_gb": round(size_gb, 2), "warnings": []}


def _entry_for(spec: dict, item: Item, info: dict) -> Optional[dict]:
    """The ProfileEntry this planned item becomes, or None when it cannot."""
    if item.errors:
        return None
    if item.kind == "embedding":
        return {"model": item.model, "kind": "embedding", "node_ids": list(item.node_ids)}
    if item.kind == "image":
        return {"model": item.model, "kind": "image", "node_ids": list(item.node_ids),
                "engine_backend": "diffusers",
                "max_image_size": info.get("max_image_size")}
    args = _without_flag(spec.get("extra_vllm_args") or [], "--max-num-seqs")
    if info.get("max_num_seqs"):
        args += ["--max-num-seqs", str(info["max_num_seqs"])]
    entry = {"model": item.model, "kind": "llm", "node_ids": list(item.node_ids),
             "strategy": "" if info.get("strategy") == "solo" else info.get("strategy", ""),
             "gpu_memory_utilization": info.get("gpu_memory_utilization"),
             "max_model_len": info.get("max_model_len"),
             "extra_vllm_args": args}
    # Only a dtype the operator chose: the node default recorded as an entry
    # would switch off serve_args' vision-model safety rule (see capture).
    if spec.get("kv_cache_dtype"):
        entry["kv_cache_dtype"] = str(spec["kv_cache_dtype"])
    return entry


def plan_household(app, draft: dict) -> dict:
    """Plan a wizard draft: ``{"models": [...], "limits": {node_id: gb}}``."""
    config = app.get("config")
    head_id = str(getattr(config, "node_id", "") or "")
    nodes = household_nodes(app, (draft or {}).get("limits") or None)
    by_id = {n.node_id: n for n in nodes}

    items: List[Item] = []
    infos: List[dict] = []
    seen = set()
    for index, spec in enumerate((draft or {}).get("models") or []):
        spec = spec if isinstance(spec, dict) else {}
        kind = str(spec.get("kind") or "llm").lower()
        mode = str(spec.get("mode") or "auto").lower()
        item = Item(id=str(spec.get("id") or f"m{index + 1}"),
                    model=str(spec.get("model") or "").strip(),
                    kind=kind if kind in ("llm", "image", "embedding") else "llm",
                    node_ids=[str(n) for n in (spec.get("node_ids") or []) if n],
                    mode=mode if mode in MODES else "auto",
                    priority=max(0.1, float(spec.get("priority") or 1)))
        info: dict = {"warnings": []}
        if not item.model:
            item.errors.append("Kein Modell gewählt.")
        elif not item.node_ids:
            item.errors.append("Keinem Node zugeordnet.")
        else:
            unknown = [n for n in item.node_ids if n not in by_id]
            if unknown:
                item.errors.append(f"Unbekannte Nodes: {', '.join(unknown)}.")
            # A replica is the same model on another node (E4) — the same model
            # twice on one node is not something a node can serve.
            for node_id in item.node_ids:
                if (item.model, node_id) in seen:
                    item.errors.append(
                        f"{item.model} ist auf {by_id.get(node_id, HouseholdNode(node_id)).name or node_id} "
                        f"schon eingeplant — ein Replikat gehört auf einen anderen Node.")
                seen.add((item.model, node_id))
        if not item.errors:
            try:
                if item.kind == "llm":
                    info = _resolve_llm(app, spec, item, by_id, head_id)
                elif item.kind == "image":
                    info = _resolve_image(app, spec, item)
                else:
                    info = _resolve_embedding(app, spec, item)
            except Exception as exc:
                logger.exception("could not plan %s", item.model)
                item.errors.append(f"Planung fehlgeschlagen: {exc}")
        items.append(item)
        infos.append(info)

    conflicts = solve(items, nodes)

    models_out = []
    entries = []
    for spec, item, info in zip((draft or {}).get("models") or [], items, infos):
        out = {"id": item.id, "model": item.model, "kind": item.kind,
               "node_ids": item.node_ids, "mode": item.mode,
               "priority": item.priority, "errors": list(item.errors),
               "warnings": list(info.get("warnings") or []),
               "fixed_per_node_gb": round(item.fixed_gb, 1)}
        for key in ("strategy", "weights_gb", "weights_per_node_gb", "weights_source",
                    "overhead_gb", "kv_source", "bytes_per_token", "kv_cache_dtype",
                    "max_image_size"):
            if key in info:
                out[key] = info[key]
        if item.has_cache and info.get("per_token_node_gb"):
            per_token = info["per_token_node_gb"]
            tokens = int(item.cache_gb / per_token) if per_token else 0
            window = info["max_model_len"]
            if item.mode == "usage":
                sessions = info["sessions"]
            else:
                sessions = tokens // window if window else 0
                if sessions < 1 and not conflicts:
                    item.errors.append(
                        f"Der Cache hält {tokens:,} Tokens — weniger als eine "
                        f"Anfrage mit {window:,}. Fenster verkleinern oder "
                        f"anderen Modellen auf diesem Node Platz nehmen.")
                    out["errors"] = list(item.errors)
            reserve = item.fixed_gb - 0.01 * max(
                (by_id[n].total_gb for n in item.node_ids), default=0) + item.cache_gb
            tight = info.get("tightest_total_gb") or 0
            gmu = math.ceil(reserve / tight * 100) / 100 if tight else 0.0
            info.update(max_num_seqs=max(1, sessions), gpu_memory_utilization=gmu)
            out.update(cache_per_node_gb=round(item.cache_gb, 1),
                       kv_tokens=tokens, max_model_len=window,
                       sessions=max(0, sessions), max_num_seqs=max(1, sessions),
                       gpu_memory_utilization=gmu,
                       cache_cap_per_node_gb=(round(item.cap_gb, 1)
                                              if item.cap_gb < math.inf else None))
        models_out.append(out)
        entry = _entry_for(spec, item, info)
        if entry is not None:
            entries.append(entry)

    nodes_out = []
    for node in nodes:
        segments = []
        used = 0.0
        for item in items:
            if item.errors or node.node_id not in item.node_ids:
                continue
            cache = item.cache_gb if item.has_cache else 0.0
            segments.append({"id": item.id, "model": item.model, "kind": item.kind,
                             "fixed_gb": round(item.fixed_gb, 1),
                             "cache_gb": round(cache, 1)})
            used += item.fixed_gb + cache
        nodes_out.append({
            "node_id": node.node_id, "name": node.name, "online": node.online,
            "total_gb": node.total_gb, "limit_gb": node.limit_gb,
            "baseline_gb": node.baseline_gb,
            "baseline_estimated": node.baseline_estimated,
            "budget_gb": node.budget_gb, "used_gb": round(used, 1),
            "free_gb": round(node.budget_gb - used, 1),
            "over": used > node.budget_gb + 0.05, "segments": segments})

    return {"nodes": nodes_out, "models": models_out, "conflicts": conflicts,
            "ok": not conflicts and all(not m["errors"] for m in models_out),
            "entries": entries, "head_node_id": head_id}
