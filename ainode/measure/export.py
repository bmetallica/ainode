"""Everything this cluster has measured, in one file to hand on.

Asked for by the operator, to give back the numbers the planner is judged by:

    ein Skript, womit ich die gemessenen Daten der bereits gelaufenen Modelle
    exportieren kann … evtl. auch direkt mit deren Startparametern … so dass
    du auf deren Basis besser einschätzen kannst, welche Modelle noch laufen
    könnten oder an welchen Stellen wir optimieren können

So the export carries, per measured model and node: what it took (host drop,
per node, the engine's own weights and cache figures, tokens, speed, load
time and phases, guard stops, the last few loads) and how it was started;
next to it the checkpoint's facts and what the planner's arithmetic predicts
for exactly that launch — the difference is the calibration. Around it: the
nodes (memory, limit, idle use, build), the engine build, what runs now, and
the profiles.

Nothing secret goes in: no credentials, API keys, the cluster key or the web
password; model ids, node names and flags only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__all__ = ["build_export", "EXPORT_VERSION"]

EXPORT_VERSION = 1


def _engine_build() -> Dict[str, str]:
    try:
        from ainode.core.config import AINODE_HOME

        out = {}
        for line in (AINODE_HOME / "engine-build.env").read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip()
        return out
    except (OSError, ImportError):
        return {}


def _nodes(app) -> List[Dict[str, Any]]:
    from ainode.core.units import gb_from_mib, node_total_gb

    cluster = app.get("cluster_state")
    config = app.get("config")
    own = str(getattr(config, "node_id", "") or "")
    out = []
    for node in (cluster.members() if cluster is not None else []):
        status = node.status.value if hasattr(node.status, "value") else str(node.status)
        limit = float((getattr(config, "memory_limit_gb", 0) if node.node_id == own
                       else getattr(node, "memory_limit_gb", 0)) or 0)
        baseline = float(getattr(node, "baseline_used_mb", 0) or 0)
        out.append({
            "node_id": node.node_id, "node_name": getattr(node, "node_name", ""),
            "head": node.node_id == own, "status": status,
            "gpu_name": getattr(node, "gpu_name", ""),
            "unified_memory": bool(getattr(node, "unified_memory", False)),
            "total_gb": round(node_total_gb(node), 1),
            "used_gb_now": round(gb_from_mib(getattr(node, "gpu_memory_used_mb", 0) or 0), 1),
            "memory_limit_gb": limit,
            "idle_use_gb": round(gb_from_mib(baseline), 1) if baseline else None,
            "version": str(getattr(node, "version", "") or ""),
        })
    return out


def _facts(app, model: str) -> Dict[str, Any]:
    manager = app.get("model_manager")
    if manager is None:
        return {}
    try:
        from ainode.planner.facts import local_facts

        facts = local_facts(manager, model)
    except Exception:
        return {}
    if not facts.weight_bytes:
        return {}
    return {
        "architecture": facts.architecture, "torch_dtype": facts.torch_dtype,
        "quantization": facts.quantization, "weights_on_disk_gb": round(facts.weights_gb, 2),
        "num_layers": facts.num_layers, "attention_layers": facts.attention_layers,
        "num_attention_heads": facts.num_attention_heads,
        "num_kv_heads": facts.num_kv_heads, "head_dim": facts.head_dim,
        "kv_lora_rank": getattr(facts, "kv_lora_rank", 0),
        "qk_rope_head_dim": getattr(facts, "qk_rope_head_dim", 0),
        "max_position_embeddings": facts.max_position_embeddings,
        "is_moe": facts.is_moe, "num_experts": facts.num_experts,
        "is_hybrid": facts.is_hybrid, "unknown": list(facts.unknown),
    }


def _prediction(app, model: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """What the planner's arithmetic says about this exact launch, beside
    what was measured — before any measurement of the model is used."""
    manager = app.get("model_manager")
    launch = row.get("launch") or {}
    ranks = int(row.get("rank_count") or 0) or int(
        (launch.get("tensor_parallel_size") or 1) * (launch.get("pipeline_parallel_size") or 1))
    if manager is None or not ranks:
        return {}
    try:
        from ainode.planner.compute import kv_bytes_per_token, weights_per_node
        from ainode.planner.facts import local_facts

        facts = local_facts(manager, model)
        if not facts.weight_bytes:
            return {}
        strategy = "tensor" if (launch.get("tensor_parallel_size") or 1) > 1 else (
            "pipeline" if (launch.get("pipeline_parallel_size") or 1) > 1 else "solo")
        estimate, _ = weights_per_node(facts, ranks, strategy)
        dtype = row.get("kv_cache_dtype") or launch.get("kv_cache_dtype") or "auto"
        formula_bpt = kv_bytes_per_token(facts, dtype)
        out: Dict[str, Any] = {
            "ranks": ranks, "strategy": strategy, "kv_cache_dtype": dtype,
            "weights_per_node_gb_estimated": round(estimate, 2),
            "kv_bytes_per_token_formula": int(formula_bpt),
        }
        measured_weights = float(row.get("weights_gb") or 0)
        if measured_weights:
            out["weights_per_node_gb_measured"] = measured_weights
            out["weights_error_percent"] = round((estimate - measured_weights)
                                                 / measured_weights * 100, 1)
        tokens = int(row.get("kv_tokens") or 0)
        cache = float(row.get("kv_cache_gb") or 0)
        if tokens and cache:
            measured_bpt = int(cache * ranks * 1e9 / tokens)
            out["kv_bytes_per_token_measured"] = measured_bpt
            if formula_bpt:
                out["kv_bytes_per_token_error_percent"] = round(
                    (formula_bpt - measured_bpt) / measured_bpt * 100, 1)
        footprint = float(row.get("memory_gb") or 0)
        if footprint and measured_weights and cache:
            out["engine_overhead_gb_measured"] = round(footprint - measured_weights - cache, 2)
        return out
    except Exception:
        logger.debug("could not predict %s", model, exc_info=True)
        return {}


def _running_and_profiles(app) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from ainode.profiles.apply import capture_profile

        out["running"] = [e.to_dict() for e in capture_profile(app, "running-now").entries]
    except Exception:
        logger.debug("could not capture what runs", exc_info=True)
        out["running"] = []
    store = app.get("profiles")
    try:
        if store is None:
            from ainode.profiles.store import ProfileStore

            store = ProfileStore()
        out["profiles"] = [p.to_dict() for p in store.all()]
        out["default_profile"] = store.default_name
    except Exception:
        logger.debug("could not read the profiles", exc_info=True)
        out["profiles"] = []
    return out


async def build_export(app) -> Dict[str, Any]:
    from ainode.measure.api_routes import gather_cluster_measurements

    config = app.get("config")
    loop = asyncio.get_event_loop()
    by_model = await gather_cluster_measurements(app)

    def _blocking() -> Dict[str, Any]:
        models = []
        for model in sorted(by_model):
            rows = []
            for row in by_model[model]:
                row = dict(row)
                prediction = _prediction(app, model, row) if row.get("kind", "llm") == "llm" else {}
                if prediction:
                    row["planner"] = prediction
                rows.append(row)
            models.append({"model": model, "facts": _facts(app, model), "measurements": rows})
        return {"models": models, **_running_and_profiles(app)}

    body = await loop.run_in_executor(None, _blocking)
    try:
        from ainode import __version__ as version
    except Exception:
        version = ""
    return {
        "export_version": EXPORT_VERSION,
        "exported_at": time.time(),
        "ainode_version": version,
        "head_node_id": str(getattr(config, "node_id", "") or ""),
        "engine_build": _engine_build(),
        "planner_constants": _planner_constants(),
        "nodes": _nodes(app),
        **body,
    }


def _planner_constants() -> Dict[str, float]:
    from ainode.planner import compute

    return {name: getattr(compute, name) for name in (
        "ENGINE_OVERHEAD_GB", "COMM_OVERHEAD_GB", "TP_REPLICATION", "SYSTEM_RESERVE_GB")
        if hasattr(compute, name)}
