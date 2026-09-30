"""Writing down what a launch did, while it is happening.

Everything here is derived from state the node already keeps: the load-phase
tracker's timings, the host memory reading the guard takes every two seconds,
and the request counters the metrics collector has been keeping all along.
Nothing new is measured — it is only that nobody was writing it down.

Driven from the cluster sync loop, which runs every five seconds on every
node whether or not telemetry is configured. A measurement that only existed
when MQTT was set up would be missing from exactly the deployments that most
need it.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["Recorder"]

_TERMINAL = ("ready", "failed")


def _engine_report(backend, record=None) -> dict:
    """The engine's own memory figures, from the log it just wrote.

    Best effort and late: only on a successful load, and only from whatever the
    backend will hand over. A log that says nothing produces {}, and the
    host-level measurement stands alone — a partial answer must not overwrite a
    whole one.
    """
    reader = getattr(backend, "logs", None)
    if not callable(reader):
        return {}
    try:
        from ainode.measure.engine_report import parse_engine_report

        report = parse_engine_report(reader(4000))
        # The split the weights figure belongs to. A weights-per-node number
        # without it cannot be compared to anything.
        if report.get("weights_gb"):
            ranks = max(1, int(getattr(record, "tensor_parallel_size", 1) or 1)
                        * int(getattr(record, "pipeline_parallel_size", 1) or 1))
            report["rank_count"] = ranks
        # The dtype the cache figures belong to, written with them so the two
        # can never come from different launches.
        if report.get("kv_tokens"):
            report["kv_cache_dtype"] = launched_kv_dtype(getattr(backend, "config", None))
        return report
    except Exception:
        logger.debug("could not read the engine's memory report", exc_info=True)
        return {}


def launched_kv_dtype(config) -> str:
    """The KV dtype vLLM was given, by the launch's own rule
    (backends/eugr.py): a --kv-cache-dtype in the extra args — the operator's
    or the recipe's — suppresses the built-in one; otherwise the node's
    setting, with fp8 turned into auto for a vision model unless asked for.

    It used to be the node's setting alone. unsloth/Qwen3.8-27B-NVFP4 ran
    with the recipe's --kv-cache-dtype auto and was recorded as fp8, so the
    planner, planning it at auto, never used its measured cost per token.
    """
    if config is None:
        return "auto"
    flag = _flag(getattr(config, "extra_vllm_args", None), "--kv-cache-dtype")
    if flag:
        return str(flag)
    try:
        from ainode.engine.serve_args import effective_kv_cache_dtype, local_model_dir

        directory = local_model_dir(str(getattr(config, "model", "") or ""),
                                    str(getattr(config, "models_dir", "") or ""))
        return str(effective_kv_cache_dtype(config, directory) or "auto")
    except Exception:
        return str(getattr(config, "kv_cache_dtype", "") or "auto")


def _flag(args, name):
    args = [str(a) for a in (args or [])]
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _launch_of(record, config) -> dict:
    """How this instance was started, in the terms a plan uses."""
    if config is None:
        return {}
    args = [str(a) for a in (getattr(config, "extra_vllm_args", None) or [])]
    tp = int(getattr(record, "tensor_parallel_size", 0) or
             getattr(config, "tensor_parallel_size", 1) or 1)
    pp = int(getattr(record, "pipeline_parallel_size", 0) or
             getattr(config, "pipeline_parallel_size", 1) or 1)
    out = {
        "engine_backend": str(getattr(config, "engine_backend", "") or "eugr"),
        "strategy": str(getattr(config, "parallel_strategy", "") or
                        ("solo" if tp * pp == 1 else "")),
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "data_parallel_size": int(getattr(record, "data_parallel_size", 0) or
                                  getattr(config, "data_parallel_size", 1) or 1),
        "nodes": 1 + len(list(getattr(record, "peer_ips", None) or [])),
        "max_model_len": int(getattr(config, "max_model_len", 0) or
                             int(_flag(args, "--max-model-len") or 0)),
        "kv_cache_dtype": launched_kv_dtype(config),
        "gpu_memory_utilization": float(getattr(config, "gpu_memory_utilization", 0) or 0),
        "max_num_seqs": int(_flag(args, "--max-num-seqs") or 0),
        "quantization": str(getattr(config, "quantization", "") or ""),
        "extra_vllm_args": args,
    }
    if out["engine_backend"] == "diffusers":
        for field in ("max_image_size", "image_steps", "image_size", "image_dtype"):
            value = getattr(config, field, None)
            if value:
                out[field] = value
    return out


class Recorder:
    """Turns instance state changes into measurements."""

    def __init__(self, app):
        self._app = app
        self._phase: Dict[str, str] = {}
        #: Host memory free when each model's load was first seen, so the
        #: cost of the load is a subtraction rather than a guess.
        self._baseline: Dict[str, float] = {}
        #: The same for the peers of a distributed load: memory IN USE per
        #: peer, from their announcements.
        self._peer_baseline: Dict[str, Dict[str, float]] = {}
        self._store = None
        #: {model: (where it runs, requests, tokens, seconds)} when this
        #: launch of it was first seen serving: its speed is counted from here.
        self._speed_base: Dict[str, tuple] = {}
        self._speed_sent: Dict[str, tuple] = {}
        #: Filled by poll(), sent by flush(): (node_id, body) and {model: port}.
        self._outbox: list = []
        self._want_version: Dict[str, int] = {}

    @property
    def store(self):
        if self._store is None:
            from ainode.measure.store import MeasurementStore

            self._store = self._app.get("measurement_store") or MeasurementStore()
            self._app["measurement_store"] = self._store
        return self._store

    # -- the loop calls this ------------------------------------------------

    def poll(self) -> None:
        """Never raises: a bookkeeping failure must not disturb a node."""
        try:
            self._poll()
        except Exception:
            logger.debug("could not record measurements", exc_info=True)

    def _poll(self) -> None:
        available = self._host_available_gb()
        seen = set()
        for model, instance in self._instances():
            seen.add(model)
            backend = instance.backend
            phase = str(getattr(backend, "load_phase", "") or "")
            previous = self._phase.get(model)
            self._phase[model] = phase

            if previous is None:
                # First sight. If it is still loading, remember what the node
                # had free before it finished — that is the only moment the
                # baseline can be taken.
                if phase not in _TERMINAL and available:
                    self._baseline.setdefault(model, available)
                if phase not in _TERMINAL:
                    peers = self._peer_used_gb(instance)
                    if peers:
                        self._peer_baseline.setdefault(model, peers)
                continue
            if phase == previous or phase not in _TERMINAL:
                continue
            self._record(model, instance, phase, available)

        for model in [m for m in self._phase if m not in seen]:
            del self._phase[model]
            self._baseline.pop(model, None)
            self._peer_baseline.pop(model, None)

        self._record_speeds()

    # -- writing ------------------------------------------------------------

    def _record(self, model: str, instance, phase: str,
                available: float) -> None:
        backend = instance.backend
        record = getattr(instance, "record", None)
        config = getattr(backend, "config", None)
        baseline = self._baseline.pop(model, 0.0)
        cost = 0.0
        if phase == "ready" and baseline and available:
            cost = max(0.0, baseline - available)
        by_node: Dict[str, float] = {}
        peer_before = self._peer_baseline.pop(model, {})
        if phase == "ready" and peer_before:
            own = str(getattr(self._app.get("config"), "node_id", "") or "")
            if cost:
                by_node[own or "head"] = round(cost, 1)
            after = self._peer_used_gb(instance)
            for node_id, before in peer_before.items():
                if node_id in after:
                    by_node[node_id] = round(max(0.0, after[node_id] - before), 1)
            # The tightest node is the one that counts: a plan is per node, and
            # the next launch fits only if it fits on the fullest one.
            if by_node:
                cost = max(by_node.values())

        report = (self._with_checkpoint(model, _engine_report(backend, record))
                  if phase == "ready" else None)
        timeline = list(getattr(backend, "load_timeline", None) or [])
        seconds = sum(e.get("seconds", 0) for e in timeline) or float(
            getattr(backend, "load_seconds", 0) or 0)

        self.store.record_launch(
            model,
            ok=(phase == "ready"),
            kind=str(getattr(record, "kind", "") or "llm") or "llm",
            node_id=str(getattr(self._app.get("config"), "node_id", "") or ""),
            engine_backend=str(getattr(config, "engine_backend", "") or ""),
            load_seconds=seconds,
            load_timeline=timeline,
            memory_gb=cost,
            memory_by_node=by_node,
            launch=_launch_of(record, config),
            error=str(getattr(backend, "load_error", "") or "") if phase != "ready" else "",
            node_total_gb=self._node_total_gb(),
            max_model_len=int(getattr(config, "max_model_len", 0) or 0),
            gpu_memory_utilization=float(
                getattr(config, "gpu_memory_utilization", 0) or 0),
            max_image_size=int(getattr(config, "max_image_size", 0) or 0)
            if str(getattr(config, "engine_backend", "")) == "diffusers" else 0,
            engine_report=report,
        )
        if report is not None and not report.get("engine_version"):
            # Not in the log (it prints the version once, at the start, and a
            # long distributed log has scrolled past it): ask the engine.
            port = int(getattr(config, "api_port", 0) or 0)
            if port and str(getattr(config, "engine_backend", "") or "") != "diffusers":
                self._want_version[model] = port

    def _with_checkpoint(self, model: str, report: dict) -> dict:
        """Add the checkpoint's on-disk size and MoE-ness to a report that
        has a weights figure, so the two are kept side by side."""
        if not report.get("weights_gb"):
            return report
        manager = self._app.get("model_manager")
        if manager is None:
            return report
        try:
            from ainode.planner.facts import local_facts

            facts = local_facts(manager, model)
            if facts.weights_gb:
                report["disk_weights_gb"] = round(facts.weights_gb, 1)
                report["is_moe"] = bool(facts.is_moe)
        except Exception:
            logger.debug("could not read %s's checkpoint", model,
                         exc_info=True)
        return report

    def _record_speeds(self) -> None:
        """How fast each model answers, from what it has served since this
        launch of it started.

        Requests are counted where they arrive — the head — and the counters
        run for the life of that AINode process. So the speed is the
        difference since the model was last seen coming up, not the lifetime
        average (which blended every launch of it, MTP on and off alike); and
        it goes to the node that runs the model. It used to be written into
        the head's own entry for the model — an older launch of it, where the
        head had taken part — while the entry of the launch that served was
        left at 0: nvidia/Qwen3.8-Flash-Next-NVFP4 on spark-1432 and
        spark-659b.

        Only with enough requests behind it — a speed from three is noise.
        An image model has no tokens: its speed is seconds per picture.
        """
        collector = self._app.get("metrics_collector")
        if collector is None:
            return
        try:
            stats = collector.model_stats() or {}
        except Exception:
            return
        kinds = self._kinds()
        where = self._serving_where()
        now = time.time()
        for model in [m for m in self._speed_base if m not in where]:
            del self._speed_base[model]          # gone: its next launch starts afresh
        for model, entry in stats.items():
            if not isinstance(entry, dict) or model not in where:
                continue
            requests = int(entry.get("requests") or 0)
            tokens = float(entry.get("tokens_generated") or 0)
            seconds = float(entry.get("avg_latency_ms") or 0) * requests / 1000.0
            base = self._speed_base.get(model)
            if base is None or base[0] != where[model]:
                self._speed_base[model] = (where[model], requests, tokens, seconds)
                continue
            d_req, d_tok, d_sec = requests - base[1], tokens - base[2], seconds - base[3]
            if d_req < 5 or d_sec <= 0:
                continue
            if kinds.get(model) == "image":
                payload = {"seconds_per_image": round(d_sec / d_req, 2)}
            elif d_tok > 0:
                payload = {"tokens_per_second": round(d_tok / d_sec, 1)}
            else:
                continue
            node_id = where[model][0]
            try:
                if node_id == self._own_id():
                    self.store.record_speed(model, **payload)
                    continue
                sent = self._speed_sent.get(model)
                # To a peer: when it moved, and otherwise once a minute.
                if sent and sent[1] == payload and now - sent[0] < 60:
                    continue
                self._outbox.append((node_id, {"model": model, **payload}))
                self._speed_sent[model] = (now, payload)
            except Exception:
                logger.debug("could not record speed for %s", model, exc_info=True)

    def _own_id(self) -> str:
        return str(getattr(self._app.get("config"), "node_id", "") or "")

    def _serving_where(self) -> Dict[str, tuple]:
        """{model: (node that runs it, launch identity)} for every model that
        serves now — here, or on the node that leads it."""
        out: Dict[str, tuple] = {}
        own = self._own_id()
        cluster = self._app.get("cluster_state")
        try:
            for node in (cluster.members() if cluster is not None else []):
                if node.node_id == own:
                    continue
                for inst in (getattr(node, "instances", None) or []):
                    if not isinstance(inst, dict) or not inst.get("model"):
                        continue
                    if str(inst.get("status") or "") != "serving":
                        continue
                    leader = str(inst.get("head_node_id") or "") or node.node_id
                    if leader != node.node_id:
                        continue                 # announced by a member, led elsewhere
                    # Not load_seconds: on a serving instance that is the time
                    # since it started, so it changed every poll and every
                    # poll began the count again — no speed was ever written.
                    # A relaunch shows as the model leaving "serving" in
                    # between, which drops its base (_record_speeds).
                    out[inst["model"]] = (node.node_id, str(inst.get("instance_id") or ""))
        except Exception:
            logger.debug("could not read the cluster's instances", exc_info=True)
        for model, instance in self._instances():
            backend = instance.backend
            if str(getattr(backend, "load_phase", "") or "") != "ready":
                continue
            record = getattr(instance, "record", None)
            out[model] = (own, str(getattr(record, "instance_id", "") or ""))
        return out

    async def flush(self) -> None:
        """The network half of a poll: speeds to the nodes that run the
        models, and each new launch's engine version from the engine."""
        session = self._app.get("client_session")
        outbox, self._outbox = self._outbox, []
        wanted, self._want_version = self._want_version, {}
        if session is None:
            return
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=5)
        cluster = self._app.get("cluster_state")
        nodes = {n.node_id: n for n in (cluster.members() if cluster is not None else [])}
        for node_id, body in outbox:
            node = nodes.get(node_id)
            host = str(getattr(node, "fabric_ip", "") or "").strip()
            if not host:
                continue
            url = f"http://{host}:{getattr(node, 'web_port', 3000)}/api/measurements/speed"
            try:
                async with session.post(url, json=body, timeout=timeout) as resp:
                    await resp.read()
            except Exception:
                logger.debug("could not send %s's speed to %s", body.get("model"),
                             node_id, exc_info=True)
        for model, port in wanted.items():
            try:
                async with session.get(f"http://127.0.0.1:{port}/version",
                                       timeout=timeout) as resp:
                    data = await resp.json(content_type=None) if resp.status == 200 else {}
                version = str((data or {}).get("version") or "")
                if version:
                    self.store.record_engine_version(model, version)
            except Exception:
                logger.debug("could not ask %s's engine for its version", model,
                             exc_info=True)

    def _kinds(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        # What the other nodes run, too: requests are counted where they
        # arrive — usually the head — and an image model running on a peer
        # was taken for an LLM here, so its seconds per image were never
        # written down.
        cluster = self._app.get("cluster_state")
        try:
            for node in (cluster.members() if cluster is not None else []):
                for inst in (getattr(node, "instances", None) or []):
                    if isinstance(inst, dict) and inst.get("model"):
                        out[inst["model"]] = str(inst.get("kind") or "llm")
        except Exception:
            logger.debug("could not read the cluster's instances", exc_info=True)
        for model, instance in self._instances():
            record = getattr(instance, "record", None)
            out[model] = str(getattr(record, "kind", "") or "llm")
        return out

    def _node_total_gb(self) -> float:
        collector = self._app.get("metrics_collector")
        try:
            from ainode.core.units import gb_from_mib

            metrics = collector.get_gpu_metrics() if collector is not None else {}
            return gb_from_mib((metrics or {}).get("memory_total_mb") or 0)
        except Exception:
            return 0.0

    # -- reading ------------------------------------------------------------

    def _instances(self):
        manager = self._app.get("instances")
        out = []
        for instance in (manager.instances() if manager is not None else []):
            record = getattr(instance, "record", None)
            model = str(getattr(record, "model", "") or "")
            if model and getattr(instance, "backend", None) is not None:
                out.append((model, instance))
        return out

    def _host_available_gb(self) -> float:
        """Free host memory now, from the guard if it is running.

        The guard reads /proc/meminfo every two seconds anyway; asking it
        costs nothing and keeps one definition of "free" in the process.
        """
        # Decimal GB, like everything this is compared with: the weights and
        # cache figures from the engine report and the plan's footprint. The
        # guard's MB are MiB, and dividing them by 1024 gave GiB — a footprint
        # read 7% below the plan it was set beside.
        from ainode.core.units import gb_from_mib

        guard = self._app.get("memory_guard")
        if guard is not None:
            try:
                reading = guard.read()
                if reading.readable and reading.available_mb:
                    return gb_from_mib(reading.available_mb)
            except Exception:
                logger.debug("could not read host memory", exc_info=True)
        from ainode.safety.memory_guard import host_available_mb

        value = host_available_mb()
        return gb_from_mib(value) if value else 0.0

    def _peer_used_gb(self, instance) -> Dict[str, float]:
        """{node id: GB in use} for the peers this instance runs on, from
        their own announcements (P6).

        The head measured only itself, and in a distributed launch the peer
        is often the tighter node — the one whose figure decides whether the
        next launch fits. Each node broadcasts its memory in use every few
        seconds; before and after the load, that is the peer's cost.
        """
        from ainode.core.units import gb_from_mib

        record = getattr(instance, "record", None)
        peers = {str(ip) for ip in (getattr(record, "peer_ips", None) or [])
                 if ip}
        cluster = self._app.get("cluster_state")
        if not peers or cluster is None:
            return {}
        out: Dict[str, float] = {}
        try:
            members = cluster.members()
        except Exception:
            return {}
        for node in members:
            addresses = {str(getattr(node, "fabric_ip", "") or ""),
                         str(getattr(node, "peer_ip", "") or "")}
            addresses.update(str(a) for a in (getattr(node, "ib_ips", None)
                                               or []))
            if not (addresses & peers):
                continue
            used = float(getattr(node, "gpu_memory_used_mb", 0) or 0)
            if used > 0:
                out[str(node.node_id)] = gb_from_mib(used)
        return out


def guard_stopped_for(app, model: str) -> Optional[dict]:
    """The record of the guard stopping this model, if it is the last word.

    Separate from :func:`measured_for`, which answers only for models that
    have actually served — a model that has never come up here has no
    measurement, and being killed by the guard is precisely the case where
    there is something to say about a model that never served.

    None once the model has run successfully since: whatever was in the way
    is evidently no longer there.
    """
    try:
        from ainode.measure.store import MeasurementStore

        store = app.get("measurement_store") or MeasurementStore()
        entry = store.get(model)
    except Exception:
        return None
    if entry is None or not entry.guard_stops:
        return None
    if entry.last_ok and entry.last_ok > entry.last_guard_stop:
        return None
    return entry.to_dict()


def measured_for(app, model: str) -> Optional[dict]:
    """The measurement for one model on this node, or None."""
    try:
        from ainode.measure.store import MeasurementStore

        store = app.get("measurement_store") or MeasurementStore()
        entry = store.get(model)
    except Exception:
        return None
    return entry.to_dict() if entry is not None and entry.measured else None
