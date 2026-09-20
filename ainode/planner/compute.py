"""Does it fit, how should it be split, and what will it hold.

Four questions get asked before every launch on this cluster, and until now
each was answered by a person with a calculator and the checkpoint's
config.json open:

  * does the model fit at all, and on how many nodes;
  * which parallel axis — tensor where the node count allows it, pipeline
    where it does not, neither when one node is enough;
  * what --max-model-len the leftover memory can actually back;
  * how many people can use it at once at that context length.

All four come out of the same arithmetic, and none of it is guesswork: the
weights are the bytes on disk, the KV cost per token is fixed by the layer,
head and dimension counts in the config, and the free memory is what the
nodes report. Where a figure cannot be computed the plan says so rather than
substituting a plausible one — an estimate presented as a plan is worse than
no plan, because it is acted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ainode.planner.facts import ModelFacts

__all__ = ["NodeBudget", "Plan", "kv_bytes_per_token", "plan_for"]

#: CUDA context, the engine itself, activation buffers and the captured graphs.
#: Measured on this hardware as roughly this much above the weights, and it
#: does not scale with the model.
ENGINE_OVERHEAD_GB = 2.5

#: NCCL buffers and the extra staging a multi-rank launch needs, per node.
COMM_OVERHEAD_GB = 0.5

#: Tensor parallelism splits the layers but not the embedding table or the
#: norms, so a rank holds slightly more than its share.
TP_REPLICATION = 1.05

#: Never plan into the last of a node's memory. On unified-memory hardware
#: this pool is also the operating system's, and a launch that takes all of it
#: does not fail cleanly — it takes the node with it.
SYSTEM_RESERVE_GB = 4.0

#: Tensor parallelism splits attention heads, and head counts are powers of
#: two. TP=3 has no models behind it.
TENSOR_SIZES = (1, 2, 4, 8)

#: A context window is only useful in round numbers, and vLLM pages the cache
#: in blocks. Anything the planner recommends is rounded down to this.
LEN_GRANULARITY = 4096


@dataclass
class NodeBudget:
    """One node, as the planner sees it."""

    node_id: str
    name: str = ""
    total_gb: float = 0.0
    free_gb: float = 0.0

    @property
    def usable_gb(self) -> float:
        return max(0.0, self.free_gb - SYSTEM_RESERVE_GB)


@dataclass
class Plan:
    model: str = ""
    fits: bool = False
    node_ids: List[str] = field(default_factory=list)
    strategy: str = ""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.0
    max_model_len: int = 0
    #: What the whole cluster can hold at once, across every rank.
    kv_tokens: int = 0
    kv_bytes_per_token: int = 0
    kv_gb: float = 0.0
    weights_gb: float = 0.0
    weights_per_node_gb: float = 0.0
    concurrent_requests: int = 0
    #: The arithmetic, in the order it was done. This is the point: a number
    #: an operator cannot check is a number they cannot overrule.
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Why it does not fit, when it does not.
    blocker: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "fits": self.fits,
            "node_ids": list(self.node_ids),
            "strategy": self.strategy,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": round(self.gpu_memory_utilization, 2),
            "max_model_len": self.max_model_len,
            "kv_tokens": self.kv_tokens,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_gb": round(self.kv_gb, 1),
            "weights_gb": round(self.weights_gb, 1),
            "weights_per_node_gb": round(self.weights_per_node_gb, 1),
            "concurrent_requests": self.concurrent_requests,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "blocker": self.blocker,
        }


def kv_bytes_per_token(facts: ModelFacts, kv_cache_dtype: str = "auto") -> int:
    """Bytes of KV cache one token costs across the whole model.

    Two formulas, because there are two kinds of attention in service here.
    Ordinary grouped-query attention caches a key and a value per KV head per
    layer. Multi-head latent attention — DeepSeek's — caches one compressed
    vector per layer instead, which is why a model of that size can hold a
    million tokens where a conventional one holds a tenth of that.
    """
    dtype_bytes = 1 if str(kv_cache_dtype).lower().startswith("fp8") \
        else facts.dtype_bytes
    layers = facts.attention_layers or facts.num_layers
    if not layers:
        return 0
    if facts.kv_lora_rank:
        per_layer = facts.kv_lora_rank + facts.qk_rope_head_dim
        return int(layers * per_layer * dtype_bytes)
    if not (facts.num_kv_heads and facts.head_dim):
        return 0
    return int(2 * layers * facts.num_kv_heads * facts.head_dim * dtype_bytes)


def _round_len(tokens: int) -> int:
    return max(0, int(tokens) // LEN_GRANULARITY * LEN_GRANULARITY)


def _tensor_ok(facts: ModelFacts, size: int) -> Optional[str]:
    """Why this tensor-parallel size is impossible, or None."""
    if size not in TENSOR_SIZES:
        return (f"tensor parallelism needs 2, 4 or 8 ranks — no model splits "
                f"attention heads {size} ways")
    if size == 1:
        return None
    if facts.num_attention_heads and facts.num_attention_heads % size:
        return (f"{facts.num_attention_heads} attention heads do not divide "
                f"by {size}")
    if facts.num_kv_heads and facts.num_kv_heads % size and \
            size % max(1, facts.num_kv_heads):
        # vLLM either splits the KV heads or replicates them; it can do
        # neither when the counts are coprime.
        return (f"{facts.num_kv_heads} KV heads can be neither split across "
                f"nor replicated over {size} ranks")
    return None


@dataclass
class _Candidate:
    nodes: List[NodeBudget]
    strategy: str
    tp: int
    pp: int
    weights_per_node: float
    kv_gb: float
    tokens: int


def _evaluate(facts: ModelFacts, nodes: Sequence[NodeBudget], strategy: str,
              bytes_per_token: int) -> Optional[_Candidate]:
    count = len(nodes)
    weights = facts.weights_gb
    if strategy == "tensor":
        per_node = weights / count * (TP_REPLICATION if count > 1 else 1.0)
        tp, pp = count, 1
    else:
        per_node = weights / count
        tp, pp = 1, count
    overhead = ENGINE_OVERHEAD_GB + (COMM_OVERHEAD_GB if count > 1 else 0.0)

    # The tightest node decides: a rank cannot borrow memory from its peers.
    smallest = min(n.usable_gb for n in nodes)
    kv_per_node = smallest - per_node - overhead
    if kv_per_node <= 0:
        return None
    kv_gb = kv_per_node * count
    tokens = int(kv_gb * 1e9 / bytes_per_token) if bytes_per_token else 0
    return _Candidate(list(nodes), strategy, tp, pp, per_node, kv_gb, tokens)


def plan_for(facts: ModelFacts, nodes: Sequence[NodeBudget], *,
             strategy: str = "auto",
             max_model_len: int = 0,
             kv_cache_dtype: str = "auto",
             concurrency: int = 1,
             supports_pipeline: bool = True,
             recipe_context: int = 0,
             recommended_gmu: float = 0.0) -> Plan:
    """The launch this model should get on these nodes."""
    plan = Plan(model=facts.repo)
    nodes = [n for n in nodes if n.total_gb > 0 or n.free_gb > 0]
    if not nodes:
        plan.blocker = "no nodes reported any memory"
        return plan
    if not facts.weight_bytes:
        plan.blocker = (f"{facts.repo or 'the model'} is not on this node's disk, "
                        f"so its real size is unknown. Download it first — a "
                        f"plan from the parameter count in the name is off by "
                        f"between two and eight times on a quantised "
                        f"checkpoint.")
        return plan

    bytes_per_token = kv_bytes_per_token(facts, kv_cache_dtype)
    plan.kv_bytes_per_token = bytes_per_token
    plan.weights_gb = facts.weights_gb
    if not bytes_per_token:
        plan.warnings.append(
            "The checkpoint does not state its layer or head counts, so the "
            "KV cache cannot be computed. Only the fit is planned below; "
            "--max-model-len is left to the engine.")

    ceiling = facts.max_position_embeddings or recipe_context or 0
    wanted_len = max_model_len or recipe_context or ceiling
    if ceiling and wanted_len > ceiling:
        plan.warnings.append(
            f"{wanted_len} exceeds the {ceiling} this checkpoint was trained "
            f"for; planning at {ceiling}.")
        wanted_len = ceiling

    axes = ["tensor", "pipeline"] if strategy == "auto" else [strategy]
    wanted_total = wanted_len * max(1, concurrency)
    best: Optional[_Candidate] = None
    # Every split the weights fit into, whether or not it backs the context
    # that was asked for. A model that fits but only at a shorter window is a
    # plan with a smaller number in it, not a refusal.
    viable: List[_Candidate] = []
    refusals: List[str] = []

    for count in range(1, len(nodes) + 1):
        chosen = sorted(nodes, key=lambda n: n.usable_gb, reverse=True)[:count]
        for axis in axes:
            if axis == "tensor":
                why = _tensor_ok(facts, count)
                if why:
                    refusals.append(f"{count} node(s), tensor: {why}")
                    continue
            else:
                if count == 1:
                    continue
                if not supports_pipeline:
                    refusals.append(
                        f"{count} node(s), pipeline: this architecture does not "
                        f"implement pipeline parallelism in vLLM")
                    continue
                if facts.num_layers and count > facts.num_layers:
                    refusals.append(
                        f"{count} node(s), pipeline: only {facts.num_layers} layers")
                    continue
            candidate = _evaluate(facts, chosen, axis, bytes_per_token)
            if candidate is None:
                refusals.append(
                    f"{count} node(s), {axis}: the weights plus engine overhead "
                    f"do not leave room on the smallest node")
                continue
            viable.append(candidate)
            if bytes_per_token and wanted_total and candidate.tokens < wanted_total:
                refusals.append(
                    f"{count} node(s), {axis}: holds {candidate.tokens:,} "
                    f"tokens, short of {wanted_len:,} × {max(1, concurrency)}")
                continue
            best = candidate
            break
        if best is not None:
            break

    short = False
    if best is None and viable:
        # Nothing reaches the requested window. The largest cache is still the
        # right answer; the window comes down to meet it, and the plan says so.
        best = max(viable, key=lambda c: c.tokens)
        short = True
    if best is None:
        plan.blocker = _blocker(facts, nodes, refusals)
        plan.notes = refusals
        return plan

    plan.fits = True
    plan.node_ids = [n.node_id for n in best.nodes]
    plan.strategy = best.strategy if len(best.nodes) > 1 else "solo"
    plan.tensor_parallel_size = best.tp
    plan.pipeline_parallel_size = best.pp
    plan.weights_per_node_gb = best.weights_per_node
    plan.kv_gb = best.kv_gb
    plan.kv_tokens = best.tokens

    smallest = min(best.nodes, key=lambda n: n.usable_gb)
    claim = best.weights_per_node + ENGINE_OVERHEAD_GB + \
        (COMM_OVERHEAD_GB if len(best.nodes) > 1 else 0.0) + \
        best.kv_gb / len(best.nodes)
    total = smallest.total_gb or smallest.free_gb or claim
    gmu = min(0.95, round(claim / total, 2)) if total else 0.85
    if recommended_gmu:
        gmu = min(gmu, recommended_gmu) if gmu else recommended_gmu
    plan.gpu_memory_utilization = gmu

    if bytes_per_token:
        headroom = best.tokens
        if short:
            # Divided by the concurrency that was asked for, not by one: a
            # window only a single request can hold is not what was wanted.
            plan.max_model_len = _round_len(headroom / max(1, concurrency))
            plan.warnings.append(
                f"{wanted_len:,} tokens × {max(1, concurrency)} request(s) needs "
                f"more cache than these nodes have free. Planned at "
                f"{plan.max_model_len:,} instead; free memory on a node, or "
                f"halve the cost per token with --kv-cache-dtype fp8.")
        else:
            plan.max_model_len = (min(wanted_len, _round_len(headroom))
                                  or _round_len(headroom))
        if plan.max_model_len:
            plan.concurrent_requests = max(1, headroom // plan.max_model_len)
        else:
            # Less cache left than the smallest window worth serving. The
            # weights fit, and that is exactly the trap: the engine would
            # start, take minutes doing it, and refuse at the last step.
            plan.fits = False
            plan.blocker = (
                f"The weights fit, but only {best.kv_gb:.1f} GB is left for the "
                f"KV cache — under {LEN_GRANULARITY:,} tokens, which is less "
                f"than one request. Free memory on a node, add a node, or use "
                f"--kv-cache-dtype fp8.")
            plan.notes = _explain(facts, plan, best, bytes_per_token,
                                  kv_cache_dtype)
            return plan
    plan.notes = _explain(facts, plan, best, bytes_per_token, kv_cache_dtype)
    plan.warnings.extend(_caveats(facts))
    return plan


def _blocker(facts: ModelFacts, nodes: Sequence[NodeBudget],
             refusals: Sequence[str]) -> str:
    pooled = sum(n.usable_gb for n in nodes)
    need = facts.weights_gb + ENGINE_OVERHEAD_GB
    if pooled < need:
        return (f"{facts.weights_gb:.0f} GB of weights plus overhead do not fit "
                f"in the {pooled:.0f} GB free across {len(nodes)} node(s). "
                f"Unload something, or use a smaller checkpoint.")
    return ("The memory is there in total but not in any split this cluster can "
            "form: " + ("; ".join(refusals[-3:]) if refusals else "no valid split"))


def _explain(facts: ModelFacts, plan: Plan, best: _Candidate,
             bytes_per_token: int, kv_cache_dtype: str) -> List[str]:
    notes = [
        f"Weights: {facts.weights_gb:.1f} GB on disk"
        + (f", split {len(best.nodes)} ways = {best.weights_per_node:.1f} GB "
           f"per node" if len(best.nodes) > 1 else ""),
        f"Engine overhead: {ENGINE_OVERHEAD_GB:.1f} GB per node"
        + (f" plus {COMM_OVERHEAD_GB:.1f} GB for the ranks"
           if len(best.nodes) > 1 else ""),
        f"Left for KV: {best.kv_gb:.1f} GB across "
        f"{len(best.nodes)} node(s), after keeping {SYSTEM_RESERVE_GB:.0f} GB "
        f"per node for the system",
    ]
    if bytes_per_token:
        dtype = "fp8" if str(kv_cache_dtype).lower().startswith("fp8") \
            else (facts.torch_dtype or "the model dtype")
        shape = (f"{facts.attention_layers} attention layers x "
                 f"{facts.num_kv_heads} KV heads x {facts.head_dim} head dim")
        if facts.kv_lora_rank:
            shape = (f"{facts.attention_layers} layers x "
                     f"{facts.kv_lora_rank + facts.qk_rope_head_dim} latent "
                     f"dimensions (MLA)")
        notes.append(
            f"KV per token: {bytes_per_token / 1024:.1f} KiB "
            f"({shape}, {dtype}) -> {plan.kv_tokens:,} tokens")
        if plan.max_model_len:
            notes.append(
                f"At {plan.max_model_len:,} context that is "
                f"{plan.concurrent_requests} concurrent request(s)")
    return notes


def _caveats(facts: ModelFacts) -> List[str]:
    out = []
    if facts.is_hybrid:
        out.append(
            "This is a hybrid stack: only its full-attention layers cache per "
            "token, while the recurrent layers hold a fixed state per "
            "sequence. The token figure is therefore a floor — the real "
            "capacity does not scale with context the way it implies.")
    if facts.unknown:
        out.append("Not stated by the checkpoint: " + ", ".join(facts.unknown))
    return out
