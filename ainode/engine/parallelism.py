"""How a model is split across cluster nodes: tensor, pipeline or data parallel.

Until now the launch path only ever built tensor parallelism — TP = node
count, PP pinned to 1 — which works because every cluster AINode shipped
against had a power-of-two node count. A 3-node mesh breaks that: **no
commonly used model supports TP=3**, because tensor parallelism splits
attention heads across ranks and head counts are powers of two. Three nodes
therefore need a different axis.

The three axes, and when each is the right answer:

* **tensor** — splits every layer's weights across ranks. Lowest added
  latency per node, but needs a rank count the model's head count divides
  by, so in practice 1/2/4/8. The proven path on this hardware.
* **pipeline** — gives each node a contiguous slice of layers. Any node
  count works. Lets a model run that does not fit on two nodes; adds no
  single-stream speed, though total throughput can improve.
* **data** — a full model replica per node, requests spread across them.
  Only for a model that already fits one node; buys concurrency, not
  capacity.

That framing, and the "TP=3 is not an option" constraint, is upstream's:
see eugr/spark-vllm-docker's README § "Support for 3-node mesh setups" and
``recipes/3x-spark-cluster/`` (MIT), whose working 3-node recipe runs
``-tp 1 -pp 3``.

A note on confidence: pipeline parallelism across nodes is proven on this
hardware by that recipe. Data parallelism is documented upstream but has no
recipe behind it and is unverified here, so :func:`recommend_strategy` never
picks it on its own — a caller has to ask for it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Rank counts tensor parallelism can actually use. A model's attention-head
# count has to divide by the TP size, and head counts are powers of two — so
# TP=3, TP=5, TP=6 have no models behind them even though the arithmetic works.
TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)


class ParallelPlanError(ValueError):
    """A requested split cannot run on the selected nodes."""


class Strategy(str, Enum):
    """Which axis to split the model along."""

    TENSOR = "tensor"
    PIPELINE = "pipeline"
    DATA = "data"
    #: Pick for the caller based on node count. Never resolves to DATA.
    AUTO = "auto"

    @classmethod
    def parse(cls, value: Optional[str]) -> "Strategy":
        """Accept every spelling in circulation, or raise.

        The UI posts ``"tensor"`` while the API docs and ``ShardingStrategy``
        say ``"tensor_parallel"``; both have always been sent and neither was
        ever acted on, so both have to keep working now that the value
        actually decides something. ``"tp"``/``"pp"``/``"dp"`` are accepted
        too — that is how the flags read on a vLLM command line.
        """
        if value is None or value == "":
            return cls.AUTO
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", "_")
        aliases = {
            "tensor": cls.TENSOR, "tensor_parallel": cls.TENSOR, "tp": cls.TENSOR,
            "pipeline": cls.PIPELINE, "pipeline_parallel": cls.PIPELINE, "pp": cls.PIPELINE,
            "data": cls.DATA, "data_parallel": cls.DATA, "dp": cls.DATA,
            "auto": cls.AUTO,
        }
        try:
            return aliases[key]
        except KeyError:
            raise ParallelPlanError(
                f"Unknown parallelism strategy {value!r}. "
                f"Use one of: tensor, pipeline, data, auto."
            ) from None


@dataclass(frozen=True)
class ParallelPlan:
    """A concrete split: one rank per GPU, one GPU per GB10 node."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    strategy: Strategy = Strategy.TENSOR

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.data_parallel_size
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def label(self) -> str:
        """Short form for a UI badge, e.g. ``"PP=3"`` or ``"TP=2 · PP=2"``."""
        parts = []
        if self.tensor_parallel_size > 1:
            parts.append(f"TP={self.tensor_parallel_size}")
        if self.pipeline_parallel_size > 1:
            parts.append(f"PP={self.pipeline_parallel_size}")
        if self.data_parallel_size > 1:
            parts.append(f"DP={self.data_parallel_size}")
        return " · ".join(parts) or "TP=1"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "world_size": self.world_size,
            "label": self.label(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParallelPlan":
        """Parse a wire dict, defaulting every missing size to 1.

        Tolerant by design: a record written by an older build carries only
        ``tensor_parallel_size``, and must read back as the plan it described.
        """
        def _size(key: str) -> int:
            try:
                return max(1, int(data.get(key, 1) or 1))
            except (TypeError, ValueError):
                return 1

        return cls(
            tensor_parallel_size=_size("tensor_parallel_size"),
            pipeline_parallel_size=_size("pipeline_parallel_size"),
            data_parallel_size=_size("data_parallel_size"),
            strategy=Strategy.parse(data.get("strategy")),
        )


def recommend_strategy(node_count: int) -> Strategy:
    """The axis to use when the caller did not choose one.

    Tensor parallelism where the node count allows it — it is the proven
    path on this hardware and the only one that improves per-node memory
    pressure without a pipeline bubble. Otherwise pipeline, which works at
    any node count. Never data parallelism: it changes what the cluster is
    *for* (concurrency instead of capacity), which is a decision for the
    operator, and it is the least verified of the three here.
    """
    if node_count in TENSOR_PARALLEL_SIZES:
        return Strategy.TENSOR
    return Strategy.PIPELINE


def plan_for(strategy, node_count: int) -> ParallelPlan:
    """Build the plan that spreads ``node_count`` nodes along ``strategy``.

    Raises :class:`ParallelPlanError` with an actionable message when the
    combination has no models behind it — most importantly TP on a node
    count that is not a power of two.
    """
    if node_count < 1:
        raise ParallelPlanError(f"Need at least one node, got {node_count}.")

    resolved = Strategy.parse(strategy)
    if resolved is Strategy.AUTO:
        resolved = recommend_strategy(node_count)

    if resolved is Strategy.TENSOR:
        if node_count not in TENSOR_PARALLEL_SIZES:
            raise ParallelPlanError(
                f"Tensor-parallel across {node_count} nodes is not supported: "
                f"TP splits attention heads across ranks, and head counts are "
                f"powers of two, so no commonly used model divides by "
                f"{node_count}. Options: pipeline parallelism across all "
                f"{node_count} nodes (runs a model too large for fewer), data "
                f"parallelism for {node_count} replicas of a model that fits "
                f"one node, or select "
                f"{max(n for n in TENSOR_PARALLEL_SIZES if n < node_count)} "
                f"nodes for tensor-parallel."
            )
        plan = ParallelPlan(tensor_parallel_size=node_count, strategy=resolved)
    elif resolved is Strategy.PIPELINE:
        plan = ParallelPlan(pipeline_parallel_size=node_count, strategy=resolved)
    else:
        plan = ParallelPlan(data_parallel_size=node_count, strategy=resolved)

    validate_plan(plan, node_count)
    return plan


def validate_plan(plan: ParallelPlan, node_count: int) -> None:
    """Raise unless ``plan`` exactly fills ``node_count`` nodes.

    One rank per GPU and one GPU per GB10 node, so the world size has to
    equal the node count — a plan that under-fills silently idles hardware
    the operator selected, and one that over-fills fails deep inside vLLM
    with a message that points nowhere near the cause.

    Mirrors the ``tp * pp * dp`` check in eugr/spark-vllm-docker's
    ``launch-cluster.sh`` (``parse_parallelism_from_text``, MIT), which
    trims or rejects the node list the same way.
    """
    for name, size in (
        ("tensor", plan.tensor_parallel_size),
        ("pipeline", plan.pipeline_parallel_size),
        ("data", plan.data_parallel_size),
    ):
        if size < 1:
            raise ParallelPlanError(
                f"{name}-parallel size must be at least 1, got {size}."
            )

    if plan.tensor_parallel_size not in TENSOR_PARALLEL_SIZES:
        raise ParallelPlanError(
            f"Tensor-parallel size {plan.tensor_parallel_size} is not supported; "
            f"use one of {', '.join(str(n) for n in TENSOR_PARALLEL_SIZES)}."
        )

    if plan.world_size != node_count:
        raise ParallelPlanError(
            f"{plan.label()} needs {plan.world_size} GPU(s) but "
            f"{node_count} node(s) were selected. One GPU per node, so "
            f"tp x pp x dp must equal the node count."
        )


__all__ = [
    "TENSOR_PARALLEL_SIZES",
    "ParallelPlan",
    "ParallelPlanError",
    "Strategy",
    "plan_for",
    "recommend_strategy",
    "validate_plan",
]
