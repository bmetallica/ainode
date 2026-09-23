"""How much of the machine an engine may claim.

``gpu_memory_utilization`` is a fraction of the device's TOTAL memory, and
vLLM sizes its KV cache to fill it: it measures its own peak and takes the
rest. On a discrete GPU that is a reasonable thing to ask for — the operating
system lives somewhere else, and an engine that asks for too much gets a CUDA
error. On GB10 the GPU's memory IS the host's memory, so 0.97 of 128 GB means
124 GB for one engine and four for Linux, docker, AINode and the page cache.
The node does not fail; it dies, and has to be power-cycled.

The number is also blind to everything that is not itself. It is a share of
total, not of free, so two engines at 0.6 each overcommit, and an engine
launched onto a node that is already half full overcommits on its own.

So the ceiling here is computed from what is actually free, minus what the
memory guard refuses to fall below, expressed back as a fraction of total::

    ceiling = (free - reserve) / total

On an idle 128 GB Spark with a 12 GB reserve that is about 0.90; with a model
already loaded it is correspondingly less, which is the point. The guard is
the last line and cannot win this race — a 100 GB allocation crosses its
threshold in less time than it takes to notice — so the launch has to be the
thing that does not ask.

``force`` skips it: an operator who knows what a checkpoint needs is allowed
to be right, and the refusal says so.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["utilization_ceiling", "cap_utilization", "MIN_UTILIZATION"]

#: Never cap below this. A ceiling of 0.02 is not a safer launch, it is a
#: launch that cannot hold its own weights — better to refuse in the admission
#: gate, which says why, than to start something that will fail obscurely.
MIN_UTILIZATION = 0.15


def utilization_ceiling(app, node_ids=None) -> Tuple[float, str]:
    """(ceiling, the node that sets it). (0.0, "") when it cannot be computed.

    The tightest node decides: every rank of a tensor-parallel launch gets the
    same fraction, so the launch is bounded by whichever node has least.
    """
    try:
        from ainode.safety.admission import _budgets_with_reserve
    except Exception:  # pragma: no cover - defensive
        return 0.0, ""

    try:
        budgets = _budgets_with_reserve(app, node_ids)
    except Exception:
        logger.debug("could not read node budgets", exc_info=True)
        return 0.0, ""

    ceiling = 0.0
    where = ""
    for budget in budgets:
        if budget.total_gb <= 0:
            continue
        share = max(0.0, budget.usable_gb) / budget.total_gb
        if not where or share < ceiling:
            ceiling, where = share, (budget.name or budget.node_id)
    return (round(ceiling, 3), where) if where else (0.0, "")


def cap_utilization(app, requested: Optional[float], *, node_ids=None,
                    force: bool = False) -> Tuple[Optional[float], str]:
    """(value to launch with, a note for the log and the UI).

    Returns ``requested`` unchanged — and an empty note — whenever there is
    nothing to say: force, no budgets, or a request that already fits.
    """
    if requested is None or force:
        return requested, ""
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return requested, ""

    ceiling, where = utilization_ceiling(app, node_ids)
    # An empty name means there were no budgets to read — silence, not a cap
    # of zero. A ceiling of zero WITH a name is a node that genuinely has no
    # room, and the floor below applies: the admission gate refuses that
    # launch anyway, and this is the second lock on the same door.
    if not where or value <= ceiling:
        return requested, ""
    capped = max(MIN_UTILIZATION, round(ceiling, 2))
    if capped >= value:
        return requested, ""
    note = (
        f"gpu-memory-utilization lowered from {value:.2f} to {capped:.2f}: "
        f"that fraction of {where}'s total memory is what is free there once "
        f"the host reserve is held back. On this hardware the engine's memory "
        f"and the operating system's are the same memory, so asking for more "
        f"than is free does not fail the launch — it takes the node down. "
        f"Pass \"force\": true to launch with {value:.2f} anyway."
    )
    logger.warning("%s", note)
    return capped, note
