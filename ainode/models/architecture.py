"""Flags a checkpoint's architecture requires at a given split.

Not a recipe and not a preference: things that are wrong without them, which
the checkpoint itself tells us.

The one that exists for, reported from the cluster:

    also immer wenn ich versuche das minimax3 zu laden lande ich im
    speicherüberlauf … ich bin mit dem max context mittlerweile bis auf 3072
    runter gegangen, aber keinerlei Veränderung … das Modell müsste doch
    eigentlich auf node1 und node2 (tp=2) passen.

It should, and it does not, and the context length being irrelevant is the
clue: the KV cache is not what is filling the node. Tensor parallelism splits
attention heads and dense layers across ranks, but **not** a mixture-of-
experts' expert weights — those are sharded only when expert parallelism is
switched on. Without it every rank materialises every expert, so a 129 GB
checkpoint needs 129 GB *per node* no matter how short the context is.

The curated catalog has known this since MiniMax M2.7 was added:

    # 256 experts with 6 active — without this every rank holds every
    # expert and the weights do not fit.
    "--enable-expert-parallel",

which is exactly right, and useless to a model the operator downloaded
themselves: an uncurated checkpoint has no recipe. The architecture is in
config.json either way, so it does not have to be curated to be known.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["architecture_args", "facts_for"]

#: Sharding the experts across ranks instead of replicating them.
EXPERT_PARALLEL = "--enable-expert-parallel"


def facts_for(app, model: str):
    """The checkpoint's facts, or None when it cannot be read here."""
    manager = app.get("model_manager") if hasattr(app, "get") else None
    if manager is None or not model:
        return None
    try:
        from ainode.planner.facts import local_facts

        facts = local_facts(manager, model)
    except Exception:
        logger.debug("could not read the facts for %s", model, exc_info=True)
        return None
    return facts if getattr(facts, "weight_bytes", 0) else facts


# There is no flag for this one.
#
# Passing --quantization modelopt_fp4 for a config with no quant_method was
# the obvious move and it does not work:
#
#     Value error, Quantization method specified in the model config (None)
#     does not match the quantization method specified in the `quantization`
#     argument (modelopt_fp4).
#
# vLLM derives a method from the config first and then refuses an argument
# that disagrees with it — and for a config without the key it derives None,
# which disagrees with everything. The fix has to be the key itself; see
# ainode/models/quantization.py and repair_quant_method().


def architecture_args(app, model: str, node_count: int) -> List[str]:
    """What this model needs at this split, beyond anyone's preferences.

    Merged the way a recipe is — the caller wins per flag, and
    ``drop:--enable-expert-parallel`` removes it — so this is a default with
    reasons, not a rule. An operator who knows the model replicates its
    experts deliberately can still say so.
    """
    if node_count <= 1:
        return []
    facts: Optional[object] = facts_for(app, model)
    if facts is None or not getattr(facts, "is_moe", False):
        return []
    return [EXPERT_PARALLEL]


def would_add(app, model: str, node_count: int) -> List[str]:
    """Same list, for callers asking "what is different about this launch?"."""
    return architecture_args(app, model, node_count)
