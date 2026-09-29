"""Which replica gets a request, when a model runs on more than one node.

wizzard.md, E4. A profile may run the same model once per node — two copies at
TP=1 rather than one at TP=2, faster for a model that fits one node and still
serving when a node goes down. The proxy already knew every node serving a
model (_routing_candidates), but only to fail over: it always tried the local
copy first, so the second copy did nothing until the first was gone.

Two rules, in this order:

* **The same conversation goes back to the same replica**, as long as that
  replica is not clearly busier than the others. A coding agent sends the
  whole, growing conversation with every step; the replica that served the
  previous step has it in its prefix cache and answers without recomputing
  it, the other replica would start from nothing. The conversation is known by
  its opening — the first messages do not change as it grows.
* Otherwise, **fewest requests in flight** from this node, ties to the order
  the candidates came in (local first: the cheapest hop).

Only what passes through this node is counted — each node balances its own
traffic, which is what a client pointed at one address produces.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

__all__ = ["Balancer", "affinity_key"]

Target = Tuple[str, int]

#: How much busier the remembered replica may be before a conversation is
#: moved: one more request in flight is worth a warm prefix cache, three are not.
AFFINITY_SLACK = 2

#: Conversations remembered.
AFFINITY_SIZE = 4096

#: Characters of the opening that identify a conversation.
OPENING_CHARS = 2000


def affinity_key(model: str, body: Optional[bytes]) -> str:
    """A short key for the conversation a request continues, or ""."""
    if not body:
        return ""
    try:
        data = json.loads(body)
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    opening = ""
    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages[:2]:
            if isinstance(message, dict):
                content = message.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True)
                parts.append(f"{message.get('role')}:{content}")
        opening = "\n".join(parts)
    elif isinstance(data.get("prompt"), str):
        opening = data["prompt"]
    if not opening:
        return ""
    digest = hashlib.sha1(f"{model}\n{opening[:OPENING_CHARS]}".encode()).hexdigest()
    return digest[:16]


class Balancer:
    def __init__(self):
        self.inflight: Dict[Target, int] = {}
        self._affinity: "OrderedDict[str, Target]" = OrderedDict()

    def order(self, candidates: List[Target], key: str = "") -> List[Target]:
        """The candidates in the order to try them."""
        if len(candidates) < 2:
            return list(candidates)
        load = {c: self.inflight.get(c, 0) for c in candidates}
        ranked = sorted(candidates, key=lambda c: (load[c], candidates.index(c)))
        remembered = self._affinity.get(key) if key else None
        if remembered in load and load[remembered] <= load[ranked[0]] + AFFINITY_SLACK:
            ranked.remove(remembered)
            ranked.insert(0, remembered)
        return ranked

    def acquire(self, target: Target, key: str = "") -> None:
        self.inflight[target] = self.inflight.get(target, 0) + 1
        if key:
            self._affinity[key] = target
            self._affinity.move_to_end(key)
            while len(self._affinity) > AFFINITY_SIZE:
                self._affinity.popitem(last=False)

    def release(self, target: Target) -> None:
        left = self.inflight.get(target, 0) - 1
        if left > 0:
            self.inflight[target] = left
        else:
            self.inflight.pop(target, None)
