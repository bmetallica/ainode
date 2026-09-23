"""Can this engine read the checkpoint's quantization at all?

Asked before a launch, not after. A distributed start takes two nodes, a Ray
cluster and several minutes to reach the point where vLLM parses
``quantization_config`` — and when the answer is no, it is no for every
combination of flags, so those minutes bought nothing.

The case this exists for, from the cluster
(``aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx``)::

    Value error, Unsupported weight_bits: 16, currently only support
    {2, 3, 4, 5, 6, 7, 8}

The 16 is not a mistake. A mixed-bit checkpoint can be written two ways
round, and only one of them loads:

* **quantized default, unquantized exceptions** — ``bits`` is 4, and
  ``extra_config`` names the modules kept at 16. vLLM handles this: its
  per-layer lookup treats ``bits >= 16`` as "leave this one alone".
* **unquantized default, quantized exceptions** — ``bits`` is 16, and
  ``extra_config`` carries the real widths per module. vLLM validates the
  global value against its supported set *in the config constructor*, before
  any per-layer entry is read, and 16 is not in that set.

So the second form stops at construction. Reading it needs the vendor's own
quantization plugin in the engine image, which is a deployment decision, not
a serve flag — the model card names the stack it was built against.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

__all__ = ["quantization_verdict", "mixed_bit_widths"]

#: What vLLM's own config classes accept as a global width. Kept here rather
#: than imported: this runs on the orchestrator, which has no vLLM.
_SUPPORTED_BITS = {2, 3, 4, 5, 6, 7, 8}

#: Methods whose config carries a global ``bits`` plus per-module overrides.
_PER_LAYER_METHODS = {"autoround", "auto-round", "auto_round", "gptq",
                      "awq", "intel/auto-round"}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def mixed_bit_widths(extra_config: Dict[str, Any]) -> List[int]:
    """The distinct widths named per module, smallest first.

    Reported back to the operator because it is the one part of this that is
    genuinely informative: a checkpoint advertised as "3.2 bit" turns out to
    hold 2-, 3- and 4-bit modules, and knowing that is how you recognise the
    same problem in the next one.
    """
    widths = set()
    for entry in (extra_config or {}).values():
        if isinstance(entry, dict):
            bits = _as_int(entry.get("bits"))
            if bits and bits < 16:
                widths.add(bits)
    return sorted(widths)


def quantization_verdict(config: Dict[str, Any]) -> Tuple[bool, str]:
    """(servable, reason). ``reason`` is empty when there is nothing to say.

    Silent on everything it does not recognise — an unreadable config.json or
    an unfamiliar method must not block a launch that would have worked. The
    only refusal here is the one that is certain.
    """
    if not isinstance(config, dict):
        return True, ""
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return True, ""

    method = str(quant.get("quant_method") or "").strip().lower()
    if method and method not in _PER_LAYER_METHODS:
        return True, ""

    bits = _as_int(quant.get("bits"))
    extra = quant.get("extra_config")
    if bits in _SUPPORTED_BITS or not isinstance(extra, dict) or not extra:
        return True, ""
    if bits < 16:
        return True, ""

    widths = mixed_bit_widths(extra)
    detail = (f"{', '.join(str(w) for w in widths)}-bit"
              if widths else "several widths")
    version = str(quant.get("autoround_version") or "").strip()
    built_with = f" (autoround {version})" if version else ""
    return False, (
        f"this checkpoint is mixed-bit{built_with}: its quantization_config "
        f"sets bits={bits} — unquantized — as the global default and records "
        f"the real widths per module ({detail}) across "
        f"{len(extra)} entries. vLLM validates the global value before it "
        f"reads any per-module entry, and 16 is not in the set it accepts, so "
        f"the engine stops at config parsing. No serve flag changes that: "
        f"reading this layout needs the quantization plugin the model card "
        f"names, installed in the engine image. Written the other way round — "
        f"a quantized default with unquantized exceptions — the same checkpoint "
        f"would load."
    )


#: Hugging Face repo names that say "mixed-bit" out loud. Only used to say so
#: earlier, in the download list, where there is no config.json to read yet.
_MIXED_NAME = re.compile(r"(\d+\.\d+)\s*bit", re.IGNORECASE)


def name_suggests_mixed_bits(repo: str) -> bool:
    """True for a repo whose name claims a fractional bit width.

    A width like 3.2 bits is not a thing a uniform quantizer produces; it is
    the average over modules of different widths. Weak evidence — it is a
    name, not a config — so it is only ever a warning.
    """
    return bool(_MIXED_NAME.search(repo or ""))
