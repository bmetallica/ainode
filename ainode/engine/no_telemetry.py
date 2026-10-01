"""Nothing about this cluster leaves it on its own.

vLLM reports usage — hardware, model architecture, some settings — to its
developers (stats.vllm.ai) from a background thread in every engine, unless
told not to; a py-spy dump of a Qwen3.8-Flash-Next worker on spark-1432 showed
the thread (``_report_usage_worker``). huggingface_hub and the libraries on top
of it send telemetry of their own. Neither is needed for anything here, and on
a cluster without internet they only fail quietly.

So every engine container gets these, and AINode's own process too. The
operator's (or a recipe's) environment still wins: they are defaults.
"""

from __future__ import annotations

import os
from typing import Dict

__all__ = ["NO_TELEMETRY_ENV", "with_no_telemetry", "apply_to_process"]

NO_TELEMETRY_ENV: Dict[str, str] = {
    "VLLM_NO_USAGE_STATS": "1",     # vLLM's usage reporting
    "DO_NOT_TRACK": "1",            # the cross-tool convention vLLM honours too
    "HF_HUB_DISABLE_TELEMETRY": "1",  # huggingface_hub / transformers / diffusers
}


def with_no_telemetry(env: Dict[str, str]) -> Dict[str, str]:
    """``env`` with the defaults underneath it."""
    return {**NO_TELEMETRY_ENV, **{str(k): str(v) for k, v in (env or {}).items()}}


def apply_to_process() -> None:
    """For AINode itself, which uses huggingface_hub for search and downloads."""
    for key, value in NO_TELEMETRY_ENV.items():
        os.environ.setdefault(key, value)
