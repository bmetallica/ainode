"""Model registry and manager — dynamic catalog + download/delete/recommend.

The catalog is now assembled dynamically from live sources (HuggingFace Hub,
Ollama library, NVIDIA NIM) with a 24-hour on-disk cache and a small static
fallback for offline/error situations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Bytes-per-element for the dtypes HF reports in safetensors metadata. Lets us
# compute real download size for quantized models (NVFP4 weights land as U8).
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 0.5, "FP4": 0.5,
}


def _safetensors_size_gb(safetensors) -> float:
    """Size (decimal GB) ESTIMATED from the HF safetensors dtype breakdown.

    An estimate, and for a packed low-bit checkpoint a bad one. The metadata
    counts tensor elements by declared dtype, and a 4-bit format stores many
    values inside one U8 or I32 element — so the arithmetic below multiplies a
    packed count by the container's width. Measured against usedStorage:

        MiniMax-M2.7 AWQ-4bit     111.6 GiB actual   912 GB estimated
        DeepSeek-V4-Flash NVFP4   164.1 GiB actual   306 GB estimated

    The second one is why this matters: AINode marked a model that fits two
    nodes "Too large for cluster" and hid it from the search. Prefer
    :func:`repo_size_gb`, which uses the exact figure when the Hub gives one.
    """
    params = getattr(safetensors, "parameters", None)
    if not params:
        return 0.0
    total_bytes = sum(_DTYPE_BYTES.get(dt, 2) * n for dt, n in params.items())
    return round(total_bytes / 1e9, 1)


#: Below this, the dtype estimate is good enough: a model that small fits any
#: node whichever way the arithmetic lands, so paying a request to sharpen it
#: buys nothing. Above it, the fit verdict depends on the number.
_EXACT_SIZE_THRESHOLD_GB = 40.0

#: Search returns up to 50 rows; verifying every large one serially would make
#: the box feel broken. Bounded, in parallel, and the estimate stands for the
#: rest — which only ever over-states, so nothing that fits is hidden.
_EXACT_SIZE_LOOKUPS = 20


def exact_repo_size_gb(repo_id: str) -> float:
    """The Hub's own byte count for ``repo_id``, or 0.0 if it will not say.

    ``usedStorage`` is exact for every format, including the packed low-bit
    ones the dtype breakdown misreads by a factor of eight. It is available
    only on the single-model endpoint — asking for it on the list endpoint is
    a 400, which is how the search came to return nothing at all.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, expand=["usedStorage"])
        used = getattr(info, "used_storage", None)
        if used and int(used) > 0:
            return round(int(used) / 1e9, 1)
    except Exception:
        logger.debug("no usedStorage for %s", repo_id, exc_info=True)
    return 0.0


def _sharpen_sizes(results: list) -> None:
    """Replace the estimate with the Hub's byte count where it matters."""
    from concurrent.futures import ThreadPoolExecutor

    big = [r for r in results if (r.get("size_gb") or 0) >= _EXACT_SIZE_THRESHOLD_GB]
    big = big[:_EXACT_SIZE_LOOKUPS]
    if not big:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        exact = list(pool.map(lambda r: exact_repo_size_gb(r["hf_repo"]), big))
    for row, size in zip(big, exact):
        if size > 0:
            row["size_gb"] = size


def repo_size_gb(model) -> float:
    """On-disk size (decimal GB) of a Hub repo, exact where the Hub says so.

    ``usedStorage`` is the byte count the Hub itself reports for the repo. It
    is exact for every format, including the packed ones the dtype breakdown
    cannot read. The estimate remains the fallback for a repo that reports no
    storage figure.
    """
    used = getattr(model, "used_storage", None)
    if used is None:
        used = getattr(model, "usedStorage", None)
    try:
        if used and int(used) > 0:
            return round(int(used) / 1e9, 1)
    except (TypeError, ValueError):
        pass
    return _safetensors_size_gb(getattr(model, "safetensors", None))


def _download_max_workers() -> int:
    """Parallel-connection cap for model downloads (AINODE_DOWNLOAD_MAX_WORKERS,
    default 4). Keeps a fat HF pull from saturating the uplink."""
    try:
        return max(1, int(os.environ.get("AINODE_DOWNLOAD_MAX_WORKERS", "4")))
    except (TypeError, ValueError):
        return 4

from ainode.core.config import AINODE_HOME, MODELS_DIR  # noqa: E402


@dataclass
class ModelInfo:
    """Metadata for a model in the catalog."""

    id: str
    name: str
    hf_repo: str
    size_gb: float
    description: str
    quantization: Optional[str] = None
    min_memory_gb: float = 0.0
    family: str = ""
    params_b: float = 0.0
    context_length: int = 0
    license: str = ""
    recommended: bool = False
    # Cluster-proven config: proven_tp = node count to launch at; verified = we've
    # actually served it on this hardware (drives the picker default + a ✓ badge).
    proven_tp: int = 0
    verified: bool = False
    #: False for a model vLLM cannot split along the pipeline axis — it has to
    #: implement the SupportsPP interface and not every architecture does.
    #: Without this the planner's helpful downgrade ("three nodes cannot do
    #: tensor-parallel, so pipeline it") produces a launch that loads for
    #: minutes and then raises NotImplementedError.
    supports_pipeline: bool = True
    # True for our hand-picked CURATED_CLUSTER_MODELS — drives the "Catalog"
    # (known-good to grab) list, separate from on-disk / HF-sweep entries.
    curated: bool = False
    created_at: str = ""
    downloads: int = 0
    likes: int = 0
    # Capabilities — inferred from HF tags or model ID
    capabilities: list = None  # ["vision", "tool_use", "reasoning", "code", "multilingual"]
    architecture: str = ""
    format: str = ""  # "safetensors", "gguf", "awq", etc.
    # ---- Launch recipe (proven config, applied automatically on load) --------
    # Some models only serve correctly with a specific engine build and flag set
    # (speculative decoding, MoE/mamba backends, reasoning + tool-call parsers).
    # Carrying that here is what makes them a one-click catalog load instead of a
    # hand-rolled container. A caller's explicit /api/models/load value always
    # wins over the recipe; the recipe only fills what wasn't specified.
    engine_image: str = ""
    #: Image for the eugr launcher path, when the launcher's own default will
    #: not do. Empty means the default (`vllm-node`), which is what every
    #: upstream recipe but the experimental ones asks for.
    engine_image_eugr: str = ""          # "" = fleet default engine image
    extra_vllm_args: list = None    # verbatim `vllm serve` flags
    extra_env: dict = None          # engine-container env (e.g. b12x kernel selection)
    recommended_gmu: float = 0.0    # 0 = use node default gpu_memory_utilization

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.extra_vllm_args is None:
            self.extra_vllm_args = []
        if self.extra_env is None:
            self.extra_env = {}

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Fallback catalog ------------------------------------------------------
#
# Used when all live sources fail (offline, rate-limited, etc.). Kept small.

FALLBACK_CATALOG: dict[str, ModelInfo] = {
    "llama-3.2-3b": ModelInfo(
        id="llama-3.2-3b",
        name="Llama 3.2 3B Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        size_gb=6.0,
        description="Compact, fast model for everyday tasks. Great starter model.",
        min_memory_gb=8,
        family="llama",
        params_b=3.21,
        context_length=131072,
        license="Llama 3.2",
        recommended=True,
    ),
    "qwen-2.5-7b": ModelInfo(
        id="qwen-2.5-7b",
        name="Qwen 2.5 7B Instruct",
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        size_gb=15.0,
        description="Strong 7B with excellent multilingual and reasoning capability.",
        min_memory_gb=16,
        family="qwen",
        params_b=7.62,
        context_length=131072,
        license="Qwen",
        recommended=True,
    ),
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B Instruct v0.3",
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
        size_gb=14.0,
        description="Fast, efficient 7B with strong instruction following.",
        min_memory_gb=16,
        family="mistral",
        params_b=7.25,
        context_length=32768,
        license="Apache 2.0",
        recommended=True,
    ),
    "phi-3-mini": ModelInfo(
        id="phi-3-mini",
        name="Phi-3 Mini 4K Instruct",
        hf_repo="microsoft/Phi-3-mini-4k-instruct",
        size_gb=7.5,
        description="Microsoft's compact model. Strong reasoning for its size.",
        min_memory_gb=8,
        family="phi",
        params_b=3.82,
        context_length=4096,
        license="MIT",
        recommended=True,
    ),
    "gemma-2-9b": ModelInfo(
        id="gemma-2-9b",
        name="Gemma 2 9B IT",
        hf_repo="google/gemma-2-9b-it",
        size_gb=18.5,
        description="Google Gemma 2 9B. Strong mid-size open model.",
        min_memory_gb=20,
        family="gemma",
        params_b=9.24,
        context_length=8192,
        license="Gemma",
        recommended=True,
    ),
}


# ---- Curated cluster models (always discoverable) --------------------------
#
# The live HF sweep (top-downloads) misses the frontier/NVFP4 models this GB10
# cluster actually runs — so they were undiscoverable in the catalog and only
# appeared once already on disk. These curated entries are ALWAYS merged into
# the catalog (see ModelManager.get_catalog) so an operator can find + download
# them. NVFP4 is native on Blackwell; these run distributed (TP=N) across nodes.

CURATED_CLUSTER_MODELS: dict[str, ModelInfo] = {
    # --- Recipe-carrying models (need a newer engine + model-specific flags) ---
    # Both were validated end-to-end on the GB10 fleet 2026-08-13/15; the flag
    # sets below are the vendor/community recipes verbatim. They require vLLM
    # 0.27.1 — hence engine_image. Do NOT add --enforce-eager: it's a 0.17-era
    # workaround and only costs throughput here (see nvidia.py module header).
    "nemotron-3.5-lightning-nvfp4": ModelInfo(
        id="nemotron-3.5-lightning-nvfp4",
        name="Nemotron 3.5 Lightning 30B-A3B (NVFP4)",
        hf_repo="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        size_gb=21.0,
        description=(
            "MoE hybrid Mamba-2 (3B active/token) with DSpark speculative decoding — "
            "104 tok/s single-stream and 504 tok/s across 16 streams on one GB10, the "
            "fastest model on this hardware. 1M context. The sub-agent workhorse. "
            "Text only (no vision). First launch also pulls the 1.3 GB DSpark drafter."
        ),
        quantization="NVFP4", min_memory_gb=30, family="nemotron", params_b=30.0,
        proven_tp=1, verified=True, curated=True,
        context_length=1048576, license="OpenMDW-1.1", recommended=True,
        format="safetensors", capabilities=["tool_use", "reasoning", "code"],
        engine_image="vllm/vllm-openai:v0.27.1",
        extra_vllm_args=[
            # From eugr/spark-vllm-docker's recipes/nemotron-3.5-lightning.yaml
            # (MIT): instanttensor loads an NVFP4 checkpoint directly instead of
            # going through the generic safetensors path, and a 30 GB checkpoint
            # pays the difference on every launch.
            "--load-format", "instanttensor",
            "--moe-backend", "marlin",
            "--enable-prefix-caching",
            "--speculative_config.model",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
            "--speculative_config.num_speculative_tokens", "3",
            "--mamba-backend", "flashinfer",
            "--mamba-cache-mode", "align",
            "--reasoning-parser", "nemotron_v3",
            "--tool-call-parser", "qwen3_xml",
            "--enable-auto-tool-choice",
        ],
        extra_env={
            # Capped. The loader stages weights through one contiguous
            # buffer and asks the driver how much device memory it may use;
            # on GB10 it gets an answer that has nothing to do with the
            # machine. Measured on an idle node with 28 GB genuinely free:
            #
            #   RuntimeError: buffer_size (5086090240 B) exceeds device
            #   memory budget (825161728 B)
            #
            # and 862404608 B on the next attempt — a figure that moves
            # between runs, so it comes from a runtime query rather than from
            # gpu-memory-utilization. nvidia-smi reports [N/A] for memory on
            # this hardware too; unified memory is the common thread.
            #
            # 64 MiB is what the GLM recipe uses and what has loaded a 175 GB
            # checkpoint here. Raise or drop it (drop:--load-format) if a
            # future engine image reports the budget correctly.
            "INSTANTTENSOR_BUFFER_SIZE": "67108864",
        },
        recommended_gmu=0.91,
    ),
    "qwen3.8-27b-nvfp4": ModelInfo(
        id="qwen3.8-27b-nvfp4",
        name="Qwen3.8 27B (NVFP4, vision)",
        hf_repo="unsloth/Qwen3.8-27B-NVFP4",
        size_gb=23.4,
        description=(
            "Dense 27B native vision-language model (images + video) with built-in MTP "
            "speculative decoding — 19 tok/s single-stream on one GB10 (dense is "
            "bandwidth-bound; batching reaches 147 tok/s at 16 streams). 262K context, "
            "excellent instruction-following and tool use. The quality-and-eyes model. "
            "Use temperature 0 for OCR/transcription."
        ),
        quantization="NVFP4", min_memory_gb=32, family="qwen", params_b=27.0,
        proven_tp=1, verified=True, curated=True,
        context_length=262144, license="Apache 2.0", recommended=True,
        format="safetensors",
        capabilities=["vision", "tool_use", "reasoning", "code", "multilingual"],
        engine_image="vllm/vllm-openai:v0.27.1",
        extra_vllm_args=[
            # As in eugr's recipes/qwen3.8-27b-nvfp4-dflash2.yaml (MIT) — see the
            # note on the Nemotron entry above.
            "--load-format", "instanttensor",
            "--enable-prefix-caching",
            # Vision models must NOT get fp8 KV on GB10 — it corrupts generation
            # (proven 2026-07-06). The automatic fp8→auto downgrade only fires
            # when the model is on local disk (it reads config.json), and this
            # one serves straight from the HF cache, so state it explicitly.
            "--kv-cache-dtype", "auto",
            "--reasoning-parser", "qwen3",
            # REQUIRED: the template emits <tool_call><function=..><parameter=..>.
            # With the hermes parser, tool calls silently never parse (0 emitted).
            "--tool-call-parser", "qwen3_xml",
            "--enable-auto-tool-choice",
            "--speculative_config", '{"method":"qwen3_5_mtp","num_speculative_tokens":2}',
        ],
        extra_env={
            # Capped. The loader stages weights through one contiguous
            # buffer and asks the driver how much device memory it may use;
            # on GB10 it gets an answer that has nothing to do with the
            # machine. Measured on an idle node with 28 GB genuinely free:
            #
            #   RuntimeError: buffer_size (5086090240 B) exceeds device
            #   memory budget (825161728 B)
            #
            # and 862404608 B on the next attempt — a figure that moves
            # between runs, so it comes from a runtime query rather than from
            # gpu-memory-utilization. nvidia-smi reports [N/A] for memory on
            # this hardware too; unified memory is the common thread.
            #
            # 64 MiB is what the GLM recipe uses and what has loaded a 175 GB
            # checkpoint here. Raise or drop it (drop:--load-format) if a
            # future engine image reports the budget correctly.
            "INSTANTTENSOR_BUFFER_SIZE": "67108864",
        },
        recommended_gmu=0.60,
    ),
    "gemma4-26b-a4b-nvfp4": ModelInfo(
        id="gemma4-26b-a4b-nvfp4",
        name="Gemma 4 26B-A4B (NVFP4)",
        hf_repo="nvidia/Gemma-4-26B-A4B-NVFP4",
        size_gb=18.0,
        description=(
            "MoE (4B active/token) in Blackwell-native NVFP4 with MTP speculative "
            "decoding — the everyday chat model: MoE decode speed with 26B quality, "
            "262K context, tool use and reasoning. Pairs with OpenWebUI for a team. "
            "First launch also pulls the assistant drafter."
        ),
        quantization="NVFP4", min_memory_gb=24, family="gemma", params_b=26.0,
        proven_tp=1, verified=False, curated=True,
        context_length=262144, license="Gemma", recommended=True,
        format="nvfp4",
        capabilities=["vision", "audio", "tool_use", "reasoning", "multilingual"],
        # Flags taken from eugr/spark-vllm-docker's proven recipe
        # recipes/gemma4-26b-a4b-nvfp4.yaml (MIT): the gemma4 reasoning and
        # tool-call parsers, instanttensor loading and the MTP drafter are what
        # make this model serve correctly rather than merely start. The recipe's
        # -tp 2 is deliberately NOT carried over: it targets a two-GPU host,
        # and on a one-GPU-per-node Spark the split is planned from the node
        # selection instead.
        extra_vllm_args=[
            "--load-format", "instanttensor",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
            # eugr's recipe says fp8 here. We deviate, deliberately: Gemma 4
            # is multimodal, and fp8 KV corrupts vision-model generation on
            # GB10 — proven on this fleet 2026-07-06 with Qwen2.5-VL, which
            # emitted garbage on fp8 and clean output on auto. The automatic
            # downgrade in serve_args cannot help here, because an explicit
            # value in a recipe is exactly what it must not override. auto
            # costs roughly half the concurrent sequences at the same context
            # length; wrong output costs more. Set --kv-cache-dtype fp8 in the
            # launch panel to take the recipe's value back.
            "--kv-cache-dtype", "auto",
            "--max-num-batched-tokens", "8192",
            "--speculative-config",
            '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant",'
            '"num_speculative_tokens":4,"moe_backend":"triton"}',
        ],
        extra_env={
            # Capped. The loader stages weights through one contiguous
            # buffer and asks the driver how much device memory it may use;
            # on GB10 it gets an answer that has nothing to do with the
            # machine. Measured on an idle node with 28 GB genuinely free:
            #
            #   RuntimeError: buffer_size (5086090240 B) exceeds device
            #   memory budget (825161728 B)
            #
            # and 862404608 B on the next attempt — a figure that moves
            # between runs, so it comes from a runtime query rather than from
            # gpu-memory-utilization. nvidia-smi reports [N/A] for memory on
            # this hardware too; unified memory is the common thread.
            #
            # 64 MiB is what the GLM recipe uses and what has loaded a 175 GB
            # checkpoint here. Raise or drop it (drop:--load-format) if a
            # future engine image reports the budget correctly.
            "INSTANTTENSOR_BUFFER_SIZE": "67108864",
        },
        recommended_gmu=0.70,
    ),
    # --- Fast single-node quantized chat models (AWQ-4bit, awq_marlin on GB10) ---
    # The everyday "always-on" tier: fit one node, serve at interactive speed, and
    # stack several per node. proven_tp=1 (no distribution). verified=True is set
    # ONLY after a real completion was observed on the cluster.
    "qwen3.5-9b-awq": ModelInfo(
        id="qwen3.5-9b-awq",
        name="Qwen3.5 9B (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-9B-AWQ",
        size_gb=12.0,
        description="Fast dense 9B, AWQ-4bit (awq_marlin). ~19 tok/s single-stream on one GB10. Great default chat model.",
        quantization="AWQ", min_memory_gb=14, family="qwen", params_b=9.0,
        proven_tp=1, verified=True,
        context_length=262144, license="Apache 2.0", recommended=True, format="awq",
    ),
    "qwen3.5-4b-awq": ModelInfo(
        id="qwen3.5-4b-awq",
        name="Qwen3.5 4B (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-4B-AWQ",
        size_gb=4.0,
        description="Tiny dense 4B, AWQ-4bit. ~15 tok/s single-stream (dense AWQ is dequant-bound on GB10, not size-bound — the MoE is the fast pick). Lowest memory / highest QPS for batched routes.",
        quantization="AWQ", min_memory_gb=6, family="qwen", params_b=4.0,
        proven_tp=1, verified=True,
        context_length=262144, license="Apache 2.0", format="awq",
    ),
    "qwen3.5-35b-a3b-awq": ModelInfo(
        id="qwen3.5-35b-a3b-awq",
        name="Qwen3.5 35B-A3B MoE (AWQ-4bit)",
        hf_repo="QuantTrio/Qwen3.5-35B-A3B-AWQ",
        size_gb=24.0,
        description="MoE (3B active/token), AWQ-4bit. ~27 tok/s single-stream on one GB10 — fast decode AND large-model quality. The flagship single-node model.",
        quantization="AWQ", min_memory_gb=28, family="qwen", params_b=35.0,
        proven_tp=1, verified=True,
        context_length=262144, license="Apache 2.0", recommended=True, format="awq",
    ),
    "llama-3.1-8b-nvfp4": ModelInfo(
        id="llama-3.1-8b-nvfp4",
        name="Llama 3.1 8B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        size_gb=6.0,
        description="Dense 8B, Blackwell-native NVFP4 — ~18 tok/s single-stream on one GB10 (dense is bandwidth-bound). Solid general-purpose chat model, light enough to stack.",
        quantization="NVFP4", min_memory_gb=8, family="llama", params_b=8.0,
        proven_tp=1, verified=True,
        context_length=131072, license="Llama 3.1", recommended=True, format="nvfp4",
    ),
    # --- Community daily-driver MoE picks (DGX Spark forum + r/LocalLLaMA, 2026) ---
    "nemotron-cascade-2-30b-a3b-nvfp4": ModelInfo(
        id="nemotron-cascade-2-30b-a3b-nvfp4",
        name="Nemotron Cascade 2 30B-A3B (NVFP4)",
        hf_repo="chankhavu/Nemotron-Cascade-2-30B-A3B-NVFP4",
        size_gb=18.0,
        description="NVIDIA's distilled hybrid (mamba+attention) MoE, 3B active. Blackwell-native NVFP4 — ~32 tok/s single-stream on one GB10 (eager-on; the ~60 t/s Spark forum reports need CUDA graphs/eager-off). Fast daily driver, great for stacking.",
        quantization="NVFP4", min_memory_gb=22, family="nemotron", params_b=30.0,
        proven_tp=1, verified=True,
        context_length=131072, license="NVIDIA Open Model", recommended=True, format="nvfp4",
    ),
    "deepseek-v4-flash-dspark": ModelInfo(
        id="deepseek-v4-flash-dspark",
        name="DeepSeek V4 Flash + DSpark (2 nodes)",
        hf_repo="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        # MEASURED from the Hub's usedStorage: 155.4 GiB. Over two nodes that
        # is 77.7 GiB each against ~106 GiB addressable at 0.87 — the largest
        # model that fits this cluster with room left for a real KV cache.
        size_gb=166.9, min_memory_gb=175,
        description=(
            "284B MoE with 13B active, hybrid compressed-sparse attention, and "
            "DeepSeek's DSpark speculative decoder (21B/~1B active) in the same "
            "checkpoint — one file serves as both target and draft. MIT, so "
            "commercial use is unencumbered. SERVED HERE on two Sparks at TP=2 "
            "with expert parallelism: 1,226,910 tokens of KV cache, 9.36x "
            "concurrency at 131,072 per request. The model advertises 1M "
            "context and does not fit it: at max_model_len 1048576 the "
            "bookkeeping leaves 3.37 GiB for the cache and the launch is "
            "refused, so the recipe pins 131072. It reasons before answering "
            "— budget output tokens accordingly and set reasoning support in "
            "the client, or answers arrive empty."
        ),
        quantization=None, family="deepseek", params_b=284.0,
        proven_tp=2, verified=True, curated=True,
        context_length=1048576, license="MIT", recommended=True,
        format="safetensors",
        capabilities=["tool_use", "reasoning", "code"],
        extra_vllm_args=[
            # Required: the sparse-attention indexer needs DeepGEMM, and the
            # checkpoint carries custom modelling code.
            "--trust-remote-code",
            # 256 experts with 6 active — without this every rank holds every
            # expert and the weights do not fit.
            "--enable-expert-parallel",
            "--kv-cache-dtype", "fp8",
            # NOT the model's 1048576. Measured: at 1M the activation and
            # CUDA-graph bookkeeping consumes ~25 of the ~28 GiB left after
            # the weights, and vLLM refuses with "5.4 GiB KV cache is needed,
            # which is larger than the available KV cache memory (3.37 GiB)".
            # At 131072 the same nodes hold 1,226,910 tokens. Raising this
            # costs cache twice over: more per sequence, less in total.
            "--max-model-len", "131072",
            "--max-num-seqs", "8",
            # The drafter that ships inside this checkpoint. NVIDIA's card
            # gives these values; accepted by the engine here.
            "--speculative-config",
            '{"method":"dspark","num_speculative_tokens":7,'
            '"draft_sample_method":"greedy"}',
            "--enable-auto-tool-choice",
            "--tool-call-parser", "deepseek_v4",
        ],
        recommended_gmu=0.87,
    ),
    "minimax-m2.7-awq": ModelInfo(
        id="minimax-m2.7-awq",
        name="MiniMax-M2.7 (AWQ-4bit)",
        # Renamed upstream: demon-zombie/... now 307-redirects here. The hub
        # client follows it, but a redirect is not a name to keep in a catalog.
        hf_repo="et0dev/MiniMax-M2.7-AWQ-4bit",
        # MEASURED from the Hub's file listing: 24 shards, 111.6 GB. The 120.0
        # that stood here was 230B x 4 bits, the same arithmetic that put Gemma
        # 4 31B at half its real size. This one happens to land the other way —
        # it is SMALLER than the guess, which is what makes two nodes
        # comfortable rather than tight.
        size_gb=111.6,
        description=(
            "The community's top agentic-coding pick — 'Sonnet at home'. MoE, "
            "256 experts with 8 active (A10B), int4 pack-quantized. ~42 tok/s "
            "across 2 Sparks (TP=2). 62 layers with 8 KV heads at head_dim "
            "128, so fp8 KV costs 124 KiB per token: at gpu_memory_utilization "
            "0.87 two nodes leave about 85 GB for the cache, near 700k tokens "
            "— roughly 11 concurrent sessions at 64K context each. TP=2 splits "
            "the 8 KV heads 4 and 4, with no replication."
        ),
        quantization="AWQ", min_memory_gb=130, family="minimax", params_b=230.0,
        proven_tp=2, verified=False,
        # From the checkpoint's config.json (max_position_embeddings), not the
        # 131072 that was here.
        context_length=196608, license="MiniMax", recommended=True, format="awq",
        capabilities=["tool_use", "reasoning", "code"],
        extra_vllm_args=[
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
        ],
        recommended_gmu=0.87,
    ),
    "qwen3-235b-a22b-nvfp4": ModelInfo(
        id="qwen3-235b-a22b-nvfp4",
        name="Qwen3-235B-A22B (NVFP4)",
        hf_repo="nvidia/Qwen3-235B-A22B-NVFP4",
        size_gb=250.0,
        description="Frontier MoE (A22B active). Runs distributed TP=4 on the cluster. NVFP4 for GB10.",
        quantization="NVFP4", min_memory_gb=275, family="qwen", params_b=235.0,
        proven_tp=4, verified=True,
        context_length=262144, license="Apache 2.0", recommended=True, format="nvfp4",
    ),
    "qwen3.5-397b-a17b-nvfp4": ModelInfo(
        id="qwen3.5-397b-a17b-nvfp4",
        name="Qwen3.5-397B-A17B (NVFP4)",
        hf_repo="nvidia/Qwen3.5-397B-A17B-NVFP4",
        size_gb=468.0,
        description="Frontier MoE (A17B active) — the cluster's design point. Distributed TP=4. NVFP4.",
        quantization="NVFP4", min_memory_gb=500, family="qwen", params_b=397.0,
        proven_tp=4, verified=False,
        context_length=262144, license="Apache 2.0", recommended=True, format="nvfp4",
    ),
    "llama-3.1-405b-nvfp4": ModelInfo(
        id="llama-3.1-405b-nvfp4",
        name="Llama 3.1 405B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.1-405B-Instruct-NVFP4",
        size_gb=437.0,
        description="Dense 405B, NVFP4. Needs the cluster's pooled memory (TP=4).",
        quantization="NVFP4", min_memory_gb=470, family="llama", params_b=405.0,
        proven_tp=4, verified=False,
        context_length=131072, license="Llama 3.1", format="nvfp4",
    ),
    "llama-3.1-405b-awq": ModelInfo(
        id="llama-3.1-405b-awq",
        name="Llama 3.1 405B Instruct (AWQ-INT4)",
        hf_repo="hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
        size_gb=408.0,
        description="Dense 405B, AWQ-INT4. Distributed TP=4.",
        quantization="AWQ", min_memory_gb=440, family="llama", params_b=405.0,
        proven_tp=4, verified=False,
        context_length=131072, license="Llama 3.1", format="awq",
    ),
    "llama-3.3-70b-nvfp4": ModelInfo(
        id="llama-3.3-70b-nvfp4",
        name="Llama 3.3 70B Instruct (NVFP4)",
        hf_repo="nvidia/Llama-3.3-70B-Instruct-NVFP4",
        size_gb=80.0,
        description="Dense 70B, NVFP4. Fits TP=2; bandwidth-bound single-stream on GB10.",
        quantization="NVFP4", min_memory_gb=88, family="llama", params_b=70.0,
        proven_tp=2, verified=True,
        context_length=131072, license="Llama 3.3", recommended=True, format="nvfp4",
    ),
    "glm-5.1": ModelInfo(
        id="glm-5.1",
        name="GLM-5.1",
        hf_repo="zai-org/GLM-5.1",
        size_gb=874.0,
        description="Large GLM. Needs the full cluster's pooled memory (TP=4).",
        quantization=None, min_memory_gb=900, family="glm", params_b=0.0,
        proven_tp=4, verified=False,
        context_length=131072, license="GLM",
    ),
    "gemma4-31b-it-nvfp4": ModelInfo(
        id="gemma4-31b-it-nvfp4",
        name="Gemma 4 31B Instruct (NVFP4)",
        hf_repo="nvidia/Gemma-4-31B-IT-NVFP4",
        # MEASURED, 31 GB on disk — not the 15.5 GB that four bits a parameter
        # would suggest. The checkpoint is not uniformly 4-bit; embeddings and
        # the head carry more. The guess mattered: on bandwidth-bound hardware
        # the weight size IS the decode speed, and half the size predicts twice
        # the tokens per second.
        size_gb=31.0, min_memory_gb=40,
        description=(
            "Dense 31B instruct model in NVFP4. Dense means bandwidth-bound "
            "decode on GB10: ~8 tok/s single-stream, which is the 273 GB/s "
            "memory bus divided by 31 GB of weights, not a misconfiguration. "
            "For an everyday chat model serving several people at once, the "
            "MoE sibling (Gemma 4 26B-A4B) reads a fraction of its weights per "
            "token and is several times faster here. This one is the better "
            "instruction-follower. Fits one node."
        ),
        quantization="NVFP4", family="gemma", params_b=31.0,
        proven_tp=1, verified=False, curated=True,
        context_length=262144, license="Gemma", recommended=False,
        format="nvfp4",
        capabilities=["vision", "audio", "tool_use", "reasoning", "multilingual"],
        # The parsers and loader are the Gemma 4 family's, taken from eugr's
        # recipes/gemma4-26b-a4b-nvfp4.yaml (MIT) — same family, same output
        # format. What is NOT carried over is that recipe's speculative config:
        # its drafter belongs to the 26B-A4B and pairing it with this model
        # would fail in a way that reads like a broken model.
        extra_vllm_args=[
            "--load-format", "instanttensor",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "gemma4",
            "--reasoning-parser", "gemma4",
            # Stated rather than left to the automatic downgrade. That
            # downgrade reads the model's config.json to decide whether it is
            # multimodal, which only works once the weights are on local disk
            # — so a first launch straight from the Hub would have served this
            # vision model on the fp8 default. See the note on the 26B entry.
            "--kv-cache-dtype", "auto",
        ],
        extra_env={
            # Capped. The loader stages weights through one contiguous
            # buffer and asks the driver how much device memory it may use;
            # on GB10 it gets an answer that has nothing to do with the
            # machine. Measured on an idle node with 28 GB genuinely free:
            #
            #   RuntimeError: buffer_size (5086090240 B) exceeds device
            #   memory budget (825161728 B)
            #
            # and 862404608 B on the next attempt — a figure that moves
            # between runs, so it comes from a runtime query rather than from
            # gpu-memory-utilization. nvidia-smi reports [N/A] for memory on
            # this hardware too; unified memory is the common thread.
            #
            # 64 MiB is what the GLM recipe uses and what has loaded a 175 GB
            # checkpoint here. Raise or drop it (drop:--load-format) if a
            # future engine image reports the budget correctly.
            "INSTANTTENSOR_BUFFER_SIZE": "67108864",
        },
        recommended_gmu=0.80,
    ),
    "glm-5.3-flash-nvfp4-spark": ModelInfo(
        id="glm-5.3-flash-nvfp4-spark",
        name="GLM 5.3 Flash (NVFP4, B12X) — 2 nodes",
        hf_repo="local-inference-lab/GLM-5.3-Flash-NVFP4-Spark",
        # Not published with a weight size we can verify; the recipe's own
        # cluster_only + TP=2 + gpu_memory_utilization 0.87 is the honest
        # statement of what it needs — more than one Spark. min_memory_gb says
        # that, size_gb stays 0 rather than inventing a number the fit
        # calculator would then present as fact.
        size_gb=0.0, min_memory_gb=130,
        description=(
            "256K context on an EXPERIMENTAL B12X serving stack, and 175 GB of "
            "weights. Needs its own engine image (vllm-node-b12x) — see "
            "docs/mesh/B12X-IMAGE.md — and exactly two nodes: this architecture "
            "does not implement SupportsPP, so pipeline is out, and tensor "
            "needs a power-of-two rank count. Served here on two Sparks at "
            "TP=2: 1,072,101 tokens of KV cache at 131,072 per request "
            "(8.18x concurrency), gpu_memory_utilization 0.87, "
            "--max-num-seqs 8. It reasons before every reply and the thinking "
            "is the bulk of the output: 979 reasoning tokens measured for "
            "\"count from 1 to 30\", so a client that caps max_tokens low "
            "gets an empty answer and finish_reason=length. Budget tens of "
            "thousands of output tokens, not hundreds. The flags are the "
            "upstream recipe's apart from three the hardware or the "
            "checkpoint contradicted."
        ),
        quantization="NVFP4", family="glm", params_b=0.0,
        # Served on this cluster on 2026-09-14 at TP=2 — the picker can
        # default to it and show the badge.
        proven_tp=2, verified=True, curated=True,
        # Measured: "NotImplementedError: Pipeline parallelism is not supported
        # for this model. Supported models implement the SupportsPP interface."
        # So the only axis is tensor, and tensor needs a power-of-two rank
        # count — which makes exactly two nodes the only shape this model has
        # on a three-node cluster, whatever the memory situation.
        supports_pipeline=False,
        context_length=262144, license="MIT", recommended=False,
        format="nvfp4", capabilities=["tool_use", "reasoning", "code"],
        # Everything below is eugr/spark-vllm-docker's recipes/glm-5.3-flash.yaml
        # (MIT), carried over verbatim apart from the parallelism flags, which
        # AINode derives from the node selection. The b12x backends and load
        # format exist ONLY in the vllm-node-b12x image — running these against
        # the default image fails in argparse.
        engine_image_eugr="vllm-node-b12x",
        extra_vllm_args=[
            "--mamba-cache-mode", "align",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--dtype", "bfloat16",
            "--kv-cache-dtype", "fp8",
            "--quantization", "modelopt_mixed",
            "--attention-backend", "B12X",
            # Required by the model, not a tuning choice. Lowering it to 16 to
            # get off the experimental attention path was refused outright:
            #
            #   ValueError: GLM C4 indexing requires a model block size
            #   divisible by 256
            #
            # So the block size and the attention backend cannot be varied
            # independently while diagnosing this model.
            "--block-size", "256",
            "--moe-backend", "b12x",
            "--linear-backend", "b12x",
            "--no-enable-flashinfer-autotune",
            # Also measured: the b12x fast loader refused to start here with
            #   RuntimeError: the initial b12x loader requires GPU host page
            #   tables
            # and the rest of the B12X stack — attention, MoE, linear — does
            # not depend on it. auto loads the same weights the ordinary way.
            "--load-format", "auto",
            # The recipe asks for 1048576. Two separate reasons not to.
            #
            # First, while the speculative config was still in play vLLM
            # refused it outright — "User-specified max_model_len (1048576) is
            # greater than the derived max_model_len (262144)". That 262144
            # came from the MTP module's own config, and with the speculative
            # config dropped the main architecture resolves instead and
            # accepts 1M. So this is no longer a hard limit.
            #
            # Second, and now the real one: it does not fit. With 1M accepted,
            # a three-node pipeline launch was killed for memory during
            # startup (exit 137) — the context length sizes per-sequence
            # bookkeeping whether or not anyone sends a million tokens.
            # 262144 is a length this cluster can actually hold; raise it in
            # the launch panel and watch the memory if you want more.
            "--max-model-len", "262144",
            "--max-num-seqs", "4",
            "--max-num-batched-tokens", "4096",
            # The recipe's --speculative-config is deliberately absent, and
            # this is measured, not assumed. With it, every launch died in
            #
            #   HFValidationError: Repo id must be in the form 'repo_name' or
            #   'namespace/repo_name': '/models/local-inference-lab--GLM-5.3-…'
            #
            # For built-in MTP vLLM resolves the draft from the same string
            # the model was given, and that string is a local path here —
            # AINode serves a downloaded model from its directory so it does
            # not fetch 175 GB again on every launch. The resolver it reaches
            # accepts only a Hub repo id.
            #
            # Dropping it, the same launch reached multi-modal warmup. Putting
            # it back means serving this model by repo id, which on this setup
            # means downloading it a second time; nobody has shown that the
            # speculative tokens are worth that. Add it in the launch panel to
            # try: --speculative-config '{"method":"mtp",...}'.
            "--reasoning-parser", "glm45",
            "--tool-call-parser", "glm47",
            "--enable-auto-tool-choice",
            # eugr's value, kept — and the arithmetic that briefly argued
            # against it was wrong. Two launches, both logged:
            #
            #   4G, max_model_len 65536, max_num_seqs 4, gmu 0.82
            #       -> GPU KV cache size:   434,176 tokens
            #   8G, max_model_len 131072, max_num_seqs 8, gmu 0.87
            #       -> GPU KV cache size: 1,072,101 tokens
            #
            # Reading 1,072,101 as "about 9.9 GB, so the 8G cap cannot have
            # been in force" assumes a constant bytes-per-token. It is not
            # constant here: this is a hybrid Mamba model launched with
            # --mamba-cache-mode align, so the state cache is sized by
            # max_num_seqs and the attention blocks are aligned, and both of
            # those changed between the two runs. The 8G cap WAS active in
            # the good one.
            #
            # What removing it would do is therefore unmeasured. Raise it in
            # the launch panel and read "GPU KV cache size" back if you want
            # to find out.
            "--kv-cache-memory-bytes", "8G",
        ],
        extra_env={
            "CUTE_DSL_ARCH": "sm_121a",
            "SAFETENSORS_FAST_GPU": "1",
            "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
            "VLLM_ROCE_ALLREDUCE_MAX_SIZE": "2MB",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "VLLM_USE_AOT_COMPILE": "1",
            "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
            "B12X_POLICY_MODE": "auto",
            "INSTANTTENSOR_BACKEND": "BUFFERED",
            "INSTANTTENSOR_BUFFER_SIZE": "67108864",
            "INSTANTTENSOR_CHUNK_SIZE": "8388608",
            "INSTANTTENSOR_CONCURRENCY": "1",
            "INSTANTTENSOR_IO_DEPTH": "3",
        },
        recommended_gmu=0.87,
    ),
    "glm-5.2-reap-504b-nvfp4": ModelInfo(
        id="glm-5.2-reap-504b-nvfp4",
        name="GLM-5.2 NVFP4 REAP-504B",
        hf_repo="madeby561/GLM-5.2-NVFP4-REAP-504B",
        size_gb=309.0,
        description="REAP-pruned GLM-5.2 MoE, NVFP4 for GB10. ~309 GB on disk — needs the cluster's pooled memory (TP=4). DeepSeek Sparse Attention. NOT yet load-tested on GB10.",
        quantization="NVFP4", min_memory_gb=360, family="glm", params_b=504.0,
        proven_tp=4, verified=False,
        context_length=131072, license="MIT", recommended=False, format="nvfp4",
    ),
}


# Backward-compat alias — external code may still import MODEL_CATALOG.
MODEL_CATALOG: dict[str, ModelInfo] = FALLBACK_CATALOG


# ---- Dynamic catalog aggregator --------------------------------------------


class CatalogAggregator:
    """Fetch and merge model metadata from HuggingFace, Ollama, NVIDIA NIM."""

    CACHE_TTL = 86400  # 24 hours
    CACHE_FILE = AINODE_HOME / "catalog-cache.json"

    def fetch(self, force_refresh: bool = False) -> list[ModelInfo]:
        """Fetch the merged catalog. Uses cache if fresh, else all sources."""
        if not force_refresh and self._cache_valid():
            cached = self._load_cache()
            if cached:
                return cached

        models: list[ModelInfo] = []
        models.extend(self._fetch_huggingface_popular(limit=100))
        models.extend(self._fetch_ollama_library())
        models.extend(self._fetch_nvidia_nim())

        # Dedupe by hf_repo (case-insensitive)
        seen: set[str] = set()
        unique: list[ModelInfo] = []
        for m in models:
            key = m.hf_repo.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(m)

        if unique:
            self._save_cache(unique)
        return unique

    # -- Source: HuggingFace Hub ---------------------------------------------

    def _fetch_huggingface_popular(self, limit: int = 100) -> list[ModelInfo]:
        """Top text-generation models on HF Hub by downloads."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return []

        try:
            api = HfApi()
            queries = [
                {"filter": "text-generation", "sort": "downloads", "limit": 50},
                {"filter": "text-generation", "tags": "instruct", "sort": "downloads", "limit": 30},
                {"filter": "text-generation", "tags": "chat", "sort": "downloads", "limit": 20},
            ]
            results: list[ModelInfo] = []
            seen_ids: set[str] = set()
            for q in queries:
                try:
                    iterator = api.list_models(**q)  # `direction` dropped in hub >=1.x
                except Exception:
                    continue
                for m in iterator:
                    if m.id in seen_ids:
                        continue
                    seen_ids.add(m.id)
                    try:
                        results.append(self._hf_to_model_info(m))
                    except Exception:
                        continue
            return results
        except Exception:
            return []

    def _hf_to_model_info(self, m) -> ModelInfo:
        """Convert a HF ModelInfo-like object to our ModelInfo."""
        size_gb = self._estimate_size_gb(m)
        params_b = self._estimate_params(m)
        family = m.id.split("/")[0].lower() if "/" in m.id else "unknown"
        slug = m.id.replace("/", "--").lower()
        name = m.id.split("/")[-1].replace("-", " ")

        card_data = getattr(m, "cardData", None) or {}
        if not isinstance(card_data, dict):
            card_data = {}

        license_str = ""
        raw_license = card_data.get("license", "")
        if isinstance(raw_license, str):
            license_str = raw_license
        elif isinstance(raw_license, list) and raw_license:
            license_str = str(raw_license[0])

        context_length = 0
        for key in ("context_length", "max_position_embeddings"):
            val = card_data.get(key, 0)
            if isinstance(val, (int, float)) and val > 0:
                context_length = int(val)
                break

        downloads = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0

        # Detect capabilities from tags + ID
        tags_raw = (card_data.get("tags", []) if isinstance(card_data, dict) else []) or []
        if not isinstance(tags_raw, list):
            tags_raw = []
        tags_joined = " ".join(str(t) for t in tags_raw).lower() + " " + m.id.lower()
        capabilities = []
        if any(k in tags_joined for k in ("vision", "multimodal", "image", "vlm", "vl-", "vl ")):
            capabilities.append("vision")
        if any(k in tags_joined for k in ("tool", "function-call", "function_call")):
            capabilities.append("tool_use")
        if any(k in tags_joined for k in ("reasoning", "thinking", "r1", "o1", "cot")):
            capabilities.append("reasoning")
        if any(k in tags_joined for k in ("code", "coder", "codellama")):
            capabilities.append("code")
        if any(k in tags_joined for k in ("multilingual", "translation")):
            capabilities.append("multilingual")

        # Architecture + format
        arch = ""
        for a in ("llama", "qwen", "mistral", "mixtral", "phi", "gemma", "deepseek", "yi", "falcon", "mpt"):
            if a in m.id.lower():
                arch = a
                break
        fmt = ""
        if "gguf" in m.id.lower():
            fmt = "GGUF"
        elif "awq" in m.id.lower():
            fmt = "AWQ"
        elif "gptq" in m.id.lower():
            fmt = "GPTQ"
        else:
            fmt = "SafeTensors"

        # Extract ISO timestamp from createdAt or lastModified
        created_at = ""
        for attr in ("createdAt", "created_at", "lastModified", "last_modified"):
            val = getattr(m, attr, None)
            if val:
                # Handle datetime objects and strings
                if hasattr(val, "isoformat"):
                    created_at = val.isoformat()
                else:
                    created_at = str(val)
                break

        return ModelInfo(
            id=slug,
            name=name,
            hf_repo=m.id,
            size_gb=size_gb,
            description=self._derive_description(m),
            quantization=self._detect_quantization(m.id),
            min_memory_gb=max(size_gb * 1.2, 2.0) if size_gb > 0 else 2.0,
            family=family,
            params_b=params_b,
            context_length=context_length,
            license=license_str,
            recommended=self._is_recommended(m.id, downloads),
            created_at=created_at,
            downloads=downloads,
            likes=likes,
            capabilities=capabilities,
            architecture=arch,
            format=fmt,
        )

    # -- Source: HuggingFace trending ---------------------------------------

    def fetch_trending(self, limit: int = 30) -> list[ModelInfo]:
        """Models trending on HF (high download velocity recently)."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # HF's trending signal is exposed as sort="trendingScore"
            models = api.list_models(
                filter="text-generation",
                sort="trendingScore",
                limit=limit,
                direction=-1,
            )
            results: list[ModelInfo] = []
            for m in models:
                try:
                    results.append(self._hf_model_to_info(m))
                except Exception:
                    continue
            return results
        except Exception:
            return []

    # Alias matching task spec naming
    def _hf_model_to_info(self, m) -> ModelInfo:
        return self._hf_to_model_info(m)

    # -- Source: HuggingFace latest (newest releases) -----------------------

    def fetch_latest(self, limit: int = 30) -> list[ModelInfo]:
        """Most recently created text-generation models on HF."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(
                filter="text-generation",
                sort="createdAt",
                limit=limit * 3,  # overfetch because many will lack metadata
                direction=-1,
            )
            results: list[ModelInfo] = []
            for m in models:
                try:
                    info = self._hf_to_model_info(m)
                    # Only keep models with real size/param info or high download count
                    # so we filter out abandoned uploads
                    if info.params_b > 0 or info.downloads > 100:
                        results.append(info)
                    if len(results) >= limit:
                        break
                except Exception:
                    continue
            return results
        except Exception:
            return []

    # -- Source: OpenRouter popular -----------------------------------------

    def fetch_openrouter_popular(self, limit: int = 30) -> list[ModelInfo]:
        """Models ranked by OpenRouter's actual API usage across their network."""
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"User-Agent": "AINode/0.1"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            models: list[ModelInfo] = []
            for m in data.get("data", [])[:limit]:
                hf_repo = m.get("id", "")
                # Skip proprietary ones (openai/, anthropic/, google/gemini)
                if hf_repo.startswith(("openai/", "anthropic/", "google/gemini", "cohere/", "perplexity/")):
                    continue
                context_length = m.get("context_length", 0)
                name = m.get("name", hf_repo)
                slug = hf_repo.replace("/", "--").lower()
                family = hf_repo.split("/")[0].lower() if "/" in hf_repo else ""
                params_b = self._estimate_params_from_name(name)
                size_gb = params_b * 2 if params_b else 0
                models.append(ModelInfo(
                    id=slug,
                    name=name,
                    hf_repo=hf_repo,
                    size_gb=size_gb,
                    description=m.get("description", "OpenRouter-ranked model") or "Text generation model",
                    quantization=None,
                    min_memory_gb=max(size_gb * 1.2, 2.0),
                    family=family,
                    params_b=params_b,
                    context_length=context_length,
                    license="",
                    recommended=True,
                ))
            return models
        except Exception:
            return []

    def _estimate_params_from_name(self, name: str) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', name)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+)\s*[Mm](?![a-zA-Z])', name)
        if match:
            return float(match.group(1)) / 1000
        return 0.0

    # -- Source: Ollama library (live) --------------------------------------

    def fetch_ollama_library(self, limit: int = 30) -> list[ModelInfo]:
        """Ollama's curated library -- scrape their public library page."""
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://ollama.com/api/library",
                headers={"User-Agent": "AINode/0.1", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode()
            try:
                data = json.loads(content)
            except Exception:
                return []
            models: list[ModelInfo] = []
            for item in (data if isinstance(data, list) else [])[:limit]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if not name:
                    continue
                models.append(ModelInfo(
                    id=f"ollama-{name}".lower(),
                    name=name,
                    hf_repo=name,
                    size_gb=0,
                    description=item.get("description", "Ollama library model"),
                    family=name.split(":")[0].lower() if ":" in name else name.lower(),
                    params_b=0,
                    context_length=0,
                    license="",
                    recommended=True,
                ))
            return models
        except Exception:
            return []

    # -- Source: Ollama library ----------------------------------------------

    def _fetch_ollama_library(self) -> list[ModelInfo]:
        """Ollama's curated set. They don't publish a JSON catalog, so we return
        a small hand-curated list that maps Ollama tags to HF repos. The
        aggregator dedupes against HF results by hf_repo, so duplicates are OK.
        """
        try:
            known = [
                ("llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct", 3.21, 6.0, "llama"),
                ("llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct", 8.03, 16.0, "llama"),
                ("qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct", 7.62, 15.0, "qwen"),
                ("mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3", 7.25, 14.0, "mistral"),
                ("gemma2:9b", "google/gemma-2-9b-it", 9.24, 18.5, "gemma"),
                ("phi3:mini", "microsoft/Phi-3-mini-4k-instruct", 3.82, 7.5, "phi"),
                ("codellama:7b", "codellama/CodeLlama-7b-Instruct-hf", 6.74, 13.5, "llama"),
                ("deepseek-r1:7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 7.0, 14.0, "deepseek"),
            ]
            results: list[ModelInfo] = []
            for tag, repo, params_b, size_gb, family in known:
                slug = repo.replace("/", "--").lower()
                results.append(ModelInfo(
                    id=slug,
                    name=repo.split("/")[-1].replace("-", " "),
                    hf_repo=repo,
                    size_gb=size_gb,
                    description=f"Available via Ollama tag '{tag}'.",
                    quantization=self._detect_quantization(repo),
                    min_memory_gb=max(size_gb * 1.2, 2.0),
                    family=family,
                    params_b=params_b,
                    context_length=0,
                    license="",
                    recommended=True,
                ))
            return results
        except Exception:
            return []

    # -- Source: NVIDIA NIM --------------------------------------------------

    def _fetch_nvidia_nim(self) -> list[ModelInfo]:
        """NVIDIA NIM catalog. Public JSON API requires auth, so we return an
        empty list unless we can successfully hit a public endpoint.
        """
        try:
            # Placeholder: NVIDIA's build.nvidia.com catalog requires auth for
            # programmatic access. Return empty to avoid spurious failures.
            return []
        except Exception:
            return []

    # -- Parsing / heuristic helpers -----------------------------------------

    def _estimate_size_gb(self, model) -> float:
        """Estimate on-disk size in GB from safetensors metadata or model id."""
        safetensors = getattr(model, "safetensors", None)
        if safetensors and isinstance(safetensors, dict):
            total = safetensors.get("total", 0)
            if total and total > 0:
                # assume bf16 = 2 bytes/param as a rough disk size
                return round((total * 2) / (1024 ** 3), 1)

        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])', model.id)
        if match:
            params_b = float(match.group(1))
            if re.search(r'awq|gptq|int4|4bit|4-bit', model.id, re.IGNORECASE):
                return round(params_b * 0.6, 1)
            if re.search(r'int8|8bit|8-bit|fp8', model.id, re.IGNORECASE):
                return round(params_b * 1.1, 1)
            return round(params_b * 2, 1)
        return 0.0

    def _estimate_params(self, model) -> float:
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])', model.id)
        if match:
            return float(match.group(1))
        match = re.search(r'(\d+)\s*[Mm](?![a-zA-Z])', model.id)
        if match:
            return float(match.group(1)) / 1000
        return 0.0

    def _detect_quantization(self, model_id: str) -> Optional[str]:
        if re.search(r'awq', model_id, re.IGNORECASE):
            return "awq"
        if re.search(r'gptq', model_id, re.IGNORECASE):
            return "gptq"
        if re.search(r'fp8', model_id, re.IGNORECASE):
            return "fp8"
        if re.search(r'int4|4bit|4-bit', model_id, re.IGNORECASE):
            return "int4"
        if re.search(r'int8|8bit|8-bit', model_id, re.IGNORECASE):
            return "int8"
        if re.search(r'gguf', model_id, re.IGNORECASE):
            return "gguf"
        return None

    def _is_recommended(self, model_id: str, downloads: int) -> bool:
        prefixes = [
            "meta-llama/Llama-3",
            "Qwen/Qwen2.5",
            "Qwen/Qwen3",
            "mistralai/Mistral",
            "google/gemma",
            "microsoft/Phi",
            "microsoft/phi",
            "deepseek-ai/DeepSeek-R1",
        ]
        if not any(model_id.startswith(p) for p in prefixes):
            return False
        if downloads and downloads < 100_000:
            return False
        lower = model_id.lower()
        return ("instruct" in lower) or ("chat" in lower) or lower.endswith("-it")

    def _derive_description(self, model) -> str:
        card = getattr(model, "cardData", None) or {}
        if not isinstance(card, dict):
            card = {}
        tags = card.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        joined_tags = " ".join(str(t).lower() for t in tags)

        pieces: list[str] = []
        if "chat" in joined_tags or "conversational" in joined_tags:
            pieces.append("Conversational model")
        elif "code" in joined_tags:
            pieces.append("Code generation model")
        else:
            pieces.append("Text generation model")

        lang = card.get("language", [])
        if isinstance(lang, list) and lang and "en" not in lang:
            pieces.append(f"Languages: {', '.join(str(x) for x in lang[:3])}")
        return " · ".join(pieces)

    # -- Cache management ----------------------------------------------------

    def _cache_valid(self) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        try:
            age = time.time() - self.CACHE_FILE.stat().st_mtime
            return age < self.CACHE_TTL
        except Exception:
            return False

    def _load_cache(self) -> list[ModelInfo]:
        try:
            data = json.loads(self.CACHE_FILE.read_text())
            return [ModelInfo(**m) for m in data]
        except Exception:
            return []

    def _save_cache(self, models: list[ModelInfo]) -> None:
        try:
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.CACHE_FILE.write_text(
                json.dumps([asdict(m) for m in models], indent=2)
            )
        except Exception:
            pass


# ---- Model manager ---------------------------------------------------------


class ModelManager:
    """Manage model downloads, listing, and deletion against a live catalog."""

    def __init__(self, models_dir: Optional[str | Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._active_downloads: dict[str, dict] = {}
        self._aggregator = CatalogAggregator()
        self._catalog_cache: Optional[dict[str, ModelInfo]] = None

    # -- Catalog access -------------------------------------------------------

    def get_catalog(self, refresh: bool = False) -> list[ModelInfo]:
        """Return the live merged catalog (uses memory + disk cache)."""
        if self._catalog_cache is None or refresh:
            models = self._aggregator.fetch(force_refresh=refresh)
            if not models:
                models = list(FALLBACK_CATALOG.values())
            merged = {m.id: m for m in models}
            # Always merge the curated cluster models — the live HF sweep misses
            # them, so without this they're undiscoverable until already on disk.
            # Skip any whose hf_repo a live entry already covers (don't clobber).
            existing_repos = {m.hf_repo.lower() for m in merged.values()}
            for cid, info in CURATED_CLUSTER_MODELS.items():
                info.curated = True  # mark our hand-picked known-good set
                if cid not in merged and info.hf_repo.lower() not in existing_repos:
                    merged[cid] = info
            self._catalog_cache = merged
        return list(self._catalog_cache.values())

    def get_catalog_map(self, refresh: bool = False) -> dict[str, ModelInfo]:
        """Same as get_catalog but indexed by id."""
        self.get_catalog(refresh=refresh)
        return dict(self._catalog_cache or {})

    def _catalog_lookup(self, model_id: str) -> Optional[ModelInfo]:
        catalog = self.get_catalog_map()
        if model_id in catalog:
            return catalog[model_id]
        # Also allow lookup by hf_repo directly
        for info in catalog.values():
            if info.hf_repo == model_id or info.hf_repo.lower() == model_id.lower():
                return info
        return None

    # -- Catalog queries ------------------------------------------------------

    def list_available(self) -> list[dict]:
        """Return catalog models annotated with download status.

        Also surfaces anything present in models_dir that is NOT in the catalog
        (user-downloaded via HF search or Trending) so the UI never loses
        track of a completed download.
        """
        results = []
        catalog_repos = set()
        for info in self.get_catalog():
            entry = info.to_dict()
            entry["downloaded"] = self._is_downloaded_info(info)
            local_size = self._local_size_gb_info(info)
            if local_size is not None:
                entry["local_size_gb"] = round(local_size, 2)
            catalog_repos.add(info.hf_repo.lower())
            results.append(entry)

        # Merge in downloaded-but-not-in-catalog entries. HF's cache layout is
        # models_dir/hub/models--<org>--<name>/ — skip control dirs like
        # .locks, xet, blobs, snapshots and anything not prefixed `models--`.
        scan_roots = [
            self.models_dir,
            self.models_dir / "hub",
            self.models_dir / "hf-cache" / "hub",  # out-of-band HF_HOME=models/hf-cache downloads
        ]
        seen_slugs: set[str] = set()
        for root in scan_roots:
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not child.name.startswith("models--"):
                    continue  # HF internals: .locks, xet, hub, blobs, snapshots
                if child.name in seen_slugs:
                    continue
                seen_slugs.add(child.name)
                # HF slug "models--org--name" → "org/name"
                hf_repo = child.name[len("models--"):].replace("--", "/", 1)
                if hf_repo.lower() in catalog_repos:
                    continue  # already merged
                results.append({
                    "id": child.name,
                    "slug": child.name,
                    "name": hf_repo.split("/")[-1],
                    "hf_repo": hf_repo,
                    "size_gb": round(self._dir_size_gb(child), 2),
                    "description": "User-downloaded model",
                    "quantization": None,
                    "min_memory_gb": 0,
                    "family": hf_repo.split("/")[0].lower() if "/" in hf_repo else "",
                    "params_b": 0,
                    "context_length": 0,
                    "license": "",
                    "recommended": False,
                    "created_at": "",
                    "downloads": 0,
                    "likes": 0,
                    "capabilities": [],
                    "architecture": "",
                    "format": "",
                    "downloaded": True,
                    "local_size_gb": round(self._dir_size_gb(child), 2),
                })
        return results

    def list_downloaded(self) -> list[dict]:
        """Scan models_dir and return info for every model present on disk.

        Handles three directory layouts:
          1. HF cache:   models_dir/hub/models--org--name/
          2. Flat cache: models_dir/models--org--name/
          3. Direct:     models_dir/org--name/   (written by _run_download_repo)
        """
        downloaded: list[dict] = []
        if not self.models_dir.exists():
            return downloaded

        seen: set[str] = set()

        def _add(child: "Path", hf_repo: str) -> None:
            if hf_repo in seen:
                return
            seen.add(hf_repo)
            catalog_entry = self._find_catalog_by_hf_repo(hf_repo)
            if catalog_entry:
                entry = catalog_entry.to_dict()
                entry["downloaded"] = True
                entry["local_size_gb"] = round(self._dir_size_gb(child), 2)
                downloaded.append(entry)
            else:
                downloaded.append({
                    "id": hf_repo,
                    "name": hf_repo.split("/")[-1] if "/" in hf_repo else hf_repo,
                    "hf_repo": hf_repo,
                    "size_gb": round(self._dir_size_gb(child), 2),
                    "description": "Downloaded model",
                    "quantization": None,
                    "min_memory_gb": 0,
                    "downloaded": True,
                    "local_size_gb": round(self._dir_size_gb(child), 2),
                })

        # Scan top-level models_dir
        for child in sorted(self.models_dir.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("models--"):
                # HF flat cache: models--org--name
                _add(child, name[len("models--"):].replace("--", "/", 1))
            elif "--" in name and not name.startswith(".") and name != "hub":
                # Direct download: org--name
                _add(child, name.replace("--", "/", 1))

        # Also scan nested HF cache layouts: models_dir/hub and the out-of-band
        # models_dir/hf-cache/hub (HF_HOME=models/hf-cache downloads land here).
        for hub in (self.models_dir / "hub", self.models_dir / "hf-cache" / "hub"):
            if hub.is_dir():
                for child in sorted(hub.iterdir()):
                    if child.is_dir() and child.name.startswith("models--"):
                        _add(child, child.name[len("models--"):].replace("--", "/", 1))

        return downloaded

    def _find_catalog_by_hf_repo(self, hf_repo: str):
        """Find a catalog entry by HF repo ID (case-insensitive)."""
        hf_lower = hf_repo.lower()
        for info in self.get_catalog():
            if info.hf_repo.lower() == hf_lower:
                return info
        return None

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Return catalog info for a model, plus local size if downloaded."""
        info = self._catalog_lookup(model_id)
        if info is None:
            return None
        entry = info.to_dict()
        entry["downloaded"] = self._is_downloaded_info(info)
        local_size = self._local_size_gb_info(info)
        if local_size is not None:
            entry["local_size_gb"] = round(local_size, 2)
        return entry

    def recommend_for_gpu(self, gpu_memory_gb: float) -> list[dict]:
        """Return catalog models that fit within the given GPU memory."""
        results = []
        for info in self.get_catalog():
            if info.min_memory_gb <= gpu_memory_gb:
                entry = info.to_dict()
                entry["downloaded"] = self._is_downloaded_info(info)
                results.append(entry)
        results.sort(key=lambda m: m["size_gb"], reverse=True)
        return results

    # -- Download / Delete ----------------------------------------------------

    def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub and return the local path."""
        info = self._catalog_lookup(model_id)
        if info is None:
            raise ValueError(
                f"Unknown model: {model_id}. Use an id from the catalog."
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloads. "
                "Install it with: pip install huggingface_hub"
            )

        local_dir = self.models_dir / self._repo_to_dirname(info.hf_repo)

        download_path = snapshot_download(
            repo_id=info.hf_repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            # Cap parallel file connections so a fat model pull can't monopolise
            # the uplink (ponytail: bounds parallelism, not absolute byte-rate —
            # upgrade to a tc/trickle shaper if a single stream still saturates).
            max_workers=_download_max_workers(),
        )

        # The size on disk just changed under a directory whose own mtime may
        # not have moved.
        self.forget_size(local_dir)
        return Path(download_path)

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model from disk; return True if deleted."""
        info = self._catalog_lookup(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")

        removed = False
        for model_dir in self.model_dirs_for_repo(info.hf_repo):
            shutil.rmtree(model_dir)
            self.forget_size(model_dir)
            removed = True
        return removed

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _repo_to_dirname(hf_repo: str) -> str:
        """Convert 'org/model-name' to 'org--model-name' for filesystem safety."""
        return hf_repo.replace("/", "--")

    def other_owners_on_disk(self, hf_repo: str) -> list[str]:
        """Repos on disk with this repo's name but a different owner.

        Upstream renames move a model between accounts —
        demon-zombie/MiniMax-M2.7-AWQ-4bit now 307-redirects to
        et0dev/MiniMax-M2.7-AWQ-4bit. huggingface_hub follows the redirect, so
        the cache directory carries the NEW owner while a catalog entry (or a
        page the operator still has open) names the old one. Deleting then
        reports

            Model not downloaded: demon-zombie/MiniMax-M2.7-AWQ-4bit

        about an aborted download that is plainly on the disk, under a name
        nothing in the UI shows. Naming it is the whole fix: the operator can
        then delete the thing that actually exists.
        """
        name = hf_repo.split("/")[-1].strip().lower()
        if not name:
            return []
        found = set()
        for entry in self.list_downloaded():
            repo = str(entry.get("hf_repo") or "")
            if not repo or repo.lower() == hf_repo.lower():
                continue
            if repo.split("/")[-1].lower() == name:
                found.add(repo)
        return sorted(found)

    def model_dirs_for_repo(self, hf_repo: str) -> list[Path]:
        """Every directory a copy of ``hf_repo`` can occupy, that exists.

        Weights arrive by several routes and land in different layouts: the
        UI's file-by-file download writes ``models_dir/<org>--<name>``, while
        anything going through huggingface_hub's cache — an out-of-band pull,
        an engine that resolved the repo itself, an aborted download — writes
        ``models--<org>--<name>`` under ``models_dir``, ``models_dir/hub`` or
        ``models_dir/hf-cache/hub``.

        list_downloaded() has always scanned all four. Deleting knew only the
        first, so a model the UI listed could not be removed through the UI:

            Delete failed: Model not downloaded: nvidia/Gemma-4-26B-A4B-NVFP4

        One definition, used by both, is the only way those two stay in
        agreement.
        """
        slug = self._repo_to_dirname(hf_repo)
        hf_slug = f"models--{slug}"
        candidates = [
            self.models_dir / slug,
            self.models_dir / hf_slug,
            self.models_dir / "hub" / hf_slug,
            self.models_dir / "hf-cache" / "hub" / hf_slug,
        ]
        return [path for path in candidates if path.is_dir()]

    def _model_dir_info(self, info: ModelInfo) -> Path:
        return self.models_dir / self._repo_to_dirname(info.hf_repo)

    def _find_model_dir(self, info: ModelInfo) -> Optional[Path]:
        """Return the on-disk dir for a model across every layout we support.

        A model can live as: direct ``org--name`` (our downloader), flat HF
        ``models--org--name``, HF cache ``hub/models--org--name``, or out-of-band
        ``hf-cache/hub/models--org--name`` (HF_HOME downloads). Catalog entries
        (incl. the curated cluster models) must detect all of them — otherwise an
        on-disk model reads as "not downloaded". Mirrors the list_available scan.
        """
        hf_slug = "models--" + info.hf_repo.replace("/", "--")
        candidates = [
            self.models_dir / self._repo_to_dirname(info.hf_repo),  # org--name
            self.models_dir / hf_slug,
            self.models_dir / "hub" / hf_slug,
            self.models_dir / "hf-cache" / "hub" / hf_slug,
        ]
        for d in candidates:
            if d.exists() and any(d.iterdir()):
                return d
        return None

    def _is_downloaded_info(self, info: ModelInfo) -> bool:
        return self._find_model_dir(info) is not None

    def _local_size_gb_info(self, info: ModelInfo) -> Optional[float]:
        d = self._find_model_dir(info)
        return self._dir_size_gb(d) if d else None

    #: Directory → (mtime, size). A model directory is written once and then
    #: only read, so its size is a constant until something changes it — and
    #: the walk that produces it is expensive enough to have been the reason
    #: the Models page took seconds to open. list_available() called it for
    #: every catalog entry on disk and list_downloaded() twice per model, on
    #: every single request, from six call sites in the UI. Hundreds of
    #: gigabytes of stat() per page view, which also evicted the page cache
    #: the next model launch was going to want.
    _SIZE_CACHE: dict = {}

    #: An entry also expires on time, not only when the directory's mtime
    #: moves. A file rewritten inside an existing tree can leave the parent
    #: untouched, and two writes inside one filesystem clock tick are
    #: indistinguishable — mtime is a cheap hint, not a guarantee. Writers
    #: call forget_size() as well; this is the backstop for everything that
    #: changes the tree without going through them.
    _SIZE_TTL_SECONDS = 60.0

    @classmethod
    def _dir_size_gb(cls, path: Path) -> float:
        key = str(path)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return 0.0
        now = time.time()
        cached = cls._SIZE_CACHE.get(key)
        if (cached is not None and cached[0] == stamp
                and now - cached[2] < cls._SIZE_TTL_SECONDS):
            return cached[1]
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size = total / (1024**3)
        cls._SIZE_CACHE[key] = (stamp, size, now)
        return size

    @classmethod
    def forget_size(cls, path=None) -> None:
        """Drop a cached size. Called after a download or a delete, where the
        directory's own mtime may not move — a file rewritten inside an
        existing tree leaves the parent untouched on some filesystems."""
        if path is None:
            cls._SIZE_CACHE.clear()
            return
        prefix = str(path)
        for key in [k for k in cls._SIZE_CACHE if k.startswith(prefix)]:
            del cls._SIZE_CACHE[key]

    #: Pipeline tags this engine can serve. A vision-language model is a text
    #: generator that also takes pictures, and vLLM treats it as one — but the
    #: Hub files it under image-text-to-text, so filtering on text-generation
    #: alone hid every multimodal model from the search. MiniMax M3 and its
    #: quantisations are tagged that way; an operator looking for one found
    #: nothing and reasonably concluded it did not exist.
    #:
    #: One query per tag, merged: list_models takes a single pipeline_tag, and
    #: dropping the filter entirely would bury the results under embeddings
    #: and classifiers that this node cannot run at all.
    SERVABLE_PIPELINE_TAGS = ("text-generation", "image-text-to-text")

    @staticmethod
    def _search_every_servable_tag(api, *, query, limit, **kwargs):
        """One search per servable pipeline tag, merged and download-sorted.

        A tag that errors is skipped rather than failing the search: the Hub
        has renamed pipeline tags before, and one unknown name should not cost
        the results of the others.
        """
        found: dict = {}
        for tag in ModelManager.SERVABLE_PIPELINE_TAGS:
            try:
                for model in api.list_models(search=query, pipeline_tag=tag,
                                             limit=limit, **kwargs):
                    found.setdefault(model.id, model)
            except Exception:
                logger.debug("search failed for pipeline_tag=%s", tag, exc_info=True)
        ranked = sorted(found.values(),
                        key=lambda m: getattr(m, "downloads", 0) or 0, reverse=True)
        return ranked[:limit]

    def search_huggingface(self, query: str, limit: int = 50) -> list[dict]:
        """Search HuggingFace Hub for models this engine could serve."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # huggingface_hub >=1.x dropped `direction`/`task`; use pipeline_tag.
            # expand=safetensors pulls the dtype breakdown so we can show real size.
            models = self._search_every_servable_tag(
                api, query=query, limit=limit,
                # NOT usedStorage: the Hub rejects it on the LIST endpoint —
                #   Invalid option: expected one of "author"|…|"safetensors"|…
                # It is valid on the single-model endpoint, which is where it
                # was verified, and adding it here turned every search into a
                # BadRequestError: no results, so nothing to download. The
                # exact size is fetched per repo below instead.
                expand=["safetensors"],
            )
            catalog_repos = {info.hf_repo.lower() for info in self.get_catalog()}
            results = []
            for m in models:
                repo = m.id
                repo_l = repo.lower()
                slug = repo.replace("/", "--").lower()
                size_gb = repo_size_gb(m)
                sf = getattr(m, "safetensors", None)
                total_params = getattr(sf, "total", 0) if sf else 0
                # Quant/engine from repo name — drives the badge AND the
                # "can it run on vLLM/GB10" filter (MLX=Apple, GGUF=llama.cpp).
                quant = ""
                for tag, label in (
                    ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("w4afp8", "W4AFP8"),
                    ("w4a16", "W4A16"), ("awq", "AWQ"), ("gptq", "GPTQ"),
                    ("int4", "INT4"), ("int8", "INT8"), ("fp8", "FP8"),
                    ("gguf", "GGUF"), ("mlx", "MLX"), ("bf16", "BF16"), ("fp16", "FP16"),
                ):
                    if tag in repo_l:
                        quant = label
                        break
                vllm_ok = not ("mlx" in repo_l or "gguf" in repo_l or "ggml" in repo_l)
                results.append({
                    "id": slug,
                    "name": repo.split("/")[-1],
                    "hf_repo": repo,
                    "size_gb": round(size_gb, 1),
                    "description": (m.pipeline_tag or "text-generation") + " model",
                    "family": repo.split("/")[0].lower(),
                    "params_b": round(total_params / 1e9, 1) if total_params else 0,
                    "context_length": 0,
                    "license": "",
                    "recommended": False,
                    "quant": quant,
                    "vllm_ok": vllm_ok,
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "in_catalog": repo_l in catalog_repos,
                })
            # Sharpen the sizes that decide a fit verdict. The dtype estimate
            # over-states a packed low-bit checkpoint — measured at 8x for an
            # AWQ 4-bit repo — and the UI hides anything it reads as too large
            # for the cluster, so an estimate is the difference between a
            # frontier model being offered and not existing. Only the large
            # ones, only a bounded number, and in parallel: a search box that
            # takes half a minute is a search box nobody uses.
            _sharpen_sizes(results)

            # Always surface curated matches for the query — HF's download-sorted
            # page often ranks our vetted pick past the limit, so inject it.
            ql = query.lower()
            have = {r["hf_repo"].lower() for r in results}
            for info in self.get_catalog():
                if not getattr(info, "curated", False):
                    continue
                hay = (info.hf_repo + " " + info.name + " " + (info.family or "")).lower()
                if ql in hay and info.hf_repo.lower() not in have:
                    results.append({
                        "id": info.hf_repo.replace("/", "--").lower(),
                        "name": info.name,
                        "hf_repo": info.hf_repo,
                        "size_gb": info.size_gb,
                        "description": info.description,
                        "family": info.family,
                        "params_b": info.params_b,
                        "context_length": info.context_length,
                        "license": info.license,
                        "recommended": info.recommended,
                        "quant": (info.quantization or info.format or "").upper(),
                        "vllm_ok": True,
                        "proven_tp": info.proven_tp,
                        "downloads": 0,
                        "likes": 0,
                        "in_catalog": True,
                    })
            # Vetted (in-catalog) pick first, then most-downloaded.
            results.sort(key=lambda r: (not r["in_catalog"], -(r["downloads"] or 0)))
            return results
        except Exception as e:
            logger.warning("HuggingFace search failed for %r: %s", query, e)
            return []

    def _find_catalog_by_dir(self, dirname: str) -> Optional[ModelInfo]:
        for info in self.get_catalog():
            if ModelManager._repo_to_dirname(info.hf_repo) == dirname:
                return info
        return None
