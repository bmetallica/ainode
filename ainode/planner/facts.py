"""What a checkpoint says about itself.

The catalog carries a recipe per model, and the recipe is right — but it was
hand-computed, once, by someone reading the checkpoint's ``config.json`` and
doing the arithmetic. That works for the handful of curated entries and for
nothing else: the moment an operator downloads a model AINode has never seen,
every number the UI shows about it is a guess from the repo name.

The checkpoint already carries the facts. Layer count, KV head count, head
dimension and context window are in ``config.json``; the true weight size is
the bytes on disk, which is the only figure that survives quantisation — a
4-bit repo of a 230B model is 130 GB, and no arithmetic on the parameter
count gets there.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ModelFacts", "snapshot_dir", "read_config", "facts_from_config",
           "weight_bytes_on_disk", "local_facts"]

#: Weight files, by extension. ``.bin`` is the old torch format; ``.gguf`` is
#: not servable here but its presence explains an otherwise empty directory.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")

#: Bytes per element, by the name the checkpoint uses for its dtype.
_DTYPE_BYTES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "float16": 2, "half": 2, "fp16": 2, "bf16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "fp8": 1, "int8": 1, "uint8": 1,
}


@dataclass
class ModelFacts:
    """Everything the planner needs about one checkpoint."""

    repo: str = ""
    architecture: str = ""
    num_layers: int = 0
    #: Layers that keep a per-token KV cache. Equal to ``num_layers`` for an
    #: ordinary transformer, smaller for a hybrid that replaces most attention
    #: with a recurrent block whose state is per-sequence, not per-token.
    attention_layers: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    torch_dtype: str = ""
    quantization: str = ""
    #: DeepSeek-style multi-head latent attention caches one compressed vector
    #: per layer instead of a K and a V per head, which is a different formula
    #: entirely — a tenth of the bytes, or less.
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    is_moe: bool = False
    num_experts: int = 0
    experts_per_token: int = 0
    #: True when the layer stack mixes attention with something else. The
    #: token arithmetic still holds for the attention layers, but the
    #: recurrent state is charged per sequence, so the cache does not scale
    #: with context the way this planner's headline number implies.
    is_hybrid: bool = False
    weight_bytes: int = 0
    #: Set when a fact could not be read rather than being genuinely absent.
    unknown: List[str] = field(default_factory=list)

    @property
    def weights_gb(self) -> float:
        return self.weight_bytes / 1e9

    @property
    def dtype_bytes(self) -> int:
        return _DTYPE_BYTES.get(str(self.torch_dtype).lower(), 2)

    @property
    def usable(self) -> bool:
        """Enough to compute a KV figure at all."""
        return bool(self.attention_layers and self.num_kv_heads and self.head_dim) \
            or bool(self.attention_layers and self.kv_lora_rank)


def snapshot_dir(path: Path) -> Optional[Path]:
    """The directory that actually holds ``config.json``.

    A direct download is flat; anything that went through huggingface_hub
    nests the real files under ``snapshots/<hash>/``. (``training/engine.py``
    applies the same rule to decide what to mount; this one is about reading,
    and neither should grow a dependency on the other.)
    """
    if not path.is_dir():
        return None
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        for sub in sorted(s for s in snapshots.iterdir() if s.is_dir()):
            if (sub / "config.json").is_file():
                return sub
    return None


def read_config(directory: Path) -> dict:
    """``config.json`` from a model directory, or {}."""
    resolved = snapshot_dir(Path(directory))
    if resolved is None:
        return {}
    try:
        return json.loads((resolved / "config.json").read_text())
    except (OSError, ValueError):
        logger.debug("could not read config.json in %s", resolved, exc_info=True)
        return {}


def weight_bytes_on_disk(directory: Path) -> int:
    """Total bytes of weight files, following the hub cache's symlinks.

    The only size that is true after quantisation. A packed 4-bit checkpoint
    of a 230B model is about 130 GB; every estimate from the parameter count
    lands a factor of two to eight away.
    """
    resolved = snapshot_dir(Path(directory)) or Path(directory)
    if not resolved.is_dir():
        return 0
    total = 0
    for entry in resolved.rglob("*"):
        if entry.suffix not in _WEIGHT_SUFFIXES:
            continue
        try:
            # stat() rather than lstat(): in the hub layout every file here is
            # a symlink into ../../blobs, and the link itself is 100 bytes.
            total += entry.stat().st_size
        except OSError:
            continue
    return total


def _first(config: dict, *names, default=0):
    for name in names:
        value = config.get(name)
        if value not in (None, ""):
            return value
    return default


def _text_config(config: dict) -> dict:
    """The language half of a multimodal config.

    A vision-language checkpoint puts the layer and head counts under
    ``text_config``, leaving the top level with none — which reads as a model
    with zero layers and an unplannable KV cache.
    """
    for key in ("text_config", "language_config", "llm_config"):
        inner = config.get(key)
        if isinstance(inner, dict) and inner.get("num_hidden_layers"):
            return inner
    return config


def _attention_layers(config: dict, num_layers: int) -> tuple:
    """(layers that cache per token, whether the stack is hybrid).

    Hybrid stacks describe themselves in one of several ways, and the
    difference matters: charging every layer of a mostly-recurrent model for a
    per-token KV cache overstates the cache by an order of magnitude, and
    charging none of them understates it.
    """
    types = config.get("layer_types")
    if isinstance(types, list) and types:
        full = sum(1 for t in types if "full" in str(t).lower()
                   or str(t).lower() in ("attention", "attn", "full_attention"))
        if full and full != len(types):
            return full, True
        return (full or num_layers), False

    # MiniMax records the same thing as a list of ints, 1 = full attention.
    attn_list = config.get("attn_type_list") or config.get("attention_type_list")
    if isinstance(attn_list, list) and attn_list:
        full = sum(1 for t in attn_list if int(t or 0) == 1)
        if full and full != len(attn_list):
            return full, True

    # Qwen3-Next / GLM style: every Nth layer is full attention.
    interval = _first(config, "full_attention_interval", "attn_interval", default=0)
    try:
        interval = int(interval)
    except (TypeError, ValueError):
        interval = 0
    if interval > 1 and num_layers:
        return max(1, num_layers // interval), True

    if any(k in config for k in ("mamba_d_state", "linear_attn_config",
                                 "ssm_cfg", "mamba_expand")):
        # Hybrid, but it does not say which layers. Charge them all: too big a
        # cache estimate refuses a launch that would have worked, too small a
        # one promises a context the engine cannot hold.
        return num_layers, True
    return num_layers, False


def facts_from_config(config: dict, repo: str = "",
                      weight_bytes: int = 0) -> ModelFacts:
    """Turn a checkpoint's config.json into the facts the planner uses."""
    facts = ModelFacts(repo=repo, weight_bytes=int(weight_bytes or 0))
    if not config:
        facts.unknown.append("config.json")
        return facts

    architectures = config.get("architectures") or []
    facts.architecture = str(architectures[0]) if architectures else ""

    text = _text_config(config)
    facts.num_layers = int(_first(text, "num_hidden_layers", "n_layer",
                                  "num_layers", default=0) or 0)
    facts.num_attention_heads = int(_first(text, "num_attention_heads", "n_head",
                                           default=0) or 0)
    facts.num_kv_heads = int(_first(text, "num_key_value_heads",
                                    "num_kv_heads", "n_kv_heads",
                                    default=facts.num_attention_heads) or 0)
    facts.hidden_size = int(_first(text, "hidden_size", "d_model",
                                   "n_embd", default=0) or 0)
    facts.head_dim = int(_first(text, "head_dim", "attention_head_dim",
                                default=0) or 0)
    if not facts.head_dim and facts.hidden_size and facts.num_attention_heads:
        facts.head_dim = facts.hidden_size // facts.num_attention_heads
    facts.vocab_size = int(_first(text, "vocab_size", default=0) or 0)
    facts.max_position_embeddings = int(
        _first(text, "max_position_embeddings", "max_sequence_length",
               "n_positions", "seq_length", default=0) or 0)
    facts.torch_dtype = str(_first(config, "torch_dtype", "dtype", default="") or "")

    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        facts.quantization = str(
            _first(quant, "quant_method", "quant_algo", default="") or "")

    facts.kv_lora_rank = int(_first(text, "kv_lora_rank", default=0) or 0)
    facts.qk_rope_head_dim = int(_first(text, "qk_rope_head_dim", default=0) or 0)
    if not facts.kv_lora_rank and facts.qk_rope_head_dim:
        # DeepSeek-V4 writes the latent geometry without naming it: one KV
        # "head" whose head_dim IS the compressed rank, beside a rope split.
        # Read as grouped-query attention that costs 2 x heads x head_dim per
        # layer, which is the K and the V counted separately — and in MLA
        # there is one latent, not two tensors. On DeepSeek-V4-Flash that is
        # 1024 bytes per layer per token against a measured 584 (the latent at
        # one byte per element plus its scales; MiaAI-Lab's DGX Spark recipe,
        # docs/PATCHES.md issue #22, MIT). Nearly twice the cache reserved,
        # which comes straight off the context the planner will offer.
        #
        # The remaining 8 bytes are the per-token scale block, which this does
        # not model: 576 against 584 is 1.4% short, and ENGINE_OVERHEAD_GB
        # covers that many times over. Fitting a scale factor to one measured
        # checkpoint would be the worse error.
        head_dim = int(_first(text, "head_dim", default=0) or 0)
        kv_heads = int(_first(text, "num_key_value_heads", default=0) or 0)
        if kv_heads == 1 and head_dim >= 256:
            facts.kv_lora_rank = head_dim

    facts.num_experts = int(_first(
        text, "num_experts", "num_local_experts", "n_routed_experts",
        "moe_num_experts", default=0) or 0)
    facts.experts_per_token = int(_first(
        text, "num_experts_per_tok", "moe_topk", "num_experts_per_token",
        default=0) or 0)
    facts.is_moe = bool(facts.num_experts)

    facts.attention_layers, facts.is_hybrid = _attention_layers(
        text, facts.num_layers)

    for name, value in (("num_hidden_layers", facts.num_layers),
                        ("num_key_value_heads", facts.num_kv_heads),
                        ("head_dim", facts.head_dim)):
        if not value:
            facts.unknown.append(name)
    return facts


def local_facts(manager, repo: str) -> ModelFacts:
    """Facts for a model already on this node's disk.

    Local only, deliberately: the planner runs before a launch, on a head that
    may have no route to the Hub, and a plan that needs the network to appear
    is a plan that is unavailable exactly when a download has just failed.
    """
    directories = []
    try:
        directories = manager.model_dirs_for_repo(repo)
    except Exception:
        logger.debug("could not locate %s on disk", repo, exc_info=True)
    for directory in directories:
        config = read_config(directory)
        if config:
            return facts_from_config(config, repo=repo,
                                     weight_bytes=weight_bytes_on_disk(directory))
    facts = ModelFacts(repo=repo)
    facts.unknown.append("not on local disk")
    return facts
