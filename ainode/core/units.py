"""One unit for memory, and a name that says which.

The planner compared two numbers that were not in the same unit. Weights come
off the disk and out of the Hub in decimal bytes::

    weight_bytes_on_disk(directory) / 1e9      # decimal GB

while the node budgets came from psutil through a MiB conversion and a divide
by 1024::

    free_gb = (total_mb - used_mb) / 1024      # GiB, called _gb

Both were labelled ``_gb`` and subtracted from each other. The error is 7.4%,
always in the same direction — the weights look bigger than the memory — and it
lands entirely on the KV cache, because the cache is what is left after the
weights.

Measured on this cluster, planning Qwen3-Coder-Next bf16 across two nodes with
nothing else loaded:

    Weights: 159.4 GB on disk, split 2 ways = 83.7 GB per node
    Left for KV: 4.1 GB across 2 node(s)

159.4 decimal GB is 148.5 GiB. Per node that is 77.9 GiB against the 83.7 the
planner reserved — 5.7 GiB too much, on a plan that had 2.05 GiB per node left
for the cache. The cache was not nearly gone because the nodes were full. It
was nearly gone because two thirds of it had been spent on a rounding
convention.

Decimal is the convention here, because it is the one the model files are
measured in: the Hub's ``usedStorage``, ``weight_bytes_on_disk``,
``_dir_size_gb`` and every model card. ``free -g`` prints GiB and will read
about 7% lower than these figures; that is the tradeoff, and it is the right
way round — a plan that thinks a model is bigger than it is refuses launches
that fit, and a plan that thinks a node is bigger than it is takes the node
down.

The memory guard keeps its own GiB arithmetic (``/proc/meminfo`` is in kB and
its thresholds were tuned there). Where a guard figure enters a plan, it is
converted explicitly rather than assumed — see ``gb_from_gib``.
"""

from __future__ import annotations

__all__ = ["BYTES_PER_GB", "BYTES_PER_GIB", "gb_from_bytes", "gb_from_mib",
           "gb_from_gib", "gib_from_gb"]

BYTES_PER_GB = 1_000_000_000
BYTES_PER_GIB = 1024 ** 3


def gb_from_bytes(value) -> float:
    """Decimal GB from a byte count."""
    return float(value or 0) / BYTES_PER_GB


def gb_from_mib(value) -> float:
    """Decimal GB from a count of MiB — what psutil and NVML hand over."""
    return float(value or 0) * (1024 * 1024) / BYTES_PER_GB


def gb_from_gib(value) -> float:
    """Decimal GB from GiB. For a threshold that was set in GiB."""
    return float(value or 0) * BYTES_PER_GIB / BYTES_PER_GB


def gib_from_gb(value) -> float:
    """GiB from decimal GB, for comparing against ``free -g``."""
    return float(value or 0) * BYTES_PER_GB / BYTES_PER_GIB
