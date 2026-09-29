"""A rolling update does not mix units.

R4 in upgrade-fixes.md. Since #212 a node renders its memory as decimal GB
(128); one not yet updated renders it as binary (119). During
update-cluster.sh the head sees both at once. Every node has always sent the
raw MiB as well, so the readers take that and convert it themselves.
"""

from __future__ import annotations

import pytest

from ainode.core.units import node_total_gb, node_used_gb


class _Node:
    def __init__(self, gb=0.0, total_mb=0.0, used_mb=0.0):
        self.gpu_memory_gb = gb
        self.gpu_memory_total_mb = total_mb
        self.gpu_memory_used_mb = used_mb


def test_old_and_new_nodes_read_the_same():
    raw = 122_000.0                                  # MiB, as psutil says
    updated = _Node(gb=127.9, total_mb=raw)
    not_yet = _Node(gb=119.1, total_mb=raw)
    assert node_total_gb(updated) == node_total_gb(not_yet)
    assert node_total_gb(updated) == pytest.approx(127.9, abs=0.1)


def test_the_rendering_is_only_a_fallback():
    assert node_total_gb(_Node(gb=128.0)) == 128.0


def test_used_is_decimal_too():
    assert node_used_gb(_Node(used_mb=1024)) == pytest.approx(1.074, abs=0.001)


def test_the_relaunch_check_does_not_mix_units():
    from ainode.engine import sharding_routes

    source = open(sharding_routes.__file__).read()
    assert "/ 1024.0" not in source.split("def free_gb")[1].split("short =")[0]
