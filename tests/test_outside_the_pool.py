"""What a launch holds beyond its memory fraction — planned and admitted alike.

Reported: the profile's nvidia/Qwen3.8-Flash-Next-NVFP4 (planned by the
wizard at 0.84) was refused at start with "cost 115 GB the last time it ran
here, and the roomiest node has 111 GB free". It had held 114.6 GB on a
130.7 GB node: 109.8 of pool and 4.8 beyond it, which the wizard never
counted.
"""

from __future__ import annotations

import pytest

from ainode.planner.household import HouseholdNode, _outside_pool_gb

_ROW = {"model": "q", "node_id": "a", "gpu_memory_utilization": 0.84, "memory_gb": 114.6,
        "node_total_gb": 130.7, "memory_by_node": {"a": 114.6, "b": 113.7},
        "launch": {"nodes": 2}}


def _nodes():
    return {n: HouseholdNode(node_id=n, total_gb=130.7) for n in ("a", "b", "c")}


class TestWhatTheWizardCounts:
    def test_the_measured_excess_over_the_pool(self):
        assert _outside_pool_gb([_ROW], ["a", "b"], _nodes(), 2) == pytest.approx(4.8, abs=0.05)

    def test_another_split_says_nothing(self):
        assert _outside_pool_gb([_ROW], ["a"], _nodes(), 1) == 0.0

    def test_a_row_without_the_nodes_total_uses_the_nodes(self):
        row = dict(_ROW, node_total_gb=0.0)
        assert _outside_pool_gb([row], ["a", "b"], _nodes(), 2) == pytest.approx(4.8, abs=0.05)

    def test_under_the_pool_is_nothing(self):
        row = dict(_ROW, memory_by_node={"a": 100.0})
        assert _outside_pool_gb([row], ["a", "b"], _nodes(), 2) == 0.0

    def test_no_measurement_no_excess(self):
        assert _outside_pool_gb(None, ["a", "b"], _nodes(), 2) == 0.0


class TestWhatTheGateCompares:
    def _app(self, free_gb):
        class _Budget:
            total_gb = 130.7
            usable_gb = free_gb
        return _Budget

    def test_a_smaller_fraction_needs_less(self, monkeypatch):
        from ainode.safety import admission

        budget = self._app(111.0)
        monkeypatch.setattr(admission, "_budgets_with_reserve", lambda app, ids=None: [budget])
        measured = {"memory_gb": 114.6, "gpu_memory_utilization": 0.84,
                    "node_total_gb": 130.7, "max_model_len": 262144}
        # At its old fraction it does not fit; at 0.80 it does.
        assert admission._measured_says({}, "q", measured, 262144, None, 0.84)
        assert admission._measured_says({}, "q", measured, 262144, None, 0.80) == ""

    def test_without_a_fraction_the_measurement_stands(self, monkeypatch):
        from ainode.safety import admission

        budget = self._app(111.0)
        monkeypatch.setattr(admission, "_budgets_with_reserve", lambda app, ids=None: [budget])
        measured = {"memory_gb": 114.6, "gpu_memory_utilization": 0.84}
        assert admission._measured_says({}, "q", measured, 0, None, None)
