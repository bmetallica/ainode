"""An engine may not ask for memory the machine does not have.

Reported from the cluster, after a two-node launch:

    wenn ich z.b. minimax3 auf node 1 und node2 starte läuft der speicher
    sofort auf 122gb voll und die sparks schmieren komplett ab

122 GB of 128 is what gpu_memory_utilization=0.97 means on unified memory —
and the model card for that checkpoint recommends exactly that, because it
was written for a machine whose GPU memory is not also the operating
system's. The memory guard cannot save this: the allocation crosses its
reserve faster than any poll interval, so the launch has to be the thing that
does not ask.
"""

from __future__ import annotations

from ainode.planner.compute import NodeBudget
from ainode.safety.utilization import (
    MIN_UTILIZATION,
    cap_utilization,
    utilization_ceiling,
)


class _App(dict):
    pass


def _app(monkeypatch, budgets):
    monkeypatch.setattr("ainode.safety.admission._budgets_with_reserve",
                        lambda app, node_ids=None: list(budgets))
    return _App()


def _spark(free_gb, node="spark-1", total=128.0):
    return NodeBudget(node_id=node, name=node, total_gb=total, free_gb=free_gb)


class TestTheCeiling:
    def test_an_idle_spark_lands_near_the_proven_value(self, monkeypatch):
        # 0.85 is the figure this cluster has been serving MoE at; the
        # ceiling arriving in that region is the arithmetic agreeing with
        # what the hardware already told us.
        ceiling, where = utilization_ceiling(_app(monkeypatch, [_spark(120.0)]))
        assert 0.80 <= ceiling <= 0.92
        assert where == "spark-1"

    def test_a_busy_node_gives_less(self, monkeypatch):
        ceiling, _ = utilization_ceiling(_app(monkeypatch, [_spark(40.0)]))
        assert 0.25 <= ceiling <= 0.32

    def test_the_tightest_node_decides(self, monkeypatch):
        # Every rank of a tensor-parallel launch gets the same fraction.
        app = _app(monkeypatch, [_spark(120.0, "spark-1"), _spark(30.0, "spark-2")])
        ceiling, where = utilization_ceiling(app)
        assert where == "spark-2"
        assert ceiling < 0.3

    def test_no_budgets_is_silence(self, monkeypatch):
        assert utilization_ceiling(_app(monkeypatch, [])) == (0.0, "")


class TestTheCap:
    def test_the_card_recommended_value_is_lowered(self, monkeypatch):
        app = _app(monkeypatch, [_spark(120.0)])
        value, note = cap_utilization(app, 0.97)
        assert value < 0.97
        assert "takes the node down" in note
        assert "force" in note

    def test_something_that_fits_is_left_alone(self, monkeypatch):
        app = _app(monkeypatch, [_spark(120.0)])
        assert cap_utilization(app, 0.5) == (0.5, "")

    def test_force_is_obeyed(self, monkeypatch):
        # An operator who knows what a checkpoint needs is allowed to be right.
        app = _app(monkeypatch, [_spark(20.0)])
        assert cap_utilization(app, 0.97, force=True) == (0.97, "")

    def test_it_never_caps_into_uselessness(self, monkeypatch):
        # Below this the engine cannot hold its own weights; refusing in the
        # admission gate says why, a 0.02 launch just fails strangely.
        app = _app(monkeypatch, [_spark(1.0)])
        value, _ = cap_utilization(app, 0.9)
        assert value == MIN_UTILIZATION

    def test_nothing_requested_is_nothing_changed(self, monkeypatch):
        assert cap_utilization(_app(monkeypatch, [_spark(120.0)]), None) == (None, "")

    def test_an_unreadable_cluster_does_not_block_a_launch(self, monkeypatch):
        assert cap_utilization(_App(), 0.9) == (0.9, "")


class TestBothLaunchPathsApplyIt:
    def test_the_solo_path_does(self):
        import inspect

        from ainode.models.api_routes import append_solo_instance

        source = inspect.getsource(append_solo_instance)
        assert "cap_utilization" in source
        # After the existing instance is stopped: a reload frees what it held.
        assert source.index("existing.backend.stop()") < source.index("cap_utilization")

    def test_the_distributed_path_does(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "cap_utilization" in source
        # Across every participating node, head included.
        assert "[config.node_id] + [n.node_id for n in chosen]" in source

    def test_the_operator_is_told(self):
        import inspect

        from ainode.engine import sharding_routes

        source = inspect.getsource(sharding_routes.handle_sharding_launch)
        assert "plan_note" in source.split("cap_note")[2]
