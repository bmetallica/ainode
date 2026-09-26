"""One panel, three numbers, three disagreements.

Reported from the cluster, the launch panel for Qwen/Qwen3-Coder-Next:

    ○ Qwen3-Coder-Next (148 GB)
    ✓ Tensor · TP=2 — 84 GB/node, 4 GB cache = 165.774 tokens
    ▣ Measured here: 83.1 GB, 12m30s to load ... (-76.3 GB vs the plan)
      Weights: 159.4 GB on disk, split 2 ways = 83.7 GB per node
      KV per token: 24.0 KiB (... bfloat16) -> 165,774 tokens
    KV cache precision: Default (fp8 — required for long context on GB10)

Three things wrong, none of them the arithmetic:

* 148 GB and 159.4 GB are the same bytes in different units. _dir_size_gb
  divided by 1024**3 and called it GB; everything else here is decimal.
* "bfloat16" against a form that says the default is fp8. The plan was
  computed at "auto" because the UI never sent the dtype and the planner
  defaulted to the model's own — twice the real cost per token, so a panel
  reporting 165,774 tokens for a launch that would hold 331,548.
* "-76.3 GB vs the plan" compared a measurement taken on ONE node against the
  weights across BOTH. The plan had said 83.7 GB per node and the measurement
  was 83.1. The plan was right to within a gigabyte and accused itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
          / "js" / "app.js").read_text()


class TestOnDiskSizeIsDecimalGb:
    def test_a_gigabyte_of_files_reads_as_one_gb(self, tmp_path):
        from ainode.models.registry import ModelManager

        ModelManager.forget_size()
        directory = tmp_path / "org--model"
        directory.mkdir()
        with open(directory / "model.safetensors", "wb") as sink:
            sink.truncate(10 ** 9)
        assert round(ModelManager._dir_size_gb(directory), 3) == 1.0

    def test_it_matches_what_the_planner_reads(self, tmp_path):
        from ainode.models.registry import ModelManager
        from ainode.planner.facts import weight_bytes_on_disk

        ModelManager.forget_size()
        directory = tmp_path / "org--model2"
        directory.mkdir()
        with open(directory / "model.safetensors", "wb") as sink:
            sink.truncate(3 * 10 ** 9)
        # The two used to differ by 7.4% — the difference between a GiB and a
        # GB — with both labelled GB in the same panel.
        assert round(ModelManager._dir_size_gb(directory), 2) == \
            round(weight_bytes_on_disk(directory) / 1e9, 2)


class TestThePlanUsesTheDtypeTheLaunchWould:
    def test_the_ui_sends_it(self):
        assert "params.set('kv_cache_dtype'" in APP_JS

    def test_it_is_part_of_the_plan_cache_key(self):
        # Otherwise changing the dtype leaves the previous plan on screen,
        # which is the same wrong number by a different route.
        assert "kv_cache_dtype: (kv && kv.value)" in APP_JS

    def test_the_route_falls_back_to_the_node_default(self):
        import inspect

        from ainode.planner import api_routes

        source = inspect.getsource(api_routes.handle_plan)
        assert 'getattr(request.app.get("config"), "kv_cache_dtype"' in source

    def test_the_node_default_really_is_fp8(self):
        # The form promises this in words. If it ever changes, the promise and
        # the plan move together.
        from ainode.core.config import NodeConfig

        assert NodeConfig().kv_cache_dtype == "fp8"

    def test_fp8_halves_the_cost_of_a_token(self):
        from ainode.planner.compute import kv_bytes_per_token
        from ainode.planner.facts import ModelFacts

        facts = ModelFacts(repo="q", num_layers=48, attention_layers=12,
                           num_kv_heads=2, head_dim=256,
                           torch_dtype="bfloat16")
        assert kv_bytes_per_token(facts, "auto") == 24576
        assert kv_bytes_per_token(facts, "fp8") == 12288


class TestTheMeasurementComparesLikeWithLike:
    def _payload(self, **plan):
        from ainode.planner.api_routes import _attach_measurement

        payload = dict(plan)

        class _App(dict):
            pass

        app = _App()
        measured = {"memory_gb": 83.1, "launches": 1, "failures": 0}

        import ainode.measure.recorder as recorder

        original = recorder.measured_for
        recorder.measured_for = lambda a, m: measured
        try:
            _attach_measurement(app, "org/model", payload)
        finally:
            recorder.measured_for = original
        return payload

    def test_a_two_node_plan_is_judged_per_node(self):
        payload = self._payload(weights_gb=159.4, weights_per_node_gb=83.7)
        assert payload["measured"]["vs_plan_gb"] == pytest.approx(-0.6, abs=0.05)
        assert payload["measured"]["vs_plan_basis"] == "per node"

    def test_it_does_not_accuse_an_accurate_plan(self):
        payload = self._payload(weights_gb=159.4, weights_per_node_gb=83.7)
        assert abs(payload["measured"]["vs_plan_gb"]) < 5

    def test_a_solo_plan_still_compares_against_the_total(self):
        payload = self._payload(weights_gb=83.5)
        assert payload["measured"]["vs_plan_gb"] == pytest.approx(-0.4, abs=0.05)
        assert payload["measured"]["vs_plan_basis"] == "total"
