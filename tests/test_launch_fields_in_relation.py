"""Context and concurrency multiply into one cache, so one of them is an answer.

Asked from the cluster:

    können wir beim laden eines modells über die Seitenleiste die werte in den
    feldern in live relation zueinander setzen? also wenn ich parallel ändere so
    das sich dann automatisch auch die davon abhängigen werte selbst anpassen?

Two things stood in the way, and the first is worse than the second.

The advanced fields had no listener at all. Typing a context length changed
nothing above them — the plan was fetched on model and node changes only, so the
hint described a launch nobody had asked for and the two numbers next to each
other were unrelated.

And the planner could not have answered anyway. Given a concurrency and no
window it reported the model's own ceiling and then a concurrency derived from
that ceiling — which answers "how many fit at maximum length", not "I want N
sessions, how long can each be". The second is the question a launch form asks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ainode.planner.compute import NodeBudget, plan_for
from ainode.planner.facts import ModelFacts

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" / "static"
          / "js" / "app.js").read_text()


def _facts(ctx=262144):
    return ModelFacts(repo="q", num_layers=48, attention_layers=12,
                      num_kv_heads=2, head_dim=256, num_attention_heads=16,
                      torch_dtype="bfloat16", weight_bytes=80_400_000_000,
                      max_position_embeddings=ctx)


NODES = [NodeBudget(node_id="n1", name="s1", total_gb=128, free_gb=116)]


class TestDrivingTheConcurrency:
    def _len(self, concurrency):
        return plan_for(_facts(), NODES, kv_cache_dtype="fp8",
                        concurrency=concurrency).max_model_len

    def test_more_sessions_means_a_shorter_window(self):
        assert self._len(40) < self._len(20) < self._len(10)

    def test_it_never_exceeds_what_the_model_was_trained_for(self):
        # A cache that could back four million tokens does not make a 262k
        # model a million-token model.
        assert self._len(1) == 262144
        assert self._len(2) == 262144

    def test_a_short_ceiling_is_still_the_ceiling(self):
        plan = plan_for(_facts(ctx=32768), NODES, kv_cache_dtype="fp8",
                        concurrency=2)
        assert plan.max_model_len == 32768

    def test_the_window_is_a_whole_number_of_blocks(self):
        from ainode.planner.compute import LEN_GRANULARITY

        assert self._len(37) % LEN_GRANULARITY == 0


class TestDrivingTheWindow:
    def _seqs(self, length):
        return plan_for(_facts(), NODES, kv_cache_dtype="fp8",
                        max_model_len=length, concurrency=1).max_num_seqs

    def test_a_shorter_window_means_more_sessions(self):
        assert self._seqs(32768) > self._seqs(131072) > self._seqs(262144)

    def test_an_explicit_window_is_never_overridden(self):
        plan = plan_for(_facts(), NODES, kv_cache_dtype="fp8",
                        max_model_len=65536, concurrency=8)
        assert plan.max_model_len == 65536


class TestTheyMoveTogether:
    def test_the_product_stays_inside_the_cache(self):
        for concurrency in (1, 2, 4, 8, 16, 32):
            plan = plan_for(_facts(), NODES, kv_cache_dtype="fp8",
                            concurrency=concurrency)
            assert plan.max_model_len * plan.max_num_seqs <= plan.kv_tokens

    def test_a_cheaper_cache_buys_a_longer_window(self):
        tight = plan_for(_facts(), NODES, kv_cache_dtype="auto",
                         concurrency=40)
        cheap = plan_for(_facts(), NODES, kv_cache_dtype="fp8", concurrency=40)
        assert cheap.max_model_len > tight.max_model_len


class TestTheFormAsksAndAnswers:
    def test_the_advanced_fields_re_plan(self):
        """They had no listener. This is the half of the bug that made the
        other half invisible."""
        for field in ("launch-max-len", "launch-max-seqs", "launch-kv-dtype",
                      "launch-gmu"):
            assert field in APP_JS, field
        assert "launchLastEdited" in APP_JS

    def test_only_the_untouched_constraint_is_sent(self):
        # Sending both pins both, and then the planner has nothing to say.
        assert "drove !== 'seqs'" in APP_JS
        assert "drove !== 'len'" in APP_JS

    def test_the_derived_value_is_written_back(self):
        assert "reflectPlanIntoFields" in APP_JS

    def test_the_write_back_does_not_count_as_an_edit(self):
        # Or the two fields would chase each other around the form.
        assert "_applyingPlan" in APP_JS

    def test_the_memory_fraction_is_left_to_the_operator(self):
        source = APP_JS[APP_JS.index("reflectPlanIntoFields"):]
        source = source[:source.index("async fetchPlan")]
        assert "launch-gmu" not in source

    @pytest.mark.parametrize("field", ["launch-max-len", "launch-max-seqs"])
    def test_a_refused_plan_writes_nothing(self, field):
        # A plan that does not fit has no numbers worth pasting into a form.
        source = APP_JS[APP_JS.index("reflectPlanIntoFields"):]
        source = source[:source.index("async fetchPlan")]
        assert "plan.fits === false" in source
