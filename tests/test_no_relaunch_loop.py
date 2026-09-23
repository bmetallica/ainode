"""A model that cannot start must not be retried forever.

Reported from the cluster, after a launch was killed for memory:

    ahh irgendwas scheint nicht zu stimmen, er versucht qwen jetzt von
    alleine zu starten

That is what a crash loop looks like from outside: the engine is killed, the
container goes with it, systemd brings it back, and the startup replay
launches the same model again. Nothing in that circle asked whether the
launch was allowed — the admission gate lived in the HTTP handler, and the
replay does not go through it.
"""

from __future__ import annotations

import inspect

from ainode.models.api_routes import append_solo_instance, replay_instances_on_startup


class TestTheSharedCoreAsks:
    def test_append_solo_instance_checks_admission(self):
        source = inspect.getsource(append_solo_instance)
        assert "check_admission" in source

    def test_it_checks_before_it_stops_the_old_instance(self):
        # A refusal must not leave the node with nothing running.
        source = inspect.getsource(append_solo_instance)
        assert source.index("check_admission") < source.index("existing.backend.stop()")

    def test_it_scopes_to_this_node(self):
        source = inspect.getsource(append_solo_instance)
        assert "node_ids=[config.node_id]" in source

    def test_force_still_goes_through(self):
        source = inspect.getsource(append_solo_instance)
        assert "force=force" in source

    def test_the_refusal_is_reported_not_raised(self):
        source = inspect.getsource(append_solo_instance)
        assert '"status": 507' in source


class TestTheReplayGoesThroughIt:
    def test_the_replay_uses_the_shared_core(self):
        # Which is how it inherits the gate rather than needing its own.
        assert "append_solo_instance" in inspect.getsource(
            replay_instances_on_startup)

    def test_a_refused_replay_does_not_retry(self):
        # _ensure_serving retries once, and only for a launch that reported
        # ok with a port. A refusal has neither.
        source = inspect.getsource(replay_instances_on_startup)
        assert 'res.get("ok") and res.get("api_port")' in source


class TestTheSingleRetryStaysSingle:
    def test_ensure_serving_says_why_it_is_one(self):
        from ainode.models.api_routes import _ensure_serving

        doc = inspect.getdoc(_ensure_serving) or ""
        assert "deliberately single" in doc
        assert "loop would just hide it" in " ".join(doc.split())
