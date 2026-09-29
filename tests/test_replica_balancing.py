"""A model on several nodes: which replica gets a request (wizzard.md, E4)."""

from __future__ import annotations

import json

from ainode.api.balancer import AFFINITY_SLACK, Balancer, affinity_key

LOCAL, REMOTE = ("localhost", 8000), ("10.0.0.2", 8000)


def _chat(system, first, *later):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": first}]
    for text in later:
        messages.append({"role": "assistant", "content": "ok"})
        messages.append({"role": "user", "content": text})
    return json.dumps({"model": "m", "messages": messages}).encode()


class TestLeastBusy:
    def test_idle_prefers_the_local_copy(self):
        assert Balancer().order([LOCAL, REMOTE]) == [LOCAL, REMOTE]

    def test_a_second_request_goes_to_the_other_replica(self):
        balancer = Balancer()
        balancer.acquire(LOCAL)
        assert balancer.order([LOCAL, REMOTE])[0] == REMOTE

    def test_a_finished_request_is_released(self):
        balancer = Balancer()
        balancer.acquire(LOCAL)
        balancer.release(LOCAL)
        assert balancer.order([LOCAL, REMOTE])[0] == LOCAL
        assert balancer.inflight == {}

    def test_one_candidate_is_one_candidate(self):
        assert Balancer().order([REMOTE], "k") == [REMOTE]


class TestTheSameConversationStays:
    def test_a_growing_conversation_has_one_key(self):
        first = affinity_key("m", _chat("sys", "fix the bug"))
        later = affinity_key("m", _chat("sys", "fix the bug", "and the tests"))
        assert first and first == later
        assert first != affinity_key("m", _chat("sys", "another task"))
        assert first != affinity_key("other", _chat("sys", "fix the bug"))

    def test_it_returns_to_its_replica(self):
        balancer = Balancer()
        key = affinity_key("m", _chat("sys", "fix the bug"))
        balancer.acquire(REMOTE, key)          # first step went to the remote
        balancer.release(REMOTE)
        assert balancer.order([LOCAL, REMOTE], key)[0] == REMOTE

    def test_but_not_to_a_replica_that_is_much_busier(self):
        balancer = Balancer()
        key = "conv"
        balancer.acquire(REMOTE, key)
        for _ in range(AFFINITY_SLACK + 1):
            balancer.acquire(REMOTE)
        assert balancer.order([LOCAL, REMOTE], key)[0] == LOCAL

    def test_a_request_without_messages_has_no_key(self):
        assert affinity_key("m", b"not json") == ""
        assert affinity_key("m", json.dumps({"model": "m"}).encode()) == ""
        assert affinity_key("m", json.dumps({"prompt": "hi"}).encode())
