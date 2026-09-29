"""The announcement loop outlives a bad cycle.

B7 in upgrade-fixes.md, from FOLLOWUPS.md: a node whose engine had died kept
advertising its model fleet-wide — "observed on a node whose ainode had been up
5 weeks". The liveness gating in the announcement was there and correct. What
was not: _cluster_sync_loop caught CancelledError and nothing else, so one
exception in one cycle ended the loop for good, silently (nobody awaits the
task), and the sender went on broadcasting the last announcement it had been
given — model included — for as long as the process lived.
"""

from __future__ import annotations

import asyncio

import pytest

from ainode.api import server


class _Announcement:
    model = "org/was-serving"
    instances = [{"model": "org/stacked"}]


class _Sender:
    def __init__(self):
        self.announcement = _Announcement()
        self.updates = []

    def update_announcement(self, **kw):
        self.updates.append(kw)


@pytest.mark.asyncio
async def test_a_failing_cycle_does_not_end_the_loop(monkeypatch):
    calls = []

    async def _once(app):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("one bad cycle")
        if len(calls) >= 3:
            raise asyncio.CancelledError

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(server, "_cluster_sync_once", _once)
    monkeypatch.setattr(server.asyncio, "sleep", _no_sleep)
    await server._cluster_sync_loop({"broadcast_sender": _Sender()})
    assert len(calls) == 3        # it went on after the first one raised


@pytest.mark.asyncio
async def test_what_could_not_be_checked_is_withdrawn(monkeypatch):
    sender = _Sender()
    calls = []

    async def _once(app):
        calls.append(1)
        if len(calls) > 1:
            raise asyncio.CancelledError
        raise RuntimeError("probe blew up")

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(server, "_cluster_sync_once", _once)
    monkeypatch.setattr(server.asyncio, "sleep", _no_sleep)
    await server._cluster_sync_loop({"broadcast_sender": sender})
    assert sender.announcement.model == ""
    assert sender.announcement.instances == []
    assert {"model": "", "instances": []} in sender.updates


@pytest.mark.asyncio
async def test_a_broken_measurement_does_not_cost_the_announcement(monkeypatch):
    """The recorder runs first in each cycle. If its failure counted as the
    cycle's, every cycle would withdraw the node's models."""

    class _Broken:
        def poll(self):
            raise ValueError("unparseable")

    class _Cluster:
        pass

    app = {"measurement_recorder": _Broken(), "broadcast_listener": None,
           "cluster_state": _Cluster()}
    await server._cluster_sync_once(app)   # must not raise
